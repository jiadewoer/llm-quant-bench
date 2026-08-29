"""VRAM estimator for Ollama / llama.cpp GGUF models.

Predicts whether a given (model, quantization, context length, parallel slots)
combination fits in a fixed VRAM budget, and if not, what fraction of the model
Ollama is likely to keep on the GPU.

All sizes are in GiB (1024**3 bytes), matching what `nvidia-smi` and
`ollama ps` report.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

GIB = 1024**3


class Precision(Enum):
    """GGUF quantization levels.

    ``bytes_per_param`` is an *empirical* ratio (file size / parameter count),
    not the nominal bit-width. Real GGUF files keep embeddings and some
    attention tensors at higher precision than the nominal level, so q4_K_M
    lands near 0.66 B/param rather than the naive 0.5.

    Calibrated against Qwen2.5-7B (7.62B params):
      q4_K_M -> 4.7 GB, q8_0 -> 8.1 GB  (matches `ollama list`)

    Caveat: models below ~2B have proportionally huge vocab embeddings
    (Qwen2.5-0.5B is 27% embedding), so these ratios underestimate them.
    """

    F16 = ("f16", 2.05)
    Q8_0 = ("q8_0", 1.14)
    Q6_K = ("q6_K", 0.90)
    Q5_K_M = ("q5_K_M", 0.78)
    Q4_K_M = ("q4_K_M", 0.66)
    Q3_K_M = ("q3_K_M", 0.52)

    def __init__(self, tag: str, bytes_per_param: float) -> None:
        self.tag = tag
        self.bytes_per_param = bytes_per_param

    def __str__(self) -> str:
        return self.tag


@dataclass(frozen=True)
class ModelSpec:
    """Architecture parameters, all taken from the model's config.json.

    num_kv_heads < num_attention_heads means GQA: the KV cache shrinks by
    num_attention_heads / num_kv_heads. For Qwen2.5-7B that is 28/4 = 7x.
    """

    name: str
    num_params: float  # in billions
    num_layers: int  # num_hidden_layers
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int  # num_key_value_heads
    max_position_embeddings: int = 32768

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def gqa_ratio(self) -> float:
        return self.num_attention_heads / self.num_kv_heads


@dataclass(frozen=True)
class VRAMEstimate:
    weights_gb: float
    kv_cache_gb: float
    overhead_gb: float

    @property
    def total_gb(self) -> float:
        return self.weights_gb + self.kv_cache_gb + self.overhead_gb

    def fits_in(self, vram_gb: float, usable_fraction: float = 0.92) -> bool:
        """Does this fit in `vram_gb`?

        `usable_fraction` accounts for VRAM the desktop, browser and CUDA
        context already hold. On an 8188 MiB laptop 4060 with a display
        attached, ~0.5-0.8 GiB is gone before Ollama starts.
        """
        return self.total_gb <= vram_gb * usable_fraction

    def predicted_gpu_ratio(self, vram_gb: float, usable_fraction: float = 0.92) -> float:
        """Fraction of the model Ollama should keep on the GPU, in [0, 1].

        Compare this against the PROCESSOR column of `ollama ps`. Ollama
        offloads whole layers, so the real value is quantized to 1/num_layers
        steps and this continuous estimate will always be slightly off.
        """
        budget = vram_gb * usable_fraction - self.overhead_gb
        demand = self.weights_gb + self.kv_cache_gb
        if demand <= 0:
            return 1.0
        return max(0.0, min(1.0, budget / demand))

    def __str__(self) -> str:
        return (
            f"weights={self.weights_gb:.2f}GB kv={self.kv_cache_gb:.2f}GB "
            f"overhead={self.overhead_gb:.2f}GB total={self.total_gb:.2f}GB"
        )


def estimate(
    model: ModelSpec,
    precision: Precision = Precision.Q4_K_M,
    num_ctx: int = 2048,
    num_parallel: int = 1,
    kv_bytes_per_elem: float = 2.0,
    overhead_gb: float = 0.7,
) -> VRAMEstimate:
    """Estimate VRAM demand.

    num_parallel maps to OLLAMA_NUM_PARALLEL. Ollama allocates KV cache for
    every concurrent slot up front, so the cache is a straight multiple of it.
    When the variable is unset Ollama picks 1 or 4 on its own based on free
    VRAM, which silently makes results irreproducible -- pin it.

    kv_bytes_per_elem is 2.0 for f16. Only lower it if you have actually
    enabled KV cache quantization (which also requires flash attention).
    """
    weights = model.num_params * 1e9 * precision.bytes_per_param / GIB

    kv = (
        2  # K and V
        * model.num_layers
        * model.num_kv_heads
        * model.head_dim
        * num_ctx
        * num_parallel
        * kv_bytes_per_elem
    ) / GIB

    return VRAMEstimate(weights_gb=weights, kv_cache_gb=kv, overhead_gb=overhead_gb)


# Values below are transcribed from each model's config.json on HuggingFace.
# Verify before trusting -- a wrong num_key_value_heads silently breaks every
# KV cache prediction in the project.
PRESETS: dict[str, ModelSpec] = {
    "qwen2.5-0.5b": ModelSpec("Qwen2.5-0.5B", 0.49, 24, 896, 14, 2),
    "qwen2.5-1.5b": ModelSpec("Qwen2.5-1.5B", 1.54, 28, 1536, 12, 2),
    "qwen2.5-3b": ModelSpec("Qwen2.5-3B", 3.09, 36, 2048, 16, 2),
    "qwen2.5-7b": ModelSpec("Qwen2.5-7B", 7.62, 28, 3584, 28, 4),
    "qwen2.5-14b": ModelSpec("Qwen2.5-14B", 14.77, 48, 5120, 40, 8),
    "qwen2.5-32b": ModelSpec("Qwen2.5-32B", 32.76, 64, 5120, 40, 8),
    "llama3.1-8b": ModelSpec("Llama-3.1-8B", 8.03, 32, 4096, 32, 8, 131072),
}
