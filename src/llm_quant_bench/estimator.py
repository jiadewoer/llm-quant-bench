"""VRAM estimator for Ollama / llama.cpp GGUF models.

Version 2, recalibrated against measurements taken on 2026-08-29. The first
version was wrong in two ways and both are documented here rather than
quietly fixed, because the corrections are the interesting part.

All sizes are in GiB (1024**3 bytes), matching `nvidia-smi` and `ollama ps`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

GIB = 1024**3

# Measured, not assumed. Exact size_vram byte counts from /api/ps across
# every offloaded configuration measured on Day 4:
#
#     7b-q4  ctx16384  P1   5.668 GiB
#     7b-q4  ctx32768  P1   5.738 GiB
#     14b-q4 ctx4096   P1   5.627 GiB
#     7b-q4  ctx32768  P4   9.014 GiB   <- exceeds the card, see below
#
# The first three span 5.627-5.738, mean 5.678, on an 7.996 GiB card with a
# ~1.2 GiB desktop baseline. Ollama leaves roughly 1.1 GiB untouched.
#
# The fourth reports more VRAM than the card physically has. Windows driver
# memory fallback silently spills to host RAM while still counting it as
# VRAM; see results/day3_offload.md section 3.
#
# Version 1 assumed usable_fraction=0.92, i.e. a 7.36 GiB budget. That was
# 30% too optimistic and is why it predicted ctx 16384 would stay fully
# resident when it does not.
MEASURED_GPU_BUDGET_GB = 5.68


class Precision(Enum):
    """GGUF quantization levels.

    ``bytes_per_param`` is an empirical ratio (file size / parameter count),
    not the nominal bit width. Real GGUF files keep embeddings and some
    attention tensors above the nominal level, so q4_K_M lands near 0.66
    B/param rather than the naive 0.5.

    Calibrated against `ollama list`, which reports DECIMAL GB (bytes / 1e9),
    not GiB. Getting this wrong inflates every weight estimate by 7.4%:

        q4_K_M   4.7e9 / 7.62e9  = 0.617   (Qwen2.5-7B)
                 9.0e9 / 14.77e9 = 0.609   (Qwen2.5-14B)
                 1.9e9 / 3.09e9  = 0.615   (Qwen2.5-3B)
        q8_0     8.1e9 / 7.62e9  = 1.063

    Version 2 of this file used GiB-derived ratios (0.66 for q4_K_M) because
    Day 3 read SIZE off `ollama ps`, which also displays decimal GB. Day 4
    read exact byte counts from /api/ps and the discrepancy surfaced.

    Caveat: models below ~2B have proportionally huge vocabulary embeddings
    (Qwen2.5-0.5B is 27% embedding), so these ratios underestimate them.
    Measured 3B error was +0.23 GiB for the same reason.
    """

    F16 = ("f16", 1.91)
    Q8_0 = ("q8_0", 1.063)
    Q6_K = ("q6_K", 0.838)
    Q5_K_M = ("q5_K_M", 0.726)
    Q4_K_M = ("q4_K_M", 0.614)
    Q3_K_M = ("q3_K_M", 0.484)

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
    extra_context_gb: float
    overhead_gb: float = 0.0

    @property
    def total_gb(self) -> float:
        return self.weights_gb + self.kv_cache_gb + self.extra_context_gb + self.overhead_gb

    @property
    def context_cost_gb(self) -> float:
        """Everything that scales with num_ctx x num_parallel."""
        return self.kv_cache_gb + self.extra_context_gb

    def fits_in(self, budget_gb: float = MEASURED_GPU_BUDGET_GB) -> bool:
        """Will Ollama keep this fully resident on the GPU?

        The budget is what Ollama actually allocates, not the card's
        capacity. Pass a different value if your desktop baseline differs
        much from the ~1.2 GiB this was measured against.
        """
        return self.total_gb <= budget_gb

    def predicted_gpu_ratio(self, budget_gb: float = MEASURED_GPU_BUDGET_GB) -> float:
        """Fraction Ollama should keep on the GPU, in [0, 1].

        Compare against the PROCESSOR column of `ollama ps`. Ollama offloads
        whole layers, so the real value is quantized to 1/num_layers steps
        and this continuous estimate will always be a little off.
        """
        if self.total_gb <= 0:
            return 1.0
        return max(0.0, min(1.0, budget_gb / self.total_gb))

    def __str__(self) -> str:
        return (
            f"weights={self.weights_gb:.2f} kv={self.kv_cache_gb:.2f} "
            f"extra={self.extra_context_gb:.2f} total={self.total_gb:.2f} GB"
        )


def estimate(
    model: ModelSpec,
    precision: Precision = Precision.Q4_K_M,
    num_ctx: int = 2048,
    num_parallel: int = 1,
    kv_bytes_per_elem: float = 2.0,
    n_batch: int = 512,
    kv_cache_only: bool = False,
    overhead_gb: float = 0.0,
) -> VRAMEstimate:
    """Estimate what `ollama ps` will report as SIZE.

    Three terms:

    1. Weights. Constant for a given model and quantization.

    2. KV cache: 2 * L * H_kv * d_head * num_ctx * num_parallel * bytes.
       num_parallel maps to OLLAMA_NUM_PARALLEL. Ollama reserves cache for
       every concurrent slot up front, so num_ctx and num_parallel are the
       same multiplier. Measured confirmation: ctx 8192 with 4 slots and ctx
       32768 with 1 slot both reported exactly 8.7 GB.

    3. Extra per-token cost, empirical: n_batch * effective_tokens * H * 4.

       MECHANISM UNKNOWN. Measured context cost on Qwen2.5-7B is 132 KiB per
       token; the KV cache accounts for only 56 KiB. This term supplies the
       remaining 76 KiB and fits 11 measured configurations to within 0.24
       GiB, but the reason it fits has not been established.

       The original hypothesis was that it is llama.cpp's materialized
       attention-score buffer, present only when flash attention is off.
       That was tested on 2026-08-29 by setting OLLAMA_FLASH_ATTENTION=1
       and re-measuring ctx 16384. SIZE did not move: 6.5 GB and a 12%/88%
       split both times, identical to the flash-attention-off run.
       Hypothesis falsified.

       Most likely reason the test showed nothing: Ollama 0.24 appears to
       ignore OLLAMA_FLASH_ATTENTION, so both runs used the same code path.
       If flash attention was on throughout, the buffer this term was named
       after never existed in any of the measurements.

       The formula is kept because it predicts well across models with
       different H / H_kv ratios -- notably 14B-q4, where a plain multiple
       of the KV cache errs by 0.85 GiB and this form errs by 0.14 GiB. That
       is weak evidence the extra cost really does scale with attention-head
       count rather than KV-head count. It is not an explanation.

       Candidate mechanisms, and the experiment that would separate them,
       are listed in results/day3_offload.md section 6.

       This term uses effective_tokens, not num_ctx. It has to: ctx 8192
       with 4 slots and ctx 32768 with 1 slot were measured to report the
       same SIZE, so every context-dependent term must see the product.

       Pass kv_cache_only=True to drop it and see the pure textbook
       prediction, which under-predicts ctx 32768 by 1.5 GiB.

    Note that overhead defaults to 0: the CUDA context and driver allocation
    show up in the nvidia-smi baseline, not in Ollama's reported SIZE.
    """
    weights = model.num_params * 1e9 * precision.bytes_per_param / GIB

    effective_tokens = num_ctx * num_parallel

    kv = (
        2  # K and V
        * model.num_layers
        * model.num_kv_heads
        * model.head_dim
        * effective_tokens
        * kv_bytes_per_elem
    ) / GIB

    extra = (
        0.0
        if kv_cache_only
        else (n_batch * effective_tokens * model.num_attention_heads * 4) / GIB
    )

    return VRAMEstimate(
        weights_gb=weights,
        kv_cache_gb=kv,
        extra_context_gb=extra,
        overhead_gb=overhead_gb,
    )


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
