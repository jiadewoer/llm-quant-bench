"""Throughput and latency benchmark against a single Ollama model.

Uses /api/chat with stream=true, not the OpenAI-compatible endpoint. Two
reasons: /api/chat accepts the `options` block (so num_ctx actually takes
effect), and its final chunk carries Ollama's own timing counters --
prompt_eval_count, eval_count, eval_duration -- which are more precise than
anything measured from the client side.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import httpx

from .monitor import LoadedModel, find_loaded, gpu_used_mib, stop_model

OLLAMA = "http://localhost:11434"

DEFAULT_PROMPT = (
    "Explain in about 120 words how grouped-query attention reduces the "
    "size of the key-value cache during autoregressive decoding."
)


@dataclass
class RequestResult:
    ttft_s: float  # client-side time to first content token
    wall_s: float  # client-side total
    prompt_tokens: int
    eval_tokens: int
    eval_duration_s: float  # server-side decode time
    load_duration_s: float

    @property
    def decode_tps(self) -> float:
        return self.eval_tokens / self.eval_duration_s if self.eval_duration_s else 0.0


@dataclass
class BenchResult:
    label: str
    model: str
    num_ctx: int
    num_parallel: int
    requests: list[RequestResult] = field(default_factory=list)
    loaded: LoadedModel | None = None
    baseline_gb: float | None = None

    def summary(self) -> dict:
        if not self.requests:
            return {"label": self.label, "error": "no successful requests"}

        ttfts = sorted(r.ttft_s for r in self.requests)
        tps = [r.decode_tps for r in self.requests]

        out: dict = {
            "label": self.label,
            "model": self.model,
            "num_ctx": self.num_ctx,
            "num_parallel": self.num_parallel,
            "n_requests": len(self.requests),
            "baseline_gb": self.baseline_gb,
            "ttft_p50_s": round(_pct(ttfts, 50), 3),
            "ttft_p95_s": round(_pct(ttfts, 95), 3),
            "decode_tps_mean": round(sum(tps) / len(tps), 2),
            "decode_tps_min": round(min(tps), 2),
            "eval_tokens_mean": round(
                sum(r.eval_tokens for r in self.requests) / len(self.requests), 1
            ),
        }
        if self.loaded:
            out["measured"] = {
                "size_gb": round(self.loaded.size_gb, 3),
                "vram_gb": round(self.loaded.vram_gb, 3),
                "gpu_ratio": round(self.loaded.gpu_ratio, 4),
            }
        return out

    def to_json(self) -> str:
        payload = self.summary()
        payload["raw_requests"] = [asdict(r) for r in self.requests]
        return json.dumps(payload, indent=2, ensure_ascii=False)


def _pct(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


async def _one_request(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    num_ctx: int,
    num_predict: int,
) -> RequestResult | None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": 0,
            "seed": 42,
        },
    }

    start = time.perf_counter()
    ttft: float | None = None
    final: dict = {}

    try:
        async with client.stream("POST", f"{OLLAMA}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if ttft is None and content:
                    ttft = time.perf_counter() - start
                if chunk.get("done"):
                    final = chunk
    except (httpx.HTTPError, json.JSONDecodeError):
        return None

    wall = time.perf_counter() - start
    return RequestResult(
        ttft_s=ttft if ttft is not None else wall,
        wall_s=wall,
        prompt_tokens=final.get("prompt_eval_count", 0),
        eval_tokens=final.get("eval_count", 0),
        eval_duration_s=final.get("eval_duration", 0) / 1e9,
        load_duration_s=final.get("load_duration", 0) / 1e9,
    )


async def run_bench(
    label: str,
    model: str,
    num_ctx: int = 2048,
    num_parallel: int = 1,
    n_requests: int = 8,
    num_predict: int = 128,
    prompt: str = DEFAULT_PROMPT,
) -> BenchResult:
    """Benchmark one configuration.

    Always unloads first. A model already resident keeps its original
    num_ctx, so skipping the unload gives every row in a sweep the same
    numbers -- the most confusing failure mode in this whole project.
    """
    stop_model(model)
    base = gpu_used_mib()
    result = BenchResult(
        label=label,
        model=model,
        num_ctx=num_ctx,
        num_parallel=num_parallel,
        baseline_gb=round(base / 1024, 3) if base is not None else None,
    )

    timeout = httpx.Timeout(600.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Warmup: loads weights and allocates KV cache. Never counted --
        # its load_duration would dominate and hide the steady-state rate.
        await _one_request(client, model, prompt, num_ctx, num_predict=16)
        result.loaded = find_loaded(model)

        for _ in range(n_requests):
            r = await _one_request(client, model, prompt, num_ctx, num_predict)
            if r is not None:
                result.requests.append(r)

    return result
