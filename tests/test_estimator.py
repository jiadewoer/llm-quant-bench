"""Tests for the VRAM estimator.

Each test is a hypothesis to be checked against real `ollama ps` output on
Day 6. When one turns out wrong, the error analysis is worth more than the
test passing was.
"""

import pytest

from llm_quant_bench.estimator import PRESETS, ModelSpec, Precision, estimate

VRAM_8GB = 8.0


def test_weights_scale_with_bytes_per_param():
    """q8_0 stores roughly twice as many bytes per weight as q4_K_M."""
    q4 = estimate(PRESETS["qwen2.5-7b"], Precision.Q4_K_M, num_ctx=2048)
    q8 = estimate(PRESETS["qwen2.5-7b"], Precision.Q8_0, num_ctx=2048)
    ratio = q8.weights_gb / q4.weights_gb
    assert 1.6 < ratio < 2.0
    assert q4.kv_cache_gb == pytest.approx(q8.kv_cache_gb)


def test_weights_match_ollama_reported_size():
    """Calibration check: predicted weights should land near `ollama list`.

    ollama list reports 4.7 GB for qwen2.5:7b and 8.1 GB for the q8_0 build.
    """
    q4 = estimate(PRESETS["qwen2.5-7b"], Precision.Q4_K_M)
    q8 = estimate(PRESETS["qwen2.5-7b"], Precision.Q8_0)
    assert q4.weights_gb == pytest.approx(4.7, abs=0.3)
    assert q8.weights_gb == pytest.approx(8.1, abs=0.3)


def test_kv_cache_is_linear_in_context():
    """Doubling num_ctx doubles the cache -- no hidden allocation granularity."""
    small = estimate(PRESETS["qwen2.5-7b"], num_ctx=2048)
    large = estimate(PRESETS["qwen2.5-7b"], num_ctx=8192)
    assert large.kv_cache_gb == pytest.approx(small.kv_cache_gb * 4, rel=1e-6)


def test_gqa_shrinks_kv_cache():
    """Qwen2.5-7B uses 4 KV heads for 28 attention heads: a 7x saving.

    Without GQA the cache at ctx 32768 would be 12 GB and nothing would fit.
    """
    real = PRESETS["qwen2.5-7b"]
    assert real.gqa_ratio == pytest.approx(7.0)

    mha = ModelSpec(
        "hypothetical-MHA-7B",
        real.num_params,
        real.num_layers,
        real.hidden_size,
        real.num_attention_heads,
        real.num_attention_heads,  # every head keeps its own K/V
    )
    with_gqa = estimate(real, num_ctx=8192)
    without_gqa = estimate(mha, num_ctx=8192)
    assert without_gqa.kv_cache_gb == pytest.approx(with_gqa.kv_cache_gb * 7, rel=1e-6)


def test_num_parallel_multiplies_kv_cache():
    """Ollama allocates KV cache per concurrent slot -- a hidden multiplier."""
    p1 = estimate(PRESETS["qwen2.5-7b"], num_ctx=8192, num_parallel=1)
    p4 = estimate(PRESETS["qwen2.5-7b"], num_ctx=8192, num_parallel=4)
    assert p4.kv_cache_gb == pytest.approx(p1.kv_cache_gb * 4, rel=0.01)
    assert p4.weights_gb == pytest.approx(p1.weights_gb)


def test_14b_does_not_fit_but_7b_does():
    """The core claim of the project, at the smallest context."""
    seven = estimate(PRESETS["qwen2.5-7b"], Precision.Q4_K_M, num_ctx=2048)
    fourteen = estimate(PRESETS["qwen2.5-14b"], Precision.Q4_K_M, num_ctx=2048)

    assert seven.fits_in(VRAM_8GB)
    assert not fourteen.fits_in(VRAM_8GB)
    # Expect partial offload, not a total failure -- compare to ollama ps.
    assert 0.3 < fourteen.predicted_gpu_ratio(VRAM_8GB) < 0.85


def test_context_axis_is_flat_until_32k():
    """The reason the Day 6 matrix needs a 32768 row.

    2048 / 8192 / 16384 are all predicted to stay fully on GPU, so a matrix
    limited to those three rows plots a flat line and the context axis says
    nothing. Only at 32768 does the estimate come within a few percent of the
    budget -- close enough that whether it actually offloads depends on how
    much VRAM the desktop is holding.

    This is the single prediction most likely to be wrong. Measure it with
    `ollama ps` on Day 3 before locking the matrix; if 32768 still reports
    100% GPU, move the context axis onto 14B or 7B-q8_0.
    """
    budget = VRAM_8GB * 0.92
    totals = {
        ctx: estimate(PRESETS["qwen2.5-7b"], num_ctx=ctx).total_gb
        for ctx in (2048, 8192, 16384, 32768)
    }
    assert totals[16384] < budget * 0.90, "16k should fit comfortably"
    assert 0.95 < totals[32768] / budget < 1.10, "32k should be marginal, not comfortable"
