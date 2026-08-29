"""Tests for the VRAM estimator.

Tests 1-5 are structural: they check the formula behaves the way the physics
says it should. Tests 6-9 are regression checks against numbers measured on
2026-08-29 with an RTX 4060 Laptop (8188 MiB), Ollama 0.24.0,
OLLAMA_NUM_PARALLEL=1, OLLAMA_FLASH_ATTENTION=0, OLLAMA_KV_CACHE_TYPE=f16.

If a regression test starts failing, either the model changed or Ollama did.
Find out which before changing the number.
"""

import pytest

from llm_quant_bench.estimator import (
    MEASURED_GPU_BUDGET_GB,
    PRESETS,
    ModelSpec,
    Precision,
    estimate,
)


def test_weights_scale_with_bytes_per_param():
    """q8_0 stores roughly twice as many bytes per weight as q4_K_M."""
    q4 = estimate(PRESETS["qwen2.5-7b"], Precision.Q4_K_M, num_ctx=2048)
    q8 = estimate(PRESETS["qwen2.5-7b"], Precision.Q8_0, num_ctx=2048)
    assert 1.6 < q8.weights_gb / q4.weights_gb < 2.0
    assert q4.kv_cache_gb == pytest.approx(q8.kv_cache_gb)


def test_weights_match_ollama_reported_size():
    """Calibration anchor: `ollama list` shows 4.7 GB and 8.1 GB."""
    assert estimate(PRESETS["qwen2.5-7b"], Precision.Q4_K_M).weights_gb == pytest.approx(
        4.7, abs=0.3
    )
    assert estimate(PRESETS["qwen2.5-7b"], Precision.Q8_0).weights_gb == pytest.approx(
        8.1, abs=0.3
    )


def test_kv_cache_is_linear_in_context():
    small = estimate(PRESETS["qwen2.5-7b"], num_ctx=2048)
    large = estimate(PRESETS["qwen2.5-7b"], num_ctx=8192)
    assert large.kv_cache_gb == pytest.approx(small.kv_cache_gb * 4, rel=1e-6)


def test_gqa_shrinks_kv_cache():
    """Qwen2.5-7B uses 4 KV heads for 28 attention heads: a 7x saving."""
    real = PRESETS["qwen2.5-7b"]
    assert real.gqa_ratio == pytest.approx(7.0)

    mha = ModelSpec(
        "hypothetical-MHA-7B",
        real.num_params,
        real.num_layers,
        real.hidden_size,
        real.num_attention_heads,
        real.num_attention_heads,
    )
    with_gqa = estimate(real, num_ctx=8192)
    without_gqa = estimate(mha, num_ctx=8192)
    assert without_gqa.kv_cache_gb == pytest.approx(with_gqa.kv_cache_gb * 7, rel=1e-6)


def test_context_and_parallel_are_interchangeable():
    """MEASURED, not merely derived.

    ctx 8192 with 4 slots and ctx 32768 with 1 slot both reported exactly
    8.7 GB. Every context-dependent term must therefore see the product
    num_ctx * num_parallel, never num_ctx alone.
    """
    a = estimate(PRESETS["qwen2.5-7b"], num_ctx=8192, num_parallel=4)
    b = estimate(PRESETS["qwen2.5-7b"], num_ctx=32768, num_parallel=1)
    assert a.total_gb == pytest.approx(b.total_gb, rel=1e-9)

    c = estimate(PRESETS["qwen2.5-7b"], num_ctx=2048, num_parallel=4)
    d = estimate(PRESETS["qwen2.5-7b"], num_ctx=8192, num_parallel=1)
    assert c.total_gb == pytest.approx(d.total_gb, rel=1e-9)


def test_disabling_flash_attention_doubles_context_cost_on_7b():
    """A coincidence worth knowing about.

    KV cache costs 2*L*H_kv*d_head*b = 56 KiB per token on Qwen2.5-7B.
    The materialized attention buffer costs n_batch*H*4 = 56 KiB per token.
    They happen to be equal, so OLLAMA_FLASH_ATTENTION=0 exactly doubles
    what a context window costs.
    """
    e = estimate(PRESETS["qwen2.5-7b"], num_ctx=16384)
    assert e.attn_buffer_gb == pytest.approx(e.kv_cache_gb, rel=0.01)

    on = estimate(PRESETS["qwen2.5-7b"], num_ctx=16384, flash_attention=True)
    assert on.attn_buffer_gb == 0.0
    assert on.context_cost_gb == pytest.approx(e.context_cost_gb / 2, rel=0.01)


@pytest.mark.parametrize(
    "preset,precision,num_ctx,num_parallel,measured_gb",
    [
        ("qwen2.5-7b", Precision.Q4_K_M, 2048, 1, 4.8),
        ("qwen2.5-7b", Precision.Q4_K_M, 4096, 1, 4.9),
        ("qwen2.5-7b", Precision.Q4_K_M, 8192, 1, 5.3),
        ("qwen2.5-7b", Precision.Q4_K_M, 16384, 1, 6.5),
        ("qwen2.5-7b", Precision.Q4_K_M, 32768, 1, 8.7),
        ("qwen2.5-7b", Precision.Q8_0, 4096, 1, 8.5),
        ("qwen2.5-14b", Precision.Q4_K_M, 4096, 1, 10.0),
        ("qwen2.5-7b", Precision.Q4_K_M, 8192, 4, 8.7),
        ("qwen2.5-7b", Precision.Q4_K_M, 16384, 4, 12.0),
    ],
)
def test_predicts_measured_size(preset, precision, num_ctx, num_parallel, measured_gb):
    """Predicted SIZE within 0.6 GB of what `ollama ps` reported."""
    e = estimate(PRESETS[preset], precision, num_ctx=num_ctx, num_parallel=num_parallel)
    assert e.total_gb == pytest.approx(measured_gb, abs=0.6)


def test_context_cliff_lands_between_8k_and_16k():
    """Where the offload actually starts, corrected by measurement.

    Version 1 predicted 7B-q4 would stay fully resident through ctx 16384
    and only get tight at 32768. It reported 15%/85% CPU/GPU at 16384. Two
    causes: the missing attention-buffer term, and a usable-VRAM assumption
    of 7.36 GiB against a measured Ollama budget of 5.5 GiB.
    """
    fits = {
        ctx: estimate(PRESETS["qwen2.5-7b"], num_ctx=ctx).fits_in()
        for ctx in (2048, 4096, 8192, 16384, 32768)
    }
    assert fits[2048] and fits[4096]
    assert not fits[16384] and not fits[32768]


def test_gpu_budget_is_measured_not_card_capacity():
    """Ollama consistently stopped at ~5.5 GiB on an 8 GiB card.

    Across six offloaded configurations the GPU-resident bytes ranged from
    5.30 to 5.61 GiB. The card holds 7.99 GiB and the desktop baseline was
    ~1.2 GiB, so Ollama leaves roughly 1.3 GiB untouched.
    """
    assert 5.0 < MEASURED_GPU_BUDGET_GB < 6.0

    e = estimate(PRESETS["qwen2.5-14b"], Precision.Q4_K_M, num_ctx=4096)
    assert e.predicted_gpu_ratio() == pytest.approx(0.53, abs=0.05)
