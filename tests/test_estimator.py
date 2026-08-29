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


def test_extra_context_term_is_empirical_and_unexplained():
    """Guards a claim the code makes about itself, not about physics.

    Measured context cost on Qwen2.5-7B is 132 KiB per token; the KV cache
    accounts for 56 KiB. The extra term supplies the rest and happens to
    equal the KV cache exactly on this model.

    That coincidence originally suggested a mechanism: llama.cpp's
    materialized attention-score buffer, which exists only when flash
    attention is off. The hypothesis was tested on 2026-08-29 by setting
    OLLAMA_FLASH_ATTENTION=1 and re-measuring ctx 16384. SIZE did not move
    -- 6.5 GB and a 12%/88% split under both settings. Falsified.

    The term stays because it fits, and the docstring says so plainly. If
    someone later renames it back to something that implies a mechanism,
    this test is the reminder that no mechanism has been established.
    """
    e = estimate(PRESETS["qwen2.5-7b"], num_ctx=16384)
    assert e.extra_context_gb == pytest.approx(e.kv_cache_gb, rel=0.01)
    assert "unknown" in estimate.__doc__.lower()


def test_flash_attention_setting_did_not_change_measured_size():
    """The falsifying measurement itself, as a regression check.

    ctx 16384 reported 6.5 GB with OLLAMA_FLASH_ATTENTION=0 and 6.5 GB with
    it set to 1. One prediction has to cover both, because the estimator has
    no flash-attention input any more -- removing that parameter was the
    consequence of this result.
    """
    e = estimate(PRESETS["qwen2.5-7b"], num_ctx=16384)
    assert e.total_gb == pytest.approx(6.5, abs=0.6)


def test_kv_cache_only_reproduces_the_original_underprediction():
    """The textbook formula, kept available for comparison.

    Version 1 of the estimator had only weights plus KV cache. It predicted
    6.43 GB for ctx 32768 against a measured 8.7 GB, a 2.3 GB miss, which is
    why the extra term exists at all.
    """
    naive = estimate(PRESETS["qwen2.5-7b"], num_ctx=32768, kv_cache_only=True)
    assert naive.extra_context_gb == 0.0
    assert naive.total_gb == pytest.approx(6.43, abs=0.1)
    assert naive.total_gb < 8.7 - 2.0  # under-predicts the measurement badly


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
