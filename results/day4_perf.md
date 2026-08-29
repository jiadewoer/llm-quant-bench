# Day 4 · Throughput and latency, measured

**Date** 2026-08-29
**Hardware** RTX 4060 Laptop, 8188 MiB (7.996 GiB), driver 580.88, 32 GB RAM
**Runtime** Ollama 0.24.0, Windows 11 (26100)
**Pinned** `OLLAMA_NUM_PARALLEL=1` (except the parallel-axis row),
`OLLAMA_KEEP_ALIVE=30m`, `OLLAMA_FLASH_ATTENTION=0`, `OLLAMA_KV_CACHE_TYPE=f16`

**Method**: every configuration unloads all resident models first, waits for
the driver to release VRAM, records a baseline, runs one uncounted warmup,
then issues 8 sequential requests (2 for the rows marked `--requests 2`).
Fixed prompt, `temperature=0`, `seed=42`, `num_predict=128`.

**Throughput is server-side**: `eval_count / eval_duration` from the final
chunk of `/api/chat`. No client-side stopwatch, so no network-stack or
Python-scheduling noise.

**Memory is exact bytes from `/api/ps`**, not the SIZE column of `ollama ps`.
That column is decimal GB and reading it as GiB carries a systematic 7.4%
error — see §6.

---

## 1. Summary

| Config | ctx | P | baseline | SIZE (GiB) | VRAM (GiB) | GPU% | decode tok/s | TTFT p50 |
|---|---|---|---|---|---|---|---|---|
| 7b-q4 | 2048 | 1 | 1.15 | 4.477 | 4.477 | 100% | 51.83 | 0.313 |
| 7b-q4 | 4096 | 1 | 1.09 | 4.586 | 4.586 | 100% | 51.98 | 0.281 |
| 7b-q4 | 8192 | 1 | 1.28 | 4.975 | 4.975 | 100% | 51.78 | 0.286 |
| 7b-q4 | 16384 | 1 | 1.12 | 6.084 | 5.668 | 93.2% | 32.16 | 0.324 |
| 7b-q4 | 32768 | 1 | 1.15 | 8.084 | 5.738 | 71.0% | 21.08 | 0.355 |
| 14b-q4 | 4096 | 1 | 1.28 | 9.602 | 5.627 | 58.6% | 8.44 | 0.406 |
| 7b-q4 | 32768 | 4 | 1.30 | 20.084 | 9.014 | 44.9% | 13.94 | 0.420 |

**Every fully resident configuration runs at the same speed** (51.83 / 51.98
/ 51.78, under 0.3% apart), so context length on its own costs nothing as
long as nothing spills. This is an important control: it establishes that
every slowdown below comes from offloading, not from longer contexts making
attention more expensive.

---

## 2. The size cliff

| | GPU% | decode tok/s |
|---|---|---|
| qwen2.5-7b q4_K_M ctx4096 | 100% | 51.98 |
| qwen2.5-14b q4_K_M ctx4096 | 58.6% | 8.44 |

**6.16× slower for a model twice the size.**

Whatever sits on the CPU crosses PCIe once per generated token. PCIe 4.0 x8
delivers roughly 12 GB/s against this card's ~256 GB/s of VRAM bandwidth —
a factor of 20. Decoding is bandwidth-bound, so the bandwidth gap is the
speed gap.

---

## 3. The context cliff is sharply non-linear

Against ctx 8192 (100% GPU, 51.78 tok/s):

| ctx | GPU% | offloaded | decode tok/s | throughput lost | amplification |
|---|---|---|---|---|---|
| 8192 | 100% | 0% | 51.78 | — | — |
| 16384 | 93.2% | **6.8%** | 32.16 | **37.9%** | **5.5×** |
| 32768 | 71.0% | 29.0% | 21.08 | 59.3% | 2.0× |

**Moving 6.8% of the bytes off the GPU costs 37.9% of the throughput.**

This is the most counter-intuitive result in the project. "93% still on the
GPU" sounds like a rounding error; it is a 1.6× slowdown.

### How much slower is the offloaded part

Split per-token time into a GPU share and a CPU share:

\[
\frac{1}{\text{tps}} = a \cdot f_{\text{gpu}} + b \cdot f_{\text{cpu}}
\]

ctx 8192 fixes \(a = 0.01931\) s. Solving for \(b\) at the other two points:

| Data point | implied \(b\) | \(b/a\) |
|---|---|---|
| ctx 16384 | 0.191 s | **9.9×** |
| ctx 32768 | 0.116 s | 6.0× |

**The offloaded portion costs 6–10× per unit what the resident portion
does.** The two points disagree, which means the linear split is only an
approximation — CPU compute, host memory bandwidth, and the fact that
`gpu_ratio` is a byte fraction rather than a layer fraction all contribute.

> ⚠️ **`gpu_ratio` is bytes, not layers.** SIZE includes the KV cache and the
> unexplained extra term, and neither is a "layer". So \(f_{\text{cpu}}\)
> above is a proxy for how much spilled, not the fraction of layers on CPU.
> A rigorous decomposition needs the layer split from Ollama's own logs.

### Why the amplification falls

From 5.6× to 2.0×. The plausible reading is marginal cost: the first thing
moved off pays a fixed round-trip, and everything moved after that is added
to an already slow path.

**Untested.** Confirming it needs denser sampling between 8192 and 16384 —
10240, 12288, 14336 — to see whether the amplification falls smoothly.

---

## 4. TTFT barely moves

| Config | GPU% | TTFT p50 | vs base | decode tok/s | vs base |
|---|---|---|---|---|---|
| 7b ctx8192 | 100% | 0.286 s | 1.00× | 51.78 | 1.00× |
| 7b ctx16384 | 93.2% | 0.324 s | 1.13× | 32.16 | 1.61× |
| 7b ctx32768 | 71.0% | 0.355 s | 1.24× | 21.08 | 2.46× |
| 14b ctx4096 | 58.6% | 0.406 s | 1.42× | 8.44 | **6.14×** |

**14B's first token is only 1.4× slower; its decoding is 6.1× slower.**

The two phases have different bottlenecks:

- **Prefill** processes the whole prompt in one batched pass. It is compute
  dense, parallelises well, and touches each weight once.
- **Decode** produces one token at a time. Every token walks the weights
  again, so it is purely bandwidth-bound, and every offloaded layer crosses
  PCIe on every single token.

**What this means in practice**: an offloaded model *feels* responsive —
text starts appearing promptly — and then types unusably slowly. Choosing a
model on TTFT alone will mislead badly.

---

## 5. The parallel axis

| Config | SIZE | VRAM | GPU% | decode tok/s |
|---|---|---|---|---|
| 7b-q4 ctx32768 P=1 | 8.084 | 5.738 | 71.0% | 21.08 |
| 7b-q4 ctx32768 P=4 | 20.084 | 9.014 | 44.9% | 13.94 |

Four slots instead of one: memory goes from 8.08 to 20.08 GiB and decoding
loses another 1.51×.

**20.084 lands exactly on the memory line** (§6) — a third independent
confirmation of the `num_ctx × num_parallel` equivalence, this time at byte
resolution rather than off a rounded display.

Reported VRAM of 9.014 GiB exceeds the card's 7.996 GiB. Windows driver
memory fallback is presenting host RAM as VRAM; see `day3_offload.md` §3.
**Losing only 1.51× suggests the spilled region is not touched often**, but
nothing quantitative should be drawn from this row.

---

## 6. `ollama ps` reports decimal GB

Day 3 read the SIZE column. Day 4 read exact bytes from `/api/ps`. They
disagree:

| Config | Day 3 display | Day 4 exact (GiB) | ratio |
|---|---|---|---|
| 7b ctx2048 | 4.8 | 4.477 | 1.072 |
| 7b ctx16384 | 6.5 | 6.084 | 1.068 |
| 7b ctx32768 | 8.7 | 8.084 | 1.076 |

The ratio holds at 1.07, and \(1024^3 / 10^9 = 1.0737\).

**`ollama ps` shows decimal GB (bytes / 1e9). Treating it as GiB overstates
everything by 7.4%.** The same applies to `ollama list`, so the
`bytes_per_param` ratios derived from file sizes were inflated too.

### Refit against exact bytes

| effective tokens | SIZE (GiB) | delta (KiB/token) |
|---|---|---|
| 2048 | 4.477 | — |
| 4096 | 4.586 | 55.8 |
| 8192 | 4.975 | 99.6 |
| 16384 | 6.084 | 142.0 |
| 32768 | 8.084 | **128.0** |
| 131072 (P=4) | 20.084 | **128.0** |

The linear region has a slope of **exactly 128.0 KiB per token** and an
intercept of 4.084 GiB. Checking back:

```
ctx  4096  fit 4.584   measured 4.586   diff -0.002
ctx 16384  fit 6.084   measured 6.084   diff  0.000
ctx 32768  fit 8.084   measured 8.084   diff  0.000
P=4 131072 fit 20.084  measured 20.084  diff  0.000
```

Four significant figures. Theoretical KV is 56.0 KiB per token, measured is
128.0 — a **ratio of 2.286**.

### Corrections to the estimator

- `bytes_per_param` recalibrated in decimal GB (q4_K_M: 0.66 → 0.614)
- `MEASURED_GPU_BUDGET_GB` 5.5 → 5.68 (exact `size_vram` readings of 5.668,
  5.738, 5.627; mean 5.678)

After both, across the six P=1 configurations: **mean absolute SIZE error
0.151 GiB, mean absolute GPU-ratio error 0.4 percentage points.**

The P=4 row still under-predicts by 1.73 GiB. The formula's slope is 112
KiB/token (56 KV + 56 extra) against a measured 128, a 12.5% shortfall that
compounds over 131072 effective tokens. That gap is now asserted in a test
so it cannot be quietly forgotten.

---

## 7. One bug, two fixes

`baseline_gb` was recorded contaminated three times:

| Config | recorded | true |
|---|---|---|
| 7b-q4-ctx8192 (first run) | 7.188 | ~1.2 |
| 14b-q4-ctx4096 (first run) | 5.912 | ~1.2 |
| 14b-q4-ctx4096 (after fix 1) | 6.349 | ~1.2 |
| both (after fix 2) | **1.283 / 1.283** | ✅ |

**Fix 1**: `/api/ps` stops listing a model before the driver has released
its VRAM. Poll `nvidia-smi` until the reading stops falling. This corrected
`7b-q4-ctx8192`, which reran at 1.049.

**14B was still wrong at 6.349.** Different root cause: `run_bench()` only
unloaded **the model it was about to benchmark**. Benchmarking 14B called
`stop_model("qwen2.5:14b")` while the previous run's 7B sat untouched, so
its 4.975 GiB went straight into the baseline.

**Fix 2**: `stop_all()` unloads every resident model before the baseline is
read. After the rerun both baselines read **1.283**, identical to three
decimals -- which is itself the evidence the fix worked.

> Throughput and memory figures are unaffected — Ollama evicts the 7B on its
> own when loading the 14B, so `size` and `gpu_ratio` are clean. Only the
> `baseline_gb` field was wrong. But that field is how comparability between
> runs is judged, so it had to be fixed.

---

## 8. Decisions for Day 6

- **Add a 14B context sweep.** There is one 14B point, so its slope is
  undetermined — and the slope's shape is exactly what separates the two
  candidate explanations for the unknown term:

  | Candidate | 14B per token |
  |---|---|
  | Scales with attention heads \(H\) (\(n_{batch} \times H \times 4\)) | 272 KiB |
  | Fixed multiple of KV (×2.286) | 439 KiB |

  A factor of 1.6 apart. Two rows — `14b-q4-ctx8192` and `14b-q4-ctx16384` —
  settle it. **Highest value per minute of any remaining experiment.**

- **Sample densely between 8192 and 16384** to test the falling-amplification
  reading in §3.

- **Keep P=4 ctx32768 out of quantitative claims**; it stands only as
  qualitative evidence of VRAM overcommit.

---

## 9. Environment caveats

- Every P=1 baseline landed between 1.09 and 1.28 GiB, so the runs are
  comparable.
- The three fully resident configurations differ by under 0.3% in throughput
  (51.78 / 51.83 / 51.98), which is a good repeatability signal.
- `eval_tokens_mean` varies from 87 to 94, and is 128 for the 14B. Different
  generation lengths affect how stable `decode_tps` is but not its value —
  it is a rate, not a duration.
- Matrix runs need the browser closed, the machine on mains power, and the
  Windows power mode set to best performance. Power saving drops the GPU
  clock and quietly depresses every throughput number.
