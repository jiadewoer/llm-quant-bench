# Day 6 · The full matrix

**Date** 2026-08-30
**Hardware** RTX 4060 Laptop, 8188 MiB (7.996 GiB), driver 580.88, 32 GB RAM
**Runtime** Ollama 0.24.0, `OLLAMA_NUM_PARALLEL=1` (except the P4 row),
`OLLAMA_KEEP_ALIVE=30m`, `OLLAMA_FLASH_ATTENTION=0`, `OLLAMA_KV_CACHE_TYPE=f16`

All eleven rows ran back to back **in a single session** with the browser
closed, on mains power, with the Windows power mode set to best performance.
Every baseline landed between 1.116 and 1.222 GiB, so the rows are
comparable.

---

## 1. Summary

| Config | ctx | SIZE (GiB) | VRAM (GiB) | GPU% | decode tok/s | TTFT p50 | baseline |
|---|---|---|---|---|---|---|---|
| 3b-q4 | 4096 | 2.226 | 2.226 | 100% | **106.14** | 0.265 | 1.215 |
| 7b-q4 | 2048 | 4.477 | 4.477 | 100% | 52.01 | 0.282 | 1.213 |
| 7b-q4 | 4096 | 4.586 | 4.586 | 100% | 51.94 | 0.273 | 1.222 |
| 7b-q4 | 8192 | 4.975 | 4.975 | 100% | 51.75 | 0.293 | 1.213 |
| 7b-q4 | 12288 | 5.420 | 5.420 | 100% | 51.84 | 0.272 | 1.116 |
| 7b-q4 | 16384 | 6.084 | 5.668 | 93.2% | 34.42 | 0.306 | 1.205 |
| 7b-q4 | 32768 | 8.084 | 5.738 | 71.0% | 18.77 | 0.359 | 1.205 |
| 7b-q8 | 4096 | 7.929 | 5.721 | 72.2% | 16.33 | 0.345 | 1.209 |
| 14b-q4 | 4096 | 9.602 | 5.787 | 60.3% | 8.27 | 0.415 | 1.219 |
| 14b-q4 | 8192 | 10.352 | 5.871 | 56.7% | 7.03 | 0.458 | 1.119 |
| 14b-q4 | 16384 | 12.571 | 5.743 | 45.7% | 5.07 | 0.502 | 1.119 |
| 7b-q4 | 32768 · P=4 | 20.084 | 9.014 | 44.9% | 13.94 | 0.420 | 1.298 |

---

## 2. The pre-registered prediction held

The Day 5 document carried a prediction, **committed to git before the data
existed**:

> Day 6 measures 7B-q8 throughput. Following the Day 4 pattern (34% offloaded
> should cost well over half the throughput), expect **15–20 tok/s**, about a
> third of q4.

Measured: **16.33 tok/s**, exactly **1/3.18** of q4. Both the interval and
the ratio hold.

The value is not in guessing right; it is that the claim was fixed before the
data arrived, and the ordering is checkable in the git history —
`day5_eval.md` was committed before any Day 6 result file existed.

---

## 3. The cliff is now bracketed

The `ctx 12288` row exists purely to locate the transition:

| ctx | Predicted total | Budget 5.68 GiB | Measured GPU% |
|---|---|---|---|
| 8192 | 5.232 | fits | 100% |
| **12288** | **5.670** | **fits, by 0.01 GiB** | **100%** |
| 16384 | 6.107 | does not fit | 93.2% |

**The transition sits between 12288 and 16384, and the estimator put 12288
within 0.01 GiB of the budget.** This is its strongest validation: it did not
just fit the points it was given, it placed the boundary correctly.

Throughput at 12288 is 51.84 tok/s against 51.75 at 8192, confirming nothing
spilled.

---

## 4. Out-of-sample validation

Five of the twelve rows were **never used for calibration**:
`3b-q4-ctx4096`, `7b-q8-ctx4096`, `7b-q4-ctx12288`, `14b-q4-ctx8192`,
`14b-q4-ctx16384`.

| Config | Predicted | Measured | Error | Pred GPU% | Meas GPU% |
|---|---|---|---|---|---|
| 3b-q4-ctx4096 † | 2.033 | 2.226 | −0.193 | 100% | 100% |
| 7b-q4-ctx2048 | 4.576 | 4.477 | +0.099 | 100% | 100% |
| 7b-q4-ctx4096 | 4.795 | 4.586 | +0.209 | 100% | 100% |
| 7b-q4-ctx8192 | 5.232 | 4.975 | +0.257 | 100% | 100% |
| 7b-q4-ctx12288 † | 5.670 | 5.420 | +0.250 | 100% | 100% |
| 7b-q4-ctx16384 | 6.107 | 6.084 | +0.023 | 93.0% | 93.2% |
| 7b-q4-ctx32768 | 7.857 | 8.084 | −0.227 | 72.3% | 71.0% |
| 7b-q8-ctx4096 † | 7.981 | 7.929 | +0.052 | 71.2% | 72.2% |
| 14b-q4-ctx4096 | 9.508 | 9.602 | −0.094 | 59.7% | 60.3% |
| 14b-q4-ctx8192 † | 10.571 | 10.352 | +0.219 | 53.7% | 56.7% |
| 14b-q4-ctx16384 † | 12.696 | 12.571 | +0.125 | 44.7% | 45.7% |
| 7b-q4-ctx32768-P4 | 18.357 | 20.084 | **−1.727** | 30.9% | 44.9% |

† = held out

**Across eleven rows (excluding P4): mean absolute SIZE error 0.159 GiB, mean
absolute GPU-ratio error 0.6 percentage points.** All five held-out points
land within 0.26 GiB.

The P=4 row remains the only large miss (−1.73 GiB) for the known reason: the
formula's slope runs 12.5% below measurement and that compounds over 131072
effective tokens. It is excluded from the accuracy claims.

---

## 5. ⭐ The unexplained term: both candidates fall

Day 4 found the measured context cost to be 2.286× the theoretical KV cache,
with the excess unexplained. Two candidates were named, along with the
observation that a 14B context sweep would separate them. The sweep is done.

### Marginal cost, decomposed

| Model | Interval | Marginal SIZE | Theoretical KV | Marginal extra |
|---|---|---|---|---|
| 7B | 2048→4096 | 55.8 KiB | 56.0 | **−0.2** |
| 7B | 4096→8192 | 99.6 | 56.0 | 43.6 |
| 7B | 8192→12288 | 113.9 | 56.0 | 57.9 |
| 7B | 12288→16384 | 170.0 | 56.0 | 114.0 |
| 7B | 16384→32768 | 128.0 | 56.0 | **72.0** |
| 7B | 32768→131072 | 128.0 | 56.0 | **72.0** |
| 14B | 4096→8192 | 192.0 | 192.0 | **0.0** |
| 14B | 8192→16384 | 284.0 | 192.0 | **92.0** |

### The verdict

Asymptotic marginal extra: **72 KiB/token for 7B, 92 KiB/token for 14B.**

| | 7B predicted | 14B predicted | 14B/7B ratio |
|---|---|---|---|
| **Measured** | 72 KiB | 92 KiB | **1.28** |
| Candidate A: \(n_{batch} \times H \times 4\) | 56 | 80 | 1.43 |
| Candidate B: 1.286 × KV | 72 | 247 | 3.43 |

**Candidate B is dead.** It predicts 247 KiB/token for the 14B against a
measured 92 — off by 2.7×. It looked perfect on the 7B only because it was
calibrated there; a different architecture breaks it immediately. This is
exactly why the 14B sweep was worth running.

**Candidate A has the right shape but runs about 15% low.** Its predicted
ratio of 1.43 is far closer to the measured 1.28 than B's 3.43, which does
say the term scales with attention-head count \(H\) rather than KV-head count
\(H_{kv}\). The magnitude is systematically under.

### A new fact: the extra term is exactly zero at small contexts

Neither candidate predicts this:

- 7B from 2048 to 4096: marginal **55.8 KiB** against a theoretical KV of 56.0
- 14B from 4096 to 8192: marginal **192.0 KiB** against a theoretical 192.0 —
  an exact match

**Over those intervals, context costs nothing but KV cache. The extra term
contributes zero.**

Both candidates are linear-from-zero and neither can produce a threshold. So
the excess is **not a simple per-token buffer** but something that switches on
once context passes a floor — between 4096 and 8192 for the 7B, between 8192
and 16384 for the 14B.

### The experiment to run next

`n_batch` appears only in candidate A and nowhere in the KV formula, which
makes it the discriminating variable:

```powershell
# add "num_batch": 128 / 1024 to the options block of the payload
ollama stop qwen2.5:7b
curl.exe -s http://localhost:11434/api/chat -H "Content-Type: application/json" -d "@payloads/ctx32768-batch128.json" > $null
curl.exe -s http://localhost:11434/api/ps
```

Halve `num_batch` from 512 to 256, then double it to 1024, and watch SIZE at
ctx 32768:

- **slope tracks batch** → the term really is batch-related, candidate A's
  shape stands and only its coefficient needs work
- **slope does not move** → candidate A's shape is wrong too, and the whole
  form needs replacing

Sampling densely around the threshold (7B at 5120 / 6144 / 7168) would also
show whether it comes on gradually or as a step.

> This section records a falsification experiment that worked as designed.
> Day 4 named two candidates and identified the measurement that would
> separate them; Day 6 took it; one candidate is refuted, the other partly
> supported, and both are shown to have missed a threshold neither predicted.
> **The estimator's comment still says the mechanism is unknown — the
> unknown is just smaller now.**

---

## 6. Throughput: size cost and offload cost separate cleanly

### With nothing offloaded, throughput tracks 1/weights

| | GPU% | Weights | decode tok/s |
|---|---|---|---|
| 3B-q4 | 100% | ~1.90 GiB | 106.14 |
| 7B-q4 | 100% | ~4.35 GiB | 51.94 |

Throughput ratio **2.04**, weight ratio **2.29**. Near inverse proportion,
which is what a bandwidth-bound decode should give.

### The 14B's 6.28× slowdown splits in two

A 14B can never be fully resident on this card, but the relationship above
extrapolates: fully resident it should run about **26.7 tok/s**.

| Factor | Multiple |
|---|---|
| Being a bigger model | 51.94 / 26.7 = **1.94×** |
| Offloading 39.7% of it | 26.7 / 8.27 = **3.23×** |
| Product | 6.28× |

Measured 51.94 / 8.27 = **6.28×** — the factors multiply out exactly.

**Offloading costs more (3.23×) than doubling the model does (1.94×).** The
practical reading: a smaller model that fits beats a larger one that almost
fits.

### The context cliff

Against ctx 8192 (100% GPU, 51.75 tok/s):

| ctx | GPU% | Offloaded | decode tok/s | Throughput lost | Amplification |
|---|---|---|---|---|---|
| 12288 | 100% | 0% | 51.84 | −0.2% | — |
| 16384 | 93.2% | **6.8%** | 34.42 | **33.5%** | **4.9×** |
| 32768 | 71.0% | 29.0% | 18.77 | 63.7% | 2.2× |

**6.8% of bytes off the GPU costs 33.5% of throughput — a 4.9×
amplification.**

The amplification falls from 4.9 to 2.2, matching the trend seen on Day 4
(5.5 → 2.0 there).

---

## 7. A new reproducibility problem

The same configurations measured on Day 4 and again on Day 6:

| Config | GPU% | Day 4 | Day 6 | Difference |
|---|---|---|---|---|
| 7b-q4-ctx2048 | 100% | 51.83 | 52.01 | 0.3% |
| 7b-q4-ctx8192 | 100% | 51.78 | 51.75 | 0.1% |
| 7b-q4-ctx16384 | 93.2% | 32.16 | 34.42 | **7.0%** |
| 7b-q4-ctx32768 | 71.0% | 21.08 | 18.77 | **11.0%** |
| 14b-q4-ctx4096 | 60.3% | 8.44 | 8.27 | 2.0% |

**Fully resident configurations repeat to within 0.3%; offloaded ones vary by
7–11%.**

The cause is presumably that the layer split is decided at load time against
whatever VRAM happens to be free, and the baseline drifts a little
(1.116–1.222 GiB). `SIZE` is identical between the two runs, so the
configuration itself is deterministic; what varies is how the layers land.

**Implication: throughput for an offloaded configuration must be reported
with an error range, never as a single value.** Treat every offloaded row in
this project as ±10% and every resident row as ±0.5%.

---

## 8. Known limitations

- **Offloaded throughput was measured once** and repeats only to ±10% (§7).
  A defensible interval needs three to five repeats per configuration.
- **`gpu_ratio` is a byte fraction, not a layer fraction.** SIZE includes the
  KV cache and the unexplained term, neither of which is a "layer". The "6.8%
  offloaded" in §6 is a proxy for how much spilled, not the share of layers
  moved. A rigorous split needs Ollama's own layer-assignment logs.
- **The P=4 ctx32768 row supports no quantitative claim.** It reports 9.014
  GiB resident on a 7.996 GiB card because the Windows driver is presenting
  host RAM as VRAM (see `day3_offload.md` §3). It stands only as qualitative
  evidence of overcommit.
- **One machine, one GPU.** Nothing generalises to a desktop 4060 Ti 16GB or
  to Apple unified memory without re-measurement.
- **`extra_context_gb` still has no mechanism**, only a narrower range: it
  scales with \(H\) rather than \(H_{kv}\), and it is zero at small contexts.

---

## 9. What to run next

| Experiment | Purpose | Cost |
|---|---|---|
| `num_batch` sweep at 128 / 512 / 1024 | Decide candidate A's shape | 20 min |
| 7B at 5120 / 6144 / 7168 | Is the threshold a step or a ramp? | 20 min |
| Three repeats per offloaded config | Error bars on throughput | 1 hour |
| llama3.1:8b context sweep | A third \(H/H_{kv}\) ratio to test extrapolation | 40 min |

The first two are the highest value and take 40 minutes together.
