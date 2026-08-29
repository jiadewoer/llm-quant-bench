# Day 3 · Offload behaviour, measured

**Date** 2026-08-29
**Hardware** RTX 4060 Laptop, 8188 MiB, driver 580.88, 32 GB RAM
**Runtime** Ollama 0.24.0, Windows 11 (26100)
**Pinned** `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_KEEP_ALIVE=30m`,
`OLLAMA_FLASH_ATTENTION=0`, `OLLAMA_KV_CACHE_TYPE=f16`

Desktop VRAM baseline, no model loaded: **1.13 – 1.75 GiB** across five
readings. Every measurement below was taken in that range.

---

## 1. Size and quantization axis

Taken with `ollama run <model> "hi"`, so all four used Ollama's default
context of **4096**, not 2048.

| Model | Quant | SIZE | PROCESSOR | Predicted | Error |
|---|---|---|---|---|---|
| qwen2.5:3b | q4_K_M | 2.4 GB | 100% GPU | 2.16 | −0.24 |
| qwen2.5:7b | q4_K_M | 4.9 GB | 100% GPU | 5.12 | +0.22 |
| qwen2.5:7b-instruct-q8_0 | q8_0 | 8.5 GB | 34%/66% CPU/GPU | 8.53 | +0.03 |
| qwen2.5:14b | q4_K_M | 10 GB | 47%/53% CPU/GPU | 10.14 | +0.14 |

The 3B row is under-predicted for the reason already noted in the estimator:
its vocabulary embedding is 27% of the model and is not stored at q4.

---

## 2. Context axis (qwen2.5:7b q4_K_M, P=1)

| num_ctx | SIZE | PROCESSOR | Predicted | Error |
|---|---|---|---|---|
| 2048 | 4.8 GB | 100% GPU | 4.90 | +0.10 |
| 8192 | 5.3 GB | 100% GPU | 5.56 | +0.26 |
| **16384** | **6.5 GB** | **15%/85% CPU/GPU** | 6.43 | −0.07 |
| 32768 | 8.7 GB | 36%/64% CPU/GPU | 8.18 | −0.52 |

**The cliff is between 8192 and 16384.** The pre-measurement estimator
predicted all four rows would stay fully resident. It was wrong; see §5.

---

## 3. Parallel axis (qwen2.5:7b q4_K_M, P=4)

| num_ctx | SIZE | PROCESSOR | Equivalent P=1 config | Its SIZE |
|---|---|---|---|---|
| 2048 | 5.3 GB | 100% GPU | ctx 8192 | **5.3 GB** |
| 8192 | 8.7 GB | 31%/69% CPU/GPU | ctx 32768 | **8.7 GB** |
| 16384 | 12 GB | 54%/46% CPU/GPU | ctx 65536 | not measured |
| 32768 | 21 GB | 55%/45% CPU/GPU | ctx 131072 | not measured |

### The cross-validation held exactly

The estimator treats `num_ctx` and `num_parallel` as a single product, which
predicts that ctx 8192 × 4 slots and ctx 32768 × 1 slot must consume
identical memory. Two independent pairs were measured and both matched to
the last reported digit.

This is the strongest result of the day. It means `OLLAMA_NUM_PARALLEL`,
which Ollama picks automatically between 1 and 4 when unset, silently
multiplies VRAM consumption by up to 4× — enough to move a configuration
from fully resident to half-offloaded without any visible cause.

### The "impossible" data point is not a display artifact

ctx 32768 × P=4 reports 21 GB at 45% GPU, implying ~9.5 GiB resident on a
7.996 GiB card. Re-read through `/api/ps` for exact bytes:

```
size      = 21,565,124,608 B = 20.084 GiB
size_vram =  9,678,372,864 B =  9.014 GiB
ratio     = 44.9%   (matches the displayed 45%)
```

**Rounding is ruled out. Ollama reports 9.01 GiB resident in VRAM, 1.02 GiB
beyond the card's physical capacity and 2.36 GiB beyond what was free after
the 1.34 GiB desktop baseline.**

#### Explanation: Windows system memory fallback

NVIDIA's Windows driver enables memory fallback by default (since 536.xx;
this machine runs 580.88). When VRAM runs out the driver does not return
OOM — it maps host RAM into the GPU's address space and serves it over PCIe.

From CUDA's point of view the allocation succeeded, so Ollama honestly
reports 9.01 GiB. Roughly 2.4 GiB of that lives in DDR5 and is reached at
two orders of magnitude less bandwidth than real VRAM.

**This is a completely silent performance cliff.** A user seeing `45% GPU`
would assume half the compute is on the card. Half of that half is host
memory wearing a costume, and nothing warns them.

#### To be confirmed

Day 4 throughput on this configuration should fall well below what a 45%
split implies. If it is more than 2× slower than `7b-q4-ctx16384` at 88%
GPU, the explanation holds.

---

## 4. How much VRAM will Ollama actually use?

Reverse-computed from `SIZE × GPU%` on every offloaded configuration:

| Config | SIZE | GPU% | Resident |
|---|---|---|---|
| 7b-q8 ctx 4096 | 8.5 | 66% | 5.61 GB |
| 14b-q4 ctx 4096 | 10.0 | 53% | 5.30 GB |
| 7b-q4 ctx 16384 | 6.5 | 85% | 5.52 GB |
| 7b-q4 ctx 32768 | 8.7 | 64% | 5.57 GB |
| 7b-q4 ctx 8192 P4 | 8.7 | 69% | 6.00 GB |
| 7b-q4 ctx 16384 P4 | 12.0 | 46% | 5.52 GB |

Range 5.30 – 6.00 GiB, mean 5.59. On a card holding 7.99 GiB with a ~1.2 GiB
desktop baseline, Ollama leaves roughly **1.3 GiB untouched**.

The pre-measurement estimator assumed 92% of the card was usable — a 7.36
GiB budget, 34% too generous.

---

## 5. What the estimator got wrong, and why

### Error 1 — a missing memory term

Fitting measured SIZE against effective tokens (`num_ctx × num_parallel`)
gives **132 KiB per token**. The KV cache alone accounts for only 56 KiB:

```
KV per token = 2 × L × H_kv × d_head × b = 2 × 28 × 4 × 128 × 2 = 56 KiB
```

The missing 76 KiB was attributed to the materialized attention buffer. With
`OLLAMA_FLASH_ATTENTION=0`, llama.cpp should be unable to fuse the attention
computation and would have to allocate space for the scores:

```
attn buffer per token = n_batch × H × 4 = 512 × 28 × 4 = 56 KiB
```

For Qwen2.5-7B these two terms are coincidentally equal, which made the
explanation look convincing. Adding the term dropped the mean absolute SIZE
error from 1.5 GB to 0.24 GB.

> ⚠️ **This explanation was later falsified by experiment. See §6.** The
> formula still fits; the mechanism it names is wrong.

### Error 2 — a guessed VRAM budget

`usable_fraction = 0.92` was an assumption. The measured budget is 5.5 GiB
(§4). This is now a named constant with the measurement recorded next to it.

### After both corrections

Across 11 configurations (excluding the suspect row in §3):

- Mean absolute SIZE error: **0.24 GB**
- Mean absolute GPU-ratio error: **1.0 percentage point**

---

## 6. The hypothesis was falsified

§5 attributed the extra 76 KiB per token to an attention-score buffer that
exists only when flash attention is off. That claim is falsifiable, and it
was tested.

### Design

One variable changed. `OLLAMA_FLASH_ATTENTION` set from 0 to 1, service and
terminal restarted, `check_env.py` confirming `1 (expected 0)` before
measuring. Everything else held constant: P=1, KV f16, same model, same
`payloads/ctx16384.json`.

### Result

| | SIZE | PROCESSOR |
|---|---|---|
| FA = 0 (original) | 6.5 GB | 12%/88% CPU/GPU |
| FA = 1 (predicted 5.56 GB) | **6.5 GB** | **12%/88% CPU/GPU** |

**Nothing moved, down to the split. The hypothesis does not hold.**

### Why the experiment showed nothing

Most likely **Ollama 0.24 ignores `OLLAMA_FLASH_ATTENTION` entirely**. Recent
versions enable flash attention by default and the variable is vestigial. A
switch that produces bit-identical results in both positions is behaving
exactly like a switch that is not wired to anything.

If that is what happened, every measurement in this document was taken with
flash attention on, and the buffer the term was named after never existed in
any of them.

### So what is the 76 KiB per token?

**Not known.** This is an honest gap, not an omission.

Constraints established so far:

- It scales linearly with `num_ctx × num_parallel` (§3 guarantees this).
- It appears to track attention-head count \(H\) rather than KV-head count
  \(H_{kv}\). Evidence: 14B has a different \(H/H_{kv}\) ratio than 7B,
  and fitting it as a fixed multiple of the KV cache errs by 0.85 GiB while
  the \(n_{batch} \times H \times 4\) form errs by 0.14 GiB. One data
  point, so weak evidence.

Candidate mechanisms and the experiment that would separate them:

| Candidate | How to test |
|---|---|
| Ollama's memory estimator carries a fixed margin | Compare `/api/ps` `size` against actual `nvidia-smi` usage |
| A buffer tied to `n_batch` | Set `num_batch` to 128 and 1024; see whether the slope tracks it |
| KV cache is not stored at f16 after all | Set `OLLAMA_KV_CACHE_TYPE=q8_0`; see whether the slope halves |
| Architecture-dependent fixed overhead | Repeat the context sweep on llama3.1:8b, where \(H/H_{kv}=4\) |

**The `n_batch` test is the cheapest and most discriminating**: `n_batch`
appears only in this term and nowhere in the KV cache formula. Halve the
batch; if the slope halves, the term really is batch-related. If the slope
does not move, the whole form should be replaced.

### What the estimator does about it

The term is kept but renamed `extra_context_gb`, with the docstring stating
plainly that the mechanism is unknown. It is a well-fitting empirical term
(0.24 GiB mean error across 11 configurations), not an explained physical
one. Presenting it as physics in the README would be dishonest.

> This section records a failed hypothesis. Keeping it is worth more than
> deleting it: a hypothesis that its own author could falsify is one that was
> stated precisely enough to be worth making.

---

## 7. Decisions for Day 6

- **Context axis stays on 7B-q4_K_M.** The cliff is at 16384, well inside
  the model's native 32768 limit, so the curve has three usable points
  either side of the transition. No need to move it to q8_0.
- **Add ctx 4096** to the matrix — it is Ollama's default and therefore the
  configuration most readers will actually be running.
- **Promote the parallel axis from optional to core.** Two exact matches is
  a result worth reporting, not a footnote.
- **Add a flash-attention pair** (ctx 16384, FA on vs off) if time allows.

---

## 8. Environment caveats

- Desktop VRAM baseline varied by 0.6 GiB across readings depending on what
  was open. All Day 6 runs must be done in one session with the browser
  closed, and `baseline_gb` recorded per run.
- **SIZE is reproducible; the GPU split is not.** ctx 16384 was measured
  twice:

  | | SIZE | PROCESSOR | baseline |
  |---|---|---|---|
  | first | 6.5 GB | 15%/85% | ~1.2 GiB |
  | second | 6.5 GB | 12%/88% | 1.65 GiB |

  Identical SIZE, so it depends only on configuration. The layer split is
  decided at load time against whatever VRAM happens to be free. **A
  gpu_ratio reported without its baseline is meaningless.**
- `ollama ps` rounds SIZE to two significant figures. Day 6 should read
  `/api/ps` for exact byte counts.
- Ollama's default context is 4096, not 2048. Any comparison against a
  prediction has to use the same value.
