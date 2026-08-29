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

### One suspect data point

ctx 32768 × P=4 reports 21 GB at 45% GPU, which implies 9.45 GiB resident on
an 7.99 GiB card. That is impossible, so either the percentage is computed
from layer count rather than bytes at this extreme, or the display rounds
badly. Re-measure via `/api/ps`, which returns exact `size` and `size_vram`
in bytes. Excluded from the error statistics below.

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

The missing 76 KiB is the materialized attention buffer. With
`OLLAMA_FLASH_ATTENTION=0`, llama.cpp cannot fuse the attention computation
and must allocate space for the scores:

```
attn buffer per token = n_batch × H × 4 = 512 × 28 × 4 = 56 KiB
```

For Qwen2.5-7B these two terms are coincidentally equal, so **turning flash
attention off doubles what a context window costs**. Adding this term
dropped the mean absolute SIZE error from 1.5 GB to 0.24 GB.

### Error 2 — a guessed VRAM budget

`usable_fraction = 0.92` was an assumption. The measured budget is 5.5 GiB
(§4). This is now a named constant with the measurement recorded next to it.

### After both corrections

Across 11 configurations (excluding the suspect row in §3):

- Mean absolute SIZE error: **0.24 GB**
- Mean absolute GPU-ratio error: **1.0 percentage point**

---

## 6. The experiment this suggests

The attention-buffer hypothesis is falsifiable and cheap to test. Set
`OLLAMA_FLASH_ATTENTION=1`, restart the service, and re-measure ctx 16384.

| Prediction | ctx 16384 | ctx 32768 |
|---|---|---|
| FA off (measured) | 6.5 GB | 8.7 GB |
| FA on (predicted) | 5.56 GB | 6.43 GB |

If SIZE drops by roughly the KV cache amount, the hypothesis holds and flash
attention becomes a legitimate fourth axis: it buys back about 0.9 GiB at
16K and 2.3 GiB at 32K, for free.

If SIZE does not drop, the excess is something else and the model needs
another look.

> Note that `OLLAMA_FLASH_ATTENTION=0` was set deliberately, to keep
> `kv_bytes_per_elem = 2.0` honest. That decision turned out to have a cost
> nobody documents.

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
- `ollama ps` rounds SIZE to two significant figures. Day 6 should read
  `/api/ps` for exact byte counts.
- Ollama's default context is 4096, not 2048. Any comparison against a
  prediction has to use the same value.
