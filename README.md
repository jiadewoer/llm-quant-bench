# llm-quant-bench

**How large a model can an 8GB consumer GPU actually run?**

A reproducible benchmark of model size × quantization × context length on a
single RTX 4060 Laptop (8188 MiB), served by Ollama. Every number here is
measured on one machine, and every prediction is compared against what
actually happened.

> Status: work in progress. Results land 2026-09-05.

---

## The three claims

*(placeholders — replace with measured numbers as results come in)*

1. **The size cliff.** Qwen2.5-7B at q4_K_M stays entirely on the GPU;
   14B at the same quantization does not, and the partial CPU offload costs
   roughly _N_× in tokens/sec.
2. **Quantization buys more than context does.** Dropping 7B from q8_0 to
   q4_K_M frees ~3.4 GiB — more than a 32K context window costs in KV cache.
3. **`OLLAMA_NUM_PARALLEL` is a hidden multiplier.** Left unset, Ollama picks
   1 or 4 slots depending on free VRAM, changing KV cache size 4× without
   telling you. Half of "unreproducible local LLM benchmarks" are this.

---

## Why the VRAM math works out the way it does

KV cache is the only term that grows with context:

```
KV_bytes = 2 × layers × kv_heads × head_dim × num_ctx × num_parallel × bytes_per_elem
```

The leading 2 is K and V. `num_parallel` maps to `OLLAMA_NUM_PARALLEL` —
Ollama reserves cache for every concurrent slot up front, so `num_ctx` and
`num_parallel` enter the formula identically. That has a testable
consequence: **ctx 8192 with 4 slots should occupy exactly as much VRAM as
ctx 32768 with 1 slot.** If the measurements agree, the model is right; if
they don't, something is allocating cache differently than documented.

Grouped-query attention is what makes any of this fit. Qwen2.5-7B has 28
attention heads but only 4 KV heads, a 7× saving. Without it, a 32K context
would need ~12 GiB of cache alone.

---

## Quick start

```cmd
git clone https://github.com/jiadewoer/llm-quant-bench.git
cd llm-quant-bench
uv venv
.venv\Scripts\activate.bat
uv pip install -e ".[dev]"

python scripts\check_env.py
python scripts\scratch.py
pytest -v
```

`dev.bat` does the UTF-8 codepage switch and venv activation in one step —
double-click it instead of the first four lines above.

### Pin the Ollama runtime first

Benchmark results are not reproducible until these are fixed. Set them, then
quit Ollama from the tray, restart it, and open a fresh terminal — neither
running CMD windows nor the running Ollama service reload environment
variables.

```cmd
setx OLLAMA_NUM_PARALLEL 1
setx OLLAMA_KEEP_ALIVE 30m
setx OLLAMA_FLASH_ATTENTION 0
setx OLLAMA_KV_CACHE_TYPE f16
```

`check_env.py` verifies all four and warns if any is missing.

---

## Hardware

| | |
|---|---|
| GPU | RTX 4060 Laptop, 8188 MiB, driver 580.88 |
| RAM | 32 GB |
| OS | Windows 11 (26100) |
| Runtime | Ollama 0.24.0 |
| Python | 3.11.9 |

---

## Known limitations

- Single machine, single GPU. Nothing here generalizes to desktop 4060 Ti
  16GB or to Apple unified memory without re-measurement.
- Weight-size estimates are calibrated against Qwen2.5 GGUF builds and
  underestimate models below ~2B, where vocabulary embeddings dominate.
- KV cache quantization is switched off deliberately, so f16 is the only
  cache precision measured.

## License

MIT
