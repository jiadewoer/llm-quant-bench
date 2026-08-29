"""Print estimator predictions for the planned benchmark matrix.

Run this before locking configs/matrix.json -- it tells you which axes will
actually produce a curve and which will plot a flat line.

    python scripts/scratch.py
"""

from llm_quant_bench.estimator import PRESETS, Precision, estimate

VRAM = 8.0

print(f"\nVRAM budget: {VRAM} GiB (usable ~{VRAM * 0.92:.2f} GiB)\n")

print("--- context axis: qwen2.5-7b q4_K_M, num_parallel=1 ---")
for ctx in (2048, 8192, 16384, 32768):
    e = estimate(PRESETS["qwen2.5-7b"], Precision.Q4_K_M, num_ctx=ctx)
    print(
        f"ctx={ctx:>6}  {e}  fits={e.fits_in(VRAM)!s:<5} "
        f"gpu_ratio={e.predicted_gpu_ratio(VRAM):.2f}"
    )

print("\n--- size / quantization axis, ctx=2048 ---")
for key, prec in [
    ("qwen2.5-3b", Precision.Q4_K_M),
    ("qwen2.5-7b", Precision.Q4_K_M),
    ("qwen2.5-7b", Precision.Q8_0),
    ("qwen2.5-14b", Precision.Q4_K_M),
    ("qwen2.5-32b", Precision.Q4_K_M),
]:
    e = estimate(PRESETS[key], prec, num_ctx=2048)
    print(
        f"{key:<14} {prec!s:<8} {e}  fits={e.fits_in(VRAM)!s:<5} "
        f"gpu_ratio={e.predicted_gpu_ratio(VRAM):.2f}"
    )

print("\n--- num_parallel axis: qwen2.5-7b q4_K_M, ctx=8192 ---")
for p in (1, 2, 4):
    e = estimate(PRESETS["qwen2.5-7b"], Precision.Q4_K_M, num_ctx=8192, num_parallel=p)
    print(
        f"parallel={p}  {e}  fits={e.fits_in(VRAM)!s:<5} "
        f"gpu_ratio={e.predicted_gpu_ratio(VRAM):.2f}"
    )
print()
