"""Turn results/*.json into the charts and table the README needs.

Axis labels are English on purpose: matplotlib's default font has no CJK
glyphs, so Chinese labels render as empty boxes unless a font is shipped.

Two design decisions worth stating, because the first version got both wrong:

1. **Nothing is plotted in label order.** Sorting configurations
   alphabetically puts 14b-ctx16384 before 3b-ctx4096 and turns the
   throughput line into noise. Each chart holds one axis of the experiment,
   sorted by the quantity that axis varies.

2. **The y-axis says bytes, not layers.** `gpu_ratio` is size_vram / size,
   and size includes the KV cache and the unexplained per-token term. Neither
   is a layer. Calling it "layers resident" would overclaim.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .estimator import (  # noqa: E402
    MEASURED_GPU_BUDGET_GB,
    PRESETS,
    Precision,
    estimate,
)

RESULTS = Path("results")
IMAGES = Path("docs/images")

PURPLE, ORANGE, TEAL, GREY = "#7F77DD", "#D85A30", "#1D9E75", "#888780"

PLOT_STYLE = {
    "figure.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 9,
}

# The P=4 row reports 9.014 GiB resident on a 7.996 GiB card: the Windows
# driver is presenting host RAM as VRAM. It is qualitative evidence of
# overcommit and supports no quantitative claim, so it stays out of the
# charts and the accuracy statistics. See results/day6_matrix.md section 8.
EXCLUDE_FROM_CLAIMS = {"7b-q4-ctx32768-p4"}

# Day 4 wrote these under auto-generated labels before the matrix existed.
# They duplicate 7b-q4-ctx2048 and 7b-q4-ctx32768 exactly.
STALE_LABELS = {"qwen2.5-7b-ctx2048-p1", "qwen2.5-7b-ctx32768-p1"}


def load_results(directory: str | Path = RESULTS) -> pd.DataFrame:
    """Flatten every bench_*.json into one row per configuration."""
    rows = []
    for path in sorted(Path(directory).glob("bench_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d["label"] in STALE_LABELS:
            continue
        measured = d.get("measured", {})
        rows.append(
            {
                "label": d["label"],
                "model": d["model"],
                "num_ctx": d["num_ctx"],
                "num_parallel": d.get("num_parallel", 1),
                "size_gb": measured.get("size_gb"),
                "vram_gb": measured.get("vram_gb"),
                "gpu_ratio": measured.get("gpu_ratio"),
                "decode_tps": d.get("decode_tps_mean"),
                "ttft_p50_s": d.get("ttft_p50_s"),
                "baseline_gb": d.get("baseline_gb"),
            }
        )
    return pd.DataFrame(rows)


def add_predictions(
    df: pd.DataFrame,
    preset_map: dict[str, tuple[str, Precision]],
    budget_gb: float = MEASURED_GPU_BUDGET_GB,
) -> pd.DataFrame:
    """Attach estimator output next to every measurement."""
    pred_total, pred_ratio = [], []
    for _, row in df.iterrows():
        entry = preset_map.get(row["model"])
        if entry is None:
            pred_total.append(None)
            pred_ratio.append(None)
            continue
        key, precision = entry
        e = estimate(
            PRESETS[key],
            precision,
            num_ctx=int(row["num_ctx"]),
            num_parallel=int(row["num_parallel"]),
        )
        pred_total.append(round(e.total_gb, 3))
        pred_ratio.append(round(e.predicted_gpu_ratio(budget_gb), 4))

    df = df.copy()
    df["predicted_total_gb"] = pred_total
    df["predicted_gpu_ratio"] = pred_ratio
    df["size_error"] = df["predicted_total_gb"] - df["size_gb"]
    df["ratio_error"] = df["predicted_gpu_ratio"] - df["gpu_ratio"]
    return df


def _dual_axis(ax, x, gpu_pct, tps, xticklabels, title, xlabel):
    ax.bar(x, gpu_pct, color=PURPLE, alpha=0.75, width=0.6, zorder=2)
    ax.set_ylabel("GPU-resident share of reported bytes (%)", color="#534AB7")
    ax.set_ylim(0, 108)
    ax.tick_params(axis="y", labelcolor="#534AB7")
    ax.set_xticks(list(x))
    ax.set_xticklabels(xticklabels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)

    right = ax.twinx()
    right.plot(x, tps, "o-", color=ORANGE, linewidth=2, markersize=5, zorder=3)
    right.set_ylabel("Decode throughput (tok/s)", color="#993C1D")
    right.tick_params(axis="y", labelcolor="#993C1D")
    right.set_ylim(0, max(tps) * 1.18)
    right.grid(False)
    right.spines["top"].set_visible(False)
    return right


def plot_cliff(
    df: pd.DataFrame,
    model_tag: str = "qwen2.5:7b",
    size_ctx: int = 4096,
    out: str | Path = IMAGES / "cliff.png",
) -> Path:
    """The money chart: context axis on the left, size/quantization on the right.

    Both panels share the same encoding — bars are GPU residency, the line is
    decode throughput — so the reader learns to read it once.
    """
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    work = df[~df["label"].isin(EXCLUDE_FROM_CLAIMS)].dropna(
        subset=["gpu_ratio", "decode_tps"]
    )

    ctx = (
        work[(work["model"] == model_tag) & (work["num_parallel"] == 1)]
        .sort_values("num_ctx")
        .reset_index(drop=True)
    )
    size = (
        work[(work["num_ctx"] == size_ctx) & (work["num_parallel"] == 1)]
        .sort_values("size_gb")
        .reset_index(drop=True)
    )

    with plt.rc_context(PLOT_STYLE):
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))

        _dual_axis(
            a,
            range(len(ctx)),
            ctx["gpu_ratio"] * 100,
            ctx["decode_tps"],
            [f"{c // 1024}K" for c in ctx["num_ctx"]],
            f"Context length — {model_tag} q4_K_M",
            "num_ctx",
        )
        _dual_axis(
            b,
            range(len(size)),
            size["gpu_ratio"] * 100,
            size["decode_tps"],
            [lbl.replace(f"-ctx{size_ctx}", "") for lbl in size["label"]],
            f"Model size and quantization — ctx {size_ctx}",
            "",
        )

        fig.suptitle(
            "Where an 8GB GPU gives up, and what it costs", fontsize=12, y=0.99
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out)
        plt.close(fig)
    return Path(out)


def _scatter_panel(ax, measured, predicted, labels, lo, hi, xlabel, ylabel, title):
    ax.plot([lo, hi], [lo, hi], "--", color=GREY, linewidth=1, zorder=1)
    ax.scatter(measured, predicted, s=55, color=TEAL, zorder=3)

    # Points that land on the same spot get one shared annotation. Every
    # fully-resident configuration sits at exactly (1.0, 1.0) on the ratio
    # panel, and labelling each of them separately produced an unreadable
    # pile in the first version of this chart.
    span = hi - lo
    groups: dict[tuple[float, float], list[str]] = {}
    for m, p, lbl in zip(measured, predicted, labels, strict=True):
        key = (round(m / span, 2), round(p / span, 2))
        groups.setdefault(key, []).append(lbl)

    for i, ((kx, ky), members) in enumerate(sorted(groups.items())):
        x, y = kx * span, ky * span
        text = members[0] if len(members) == 1 else f"{len(members)} configs here"
        dx, dy = (7, 5) if i % 2 == 0 else (7, -11)
        ax.annotate(
            text,
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=6.5,
            color="#555",
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="box")


def plot_prediction_vs_actual(
    df: pd.DataFrame, out: str | Path = IMAGES / "prediction.png"
) -> Path:
    """Two panels: predicted VRAM demand, and predicted GPU residency.

    The size panel matters more than the ratio panel. Ratio is bounded in
    [0,1] and saturates at 1 for every resident configuration, which flatters
    the estimator; predicted GiB spans 2 to 13 and has nowhere to hide.
    """
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    work = df[~df["label"].isin(EXCLUDE_FROM_CLAIMS)].dropna(
        subset=["gpu_ratio", "predicted_gpu_ratio", "size_gb", "predicted_total_gb"]
    )
    short = [
        lbl.replace("qwen2.5-", "").replace("-q4", "").replace("ctx", "")
        for lbl in work["label"]
    ]

    with plt.rc_context(PLOT_STYLE):
        fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 5))

        hi = max(work["size_gb"].max(), work["predicted_total_gb"].max()) * 1.08
        _scatter_panel(
            a,
            work["size_gb"].to_numpy(),
            work["predicted_total_gb"].to_numpy(),
            short,
            0,
            hi,
            "Measured SIZE (GiB)",
            "Predicted SIZE (GiB)",
            f"VRAM demand — mean error {work['size_error'].abs().mean():.3f} GiB",
        )
        _scatter_panel(
            b,
            work["gpu_ratio"].to_numpy(),
            work["predicted_gpu_ratio"].to_numpy(),
            short,
            0.3,
            1.08,
            "Measured GPU ratio",
            "Predicted GPU ratio",
            f"GPU residency — mean error "
            f"{100 * work['ratio_error'].abs().mean():.1f} pp",
        )

        fig.suptitle("Estimator accuracy (dashed line = perfect)", fontsize=12, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out)
        plt.close(fig)
    return Path(out)


def markdown_table(df: pd.DataFrame) -> str:
    """The prediction-vs-actual table. The centrepiece of the README."""
    cols = [
        "label",
        "num_ctx",
        "size_gb",
        "predicted_total_gb",
        "size_error",
        "gpu_ratio",
        "predicted_gpu_ratio",
        "decode_tps",
        "ttft_p50_s",
    ]
    present = [c for c in cols if c in df.columns]
    ordered = df.sort_values(["model", "num_parallel", "num_ctx"])
    return ordered[present].to_markdown(index=False, floatfmt=".3f")


def build_all(preset_map: dict[str, tuple[str, Precision]], **kwargs) -> None:
    df = add_predictions(load_results(), preset_map, **kwargs)
    if df.empty:
        print("No results found in results/. Run the bench first.")
        return

    plot_cliff(df)
    plot_prediction_vs_actual(df)

    table = markdown_table(df)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.md").write_text(table + "\n", encoding="utf-8")

    scored = df[~df["label"].isin(EXCLUDE_FROM_CLAIMS)]
    print(table)
    print(
        f"\n{len(scored)} rows scored "
        f"({len(df) - len(scored)} excluded: {', '.join(sorted(EXCLUDE_FROM_CLAIMS))})"
    )
    print(f"  mean |SIZE error|      {scored['size_error'].abs().mean():.3f} GiB")
    print(
        f"  mean |GPU ratio error| "
        f"{100 * scored['ratio_error'].abs().mean():.1f} pp"
    )
    print("\nWrote docs/images/cliff.png, docs/images/prediction.png, results/summary.md")
