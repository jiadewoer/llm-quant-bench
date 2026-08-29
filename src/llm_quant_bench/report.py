"""Turn results/*.json into the two charts and one table the README needs.

Axis labels are English on purpose. matplotlib's default font has no CJK
glyphs, so Chinese labels render as empty boxes unless you ship a font --
and an English-facing repo wants English axes anyway.
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

PLOT_STYLE = {
    "figure.figsize": (8, 4.5),
    "figure.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
}


def load_results(directory: str | Path = RESULTS) -> pd.DataFrame:
    """Flatten every bench_*.json into one row per configuration."""
    rows = []
    for path in sorted(Path(directory).glob("bench_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        measured = d.get("measured", {})
        rows.append(
            {
                "label": d["label"],
                "model": d["model"],
                "num_ctx": d["num_ctx"],
                "num_parallel": d.get("num_parallel", 1),
                "size_gb": measured.get("size_gb"),
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
    """Attach estimator output next to every measurement.

    preset_map maps an ollama tag to (PRESETS key, Precision), e.g.
        {"qwen2.5:7b": ("qwen2.5-7b", Precision.Q4_K_M)}
    """
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
    df["ratio_error"] = df["predicted_gpu_ratio"] - df["gpu_ratio"]
    return df


def plot_cliff(df: pd.DataFrame, out: str | Path = IMAGES / "cliff.png") -> Path:
    """GPU residency and throughput against configuration -- the money chart."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    data = df.dropna(subset=["gpu_ratio", "decode_tps"]).reset_index(drop=True)

    with plt.rc_context(PLOT_STYLE):
        fig, ax1 = plt.subplots()
        x = range(len(data))

        ax1.bar(x, data["gpu_ratio"] * 100, color="#7F77DD", alpha=0.75, width=0.6)
        ax1.set_ylabel("Layers resident on GPU (%)", color="#534AB7")
        ax1.set_ylim(0, 105)
        ax1.tick_params(axis="y", labelcolor="#534AB7")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(data["label"], rotation=30, ha="right", fontsize=8)

        ax2 = ax1.twinx()
        ax2.plot(x, data["decode_tps"], "o-", color="#D85A30", linewidth=2)
        ax2.set_ylabel("Decode throughput (tok/s)", color="#993C1D")
        ax2.tick_params(axis="y", labelcolor="#993C1D")
        ax2.grid(False)

        ax1.set_title("Offload cliff on an 8GB GPU")
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    return Path(out)


def plot_prediction_vs_actual(
    df: pd.DataFrame, out: str | Path = IMAGES / "prediction.png"
) -> Path:
    """How well the estimator did. The diagonal is a perfect prediction."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    data = df.dropna(subset=["gpu_ratio", "predicted_gpu_ratio"])

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(5.2, 5))
        ax.plot([0, 1], [0, 1], "--", color="#888780", linewidth=1)
        ax.scatter(
            data["gpu_ratio"], data["predicted_gpu_ratio"], s=70, color="#1D9E75"
        )
        for _, r in data.iterrows():
            ax.annotate(
                r["label"],
                (r["gpu_ratio"], r["predicted_gpu_ratio"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
            )
        ax.set_xlabel("Measured GPU ratio")
        ax.set_ylabel("Predicted GPU ratio")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Estimator accuracy")
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    return Path(out)


def markdown_table(df: pd.DataFrame) -> str:
    """The prediction-vs-actual table. This is the centerpiece of the README."""
    cols = [
        "label",
        "num_ctx",
        "size_gb",
        "gpu_ratio",
        "predicted_gpu_ratio",
        "ratio_error",
        "decode_tps",
        "ttft_p50_s",
    ]
    present = [c for c in cols if c in df.columns]
    return df[present].to_markdown(index=False, floatfmt=".3f")


def build_all(preset_map: dict[str, tuple[str, Precision]], **kwargs) -> None:
    df = add_predictions(load_results(), preset_map, **kwargs)
    if df.empty:
        print("No results found in results/. Run the bench first.")
        return
    plot_cliff(df)
    plot_prediction_vs_actual(df)
    table = markdown_table(df)
    Path(RESULTS / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print("\nWrote docs/images/cliff.png, docs/images/prediction.png, "
          "results/summary.md")
