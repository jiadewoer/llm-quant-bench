"""Command line entry point. Installed as `lqb` by pyproject.toml."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import monitor
from .bench import run_bench
from .estimator import MEASURED_GPU_BUDGET_GB, PRESETS, Precision, estimate
from .evaluate import review as review_eval
from .evaluate import run_eval

app = typer.Typer(add_completion=False, help="VRAM and throughput benchmarks for Ollama.")
console = Console()

RESULTS = Path("results")


def _precision(name: str) -> Precision:
    for p in Precision:
        if p.tag.casefold() == name.casefold():
            return p
    raise typer.BadParameter(f"unknown precision {name!r}; try q4_K_M or q8_0")


@app.command()
def estimate_cmd(
    model: str = typer.Option(..., "--model", "-m", help="PRESETS key, e.g. qwen2.5-7b"),
    precision: str = typer.Option("q4_K_M", "--precision", "-p"),
    ctx: int = typer.Option(2048, "--ctx"),
    parallel: int = typer.Option(1, "--parallel"),
    budget: float = typer.Option(
        MEASURED_GPU_BUDGET_GB, "--budget", help="GiB Ollama will actually use (measured)"
    ),
) -> None:
    """Predict VRAM demand for one configuration."""
    if model not in PRESETS:
        raise typer.BadParameter(f"unknown model; known: {', '.join(PRESETS)}")
    e = estimate(PRESETS[model], _precision(precision), num_ctx=ctx, num_parallel=parallel)
    ratio = e.predicted_gpu_ratio(budget)
    fits = e.fits_in(budget)

    console.print(f"[bold]{model}[/bold] {precision} ctx={ctx} parallel={parallel}")
    console.print(f"  weights   {e.weights_gb:6.2f} GiB")
    console.print(f"  kv cache  {e.kv_cache_gb:6.2f} GiB")
    console.print(f"  extra     {e.extra_context_gb:6.2f} GiB  (empirical, mechanism unknown)")
    console.print(f"  total     {e.total_gb:6.2f} GiB  of {budget:.2f} budget")
    verdict = "[green]fits[/green]" if fits else "[yellow]needs CPU offload[/yellow]"
    console.print(f"  {verdict}, predicted GPU ratio {ratio:.0%}")


app.command("estimate")(estimate_cmd)


@app.command()
def ps() -> None:
    """What Ollama currently has loaded, and how much of it is on the GPU."""
    loaded = monitor.ps()
    if not loaded:
        console.print("[dim]nothing loaded[/dim]")
    else:
        table = Table("model", "size", "vram", "gpu")
        for m in loaded:
            table.add_row(
                m.name, f"{m.size_gb:.2f} GB", f"{m.vram_gb:.2f} GB", f"{m.gpu_ratio:.0%}"
            )
        console.print(table)

    used = monitor.gpu_used_mib()
    total = monitor.gpu_total_mib()
    if used is not None and total is not None:
        console.print(f"[dim]GPU {used / 1024:.2f} / {total / 1024:.2f} GiB in use[/dim]")


@app.command()
def bench(
    model: str = typer.Option(..., "--model", "-m", help="ollama tag, e.g. qwen2.5:7b"),
    ctx: int = typer.Option(2048, "--ctx"),
    parallel: int = typer.Option(
        1, "--parallel", help="Recorded only; set OLLAMA_NUM_PARALLEL in the environment"
    ),
    n: int = typer.Option(8, "--requests", "-n"),
    label: str = typer.Option("", "--label"),
) -> None:
    """Benchmark one configuration and write results/bench_<label>.json."""
    label = label or f"{model.replace(':', '-')}-ctx{ctx}-p{parallel}"
    console.print(f"[bold]{label}[/bold] warming up...")
    result = asyncio.run(run_bench(label, model, num_ctx=ctx, num_parallel=parallel, n_requests=n))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"bench_{label}.json"
    out.write_text(result.to_json(), encoding="utf-8")
    console.print_json(json.dumps(result.summary(), ensure_ascii=False))
    console.print(f"[dim]wrote {out}[/dim]")


@app.command()
def matrix(
    config: Path = typer.Option(Path("configs/matrix.json"), "--config", "-c"),
) -> None:
    """Run every row of the benchmark matrix, sequentially."""
    rows = json.loads(config.read_text(encoding="utf-8"))["runs"]
    RESULTS.mkdir(exist_ok=True)

    for i, row in enumerate(rows, 1):
        label = row["label"]
        console.rule(f"[{i}/{len(rows)}] {label}")
        result = asyncio.run(
            run_bench(
                label,
                row["model"],
                num_ctx=row.get("num_ctx", 2048),
                num_parallel=row.get("num_parallel", 1),
                n_requests=row.get("n_requests", 8),
            )
        )
        (RESULTS / f"bench_{label}.json").write_text(result.to_json(), encoding="utf-8")
        s = result.summary()
        gpu = s.get("measured", {}).get("gpu_ratio")
        console.print(
            f"  gpu={gpu:.0%} " if gpu is not None else "  gpu=? ",
            f"tps={s.get('decode_tps_mean')} ttft={s.get('ttft_p50_s')}s",
        )


@app.command("eval")
def eval_cmd(
    model: str = typer.Option(..., "--model", "-m"),
    ctx: int = typer.Option(2048, "--ctx"),
    eval_set: Path = typer.Option(Path("data/eval_set.jsonl"), "--set"),
    limit: int = typer.Option(0, "--limit", help="0 means all"),
    label: str = typer.Option("", "--label"),
) -> None:
    """Score a model on the eval set."""
    label = label or f"{model.replace(':', '-')}-ctx{ctx}"
    result = run_eval(label, model, eval_set, num_ctx=ctx, limit=limit or None)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"eval_{label}.json"
    out.write_text(json.dumps(result.summary(), indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[bold]{label}[/bold] {result.correct}/{result.total} = {result.accuracy:.1%}")
    console.print(f"[dim]wrote {out}[/dim]")


@app.command("review")
def review_cmd(
    result: Path = typer.Argument(..., help="results/eval_<label>.json"),
    eval_set: Path = typer.Option(Path("data/eval_set.jsonl"), "--set"),
) -> None:
    """Show every wrong item with its question and the model's actual reply.

    Run this before trusting a score: it separates "the model got it wrong"
    from "the question was bad".
    """
    review_eval(result, eval_set)


@app.command()
def report(
    config: Path = typer.Option(Path("configs/matrix.json"), "--config", "-c"),
    budget: float = typer.Option(MEASURED_GPU_BUDGET_GB, "--budget"),
) -> None:
    """Build charts and the prediction-vs-actual table."""
    from .report import build_all

    spec = json.loads(config.read_text(encoding="utf-8"))
    preset_map = {
        tag: (entry["preset"], _precision(entry["precision"]))
        for tag, entry in spec["model_map"].items()
    }
    build_all(preset_map, budget_gb=budget)


if __name__ == "__main__":
    app()
