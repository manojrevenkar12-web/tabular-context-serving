"""Charts for the README, generated from reports/ (never drawn by hand).

python3 -m evals.charts
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "charts"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
HIGHLIGHT, OTHER, SECOND = "#2a78d6", "#b9b8b1", "#eb6834"
ORDER = ["platform", "lru_no_host", "no_batching", "fifo_no_quota", "no_cache_routing", "no_learning", "fixed_chunks", "llm_style", "baseline"]


def _style(ax) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def evaluation() -> None:
    dev = json.loads((ROOT / "reports" / "dev.json").read_text())
    names = [n for n in ORDER if n in dev["designs"]]
    panels = [
        ("slo_attainment", "Requests answered within their SLO", lambda v: f"{v * 100:.1f}%", False),
        ("eur_per_million_rows", "GPU EUR per million predicted rows", lambda v: f"{v:.2f}", False),
        ("oom_events", "Out-of-memory errors per 6-hour day", lambda v: f"{v:,.0f}", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (key, title, fmt, log) in zip(axes, panels, strict=True):
        vals = [dev["designs"][n]["summary"][key]["mean"] for n in names]
        errs = [dev["designs"][n]["summary"][key]["ci95"] or 0 for n in names]
        ys = range(len(names))
        colors = [HIGHLIGHT if n == "platform" else OTHER for n in names]
        ax.barh(ys, vals, height=0.55, color=colors, xerr=errs, error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 0})
        if log:
            ax.set_xscale("log")
        for y, v, e in zip(ys, vals, errs, strict=True):
            ax.text((v + e) * (1.08 if log else 1.0), y, "  " + fmt(v), va="center", fontsize=8.5, color=INK)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=10)
        _style(ax)
        if key == "slo_attainment":
            ax.set_xlim(0.4, 1.08)
        ax.margins(x=0.18)
    axes[0].set_yticks(range(len(names)), [n.replace("_", " ") for n in names], color=INK)
    axes[0].invert_yaxis()
    fig.text(0.01, 0.01, f"Mean over {len(dev['seeds'])} seeds with 95% intervals. Same traffic and the same GPUs for every design.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "evaluation.png", dpi=160)
    plt.close(fig)


def bench() -> None:
    b = json.loads((ROOT / "reports" / "bench.json").read_text())
    rows = [s["train_rows"] for s in b["speed"]]
    fit = [s["fit_and_predict_s"] for s in b["speed"]]
    hit = [s["predict_from_cache_s"] for s in b["speed"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    xs = range(len(rows))
    ax.bar([x - 0.18 for x in xs], fit, width=0.34, color=SECOND, label="fit and predict (cache miss)")
    ax.bar([x + 0.18 for x in xs], hit, width=0.34, color=HIGHLIGHT, label="predict from the KV cache")
    ax.set_yscale("log")
    for x, f, h in zip(xs, fit, hit, strict=True):
        ax.text(x - 0.18, f * 1.15, f"{f:.2f}s", ha="center", fontsize=8.5, color=INK)
        ax.text(x + 0.18, h * 1.15, f"{h:.3f}s", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(list(xs), [f"{r:,} training rows" for r in rows], color=INK)
    ax.set_title("100 test rows on the small PyTorch model, this machine's CPU", fontsize=10.5, color=INK, loc="left", pad=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.margins(y=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "bench.png", dpi=160)
    plt.close(fig)


def main() -> None:
    evaluation()
    bench()
    print("wrote", ", ".join(str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
