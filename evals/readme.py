"""Regenerate the README's results block from reports/. Never edit the block by hand.

python3 -m evals.readme
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- results:start -->", "<!-- results:end -->"
ORDER = ["platform", "baseline", "llm_style", "no_cache_routing", "fixed_chunks", "no_learning", "no_batching", "fifo_no_quota", "lru_no_host"]


def _load(name: str) -> dict | None:
    p = ROOT / "reports" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _pct(v: float | None, ci: float | None = None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" + (f" ± {ci * 100:.1f}" if ci else "")


def _num(v: float | None, fmt: str = "{:.2f}", ci: float | None = None) -> str:
    if v is None:
        return "n/a"
    return fmt.format(v) + (f" ± {fmt.format(ci)}" if ci else "")


def design_table(rep: dict, with_ci: bool) -> list[str]:
    lines = [
        "| Design | Requests in SLO | Interactive p95 | OOM errors / day | Failed requests | GPU EUR per M rows | Cache hit rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for n in [n for n in ORDER if n in rep["designs"]]:
        s = rep["designs"][n]["summary"]

        def c(k: str, s: dict = s) -> float | None:
            return s[k]["ci95"] if with_ci else None

        name = f"**{n}**" if n == "platform" else n
        lines.append(
            f"| {name} | {_pct(s['slo_attainment']['mean'], c('slo_attainment'))} | {_num(s['interactive_p95_s']['mean'], '{:,.1f}')} s "
            f"| {_num(s['oom_events']['mean'], '{:,.0f}')} | {_num(s['failed']['mean'], '{:,.0f}')} "
            f"| {_num(s['eur_per_million_rows']['mean'], '{:.2f}', c('eur_per_million_rows'))} | {_pct(s['cache_hit_rate']['mean'])} |"
        )
    return lines


def gate_table(rep: dict) -> list[str]:
    lines = ["| Gate | Value | Result |", "|---|---:|---|"]
    for g in rep["gates"]:
        lines.append(f"| {g['gate'].replace('_', ' ')} | {g['value']:.4g} | {'pass' if g['pass'] else '**fail**'} |")
    return lines


def render_block() -> str:
    dev = _load("dev")
    if dev is None:
        return f"{START}\n_No reports yet: run `python3 -m evals.run`._\n{END}"
    p = {k: v["mean"] for k, v in dev["designs"]["platform"]["summary"].items()}
    b = {k: v["mean"] for k, v in dev["designs"]["baseline"]["summary"].items()}
    llm = {k: v["mean"] for k, v in dev["designs"]["llm_style"]["summary"].items()}
    out = [
        START,
        "",
        f"Dev campaign: {len(dev['seeds'])} seeds × {dev['hours']:.0f} simulated hours on 8 GPUs (2 H100, 2 A100, 4 L4), "
        f"{p['requests']:,.0f} customer requests a day plus a flooding tenant, every design on the same traffic. "
        f"Commit `{dev['commit'][:7]}`.",
        "",
        "| | Status quo | LLM-style serving | **This plane** |",
        "|---|---:|---:|---:|",
        f"| Requests answered within their SLO | {_pct(b['slo_attainment'])} | {_pct(llm['slo_attainment'])} | **{_pct(p['slo_attainment'])}** |",
        f"| Interactive p95 latency | {b['interactive_p95_s']:,.0f} s | {llm['interactive_p95_s']:,.1f} s | **{p['interactive_p95_s']:.1f} s** |",
        f"| Out-of-memory errors per day | {b['oom_events']:,.0f} | {llm['oom_events']:,.0f} | **{p['oom_events']:,.0f}** |",
        f"| GPU EUR per million predicted rows | {b['eur_per_million_rows']:.2f} | {llm['eur_per_million_rows']:.2f} | **{p['eur_per_million_rows']:.2f}** |",
        f"| Context fits (forward passes over the training rows) | {b['fits']:,.0f} | {llm['fits']:,.0f} | **{p['fits']:,.0f}** |",
        f"| Retries charged twice | {b['double_charges']:,.0f} | {llm['double_charges']:,.0f} | **{p['double_charges']:,.0f}** |",
        f"| Erased bytes still on GPUs or in the bucket 15 min after erasure | {b['erasure_residual_bytes'] / 1e9:,.2f} GB | {llm['erasure_residual_bytes'] / 1e9:,.2f} GB | **{p['erasure_residual_bytes'] / 1e9:,.2f} GB** |",
        "",
        "![Evaluation](docs/charts/evaluation.png)",
        "",
        "<details><summary>Every design, with 95% intervals over seeds</summary>",
        "",
        *design_table(dev, True),
        "",
        *[f"- **{n}**: {d['description']}" for n, d in dev["designs"].items()],
        "",
        "</details>",
        "",
        "Release gates (config/gates.toml):",
        "",
        *gate_table(dev),
        "",
    ]
    held = _load("heldout")
    lock = ROOT / "reports" / "heldout.lock"
    if held:
        out += [
            f"**Held-out world** (run once at commit `{held['commit'][:7]}`, behind `reports/heldout.lock`): different hidden memory and "
            "speed coefficients, allocator overhead and noise, a throttled L4, more large contexts, a heavier flood, more retries. See docs/heldout.md.",
            "",
            *design_table(held, False),
            "",
            *gate_table(held),
            "",
        ]
    elif lock.exists():
        out += ["_Held-out run started but no report written._", ""]
    bench = _load("bench")
    if bench:
        s = bench["speed"][-1]
        m = bench["memory"]
        acc = bench["accuracy"]
        mean = {k: sum(v[k] for v in acc.values()) / len(acc) for k in ("tiny_icl", "logistic_regression", "knn_5")}
        out += [
            f"**The real model** (`scripts/bench.py`, PyTorch on this machine's CPU, {bench['parameters']:,} parameters):",
            "",
            f"- Cached and chunked predictions match the uncached forward pass to {max(bench['max_abs_diff_cached_vs_reference'], bench['max_abs_diff_chunked_vs_reference']):.1e} (largest absolute difference in a probability).",
            f"- At {s['train_rows']:,} training rows, 100 test rows take {s['fit_and_predict_s']:.2f} s with a fit and {s['predict_from_cache_s']:.3f} s from the KV cache ({s['speedup']:.0f}×). The cache costs exactly {bench['kv_bytes_per_row']:,} bytes per training row.",
            f"- Fitting {m['train_rows']:,} rows with whole-context attention grows resident memory by {m['whole_context_peak_bytes'] / 1e6:,.0f} MB; in 512-row query chunks by {m['chunked_512_peak_bytes'] / 1e6:,.0f} MB, with the same answers.",
            f"- Mean accuracy over {len(acc)} datasets: small model {mean['tiny_icl'] * 100:.1f}%, logistic regression {mean['logistic_regression'] * 100:.1f}%, 5-NN {mean['knn_5'] * 100:.1f}%. "
            "It learns in the forward pass but is a toy next to TabPFN; it is here so the serving code runs a real model.",
            "",
            "![Benchmark](docs/charts/bench.png)",
            "",
        ]
    adv = _load("advisor")
    if adv:
        sm = adv["summary"]
        out += [
            f"**Sizing advisor** for customer VPC deployments (`evals/advisor_eval.py`, {len(adv['profiles'])} customer workloads, target {adv['target'] * 100:.0f}% of requests in SLO, "
            "each choice then run in the simulated world to see what really happens):",
            "",
            "| Choice | Meets the target in truth | Mean EUR per hour |",
            "|---|---:|---:|",
        ]
        for k, label in (
            ("advisor", "**Advisor** (replays the customer's log through this plane)"),
            ("two_h100", "Two H100s"),
            ("one_card_that_holds_the_largest", "One card that holds the largest context"),
        ):
            v = sm[k]
            out.append(f"| {label} | {v['meets_target']} of {v['profiles']} | {v['mean_eur_per_hour']:.2f} |")
        oracle = [pr["oracle"]["eur_per_hour"] for pr in adv["profiles"] if pr["oracle"]]
        out += [
            "",
            f"The cheapest pools that really meet the target average EUR {sum(oracle) / max(1, len(oracle)):.2f} per hour. "
            "The advisor pays more on two busy workloads, where two L4s were enough and it chose A100s.",
            "",
        ]
    out.append(END)
    return "\n".join(out)


def main() -> None:
    path = ROOT / "README.md"
    text = path.read_text()
    a, b = text.index(START), text.index(END) + len(END)
    path.write_text(text[:a] + render_block() + text[b:])
    print("README results block regenerated")


if __name__ == "__main__":
    main()
