"""Data for the operator console (web/), generated from one platform day and from reports/.

python3 -m evals.console_data      -> web/public/data/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

from evals.operate import operate
from world.traffic import Predict, Upload

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "public" / "data"
SEED = 101
STEP_S = 600.0


def fleet_day() -> dict:
    run = operate("platform", "world", SEED, snapshot_every=STEP_S)
    meta = {e.digest: {"rows": e.rows, "features": e.features, "tenant": e.tenant} for e in run["events"] if isinstance(e, Upload)}
    predicts = {e.request_id: e for e in run["events"] if isinstance(e, Predict)}
    frames = []
    for snap in run["snapshots"]:
        gpus = []
        for name, g in snap["plane"].items():
            caches = sorted(({"id": d[:8], "bytes": b, **meta.get(d, {})} for d, b in g["hbm"].items()), key=lambda c: -c["bytes"])
            gpus.append(
                {
                    "name": name,
                    "kind": g["kind"],
                    "hbm_bytes": run["plane"].gpus[name].hbm_bytes,
                    "true_capacity": snap["capacity"][name],
                    "capacity_estimate": g["capacity_estimate"],
                    "caches": caches,
                    "host_caches": len(g["host"]),
                    "host_bytes": sum(g["host"].values()),
                    "queued": g["queued"],
                    "busy": g["running"] is not None,
                }
            )
        frames.append({"t": snap["t"], "gpus": gpus})
    windows = []
    for f in frames:
        lo, hi = f["t"], f["t"] + STEP_S
        outs = [o for o in run["outcomes"] if lo <= o.arrived < hi and o.request_id in predicts and not predicts[o.request_id].is_retry]
        ok = [o for o in outs if o.status == "ok"]
        windows.append(
            {
                "t": lo,
                "requests": len(outs),
                "in_slo": sum(1 for o in ok if o.latency <= predicts[o.request_id].slo_s),
                "throttled": sum(1 for o in outs if o.status == "throttled"),
                "oom": sum(1 for x in run["log"] if x.oom and lo <= x.start < hi),
                "fits": sum(1 for x in run["log"] if x.ok and x.kind == "fit_predict" and lo <= x.start < hi),
                "hits": sum(1 for x in run["log"] if x.ok and x.kind != "fit_predict" and lo <= x.start < hi),
            }
        )
    certs = [
        {
            "tenant": c["tenant"],
            "requested_s": c["requested"],
            "datasets": len(c["digests"]),
            "verified": c["verified"],
            "seconds": (c["completed"] - c["requested"]) if c["completed"] is not None else None,
        }
        for c in run["certificates"]
    ]
    audits = run["audits"]
    return {"seed": SEED, "step_s": STEP_S, "frames": frames, "windows": windows, "erasures": certs, "audits": audits}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fleet.json").write_text(json.dumps(fleet_day(), separators=(",", ":")))
    dev = json.loads((ROOT / "reports" / "dev.json").read_text())
    designs = {
        "seeds": dev["seeds"],
        "hours": dev["hours"],
        "gates": dev["gates"],
        "designs": [
            {
                "name": n,
                "description": d["description"],
                "summary": {k: v["mean"] for k, v in d["summary"].items()},
                "ci": {k: v["ci95"] for k, v in d["summary"].items()},
            }
            for n, d in dev["designs"].items()
        ],
    }
    (OUT / "designs.json").write_text(json.dumps(designs, separators=(",", ":")))
    for name in ("bench", "advisor", "heldout"):
        src = ROOT / "reports" / f"{name}.json"
        if src.exists():
            data = json.loads(src.read_text())
            if name == "advisor":
                data.pop("truth_grid", None)
            if name == "heldout":
                data = {
                    "gates": data["gates"],
                    "designs": [{"name": n, "summary": {k: v["mean"] for k, v in d["summary"].items()}} for n, d in data["designs"].items()],
                }
            (OUT / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")))
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.json"))))


if __name__ == "__main__":
    main()
