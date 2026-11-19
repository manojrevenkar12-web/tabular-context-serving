"""Usage ledger: one charge per idempotency key, whatever the client resends.

SQLite with a primary key on the charge key: a second insert for the same key is ignored, so a retried request is
never charged twice. Designs without idempotency charge by request id, which is what double-charges a retry.
"""

from __future__ import annotations

import sqlite3


class Ledger:
    def __init__(self, path: str = ":memory:") -> None:
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS charges (key TEXT PRIMARY KEY, tenant TEXT NOT NULL, rows INTEGER NOT NULL, gpu_ms INTEGER NOT NULL, at REAL NOT NULL)"
        )

    def charge(self, key: str, tenant: str, rows: int, gpu_s: float, at: float) -> bool:
        """True if this key was charged now, False if it had been charged before."""
        cur = self.db.execute("INSERT OR IGNORE INTO charges VALUES (?, ?, ?, ?, ?)", (key, tenant, int(rows), int(round(gpu_s * 1000)), at))
        return cur.rowcount == 1

    def usage(self, tenant: str) -> dict:
        rows, gpu_ms, n = self.db.execute("SELECT COALESCE(SUM(rows),0), COALESCE(SUM(gpu_ms),0), COUNT(*) FROM charges WHERE tenant = ?", (tenant,)).fetchone()
        return {"tenant": tenant, "requests": n, "rows": rows, "gpu_seconds": gpu_ms / 1000}

    def charged_keys(self) -> list[tuple[str, str]]:
        return list(self.db.execute("SELECT key, tenant FROM charges"))
