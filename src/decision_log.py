"""Persistent decision log (SQLite).

Every automated decision is reconstructable later: one row per stage per ticket carrying the minimum
record from the governance framework, plus a `tickets` table holding each ticket's final disposition.
`reconcile()` checks that the two agree with the set of tickets processed (acceptance criterion A8) -
gaps mean some path took a decision without recording it.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    ts          TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    ticket_id   TEXT NOT NULL,
    stage       TEXT NOT NULL,
    action      TEXT NOT NULL,
    record      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_decisions_ticket ON decisions(run_id, ticket_id);
CREATE TABLE IF NOT EXISTS tickets (
    run_id      TEXT NOT NULL,
    ticket_id   TEXT NOT NULL,
    ts          TEXT NOT NULL,
    disposition TEXT NOT NULL,
    PRIMARY KEY (run_id, ticket_id)
);
"""

STAGES = ("classification", "retrieval", "routing", "generation", "validation")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class DecisionLog:
    def __init__(self, path: Path | str, run_id: str | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.run_id = run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S-") + uuid.uuid4().hex[:6]
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._db.commit()
        self.write_errors = 0

    def record(self, ticket_id: str, stage: str, action: str, **fields) -> str:
        """Write one decision. Returns its id. Never raises: a logging fault is counted and surfaced in metrics."""
        decision_id = uuid.uuid4().hex
        ts = now_iso()
        rec = {"decision_id": decision_id, "timestamp": ts, "ticket_id": ticket_id, "stage": stage,
               "action_taken": action, **fields}
        try:
            with self._lock:
                self._db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?)",
                                 (decision_id, ts, self.run_id, ticket_id, stage, action,
                                  json.dumps(rec, default=str)))
                self._db.commit()
        except Exception:
            self.write_errors += 1
        return decision_id

    def finish_ticket(self, ticket_id: str, disposition: str) -> None:
        try:
            with self._lock:
                self._db.execute("INSERT OR REPLACE INTO tickets VALUES (?,?,?,?)",
                                 (self.run_id, ticket_id, now_iso(), disposition))
                self._db.commit()
        except Exception:
            self.write_errors += 1

    # ---------------------------------------------------------------- read side
    def decisions_for(self, ticket_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT record FROM decisions WHERE run_id=? AND ticket_id=? ORDER BY ts",
                                    (self.run_id, ticket_id)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def reconcile(self, processed_ids: list[str]) -> dict:
        """Compare logged decisions with tickets processed. `ok` only if every ticket has a final
        disposition and a routing decision, and nothing is logged for tickets that were not processed."""
        with self._lock:
            finished = {r[0] for r in self._db.execute("SELECT ticket_id FROM tickets WHERE run_id=?", (self.run_id,))}
            routed = {r[0] for r in self._db.execute(
                "SELECT DISTINCT ticket_id FROM decisions WHERE run_id=? AND stage='routing'", (self.run_id,))}
            total = self._db.execute("SELECT COUNT(*) FROM decisions WHERE run_id=?", (self.run_id,)).fetchone()[0]
            by_stage = dict(self._db.execute(
                "SELECT stage, COUNT(*) FROM decisions WHERE run_id=? GROUP BY stage", (self.run_id,)).fetchall())
        processed = set(processed_ids)
        report = {
            "tickets_processed": len(processed_ids),
            "tickets_with_final_disposition": len(finished & processed),
            "tickets_with_routing_decision": len(routed & processed),
            "decisions_logged": total,
            "decisions_by_stage": by_stage,
            "missing_disposition": sorted(processed - finished)[:20],
            "missing_routing": sorted(processed - routed)[:20],
            "unexpected_tickets": sorted((finished | routed) - processed)[:20],
            "log_write_errors": self.write_errors,
        }
        report["ok"] = (not report["missing_disposition"] and not report["missing_routing"]
                        and not report["unexpected_tickets"] and self.write_errors == 0
                        and len(processed_ids) == len(processed))
        return report

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass
