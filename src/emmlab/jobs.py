"""Experiment jobs persisted in SQLite: per-unit results make cancel/resume exact (X7)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .spec import ExperimentSpec

SCHEMA = """
create table if not exists jobs (id text primary key, name text, spec text, status text, created real,
                                 updated real, total integer, done integer, error text);
create table if not exists units (job_id text, key text, payload text, primary key (job_id, key));
"""


class JobDB:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.execute("pragma journal_mode=wal")
        return c

    def create(self, spec: ExperimentSpec, total: int) -> str:
        jid = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock, self._conn() as c:
            c.execute("insert into jobs values (?,?,?,?,?,?,?,?,?)",
                      (jid, spec.name, spec.model_dump_json(by_alias=True), "queued", now, now, total, 0, None))
        return jid

    def set_status(self, jid: str, status: str, error: str | None = None, total: int | None = None) -> None:
        with self._lock, self._conn() as c:
            if total is not None:
                c.execute("update jobs set total=? where id=?", (total, jid))
            c.execute("update jobs set status=?, updated=?, error=? where id=?", (status, time.time(), error, jid))

    def add_result(self, jid: str, res: dict[str, Any]) -> None:
        with self._lock, self._conn() as c:
            c.execute("insert or replace into units values (?,?,?)", (jid, res["unit"], json.dumps(res)))
            c.execute("update jobs set done=(select count(*) from units where job_id=?), updated=? where id=?",
                      (jid, time.time(), jid))

    def results(self, jid: str) -> dict[str, dict]:
        with self._conn() as c:
            rows = c.execute("select key, payload from units where job_id=?", (jid,)).fetchall()
        return {k: json.loads(p) for k, p in rows}

    def get(self, jid: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("select id,name,spec,status,created,updated,total,done,error from jobs where id=?",
                          (jid,)).fetchone()
        if r is None:
            return None
        return {"id": r[0], "name": r[1], "spec": json.loads(r[2]), "status": r[3], "created": r[4],
                "updated": r[5], "total": r[6], "done": r[7], "error": r[8]}

    def list(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("select id,name,status,created,total,done from jobs order by created desc").fetchall()
        return [{"id": r[0], "name": r[1], "status": r[2], "created": r[3], "total": r[4], "done": r[5]}
                for r in rows]

    def spec(self, jid: str) -> ExperimentSpec:
        j = self.get(jid)
        assert j is not None
        return ExperimentSpec.model_validate(j["spec"])
