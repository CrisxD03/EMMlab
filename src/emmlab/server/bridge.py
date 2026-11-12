"""Socket-free API bridge: JSON lines on stdin -> FastAPI app in-process -> JSON lines on stdout.

Used by frontend integration tests (and any environment that cannot bind ports): each request line is
``{"id": ..., "method": "GET", "path": "/api/...", "body": {...}}``; each response line is
``{"id": ..., "status": 200, "contentType": "...", "body": "<text>"}``. Server-sent-event routes return
a single snapshot event with a short ``retry`` so EventSource keeps polling.
"""

from __future__ import annotations

import json
import sys
import warnings


def main() -> None:
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

    from .app import app, db

    client = TestClient(app)
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        rid, method, path = req.get("id"), req.get("method", "GET").upper(), req["path"]
        try:
            if path.endswith("/events") and path.startswith("/api/experiments/"):
                jid = path.split("/")[3]
                j = db.get(jid) or {}
                payload = {"status": j.get("status"), "done": j.get("done"), "total": j.get("total")}
                body, status, ctype = f"retry: 500\ndata: {json.dumps(payload)}\n\n", 200, "text/event-stream"
            else:
                kw = {"json": req["body"]} if req.get("body") is not None else {}
                r = client.request(method, path, **kw)
                body, status, ctype = r.text, r.status_code, r.headers.get("content-type", "")
        except Exception as e:  # report, keep serving
            body, status, ctype = json.dumps({"detail": repr(e)}), 500, "application/json"
        out.write(json.dumps({"id": rid, "status": status, "contentType": ctype, "body": body}) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
