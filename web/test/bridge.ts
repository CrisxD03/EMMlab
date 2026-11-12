// Socket-free connection to the REAL backend for integration tests.
//
// Spawns `python -m emmlab.server.bridge` (FastAPI app in-process, JSON lines on stdin/stdout) and
// installs a `fetch` that routes /api/* through it, plus an EventSource shim (the bridge answers
// SSE routes with one snapshot event and `retry: 500`, so the shim reconnects like a browser would).

import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

/** Repository root (the uv project); vitest.config.ts passes it in. */
export const ROOT = process.env.EMMLAB_ROOT ?? resolve(process.cwd(), "..");

interface BridgeResponse {
  id: number;
  status: number;
  contentType: string;
  body: string;
}

export class Bridge {
  private proc: ChildProcessWithoutNullStreams | null = null;
  private pending = new Map<number, (r: BridgeResponse) => void>();
  private buf = "";
  private nextId = 1;
  readonly dataDir: string;
  /** Log of every request (method + path), for tests that check which endpoints the UI used. */
  readonly calls: { method: string; path: string; body?: unknown }[] = [];

  constructor() {
    this.dataDir = mkdtempSync(join(process.env.TMPDIR ?? tmpdir(), "emmlab-web-test-"));
  }

  start() {
    if (this.proc) return;
    this.proc = spawn("uv", ["run", "--project", ROOT, "python", "-m", "emmlab.server.bridge"], {
      env: { ...process.env, EMMLAB_DATA: this.dataDir, PYTHONUNBUFFERED: "1" },
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.proc.stdout.setEncoding("utf8");
    this.proc.stdout.on("data", (chunk: string) => {
      this.buf += chunk;
      let i = this.buf.indexOf("\n");
      while (i >= 0) {
        const line = this.buf.slice(0, i).trim();
        this.buf = this.buf.slice(i + 1);
        if (line) {
          try {
            const r = JSON.parse(line) as BridgeResponse;
            this.pending.get(r.id)?.(r);
            this.pending.delete(r.id);
          } catch {
            /* non-JSON output (warnings): ignore */
          }
        }
        i = this.buf.indexOf("\n");
      }
    });
    this.proc.stderr.setEncoding("utf8");
    this.proc.stderr.on("data", (d: string) => {
      if (process.env.EMMLAB_BRIDGE_DEBUG) process.stderr.write(d);
    });
    this.proc.on("exit", (code) => {
      for (const [, resolve] of this.pending)
        resolve({ id: -1, status: 599, contentType: "text/plain", body: `bridge exited (${code})` });
      this.pending.clear();
      this.proc = null;
    });
  }

  request(method: string, path: string, body?: unknown): Promise<BridgeResponse> {
    this.start();
    const id = this.nextId++;
    this.calls.push({ method, path, body });
    return new Promise((resolve) => {
      this.pending.set(id, resolve);
      this.proc!.stdin.write(`${JSON.stringify({ id, method, path, body: body ?? null })}\n`);
    });
  }

  async json<T = unknown>(method: string, path: string, body?: unknown): Promise<T> {
    const r = await this.request(method, path, body);
    return JSON.parse(r.body) as T;
  }

  stop() {
    this.proc?.stdin.end();
    this.proc?.kill();
    this.proc = null;
  }

  /** fetch() for the app under test: /api/* goes to the bridge. */
  fetch: typeof fetch = async (input, init) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const u = new URL(url, "http://localhost");
    if (!u.pathname.startsWith("/api/")) throw new Error(`unexpected fetch ${url}`);
    const raw = init?.body;
    const body = typeof raw === "string" && raw ? JSON.parse(raw) : undefined;
    const r = await this.request(init?.method ?? "GET", u.pathname + u.search, body);
    return new Response(r.body, { status: r.status, headers: { "content-type": r.contentType } });
  };
}

/** Minimal EventSource on top of the bridge fetch (reconnects after `retry` like a browser). */
export function makeEventSource(doFetch: () => typeof fetch) {
  return class BridgeEventSource extends EventTarget {
    static readonly CONNECTING = 0;
    static readonly OPEN = 1;
    static readonly CLOSED = 2;
    readonly url: string;
    readyState = 0;
    onmessage: ((e: MessageEvent) => void) | null = null;
    onerror: ((e: Event) => void) | null = null;
    onopen: ((e: Event) => void) | null = null;
    private timer: ReturnType<typeof setTimeout> | undefined;

    constructor(url: string | URL) {
      super();
      this.url = String(url);
      void this.connect();
    }

    private async connect() {
      if (this.readyState === 2) return;
      let retry = 500;
      try {
        const r = await doFetch()(this.url);
        if (this.readyState === 2) return;
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.readyState = 1;
        const open = new Event("open");
        this.onopen?.(open);
        this.dispatchEvent(open);
        const text = await r.text();
        for (const block of text.split("\n\n")) {
          let data = "";
          let type = "message";
          for (const line of block.split("\n")) {
            if (line.startsWith("retry:")) retry = Number(line.slice(6).trim()) || retry;
            else if (line.startsWith("data:")) data += (data ? "\n" : "") + line.slice(5).trim();
            else if (line.startsWith("event:")) type = line.slice(6).trim();
          }
          if (!data || this.readyState === 2) continue;
          const ev = new MessageEvent(type, { data });
          if (type === "message") this.onmessage?.(ev);
          this.dispatchEvent(ev);
        }
      } catch {
        const err = new Event("error");
        this.onerror?.(err);
        this.dispatchEvent(err);
      }
      if (this.readyState !== 2) {
        this.readyState = 0;
        this.timer = setTimeout(() => void this.connect(), retry);
      }
    }

    close() {
      this.readyState = 2;
      clearTimeout(this.timer);
    }
  };
}
