import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { defineConfig } from "@playwright/test";

// Real-browser checks (UX4 layout/contrast, U2 timing, drag-to-connect, offline reports).
// Needs a built frontend (`pnpm build`) and installed Chrome. `pnpm e2e` starts `lab serve` itself.
const root = resolve(import.meta.dirname, "..");
const data = join(tmpdir(), `emmlab-e2e-${process.pid}`);
// E2E_PORT picks a free port when 8765 is taken by something else.
const port = Number(process.env.E2E_PORT ?? 8765);

export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    channel: "chrome",
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
  },
  webServer: {
    command: `uv run lab serve --port ${port} --data ${data}`,
    cwd: root,
    url: `http://127.0.0.1:${port}/api/presets`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
