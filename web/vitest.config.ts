import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// Integration tests run the real FastAPI app through a socket-free stdin/stdout bridge
// (see test/bridge.ts), so they need no port and no browser.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: false,
    setupFiles: ["./test/setup.ts"],
    env: { EMMLAB_ROOT: resolve(import.meta.dirname, ".."), DEBUG_PRINT_LIMIT: "600" },
    include: ["test/**/*.test.{ts,tsx}"],
    testTimeout: 90_000,
    hookTimeout: 90_000,
    css: false,
    fileParallelism: true,
    maxWorkers: 4,
  },
});
