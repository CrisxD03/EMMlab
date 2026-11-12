import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `uv run lab serve` serves the build: index.html at /, bundles under /assets.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/",
  build: {
    outDir: "../src/emmlab/server/static",
    emptyOutDir: true,
    assetsDir: "assets",
    chunkSizeWarningLimit: 2000,
  },
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
});
