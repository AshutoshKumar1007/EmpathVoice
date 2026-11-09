import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, proxy API + probe routes to the FastAPI backend so the frontend can
// use same-origin relative URLs (no CORS, matching production where FastAPI
// serves the built assets).
const backend = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": backend,
      "/health": backend,
      "/ready": backend,
    },
  },
  build: {
    outDir: "dist",
  },
});
