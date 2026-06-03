import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The Flask server is reachable as `flame3d-core:5005` on the docker-compose
// network, and as `localhost:5005` when running the dev server outside docker.
// `VITE_API_TARGET` lets docker-compose override the proxy target.
const apiTarget = process.env.VITE_API_TARGET ?? "http://localhost:5005";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // listen on 0.0.0.0 so the port is reachable from the host
    port: 5173,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
});
