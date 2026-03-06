import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/events": "http://localhost:5001",
      "/infer": "http://localhost:5001",
      "/gallery-image": "http://localhost:5001",
    },
  },
});
