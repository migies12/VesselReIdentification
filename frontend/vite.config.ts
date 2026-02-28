import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/events": "http://localhost:5000",
      "/infer": "http://localhost:5000",
      "/gallery-image": "http://localhost:5000",
    },
  },
});
