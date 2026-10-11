import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

const proxy = Object.fromEntries(
  ["/start", "/sessions", "/api/offer", "/status"].map((route) => [
    route,
    { target: "http://127.0.0.1:7860", changeOrigin: true },
  ])
)
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "./src") } },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    allowedHosts: [".ngrok.app"],
    proxy,
  },
  preview: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    allowedHosts: [".ngrok.app"],
    proxy,
  },
})
