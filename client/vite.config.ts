import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// BOT_URL / CLIENT_PORT let a second UI front a second bot (e.g. the fully local copy).
const target = process.env.BOT_URL ?? "http://127.0.0.1:7860"
const port = Number(process.env.CLIENT_PORT ?? 5173)
const proxy = Object.fromEntries(
  ["/start", "/sessions", "/api/offer", "/status", "/models"].map((route) => [
    route,
    { target, changeOrigin: true },
  ])
)
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "./src") } },
  server: {
    host: "127.0.0.1",
    port,
    strictPort: true,
    allowedHosts: [".ngrok.app", ".ts.net"],
    proxy,
  },
  preview: {
    host: "127.0.0.1",
    port,
    strictPort: true,
    allowedHosts: [".ngrok.app", ".ts.net"],
    proxy,
  },
})
