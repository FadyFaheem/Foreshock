import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// API proxy target: defaults to :8000 for host dev (macOS reserves :5000 for
// AirPlay). The Podman pod sets VITE_API_PROXY=http://localhost:5000.
const apiTarget = process.env.VITE_API_PROXY ?? 'http://localhost:8000'

// Hostnames allowed through Vite's host check (e.g. a Cloudflare Tunnel host).
const allowedHosts = (process.env.VITE_ALLOWED_HOSTS ?? '')
  .split(',')
  .map((h) => h.trim())
  .filter(Boolean)

export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts,
    // Polling is needed when the source is bind-mounted into a container VM
    // (Podman/libkrun, WSL2), where inotify file events don't propagate. The
    // dev pod sets VITE_USE_POLLING=1; host dev leaves it off for performance.
    watch:
      process.env.VITE_USE_POLLING === '1'
        ? { usePolling: true, interval: 1000 }
        : undefined,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})
