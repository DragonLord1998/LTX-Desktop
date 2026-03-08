/**
 * vite.config.web.ts
 *
 * Standalone web build config — no Electron plugins.
 * Output is a static site that FastAPI can serve directly.
 *
 * Build with:  pnpm run build:web
 * Output dir:  dist-web/
 */
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './frontend'),
    },
  },
  // Use absolute paths so FastAPI can serve the SPA from /
  base: '/',
  build: {
    outDir: 'dist-web',
    emptyOutDir: true,
  },
  // Root is the project root; index.html is at the top level
  root: '.',
})
