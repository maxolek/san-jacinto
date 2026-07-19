import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist',
  },
  server: {
    port: 3000,
    open: true,
  },
  optimizeDeps: {
    exclude: ['@duckdb/duckdb-wasm'],
  },
  worker: {
    format: 'es',
  },
});
