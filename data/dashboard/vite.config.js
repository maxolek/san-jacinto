import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  // Relative assets work at both a domain root and a repository Pages subpath.
  base: './',
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
