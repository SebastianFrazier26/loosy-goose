import { defineConfig } from "vite";

// Relative base: the same build serves from /loosy-goose/ on Pages, a local preview, or an iframe.
export default defineConfig({
  base: "./",
  build: { target: "es2022" },
});
