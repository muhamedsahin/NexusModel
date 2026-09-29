import fs from "node:fs";
import path from "node:path";
const root = path.resolve("..");
const out = path.resolve("public/source");
const entries = [
  "README.md",
  "CMakeLists.txt",
  "include",
  "src",
  "tests",
  "benchmarks",
  "examples",
  "docs/benchmarks",
  "docs/examples",
];
function copy(rel) {
  if (rel.split(path.sep).some((part) => part.startsWith("."))) return;
  const file = path.join(root, rel);
  if (fs.statSync(file).isDirectory()) {
    for (const name of fs.readdirSync(file)) copy(path.join(rel, name));
    return;
  }
  if (!/\.(hpp|cuh|cu|cpp|txt|md|py)$/.test(file)) return;
  const target = path.join(out, rel + ".txt");
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.copyFileSync(file, target);
}
entries.forEach(copy);
fs.copyFileSync(path.resolve("NexusModel-Kullanim-Kilavuzu.pdf"), path.resolve("public/NexusModel-Kullanim-Kilavuzu.pdf"));
fs.copyFileSync(
  path.join(root, "benchmarks/RESULTS.md"),
  path.resolve("public/benchmark-results.md"),
);
console.log("Source snapshots synchronized.");
