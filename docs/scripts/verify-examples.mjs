import { spawnSync } from "node:child_process";
const env = Object.fromEntries(
  Object.entries(process.env).map(([k, v]) => [k.toUpperCase(), v]),
);
for (const [cmd, args] of [
  ["cmake", ["--fresh", "-S", "examples", "-B", ".example-build"]],
  ["cmake", ["--build", ".example-build", "--config", "Release"]],
  [
    "ctest",
    ["--test-dir", ".example-build", "-C", "Release", "--output-on-failure"],
  ],
]) {
  const result = spawnSync(cmd, args, { env, stdio: "inherit" });
  if (result.status !== 0) process.exit(result.status ?? 1);
}
