import fs from "node:fs";
const quick = fs
  .readFileSync("lib/quick-code.ts", "utf8")
  .match(/export const quickCode\s*=\s*(`[\s\S]*?`);/)[1];
const train = fs
  .readFileSync("lib/content-reference.ts", "utf8")
  .match(/export const trainingCode\s*=\s*(`[\s\S]*?`);/)[1];
fs.writeFileSync("examples/quickstart.cpp", Function("return " + quick)());
fs.writeFileSync("examples/train_mse.cpp", Function("return " + train)());
