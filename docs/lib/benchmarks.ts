import cpu from "./cpu-results.json";
export const cpuRows = cpu;
// Historical GPU snapshot; not remeasured for this revision.
export const gpuRows = [
  { op: "ReLU", shape: "2²⁰", cpu: 0.394, gpu: 0.0232 },
  { op: "ReLU", shape: "2²⁴", cpu: 6.85, gpu: 0.329 },
  { op: "GELU tanh", shape: "2²⁰", cpu: 0.992, gpu: 0.028 },
  { op: "GELU tanh", shape: "2²⁴", cpu: 17.6, gpu: 0.334 },
  { op: "Softmax", shape: "256 × 256", cpu: 0.0762, gpu: 0.012 },
  { op: "Softmax", shape: "1024 × 1024", cpu: 0.926, gpu: 0.0318 },
  { op: "LayerNorm", shape: "256 × 256", cpu: 0.215, gpu: 0.0133 },
  { op: "LayerNorm", shape: "1024 × 1024", cpu: 3.38, gpu: 0.0283 },
  { op: "GEMM", shape: "32 × 256 × 256", cpu: 0.151, gpu: 0.0355 },
  { op: "GEMM", shape: "1024 × 1024 × 1024", cpu: 82.9, gpu: 0.248 },
  { op: "GEMM", shape: "4096 × 4096 × 4096", cpu: 10420, gpu: 20.2 },
];
