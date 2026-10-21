# Reproducible CPU measurements

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

## 2026-10-21 evening protocol

This machine, evening of 2026-10-21, AMD Ryzen 7 5800H. There is no world ranking.

GEMM kept kernel `avx2-4x16`. At N=2048 Nexus recorded 330, 362, and 369 GFLOPS; TensorFlow recorded 389 GFLOPS.

CPU Conv2d on the competitor protocol was behind PyTorch at 31.6 GFLOPS. Nexus that evening was about 4.9 to 6.8 GFLOPS.

Depthwise AVX was ahead of PyTorch 4.93 and TensorFlow 0.688 on that evening column.

AVX-512 is not selected on the Ryzen 7 5800H.

The commands below reproduce the 2026-09-29 single-thread harness (small GEMM and ReLU, millisecond medians). A lowest median in 7 of those 9 workloads describes that harness only. Do not overwrite raw logs under `benchmarks/results/` with mixed ratios from the evening GFLOPS column. Historical logs stay labeled as historical; `benchmarks/results/legacy-results.md` points at this protocol.

## September harness

Run from the repository root. Install NumPy, CPU PyTorch and threadpoolctl in an isolated Python environment (see requirements.txt).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
cmake -S docs/examples -B docs/.example-build -DCMAKE_BUILD_TYPE=Release
cmake --build docs/.example-build --config Release
python docs/benchmarks/compare_revisions.py --prepare
python docs/benchmarks/compare_libraries.py
```

The revision script rebuilds the archived original sources with the same 2026-09-29 native timing harness. Three rounds alternate original/optimized order. It writes before/after medians and 90 raw samples per operation to optimization-results.json. Backward includes clearing dx/dw.

The cross-library script supports Release subdirectories and single-config generators. It rotates NexusModel, NumPy and PyTorch through first/middle/last positions over three rounds. Each round uses 20 warmups and 30 samples of 5 calls. Median and interpolated p95 summarize 90 sample means, not independent-call tail latency. There are 6 GEMM shapes (including batch-one and irregular tails) and 3 ReLU sizes: 27 results total.

Scope: CPU float32 forward-only GEMM without bias and ReLU, preallocated outputs, one compute thread. NexusModel weight packing is included. Python dispatch is included for NumPy/PyTorch; C++ dispatch for NexusModel. This is not full training, autograd, transfers or full-module memory copying. NumPy is an array library, not a neural-network framework.

Full outputs are checked against float64 every round (rtol 1e-4, atol 1e-5); ReLU is exact. Failed validation prevents publication. Inputs are deterministic periodic patterns; independent native tests additionally use seeded random/adversarial inputs.

One machine; warm caches, uncontrolled affinity, power and thermals. Near ties can reverse. JSON records CPU/compiler/package/BLAS information, source/executable/harness hashes, and all samples. GPU tables are historical and not remeasured.

## Refresh all artifacts

Save the output of `build/Release/nexus_model_bench.exe` to `benchmarks/results/optimized-modules.csv` (PowerShell: `| Set-Content -Encoding utf8 ...`). Module medians use 20 warmups and 30×5 calls, one round.

```sh
python docs/benchmarks/export_reports.py
cd docs
node scripts/render-guide.mjs
npm run build
npm run typecheck
npm start -- --port 3100
# another terminal
npm test
```

Generated outputs from that refresh: RESULTS.md, cpu-results.json, comparison-results.json, optimization-results.json, public CSV/JSON downloads, and the Next.js static export. `node scripts/render-guide.mjs` prints `kilavuz.html` to the PDF. On 2026-10-21 that PDF step was not run. Pandoc is not installed. The usage-guide source is `docs/kullanim.md`. Archived original source and pre-change observations are under benchmarks/results; temporary baseline builds are ignored under benchmarks/.baseline. Do not fold the 2026-10-21 GFLOPS column into those raw logs.
