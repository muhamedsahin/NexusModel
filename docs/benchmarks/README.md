# Reproducible CPU measurements

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

The revision script rebuilds the archived original sources with the same current native timing harness. Three rounds alternate original/optimized order. It writes before/after medians and 90 raw samples per operation to optimization-results.json. Backward includes clearing dx/dw.

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

Generated outputs: RESULTS.md, cpu-results.json, comparison-results.json, optimization-results.json, public CSV/JSON downloads, HTML/PDF guide and Next.js static export. Archived original source and pre-change observations are under benchmarks/results; temporary baseline builds are ignored under benchmarks/.baseline.
