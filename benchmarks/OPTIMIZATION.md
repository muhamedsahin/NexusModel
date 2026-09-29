# CPU optimization audit

## Findings and implemented changes

| Area inspected | Finding | Change / verification |
|---|---|---|
| Linear forward | Repeated row-wise dot products reload weights and serialize FMA chains | 6×16 register tile, 128-depth weight packing, small-batch fallback; independent float64 reference tests |
| Linear backward | One AXPY per output feature repeatedly reads/writes gradient rows | Blocked dX/dW products with transpose strides and optional outputs; repeated nonzero-gradient accumulation tests |
| Convolution | Dot/AXPY nested loops bypass matrix reuse | Express forward and backward through the shared matrix engine; grouped padded convolution checked against direct loops |
| LayerNorm | AVX2 entry delegated to scalar implementation | Two-pass FP64 vector statistics and FP32 affine output; large-offset, small-variance tests |
| Dot / softmax | Single FMA dependency chain / scalar maximum reduction | Four independent dot accumulators and vector maximum; tail and scalar-reference checks |
| CPU dispatch | FMA missing from feature checks; non-MSVC OS state not checked; table mutation could race | Explicit FMA/XGETBV checks, immutable tables, acquire/release pointer publication |
| Build portability | Global `-march=native` could insert unsupported instructions outside dispatch | ISA flags confined to specialized translation units; no global fast-math |
| Attention tape | Growing vector could invalidate live references; Add propagated only one branch | Stable deque slots, gradients to both inputs, 100-node growth/branch regression test |
| Attention constructor | Division by zero before head validation | Validate through a guarded dimension initializer |
| Source layout | Convolution, normalization, pooling, embedding and dropout shared one large implementation | Separate implementations with folder-level `details.txt`; stable public API |
| Benchmark/site | Old hardcoded results and too few module repetitions | Raw samples, three rotated rounds, expanded workloads, generated reports, explicit historical GPU labeling |

## Decisions

- Weight packing is rebuilt on every call and included in timing. Public weight
  tensors remain mutable, so a persistent cache without versioning would be unsafe.
- The GEMM workspace is 8 KiB on the stack. No new heap workspace or thread pool.
- A 256-depth candidate did not consistently improve backward; retain 128.
- Do not replace stable variance with float32 `E[x²] - E[x]²` or enable global
  fast-math to make a benchmark appear faster.
- AVX-512 compilation is retained; blocked Linear uses the AVX2 matrix engine.
  The measured Ryzen host does not validate AVX-512 execution.
- The initial module benchmark process crashed. Recompiling it restored execution;
  no usable pre-change module baseline was recorded. Do not compare its current
  medians to historical module timings as an isolated optimization effect.

## Scope and remaining work

Tensor ownership/reuse, module gradient accumulation, recurrent projections,
attention, normalization, convolution, activation dispatch, CUDA interfaces and
benchmark/documentation paths were reviewed. The changes target measured CPU
bottlenecks; they are not an exhaustive formal audit of every public API.

Full training throughput, GPU end-to-end performance, multiple machines, ARM,
multi-thread scaling, cold caches, all IEEE exceptional inputs, all possible
matrix shapes and AVX-512 execution are not established by these measurements.
CUDA kernels and MatrixFlash integration were not optimized in this pass. Large
GEMM can still favor BLAS-backed competitors. Full models still incur saved-input
copies; the cross-library study measures raw forward kernels, not those copies.

Numerical equivalence is tolerance-based, not bitwise: FMA and reduction order
change rounding. The independent C++ suite exercises scalar and automatic paths;
the Python study checks every element against a float64 reference.

## Evidence

`results/baseline-source.zip` preserves the pre-change source snapshot; its
include/src content digest was checked against the original baseline metadata.
`results/revision-comparison.json` records paired native measurements and binary
hashes. `docs/public/benchmark-comparison.json` records versions, hardware, BLAS,
source/binary/harness hashes and all timing samples. `RESULTS.md` is the readable
report. The archived source is for reproduction, not part of the live build.
