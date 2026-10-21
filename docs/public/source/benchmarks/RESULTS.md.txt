# NexusModel CPU benchmark report

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

**21 Ekim 2026 akşamı, bu makine (AMD Ryzen 7 5800H).** Aşağıdaki tablolar 29 Eylül 2026 tek iş parçacıklı çalışmadır. Dünya sıralaması yoktur. Rakip protokolünde GEMM çekirdeği `avx2-4x16` olarak kaldı. N=2048 için Nexus 330, 362 ve 369; TensorFlow 389. Birim o protokolün GFLOPS sütunudur. CPU Conv2d aynı protokolde PyTorch’un 31.6 GFLOPS değerinin gerisinde kaldı; Nexus o akşam yaklaşık 4.9 ile 6.8 GFLOPS arasındaydı. Depthwise AVX, o akşamki sütunda PyTorch 4.93 ve TensorFlow 0.688 değerlerinin önündeydi. AVX-512 Ryzen 7 5800H üzerinde seçilmez. Bu GFLOPS sütunu aşağıdaki milisaniye tablolarının üzerine yazılmaz. Ham günlükler `benchmarks/results/` altındadır.

Measured: 2026-09-29T19:13:14.340268+00:00

AMD Ryzen 7 5800H with Radeon Graphics · Windows-11-10.0.26200-SP0 · MSVC 19.44.34823.2 / Release /O2

Float32, one compute thread. Three rotated rounds, 20 warmups per round, 30 samples per round, 5 calls per sample. Tables report medians of 90 sample means; p95 and all raw samples are in JSON.

## Same-harness revision comparison

| Operation | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| linear_forward_128_512_512 | 5.583850 | 1.833240 | 3.05× |
| linear_backward_128_512_512 | 14.219690 | 4.022970 | 3.53× |
| softmax_128_512 | 0.142120 | 0.087740 | 1.62× |
| layernorm_128_512 | 0.311520 | 0.091510 | 3.40× |

Baseline rebuilt from the archived, digest-verified source snapshot. Same native harness for both revisions; backward includes dx/dw clearing. Revision order alternates. This compares the library revisions, not scalar versus SIMD.

## NexusModel / NumPy / PyTorch

| Operation / shape | NexusModel ms | NumPy ms | PyTorch ms | Lowest median |
|---|---:|---:|---:|---|
| GEMM 1 × 256 × 256 | 0.006020 | 0.013710 | 0.019380 | NexusModel |
| GEMM 7 × 129 × 33 | 0.003940 | 0.015930 | 0.007290 | NexusModel |
| GEMM 32 × 256 × 256 | 0.154630 | 0.194050 | 0.142130 | PyTorch |
| GEMM 96 × 768 × 192 | 0.741060 | 0.892920 | 0.808200 | NexusModel |
| GEMM 128 × 512 × 512 | 1.765330 | 1.907980 | 1.861630 | NexusModel |
| GEMM 256 × 1024 × 1024 | 15.062200 | 13.732030 | 14.438170 | NumPy |
| ReLU 65,536 | 0.018090 | 0.100000 | 0.028980 | NexusModel |
| ReLU 1,048,576 | 0.238820 | 1.387470 | 0.311280 | NexusModel |
| ReLU 4,194,304 | 2.198630 | 5.541180 | 2.312790 | NexusModel |

In this 2026-09-29 table, NexusModel recorded the lowest median in 7 of the 9 workloads. That sentence describes only this single-thread table. It does not replace the 2026-10-21 competitor protocol at the top of this file, and it is not a world ranking.

Versions: NumPy 2.3.5; PyTorch 2.14.0+cpu.

## Full CPU modules

Medians from this 2026-09-29 module harness: 20 warmups and 30×5 calls (one round). Linear/Conv/MHA include forward and backward; ReLU forward only. These use the full module API with saved inputs. The Conv2D row is a millisecond on that harness. It is not the 2026-10-21 competitor-protocol GFLOPS column, where CPU Conv2d was behind PyTorch at 31.6 GFLOPS and Nexus was about 4.9 to 6.8 GFLOPS. No valid pre-change module baseline was recorded.

| Module | Shape | Median ms |
|---|---|---:|
| Linear | 64 × 784 → 256 | 2.180140 |
| Linear · scalar | 32 × 256 → 256 | 5.303940 |
| Linear · SIMD | 32 × 256 → 256 | 0.400470 |
| ReLU · scalar | 1 × 2²⁰ | 0.652760 |
| ReLU · SIMD | 1 × 2²⁰ | 0.625370 |
| Conv2D | 8 × 16 × 28 × 28 · K32 · k3 | 34.522300 |
| MultiHeadAttention | 4 × 32 × 64 · H4 | 0.685370 |

## Correctness and limits

- Every cross-library output element was checked each round against float64 (rtol 1e-4, atol 1e-5); ReLU is exact.
- Native tests independently cover seeded random matrices, tails, bias, optional gradients, repeated accumulation, grouped convolution, stable LayerNorm and tape growth.
- NexusModel uses C++ calls; NumPy/PyTorch use Python APIs. Tiny-case timings include unequal dispatch overhead.
- Preallocated outputs and warm caches; deterministic periodic benchmark inputs. Weight packing is timed. This is not full training or model throughput.
- OS scheduling, affinity, thermals and power are uncontrolled. Close results can reverse between runs.
- GPU numbers are historical and are not refreshed or mixed into the 2026-09-29 CPU tables or the 2026-10-21 GFLOPS column.

## Reproduce and raw evidence

See `docs/benchmarks/README.md`. Machine metadata, loaded BLAS backends, package versions, source/binary/harness hashes and raw timing samples are in `docs/public/benchmark-comparison.json`. Revision evidence: `benchmarks/results/revision-comparison.json`. Archived sources: `benchmarks/results/baseline-source.zip`. Historical report: `benchmarks/results/legacy-results.md`.
