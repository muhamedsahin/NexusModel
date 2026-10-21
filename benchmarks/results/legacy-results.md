# Benchmark

> **21 Ekim 2026 notu.** Bu dosya tarihsel bir gündür; aşağıdaki sayılar yerinde bırakıldı. Ryzen 7 5800H üzerinde AVX-512 seçilmez. Aynı makinenin 21 Ekim 2026 akşam protokolü `benchmarks/RESULTS.md` başındaki nottadır: GEMM çekirdeği `avx2-4x16`, N=2048 Nexus 330/362/369 ve TensorFlow 389 GFLOPS; CPU Conv2d PyTorch 31.6 GFLOPS gerisinde, Nexus yaklaşık 4.9–6.8 GFLOPS; Depthwise AVX o sütunda PyTorch 4.93 ve TensorFlow 0.688 önünde. Bu oranlar ham günlüğün üzerine yazılmaz.
>
> Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

CPU: Release, MSVC 19.44, `/O2`, AVX2/AVX-512. GPU: NVIDIA GeForce RTX 3070 Laptop GPU, CUDA 13.4, `sm_86`. GPU süreleri `cudaEvent` ile ölçülür; aktarım (H2D/D2H) dahil değildir. Çekirdek zaten cihazda duruyormuş gibi sayılır.

## CPU katmanları

Süre, ısınma sonrası bir `forward` + `backward` çiftidir. ReLU yalnız `forward`.

| op | shape | ms |
|---|---|---|
| linear | 64 × 784 → 256 | 2.91 |
| linear scalar | 32 × 256 → 256 | 2.70 |
| linear SIMD | 32 × 256 → 256 | 0.413 |
| ReLU scalar | 1 × 2^20 | 0.619 |
| ReLU SIMD | 1 × 2^20 | 0.421 |
| Conv2D | 8 × 16 × 28 × 28, 32 filtre, k=3 | 22.1 |
| MultiHeadAttention | 4 × 32 × 64, 4 baş | 0.818 |

Aynı 256-geniş lineer katmanda SIMD yol skalar yoldan yaklaşık 6.5 kat kısadır. ReLU farkı daha küçüktür; skalar döngü derleyici tarafından da vektörlenir.

## CPU ve GPU

`nexus_model_bench_gpu`. ReLU ve GELU, NexusModel CUDA çekirdekleridir. Softmax ve LayerNorm fused cihaz çekirdekleridir. GEMM, aynı `Y = X Wᵀ` düzeninde CPU AVX `linear_forward` ile cuBLAS `sgemm` karşılaştırmasıdır. Ampere’de cuBLAS SGEMM varsayılan olarak TF32 tensor çekirdeği kullanır. En büyük mutlak sapma 4096’lık çarpımda 1.8e-5 oldu.

| op | şekil | CPU ms | GPU ms |
|---|---|---:|---:|
| ReLU | 2^20 | 0.394 | 0.0232 |
| ReLU | 2^24 | 6.85 | 0.329 |
| GELU tanh | 2^20 | 0.992 | 0.0280 |
| GELU tanh | 2^24 | 17.6 | 0.334 |
| softmax | 256 × 256 | 0.0762 | 0.0120 |
| softmax | 1024 × 1024 | 0.926 | 0.0318 |
| LayerNorm | 256 × 256 | 0.215 | 0.0133 |
| LayerNorm | 1024 × 1024 | 3.38 | 0.0283 |
| GEMM | 32 × 256 × 256 | 0.151 | 0.0355 |
| GEMM | 1024 × 1024 × 1024 | 82.9 | 0.248 |
| GEMM | 4096 × 4096 × 4096 | 10420 | 20.2 |

Küçük ReLU’da GPU yaklaşık 17 kat, 16 milyon elemanda yaklaşık 21 kat kısadır. 1024’lük çarpımda GPU yaklaşık 330 kat, 4096’lık çarpımda yaklaşık 500 kat kısadır. 32 × 256 × 256 çarpımında kazanç 4 katta kalır; çekirdek kısa sürdüğü için başlatma maliyeti görünür.
