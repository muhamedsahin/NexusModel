import { Article, s, tx } from "./types";

export const performanceGuide: Article = {
  slug: "performance-engine",
  group: 3,
  title: tx("Optimize hesap motoru", "Optimized compute engine"),
  description: tx("Bloklu matris hesabı, doğruluk ve ölçülen kazanımlar.", "Blocked matrix computation, correctness, and measured improvements."),
  source: "src/simd/avx2/details.txt",
  sections: [
    s("matrix", "Matris motoru", "Matrix engine",
      "Bu depodaki Linear, 6×16 çıktı bloğunu 12 bağımsız FMA akümülatörüyle hesaplar. 128 eleman derinliğindeki ağırlık paneli 8 KiB yerel tamponda tutulur ve bütün girdi satırlarında kullanılır. Küçük batch için dört bağımsız toplama zincirli dot yolu seçilir. Ağırlık paketleme her çağrının süresine dahildir. 21 Ekim 2026 akşamının rakip protokolünde duran GEMM çekirdeği avx2-4x16’dır; N=2048 GFLOPS sütunu (Nexus 330, 362, 369; TensorFlow 389) bu 6×16 karosunun ölçümü değildir.",
      "Linear in this repository computes a 6×16 output tile with 12 independent FMA accumulators. A 128-deep weight panel occupies 8 KiB of local storage and is reused across input rows. Small batches use a four-chain dot path. Packing is timed on every call. The GEMM kernel kept on the 2026-10-21 evening competitor protocol is avx2-4x16; the N=2048 GFLOPS column (Nexus 330, 362, 369; TensorFlow 389) is not a measurement of this 6×16 tile."),
    s("backward", "Geri geçiş ve convolution", "Backward and convolution",
      "dX ve dW, geçici transpoz oluşturmadan aynı bloklu çekirdeği kullanır. İstenmeyen gradyan ürünleri hesaplanmaz. Gradyanlar mevcut değerlere eklenir. Convolution im2col düzeninden doğrudan kanal-major çıktıya matris çarpımı yapar; ileri ve geri geçiş aynı motoru paylaşır.",
      "dX and dW share the blocked kernel without materialized transposes. Unrequested gradient products are skipped. Gradients accumulate into existing values. Convolution multiplies the im2col layout directly into channel-major output; forward and backward share the same engine."),
    s("precision", "Hız ve sayısal doğruluk", "Speed and numerical accuracy",
      "LayerNorm iki geçişli float64 toplam/varyans hesabını SIMD ile hızlandırır. Büyük ortalama ve küçük varyansta hassasiyet korunur. Genel fast-math açılmaz. Kernel testleri rastgele girdiler, 16/128 sınırlarının iki tarafı, tek satır, bias, opsiyonel gradyanlar, tekrarlı birikim ve gruplu convolution içerir.",
      "LayerNorm vectorizes two-pass float64 sum/variance reductions, preserving accuracy for large means and small variance. Global fast-math is disabled. Kernel tests cover random inputs, both sides of 16/128 boundaries, single rows, bias, optional gradients, repeated accumulation, and grouped convolution."),
    s("layout", "Dosya düzeni", "Source organization",
      "src/core altında convolution, normalization, pooling, embedding ve dropout ayrı dosyalardadır. src/simd/avx2 altında matris mikroçekirdeği, Linear ve LayerNorm ayrılmıştır. Her klasörde details.txt, veri düzenini ve kararların nedenini açıklar. Dış C++ API korunmuştur.",
      "src/core separates convolution, normalization, pooling, embedding, and dropout. src/simd/avx2 separates the matrix microkernel, Linear, and LayerNorm. Each folder's details.txt explains layouts and design decisions. The public C++ API is preserved."),
    s("measure", "Sonuçları nasıl okuyalım?", "Reading the results",
      "Benchmark sayfasında 29 Eylül 2026 NumPy/PyTorch karşılaştırması ile eski/yeni NexusModel süreleri ayrıdır. Üç turda sıra değişir; her tur 20 ısınma ve 30×5 çağrıdır. 21 Ekim 2026 akşamı, bu Ryzen 7 5800H, ayrı protokoldür. GEMM avx2-4x16 kaldı. N=2048 Nexus 330, 362 ve 369 GFLOPS; TensorFlow 389. CPU Conv2d PyTorch 31.6 GFLOPS gerisindeydi; Nexus yaklaşık 4.9 ile 6.8 GFLOPS arasındaydı. Depthwise AVX, PyTorch 4.93 ve TensorFlow 0.688 önündeydi. AVX-512 seçilmez. Dünya sıralaması yoktur. GPU verileri eski arşivdir.",
      "The benchmark page keeps the 2026-09-29 NumPy/PyTorch comparison apart from the old/new NexusModel times. Three rounds rotate order; each has 20 warmups and 30×5 calls. The evening of 2026-10-21 on this Ryzen 7 5800H is a separate protocol. GEMM kept avx2-4x16. At N=2048 Nexus recorded 330, 362, and 369 GFLOPS; TensorFlow recorded 389. CPU Conv2d was behind PyTorch at 31.6 GFLOPS; Nexus was about 4.9 to 6.8 GFLOPS. Depthwise AVX was ahead of PyTorch 4.93 and TensorFlow 0.688. AVX-512 is not selected. There is no world ranking. GPU data is a historical archive.",
      { code: "cmake -S . -B build\ncmake --build build --config Release\nctest --test-dir build -C Release --output-on-failure\ncmake -S docs/examples -B docs/.example-build\ncmake --build docs/.example-build --config Release\npython docs/benchmarks/compare_libraries.py" }),
  ],
};
