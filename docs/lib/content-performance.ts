import { Article, s, tx } from "./types";

export const performanceGuide: Article = {
  slug: "performance-engine",
  group: 3,
  title: tx("Optimize hesap motoru", "Optimized compute engine"),
  description: tx("Bloklu matris hesabı, doğruluk ve ölçülen kazanımlar.", "Blocked matrix computation, correctness, and measured improvements."),
  source: "src/simd/avx2/details.txt",
  sections: [
    s("matrix", "Matris motoru", "Matrix engine",
      "Linear, 6×16 çıktı bloğunu 12 bağımsız FMA akümülatörüyle hesaplar. 128 eleman derinliğindeki ağırlık paneli 8 KiB yerel tamponda tutulur ve bütün girdi satırlarında kullanılır. Küçük batch için dört bağımsız toplama zincirli dot yolu seçilir. Ağırlık paketleme her çağrının süresine dahildir; ağırlıkları değiştirmek için önbellek temizleme gerekmez.",
      "Linear computes a 6×16 output tile with 12 independent FMA accumulators. A 128-deep weight panel occupies 8 KiB of local storage and is reused across input rows. Small batches use a four-chain dot path. Packing is timed on every call; modifying weights requires no cache invalidation."),
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
      "Benchmark sayfasında NumPy ve PyTorch karşılaştırmaları ile eski/yeni NexusModel sonuçları ayrı gösterilir. Üç turda çalışma sırası değiştirilir; her tur 20 ısınma ve 30×5 çağrı içerir. Tek makine ölçümü genel dünya sıralaması değildir. Küçük işlemlerde Python çağrı maliyeti görünür. GPU verileri eski sürüm arşividir; bu CPU optimizasyonuyla yeniden ölçülmemiştir.",
      "The benchmark page separates NumPy/PyTorch comparisons from old/new NexusModel measurements. Three rounds rotate execution order; each has 20 warmups and 30×5 calls. One machine does not establish a worldwide ranking. Python dispatch matters for small operations. GPU data is historical and was not remeasured for this CPU optimization.",
      { code: "cmake -S . -B build\ncmake --build build --config Release\nctest --test-dir build -C Release --output-on-failure\ncmake -S docs/examples -B docs/.example-build\ncmake --build docs/.example-build --config Release\npython docs/benchmarks/compare_libraries.py" }),
  ],
};
