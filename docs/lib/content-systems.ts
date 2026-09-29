import { Article, s, tx } from "./types";
export const systems: Article[] = [
  {
    slug: "initialization",
    group: 3,
    title: tx("Ağırlık başlatma", "Weight initialization"),
    description: tx(
      "Kaiming, Xavier, orthogonal ve tekrar üretim.",
      "Kaiming, Xavier, orthogonal, and reproducibility.",
    ),
    source: "include/nexus_model/core/initializer.hpp",
    sections: [
      s(
        "random",
        "Tohum ve temel işlemler",
        "Seeds and primitives",
        "init::manual_seed(uint64_t) thread-local mt19937 motorunu tohumlar; seed result_type’a dönüştürülür. constant_, zeros_, ones_, uniform_(low,high), normal_(mean,stddev) yerinde doldurur. Seed yeni parametreler oluşturulmadan önce verilmelidir.",
        "init::manual_seed(uint64_t) seeds a thread-local mt19937; the seed is cast to result_type. constant_, zeros_, ones_, uniform_(low,high), normal_(mean,stddev) fill in place. Set the seed before creating parameters.",
      ),
      s(
        "fans",
        "Fan-in ve fan-out",
        "Fan-in and fan-out",
        "[out,in,spatial...] için fan_in=in·çekirdek_hacmi ve fan_out=out·çekirdek_hacmi. rank<2 için max(numel,1). Başlatma, aktivasyon ve türev ölçeğinin derinlik boyunca büyüyüp küçülmesini azaltmayı amaçlar.",
        "For [out,in,spatial...], fan_in=in·kernel_volume and fan_out=out·kernel_volume. For rank<2, both equal max(numel,1). Initialization aims to reduce growth or decay of activation and gradient scales with depth.",
      ),
      s(
        "kaiming",
        "Kaiming",
        "Kaiming",
        "kaiming_uniform_/kaiming_normal_(tensor,negative_slope=0,fan_out=false). Gain √(2/(1+a²)); normal std=gain/√fan, uniform sınır=√3·gain/√fan. Linear ve Conv varsayılanı uniform.",
        "kaiming_uniform_/kaiming_normal_(tensor,negative_slope=0,fan_out=false). Gain √(2/(1+a²)); normal std=gain/√fan, uniform bound=√3·gain/√fan. Linear and Conv default to uniform.",
        {
          formula:
            "W\\sim U\\left(-\\sqrt{\\frac6{(1+a^2)fan}},\\sqrt{\\frac6{(1+a^2)fan}}\\right)",
        },
      ),
      s(
        "xavier",
        "Xavier ve orthogonal",
        "Xavier and orthogonal",
        "xavier_uniform_/xavier_normal_(tensor,gain=1) fan_in+fan_out kullanır. orthogonal_(tensor,gain=1) rank-2 kabul eder; normal örnekleri Gram–Schmidt ile ortogonalleştirir. Recurrent varsayılan gain=1; farklı gain için kendi sayısal testinizi yapın.",
        "xavier_uniform_/xavier_normal_(tensor,gain=1) use fan_in+fan_out. orthogonal_(tensor,gain=1) accepts rank-2 and applies Gram–Schmidt to normal samples. Recurrent defaults use gain=1; verify other gains numerically.",
        { formula: "\\sigma_{Xavier}=g\\sqrt{\\frac2{fan_{in}+fan_{out}}}" },
      ),
    ],
  },
  {
    slug: "serialization",
    group: 3,
    title: tx("Kaydetme ve yükleme", "Saving & loading"),
    description: tx(
      "StateDict, NXM1 biçimi ve checkpoint sınırları.",
      "StateDict, NXM1 format, and checkpoint limitations.",
    ),
    source: "include/nexus_model/core/serialization.hpp",
    sections: [
      s(
        "snapshot",
        "StateDict anlık görüntüsü",
        "StateDict snapshots",
        "state_dict parametre verilerini ve buffer’ları clone eder. Anahtarlar 0.weight veya norm.running_mean gibi hiyerarşiktir. Gradyan, optimizer, RNG, eğitim modu ve mimari saklanmaz. Yüklemeden önce aynı mimariyi kurun.",
        "state_dict clones parameter data and buffers. Keys are hierarchical, such as 0.weight or norm.running_mean. Gradients, optimizer, RNG, training mode, and architecture are not saved. Construct the same architecture first.",
        {
          code: 'save_state_dict(model.state_dict(), "model.nxm");\nauto state=load_state_dict_file("model.nxm");\nmodel.load_state_dict(state); // strict=true\nmodel.eval();',
        },
      ),
      s(
        "strict",
        "Strict yükleme",
        "Strict loading",
        "load_state_dict(state,strict=true) eksik/fazla anahtarları reddeder. strict=false onları tolere eder; eşleşen anahtarların şekilleri yine uymalıdır. Değerler mevcut parametrelere kopyalanır.",
        "load_state_dict(state,strict=true) rejects missing/extra keys. strict=false tolerates them; shapes of matching keys still must agree. Values copy into existing parameters.",
      ),
      s(
        "format",
        "NXM1 düzeni",
        "NXM1 layout",
        "NXM1 sihri + uint32 kayıt sayısı. Her kayıt: uint32 isim uzunluğu, isim baytları, uint32 rank, uint64 boyutlar, ham float32. Host byte sırası kullanılır; endian dönüşümü yoktur.",
        "NXM1 magic + uint32 entry count. Each entry: uint32 name length, name bytes, uint32 rank, uint64 dimensions, raw float32. Uses host byte order without endian conversion.",
      ),
      s(
        "limits",
        "Doğrulama sınırları",
        "Validation limits",
        "Okuyucu sihir, rank≤8 ve kesilmiş dosya kontrolü yapar; toplam boyut, isim uzunluğu veya kayıt sayısı kotası yoktur. Kontrollü checkpoint alışverişi içindir. Güvenilmeyen yüklemeler öncesinde sınırlandırılmış doğrulama ekleyin.",
        "Reader checks magic, rank≤8, and truncation, without quotas on total size, name length, or entry count. Intended for controlled checkpoint exchange. Add bounded validation before accepting untrusted uploads.",
      ),
    ],
  },
  {
    slug: "simd",
    group: 3,
    title: tx("CPU ve SIMD", "CPU & SIMD"),
    description: tx(
      "Çalışma zamanı dispatch ve vektör çekirdekleri.",
      "Runtime dispatch and vector kernels.",
    ),
    source: "src/simd/simd_dispatch.cpp",
    sections: [
      s(
        "dispatch",
        "Yönerge seçimi",
        "Instruction selection",
        "Donanım özellikleri ve OS XCR0 kayıt desteği denetlenir. AVX2 yolunda FMA kullanan çekirdekler vardır. AVX-512F derlenmiş ve destekleniyorsa ReLU ve dot AVX-512 kullanır; linear_forward bloklu AVX2 motoruna yönlenir; tüm işlemler AVX-512 değildir.",
        "Hardware features and OS XCR0 register support are checked. The AVX2 path includes kernels using FMA. When compiled and supported, AVX-512F uses AVX-512 for ReLU/dot; linear_forward delegates to the blocked AVX2 engine; not every operation uses AVX-512.",
      ),
      s(
        "fallback",
        "Skalar referans",
        "Scalar reference",
        "kernels::set_force_scalar(true) referans yolu, false otomatik seçimi açar. Derleyici skalar döngüyü yine vektörleştirebilir. Tablo seçimi atomiktir; tekrarlanabilir ölçümler için hesap sırasında değiştirmeyin.",
        "kernels::set_force_scalar(true) selects the reference path; false restores automatic selection. Compilers may still vectorize scalar loops. Table selection is atomic; keep it fixed during a measurement for reproducibility.",
        {
          code: "#include <nexus_model/core/kernels.hpp>\nkernels::set_force_scalar(true);\n// Reference measurement\nkernels::set_force_scalar(false);",
        },
      ),
      s(
        "portability",
        "Derleme ve taşınabilirlik",
        "Build and portability",
        "MSVC /arch:AVX2 ve /arch:AVX512; GCC/Clang -mavx2/-mfma ve -mavx512f kullanır. Release genelinde -march=native kullanılmaz; ISA bayrakları yalnız ilgili kaynak dosyalarına uygulanır.",
        "MSVC uses /arch:AVX2 and /arch:AVX512; GCC/Clang use -mavx2/-mfma and -mavx512f. Release does not apply -march=native globally; ISA flags are confined to the relevant source files.",
      ),
      s(
        "measured",
        "Ölçülen kazanç",
        "Measured improvement",
        "Linear [32,256]→[32,256] forward+backward: skalar ve SIMD süreleri güncel CPU tablosunda gösterilir. Bu belirli ölçümdür; her model için genel hız iddiası değildir.",
        "Linear [32,256]→[32,256] forward+backward: current scalar and SIMD times are displayed in the CPU table. This is a specific measurement, not a universal speedup.",
      ),
    ],
  },
  {
    slug: "cuda",
    group: 3,
    title: tx("CUDA ve MatrixFlash", "CUDA & MatrixFlash"),
    description: tx(
      "Cihaz işaretçileri, fused çekirdekler ve GEMM köprüsü.",
      "Device pointers, fused kernels, and GEMM bridge.",
    ),
    source: "include/nexus_model/cuda/fused_softmax.cuh",
    sections: [
      s(
        "boundary",
        "Host / cihaz sınırı",
        "Host / device boundary",
        "NEXUS_MODEL_WITH_CUDA cihaz fonksiyonlarını derler. Host Tensor otomatik taşınmaz. cudaMalloc, transfer, eşzamanlama, hata kontrolü ve cudaFree çağırana aittir. Bu API cihaz üstü backward sunmaz.",
        "NEXUS_MODEL_WITH_CUDA compiles device functions. Host Tensor is not transferred automatically. The caller owns cudaMalloc, transfers, synchronization, error checks, and cudaFree. This API does not provide device-side backward.",
      ),
      s(
        "kernels",
        "Fused arayüzler",
        "Fused interfaces",
        "Softmax max/exp/toplam/bölmeyi; LayerNorm ortalama/varyans/normalize/scale/shift’i birleştirir. Tüm işaretçiler cihaz belleği, rows/cols pozitif olmalıdır.",
        "Softmax fuses max/exp/sum/division; LayerNorm fuses mean/variance/normalize/scale/shift. All pointers must refer to device memory, with positive rows/cols.",
        {
          code: "namespace nexus_model::cuda {\n void relu(const float* x,float* y,int n);\n void gelu_tanh(const float* x,float* y,int n);\n void fused_softmax(const float* x,float* y,int rows,int cols);\n void fused_layernorm(const float* x,const float* gamma,\n   const float* beta,float* y,int rows,int cols,float eps);\n}",
        },
      ),
      s(
        "bridge",
        "MatrixFlash köprüsü",
        "MatrixFlash bridge",
        "WITH_MATRIXFLASH açıkken bridge/matrixflash.hpp, matrix_pro::multiply için gemm(left,right) sunar. CPU GEMM NexusModel’dedir. Bu başlık Linear forward’u otomatik değiştirmez; tüketen hedef köprüyü açıkça kullanır.",
        "With WITH_MATRIXFLASH, bridge/matrixflash.hpp provides gemm(left,right) around matrix_pro::multiply. CPU GEMM belongs to NexusModel. This header does not automatically change Linear forward; consumers call it explicitly.",
      ),
      s(
        "timing",
        "Ölçümün kapsamı",
        "Measurement scope",
        "GPU ölçümleri cihazda hazır verinin cudaEvent süresidir. Transfer, bellek ayırma, backward veya tam eğitim pipeline’ı değildir. GEMM benchmark doğrudan cuBLAS çağırır; NexusModel’in kendi GPU GEMM’i değildir.",
        "GPU numbers are cudaEvent timings with device-resident data. They exclude transfers, allocation, backward, and the full training pipeline. GEMM benchmark calls cuBLAS directly; it is not a NexusModel GPU GEMM implementation.",
      ),
    ],
  },
  {
    slug: "benchmark-methodology",
    group: 3,
    title: tx("Benchmark metodolojisi", "Benchmark methodology"),
    description: tx(
      "Kaynak, kapsam ve tekrar üretilebilir ölçümler.",
      "Sources, scope, and reproducible measurements.",
    ),
    source: "benchmarks/RESULTS.md",
    sections: [
      s(
        "source",
        "Yayımlanmış rapor",
        "Published report",
        "Sayılar benchmarks/RESULTS.md anlık görüntüsüdür; tarayıcı canlı C++ benchmark çalıştırmaz. Güncel CPU modeli ve derleyici sürümü JSON metadata içindedir; GPU verileri eski sürüm arşividir. MSVC 19.44, /O2, Release, AVX2/AVX-512. GPU RTX 3070 Laptop, CUDA 13.4, sm_86.",
        "Numbers are a snapshot of benchmarks/RESULTS.md; the browser does not run live C++ benchmarks. Current CPU and compiler information is in JSON metadata; GPU figures are historical. MSVC 19.44, /O2, Release, AVX2/AVX-512. GPU RTX 3070 Laptop, CUDA 13.4, sm_86.",
      ),
      s(
        "cpu",
        "CPU yöntemi",
        "CPU method",
        "Katman testleri 20 ısınma ve 30×5 çağrının medyanını raporlar. ReLU yalnız forward, diğerleri forward+backward. Kütüphane karşılaştırması ve eski/yeni çekirdek ölçümü ise üç turda 90 örnek içerir; ham örnekler indirilebilir.",
        "Module tests report medians from 20 warmups and 30×5 calls. ReLU is forward-only; the others include forward+backward. Cross-library and revision kernel comparisons use three rounds and 90 samples; raw samples are downloadable.",
      ),
      s(
        "gpu",
        "GPU ve hassasiyet",
        "GPU and precision",
        "cudaEvent, cihazda hazır veri; H2D/D2H hariç. Kaynak rapor Ampere cuBLAS SGEMM için varsayılan TF32 tensor çekirdekleri ve 4096 boyutunda 1.8e−5 en büyük mutlak sapma bildirir. Bunları farklı sürüm veya donanıma genellemeyin.",
        "cudaEvent, resident device data; H2D/D2H excluded. The source reports default TF32 tensor cores for Ampere cuBLAS SGEMM and maximum absolute deviation 1.8e−5 at size 4096. Do not generalize to other versions or hardware.",
      ),
      s(
        "reproduce",
        "Tekrar ölçün",
        "Reproduce",
        "Release derleyin, donanım ve güç profilini kaydedin. Daha çok ısınma/tekrar ve medyan/p95 raporu karşılaştırmayı güçlendirir. CPU forward+backward çiftleri ile GPU forward çekirdeklerini doğrudan karşılaştırmayın.",
        "Build Release and record hardware and power profile. More warmups/repeats and median/p95 reporting strengthen comparisons. Do not directly compare CPU forward+backward pairs with GPU forward kernels.",
        {
          code: ".\\build\\Release\\nexus_model_bench.exe\n.\\build-cuda\\Release\\nexus_model_bench_gpu.exe",
        },
      ),
    ],
  },
  {
    slug: "testing",
    group: 3,
    title: tx("Doğrulama ve sınırlar", "Validation & limitations"),
    description: tx(
      "Gradyan kontrolü, bellek testi ve kullanım sözleşmeleri.",
      "Gradient checks, memory tests, and usage contracts.",
    ),
    source: "tests/test_model.cpp",
    sections: [
      s(
        "checks",
        "Doğruluk kontrolleri",
        "Correctness checks",
        "Testler Linear/Conv referansları, merkezi sonlu fark, aktivasyonlar, normalizasyon, recurrent/attention, state_dict, SIMD ve sıcak yol ayırmalarını kapsar. Parçalı fonksiyonların kink noktaları atlanır.",
        "Tests cover Linear/Conv references, central finite differences, activations, normalization, recurrent/attention, state_dict, SIMD, and hot-path allocations. Kink points of piecewise functions are skipped.",
        {
          formula:
            "g_i\\approx\\frac{L(\\theta_i+\\varepsilon)-L(\\theta_i-\\varepsilon)}{2\\varepsilon}",
        },
      ),
      s(
        "tolerance",
        "Tolerans seçimi",
        "Choosing tolerances",
        "Çok küçük epsilon float32 yuvarlama, büyük epsilon yaklaşım hatası oluşturur. Mutlak ve bağıl hata kullanın. Dropout’u kapatın veya maskeyi kontrol edin.",
        "Too-small epsilon suffers float32 rounding; too-large epsilon increases approximation error. Use absolute and relative errors. Disable dropout or control its mask.",
        { formula: "e_{rel}=\\frac{|g_a-g_n|}{\\max(1,|g_a|,|g_n|)}" },
      ),
      s(
        "memory",
        "Ayırma iddiası",
        "Allocation claim",
        "Sabit şekilli, ısıtılmış yolda çıktı başlıkları bırakıldığında allocation_count artmaması denetlenir. Bu tüm uygulamanın heap kullanmadığı anlamına gelmez; sayaç sadece AlignedStorage içindir.",
        "Tests check allocation_count stays unchanged on warmed fixed-shape paths after output headers are released. This does not imply no heap use across the application; the counter covers AlignedStorage only.",
      ),
      s(
        "contracts",
        "Çağıranın sorumlulukları",
        "Caller responsibilities",
        "Pozitif boyutlar, doğru backward şekilleri, geçerli indeksler, head divisibility ve bellek ömrü gerekir. Her kernel kapsamlı kontrol yapmaz. Tek modülde eşzamanlı forward yapmayın. Öğrenilebilir PositionalEncoding için max_len aşmayın.",
        "Ensure positive dimensions, correct backward shapes, valid indices, head divisibility, and memory lifetime. Not every kernel performs full validation. Do not run concurrent forward on one module. Keep learnable PositionalEncoding within max_len.",
      ),
      s(
        "troubleshoot",
        "Sorun giderme",
        "Troubleshooting",
        "Büyük gradyan: zero_grad ve batch ortalaması. Şekil hatası: Linear son ekseni, Conv NCHW. Yeni ayırma: eski çıktılar hâlâ canlı mı? CUDA açık ama CPU kullanılıyor: host Module bilinçli olarak CPU yolunu kullanır.",
        "Large gradients: check zero_grad and batch averaging. Shape errors: check the last Linear axis and Conv NCHW. New allocations: are old outputs still alive? CPU use with CUDA enabled: host Module intentionally uses the CPU path.",
      ),
    ],
  },
];
