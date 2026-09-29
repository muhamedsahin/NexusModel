import { Article, s, tx } from "./types";
export const comparisonGuide: Article = {
  slug: "library-comparison",
  group: 3,
  title: tx("Kütüphane karşılaştırması", "Library comparison"),
  description: tx(
    "NexusModel, NumPy ve PyTorch ile aynı CPU işlemini ölçmek.",
    "Measuring the same CPU operation with NexusModel, NumPy, and PyTorch.",
  ),
  source: "docs/benchmarks/compare_libraries.py",
  sections: [
    s(
      "scope",
      "Karşılaştırma tam olarak ne?",
      "What exactly is compared?",
      "NexusModel bir C++ sinir ağı katman kütüphanesi, PyTorch bir tensör/eğitim çatısı, NumPy sayısal dizi kütüphanesidir. Burada ortak payda olan CPU float32 GEMM ve ReLU ileri işlemleri ölçülür. Model kalitesi, eğitim kapasitesi veya tüm API’lerin genel hız sıralaması ölçülmez.",
      "NexusModel is a C++ neural-network layer library, PyTorch a tensor/training framework, and NumPy a numerical array library. We measure their shared CPU float32 GEMM and ReLU forward operations. This does not measure model quality, training capability, or an overall ranking of all APIs.",
    ),
    s(
      "nexus",
      "NexusModel çağrıları",
      "NexusModel calls",
      "Aynı X=[M,K], W=[N,K], Y=[M,N] düzeninde bias=nullptr ile ham linear_forward çekirdeği çalışır. Tam Linear modülündeki eğitim önbelleği bu ölçümde yoktur. ReLU için girdi/çıktı tamponları ayrıdır ve önceden ayrılmıştır.",
      "Use X=[M,K], W=[N,K], Y=[M,N] with bias=nullptr and the raw linear_forward kernel. This excludes the full Linear module training cache. ReLU uses separate preallocated input/output buffers.",
      {
        code: "kernels::linear_forward(x.data(), w.data(), nullptr,\n  y.data(), rows, in_features, out_features);\nkernels::relu(x.data(), y.data(), x.numel());",
      },
    ),
    s(
      "numpy",
      "NumPy çağrıları",
      "NumPy calls",
      "Diziler float32 ve satır-major oluşturulur. W.T görünümü ölçümden önce hazırlanır; matmul out ile çıktı tamponunu yeniden kullanır. maximum ReLU işlemini uygular. Yüklü BLAS backend bilgisi threadpoolctl ile JSON’a kaydedilir.",
      "Arrays are float32 and row-major. W.T is prepared before timing; matmul reuses the output buffer via out. maximum implements ReLU. Loaded BLAS backend information is recorded through threadpoolctl in the JSON.",
      {
        code: "import numpy as np\ny = np.empty((rows, out_features), dtype=np.float32)\nwt = w.T\nnp.matmul(x, wt, out=y)\nnp.maximum(relu_input, np.float32(0), out=relu_output)",
      },
    ),
    s(
      "pytorch",
      "PyTorch çağrıları",
      "PyTorch calls",
      "torch.set_num_threads(1) ve set_num_interop_threads(1) ölçüm öncesi çağrılır. inference_mode içinde mm/clamp out tamponlarını kullanır. CPU’da float32 çalışır; CUDA ve autograd bu ölçümün dışında kalır.",
      "Call torch.set_num_threads(1) and set_num_interop_threads(1) before timing. mm/clamp reuse out buffers within inference_mode. Operations use float32 on CPU; CUDA and autograd are outside this measurement.",
      {
        code: "import torch\ntorch.set_num_threads(1)\ntorch.set_num_interop_threads(1)\nwith torch.inference_mode():\n    torch.mm(x, wt, out=y)\n    torch.clamp(relu_input, min=0, out=relu_output)",
      },
    ),
    s(
      "method",
      "Ölçüm ve doğruluk",
      "Timing and correctness",
      "Üç turda kütüphane sırası değiştirilir. Her turda 20 ısınma çağrısından sonra 30 örnek alınır; her örnek 5 çağrının ortalama süresidir. Medyan ve p95 birleştirilmiş 90 değerden hesaplanır. Her çıktı elemanı float64 referansına karşı rtol=1e−4, atol=1e−5 ile doğrulanır. ReLU sonuçları birebir eşittir. Doğrulama başarısızsa script sonuç yayımlamaz.",
      "Rotate framework order across three rounds. After 20 warmup calls per round, collect 30 samples, each averaging 5 calls. Median and p95 are computed from all 90 values. Every output element is validated against a float64 reference with rtol=1e−4, atol=1e−5. ReLU outputs match exactly. The script refuses to publish failed validation.",
    ),
    s(
      "limits",
      "Nasıl yorumlanmalı?",
      "How to interpret results",
      "NexusModel C++ çağrı maliyeti, diğerlerinde Python API çağrı maliyeti dahildir. Tek makine, üç turda dönüşümlü framework sırası; sıcaklık ve OS zamanlaması kontrol edilmemiştir. Küçük boyutlarda dispatch etkisi büyür. Bu nedenle tek bir “en hızlı kütüphane” iddiası yerine işlem/şekil bazında sonucu okuyun.",
      "NexusModel includes C++ call overhead; the others include Python API dispatch. One machine, three rotated framework rounds, uncontrolled temperature and OS scheduling. Dispatch matters more for small workloads. Interpret by operation and shape rather than declaring one library universally fastest.",
    ),
    s(
      "reproduce",
      "Kendi ölçümünüz",
      "Your own measurement",
      "Komutlar docs içinde çalıştırılır. NumPy, CPU PyTorch ve threadpoolctl bulunan izole bir Python ortamı kullanın. Script paket sürümleri, iş parçacığı sayıları, BLAS backend, kaynak ve çalıştırılabilir dosya SHA-256 değerlerini kaydeder.",
      "Run from docs using an isolated Python environment containing NumPy, CPU PyTorch, and threadpoolctl. The script records package versions, thread counts, BLAS backends, and source/executable SHA-256 hashes.",
      {
        code: "cmake -S examples -B .example-build\ncmake --build .example-build --config Release\npython benchmarks/compare_libraries.py\nnpm run build",
      },
    ),
  ],
};
