<div align="center">

# 🧱 NexusModel

### Nexus ekosisteminin C++20 katman kütüphanesi

*Tensörü katmanlardan geçirir, `backward` ile girdi gradyanını üretir, parametre gradyanlarını biriktirir.*

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?style=for-the-badge&logo=cmake&logoColor=white)](CMakeLists.txt)
[![AVX2](https://img.shields.io/badge/SIMD-AVX2_%7C_AVX--512-FF6F00?style=for-the-badge)](src/simd)
[![License](https://img.shields.io/badge/Lisans-Apache--2.0-D22128?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)

[Kurulum](#-kurulum) • [Kısa örnek](#-kısa-ileri-geçiş) • [Katman kataloğu](#-katman-kataloğu) • [Ölçümler](#-bu-makine-21-ekim-2026-akşamı) • [Nexus](https://github.com/muhamedsahin/Nexus)

</div>

> Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

---

## 🎯 Bu kütüphane ne yapar?

| ✅ Yapar | 🚫 Yapmaz (ekosistemdeki yeri) |
| --- | --- |
| Katman `forward` / `backward` | Veri yükleme → [NexusData](https://github.com/muhamedsahin/NexusData) |
| Parametre gradyanı biriktirme | Kayıp → [NexusLoss](https://github.com/muhamedsahin/NexusLoss) |
| AVX2 (6×16 karo) sıcak yollar | Parametre güncellemesi → [NexusOptim](https://github.com/muhamedsahin/NexusOptim) |
| `train()` / `eval()` modları | Büyük GEMM → [MatrixFlash-Pro](https://github.com/muhamedsahin/MatrixFlash_Pro) |

Nexus'ta bu katmanlar `nexus::nn::` adlarıyla ve `nexus::autograd::module_function` köprüsüyle autograd grafiğine de bağlanır.

---

## 🛠 Kurulum

CMake 3.20+ ve bir C++20 derleyicisi (MSVC 19.30+, GCC 11+ veya Clang 13+).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

| Seçenek | Varsayılan |
| --- | --- |
| `NEXUS_MODEL_BUILD_TESTS` | `ON` |
| `NEXUS_MODEL_BUILD_BENCHMARKS` | `ON` |
| `NEXUS_MODEL_BUILD_EXAMPLES` | `OFF` |
| `NEXUS_MODEL_ENABLE_AVX512` | `ON` (Ryzen 7 5800H'de seçilmez) |
| `NEXUS_MODEL_WITH_CUDA` | `OFF` |
| `NEXUS_MODEL_WITH_MATRIXFLASH` | `OFF` |
| `NEXUS_MODEL_WARNINGS_AS_ERRORS` | `ON` |

Bu sürüm `find_package` yapılandırması dışa aktarmaz; `add_subdirectory` ile bağlanır:

```cmake
add_subdirectory(path/to/NexusModel)
target_link_libraries(my_app PRIVATE nexus_model::nexus_model)
```

Başlık girişi: `nexus_model/nexus_model.hpp`.

---

## 🚀 Kısa ileri geçiş

```cpp
#include <nexus_model/nexus_model.hpp>

int main() {
    nexus_model::Linear layer(4, 2);
    nexus_model::Tensor x = nexus_model::Tensor::from_values(
        {1, 4}, {1.0f, 0.5f, -0.2f, 0.8f});
    nexus_model::Tensor y = layer.forward(x);
    return y.size(1) == 2 ? 0 : 1;
}
```

`parameters()`, `train()` ve `eval()` için: [docs/kullanim.md](docs/kullanim.md).

---

## 📚 Katman kataloğu

Hepsi `nexus_model` ad uzayındadır ve `Module` üzerinden `forward` / `backward` uygular.

| Aile | Sınıflar | Başlık |
| --- | --- | --- |
| 🔗 Doğrusal | `Linear` | `layers/linear.hpp` |
| 🌀 Evrişim | `Conv1D`, `Conv2D`, `Conv3D`, `DepthwiseConv2D` | `layers/conv.hpp` |
| 📐 Normalizasyon | `BatchNorm`, `BatchNorm1D`, `BatchNorm2D`, `LayerNorm`, `GroupNorm`, `InstanceNorm` | `layers/normalization.hpp` |
| 🔁 Tekrarlayan | `RNN`, `LSTM`, `GRU` | `layers/recurrent.hpp` |
| 👁 Dikkat | `MultiHeadAttention`, `PositionalEncoding` | `layers/attention.hpp` |
| 🤖 Transformer | `TransformerEncoderLayer`, `TransformerDecoderLayer` | `layers/attention.hpp` |
| 🏊 Havuzlama | `MaxPool2D`, `AvgPool2D`, `GlobalAvgPool2D` | `layers/pooling.hpp` |
| 🔤 Gömme | `Embedding` | `layers/embedding.hpp` |
| 📦 Şekil | `Flatten`, `Reshape` | `layers/flatten_reshape.hpp` |
| 🎲 Düzenleme | `Dropout` | `layers/dropout.hpp` |
| ➕ Artık | `ResidualBlock` | `layers/residual.hpp` |
| 🗂 Kapsayıcı | `Sequential`, `ModuleList`, `ModuleDict` | `containers/` |
| ⚡ Aktivasyon | `ReLU`, `LeakyReLU`, `ELU`, `GELU`, `Sigmoid`, `Tanh`, `SiLU`, `Mish`, `Softmax`, `LogSoftmax`, `PReLU` | `activations/activations.hpp` |

<details>
<summary><b>Notlar</b></summary>

- `ConvNd` kurucusu korumalıdır; çağrı `Conv1D`, `Conv2D`, `Conv3D` veya `DepthwiseConv2D` üzerinden yapılır.
- `DepthwiseConv2D`, `groups = channels` olan bir `Conv2D`'dir. `InstanceNorm`, kanal sayısı kadar gruplu `GroupNorm`'dur.
- Evrişim düzeni NCHW (Conv1D için `[N,C,L]`).
- `TransformerDecoderLayer::forward(target)` bellek tensörü verilmeden çağrılırsa hata verir; `forward(target, memory)` kullanılır.

</details>

---

## 📊 Bu makine, 21 Ekim 2026 akşamı

AMD Ryzen 7 5800H. **Dünya sıralaması değildir.**

| İş | Nexus | Karşılaştırma |
| --- | --- | --- |
| GEMM N=2048 (`avx2-4x16`) | 330 · 362 · 369 GFLOPS | TensorFlow **389** |
| Conv2d (CPU) | ~4.9 – 6.8 GFLOPS | PyTorch **31.6** — Nexus geride |
| Depthwise (AVX) | **önde** | PyTorch 4.93, TensorFlow 0.688 |

- Bu işlemcide AVX-512 seçilmez; `avx512::linear_forward` AVX2 `linear_forward`'u çağırır.
- `Linear` sıcak yolu `src/simd/avx2/gemm.hpp` içindeki 6×16 AVX2 karosudur; yukarıdaki N=2048 değerleri o karoya yazılmaz.
- 29 Eylül 2026 tek iş parçacıklı milisaniye tabloları `benchmarks/RESULTS.md`, ham günlükler `benchmarks/results/` altındadır.

---

## 📄 Lisans

Apache-2.0 — koşullar [LICENSE](LICENSE) dosyasındadır.

---

<div align="center">

**🎓 Bu proje BTÜ (Bursa Teknik Üniversitesi) Bilgisayar Mühendisliği 1. sınıf öğrencisi tarafından yapay zeka destekli olarak yapıldı.**

<sub>Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.</sub>

<sub>⚡ [Nexus](https://github.com/muhamedsahin/Nexus) ekosisteminin bir parçasıdır.</sub>

</div>
