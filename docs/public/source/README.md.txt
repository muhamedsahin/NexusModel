# NexusModel

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

NexusModel, Nexus ekosisteminin C++20 katman kütüphanesidir. Tensörleri katmanlardan geçirir, `backward` ile girdi gradyanını üretir ve öğrenilebilir parametre gradyanlarını biriktirir. Veri yükleme NexusData, kayıp NexusLoss, parametre güncellemesi NexusOptim tarafındadır.

Lisans **Apache-2.0**. Koşullar [LICENSE](LICENSE) dosyasındadır.

## Bu makine, 21 Ekim 2026 akşamı

Ölçüm bu bilgisayardadır: AMD Ryzen 7 5800H. Dünya sıralaması yoktur.

Rakip protokolünde GEMM çekirdeği `avx2-4x16` olarak kaldı. N=2048 sütununda Nexus 330, 362 ve 369; TensorFlow 389. Birim, o protokolün GFLOPS sütunudur.

CPU Conv2d aynı protokolde PyTorch’un 31.6 GFLOPS değerinin gerisinde kaldı. Nexus o akşam yaklaşık 4.9 ile 6.8 GFLOPS arasındaydı.

Depthwise AVX, o akşamki sütunda PyTorch 4.93 ve TensorFlow 0.688 değerlerinin önündeydi.

Ryzen 7 5800H üzerinde AVX-512 seçilmez. `NEXUS_MODEL_ENABLE_AVX512` çekirdekleri derleyebilir; bu işlemci o yolu almaz. `avx512::linear_forward` AVX2 `linear_forward` çağırır.

Bu depodaki `Linear` sıcak yolu `src/simd/avx2/gemm.hpp` içindeki 6×16 AVX2 karosudur. Yukarıdaki N=2048 GFLOPS değerleri o karoya yazılmaz; akşam protokolünün çekirdek adı `avx2-4x16`dir.

29 Eylül 2026 tarihli tek iş parçacıklı milisaniye tabloları `benchmarks/RESULTS.md` içindedir. Ham günlükler `benchmarks/results/` altındadır. O tablolar bu akşamın GFLOPS sütunuyla karıştırılmaz.

Kullanım ayrıntısı: [docs/kullanim.md](docs/kullanim.md).

## Kurulum

CMake 3.20+ ve bir C++20 derleyicisi gerekir (MSVC 19.30+, GCC 11+ veya Clang 13+). Komutlar bu klasörde çalışır.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

| Seçenek | Varsayılan |
|---|---|
| `NEXUS_MODEL_BUILD_TESTS` | `ON` |
| `NEXUS_MODEL_BUILD_BENCHMARKS` | `ON` |
| `NEXUS_MODEL_BUILD_EXAMPLES` | `OFF` |
| `NEXUS_MODEL_ENABLE_AVX512` | `ON` (bu Ryzen’de seçilmez) |
| `NEXUS_MODEL_WITH_CUDA` | `OFF` |
| `NEXUS_MODEL_WITH_MATRIXFLASH` | `OFF` |
| `NEXUS_MODEL_WARNINGS_AS_ERRORS` | `ON` |

Bu sürüm `find_package` yapılandırması dışa aktarmaz. Uygulama `add_subdirectory` ile bağlanır:

```cmake
add_subdirectory(path/to/NexusModel)
target_link_libraries(my_app PRIVATE nexus_model::nexus_model)
```

Başlık girişi `nexus_model/nexus_model.hpp`.

## Kısa ileri geçiş

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

`parameters()`, `train()` ve `eval()` için [docs/kullanim.md](docs/kullanim.md).

## Katman kataloğu

Sınıf adları `include/nexus_model` başlıklarındaki adlardır. Hepsi `nexus_model` ad uzayındadır ve `Module` üzerinden `forward` / `backward` uygular.

| Sınıf | Başlık |
|---|---|
| `Linear` | `layers/linear.hpp` |
| `Conv1D`, `Conv2D`, `Conv3D`, `DepthwiseConv2D` | `layers/conv.hpp` |
| `BatchNorm`, `BatchNorm1D`, `BatchNorm2D`, `LayerNorm`, `GroupNorm`, `InstanceNorm` | `layers/normalization.hpp` |
| `RNN`, `LSTM`, `GRU` | `layers/recurrent.hpp` |
| `MultiHeadAttention`, `PositionalEncoding` | `layers/attention.hpp` |
| `TransformerEncoderLayer`, `TransformerDecoderLayer` | `layers/attention.hpp` |
| `MaxPool2D`, `AvgPool2D`, `GlobalAvgPool2D` | `layers/pooling.hpp` |
| `Embedding` | `layers/embedding.hpp` |
| `Flatten`, `Reshape` | `layers/flatten_reshape.hpp` |
| `Dropout` | `layers/dropout.hpp` |
| `ResidualBlock` | `layers/residual.hpp` |
| `Sequential`, `ModuleList`, `ModuleDict` | `containers/` |
| `ReLU`, `LeakyReLU`, `ELU`, `GELU`, `Sigmoid`, `Tanh`, `SiLU`, `Mish`, `Softmax`, `LogSoftmax`, `PReLU` | `activations/activations.hpp` |

`ConvNd` kurucusu korumalıdır; çağrı `Conv1D`, `Conv2D`, `Conv3D` veya `DepthwiseConv2D` üzerinden yapılır. `DepthwiseConv2D`, `groups = channels` olan bir `Conv2D`dir. `BatchNorm1D` ve `BatchNorm2D`, `BatchNorm` kurucusunu kullanır. `InstanceNorm`, kanal sayısı kadar gruplu `GroupNorm`dur.

Evrişim düzeni NCHW (Conv1D için `[N,C,L]`). `TransformerDecoderLayer::forward(target)` bellek tensörü verilmeden çağrılırsa hata verir; `forward(target, memory)` kullanılır.
