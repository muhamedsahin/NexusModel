# NexusModel

## Ölçülmüş CPU optimizasyonu

Linear ve convolution artık bloklu AVX2/FMA matris motorunu kullanır. İleri
geçiş 6×16 register blokları ve 8 KiB ağırlık panelleri kullanır; geri geçiş
geçici transpoz oluşturmaz. LayerNorm, kararlı iki geçişli float64 istatistik
hesabını SIMD ile yürütür. Küçük batch için doğrudan dot yolu korunur.

Güncel sayılar ve sınırlamalar: [benchmark raporu](benchmarks/RESULTS.md).
NumPy ve PyTorch ile 9 iş yükünde 27 sonuç, her sonuçta 90 zaman örneği ve
tam çıktı doğrulaması yayımlanır. Eski/yeni sürümler aynı C++ ölçüm koduyla
dönüşümlü çalıştırılır. Tek makine ölçümü evrensel "dünyanın en hızlısı"
iddiasını desteklemez; rakiplerin kazandığı sonuçlar da raporda yer alır.

Dosya düzeni: [CPU çekirdekleri](src/core/details.txt),
[AVX2 hesap motoru](src/simd/avx2/details.txt),
[inceleme ve kararlar](benchmarks/OPTIMIZATION.md).

```text
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
cmake -S docs/examples -B docs/.example-build
cmake --build docs/.example-build --config Release
python docs/benchmarks/compare_libraries.py
```

Nexus yığınında katman, ileri geçiş ve geri yayılım. Kayıp NexusLoss'ta, parametre güncellemesi NexusOptim'de, veri NexusData'da, ham matris cebri MatrixFlash Pro'dadır.

## Autograd

MatrixFlash `Variable::backward()` kök gradyanı birlerle doldurur ve dış `dL/dy` kabul etmez. Kardeş kütüphaneler aynı teybi paylaşmaz. NexusModel bu yüzden **Yol B** kullanır:

- Linear, Conv, Pool, Norm, Dropout, Embedding, RNN/LSTM/GRU ve aktivasyonlar analitik backward yazar.
- Attention'ın softmax ve matmul zinciri katman-lokal `MicroTape` ile türetilir.

Ayrıntı: `include/nexus_model/core/details.txt`.

## Hesap

Varsayılan yol host float32 ve AVX2'dir. AVX-512F derlenir ve işlemci destekliyorsa dot/ReLU oraya gider; Linear bloklu AVX2 motorunu kullanır. MatrixFlash `multiply` yalnızca CUDA olduğundan CPU GEMM bu kütüphanededir. GPU çarpımı `NEXUS_MODEL_WITH_MATRIXFLASH` ile MatrixFlash'e bırakılır; NexusModel cuBLAS yazmaz.

Fused softmax, LayerNorm ve aktivasyon CUDA çekirdekleri `NEXUS_MODEL_WITH_CUDA=ON` ile derlenir. Host `Tensor` onları otomatik çağırmaz; cihaz işaretçisi `nexus_model::cuda` altından verilir.

## Derleme

```text
cmake -S NexusModel -B NexusModel/build
cmake --build NexusModel/build --config Release
NexusModel/build/Release/nexus_model_tests.exe
```

`NEXUS_MODEL_BUILD_EXAMPLES=ON` NexusData, NexusLoss ve NexusOptim'i yalnızca `train_end_to_end` hedefine bağlar. Çekirdek bu üçüne bağlanmaz.

## Doğruluk

`tests/test_model.cpp` elle kurulmuş Linear ve Conv referansları ile merkezi sonlu fark gradyan kontrolü içerir. Parçalı aktivasyonlarda tam sıfır noktası (ReLU ailesi) atlanır; orada analitik alttürev 0'dır. Sıcak yol, dönen tensör başlığı bırakıldıktan sonra ikinci forward+backward'ta `allocation_count` artmaz.

## Sınırlar

- Bir modül nesnesi eşzamanlı forward için güvenli değildir. Replika gradyanı `Parameter::accumulate_grad_locked` ile adım dışında birleşir.
- Hesap float32'dir.
- Transformer decoder, bellek gradyanını dönüş değerine eklemez; `grad_key_input` ve `grad_value_input` içindedir. Projeksiyon parametrelerinin gradyanı birikir.
