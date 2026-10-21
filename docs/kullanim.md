# NexusModel kullanım kılavuzu

Bu yazılım ekosistemi Bursa Teknik Üniversitesi Bilgisayar Mühendisliği 1. sınıf öğrencisi Muhammed Fatih Şahin tarafından yapay zeka destekli olarak geliştirilmiştir.

Bu dosya kullanım kılavuzunun kaynak metnidir. Pandoc bu makinede yüklü değildir. `docs/NexusModel-Kullanim-Kilavuzu.pdf` ve `docs/public` kopyası bu metinden yeniden üretilmedi.

Lisans Apache-2.0. Metin [LICENSE](../LICENSE) dosyasındadır.

NexusModel, C++20 katman kütüphanesidir. Tek giriş başlığı:

```cpp
#include <nexus_model/nexus_model.hpp>
using namespace nexus_model;
```

Derleme, depo kökündeki `README.md` ile aynıdır: CMake 3.20+, C++20, `nexus_model::nexus_model`.

## İleri geçiş

Yeni bir modül eğitim modunda başlar (`is_training()` doğrudur). `forward` çıktı tensörünü katmanın kendi tamponunda üretir. Aynı nesnede bir sonraki `forward`, bir önceki geri geçiş önbelleğinin üzerine yazar.

```cpp
Linear layer(4, 2, /*use_bias=*/true);
Tensor x = Tensor::from_values({2, 4}, {
    1.0f, 0.5f, -0.2f, 0.8f,
    0.0f, 1.2f,  0.3f, -0.5f
});
Tensor y = layer.forward(x);  // [2, 2]
```

Geri geçiş, dışarıdan verilen `dL/dy` ile girdi gradyanını döndürür. Parametre gradyanı `+=` ile birikir. Yeni adımdan önce `zero_grad()` çağrılır.

```cpp
layer.zero_grad();
Tensor dy = Tensor::full({2, 2}, 1.0f);
Tensor dx = layer.backward(dy);
```

## parameters()

`parameters()` modülün ve kayıtlı çocukların `Parameter*` listesini döndürür. `named_parameters()` aynı yapraklara `"weight"`, `"0.bias"`, `"l0.weight_ih"` gibi adlarla ulaşır. `Sequential` çocukları `"0"`, `"1"`, … diye kaydeder.

```cpp
auto model = Sequential({
    std::make_shared<Linear>(8, 16),
    std::make_shared<ReLU>(),
    std::make_shared<Linear>(16, 3)
});
for (Parameter* p : model.parameters()) {
    p->data;            // ağırlık
    p->grad;            // biriken gradyan
    p->requires_grad;   // false ise backward o parametreye yazmaz
}
for (const auto& [name, p] : model.named_parameters()) {
    // name: "0.weight", "0.bias", "2.weight", "2.bias"
    (void)name;
    (void)p;
}
model.zero_grad();
```

`state_dict()` parametreleri ve `BatchNorm` koşu istatistiklerini kopyalar. `load_state_dict` varsayılan olarak katı anahtar ve şekil kontrolü yapar. Dosya yardımcıları `save_state_dict` ve `load_state_dict_file`.

## train() ve eval()

`train()` ağacı eğitim moduna, `eval()` değerlendirme moduna alır. Çağrı çocuklara iner. `is_training()` o nesnenin bayrağını okur.

Bu bayrağı okuyan katmanlar:

- `Dropout`: eğitimde ters dropout (tutulan değer `1/(1-p)` ile ölçeklenir). `eval()` veya `p == 0` kimlik dönüşümüdür. `backward`, son `forward` anındaki modu ve maskeyi kullanır.
- `BatchNorm`, `BatchNorm1D`, `BatchNorm2D`: eğitimde mini-batch istatistiği ve `running_mean` / `running_var` güncellemesi. `eval()` koşu istatistiğini kullanır. Koşu tamponları `state_dict` içindedir.
- `MultiHeadAttention`: eğitimde ve `dropout > 0` iken dikkat dropout’u uygular.

`LayerNorm`, `GroupNorm`, `InstanceNorm`, `Linear`, evrişim, havuzlama, gömme ve tekrarlayan katmanlar bu bayraktan bağımsız hesaplar. Yine de modeli `eval()` ile çıkarıma almak, Dropout ve BatchNorm içeren ağlarda gerekir.

```cpp
model.train();
Tensor logits = model.forward(batch);
model.zero_grad();
model.backward(grad_logits);

model.eval();
Tensor inference = model.forward(batch);
```

## Katmanlar

Kurucular başlıklardaki imzalardır. Boyutlar pozitif olmalıdır; aksi halde `ModelError`.

### Linear

`Linear(in_features, out_features, use_bias = true)`. Girdi son ekseni `in_features`. Ağırlık `[out, in]`, `Y = X Wᵀ + b`. Kaiming uniform, bias sıfır.

### Evrişim

`Conv1D(in_channels, out_channels, kernel, stride = 1, padding = 0, dilation = 1, groups = 1, bias = true)`  
Girdi `[N, C, L]`.

`Conv2D(in_channels, out_channels, kernel, stride = 1, padding = 0, dilation = 1, groups = 1, bias = true)`  
Girdi `[N, C, H, W]`. Eksen başına kernel, stride, padding ve dilation alan uzun kurucu da vardır.

`Conv3D(in_channels, out_channels, kernel, stride = 1, padding = 0, dilation = 1, groups = 1, bias = true)`  
Girdi `[N, C, D, H, W]`.

`DepthwiseConv2D(channels, kernel, stride = 1, padding = 0)`  
`out_channels = channels` ve `groups = channels`. Kanal çarpanı yoktur.

`groups` giriş ve çıkış kanallarını bölmelidir. Çekirdek ters çevrilmez (çapraz korelasyon). `ConvNd` doğrudan kurulmaz.

21 Ekim 2026 akşamı, bu makinede, rakip protokolündeki CPU Conv2d PyTorch’un 31.6 GFLOPS değerinin gerisindeydi. Nexus o akşam yaklaşık 4.9 ile 6.8 GFLOPS arasındaydı.

### Normalizasyon

`BatchNorm(num_features, eps = 1e-5, momentum = 0.1, affine = true)`  
`BatchNorm1D` ve `BatchNorm2D` aynı kurucuyu kullanır. Kanal ekseni 1’dir.

`LayerNorm(normalized_size, eps = 1e-5)` son ekseni normalize eder.

`GroupNorm(num_groups, num_channels, eps = 1e-5, affine = true)`  
`InstanceNorm(num_channels, eps = 1e-5, affine = true)` her kanalı kendi grubu sayar. İkisi de koşu istatistiği tutmaz.

### Tekrarlayan katmanlar

`RNN`, `LSTM` ve `GRU`: `(input_size, hidden_size, num_layers = 1, truncate_bptt = 0)`.

Girdi `[B, T, F]`, çıktı `[B, T, H]`, batch-first, tek yön. Her `forward` sıfır durumla başlar. `truncate_bptt > 0` ise gradyan yalnız son k adıma akar. LSTM kapı sırası i, f, g, o. Unutma kapısı bias’ı 1 ile başlar.

### Dikkat ve Transformer

`MultiHeadAttention(embed, num_heads, dropout = 0, causal = false)`. `embed`, baş sayısına bölünür. Öz-dikkat girdisi `[B, T, E]`. Ayrı Q, K, V için `forward(query, key, value, mask)`.

`PositionalEncoding(embed, max_len = 512, learnable = false)`. Girdi `[B, T, E]`.

`TransformerEncoderLayer(embed, heads, ff = 0, dropout = 0.1)`. `ff == 0` iken ara genişlik `4 * embed`. Post-norm: dikkat artığı, sonra GELU ileri besleme artığı.

`TransformerDecoderLayer(embed, heads, ff = 0, dropout = 0.1)`. `forward(target, memory)` gerekir. Öz-dikkat nedenseldir.

### Havuzlama, gömme, şekil

`MaxPool2D(kernel, stride = -1, padding = 0)` ve `AvgPool2D(kernel, stride = -1, padding = 0, count_include_pad = true)`. Negatif stride, kernel boyutuna döner. Girdi NCHW.

`GlobalAvgPool2D()` çıktısı `[N, C]`.

`Embedding(num_embeddings, embedding_dim, padding_idx = -1)`. İndeks float tensörde taşınır. `[B, T]` girdisi `[B, T, D]` olur.

`Flatten` rank ≥ 2 girdiyi `[N, -1]` yapar. `Reshape` kurucuya verilen şekli uygular.

`Dropout(probability = 0.5, seed = 0xD1CE)`. `ResidualBlock(shared_ptr<Module>)` artığı `F(x) + x` hesaplar.

`Sequential`, `ModuleList` ve `ModuleDict` çocuk modülleri kaydeder. `train`, `eval`, `parameters` ve `state_dict` bu kayıttan yürür.

### Aktivasyonlar

`ReLU`, `LeakyReLU(alpha = 0.01)`, `ELU`, `GELU`, `Sigmoid`, `Tanh`, `SiLU`, `Mish`, `Softmax`, `LogSoftmax`, `PReLU`. `PReLU` dışında parametre yoktur. Softmax ve LogSoftmax son eksendedir.

## 21 Ekim 2026 akşamı

Bu makine AMD Ryzen 7 5800H. Dünya sıralaması yoktur.

GEMM çekirdeği `avx2-4x16` olarak kaldı. N=2048 için Nexus 330, 362 ve 369; TensorFlow 389. Birim o protokolün GFLOPS sütunudur.

CPU Conv2d, PyTorch 31.6 GFLOPS değerinin gerisindeydi. Nexus yaklaşık 4.9 ile 6.8 GFLOPS arasındaydı.

Depthwise AVX, o sütunda PyTorch 4.93 ve TensorFlow 0.688 değerlerinin önündeydi.

AVX-512 bu işlemcide seçilmez.

29 Eylül 2026 tek iş parçacıklı tabloları `benchmarks/RESULTS.md` dosyasındadır. `benchmarks/results/` altındaki ham günlüklerin üzerine bu oranlar yazılmaz.
