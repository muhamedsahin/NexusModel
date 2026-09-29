import { Article, s, tx } from "./types";
export const layers: Article[] = [
  {
    slug: "linear",
    group: 2,
    title: tx("Linear", "Linear"),
    description: tx(
      "Tam bağlı katman, ağırlık düzeni ve analitik türevler.",
      "Fully connected layers, weight layout, and analytic derivatives.",
    ),
    source: "include/nexus_model/layers/linear.hpp",
    sections: [
      s(
        "signature",
        "Kurucu ve parametreler",
        "Constructor and parameters",
        "Linear(in_features,out_features,use_bias=true). Boyutlar pozitif olmalıdır. weight() [out,in] Parameter, bias() [out] Parameter döndürür; bias kapalıysa nullptr olur. Ağırlık Kaiming uniform, bias sıfır ile başlar.",
        "Linear(in_features,out_features,use_bias=true). Feature counts must be positive. weight() returns an [out,in] Parameter; bias() returns an [out] Parameter or nullptr when disabled. Weights start with Kaiming uniform, biases with zeros.",
        {
          code: "Linear layer(64,128,true);\nauto x=Tensor::zeros({4,16,64});\nauto y=layer.forward(x); // [4,16,128]",
        },
      ),
      s(
        "forward",
        "İleri geçiş",
        "Forward pass",
        "Son eksen in_features ile eşleşir. Önceki eksenler düzleştirilmiş satırlar olarak işlenir ve çıktıda korunur. W=[out,in] düzeni fiziksel ağırlık transpozisyonu gerektirmez.",
        "The final axis must match in_features. Earlier axes are flattened into rows and preserved in the output. W=[out,in] layout avoids physically transposing weights.",
        { formula: "Y=XW^T+\\mathbf1b^T" },
      ),
      s(
        "backward",
        "Analitik geri geçiş",
        "Analytic backward",
        "dY=[M,O], X=[M,I], W=[O,I]. Katman dX tamponunu sıfırlar, dW/db mevcut birikime eklenir. requires_grad=false yalnız ilgili parametre katkısını kapatır.",
        "dY=[M,O], X=[M,I], W=[O,I]. The layer clears dX; dW/db add to existing accumulations. requires_grad=false disables only the corresponding parameter contribution.",
        {
          formula:
            "dX=dY\\,W,\\qquad dW\\mathrel{+}=dY^TX,\\qquad db\\mathrel{+}=\\sum_m dY_m",
        },
      ),
      s(
        "cost",
        "Hesap maliyeti",
        "Compute cost",
        "İleri çarpım yaklaşık 2MIO FLOP, ağırlıklar bias hariç 4IO bayt gerektirir. Eğitimde gradyanlar ve önbellekler ek bellek tutar. Aynı modülde yeni forward eski backward önbelleğini değiştirir.",
        "Forward multiplication takes approximately 2MIO FLOPs; weights occupy 4IO bytes excluding bias. Training adds gradients and caches. A new forward on the same module replaces the previous backward cache.",
      ),
    ],
  },
  {
    slug: "convolution",
    group: 2,
    title: tx("Evrişim katmanları", "Convolution layers"),
    description: tx(
      "Conv1D/2D/3D, groups, dilation ve depthwise.",
      "Conv1D/2D/3D, groups, dilation, and depthwise.",
    ),
    source: "include/nexus_model/layers/conv.hpp",
    sections: [
      s(
        "api",
        "Kurucular ve veri düzeni",
        "Constructors and layout",
        "Conv1D/2D/3D(in_channels,out_channels,kernel,stride=1,padding=0,dilation=1,groups=1,bias=true). Girdi [N,C,L], [N,C,H,W] veya [N,C,D,H,W]. W=[K,C/groups,çekirdek...]. Giriş ve çıkış kanalları groups ile bölünebilmelidir. Kanal-last desteklenmez.",
        "Conv1D/2D/3D(in_channels,out_channels,kernel,stride=1,padding=0,dilation=1,groups=1,bias=true). Inputs use [N,C,L], [N,C,H,W], or [N,C,D,H,W]. W=[K,C/groups,kernel...]. Input and output channels must divide by groups. Channels-last is unsupported.",
      ),
      s(
        "shape",
        "Çıktı boyutu",
        "Output size",
        "Her uzamsal eksen bağımsız hesaplanır. padding sıfır dolgu, dilation çekirdek örnekleri arasındaki aralıktır. Conv2D eksen başına kernel/stride/padding/dilation alan uzun bir kurucu da sunar.",
        "Each spatial axis is computed independently. Padding adds zeros; dilation spaces kernel samples. Conv2D also exposes a longer constructor with per-axis kernel/stride/padding/dilation.",
        {
          formula: "O=\\left\\lfloor\\frac{I+2P-D(K-1)-1}{S}\\right\\rfloor+1",
          code: "Conv2D conv(3,32,3,1,1);\nauto x=Tensor::zeros({8,3,28,28});\nauto y=conv.forward(x); // [8,32,28,28]",
        },
      ),
      s(
        "math",
        "Çapraz korelasyon ve im2col",
        "Cross-correlation and im2col",
        "Çekirdek ters çevrilmez: çapraz korelasyon uygulanır. im2col pencereleri çalışma alanına açar, GEMM filtrelerle çarpar. Backward dW ve db biriktirir; col2im örtüşen giriş katkılarını toplar. Workspace maliyeti çekirdek hacmi ve çıktı konumlarıyla büyür.",
        "The kernel is not flipped: this is cross-correlation. im2col unfolds windows into workspace and GEMM multiplies by filters. Backward accumulates dW and db; col2im sums overlapping input contributions. Workspace grows with kernel volume and output positions.",
        { formula: "Y_{n,k,p}=b_k+\\sum_{c,r}W_{k,c,r}X_{n,c,pS-P+rD}" },
      ),
      s(
        "depthwise",
        "DepthwiseConv2D",
        "DepthwiseConv2D",
        "DepthwiseConv2D(channels,kernel,stride=1,padding=0), groups=channels ve out_channels=channels kullanır. Kanal çarpanı seçeneği yoktur. Ardından 1×1 Conv2D ile kanallar karıştırılabilir.",
        "DepthwiseConv2D(channels,kernel,stride=1,padding=0) uses groups=channels and out_channels=channels. There is no channel multiplier option. Follow with 1×1 Conv2D to mix channels.",
      ),
    ],
  },
  {
    slug: "pooling",
    group: 2,
    title: tx("Havuzlama", "Pooling"),
    description: tx(
      "Uzamsal indirgeme ve gradyan dağılımı.",
      "Spatial reduction and gradient distribution.",
    ),
    source: "include/nexus_model/layers/pooling.hpp",
    sections: [
      s(
        "max",
        "MaxPool2D",
        "MaxPool2D",
        "MaxPool2D(kernel,stride=-1,padding=0). Negatif varsayılan stride kernel boyutuna dönüşür. Rank-4 NCHW girdi ister. İlk maksimum kazanır; ileri geçiş argmax saklar. Backward kazanan hücrelere scatter-add yapar.",
        "MaxPool2D(kernel,stride=-1,padding=0). Negative default stride becomes the kernel size. Requires rank-4 NCHW. First maximum wins; forward stores argmax. Backward scatter-adds to winning cells.",
        {
          code: "MaxPool2D pool(2);\nauto x=Tensor::zeros({8,32,28,28});\nauto y=pool.forward(x); // [8,32,14,14]",
        },
      ),
      s(
        "avg",
        "AvgPool2D",
        "AvgPool2D",
        "AvgPool2D(kernel,stride=-1,padding=0,count_include_pad=true). Payda varsayılan olarak dolgu hücrelerini içerir. false ise yalnız geçerli hücreler sayılır. Backward aynı payda ile gradyanı dağıtır, örtüşmeler toplanır.",
        "AvgPool2D(kernel,stride=-1,padding=0,count_include_pad=true). The default denominator includes padding. With false, only valid cells count. Backward distributes gradients with the same denominator and sums overlaps.",
        {
          formula:
            "y=\\frac1m\\sum_{i\\in\\mathcal W}x_i,\\qquad dx_i\\mathrel{+}=\\frac{dy}{m}",
        },
      ),
      s(
        "global",
        "GlobalAvgPool2D",
        "GlobalAvgPool2D",
        "GlobalAvgPool2D() her kanalın H×W ortalamasını alır. Çıktı doğrudan [N,C] şeklindedir; ek Flatten gerekmez. Her hücre 1/(HW) oranında gradyan alır.",
        "GlobalAvgPool2D() averages H×W per channel. Output is directly [N,C]; no extra Flatten is needed. Every cell receives gradient scaled by 1/(HW).",
      ),
    ],
  },
  {
    slug: "normalization",
    group: 2,
    title: tx("Normalizasyon", "Normalization"),
    description: tx(
      "BatchNorm, LayerNorm, GroupNorm ve InstanceNorm.",
      "BatchNorm, LayerNorm, GroupNorm, and InstanceNorm.",
    ),
    source: "include/nexus_model/layers/normalization.hpp",
    sections: [
      s(
        "equation",
        "Ortak dönüşüm",
        "Common transformation",
        "Seçilen eksenlerde ortalama ve varyans hesaplanır. epsilon sayısal kararlılık, gamma öğrenilebilir ölçek, beta kaymadır. Gruplama eksenleri her türün istatistiksel anlamını belirler.",
        "Mean and variance are computed over selected axes. Epsilon provides stability, gamma is learnable scale, beta is shift. Grouping axes define the statistical meaning of each variant.",
        {
          formula:
            "\\hat x=\\frac{x-\\mu}{\\sqrt{\\sigma^2+\\epsilon}},\\qquad y=\\gamma\\hat x+\\beta",
        },
      ),
      s(
        "batch",
        "BatchNorm1D / BatchNorm2D",
        "BatchNorm1D / BatchNorm2D",
        "BatchNorm(num_features,eps=1e-5,momentum=0.1,affine=true). 1D/2D aynı tabanı kullanır, kanal ekseni 1’dir. Eğitimde batch ve uzamsal eksenler indirgenir. Normalizasyonda /m varyansı, running_var için m>1 olduğunda /(m−1) düzeltmesi vardır. Eval running_mean/running_var kullanır; bu buffer’lar state_dict içinde saklanır.",
        "BatchNorm(num_features,eps=1e-5,momentum=0.1,affine=true). 1D/2D share the base, with channel axis 1. Training reduces batch and spatial axes. Normalization uses /m variance; running_var uses the /(m−1) correction when m>1. Eval uses running_mean/running_var, which are saved in state_dict.",
        {
          formula:
            "\\mu_{run}\\leftarrow(1-\\alpha)\\mu_{run}+\\alpha\\mu_{batch}",
        },
      ),
      s(
        "layer",
        "LayerNorm",
        "LayerNorm",
        "LayerNorm(normalized_size,eps=1e-5) yalnız son ekseni normalize eder; o eksen normalized_size olmalı. Batch bağımsızdır, running istatistiği yoktur. weight=1, bias=0 başlar; affine kapatma seçeneği yoktur.",
        "LayerNorm(normalized_size,eps=1e-5) normalizes only the final axis, which must equal normalized_size. It is batch-independent with no running statistics. weight starts at 1, bias at 0; there is no affine-disable option.",
        {
          code: "LayerNorm norm(64);\nauto y=norm.forward(Tensor::zeros({4,32,64}));",
        },
      ),
      s(
        "group",
        "GroupNorm / InstanceNorm",
        "GroupNorm / InstanceNorm",
        "GroupNorm(num_groups,num_channels,eps=1e-5,affine=true) her örneğin kanallarını eşit gruplara ayırır. Kanal sayısı gruplara bölünür. InstanceNorm(num_channels,eps=1e-5,affine=true) groups=C olan GroupNorm’dur. Running istatistiği tutmazlar.",
        "GroupNorm(num_groups,num_channels,eps=1e-5,affine=true) groups channels equally within each sample. Channels must divide by groups. InstanceNorm(num_channels,eps=1e-5,affine=true) is GroupNorm with groups=C. Neither keeps running statistics.",
      ),
      s(
        "gradient",
        "Normalizasyon türevi",
        "Normalization derivative",
        "Bir grup için u=dy⊙gamma olsun. dgamma=Σ(dy⊙x̂), dbeta=Σdy uygun eksenlerde birikir. Eval BatchNorm’da istatistikler sabit olduğu için eğitim türevi yerine doğrudan ölçek uygulanır.",
        "For a group, define u=dy⊙gamma. dgamma=Σ(dy⊙x̂), dbeta=Σdy accumulate over appropriate axes. In eval BatchNorm, statistics are constant, so backward applies direct scaling.",
        {
          formula:
            "dx=\\frac{mu-\\sum u-\\hat x\\sum(u\\hat x)}{m\\sqrt{\\sigma^2+\\epsilon}}",
        },
      ),
    ],
  },
  {
    slug: "activations",
    group: 2,
    title: tx("Aktivasyonlar", "Activations"),
    description: tx(
      "11 aktivasyonun matematiği, türevleri ve varsayılanları.",
      "Mathematics, derivatives, and defaults for 11 activations.",
    ),
    source: "include/nexus_model/activations/activations.hpp",
    sections: [
      s(
        "elementwise",
        "Eleman bazlı dönüşümler",
        "Elementwise transformations",
        "Şekil korunur. PReLU tek ortak alpha öğrenir, kanal başına ayrı eğim yoktur. Diğer aktivasyonların parametresi yoktur. GELU varsayılanı kesin erf formu; true tanh yaklaşımını seçer.",
        "Shape is preserved. PReLU learns a single shared alpha, not a per-channel slope. Other activations have no parameters. GELU defaults to the exact erf form; true selects the tanh approximation.",
        {
          table: {
            headers: [
              tx("API", "API"),
              tx("İleri", "Forward"),
              tx("Yerel türev", "Local derivative"),
            ],
            rows: [
              ["ReLU()", "max(0,x)", "x>0 ? 1 : 0"],
              ["LeakyReLU(0.01f)", "x>0 ? x : αx", "x>0 ? 1 : α"],
              ["PReLU(0.25f)", "x>0 ? x : αx", "dα += Σ(x≤0) dy·x"],
              ["ELU(1.f)", "x>0 ? x : α(exp(x)−1)", "x>0 ? 1 : y+α"],
              ["GELU(false)", "x·Φ(x)", "Φ(x)+x·φ(x)"],
              [
                "GELU(true)",
                "0.5x[1+tanh(c(x+0.044715x³))]",
                "0.5(1+tanh u)+0.5x(1−tanh²u)c(1+0.134145x²); u=c(x+0.044715x³)",
              ],
              ["Sigmoid()", "σ=1/(1+exp(−x))", "σ(1−σ)"],
              ["Tanh()", "tanh(x)", "1−y²"],
              ["SiLU()", "x·σ(x)", "σ+xσ(1−σ)"],
              ["Mish()", "x·tanh(softplus(x))", "tanh(s)+xσ(x)(1−tanh²(s))"],
            ],
          },
        },
      ),
      s(
        "softmax",
        "Softmax",
        "Softmax",
        "Son eksende satır maksimumu çıkarılıp üstel hesaplanır. Her satır yaklaşık 1 toplar. Backward tam Jacobian yerine iç çarpım kullanır. Tümü −∞ satırı geçerli dağılım değildir.",
        "Subtract the row maximum and exponentiate on the final axis. Rows sum approximately to 1. Backward uses a dot product instead of a full Jacobian. An all−∞ row is not a valid distribution.",
        {
          formula:
            "p_i=\\frac{e^{x_i-\\max x}}{\\sum_je^{x_j-\\max x}},\\qquad dx_i=p_i(g_i-\\sum_jg_jp_j)",
        },
      ),
      s(
        "log",
        "LogSoftmax",
        "LogSoftmax",
        "Son eksende kararlı x−logsumexp(x) uygulanır. Çıktı log-olasılıktır. Çok küçük olasılıkların önce hesaplanıp log alınması yerine doğrudan bu dönüşüm kullanılır.",
        "Uses stable x−logsumexp(x) on the final axis. Outputs are log probabilities, computed directly instead of first forming very small probabilities and then taking their logarithm.",
        { formula: "dx_i=g_i-e^{y_i}\\sum_jg_j" },
      ),
      s(
        "corners",
        "Sıfır noktası ve testler",
        "Zero and testing",
        "ReLU sıfırda türev 0 seçer. LeakyReLU/PReLU negatif dalı kullanır. Sonlu fark testlerinde kink noktalarını atlayın. SIMD ve float32 farkları için bit eşitliği yerine tolerans kullanın.",
        "ReLU chooses derivative 0 at zero. LeakyReLU/PReLU use the negative branch. Skip kink points in finite-difference checks. Use tolerances rather than bitwise equality for SIMD and float32 differences.",
      ),
    ],
  },
  {
    slug: "dropout",
    group: 2,
    title: tx("Dropout", "Dropout"),
    description: tx(
      "Ters dropout, tohum ve train/eval davranışı.",
      "Inverted dropout, seeds, and train/eval behavior.",
    ),
    source: "include/nexus_model/layers/dropout.hpp",
    sections: [
      s(
        "api",
        "Kurucu",
        "Constructor",
        "Dropout(probability=0.5f,seed=0xD1CEU), probability ∈ [0,1]. Maske her eğitim forward çağrısında değişir. Aynı tohum ve çağrı sırası tekrarlanabilir dizi verir. init::manual_seed bu ayrı seed alanını değiştirmez.",
        "Dropout(probability=0.5f,seed=0xD1CEU), probability ∈ [0,1]. Masks change on each training forward. Same seed and call sequence give a repeatable series. init::manual_seed does not change this separate seed field.",
      ),
      s(
        "math",
        "Beklenen değeri koruma",
        "Preserve the expected value",
        "Tutulan değerler 1/(1−p) ile ölçeklenir. p=1 özel durumunda çıktı ve türev sıfır; eval veya p=0 için kimlik dönüşümüdür.",
        "Retained values scale by 1/(1−p). At p=1 output and derivative are zero; eval or p=0 applies the identity.",
        {
          formula:
            "m_i\\sim Bernoulli(1-p),\\qquad y_i=\\frac{x_im_i}{1-p},\\qquad dx_i=\\frac{g_im_i}{1-p}",
        },
      ),
      s(
        "mode",
        "Kaydedilen ileri mod",
        "Recorded forward mode",
        "training_at_forward saklanır; backward son forward maskesini ve modunu kullanır. Attention dropout ayrı uygulanır ve p<1 gerektirir.",
        "training_at_forward is saved; backward uses that forward mask and mode. Attention dropout is implemented separately and requires p<1.",
      ),
    ],
  },
  {
    slug: "embedding",
    group: 2,
    title: tx("Embedding", "Embedding"),
    description: tx(
      "Token gather ve satır bazlı scatter-add.",
      "Token gathering and row-wise scatter-add.",
    ),
    source: "include/nexus_model/layers/embedding.hpp",
    sections: [
      s(
        "api",
        "Tablo ve şekiller",
        "Table and shapes",
        "Embedding(num_embeddings,embedding_dim,padding_idx=-1). [V,D] ağırlık normal(0,1) ile başlar. Girdi şekline D eklenir: [B,T]→[B,T,D]. Ek eksen için girdi rank≤7 olmalı.",
        "Embedding(num_embeddings,embedding_dim,padding_idx=-1). [V,D] weights start normal(0,1). D is appended to the input shape: [B,T]→[B,T,D]. Keep input rank≤7 for the extra axis.",
        {
          code: "Embedding tokens(1000,64,0);\nauto ids=Tensor::from_values({1,4},{12,9,0,4});\nauto y=tokens.forward(ids); // [1,4,64]",
        },
      ),
      s(
        "indices",
        "İndeks sözleşmesi",
        "Index contract",
        "İndeksler float32 Tensor içinde taşınır ve çekirdekte int’e çevrilir. Tamsayı değerli, geçerli aralıklı indeksler kullanın; kesirli değerler reddedilmeyebilir. padding_idx negatif değilse [0,V) içinde olmalı.",
        "Indices are stored in float32 Tensor and cast to int in the kernel. Use integer-valued indices in range; fractional values may not be rejected. Nonnegative padding_idx must lie in [0,V).",
      ),
      s(
        "gradient",
        "Tekrarlanan token türevleri",
        "Repeated-token gradients",
        "Aynı token katkıları aynı satırda birikir. padding_idx çıktı ve gradyanı sıfırdır. backward indekslere sıfır türev döndürür. Yalnız dokunulan satırlar güncellense de grad yoğun tensördür.",
        "Repeated tokens add into the same row. padding_idx has zero output and no gradient contribution. backward returns zero index gradients. grad remains dense despite updating only touched rows.",
        { formula: "Y_i=W_{k_i},\\qquad dW_j\\mathrel{+}=\\sum_{i:k_i=j}dY_i" },
      ),
    ],
  },
  {
    slug: "recurrent",
    group: 2,
    title: tx("RNN, LSTM ve GRU", "RNN, LSTM & GRU"),
    description: tx(
      "Kapı denklemleri ve zaman boyunca geri yayılım.",
      "Gate equations and backpropagation through time.",
    ),
    source: "include/nexus_model/layers/recurrent.hpp",
    sections: [
      s(
        "api",
        "Ortak arayüz",
        "Shared interface",
        "RNN/LSTM/GRU(input_size,hidden_size,num_layers=1,truncate_bptt=0). Girdi [B,T,F], çıktı [B,T,H]; tek yönlü ve batch-first. Ayrı başlangıç/son durum API’si yoktur. Her forward sıfır durumla başlar. Boyutları, pozitif T ve katman sayısını çağıran sağlamalı.",
        "RNN/LSTM/GRU(input_size,hidden_size,num_layers=1,truncate_bptt=0). Inputs [B,T,F], outputs [B,T,H]; unidirectional and batch-first. No separate initial/final-state API. Every forward starts from zeros. Callers must ensure valid dimensions, positive T, and positive layer count.",
        {
          code: "LSTM sequence(32,64,2);\nauto x=Tensor::zeros({4,20,32});\nauto y=sequence.forward(x); // [4,20,64]",
        },
      ),
      s(
        "rnn",
        "Tanh RNN",
        "Tanh RNN",
        "Giriş ağırlıkları Kaiming, recurrent ağırlıkları orthogonal başlar. Katman başına lN.weight_ih, lN.weight_hh, lN.bias_ih, lN.bias_hh kaydedilir.",
        "Input weights use Kaiming, recurrent weights use orthogonal initialization. Per-layer keys are lN.weight_ih, lN.weight_hh, lN.bias_ih, lN.bias_hh.",
        { formula: "h_t=\\tanh(W_{ih}x_t+b_{ih}+W_{hh}h_{t-1}+b_{hh})" },
      ),
      s(
        "lstm",
        "LSTM kapıları",
        "LSTM gates",
        "Düzen i,f,g,o: giriş, unutma, aday, çıkış. Giriş bias’ının forget bloğu 1 başlar. Backward gizli ve hücre gradyanlarını zamanda taşır.",
        "Order i,f,g,o: input, forget, candidate, output. The forget block of the input bias starts at 1. Backward propagates both hidden and cell gradients through time.",
        {
          formula:
            "c_t=f_t\\odot c_{t-1}+i_t\\odot g_t,\\qquad h_t=o_t\\odot\\tanh(c_t)",
        },
      ),
      s(
        "gru",
        "GRU kapıları",
        "GRU gates",
        "r ve z sigmoid; n tanh kullanır. Reset kapısı recurrent aday projeksiyonuna bias dahil uygulandıktan sonra giriş projeksiyonuyla toplanır. Ağırlık aktarırken GRU varyantı uyumunu kontrol edin.",
        "r and z use sigmoid; n uses tanh. Reset is applied to the recurrent candidate projection including bias, before adding the input projection. Check GRU variant compatibility when transferring weights.",
        {
          formula:
            "n_t=\\tanh(W_{in}x_t+b_{in}+r_t\\odot(W_{hn}h_{t-1}+b_{hn})),\\quad h_t=(1-z_t)\\odot n_t+z_t\\odot h_{t-1}",
        },
      ),
      s(
        "bptt",
        "Kesilmiş BPTT",
        "Truncated BPTT",
        "truncate_bptt=k>0 yalnız son k adımı türetir: t0=max(0,T−k). Önceki giriş konumları sıfır gradyan alır. Geçmiş durum forward’da kullanılır, kesme sınırını türev geçmez. Bu otomatik kayan pencereli eğitim değildir.",
        "truncate_bptt=k>0 differentiates only the last k steps: t0=max(0,T−k). Earlier inputs keep zero gradients. Historical states still affect forward, but gradients stop at the cutoff. This is not automatic sliding-window training.",
      ),
    ],
  },
  {
    slug: "attention",
    group: 2,
    title: tx("Multi-head attention", "Multi-head attention"),
    description: tx(
      "Q/K/V, maskeler ve self/cross attention türevleri.",
      "Q/K/V, masks, and self/cross-attention derivatives.",
    ),
    source: "include/nexus_model/layers/attention.hpp",
    sections: [
      s(
        "api",
        "Boyut sözleşmesi",
        "Shape contract",
        "MultiHeadAttention(embed,num_heads,dropout=0,causal=false). num_heads pozitif olmalı ve embed’i bölmeli; bölme kontrol öncesinde yapıldığından sıfır vermeyin. Q/K/V rank-3 [B,T,E]. Dört projeksiyon bias içeren Linear katmanlarıdır.",
        "MultiHeadAttention(embed,num_heads,dropout=0,causal=false). num_heads must be positive and divide embed; division precedes validation, so never pass zero. Q/K/V use rank-3 [B,T,E]. All four projections are Linear layers with bias.",
      ),
      s(
        "math",
        "Ölçekli noktasal çarpım",
        "Scaled dot-product",
        "Başlar [B·H,T,D], D=E/H şekline ayrılır. Maskeli ve ölçekli skorlardan softmax, isteğe bağlı dropout ve V çarpımı gelir. Başlar birleştirilip çıkış projeksiyonuna gider.",
        "Heads split into [B·H,T,D], D=E/H. Scaled masked scores pass through softmax, optional dropout, and multiplication by V. Heads merge and pass through the output projection.",
        {
          formula:
            "Attention(Q,K,V)=softmax\\left(\\frac{QK^T}{\\sqrt{d_k}}+M\\right)V",
        },
      ),
      s(
        "mask",
        "Maskeler",
        "Masks",
        "forward(query,key,value,mask), nullptr veya [Tq,Tk] toplamsal float maske alır. 0 izin, büyük negatif değer bastırmadır; boolean maske değildir. causal geleceğe −1e9 ekler ve Tq=Tk ister. Her satırda geçerli anahtar bırakın.",
        "forward(query,key,value,mask) takes nullptr or an additive float [Tq,Tk] mask. Zero permits, large negative suppresses; it is not boolean. causal adds −1e9 to future positions and requires Tq=Tk. Leave a valid key in each row.",
        {
          code: "MultiHeadAttention attn(64,4,0.f,true);\nauto x=Tensor::zeros({4,32,64});\nauto y=attn.forward(x);\nauto dx=attn.backward(Tensor::full({4,32,64},1.f));",
        },
      ),
      s(
        "gradient",
        "Self / cross gradyan",
        "Self / cross gradients",
        "forward(x) dq+dk+dv döndürür. Çok argümanlı overload self durumunu nesne adresiyle tanır; aynı depolu farklı Tensor başlıkları self sayılmaz. Cross backward dq döndürür. dk/dv grad_key_input()/grad_value_input() içindedir; aynı memory’den geliyorlarsa toplanmalıdır.",
        "forward(x) returns dq+dk+dv. The multi-argument overload detects self by object addresses; separate Tensor headers sharing storage are not self. Cross backward returns dq. dk/dv are in grad_key_input()/grad_value_input(); sum them if they refer to the same memory.",
      ),
      s(
        "cost",
        "Karesel skor belleği",
        "Quadratic score memory",
        "Skorlar B·H·Tq·Tk değer tutar. Bu FlashAttention değildir. Analitik Linear projeksiyonları, yerel MicroTape softmax/matmul zinciriyle birlikte çalışır.",
        "Scores store B·H·Tq·Tk values. This is not FlashAttention. Analytic Linear projections work alongside a local MicroTape softmax/matmul chain.",
      ),
    ],
  },
  {
    slug: "transformer",
    group: 2,
    title: tx("Transformer blokları", "Transformer blocks"),
    description: tx(
      "Konumsal kodlama, post-norm encoder ve decoder.",
      "Positional encoding, post-norm encoder, and decoder.",
    ),
    source: "include/nexus_model/layers/attention.hpp",
    sections: [
      s(
        "position",
        "PositionalEncoding",
        "PositionalEncoding",
        "PositionalEncoding(embed,max_len=512,learnable=false) sinüs/kosinüsü [B,T,E] girdisine ekler. Sabit tablo büyür; learnable parametre tablosu ilk boyutta kalır, T≤max_len koruyun. Öğrenilebilir tablo sinüsoidal başlar ve batch boyunca gradyan toplar.",
        "PositionalEncoding(embed,max_len=512,learnable=false) adds sine/cosine values to [B,T,E]. Fixed tables grow, but learnable parameter tables retain their initial size; maintain T≤max_len. Learnable tables start sinusoidal and accumulate gradients over the batch.",
        {
          formula:
            "PE_{p,2i}=\\sin(p/10000^{2i/E}),\\quad PE_{p,2i+1}=\\cos(p/10000^{2i/E})",
        },
      ),
      s(
        "encoder",
        "TransformerEncoderLayer",
        "TransformerEncoderLayer",
        "TransformerEncoderLayer(embed,heads,ff=0,dropout=0.1); ff≤0 ise 4·embed. Self-attention→dropout→residual→LayerNorm; Linear→GELU→Linear→dropout→residual→LayerNorm. Post-norm mimarisidir.",
        "TransformerEncoderLayer(embed,heads,ff=0,dropout=0.1); ff≤0 uses 4·embed. Self-attention→dropout→residual→LayerNorm; Linear→GELU→Linear→dropout→residual→LayerNorm. This is post-norm.",
        {
          formula: "u=LN(x+D(A(x))),\\qquad y=LN(u+D(FFN(u)))",
          code: "Sequential encoder({\n  std::make_shared<PositionalEncoding>(64,128),\n  std::make_shared<TransformerEncoderLayer>(64,4,256,0.1f)\n});\nauto y=encoder.forward(Tensor::zeros({4,32,64}));",
        },
      ),
      s(
        "decoder",
        "TransformerDecoderLayer",
        "TransformerDecoderLayer",
        "TransformerDecoderLayer(embed,heads,ff=0,dropout=0.1) maskeli self-attention, cross-attention ve FFN içerir. Önce forward(target,memory) çağırın. Tek argümanlı forward daha önce kaydedilmiş memory ister; ilk çağrıda ModelError üretir.",
        "TransformerDecoderLayer(embed,heads,ff=0,dropout=0.1) contains masked self-attention, cross-attention, and FFN. Call forward(target,memory) first. Single-argument forward requires previously cached memory and throws ModelError on the first call.",
      ),
      s(
        "memory",
        "Memory türevi sınırı",
        "Memory-gradient limitation",
        "Decoder backward yalnız target türevi döndürür. İç cross-attention’ın dk/dv türevleri dönüşe eklenmez ve decoder public API’sinde sunulmaz. Tam encoder-decoder eğitimi için bu aktarımı ekleyen genişletme veya açık cross-attention akışı gerekir.",
        "Decoder backward returns only target gradients. Inner cross-attention dk/dv are not included or exposed by the decoder public API. Full encoder-decoder training requires an extension that propagates them or an explicit cross-attention pipeline.",
      ),
    ],
  },
];

// Additional derivations are kept next to the layer reference.
layers
  .find((a) => a.slug === "recurrent")!
  .sections.push(
    s(
      "gate-equations",
      "LSTM kapılarının açılımı",
      "Expanded LSTM gate equations",
      "Ağırlık projeksiyonu a_t=W_ih x_t+b_ih+W_hh h_(t−1)+b_hh dört H genişlikli bloğa ayrılır. i/f/o sigmoid, g tanh kullanır. Sigmoid kapılar [0,1] aralığında bilgi geçişini ölçekler; hücrenin toplamsal yolu uzun dönem türevi taşır.",
      "The projection a_t=W_ih x_t+b_ih+W_hh h_(t−1)+b_hh splits into four H-wide blocks. i/f/o use sigmoid, g uses tanh. Sigmoid gates scale information flow in [0,1]; the additive cell path carries long-term derivatives.",
      {
        formula:
          "i_t=\\sigma(a_t^i),\\quad f_t=\\sigma(a_t^f),\\quad g_t=\\tanh(a_t^g),\\quad o_t=\\sigma(a_t^o)",
      },
    ),
    s(
      "cell-gradient",
      "Hücre ve kapı türevleri",
      "Cell and gate derivatives",
      "d c_t, gelecekten gelen hücre türeviyle h_t yolunun katkısının toplamıdır. Hücre gradyanı f_t ile geçmişe taşınır. i, f, o için sigmoid türevi q(1−q), g için 1−g² ile yerel çarpım yapılır; ardından projeksiyonun Linear türevleri hesaplanır.",
      "d c_t combines the future cell derivative with the h_t path. It propagates to the previous cell through f_t. Apply sigmoid derivative q(1−q) to i, f, o and 1−g² to g, then differentiate the Linear projections.",
      {
        formula:
          "dc_t\\mathrel{+}=dh_t\\odot o_t\\odot(1-\\tanh^2c_t),\\quad dc_{t-1}=dc_t\\odot f_t",
      },
    ),
  );
layers
  .find((a) => a.slug === "attention")!
  .sections.push(
    s(
      "attention-gradient",
      "Attention türev zinciri",
      "Attention derivative chain",
      "Dropout kapalıyken C=PV ve P=softmax(S), S=QKᵀ/√d olsun. dP=dC Vᵀ, dV=Pᵀ dC; softmax vektör-Jacobian çarpımından dS bulunur. Sonra dQ=dS K/√d ve dK=dSᵀ Q/√d hesaplanır. MicroTape bu zinciri, Linear projeksiyonlar ise parametre türevlerini uygular.",
      "With dropout disabled, let C=PV, P=softmax(S), S=QKᵀ/√d. dP=dC Vᵀ, dV=Pᵀ dC; obtain dS through the softmax vector-Jacobian product. Then compute dQ=dS K/√d and dK=dSᵀ Q/√d. MicroTape implements this chain; Linear projections compute parameter derivatives.",
      {
        formula:
          "dS=P\\odot(dP-\\operatorname{rowsum}(dP\\odot P)),\\quad dQ=\\frac{dS K}{\\sqrt d},\\quad dK=\\frac{dS^T Q}{\\sqrt d}",
      },
    ),
  );
