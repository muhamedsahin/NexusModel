import { Article, s, tx } from "./types";
import { quickCode } from "./quick-code";
export const basics: Article[] = [
  {
    slug: "introduction",
    group: 0,
    title: tx("NexusModel nedir?", "What is NexusModel?"),
    description: tx(
      "Sinir ağının hesaplama katmanı. Her adım sizin kontrolünüzde.",
      "The computation layer of a neural network. Every step under your control.",
    ),
    source: "README.md",
    sections: [
      s(
        "purpose",
        "Bir modelin içindeki motor",
        "The engine inside a model",
        "NexusModel, Nexus ekosisteminin C++20 katman kütüphanesidir. Tensörleri katmanlardan geçirir, girdilere göre türevleri hesaplar ve öğrenilebilir parametrelerin gradyanlarını biriktirir. MLP, evrişimli ağ, tekrarlayan ağ ve Transformer blokları için ortak bir Module arayüzü sunar.",
        "NexusModel is the C++20 layer library in the Nexus ecosystem. It runs tensors through layers, computes input derivatives, and accumulates learnable-parameter gradients. A common Module interface supports MLPs, convolutional networks, recurrent networks, and Transformer blocks.",
      ),
      s(
        "scope",
        "Sorumlulukların ayrılması",
        "Separation of responsibilities",
        "NexusData veri yükler; NexusModel tahmin üretir; NexusLoss kaybı ve dış dL/dy gradyanını hesaplar; NexusOptim parametreleri günceller. MatrixFlash Pro isteğe bağlı GPU matris çarpımı köprüsüdür. Çekirdek NexusModel bu üç kardeş eğitim kütüphanesine bağlanmaz; yalnız uçtan uca örnek onları bağlar.",
        "NexusData loads data; NexusModel predicts; NexusLoss computes loss and external dL/dy; NexusOptim updates parameters. MatrixFlash Pro provides an optional GPU matrix multiplication bridge. The core does not link the three sibling training libraries; only the end-to-end example links them.",
      ),
      s(
        "contract",
        "Temel sözleşme",
        "The central contract",
        "forward(x) çıktıyı ve backward önbelleğini üretir. backward(dy) giriş gradyanını döndürür; parametre gradyanlarına += ile yazar. Yeni adım başında zero_grad() çağırın. Aynı nesnede forward ile ona ait backward arasına başka forward koymayın.",
        "forward(x) produces output and a backward cache. backward(dy) returns input gradients and adds to parameter gradients with +=. Call zero_grad() at the start of a new step. Do not insert another forward on the same object before its corresponding backward.",
        {
          formula:
            "x \\xrightarrow{f_\\theta} y, \\qquad \\nabla_x L = J_f(x)^T \\nabla_y L",
        },
      ),
      s(
        "boundaries",
        "Mevcut kapsam",
        "Current scope",
        "Hesap ve depolama float32, varsayılan yürütme host CPU üzerindedir. Global autograd, otomatik cihaz taşıma, FP16/BF16 eğitim, optimizer veya veri yükleyici bu çekirdeğin özellikleri değildir. Modüller değişebilir çalışma tamponları taşır; eşzamanlı kullanım ayrı replikalar gerektirir.",
        "Computation and storage use float32, with host CPU execution by default. Global autograd, automatic device transfer, FP16/BF16 training, optimizers, and data loaders are outside this core. Modules own mutable buffers; concurrency requires separate replicas.",
      ),
    ],
  },
  {
    slug: "installation",
    group: 0,
    title: tx("Kurulum ve derleme", "Installation & build"),
    description: tx(
      "CMake, derleme seçenekleri ve projenize bağlantı.",
      "CMake, configuration options, and application integration.",
    ),
    source: "CMakeLists.txt",
    sections: [
      s(
        "requirements",
        "Gereksinimler",
        "Requirements",
        "CMake 3.20+ ve C++20 derleyicisi gerekir. SIMD kaynakları x86 AVX yönergeleriyle derlenir; mevcut derleme genel ARM taşınabilirlik garantisi sunmaz. CUDA yalnız cihaz çekirdekleri etkinse gerekir.",
        "Requires CMake 3.20+ and a C++20 compiler. SIMD sources compile with x86 AVX instructions; the build does not guarantee general ARM portability. CUDA is needed only for device kernels.",
      ),
      s(
        "build",
        "Kaynak kökünden derleyin",
        "Build from the repository root",
        "Bu komutları NexusModel klasöründe çalıştırın. Testler ve CPU benchmark varsayılan açık. Çoklu yapılandırmalı derleyicilerde yürütülebilir dosyalar Release altında oluşur.",
        "Run inside NexusModel. Tests and CPU benchmarks are enabled by default. Multi-configuration generators place executables under Release.",
        {
          code: "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\ncmake --build build --config Release\nctest --test-dir build -C Release --output-on-failure\n# Windows / MSVC\n.\\build\\Release\\nexus_model_bench.exe",
        },
      ),
      s(
        "options",
        "Derleme seçenekleri",
        "Build options",
        "CUDA mimarisi varsayılan 86; kendi GPU mimarinizi seçin. MatrixFlash seçeneği köprü başlığını açar; tüketen hedef ayrıca MatrixFlash bağımlılığını karşılamalıdır.",
        "CUDA architecture defaults to 86; select your GPU architecture. The MatrixFlash option enables the bridge header; the consumer must separately supply MatrixFlash dependencies.",
        {
          table: {
            headers: [tx("Seçenek", "Option"), tx("Varsayılan", "Default")],
            rows: [
              ["NEXUS_MODEL_WITH_CUDA", "OFF"],
              ["NEXUS_MODEL_WITH_MATRIXFLASH", "OFF"],
              ["NEXUS_MODEL_BUILD_TESTS", "ON"],
              ["NEXUS_MODEL_BUILD_BENCHMARKS", "ON"],
              ["NEXUS_MODEL_BUILD_EXAMPLES", "OFF"],
              ["NEXUS_MODEL_ENABLE_AVX512", "ON"],
              ["NEXUS_MODEL_WARNINGS_AS_ERRORS", "ON"],
            ],
          },
        },
      ),
      s(
        "integrate",
        "Uygulamaya bağlayın",
        "Link your application",
        "CMake kurulum kuralları arşiv ve başlıkları kopyalar; bu sürüm find_package yapılandırması dışa aktarmaz. Alias hedefini add_subdirectory ile kullanın.",
        "Installation rules copy the archive and headers; this version does not export a find_package configuration. Use the alias target with add_subdirectory.",
        {
          code: "cmake_minimum_required(VERSION 3.20)\nproject(example LANGUAGES CXX)\nadd_subdirectory(NexusModel)\nadd_executable(example main.cpp)\ntarget_link_libraries(example PRIVATE nexus_model::nexus_model)",
        },
      ),
      s(
        "cuda",
        "CUDA derlemesi",
        "CUDA build",
        "CUDA Toolkit gerekir. GPU benchmark ayrıca cuBLAS bağlar. Bu yapılandırma host katmanlarını otomatik GPU’ya taşımaz.",
        "Requires the CUDA Toolkit. The GPU benchmark also links cuBLAS. This configuration does not automatically move host layers to GPU.",
        {
          code: "cmake -S . -B build-cuda -DNEXUS_MODEL_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86\ncmake --build build-cuda --config Release",
        },
      ),
    ],
  },
  {
    slug: "quickstart",
    group: 0,
    title: tx("İlk modeliniz", "Your first model"),
    description: tx(
      "Çalıştırılabilir bir MLP ile açık geri yayılım.",
      "Explicit backward with an executable MLP.",
    ),
    source: "include/nexus_model/containers/sequential.hpp",
    sections: [
      s(
        "example",
        "Üç adımda sinir ağı",
        "A network in three steps",
        "Bu tam C++ örneği [32,784] girdisini [32,10] çıktısına taşır. GELU aktivasyon, Linear öğrenilebilir katmandır. Mekanizmayı göstermek için L=sum(y) kullanılır; sınıflandırmada gerçek kayıp gradyanını verin.",
        "This complete C++ example maps [32,784] to [32,10]. GELU is an activation; Linear is a learnable layer. It uses L=sum(y) to demonstrate the mechanism; supply your actual loss gradient for classification.",
        { code: quickCode },
      ),
      s(
        "shapes",
        "Şekilleri izleyin",
        "Follow the shapes",
        "İlk Linear son ekseni 784’ten 256’ya dönüştürür; GELU korur; ikinci Linear 10 logit üretir. backward girdisi çıktı şekline tam uymalıdır. dx, x ile aynı şekildedir. Parametre gradyanları model.parameters() altındadır.",
        "The first Linear maps the last axis from 784 to 256; GELU preserves it; the second Linear produces 10 logits. backward input must exactly match output shape. dx has the same shape as x. Parameter gradients are in model.parameters().",
      ),
      s(
        "parameters",
        "203530 öğrenilebilir değer",
        "203530 learnable values",
        "Sequential sayısal isimler kullanır: 0.weight [256,784], 0.bias [256], 2.weight [10,256], 2.bias [10]. GELU parametresizdir. Örnekte optimizer adımı yoktur; eğitim döngüsü bölümünde ağırlık güncellemesini ekleyin.",
        "Sequential uses numeric names: 0.weight [256,784], 0.bias [256], 2.weight [10,256], 2.bias [10]. GELU has no parameters. There is no optimizer step in this example; add weight updates using the training guide.",
        {
          code: 'for (const auto& [name, p] : model.named_parameters()) {\n  std::cout << name << ": " << p->data.numel() << "\\n";\n}',
        },
      ),
    ],
  },
  {
    slug: "training",
    group: 0,
    title: tx("Eğitim döngüsü", "Training loop"),
    description: tx(
      "Kayıp, gradyan biriktirme ve NexusOptim.",
      "Loss, gradient accumulation, and NexusOptim.",
    ),
    source: "examples/train_end_to_end_example.cpp",
    sections: [
      s(
        "cycle",
        "Bir adımın sırası",
        "Order of a step",
        "zero_grad → forward → loss.backward → model.backward → optimizer.step. Gradyanlar backward çağrıları boyunca toplanır. Mikro-batch biriktirmede bu istenir; bağımsız adımlar arasında temizlenmezse yanlış güncelleme olur.",
        "zero_grad → forward → loss.backward → model.backward → optimizer.step. Gradients accumulate across backward calls. This enables micro-batch accumulation, but requires clearing between independent steps.",
        { formula: "\\theta_{t+1}=\\theta_t-\\eta\\nabla_\\theta L" },
      ),
      s(
        "adapter",
        "Ödünç parametre köprüsü",
        "Borrowed parameter bridge",
        "as_param_group yalnız requires_grad=true parametrelerin veri ve gradyan işaretçilerini aktarır. Optimizer yaşarken parametre depolamasını değiştirmeyin veya modeli yok etmeyin. Adaptör çekirdekten ayrı include edilir.",
        "as_param_group passes data and gradient pointers only for requires_grad=true parameters. Do not replace parameter storage or destroy the model while the optimizer is alive. The adapter is included separately from the core.",
        {
          code: "#include <nexus_model/adapters/nexus_optim.hpp>\n#include <nexus_optim/algorithms/adamw.hpp>\nnexus_optim::ParamGroupOptions group;\ngroup.learning_rate = 1e-2;\nnexus_optim::AdamW<float>::Options options;\noptions.lr = 1e-2;\noptions.weight_decay = 1e-2;\nnexus_optim::AdamW<float> optimizer(\n  {nexus_model::as_param_group(model, group)}, options);",
        },
      ),
      s(
        "loss",
        "Logitlerden gradyana",
        "From logits to gradients",
        "Örnekte CrossEntropyLoss<float> her satır için çağrılır. backward sonucu bir sonraki satırdan önce kopyalanır ve batch boyutuna bölünür. CrossEntropy’ye logit verin; önceden Softmax eklemeyin.",
        "The example calls CrossEntropyLoss<float> per row. Copy backward output before the next row and divide by batch size. Pass logits to CrossEntropy; do not prepend Softmax.",
        {
          formula:
            "L=-\\frac1B\\sum_b\\log p_{b,y_b},\\qquad \\frac{\\partial L}{\\partial z_{b,c}}=\\frac{p_{b,c}-\\mathbf1[c=y_b]}B",
        },
      ),
      s(
        "run",
        "Uçtan uca örnek",
        "End-to-end example",
        "NexusData, NexusLoss ve NexusOptim aynı üst klasörde bulunmalı. Örnek 96 sentetik örnek, 8 özellik, 3 sınıf ve 8 epoch AdamW kullanır. Bu bir entegrasyon kontrolüdür.",
        "NexusData, NexusLoss, and NexusOptim must be sibling directories. The example uses 96 synthetic samples, 8 features, 3 classes, and 8 AdamW epochs. This is an integration check.",
        {
          code: "cmake -S . -B build-e2e -DNEXUS_MODEL_BUILD_EXAMPLES=ON\ncmake --build build-e2e --config Release\n.\\build-e2e\\Release\\train_end_to_end.exe",
        },
      ),
    ],
  },
  {
    slug: "tensor",
    group: 1,
    title: tx("Tensor ve bellek", "Tensor & memory"),
    description: tx(
      "64 bayt hizalama, paylaşımlı depolama ve yaşam süresi.",
      "64-byte alignment, shared storage, and lifetimes.",
    ),
    source: "include/nexus_model/core/tensor.hpp",
    sections: [
      s(
        "layout",
        "Düzen ve oluşturma",
        "Layout and construction",
        "Yoğun, bitişik, satır-major float32; en fazla 8 eksen. Tensor(shape) belleği sıfırlamaz. zeros, full ve from_values başlatılmış veri üretir. Varsayılan boş Tensor rank=0, numel=0 taşır. AlignedStorage veri adresi 64 bayta hizalıdır.",
        "Dense contiguous row-major float32, with up to 8 axes. Tensor(shape) does not zero memory. zeros, full, and from_values create initialized data. A default empty Tensor has rank=0 and numel=0. AlignedStorage aligns data to 64 bytes.",
        {
          code: "auto a = Tensor::zeros({2, 3});\nauto b = Tensor::full({2, 3}, 0.5f);\nauto c = Tensor::from_values({2, 2}, {1, 2, 3, 4});",
        },
      ),
      s(
        "ownership",
        "Kopya veya görünüm",
        "Copy or view",
        "Atama yalnız başlığı kopyalar ve veriyi paylaşır. clone bağımsız kopyadır. reshape/view_as aynı depoya farklı şekil verir. [] veya data() ile yazma diğer görünümü de değiştirir; genel copy-on-write yoktur.",
        "Assignment copies the header and shares data. clone makes an independent copy. reshape/view_as give the storage a different shape. Writing through [] or data() changes other views too; there is no general copy-on-write.",
        {
          code: "auto shared = c;\nauto independent = c.clone();\nauto flat = c.reshape({4});\nshared[0] = 9; // c[0] and flat[0] also become 9",
        },
      ),
      s(
        "reshape",
        "Boyut çıkarımı",
        "Dimension inference",
        "Tensor::reshape tek size_t(-1) ile boyut çıkarır. Eleman sayısı korunur. Literal -1 daraltma hatası verebilir; açık dönüşüm kullanın. Reshape modülü bu çıkarımı yapmaz, somut şekil ister.",
        "Tensor::reshape infers one size_t(-1) dimension while preserving element count. Literal -1 can cause narrowing; use an explicit cast. The Reshape module does not perform this inference and needs a concrete shape.",
        {
          code: "auto t = Tensor::zeros({2, 3, 4});\nauto rows = t.reshape({2, static_cast<std::size_t>(-1)}); // [2,12]",
        },
      ),
      s(
        "borrow",
        "Ödünç bellek",
        "Borrowed memory",
        "borrow(float*,shape) sahip değildir; ömür çağırana aittir. resize/copy_from ödünç belleği koruma garantisi vermez. operator[] sınır kontrolsüz; size(axis) geçersiz eksende ModelError üretir.",
        "borrow(float*,shape) is nonowning; lifetime belongs to the caller. resize/copy_from need not preserve borrowed memory. operator[] is unchecked; size(axis) throws ModelError for invalid axes.",
      ),
      s(
        "reuse",
        "Tampon yeniden kullanımı",
        "Buffer reuse",
        "resize_dims yalnız tek sahiplik, yeterli kapasite ve sıfır offset ile depoyu kullanır. Dönen başlığı sonraki çağrıya kadar tutmak yeni ayırma doğurabilir. allocation_count sadece AlignedStorage sayar; bütün heap ayırmalarını ölçmez.",
        "resize_dims reuses storage with sole ownership, enough capacity, and zero offset. Keeping a returned header alive across the next call may force allocation. allocation_count tracks AlignedStorage only, not all heap allocations.",
      ),
    ],
  },
  {
    slug: "module",
    group: 1,
    title: tx("Module ve Parameter", "Module & Parameter"),
    description: tx(
      "Modül ağacı, gradyanlar ve modlar.",
      "Module trees, gradients, and modes.",
    ),
    source: "include/nexus_model/core/module.hpp",
    sections: [
      s(
        "interface",
        "Ortak arayüz",
        "Common interface",
        "forward(const Tensor&) ve backward(const Tensor&) saf sanaldır. Yeni modül eğitim modundadır. train/eval kayıtlı çocuklara yayılır. eval gradyanı kapatan no-grad modu değildir.",
        "forward(const Tensor&) and backward(const Tensor&) are pure virtual. New modules start in training mode. train/eval propagate to registered children. eval is not a no-grad mode.",
      ),
      s(
        "parameter",
        "Öğrenilebilir değer",
        "Learnable values",
        "Parameter data, grad, requires_grad içerir. grad sıfır başlar. requires_grad=false ilgili parametre türevini durdurur; giriş türevi hesaplanır. accumulate_grad_locked replikaları adım dışında toplar; yalnız eleman sayısını kontrol eder.",
        "Parameter holds data, grad, and requires_grad. grad starts at zero. requires_grad=false stops that parameter derivative; input gradients still compute. accumulate_grad_locked merges replicas outside the step and checks element count only.",
      ),
      s(
        "tree",
        "Kayıt ve dolaşım",
        "Registration and traversal",
        "parameters ham işaretçi listesi; named_parameters hiyerarşik isimler verir. register_parameter ağırlık, register_buffer öğrenilmeyen durum, register_module çocuk kaydeder. Aynı parametre birden çok yoldan kaydedilirse tekilleştirme yapılmaz.",
        "parameters returns raw pointers; named_parameters adds hierarchical names. register_parameter records weights, register_buffer nonlearnable state, register_module children. Repeated registration of the same parameter is not deduplicated.",
      ),
      s(
        "custom",
        "Kendi katmanınız",
        "Your own layer",
        "Parametre türevlerini += ile biriktirin. Çalışma tamponlarını nesnede tutun; son forward önbelleğinin yaşam süresine dikkat edin.",
        "Accumulate parameter derivatives with +=. Keep working buffers in the object and respect the latest-forward cache lifetime.",
        {
          code: "class Scale : public Module {\n public:\n  Tensor forward(const Tensor& x) override {\n    y_.resize_like(x);\n    for (std::size_t i=0;i<x.numel();++i) y_[i]=2*x[i];\n    return y_;\n  }\n  Tensor backward(const Tensor& dy) override {\n    dx_.resize_like(dy);\n    for (std::size_t i=0;i<dy.numel();++i) dx_[i]=2*dy[i];\n    return dx_;\n  }\n private: Tensor y_, dx_;\n};",
        },
      ),
    ],
  },
  {
    slug: "autograd",
    group: 1,
    title: tx("Açık geri yayılım", "Explicit backward"),
    description: tx(
      "Analitik türev ve katman-lokal teyp kararı.",
      "The analytic derivative and layer-local tape decision.",
    ),
    source: "include/nexus_model/core/details.txt",
    sections: [
      s(
        "why",
        "Yol B",
        "Path B",
        "MatrixFlash Variable::backward kökü birlerle başlatır, dış dL/dy kabul etmez. Kardeş kütüphaneler aynı teybi paylaşmaz. NexusModel açık gradyan girişli backward kullanır.",
        "MatrixFlash Variable::backward seeds the root with ones and accepts no external dL/dy. Sibling libraries do not share a tape. NexusModel uses an explicit-gradient backward interface.",
      ),
      s(
        "vjp",
        "Jacobian saklamadan türev",
        "Differentiate without storing a Jacobian",
        "Linear, Conv, Pool, Norm, Dropout, Embedding, aktivasyonlar ve RNN/LSTM/GRU analitik vektör-Jacobian çarpımı uygular. Sequential ters sırada dolaşır. Attention softmax/matmul zinciri MicroTape kullanır.",
        "Linear, Conv, Pool, Norm, Dropout, Embedding, activations, and RNN/LSTM/GRU implement analytic vector-Jacobian products. Sequential traverses in reverse. The attention softmax/matmul chain uses MicroTape.",
        {
          formula:
            "\\frac{\\partial L}{\\partial x_i}=\\sum_j\\frac{\\partial y_j}{\\partial x_i}\\frac{\\partial L}{\\partial y_j}",
        },
      ),
      s(
        "cache",
        "Önbellek ve biriktirme",
        "Caching and accumulation",
        "Parametreler birden fazla yoldan katkı alır; türevleri toplanır. Giriş türevi o backward çağrısına aittir. Aynı katman nesnesini grafikte tekrar forward etmek önbelleği ezer. Her kullanıma ayrı modül veya ayrı replika ayırın.",
        "Parameters receive contributions from multiple paths, so derivatives add. Input derivatives belong to that backward call. Reusing a layer object for another forward in a graph overwrites its cache. Use separate modules or replicas.",
      ),
    ],
  },
  {
    slug: "micro-tape",
    group: 1,
    title: tx("MicroTape", "MicroTape"),
    description: tx(
      "Attention için küçük ters-mod grafiği.",
      "A small reverse-mode graph for attention.",
    ),
    source: "include/nexus_model/core/micro_tape.hpp",
    sections: [
      s(
        "lifetime",
        "Yaşam süresi",
        "Lifetime",
        "reset kullanılan slot sayısını ve işlem listesini temizler, kapasiteyi korur. load tensörü slotuna kopyalar ve int kimlik döndürür. Sonraki reset eski grafik kimliklerini geçersiz kılar. Bu global autograd değildir.",
        "reset clears the used-slot count and operation list while keeping capacity. load copies a tensor into a slot and returns an int ID. The next reset invalidates old graph IDs. This is not global autograd.",
      ),
      s(
        "ops",
        "İşlemler",
        "Operations",
        "matmul ve matmul_bt rank-3 batch matrisleri kullanır. add/mul aynı şekilli eleman bazlı işlemler; add_matrix tekrar edilen sabit maske ekler ve maskeyi türetmez. Genel broadcasting varsaymayın.",
        "matmul and matmul_bt use rank-3 batched matrices. add/mul are elementwise same-shape operations; add_matrix adds a repeated constant mask without differentiating it. Do not assume general broadcasting.",
        {
          table: {
            headers: [tx("İşlem", "Operation"), tx("Kural", "Rule")],
            rows: [
              ["matmul", "[B,M,K] × [B,K,N] → [B,M,N]"],
              ["matmul_bt", "[B,M,K] × [B,N,K]ᵀ → [B,M,N]"],
              ["scale", "factor · a"],
              ["add / mul", "a+b / a⊙b"],
              ["add_matrix", "[B,M,N] + [M,N]"],
              ["softmax", "last axis"],
            ],
          },
        },
      ),
      s(
        "backward",
        "Dış gradyan",
        "External gradient",
        "backward(root,grad) tüm kullanılan slot türevlerini sıfırlar, köke verilen gradyanı kopyalar, işlemleri tersten uygular. value(id) ve grad(id) sonuçlara erişir.",
        "backward(root,grad) zeros all used-slot derivatives, copies the seed gradient to the root, and reverses the operations. value(id) and grad(id) access results.",
        {
          code: "MicroTape tape;\nauto q = Tensor::full({1,2,4}, 0.1f);\nauto k = Tensor::full({1,3,4}, 0.2f);\nint iq=tape.load(q), ik=tape.load(k);\nint p=tape.softmax(tape.scale(tape.matmul_bt(iq,ik),0.5f));\ntape.backward(p, Tensor::full({1,2,3},1.f));",
        },
      ),
    ],
  },
  {
    slug: "containers",
    group: 1,
    title: tx("Model konteynerleri", "Model containers"),
    description: tx(
      "Sequential, koleksiyonlar ve artık bağlantılar.",
      "Sequential, collections, and residual connections.",
    ),
    source: "include/nexus_model/containers/sequential.hpp",
    sections: [
      s(
        "sequential",
        "Sequential",
        "Sequential",
        "Sequential(vector<shared_ptr<Module>>) ileri sırada forward, ters sırada backward uygular. add katman ekler. Ara aktivasyonları ayrıca saklamaz. Boş Sequential girdiyi geçirir.",
        "Sequential(vector<shared_ptr<Module>>) runs forward in order and backward in reverse. add appends a layer. Intermediate activations are not cached separately. An empty Sequential passes input through.",
      ),
      s(
        "collections",
        "ModuleList / ModuleDict",
        "ModuleList / ModuleDict",
        "Kayıtlı koleksiyonlardır; kendi forward/backward metotları ModelError atar. ModuleList indeks veya at(index); ModuleDict isimle erişim verir. Akışı kullanıcı kurar; parametre dolaşımı ve train/eval yine çalışır.",
        "Registered collections whose own forward/backward throw ModelError. ModuleList provides indexing or at(index); ModuleDict provides name lookup. You implement execution flow; parameter traversal and train/eval still work.",
        {
          code: 'ModuleDict heads;\nheads.add("classify", std::make_shared<Linear>(64,10));\nauto x=Tensor::zeros({4,64});\nauto y=heads["classify"]->forward(x);',
        },
      ),
      s(
        "residual",
        "ResidualBlock",
        "ResidualBlock",
        "ResidualBlock(shared_ptr<Module>) y=F(x)+x hesaplar. Şekiller eşleşmelidir, otomatik projeksiyon yoktur. Kimlik dalı türevi doğrudan gradyana eklenir.",
        "ResidualBlock(shared_ptr<Module>) computes y=F(x)+x. Shapes must match; there is no automatic projection. The identity-path derivative adds directly to the gradient.",
        { formula: "y=F(x)+x,\\qquad \\nabla_xL=J_F(x)^Tg+g" },
      ),
      s(
        "reshape",
        "Flatten / Reshape",
        "Flatten / Reshape",
        "Flatten rank≥2 girdiyi [N,kalan] şekline getirir ve mevcut uygulamada memcpy yapar. Başlık yorumuna rağmen sıfır kopya değildir. Reshape({somut boyutlar}) ileri geçişte görünüm, geri geçişte eski şekle kopya üretir.",
        "Flatten maps rank≥2 input to [N,remaining] and uses memcpy in the implementation, despite a zero-copy header comment. Reshape({concrete dimensions}) returns a view in forward and copies into the old shape in backward.",
      ),
    ],
  },
];
