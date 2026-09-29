import { Article, s, tx } from "./types";
export const trainingCode = `#include <nexus_model/nexus_model.hpp>
#include <iostream>
using namespace nexus_model;
int main() {
  init::manual_seed(42);
  Linear model(1,1);
  auto x=Tensor::from_values({5,1},{-2,-1,0,1,2});
  auto target=Tensor::from_values({5,1},{-3,-1,1,3,5});
  auto dy=Tensor::zeros({5,1});
  float loss=0;
  for(int step=0;step<200;++step) {
    model.zero_grad();
    auto y=model.forward(x);
    loss=0;
    for(std::size_t i=0;i<y.numel();++i) {
      float e=y[i]-target[i];
      loss+=e*e/5.f;
      dy[i]=2.f*e/5.f;
    }
    model.backward(dy);
    // Educational SGD; use NexusOptim for production optimizers.
    for(auto* p:model.parameters())
      for(std::size_t i=0;i<p->data.numel();++i)
        p->data[i]-=0.03f*p->grad[i];
  }
  std::cout<<"MSE: "<<loss<<"\\n";
  return loss<1e-3f ? 0 : 1;
}`;
export const reference: Article[] = [
  {
    slug: "worked-example",
    group: 0,
    title: tx("Matematikten eğitime", "From mathematics to training"),
    description: tx(
      "Sayısal bir türev hesabı ve y=2x+1 öğrenen tam program.",
      "A numerical derivative walkthrough and a complete program learning y=2x+1.",
    ),
    source: "include/nexus_model/layers/linear.hpp",
    sections: [
      s(
        "forward",
        "Bir örneği elle hesaplayın",
        "Compute one example by hand",
        "x=[1,2], W=[3,4], b=0 için tek çıkışlı Linear y=11 verir. Hedef t=10 ve L=(y−t)²/2 olsun; kayıp 0.5, dış gradyan dy=1 olur. backward’a 1 vermek bu belirli kaybın doğru türevidir.",
        "With x=[1,2], W=[3,4], b=0, a single-output Linear gives y=11. Let target t=10 and L=(y−t)²/2; loss is 0.5 and external dy=1. Passing 1 to backward is correct for this particular loss.",
        {
          formula:
            "y=1\\cdot3+2\\cdot4=11,\\quad L=\\tfrac12(11-10)^2=0.5,\\quad dy=y-t=1",
        },
      ),
      s(
        "derivative",
        "Gradyanın anlamı",
        "What the gradient means",
        "dW=[1,2], db=1 ve dx=[3,4]. η=0.1 ile SGD ağırlıkları [2.9,3.8], bias −0.1 yapar. Aynı girdiyle yeni y=10.4 ve yeni kayıp 0.08 olur. Türev, parametrede küçük bir değişimin kaybı hangi yönde değiştireceğini söyler.",
        "dW=[1,2], db=1, dx=[3,4]. SGD with η=0.1 changes weights to [2.9,3.8] and bias to −0.1. The same input now gives y=10.4 and loss 0.08. A derivative describes the direction in which a small parameter change affects loss.",
        { formula: "W\\leftarrow W-0.1dW,\\qquad b\\leftarrow b-0.1db" },
      ),
      s(
        "mse",
        "Batch ortalaması",
        "Batch averaging",
        "Aşağıdaki program MSE=(1/B)Σ(y−t)² kullanır; önceki yarım-kare örneğinden farklı olarak 2/B çarpanı gerekir. Batch ortalamasını yalnız bir kere uygulayın. Her adımın başında zero_grad birikimi temizler.",
        "The program below uses MSE=(1/B)Σ(y−t)², so it needs a 2/B factor unlike the half-square example above. Apply batch averaging exactly once. zero_grad clears accumulation at the start of each step.",
        { formula: "\\frac{\\partial L}{\\partial y_i}=\\frac{2(y_i-t_i)}B" },
      ),
      s(
        "program",
        "Tam çalışan eğitim örneği",
        "Complete training example",
        "Bu örnek yalnız NexusModel kullanır; MSE ve sade SGD eğitim amacıyla açık yazılmıştır. Genel optimizer özellikleri için NexusOptim entegrasyonunu kullanın. Program 200 adım sonra kayıp 1e−3 altında ise başarı koduyla çıkar.",
        "This uses NexusModel alone, with MSE and simple SGD written explicitly for teaching. Use NexusOptim integration for general optimizer features. The program exits successfully when loss is below 1e−3 after 200 steps.",
        { code: trainingCode },
      ),
    ],
  },
  {
    slug: "api-reference",
    group: 3,
    title: tx("Çekirdek API kataloğu", "Core API catalog"),
    description: tx(
      "Tensor, Module, Parameter ve durum işlemlerinin hızlı referansı.",
      "Quick reference for Tensor, Module, Parameter, and state operations.",
    ),
    source: "include/nexus_model/nexus_model.hpp",
    sections: [
      s(
        "include",
        "Include ve hata tipi",
        "Includes and error type",
        "nexus_model/nexus_model.hpp katman, çekirdek, konteyner ve serileştirme API’lerini birleştirir. NexusOptim adaptörü, MatrixFlash köprüsü ve CUDA başlıkları ayrı include edilir. ModelError, std::runtime_error’dan türeyen metin mesajlı hata tipidir; bütün geçersiz girdilerin otomatik denetlendiğini varsaymayın.",
        "nexus_model/nexus_model.hpp aggregates layers, core, containers, and serialization. NexusOptim adapter, MatrixFlash bridge, and CUDA headers are separate includes. ModelError derives from std::runtime_error and carries a message; do not assume every invalid input is automatically checked.",
      ),
      s(
        "tensor-api",
        "Tensor metotları",
        "Tensor methods",
        "Aşağıdaki metotlar yoğun float32 depolama içindir. const overload’lar salt okunur erişim sunar. Paylaşımlı Tensor başlığı veriyi kopyalamaz; bunun için clone çağırın.",
        "The following methods operate on dense float32 storage. Const overloads provide read-only access. Shared Tensor headers do not copy data; use clone for that.",
        {
          table: {
            headers: [tx("Metot", "Method"), tx("Sözleşme", "Contract")],
            rows: [
              ["Tensor()", "empty: rank=0, numel=0"],
              ["Tensor(shape)", "allocate uninitialized storage"],
              ["zeros(shape)", "allocate + zero"],
              ["full(shape,value)", "allocate + fill"],
              ["from_values(shape,values)", "value count must match numel"],
              ["borrow(float*,shape)", "nonowning external memory"],
              ["rank(), numel(), empty()", "shape metadata"],
              [
                "dims(), shape_vec(), size(axis)",
                "dimensions / copied vector / checked axis",
              ],
              ["same_shape(other)", "rank and dimensions equality"],
              ["data(), operator[](i)", "raw pointer / unchecked element"],
              ["zero(), fill(value)", "in-place writes"],
              ["copy_from(src)", "resize + copy bytes"],
              ["clone()", "independent storage copy"],
              ["reshape(shape)", "shared view; one size_t(-1) allowed"],
              ["view_as(dims,rank)", "shared view; concrete shape"],
              [
                "resize(shape), resize_like(src)",
                "reuse when uniquely owned and capacity sufficient",
              ],
              ["resize_dims(dims,rank)", "low-level resize"],
              [
                "allocation_count()",
                "atomic AlignedStorage allocation counter",
              ],
            ],
          },
        },
      ),
      s(
        "module-api",
        "Module ve Parameter",
        "Module and Parameter",
        "Kayıt fonksiyonları protected; kendi Module alt sınıfınız içinden çağrılır. state_dict yalnız data ve buffer snapshot’ı üretir. API işaretçilerinin yaşam süresi bağlı oldukları modelin ömrüyle sınırlıdır.",
        "Registration methods are protected and called from your Module subclass. state_dict snapshots only data and buffers. API pointers are valid only within their owning model lifetime.",
        {
          table: {
            headers: [tx("API", "API"), tx("Anlam", "Meaning")],
            rows: [
              ["Module::forward(input)", "output + cache"],
              [
                "Module::backward(grad_output)",
                "input gradient; accumulates parameter gradients",
              ],
              ["parameters()", "vector<Parameter*>"],
              [
                'named_parameters(prefix="")',
                "vector<pair<string,Parameter*>>",
              ],
              ["zero_grad()", "clear all registered parameter gradients"],
              [
                "train(), eval(), is_training()",
                "recursive mode / current mode",
              ],
              ["state_dict()", "independent clones"],
              [
                "load_state_dict(state,strict=true)",
                "validate keys and shapes; copy values",
              ],
              ["register_parameter(name,ptr)", "register learnable state"],
              ["register_buffer(name,ptr)", "register nonlearnable state"],
              ["register_module(name,ptr)", "register child module"],
              [
                "Parameter(values,need_grad=true)",
                "data + zero grad + requires_grad",
              ],
              [
                "Parameter::accumulate_grad_locked(extra)",
                "locked replica merge; size checked",
              ],
              ["StateDict::items", "map<string,Tensor>"],
              ["save_state_dict(state,path)", "write NXM1"],
              ["load_state_dict_file(path)", "read NXM1"],
            ],
          },
        },
      ),
    ],
  },
  {
    slug: "kernel-reference",
    group: 3,
    title: tx("Sayısal çekirdekler", "Numerical kernels"),
    description: tx(
      "Ham işaretçi API’si, dispatch tablosu ve yapısal işlemler.",
      "Raw-pointer APIs, the dispatch table, and structural operations.",
    ),
    source: "include/nexus_model/core/kernels.hpp",
    sections: [
      s(
        "contract",
        "Düşük seviyeli sözleşme",
        "Low-level contract",
        "nexus_model::kernels fonksiyonları satır-major float32 işaretçilerle çalışır; Tensor sahipliği veya ömür yönetmez. Çıktı, çalışma alanı ve gradyan tamponlarını doğru boyutta çağıran ayırır. Boyutlar çoğunlukla int veya size_t’dır; taşan büyük boyutları çağıran önlemelidir.",
        "nexus_model::kernels operates on row-major float32 pointers without Tensor ownership or lifetime management. Callers allocate correctly sized output, workspace, and gradient buffers. Dimensions mostly use int or size_t; callers must prevent oversized dimension overflows.",
      ),
      s(
        "dispatch",
        "KernelTable",
        "KernelTable",
        "cpu() sabit referansla seçilmiş fonksiyon tablosunu verir; sarmalayıcılar tabloya yönlendirir. set_force_scalar tabloyu yeniden kurar. Bu ham API uygulama detayına yakındır; genel kullanımda Module tercih edin.",
        "cpu() returns a const reference to the selected function table; wrappers dispatch through it. set_force_scalar rebuilds the table. This raw API is close to implementation details; prefer Module for general usage.",
        {
          table: {
            headers: [tx("Aile", "Family"), tx("İşlemler", "Operations")],
            rows: [
              ["Vector", "fill, add, mul, scale, axpy, dot"],
              [
                "Activation",
                "relu, leaky_relu, sigmoid, tanh_fwd, gelu, gelu_tanh, silu, elu, mish",
              ],
              ["Activation backward", "corresponding *_bwd functions"],
              [
                "Row reductions",
                "softmax, log_softmax, softmax_bwd, log_softmax_bwd",
              ],
              ["Dense", "linear_forward, linear_backward"],
              ["Normalization", "layernorm, layernorm_bwd"],
            ],
          },
        },
      ),
      s(
        "gradients",
        "Biriktirme ve önbellekler",
        "Accumulation and caches",
        "linear_backward dX, dW ve db içine ekler; kendisi sıfırlamaz. LayerNorm geri geçişi forward’ın mean ve reciprocal-std tamponlarını kullanır. Sigmoid/Tanh türevi çıkış y üzerinden, ReLU türevi giriş x üzerinden alınır; yanlış önbellek aynı şekille bile yanlış sonuç üretir.",
        "linear_backward adds into dX, dW, and db without clearing. LayerNorm backward uses forward mean and reciprocal-std buffers. Sigmoid/Tanh derivatives use output y; ReLU uses input x. Passing the wrong cache gives wrong results even when shapes match.",
      ),
      s(
        "structural",
        "Yapısal çekirdekler",
        "Structural kernels",
        "ConvDesc n/c/k, üç elemanlı spatial_in/out, kernel, stride, pad, dilation dizileri, groups=1 ve rank=2 içerir. conv_col_size gerekli im2col çalışma alanını belirler. Conv içinde GEMM SIMD kullanabilir; yapısal döngüler farklı çekirdeklerdir.",
        "ConvDesc contains n/c/k, three-element spatial_in/out, kernel, stride, pad, dilation arrays, groups=1, and rank=2. conv_col_size determines im2col workspace size. Conv may use SIMD GEMM internally; structural loops are separate kernels.",
        {
          table: {
            headers: [tx("Aile", "Family"), tx("İşlemler", "Operations")],
            rows: [
              [
                "Normalization",
                "batch_norm / batch_norm_bwd / group_norm / group_norm_bwd",
              ],
              [
                "Convolution",
                "conv_out_size / conv_col_size / conv_forward / conv_backward",
              ],
              ["Max pooling", "max_pool2d / max_pool2d_bwd; saved int indices"],
              [
                "Average pooling",
                "avg_pool2d / avg_pool2d_bwd; count_include_pad",
              ],
              [
                "Embedding",
                "embedding_forward / embedding_backward; float indices",
              ],
              ["Dropout", "dropout_mask(mask,n,p,seed)"],
            ],
          },
        },
      ),
    ],
  },
];
