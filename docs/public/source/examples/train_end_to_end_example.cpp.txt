#include "nexus_model/adapters/nexus_optim.hpp"
#include "nexus_model/nexus_model.hpp"
#include "nexusdata/core/ndarray.hpp"
#include "nexusdata/dataset/in_memory.hpp"
#include "nexusdata/dataset/split.hpp"
#include "nexusdata/loading/dataloader.hpp"
#include "nexusloss/nexusloss.hpp"
#include "nexus_optim/algorithms/adamw.hpp"

#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

int main() {
  constexpr int samples = 96;
  constexpr int features = 8;
  constexpr int classes = 3;
  constexpr int epochs = 8;

  nexusdata::NDArray x(nexusdata::Shape{samples, features}, nexusdata::DType::Float32);
  nexusdata::NDArray y(nexusdata::Shape{samples}, nexusdata::DType::Float32);
  float* xf = x.data<float>();
  float* yf = y.data<float>();
  for (int n = 0; n < samples; ++n) {
    const int label = n % classes;
    yf[n] = static_cast<float>(label);
    for (int f = 0; f < features; ++f) {
      xf[n * features + f] = (f % classes == label) ? 1.f : 0.05f * static_cast<float>((n + f) % 5);
    }
  }

  nexusdata::InMemoryDataset dataset(x, y);
  auto split = nexusdata::random_split(dataset, {0.8, 0.2}, 7);
  nexusdata::DataLoaderOptions options;
  options.batch_size = 16;
  options.shuffle = true;
  options.seed = 7;
  nexusdata::DataLoader loader(split.first, options);

  auto model = nexus_model::Sequential({
      std::make_shared<nexus_model::Linear>(features, 16),
      std::make_shared<nexus_model::ReLU>(),
      std::make_shared<nexus_model::Linear>(16, classes),
  });

  nexusloss::classification::CrossEntropyLoss<float> loss_fn;
  nexus_optim::ParamGroupOptions group_options;
  group_options.learning_rate = 1e-2;
  nexus_optim::AdamW<float>::Options adam;
  adam.lr = 1e-2;
  adam.weight_decay = 1e-2;
  nexus_optim::AdamW<float> optimizer({nexus_model::as_param_group(model, group_options)}, adam);

  float last = 0.f;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    double total = 0.0;
    int seen = 0;
    for (nexusdata::Batch batch : loader) {
      optimizer.zero_grad();
      const std::size_t batch_n = batch.inputs.shape()[0];
      nexus_model::Tensor inputs = nexus_model::Tensor::zeros({batch_n, static_cast<std::size_t>(features)});
      for (std::size_t i = 0; i < inputs.numel(); ++i) inputs[i] = batch.inputs.data<float>()[i];

      nexus_model::Tensor prediction = model.forward(inputs);
      nexus_model::Tensor grad = nexus_model::Tensor::zeros(prediction.shape_vec());
      double loss = 0.0;
      for (std::size_t row = 0; row < batch_n; ++row) {
        const std::size_t label = static_cast<std::size_t>(batch.labels.data<float>()[row] + 0.5f);
        std::span<const float> logits(prediction.data() + row * classes, classes);
        loss += loss_fn.forward(logits, label);
        const std::vector<float> row_grad = loss_fn.backward();
        for (int c = 0; c < classes; ++c) {
          grad[row * classes + static_cast<std::size_t>(c)] = row_grad[static_cast<std::size_t>(c)] / static_cast<float>(batch_n);
        }
      }
      model.backward(grad);
      optimizer.step();
      total += loss;
      seen += static_cast<int>(batch_n);
    }
    last = static_cast<float>(total / seen);
    std::cout << "epoch " << epoch << " loss " << last << "\n";
  }
  return last < 0.6f ? 0 : 1;
}
