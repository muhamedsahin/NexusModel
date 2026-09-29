#include <nexus_model/nexus_model.hpp>
#include <iostream>
#include <memory>
using namespace nexus_model;

int main() {
  init::manual_seed(42);
  Sequential model({
    std::make_shared<Linear>(784, 256),
    std::make_shared<GELU>(),
    std::make_shared<Linear>(256, 10)
  });

  auto x = Tensor::full({32, 784}, 0.1f);
  model.zero_grad();
  auto y = model.forward(x);
  // Example objective: L = sum(y). dL/dy = 1.
  auto dy = Tensor::full({32, 10}, 1.0f);
  auto dx = model.backward(dy);
  std::cout << y.size(0) << " x " << y.size(1) << "\n";
  return dx.same_shape(x) ? 0 : 1;
}