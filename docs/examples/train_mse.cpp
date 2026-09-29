#include <nexus_model/nexus_model.hpp>
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
  std::cout<<"MSE: "<<loss<<"\n";
  return loss<1e-3f ? 0 : 1;
}