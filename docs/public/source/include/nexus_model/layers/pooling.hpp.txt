#pragma once

/**
 * @file pooling.hpp
 * @brief MaxPool2D, AvgPool2D, GlobalAvgPool2D.
 *
 * MaxPool gradyanı yalnızca ileri geçişte kaydedilen argmax indisine akar.
 * Eşit maksimumlarda ilk konum kazanır. AvgPool, pencere paydasını ileri ile
 * aynı sayar (`count_include_pad` varsayılanı açık, PyTorch ile aynı).
 */

#include "nexus_model/core/kernels.hpp"
#include "nexus_model/core/module.hpp"

#include <vector>

namespace nexus_model {

class MaxPool2D : public Module {
 public:
  MaxPool2D(int kernel, int stride = -1, int padding = 0) : MaxPool2D(kernel, kernel, stride < 0 ? kernel : stride, stride < 0 ? kernel : stride, padding, padding) {}
  MaxPool2D(int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w)
      : kh_(kernel_h), kw_(kernel_w), sh_(stride_h), sw_(stride_w), ph_(pad_h), pw_(pad_w) {}

  Tensor forward(const Tensor& input) override {
    if (input.rank() != 4) throw ModelError("MaxPool2D expects NCHW");
    const int n = static_cast<int>(input.size(0));
    const int c = static_cast<int>(input.size(1));
    const int h = static_cast<int>(input.size(2));
    const int w = static_cast<int>(input.size(3));
    const int oh = kernels::conv_out_size(h, ph_, 1, kh_, sh_);
    const int ow = kernels::conv_out_size(w, pw_, 1, kw_, sw_);
    if (oh <= 0 || ow <= 0) throw ModelError("MaxPool2D output is empty");
    input_.copy_from(input);
    const std::size_t dims[4] = {input.size(0), input.size(1), static_cast<std::size_t>(oh), static_cast<std::size_t>(ow)};
    output_.resize_dims(dims, 4);
    index_.resize(static_cast<std::size_t>(n * c * oh * ow));
    oh_ = oh;
    ow_ = ow;
    in_spatial_ = h * w;
    kernels::max_pool2d(input_.data(), output_.data(), index_.data(), n, c, h, w, kh_, kw_, sh_, sw_, ph_, pw_);
    return output_;
  }

  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(input_);
    grad_input_.zero();
    const int n = static_cast<int>(input_.size(0));
    const int c = static_cast<int>(input_.size(1));
    kernels::max_pool2d_bwd(grad_output.data(), index_.data(), grad_input_.data(), n, c, oh_, ow_, in_spatial_);
    return grad_input_;
  }

 private:
  int kh_, kw_, sh_, sw_, ph_, pw_;
  int oh_ = 0, ow_ = 0, in_spatial_ = 0;
  std::vector<int> index_;
  Tensor input_, output_, grad_input_;
};

class AvgPool2D : public Module {
 public:
  AvgPool2D(int kernel, int stride = -1, int padding = 0, bool count_include_pad = true)
      : kh_(kernel),
        kw_(kernel),
        sh_(stride < 0 ? kernel : stride),
        sw_(stride < 0 ? kernel : stride),
        ph_(padding),
        pw_(padding),
        count_include_pad_(count_include_pad) {}

  Tensor forward(const Tensor& input) override {
    if (input.rank() != 4) throw ModelError("AvgPool2D expects NCHW");
    const int h = static_cast<int>(input.size(2));
    const int w = static_cast<int>(input.size(3));
    const int oh = kernels::conv_out_size(h, ph_, 1, kh_, sh_);
    const int ow = kernels::conv_out_size(w, pw_, 1, kw_, sw_);
    input_.copy_from(input);
    const std::size_t dims[4] = {input.size(0), input.size(1), static_cast<std::size_t>(oh), static_cast<std::size_t>(ow)};
    output_.resize_dims(dims, 4);
    kernels::avg_pool2d(input_.data(), output_.data(), static_cast<int>(input.size(0)), static_cast<int>(input.size(1)), h, w,
                        kh_, kw_, sh_, sw_, ph_, pw_, count_include_pad_);
    return output_;
  }

  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(input_);
    grad_input_.zero();
    kernels::avg_pool2d_bwd(grad_output.data(), grad_input_.data(), static_cast<int>(input_.size(0)),
                            static_cast<int>(input_.size(1)), static_cast<int>(input_.size(2)), static_cast<int>(input_.size(3)),
                            kh_, kw_, sh_, sw_, ph_, pw_, count_include_pad_);
    return grad_input_;
  }

 private:
  int kh_, kw_, sh_, sw_, ph_, pw_;
  bool count_include_pad_;
  Tensor input_, output_, grad_input_;
};

class GlobalAvgPool2D : public Module {
 public:
  Tensor forward(const Tensor& input) override {
    if (input.rank() != 4) throw ModelError("GlobalAvgPool2D expects NCHW");
    input_.copy_from(input);
    const std::size_t dims[2] = {input.size(0), input.size(1)};
    output_.resize_dims(dims, 2);
    const int n = static_cast<int>(input.size(0));
    const int c = static_cast<int>(input.size(1));
    const int spatial = static_cast<int>(input.size(2) * input.size(3));
    const float inv = 1.f / static_cast<float>(spatial);
    for (int ni = 0; ni < n; ++ni) {
      for (int ci = 0; ci < c; ++ci) {
        const float* src = input_.data() + (static_cast<std::size_t>(ni * c + ci) * spatial);
        double sum = 0.0;
        for (int s = 0; s < spatial; ++s) sum += src[s];
        output_[static_cast<std::size_t>(ni * c + ci)] = static_cast<float>(sum * inv);
      }
    }
    return output_;
  }

  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(input_);
    const int n = static_cast<int>(input_.size(0));
    const int c = static_cast<int>(input_.size(1));
    const int spatial = static_cast<int>(input_.size(2) * input_.size(3));
    const float inv = 1.f / static_cast<float>(spatial);
    for (int ni = 0; ni < n; ++ni) {
      for (int ci = 0; ci < c; ++ci) {
        const float g = grad_output[static_cast<std::size_t>(ni * c + ci)] * inv;
        float* dst = grad_input_.data() + (static_cast<std::size_t>(ni * c + ci) * spatial);
        for (int s = 0; s < spatial; ++s) dst[s] = g;
      }
    }
    return grad_input_;
  }

 private:
  Tensor input_, output_, grad_input_;
};

}  // namespace nexus_model
