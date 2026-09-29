#pragma once

/**
 * @file conv.hpp
 * @brief Conv1D / Conv2D / Conv3D, im2col + MatrixFlash ile aynı GEMM sözleşmesi.
 *
 * Çapraz korelasyon (PyTorch conv):
 *   Y[n, k, p] = bias[k] + Σ_{c,u} X[n, c, p·s + u·d - pad] W[k, c, u]
 * Geri yayılım im2col'u yeniden kurar; dW birikir, dX col2im ile saçılır.
 * Workspace katmanda bir kez ayrılır (col ve dcol). Groups ve dilation desteklenir.
 *
 * DepthwiseConv2D, groups = in_channels olan Conv2D'dir.
 */

#include "nexus_model/core/initializer.hpp"
#include "nexus_model/core/kernels.hpp"
#include "nexus_model/core/module.hpp"

#include <vector>

namespace nexus_model {

class ConvNd : public Module {
 public:
  Tensor forward(const Tensor& input) override {
    const int spatial_rank = rank_;
    if (static_cast<int>(input.rank()) != spatial_rank + 2) throw ModelError("conv input rank mismatch");
    if (input.size(1) != static_cast<std::size_t>(in_channels_)) throw ModelError("conv channel mismatch");
    kernels::ConvDesc desc = make_desc(input);
    for (int i = 0; i < spatial_rank; ++i) {
      desc.spatial_out[i] = kernels::conv_out_size(desc.spatial_in[i], desc.pad[i], desc.dilation[i], desc.kernel[i], desc.stride[i]);
      if (desc.spatial_out[i] <= 0) throw ModelError("conv output spatial size is empty");
    }
    std::size_t out_dims[kMaxRank];
    out_dims[0] = input.size(0);
    out_dims[1] = static_cast<std::size_t>(out_channels_);
    for (int i = 0; i < spatial_rank; ++i) out_dims[2 + i] = static_cast<std::size_t>(desc.spatial_out[i]);
    output_.resize_dims(out_dims, static_cast<std::uint8_t>(spatial_rank + 2));
    input_.copy_from(input);
    const std::size_t col = kernels::conv_col_size(desc);
    std::size_t work_dims[1] = {col * 2};
    workspace_.resize_dims(work_dims, 1);
    last_desc_ = desc;
    kernels::conv_forward(input_.data(), weight_->data.data(), bias_ ? bias_->data.data() : nullptr, output_.data(),
                          workspace_.data(), desc);
    return output_;
  }

  Tensor backward(const Tensor& grad_output) override {
    grad_input_.resize_like(input_);
    grad_input_.zero();
    float* dw = weight_->requires_grad ? weight_->grad.data() : nullptr;
    float* db = (bias_ && bias_->requires_grad) ? bias_->grad.data() : nullptr;
    kernels::conv_backward(input_.data(), weight_->data.data(), grad_output.data(), grad_input_.data(), dw, db,
                           workspace_.data(), last_desc_);
    return grad_input_;
  }

 protected:
  ConvNd(int rank, int in_channels, int out_channels, const int* kernel, const int* stride, const int* pad,
         const int* dilation, int groups, bool use_bias)
      : rank_(rank),
        in_channels_(in_channels),
        out_channels_(out_channels),
        groups_(groups) {
    if (rank < 1 || rank > 3) throw ModelError("conv rank must be 1, 2 or 3");
    if (groups <= 0 || in_channels % groups != 0 || out_channels % groups != 0) {
      throw ModelError("conv groups must divide channels");
    }
    for (int i = 0; i < rank; ++i) {
      kernel_[i] = kernel[i];
      stride_[i] = stride[i];
      pad_[i] = pad[i];
      dilation_[i] = dilation[i];
    }
    std::size_t weight_dims[kMaxRank];
    weight_dims[0] = static_cast<std::size_t>(out_channels);
    weight_dims[1] = static_cast<std::size_t>(in_channels / groups);
    for (int i = 0; i < rank; ++i) weight_dims[2 + i] = static_cast<std::size_t>(kernel[i]);
    weight_ = std::make_shared<Parameter>(Tensor::zeros(std::vector<std::size_t>(weight_dims, weight_dims + rank + 2)));
    init::kaiming_uniform_(weight_->data);
    register_parameter("weight", weight_);
    if (use_bias) {
      bias_ = std::make_shared<Parameter>(Tensor::zeros({static_cast<std::size_t>(out_channels)}));
      register_parameter("bias", bias_);
    }
  }

 private:
  kernels::ConvDesc make_desc(const Tensor& input) const {
    kernels::ConvDesc desc;
    desc.n = static_cast<int>(input.size(0));
    desc.c = in_channels_;
    desc.k = out_channels_;
    desc.groups = groups_;
    desc.rank = rank_;
    for (int i = 0; i < rank_; ++i) {
      desc.spatial_in[i] = static_cast<int>(input.size(2 + i));
      desc.kernel[i] = kernel_[i];
      desc.stride[i] = stride_[i];
      desc.pad[i] = pad_[i];
      desc.dilation[i] = dilation_[i];
    }
    return desc;
  }

  int rank_;
  int in_channels_;
  int out_channels_;
  int groups_;
  int kernel_[3]{};
  int stride_[3]{};
  int pad_[3]{};
  int dilation_[3]{};
  kernels::ConvDesc last_desc_{};
  std::shared_ptr<Parameter> weight_;
  std::shared_ptr<Parameter> bias_;
  Tensor input_, output_, grad_input_, workspace_;
};

class Conv1D : public ConvNd {
 public:
  Conv1D(int in_channels, int out_channels, int kernel, int stride = 1, int padding = 0, int dilation = 1,
         int groups = 1, bool bias = true)
      : ConvNd(1, in_channels, out_channels, &kernel, &stride, &padding, &dilation, groups, bias) {}
};

class Conv2D : public ConvNd {
 public:
  Conv2D(int in_channels, int out_channels, int kernel, int stride = 1, int padding = 0, int dilation = 1,
         int groups = 1, bool bias = true)
      : Conv2D(in_channels, out_channels, kernel, kernel, stride, stride, padding, padding, dilation, dilation, groups,
               bias) {}

  Conv2D(int in_channels, int out_channels, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w,
         int dil_h, int dil_w, int groups = 1, bool bias = true)
      : ConvNd(2, in_channels, out_channels, kernel_pair(kernel_h, kernel_w), stride_pair(stride_h, stride_w),
               pad_pair(pad_h, pad_w), dil_pair(dil_h, dil_w), groups, bias) {}

 private:
  static const int* kernel_pair(int a, int b) {
    static thread_local int v[2];
    v[0] = a;
    v[1] = b;
    return v;
  }
  static const int* stride_pair(int a, int b) {
    static thread_local int v[2];
    v[0] = a;
    v[1] = b;
    return v;
  }
  static const int* pad_pair(int a, int b) {
    static thread_local int v[2];
    v[0] = a;
    v[1] = b;
    return v;
  }
  static const int* dil_pair(int a, int b) {
    static thread_local int v[2];
    v[0] = a;
    v[1] = b;
    return v;
  }
};

class Conv3D : public ConvNd {
 public:
  Conv3D(int in_channels, int out_channels, int kernel, int stride = 1, int padding = 0, int dilation = 1,
         int groups = 1, bool bias = true)
      : ConvNd(3, in_channels, out_channels, cube_k(kernel), cube_s(stride), cube_p(padding), cube_d(dilation), groups,
               bias) {}

 private:
  static const int* cube_k(int v) { return fill_cube(v); }
  static const int* cube_s(int v) {
    static thread_local int t[3];
    t[0] = t[1] = t[2] = v;
    return t;
  }
  static const int* cube_p(int v) {
    static thread_local int t[3];
    t[0] = t[1] = t[2] = v;
    return t;
  }
  static const int* cube_d(int v) {
    static thread_local int t[3];
    t[0] = t[1] = t[2] = v;
    return t;
  }
  static const int* fill_cube(int v) {
    static thread_local int t[3];
    t[0] = t[1] = t[2] = v;
    return t;
  }
};

class DepthwiseConv2D : public Conv2D {
 public:
  DepthwiseConv2D(int channels, int kernel, int stride = 1, int padding = 0)
      : Conv2D(channels, channels, kernel, stride, padding, 1, channels, true) {}
};

}  // namespace nexus_model
