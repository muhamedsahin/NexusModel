#include "nexus_model/core/error.hpp"
#include "nexus_model/core/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nexus_model::kernels {
namespace {

int product3(const int s[3]) { return s[0] * s[1] * s[2]; }

void align3(const ConvDesc& d, int in_s[3], int out_s[3], int ker[3], int str[3], int pad[3], int dil[3]) {
  for (int i = 0; i < 3; ++i) {
    in_s[i] = 1;
    out_s[i] = 1;
    ker[i] = 1;
    str[i] = 1;
    pad[i] = 0;
    dil[i] = 1;
  }
  for (int i = 0; i < d.rank; ++i) {
    const int slot = 3 - d.rank + i;
    in_s[slot] = d.spatial_in[i];
    out_s[slot] = d.spatial_out[i];
    ker[slot] = d.kernel[i];
    str[slot] = d.stride[i];
    pad[slot] = d.pad[i];
    dil[slot] = d.dilation[i];
  }
}

void im2col_group(const float* x, float* col, int n, int g, int channels, int cin_g, const int in_s[3],
                  const int out_s[3], const int ker[3], const int str[3], const int pad[3], const int dil[3],
                  int in_spatial, int col_stride) {
  int s = 0;
  for (int od = 0; od < out_s[0]; ++od) {
    for (int oh = 0; oh < out_s[1]; ++oh) {
      for (int ow = 0; ow < out_s[2]; ++ow, ++s) {
        int col_c = 0;
        for (int ic = 0; ic < cin_g; ++ic) {
          for (int kd = 0; kd < ker[0]; ++kd) {
            for (int kh = 0; kh < ker[1]; ++kh) {
              for (int kw = 0; kw < ker[2]; ++kw, ++col_c) {
                const int id = od * str[0] - pad[0] + kd * dil[0];
                const int ih = oh * str[1] - pad[1] + kh * dil[1];
                const int iw = ow * str[2] - pad[2] + kw * dil[2];
                float value = 0.f;
                if (id >= 0 && id < in_s[0] && ih >= 0 && ih < in_s[1] && iw >= 0 && iw < in_s[2]) {
                  const int spat = (id * in_s[1] + ih) * in_s[2] + iw;
                  const int channel = g * cin_g + ic;
                  value = x[(static_cast<std::size_t>(n * channels + channel) * in_spatial) + spat];
                }
                col[static_cast<std::size_t>(s) * col_stride + col_c] = value;
              }
            }
          }
        }
      }
    }
  }
}

void col2im_group(const float* dcol, float* dx, int n, int g, int channels, int cin_g, const int in_s[3],
                  const int out_s[3], const int ker[3], const int str[3], const int pad[3], const int dil[3],
                  int in_spatial, int col_stride) {
  int s = 0;
  for (int od = 0; od < out_s[0]; ++od) {
    for (int oh = 0; oh < out_s[1]; ++oh) {
      for (int ow = 0; ow < out_s[2]; ++ow, ++s) {
        int col_c = 0;
        for (int ic = 0; ic < cin_g; ++ic) {
          for (int kd = 0; kd < ker[0]; ++kd) {
            for (int kh = 0; kh < ker[1]; ++kh) {
              for (int kw = 0; kw < ker[2]; ++kw, ++col_c) {
                const int id = od * str[0] - pad[0] + kd * dil[0];
                const int ih = oh * str[1] - pad[1] + kh * dil[1];
                const int iw = ow * str[2] - pad[2] + kw * dil[2];
                if (id < 0 || id >= in_s[0] || ih < 0 || ih >= in_s[1] || iw < 0 || iw >= in_s[2]) continue;
                const int spat = (id * in_s[1] + ih) * in_s[2] + iw;
                const int channel = g * cin_g + ic;
                dx[(static_cast<std::size_t>(n * channels + channel) * in_spatial) + spat] +=
                    dcol[static_cast<std::size_t>(s) * col_stride + col_c];
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace

int conv_out_size(int input, int pad, int dilation, int kernel, int stride) {
  if (stride <= 0 || kernel <= 0 || dilation <= 0) throw ModelError("conv geometry must be positive");
  const int numer = input + 2 * pad - dilation * (kernel - 1) - 1;
  if (numer < 0) return 0;
  return numer / stride + 1;
}

std::size_t conv_col_size(const ConvDesc& d) {
  if (d.groups <= 0 || d.c % d.groups != 0 || d.k % d.groups != 0) {
    throw ModelError("conv groups must divide channels");
  }
  int out_p = 1;
  int kvol = 1;
  for (int i = 0; i < d.rank; ++i) {
    out_p *= d.spatial_out[i];
    kvol *= d.kernel[i];
  }
  return static_cast<std::size_t>(out_p) * static_cast<std::size_t>(d.c / d.groups) * static_cast<std::size_t>(kvol);
}

void conv_forward(const float* x, const float* w, const float* bias, float* y, float* col, const ConvDesc& d) {
  int in_s[3], out_s[3], ker[3], str[3], pad[3], dil[3];
  align3(d, in_s, out_s, ker, str, pad, dil);
  const int cin_g = d.c / d.groups;
  const int cout_g = d.k / d.groups;
  const int in_spatial = product3(in_s);
  const int out_spatial = product3(out_s);
  const int col_stride = cin_g * product3(ker);
  for (int n = 0; n < d.n; ++n) {
    for (int g = 0; g < d.groups; ++g) {
      im2col_group(x, col, n, g, d.c, cin_g, in_s, out_s, ker, str, pad, dil, in_spatial, col_stride);
      float* output = y + static_cast<std::size_t>(n*d.k+g*cout_g)*out_spatial;
      // W * col^T directly produces channel-major output, without a transpose.
      linear_forward(w+static_cast<std::size_t>(g*cout_g)*col_stride,col,nullptr,
                     output,cout_g,col_stride,out_spatial);
      if(bias) for(int oc=0;oc<cout_g;++oc) {
        const float value=bias[g*cout_g+oc];
        float* row=output+static_cast<std::size_t>(oc)*out_spatial;
        for(int s=0;s<out_spatial;++s) row[s]+=value;
      }
    }
  }
}

void conv_backward(const float* x, const float* w, const float* dy, float* dx, float* dw, float* db, float* col,
                   const ConvDesc& d) {
  int in_s[3], out_s[3], ker[3], str[3], pad[3], dil[3];
  align3(d, in_s, out_s, ker, str, pad, dil);
  const int cin_g = d.c / d.groups;
  const int cout_g = d.k / d.groups;
  const int in_spatial = product3(in_s);
  const int out_spatial = product3(out_s);
  const int col_stride = cin_g * product3(ker);
  const std::size_t col_elems = static_cast<std::size_t>(out_spatial) * static_cast<std::size_t>(col_stride);
  float* dcol = col + col_elems;
  for (int n = 0; n < d.n; ++n) {
    for (int g = 0; g < d.groups; ++g) {
      im2col_group(x, col, n, g, d.c, cin_g, in_s, out_s, ker, str, pad, dil, in_spatial, col_stride);
      if(dx) std::fill(dcol, dcol + col_elems, 0.f);
      const float* grad=dy+static_cast<std::size_t>(n*d.k+g*cout_g)*out_spatial;
      linear_backward(w+static_cast<std::size_t>(g*cout_g)*col_stride,col,grad,
                      dw?dw+static_cast<std::size_t>(g*cout_g)*col_stride:nullptr,
                      dx?dcol:nullptr,nullptr,cout_g,col_stride,out_spatial);
      if(db) for(int oc=0;oc<cout_g;++oc) {
        float sum=0;
        for(int s=0;s<out_spatial;++s) sum+=grad[static_cast<std::size_t>(oc)*out_spatial+s];
        db[g*cout_g+oc]+=sum;
      }
      if (dx != nullptr) {
        col2im_group(dcol, dx, n, g, d.c, cin_g, in_s, out_s, ker, str, pad, dil, in_spatial, col_stride);
      }
    }
  }
}


}  // namespace nexus_model::kernels
