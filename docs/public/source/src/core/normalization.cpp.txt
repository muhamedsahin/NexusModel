#include "nexus_model/core/error.hpp"
#include "nexus_model/core/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nexus_model::kernels {
void batch_norm(const float* x, const float* gamma, const float* beta, float* y, float* save_mean, float* save_rstd,
                float* running_mean, float* running_var, int outer, int channels, int inner, float eps, float momentum,
                bool training, bool track_running) {
  const int m = outer * inner;
  if (m <= 0) throw ModelError("batch norm received an empty reduction");
  for (int c = 0; c < channels; ++c) {
    float mean = 0.f;
    float rstd = 0.f;
    if (training) {
      double sum = 0.0;
      for (int n = 0; n < outer; ++n) {
        for (int i = 0; i < inner; ++i) sum += x[(static_cast<std::size_t>(n * channels + c) * inner) + i];
      }
      mean = static_cast<float>(sum / m);
      double var_acc = 0.0;
      for (int n = 0; n < outer; ++n) {
        for (int i = 0; i < inner; ++i) {
          const double diff = x[(static_cast<std::size_t>(n * channels + c) * inner) + i] - mean;
          var_acc += diff * diff;
        }
      }
      const float var = static_cast<float>(var_acc / m);
      rstd = 1.f / std::sqrt(var + eps);
      if (track_running && running_mean != nullptr && running_var != nullptr) {
        running_mean[c] = (1.f - momentum) * running_mean[c] + momentum * mean;
        const float unbiased = m > 1 ? static_cast<float>(var_acc / (m - 1)) : var;
        running_var[c] = (1.f - momentum) * running_var[c] + momentum * unbiased;
      }
    } else {
      mean = running_mean != nullptr ? running_mean[c] : 0.f;
      const float var = running_var != nullptr ? running_var[c] : 1.f;
      rstd = 1.f / std::sqrt(var + eps);
    }
    save_mean[c] = mean;
    save_rstd[c] = rstd;
    const float g = gamma != nullptr ? gamma[c] : 1.f;
    const float b = beta != nullptr ? beta[c] : 0.f;
    for (int n = 0; n < outer; ++n) {
      for (int i = 0; i < inner; ++i) {
        const std::size_t idx = (static_cast<std::size_t>(n * channels + c) * inner) + i;
        y[idx] = (x[idx] - mean) * rstd * g + b;
      }
    }
  }
}

void batch_norm_bwd(const float* x, const float* gamma, const float* save_mean, const float* save_rstd, const float* dy,
                    float* dx, float* dgamma, float* dbeta, int outer, int channels, int inner) {
  const int m = outer * inner;
  const float inv_m = 1.f / static_cast<float>(m);
  for (int c = 0; c < channels; ++c) {
    const float mean = save_mean[c];
    const float rstd = save_rstd[c];
    const float g = gamma != nullptr ? gamma[c] : 1.f;
    double sum_dy = 0.0;
    double sum_dy_xhat = 0.0;
    for (int n = 0; n < outer; ++n) {
      for (int i = 0; i < inner; ++i) {
        const std::size_t idx = (static_cast<std::size_t>(n * channels + c) * inner) + i;
        const float xhat = (x[idx] - mean) * rstd;
        const float dys = dy[idx] * g;
        sum_dy += dys;
        sum_dy_xhat += static_cast<double>(dys) * xhat;
        if (dgamma != nullptr) dgamma[c] += dy[idx] * xhat;
        if (dbeta != nullptr) dbeta[c] += dy[idx];
      }
    }
    for (int n = 0; n < outer; ++n) {
      for (int i = 0; i < inner; ++i) {
        const std::size_t idx = (static_cast<std::size_t>(n * channels + c) * inner) + i;
        const float xhat = (x[idx] - mean) * rstd;
        const float dys = dy[idx] * g;
        dx[idx] = rstd * inv_m *
                  (static_cast<float>(m) * dys - static_cast<float>(sum_dy) - xhat * static_cast<float>(sum_dy_xhat));
      }
    }
  }
}

void group_norm(const float* x, const float* gamma, const float* beta, float* y, float* save_mean, float* save_rstd,
                int n, int channels, int inner, int groups, float eps) {
  if (groups <= 0 || channels % groups != 0) throw ModelError("group norm groups must divide channels");
  const int cpg = channels / groups;
  const int m = cpg * inner;
  for (int ni = 0; ni < n; ++ni) {
    for (int g = 0; g < groups; ++g) {
      double sum = 0.0;
      for (int c = 0; c < cpg; ++c) {
        const int ch = g * cpg + c;
        for (int i = 0; i < inner; ++i) sum += x[(static_cast<std::size_t>(ni * channels + ch) * inner) + i];
      }
      const float mean = static_cast<float>(sum / m);
      double var_acc = 0.0;
      for (int c = 0; c < cpg; ++c) {
        const int ch = g * cpg + c;
        for (int i = 0; i < inner; ++i) {
          const double diff = x[(static_cast<std::size_t>(ni * channels + ch) * inner) + i] - mean;
          var_acc += diff * diff;
        }
      }
      const float rstd = 1.f / std::sqrt(static_cast<float>(var_acc / m) + eps);
      save_mean[ni * groups + g] = mean;
      save_rstd[ni * groups + g] = rstd;
      for (int c = 0; c < cpg; ++c) {
        const int ch = g * cpg + c;
        const float gv = gamma != nullptr ? gamma[ch] : 1.f;
        const float bv = beta != nullptr ? beta[ch] : 0.f;
        for (int i = 0; i < inner; ++i) {
          const std::size_t idx = (static_cast<std::size_t>(ni * channels + ch) * inner) + i;
          y[idx] = (x[idx] - mean) * rstd * gv + bv;
        }
      }
    }
  }
}

void group_norm_bwd(const float* x, const float* gamma, const float* save_mean, const float* save_rstd, const float* dy,
                    float* dx, float* dgamma, float* dbeta, int n, int channels, int inner, int groups) {
  const int cpg = channels / groups;
  const int m = cpg * inner;
  const float inv_m = 1.f / static_cast<float>(m);
  for (int ni = 0; ni < n; ++ni) {
    for (int g = 0; g < groups; ++g) {
      const float mean = save_mean[ni * groups + g];
      const float rstd = save_rstd[ni * groups + g];
      double sum_dy = 0.0;
      double sum_dy_xhat = 0.0;
      for (int c = 0; c < cpg; ++c) {
        const int ch = g * cpg + c;
        const float gv = gamma != nullptr ? gamma[ch] : 1.f;
        for (int i = 0; i < inner; ++i) {
          const std::size_t idx = (static_cast<std::size_t>(ni * channels + ch) * inner) + i;
          const float xhat = (x[idx] - mean) * rstd;
          const float dys = dy[idx] * gv;
          sum_dy += dys;
          sum_dy_xhat += static_cast<double>(dys) * xhat;
          if (dgamma != nullptr) dgamma[ch] += dy[idx] * xhat;
          if (dbeta != nullptr) dbeta[ch] += dy[idx];
        }
      }
      for (int c = 0; c < cpg; ++c) {
        const int ch = g * cpg + c;
        const float gv = gamma != nullptr ? gamma[ch] : 1.f;
        for (int i = 0; i < inner; ++i) {
          const std::size_t idx = (static_cast<std::size_t>(ni * channels + ch) * inner) + i;
          const float xhat = (x[idx] - mean) * rstd;
          const float dys = dy[idx] * gv;
          dx[idx] = rstd * inv_m * (static_cast<float>(m) * dys - static_cast<float>(sum_dy) -
                                    xhat * static_cast<float>(sum_dy_xhat));
        }
      }
    }
  }
}


}  // namespace nexus_model::kernels
