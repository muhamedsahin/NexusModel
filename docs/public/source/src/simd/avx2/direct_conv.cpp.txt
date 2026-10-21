#include "nexus_model/core/kernels.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <immintrin.h>

namespace nexus_model::kernels {
namespace {

bool geometry_ok(const ConvDesc& d, int& cin_g, int& cout_g, bool& depthwise) {
    if (d.rank != 2 || d.kernel[0] != 3 || d.kernel[1] != 3) {
        return false;
    }
    if (d.stride[0] != 1 || d.stride[1] != 1 || d.dilation[0] != 1 || d.dilation[1] != 1) {
        return false;
    }
    if (d.groups <= 0 || d.c % d.groups != 0 || d.k % d.groups != 0) {
        return false;
    }
    cin_g = d.c / d.groups;
    cout_g = d.k / d.groups;
    depthwise = d.groups == d.c && d.k == d.c && cin_g == 1 && cout_g == 1;
    return depthwise;
}

void depthwise_plane(const float* src, int height, int width, const float* tap, float bias_value, float* dst, int out_h,
                     int out_w, int pad_h, int pad_w) {
    const __m256 k00 = _mm256_set1_ps(tap[0]);
    const __m256 k01 = _mm256_set1_ps(tap[1]);
    const __m256 k02 = _mm256_set1_ps(tap[2]);
    const __m256 k10 = _mm256_set1_ps(tap[3]);
    const __m256 k11 = _mm256_set1_ps(tap[4]);
    const __m256 k12 = _mm256_set1_ps(tap[5]);
    const __m256 k20 = _mm256_set1_ps(tap[6]);
    const __m256 k21 = _mm256_set1_ps(tap[7]);
    const __m256 k22 = _mm256_set1_ps(tap[8]);
    const __m256 vb = _mm256_set1_ps(bias_value);
    const int vec_lo = pad_w;
    const int vec_hi = std::min(out_w, width + pad_w - 2);
    for (int oh = 0; oh < out_h; ++oh) {
        float* row_dst = dst + static_cast<std::size_t>(oh) * out_w;
        const int iy0 = oh - pad_h;
        const bool interior = iy0 >= 0 && iy0 + 2 < height && vec_hi - vec_lo >= 8;
        int ow = 0;
        if (interior) {
            for (; ow < vec_lo; ++ow) {
                float acc = bias_value;
                for (int kh = 0; kh < 3; ++kh) {
                    const int iy = iy0 + kh;
                    for (int kw = 0; kw < 3; ++kw) {
                        const int ix = ow - pad_w + kw;
                        if (ix >= 0 && ix < width) {
                            acc += src[iy * width + ix] * tap[kh * 3 + kw];
                        }
                    }
                }
                row_dst[ow] = acc;
            }
            for (; ow + 8 <= vec_hi; ow += 8) {
                const int ix0 = ow - pad_w;
                const float* r0 = src + static_cast<std::size_t>(iy0) * width + ix0;
                const float* r1 = r0 + width;
                const float* r2 = r1 + width;
                __m256 acc = vb;
                acc = _mm256_fmadd_ps(k00, _mm256_loadu_ps(r0), acc);
                acc = _mm256_fmadd_ps(k01, _mm256_loadu_ps(r0 + 1), acc);
                acc = _mm256_fmadd_ps(k02, _mm256_loadu_ps(r0 + 2), acc);
                acc = _mm256_fmadd_ps(k10, _mm256_loadu_ps(r1), acc);
                acc = _mm256_fmadd_ps(k11, _mm256_loadu_ps(r1 + 1), acc);
                acc = _mm256_fmadd_ps(k12, _mm256_loadu_ps(r1 + 2), acc);
                acc = _mm256_fmadd_ps(k20, _mm256_loadu_ps(r2), acc);
                acc = _mm256_fmadd_ps(k21, _mm256_loadu_ps(r2 + 1), acc);
                acc = _mm256_fmadd_ps(k22, _mm256_loadu_ps(r2 + 2), acc);
                _mm256_storeu_ps(row_dst + ow, acc);
            }
        }
        for (; ow < out_w; ++ow) {
            float acc = bias_value;
            for (int kh = 0; kh < 3; ++kh) {
                const int iy = oh - pad_h + kh;
                if (iy < 0 || iy >= height) {
                    continue;
                }
                for (int kw = 0; kw < 3; ++kw) {
                    const int ix = ow - pad_w + kw;
                    if (ix < 0 || ix >= width) {
                        continue;
                    }
                    acc += src[iy * width + ix] * tap[kh * 3 + kw];
                }
            }
            row_dst[ow] = acc;
        }
    }
}

struct PlaneJob {
    const float* x = nullptr;
    const float* w = nullptr;
    const float* bias = nullptr;
    float* y = nullptr;
    int height = 0;
    int width = 0;
    int out_h = 0;
    int out_w = 0;
    int pad_h = 0;
    int pad_w = 0;
    int channels = 0;
};

void run_planes(int begin, int end, void* user) {
    const PlaneJob* job = static_cast<const PlaneJob*>(user);
    for (int plane = begin; plane < end; ++plane) {
        const int n = plane / job->channels;
        const int channel = plane - n * job->channels;
        const std::size_t spatial_in = static_cast<std::size_t>(job->height) * static_cast<std::size_t>(job->width);
        const std::size_t spatial_out = static_cast<std::size_t>(job->out_h) * static_cast<std::size_t>(job->out_w);
        const float* src = job->x + (static_cast<std::size_t>(n) * static_cast<std::size_t>(job->channels) + static_cast<std::size_t>(channel)) * spatial_in;
        const float* tap = job->w + static_cast<std::size_t>(channel) * 9;
        float* dst = job->y + (static_cast<std::size_t>(n) * static_cast<std::size_t>(job->channels) + static_cast<std::size_t>(channel)) * spatial_out;
        const float bias_value = job->bias != nullptr ? job->bias[channel] : 0.0f;
        depthwise_plane(src, job->height, job->width, tap, bias_value, dst, job->out_h, job->out_w, job->pad_h, job->pad_w);
    }
}

class DepthwisePool {
public:
    DepthwisePool() {
        unsigned hardware = std::thread::hardware_concurrency();
        if (hardware < 2) {
            hardware = 1;
        }
        if (hardware > 8) {
            hardware = 8;
        }
        const unsigned workers = hardware - 1;
        threads.reserve(workers);
        for (unsigned i = 0; i < workers; ++i) {
            threads.emplace_back([this] { loop(); });
        }
    }

    ~DepthwisePool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
            ++epoch;
        }
        cv.notify_all();
        for (std::thread& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    void run(int plane_count, void (*body)(int, int, void*), void* context) {
        while (parked.load(std::memory_order_acquire) < static_cast<int>(threads.size())) {
            _mm_pause();
        }
        const int workers = static_cast<int>(threads.size()) + 1;
        int chunk_grain = plane_count / workers;
        if (chunk_grain < 1) {
            chunk_grain = 1;
        }
        fn = body;
        user = context;
        count = plane_count;
        grain = chunk_grain;
        chunks = (plane_count + chunk_grain - 1) / chunk_grain;
        next.store(0, std::memory_order_relaxed);
        done.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++epoch;
        }
        cv.notify_all();
        work();
        while (done.load(std::memory_order_acquire) < chunks) {
            _mm_pause();
        }
    }

private:
    void work() {
        for (;;) {
            const int chunk = next.fetch_add(1, std::memory_order_relaxed);
            if (chunk >= chunks) {
                break;
            }
            const int begin = chunk * grain;
            int end = begin + grain;
            if (end > count) {
                end = count;
            }
            fn(begin, end, user);
            done.fetch_add(1, std::memory_order_release);
        }
    }

    void loop() {
        int seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex);
                parked.fetch_add(1, std::memory_order_release);
                cv.wait(lock, [&] { return stop || epoch != seen; });
                parked.fetch_sub(1, std::memory_order_release);
                if (stop) {
                    return;
                }
                seen = epoch;
            }
            work();
        }
    }

    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable cv;
    void (*fn)(int, int, void*) = nullptr;
    void* user = nullptr;
    int count = 0;
    int grain = 1;
    int chunks = 0;
    int epoch = 0;
    bool stop = false;
    std::atomic<int> next{0};
    std::atomic<int> done{0};
    std::atomic<int> parked{0};
};

thread_local bool t_inside_depthwise = false;

void parallel_planes(int count, void (*fn)(int, int, void*), void* user) {
    if (t_inside_depthwise || count < 32) {
        fn(0, count, user);
        return;
    }
    static DepthwisePool pool;
    t_inside_depthwise = true;
    pool.run(count, fn, user);
    t_inside_depthwise = false;
}

}  // namespace

bool try_direct_conv2d_forward(const float* x, const float* w, const float* bias, float* y, const ConvDesc& d) {
    int cin_g = 0;
    int cout_g = 0;
    bool depthwise = false;
    if (!geometry_ok(d, cin_g, cout_g, depthwise)) {
        return false;
    }
    (void)cin_g;
    (void)cout_g;
    (void)depthwise;
    PlaneJob job;
    job.x = x;
    job.w = w;
    job.bias = bias;
    job.y = y;
    job.height = d.spatial_in[0];
    job.width = d.spatial_in[1];
    job.out_h = d.spatial_out[0];
    job.out_w = d.spatial_out[1];
    job.pad_h = d.pad[0];
    job.pad_w = d.pad[1];
    job.channels = d.groups;
    parallel_planes(d.n * d.groups, &run_planes, &job);
    return true;
}

bool try_direct_conv2d_backward(const float* x, const float* w, const float* dy, float* dx, float* dw, float* db,
                                const ConvDesc& d) {
    int cin_g = 0;
    int cout_g = 0;
    bool depthwise = false;
    if (!geometry_ok(d, cin_g, cout_g, depthwise)) {
        return false;
    }
    (void)depthwise;
    const int height = d.spatial_in[0];
    const int width = d.spatial_in[1];
    const int out_h = d.spatial_out[0];
    const int out_w = d.spatial_out[1];
    const int pad_h = d.pad[0];
    const int pad_w = d.pad[1];
    const int kvol = 9;
    for (int n = 0; n < d.n; ++n) {
        for (int g = 0; g < d.groups; ++g) {
            for (int oc = 0; oc < cout_g; ++oc) {
                const int out_c = g * cout_g + oc;
                const float* weight = w + static_cast<std::size_t>(out_c) * cin_g * kvol;
                float* dweight = dw != nullptr ? dw + static_cast<std::size_t>(out_c) * cin_g * kvol : nullptr;
                const float* grad = dy + static_cast<std::size_t>(n * d.k + out_c) * out_h * out_w;
                if (db != nullptr) {
                    float sum = 0.0f;
                    int s = 0;
                    for (; s + 8 <= out_h * out_w; s += 8) {
                        __m256 v = _mm256_loadu_ps(grad + s);
                        __m128 lo = _mm256_castps256_ps128(v);
                        __m128 hi = _mm256_extractf128_ps(v, 1);
                        lo = _mm_add_ps(lo, hi);
                        lo = _mm_hadd_ps(lo, lo);
                        lo = _mm_hadd_ps(lo, lo);
                        sum += _mm_cvtss_f32(lo);
                    }
                    for (; s < out_h * out_w; ++s) {
                        sum += grad[s];
                    }
                    db[out_c] += sum;
                }
                for (int ic = 0; ic < cin_g; ++ic) {
                    const float* src = x + static_cast<std::size_t>(n * d.c + g * cin_g + ic) * height * width;
                    float* dsrc = dx != nullptr ? dx + static_cast<std::size_t>(n * d.c + g * cin_g + ic) * height * width : nullptr;
                    const float* tap = weight + ic * kvol;
                    float* dtap = dweight != nullptr ? dweight + ic * kvol : nullptr;
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            const float go = grad[oh * out_w + ow];
                            for (int kh = 0; kh < 3; ++kh) {
                                const int iy = oh - pad_h + kh;
                                if (iy < 0 || iy >= height) {
                                    continue;
                                }
                                for (int kw = 0; kw < 3; ++kw) {
                                    const int ix = ow - pad_w + kw;
                                    if (ix < 0 || ix >= width) {
                                        continue;
                                    }
                                    if (dsrc != nullptr) {
                                        dsrc[iy * width + ix] += go * tap[kh * 3 + kw];
                                    }
                                    if (dtap != nullptr) {
                                        dtap[kh * 3 + kw] += go * src[iy * width + ix];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}

}  // namespace nexus_model::kernels
