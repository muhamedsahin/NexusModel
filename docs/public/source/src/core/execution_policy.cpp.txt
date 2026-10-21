#include "nexus_model/core/execution_policy.hpp"

namespace nexus_model {
namespace {

MatmulPolicy g_policy = MatmulPolicy::CpuKernel;
PackedGemmFn g_gemm = nullptr;

}  // namespace

void set_matmul_policy(MatmulPolicy policy) { g_policy = policy; }
MatmulPolicy matmul_policy() { return g_policy; }
void set_packed_gemm(PackedGemmFn function) { g_gemm = function; }
PackedGemmFn packed_gemm() { return g_gemm; }

}  // namespace nexus_model
