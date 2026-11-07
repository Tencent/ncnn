// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif

namespace ncnn {

#if NCNN_BF16

#include "instancenorm_bf16s.h"

void instancenorm_bf16s_zvfbfmin(unsigned short* ptr, int size, float a, float b)
{
    instancenorm_bf16s(ptr, size, a, b);
}

void instancenorm_bf16s_compute_mean_var_zvfbfmin(const unsigned short* ptr, int size, float& mean, float& var)
{
    instancenorm_bf16s_compute_mean_var(ptr, size, mean, var);
}

void instancenorm_bf16s_packn_zvfbfmin(unsigned short* ptr, int size, int elempack, const float* gamma, const float* beta, float eps, int affine)
{
    instancenorm_bf16s_packn(ptr, size, elempack, gamma, beta, eps, affine);
}

#endif // NCNN_BF16

} // namespace ncnn
