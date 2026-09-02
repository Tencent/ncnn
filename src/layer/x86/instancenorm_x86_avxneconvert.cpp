// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "instancenorm_bf16s.h"

void instancenorm_bf16s_avxneconvert(unsigned short* ptr, int size, float a, float b)
{
    instancenorm_bf16s(ptr, size, a, b);
}

void instancenorm_bf16s_compute_mean_var_avxneconvert(const unsigned short* ptr, int size, float& mean, float& var)
{
    instancenorm_bf16s_compute_mean_var(ptr, size, mean, var);
}

#endif // NCNN_BF16

} // namespace ncnn
