// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16
#include "instancenorm_bf16s.h"
#endif // NCNN_BF16
#include "instancenorm_fp32.h"

#if NCNN_BF16
void instancenorm_bf16s_fma(unsigned short* ptr, int size, float a, float b)
{
    instancenorm_bf16s(ptr, size, a, b);
}

void instancenorm_bf16s_compute_mean_var_fma(const unsigned short* ptr, int size, float& mean, float& var)
{
    instancenorm_bf16s_compute_mean_var(ptr, size, mean, var);
}
#endif // NCNN_BF16

void instancenorm_fp32_fma(Mat& bottom_top_blob, float eps, int affine, const Mat& gamma_data, const Mat& beta_data, const Option& opt)
{
    instancenorm_fp32(bottom_top_blob, eps, affine, gamma_data, beta_data, opt);
}

} // namespace ncnn
