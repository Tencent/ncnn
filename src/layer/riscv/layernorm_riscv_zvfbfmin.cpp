// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "layernorm_bf16s.h"

void layernorm_bf16s_zvfbfmin(Mat& bottom_top_blob, const float* gamma_data, const float* beta_data, float eps, int affine_size, const Option& opt)
{
    layernorm_bf16s(bottom_top_blob, gamma_data, beta_data, eps, affine_size, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
