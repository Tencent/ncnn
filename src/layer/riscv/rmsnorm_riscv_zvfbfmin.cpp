// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "rmsnorm_bf16s.h"

void rmsnorm_bf16s_zvfbfmin(Mat& bottom_top_blob, const float* gamma_data, float eps, int affine_size, const Option& opt)
{
    rmsnorm_bf16s(bottom_top_blob, gamma_data, eps, affine_size, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
