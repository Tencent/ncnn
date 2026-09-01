// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16
#include "batchnorm_bf16s.h"
#endif // NCNN_BF16
#include "batchnorm_fp32.h"

#if NCNN_BF16
void batchnorm_bf16s_sse_fma(unsigned short* ptr, const float* a, const float* b, int size, int elempack)
{
    batchnorm_bf16s_sse(ptr, a, b, size, elempack);
}

void batchnorm_bf16s_per_element_sse_fma(unsigned short* ptr, const float* a, const float* b, int size, int num_threads)
{
    batchnorm_bf16s_per_element_sse(ptr, a, b, size, num_threads);
}
#endif // NCNN_BF16

void batchnorm_fp32_fma(Mat& bottom_top_blob, const Mat& a_data, const Mat& b_data, const Option& opt)
{
    batchnorm_fp32(bottom_top_blob, a_data, b_data, opt);
}
} // namespace ncnn
