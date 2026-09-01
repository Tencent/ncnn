// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "layer.h"
#include "mat.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "prelu_fp32.h"

#if NCNN_BF16
#include "prelu_bf16s.h"

void prelu_bf16s_sse_fma(unsigned short* ptr, const float* slope, int size, int elempack)
{
    prelu_bf16s_sse(ptr, slope, size, elempack);
}

void prelu_bf16s_per_element_sse_fma(unsigned short* ptr, const float* slope, int size, int num_threads)
{
    prelu_bf16s_per_element_sse(ptr, slope, size, num_threads);
}

void prelu_bf16s_single_slope_sse_fma(unsigned short* ptr, float slope, int size, int num_threads)
{
    prelu_bf16s_single_slope_sse(ptr, slope, size, num_threads);
}
#endif // NCNN_BF16

void prelu_fp32_fma(Mat& bottom_top_blob, const Mat& slope_data, int num_slope, const Option& opt)
{
    prelu_fp32(bottom_top_blob, slope_data, num_slope, opt);
}

} // namespace ncnn
