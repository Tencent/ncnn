// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "scale_bf16s.h"
#include "scale_fp32.h"

void scale_bf16s_sse_fma(unsigned short* ptr, const float* scale, const float* bias, int size, int elempack)
{
    scale_bf16s_sse(ptr, scale, bias, size, elempack);
}

void scale_bf16s_per_element_sse_fma(unsigned short* ptr, const float* scale, const float* bias, int size, int num_threads)
{
    scale_bf16s_per_element_sse(ptr, scale, bias, size, num_threads);
}

void scale_fp32_fma(std::vector<Mat>& bottom_top_blobs, int bias_term, const Mat& bias_data, const Option& opt)
{
    scale_fp32(bottom_top_blobs, bias_term, bias_data, opt);
}
} // namespace ncnn
