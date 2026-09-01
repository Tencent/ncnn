// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolutiondepthwise_x86.h"

#include <immintrin.h>

#include "cpu.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "convolutiondepthwise_3x3_pack4.h"
#include "convolutiondepthwise_5x5_pack4.h"
#include "convolutiondepthwise_3x3_pack8.h"
#include "convolutiondepthwise_5x5_pack8.h"
#include "convolutiondepthwise_packed.h"

void convdw3x3s1_pack4_sse_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& bias, const Option& opt)
{
    convdw3x3s1_pack4_sse(bottom_blob, top_blob, kernel, bias, opt);
}

void convdw3x3s2_pack4_sse_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& bias, const Option& opt)
{
    convdw3x3s2_pack4_sse(bottom_blob, top_blob, kernel, bias, opt);
}

void convdw5x5s1_pack4_sse_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& bias, const Option& opt)
{
    convdw5x5s1_pack4_sse(bottom_blob, top_blob, kernel, bias, opt);
}

void convdw5x5s2_pack4_sse_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& bias, const Option& opt)
{
    convdw5x5s2_pack4_sse(bottom_blob, top_blob, kernel, bias, opt);
}

void convdw3x3s1_pack8_avx_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& bias, const Option& opt)
{
    convdw3x3s1_pack8_avx(bottom_blob, top_blob, kernel, bias, opt);
}

void convdw3x3s2_pack8_avx_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& bias, const Option& opt)
{
    convdw3x3s2_pack8_avx(bottom_blob, top_blob, kernel, bias, opt);
}

void convdw5x5s1_pack8_avx_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& bias, const Option& opt)
{
    convdw5x5s1_pack8_avx(bottom_blob, top_blob, kernel, bias, opt);
}

void convdw5x5s2_pack8_avx_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& bias, const Option& opt)
{
    convdw5x5s2_pack8_avx(bottom_blob, top_blob, kernel, bias, opt);
}

void convolutiondepthwise_packed8_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int bias_term, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    convolutiondepthwise_packed8(bottom_blob, top_blob, weight_data, bias_data, bias_term, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
}

} // namespace ncnn
