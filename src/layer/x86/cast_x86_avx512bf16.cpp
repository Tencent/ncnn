// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "cast_bf16.h"

void cast_fp32_to_bf16_sse_avx512bf16(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    cast_fp32_to_bf16_sse(bottom_blob, top_blob, opt);
}

void cast_bf16_to_fp32_sse_avx512bf16(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    cast_bf16_to_fp32_sse(bottom_blob, top_blob, opt);
}

} // namespace ncnn
