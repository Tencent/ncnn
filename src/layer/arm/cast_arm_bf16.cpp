// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"

namespace ncnn {

#include "cast_bf16.h"

void cast_fp32_to_bf16_neon_bf16(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    cast_fp32_to_bf16_neon(bottom_blob, top_blob, opt);
}

void cast_bf16_to_fp32_neon_bf16(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    cast_bf16_to_fp32_neon(bottom_blob, top_blob, opt);
}

} // namespace ncnn
