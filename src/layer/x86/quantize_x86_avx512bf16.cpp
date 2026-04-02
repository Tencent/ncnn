// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "quantize_bf16s.h"

int quantize_forward_bf16s_avx512bf16(const Mat& bottom_blob, Mat& top_blob, const Mat& scale_data, int scale_data_size, const Option& opt)
{
    return quantize_forward_bf16s(bottom_blob, top_blob, scale_data, scale_data_size, opt);
}

} // namespace ncnn
