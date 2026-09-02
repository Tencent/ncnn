// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16
#include "dequantize_bf16s.h"

void dequantize_bf16s_avx2(const Mat& bottom_blob, Mat& top_blob, const Mat& scale_data, int scale_data_size, const Mat& bias_data, int bias_data_size, const Option& opt)
{
    dequantize_bf16s(bottom_blob, top_blob, scale_data, scale_data_size, bias_data, bias_data_size, opt);
}
#endif // NCNN_BF16

} // namespace ncnn
