// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "cast_bf16.h"

void cast_fp32_to_bf16_avxneconvert(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    cast_fp32_to_bf16(bottom_blob, top_blob, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
