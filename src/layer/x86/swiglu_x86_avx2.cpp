// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "layer.h"
#include "layer_type.h"
#include "mat.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "swiglu_fp32.h"

#if NCNN_BF16
#include "swiglu_bf16s.h"
#endif

void swiglu_fp32_avx2(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt, float scale, float offset, float limit)
{
    swiglu_fp32(gate, up, top_blob, opt, scale, offset, limit);
}

#if NCNN_BF16
void swiglu_bf16s_avx2(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt, float scale, float offset, float limit)
{
    swiglu_bf16s(gate, up, top_blob, opt, scale, offset, limit);
}
#endif

} // namespace ncnn
