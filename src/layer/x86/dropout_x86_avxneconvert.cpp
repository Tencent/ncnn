// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "dropout_bf16s.h"

void dropout_bf16s_avxneconvert(Mat& a, float scale, const Option& opt)
{
    dropout_bf16s(a, scale, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
