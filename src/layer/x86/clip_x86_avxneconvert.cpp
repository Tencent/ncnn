// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "clip_bf16s.h"

void clip_bf16s_avxneconvert(Mat& a, float min, float max, const Option& opt)
{
    clip_bf16s(a, min, max, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
