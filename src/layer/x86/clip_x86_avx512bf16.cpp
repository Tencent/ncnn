// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "clip_x86.h"

#include "x86_usability.h"

#include "cpu.h"
#include "mat.h"

namespace ncnn {

#include "clip_bf16s.h"

void clip_bf16s_avx512bf16(Mat& a, float min, float max, const Option& opt)
{
    clip_bf16s(a, min, max, opt);
}

} // namespace ncnn
