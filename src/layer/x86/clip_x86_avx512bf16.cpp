// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "clip_x86.h"

#include "x86_usability.h"

#include "cpu.h"
#include "mat.h"

namespace ncnn {

#include "clip_bf16s.h"

int Clip_x86::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    return clip_bf16s(bottom_top_blob, min, max, opt);
}

} // namespace ncnn
