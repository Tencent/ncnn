// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

#include "rotaryembed_fp16s.h"

void rotaryembed_fp16s_asimdhp(const Mat& bottom_blob, const Mat& cos_cache, const Mat& sin_cache, Mat& top_blob, int interleaved, const Option& opt)
{
    rotaryembed_fp16s(bottom_blob, cos_cache, sin_cache, top_blob, interleaved, opt);
}

void rotaryembed_fp16sa_asimdhp(const Mat& bottom_blob, const Mat& cos_cache, const Mat& sin_cache, Mat& top_blob, int interleaved, const Option& opt)
{
    rotaryembed_fp16sa(bottom_blob, cos_cache, sin_cache, top_blob, interleaved, opt);
}

} // namespace ncnn
