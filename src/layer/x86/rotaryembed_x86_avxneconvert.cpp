// Copyright 2026 pchar.cn
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "rotaryembed_bf16s.h"

void rotaryembed_bf16s_avxneconvert(const Mat& bottom_blob, const Mat& cos_cache, const Mat& sin_cache, Mat& top_blob, int interleaved, const Option& opt)
{
    rotaryembed_bf16s(bottom_blob, cos_cache, sin_cache, top_blob, interleaved, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
