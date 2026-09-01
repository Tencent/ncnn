// Copyright 2025 pchar.cn
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __SSE3__
#include <pmmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE3__
#endif // __SSE2__

#include "x86_usability.h"

#include "cpu.h"

namespace ncnn {

#include "rotaryembed_fp32.h"

#if NCNN_BF16
#include "rotaryembed_bf16s.h"
#endif

RotaryEmbed_x86::RotaryEmbed_x86()
{
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int RotaryEmbed_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_blobs[0].elembits() == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);
#endif

    return rotaryembed_fp32(bottom_blobs, top_blobs, interleaved, opt);
}

#if NCNN_BF16
int RotaryEmbed_x86::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& cos_cache = bottom_blobs[1];
    const Mat& sin_cache = bottom_blobs[2];

    Mat& top_blob = top_blobs[0];

    rotaryembed_bf16s(bottom_blob, cos_cache, sin_cache, top_blob, interleaved, opt);

    return top_blob.empty() ? -100 : 0;
}
#endif // NCNN_BF16

} // namespace ncnn
