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
    /* The fused layout (input_hc_swapped) is produced by the exporter together
     * with dropping a Permute in front of this layer. The overrides below still
     * assume h is the sequence and c the heads, so hand the fused case to the
     * generic implementation instead of mis-indexing the one-row caches. */
    if (input_hc_swapped)
    {
        /* 16-bit fused input goes through the generic implementation's fp32
         * conversion path, which decides fp16-vs-bf16 from the storage flags.
         * Set them to the format THIS backend would have selected for the same
         * input, so fused 16-bit models decode the bits they actually contain. */
        Option opt2 = opt;
        if (bottom_blobs[0].elembits() == 16)
        {
            opt2.use_fp16_storage = 0 ? 1 : 0;
            opt2.use_bf16_storage = opt2.use_fp16_storage ? 0 : (opt.use_bf16_storage ? 1 : 0);
        }
        return RotaryEmbed::forward(bottom_blobs, top_blobs, opt2);
    }

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_blobs[0].elembits() == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);
#endif

    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& cos_cache = bottom_blobs[1];
    const Mat& sin_cache = bottom_blobs[2];

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    rotaryembed_fp32(bottom_blob, cos_cache, sin_cache, top_blob, interleaved, opt);

    return 0;
}

#if NCNN_BF16
int RotaryEmbed_x86::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& cos_cache = bottom_blobs[1];
    const Mat& sin_cache = bottom_blobs[2];

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    rotaryembed_bf16s(bottom_blob, cos_cache, sin_cache, top_blob, interleaved, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
