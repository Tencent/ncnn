// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "swiglu_x86.h"

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#include "x86_activation.h"
#include "x86_usability.h"

#include "cpu.h"

namespace ncnn {

#include "swiglu_fp32.h"

#if NCNN_BF16
#include "swiglu_bf16s.h"
#endif

SwiGLU_x86::SwiGLU_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int SwiGLU_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() != 2)
        return -100;

    const Mat& gate = bottom_blobs[0];
    const Mat& up = bottom_blobs[1];

    if (gate.dims != up.dims || gate.w != up.w || gate.h != up.h || gate.d != up.d || gate.c != up.c || gate.elempack != up.elempack || gate.elembits() != up.elembits())
        return -100;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(gate, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if NCNN_BF16
    if (opt.use_bf16_storage && gate.elembits() == 16)
    {
        swiglu_bf16s(gate, up, top_blob, opt, sigmoid_scale, up_offset, activation_limit);
        return 0;
    }
#endif

    swiglu_fp32(gate, up, top_blob, opt, sigmoid_scale, up_offset, activation_limit);

    return 0;
}

} // namespace ncnn
