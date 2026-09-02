// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eltwise_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"

#include "cpu.h"

namespace ncnn {

#include "eltwise_fp32.h"

#if NCNN_BF16
#include "eltwise_bf16s.h"
#endif

Eltwise_x86::Eltwise_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Eltwise_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_blobs[0].elembits() == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);
#endif
    const Mat& bottom_blob = bottom_blobs[0];

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    eltwise_fp32(bottom_blobs, top_blob, op_type, coeffs, opt);

    return 0;
}

#if NCNN_BF16
int Eltwise_x86::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    eltwise_bf16s(bottom_blobs, top_blob, op_type, coeffs, opt);

    return 0;
}

#endif // NCNN_BF16

} // namespace ncnn
