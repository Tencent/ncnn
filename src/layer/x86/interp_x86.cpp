// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "interp_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include <string.h>

#include "cpu.h"
#include "x86_usability.h"

namespace ncnn {

#include "interp_bicubic.h"
#include "interp_bilinear.h"

#if __SSE2__
#include "interp_bicubic_pack4.h"
#include "interp_bilinear_pack4.h"

#if __AVX__
#include "interp_bicubic_pack8.h"
#include "interp_bilinear_pack8.h"

#if __AVX512F__
#include "interp_bicubic_pack16.h"
#include "interp_bilinear_pack16.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#include "interp_fp32.h"

Interp_x86::Interp_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Interp_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];

    int outw = reference_blob.w;
    int outh = reference_blob.h;

    if (!size_expr.empty())
    {
        std::vector<Mat> bottom_blob_shapes(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            bottom_blob_shapes[i] = bottom_blobs[i].shape();
        }
        eval_size_expr(bottom_blob_shapes, outw, outh);
    }

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_blob.elembits() == 16)
    {
        return forward_bf16s(bottom_blobs, top_blobs, opt);
    }
#endif

    Mat& top_blob = top_blobs[0];

    const int h = bottom_blob.h;
    const int w = bottom_blob.w;
    const int channels = bottom_blob.c;
    const int dims = bottom_blob.dims;
    const size_t elemsize = bottom_blob.elemsize;
    const int elempack = bottom_blob.elempack;

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
    }
    else if (dims == 2)
    {
        if (outw == w)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
    }
    else // dims == 3
    {
        if (outw == w && outh == h)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
    }

    interp_fp32(bottom_blob, top_blob, resize_type, align_corner, height_scale, width_scale, output_height, output_width, !size_expr.empty(), opt);

    return 0;
}
#if NCNN_BF16

#include "interp_bf16s.h"

int Interp_x86::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = reference_blob.w;
    int outh = reference_blob.h;

    if (!size_expr.empty())
    {
        std::vector<Mat> bottom_blob_shapes(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            bottom_blob_shapes[i] = bottom_blobs[i].shape();
        }
        eval_size_expr(bottom_blob_shapes, outw, outh);
    }

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
    }
    else if (dims == 2)
    {
        if (outw == w)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
    }
    else // dims == 3
    {
        if (outw == w && outh == h)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
    }

    interp_bf16s(bottom_blob, top_blob, resize_type, align_corner, height_scale, width_scale, output_height, output_width, !size_expr.empty(), opt);

    return 0;
}

#endif // NCNN_BF16

} // namespace ncnn
