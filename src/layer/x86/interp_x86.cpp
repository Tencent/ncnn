// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "interp_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include <float.h>
#include <limits.h>
#include <stdio.h>
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

int Interp_x86::load_param(const ParamDict& pd)
{
    source_scale_dims = 0;
    source_width_scale = 1.0;
    source_height_scale = 1.0;
    height_scale_inv = 0.f;
    width_scale_inv = 0.f;

    if (pd.get(0, 0) != 4)
        return Interp::load_param(pd);

    ParamDict pd_nearest = pd;
    pd_nearest.set(0, 1);
    int ret = Interp::load_param(pd_nearest);
    if (ret != 0)
        return ret;

    resize_type = 4;
    support_vulkan = false;

    if (size_expr.compare(0, 6, "scale(") == 0)
    {
        int consumed = 0;
        if (sscanf(size_expr.c_str(), "scale(%lf,%lf)%n", &source_width_scale, &source_height_scale, &consumed) == 2 && consumed == (int)size_expr.size())
            source_scale_dims = 2;
        else
        {
            consumed = 0;
            if (sscanf(size_expr.c_str(), "scale(%lf)%n", &source_width_scale, &consumed) == 1 && consumed == (int)size_expr.size())
                source_scale_dims = 1;
            else
                return -1;
        }

        if (!(source_width_scale > 0.0 && source_width_scale <= FLT_MAX && source_height_scale > 0.0 && source_height_scale <= FLT_MAX))
            return -1;

        width_scale_inv = (float)(1.0 / source_width_scale);
        height_scale_inv = (float)(1.0 / source_height_scale);
    }

    return 0;
}

bool Interp_x86::is_identity_resize(int w, int h, int outw, int outh, int dims) const
{
    if (resize_type != 4)
        return Interp::is_identity_resize(w, h, outw, outh, dims);

    if (outw != w || (dims != 2 && outh != h))
        return false;

    if (source_scale_dims != 0)
        return source_width_scale == 1.0 && (dims == 2 || source_height_scale == 1.0);

    const bool has_output_size = dynamic_target_size || !size_expr.empty();
    return (output_width || has_output_size || width_scale == 1.f) && (dims == 2 || output_height || has_output_size || height_scale == 1.f);
}

int Interp_x86::eval_size_expr(const std::vector<Mat>& bottom_blobs, int& outw, int& outh) const
{
    if (resize_type != 4 || source_scale_dims == 0)
        return Interp::eval_size_expr(bottom_blobs, outw, outh);

    const double w = floor(bottom_blobs[0].w * source_width_scale);
    const double h = source_scale_dims == 1 ? bottom_blobs[0].h : floor(bottom_blobs[0].h * source_height_scale);
    if (!(w >= 1.0 && w <= INT_MAX && h >= 1.0 && h <= INT_MAX))
        return -1;

    outw = (int)w;
    outh = (int)h;
    return 0;
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
        if (eval_size_expr(bottom_blob_shapes, outw, outh) != 0 && resize_type == 4)
            return -1;
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

    if ((dims == 2 || dims == 3)
            && ((resize_type == 2 && (w == 1 || (dims == 3 && h == 1)))
                || (resize_type == 3 && (w < 4 || (dims == 3 && h < 4)))))
        return forward_small_input(bottom_blob, top_blob, outw, outh, opt);

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
    }
    else if (dims == 2)
    {
        if (is_identity_resize(w, h, outw, outh, dims))
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
        if (is_identity_resize(w, h, outw, outh, dims))
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
    }

    float hs = height_scale;
    float ws = width_scale;
    if (resize_type == 4)
    {
        hs = height_scale_inv > 0.f ? height_scale_inv : ((output_height || dynamic_target_size || !size_expr.empty()) ? h / (float)outh : 1.f / height_scale);
        ws = width_scale_inv > 0.f ? width_scale_inv : ((output_width || dynamic_target_size || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale);
    }

    interp_fp32(bottom_blob, top_blob, resize_type, align_corner, hs, ws, output_height, output_width, dynamic_target_size || !size_expr.empty(), opt);

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
        if (eval_size_expr(bottom_blob_shapes, outw, outh) != 0 && resize_type == 4)
            return -1;
    }

    if ((dims == 2 || dims == 3)
            && ((resize_type == 2 && (w == 1 || (dims == 3 && h == 1)))
                || (resize_type == 3 && (w < 4 || (dims == 3 && h < 4)))))
        return forward_small_input(bottom_blob, top_blob, outw, outh, opt);

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
    }
    else if (dims == 2)
    {
        if (is_identity_resize(w, h, outw, outh, dims))
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
        if (is_identity_resize(w, h, outw, outh, dims))
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
    }

    float hs = height_scale;
    float ws = width_scale;
    if (resize_type == 4)
    {
        hs = height_scale_inv > 0.f ? height_scale_inv : ((output_height || dynamic_target_size || !size_expr.empty()) ? h / (float)outh : 1.f / height_scale);
        ws = width_scale_inv > 0.f ? width_scale_inv : ((output_width || dynamic_target_size || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale);
    }

    interp_bf16s(bottom_blob, top_blob, resize_type, align_corner, hs, ws, output_height, output_width, dynamic_target_size || !size_expr.empty(), opt);

    return 0;
}

#endif // NCNN_BF16

} // namespace ncnn
