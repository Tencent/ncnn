// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "scale_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"
#include "cpu.h"

namespace ncnn {

#include "scale_fp32.h"

#if NCNN_BF16
#include "scale_bf16s.h"
#endif

Scale_x86::Scale_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Scale_x86::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blobs[0].elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blobs, opt);
#endif

    scale_fp32(bottom_top_blobs, bias_term, bias_data, opt);

    return 0;
}

#if NCNN_BF16
int Scale_x86::forward_inplace_bf16s(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;

    // scale_blob may be bf16 (from second input) or fp32 (from scale_data weight)
    const float* scale = 0;
    Mat scale_fp32;
    if (scale_blob.elembits() == 16)
    {
        const int scale_data_size = scale_blob.w * scale_blob.elempack;
        scale_fp32.create(scale_data_size, 4u, 1, opt.workspace_allocator);
        if (scale_fp32.empty())
            return -100;
        const unsigned short* src = scale_blob;
        float* dst = scale_fp32;
        for (int i = 0; i < scale_data_size; i++)
        {
            dst[i] = bfloat16_to_float32(src[i]);
        }
        scale = scale_fp32;
    }
    else
    {
        scale = scale_blob;
    }
    const float* bias = bias_data;

    if (dims == 1)
    {
        unsigned short* ptr = (unsigned short*)bottom_top_blob;
        const int size = w * elempack;

        if (bias_term)
        {
            scale_bf16s_per_element(ptr, scale, bias, size, opt.num_threads);
        }
        else
        {
            scale_bf16s_no_bias_per_element(ptr, scale, size, opt.num_threads);
        }
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            const float* sptr = scale + i * elempack;

            if (bias_term)
            {
                const float* bptr = bias + i * elempack;
                scale_bf16s(ptr, sptr, bptr, size, elempack);
            }
            else
            {
                scale_bf16s_no_bias(ptr, sptr, size, elempack);
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            const float* sptr = scale + q * elempack;

            if (bias_term)
            {
                const float* bptr = bias + q * elempack;
                scale_bf16s(ptr, sptr, bptr, size, elempack);
            }
            else
            {
                scale_bf16s_no_bias(ptr, sptr, size, elempack);
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
