// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prelu_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__
#include "x86_activation.h"
#include "x86_usability.h"
#include "cpu.h"

namespace ncnn {

#include "prelu_fp32.h"

#if NCNN_BF16
#include "prelu_bf16s.h"
#endif

PReLU_x86::PReLU_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int PReLU_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    prelu_fp32(bottom_top_blob, slope_data, num_slope, opt);

    return 0;
}

#if NCNN_BF16
int PReLU_x86::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        unsigned short* ptr = bottom_top_blob;
        const int size = w * elempack;

        if (num_slope > 1)
        {
            prelu_bf16s_per_element(ptr, (const float*)slope_data, size, opt.num_threads);
        }
        else
        {
            prelu_bf16s_single_slope(ptr, slope_data[0], size, opt.num_threads);
        }
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);

            float slope = num_slope > 1 ? slope_data[i] : slope_data[0];
            const float* sptr = num_slope > 1 ? (const float*)slope_data + i * elempack : &slope;
            int ep = num_slope > 1 ? elempack : 1;

            prelu_bf16s(ptr, sptr, size, ep);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            float slope = num_slope > 1 ? slope_data[q] : slope_data[0];
            const float* sptr = num_slope > 1 ? (const float*)slope_data + q * elempack : &slope;
            int ep = num_slope > 1 ? elempack : 1;

            prelu_bf16s(ptr, sptr, size, ep);
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
