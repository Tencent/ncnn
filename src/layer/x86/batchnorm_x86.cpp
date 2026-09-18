// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "batchnorm_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"
#include "cpu.h"

namespace ncnn {

#include "batchnorm_fp32.h"

#if NCNN_BF16
#include "batchnorm_bf16s.h"
#endif

BatchNorm_x86::BatchNorm_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int BatchNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    batchnorm_fp32(bottom_top_blob, a_data, b_data, opt);

    return 0;
}

#if NCNN_BF16
int BatchNorm_x86::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int c = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        unsigned short* ptr = bottom_top_blob;
        const float* aptr = a_data;
        const float* bptr = b_data;

        const int size = w * elempack;

        batchnorm_bf16s_per_element(ptr, aptr, bptr, size, opt.num_threads);
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            const float* aptr = (const float*)a_data + i * elempack;
            const float* bptr = (const float*)b_data + i * elempack;

            batchnorm_bf16s(ptr, aptr, bptr, size, elempack);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            const float* aptr = (const float*)a_data + q * elempack;
            const float* bptr = (const float*)b_data + q * elempack;

            batchnorm_bf16s(ptr, aptr, bptr, size, elempack);
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
