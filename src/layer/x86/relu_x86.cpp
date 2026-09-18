// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "relu_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

#include "cpu.h"

namespace ncnn {

#include "relu_fp32.h"

#if NCNN_BF16
#include "relu_bf16s.h"
#endif

ReLU_x86::ReLU_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int ReLU_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

    if (elembits == 8)
        return forward_inplace_int8(bottom_top_blob, opt);

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    relu_fp32(bottom_top_blob, slope, opt);

    return 0;
}

int ReLU_x86::forward_inplace_int8(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;
    int elempack = bottom_top_blob.elempack;

#if __SSE2__
    if (elempack == 8)
    {
        if (slope == 0.f)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                signed char* ptr = bottom_top_blob.channel(q);

                int i = 0;
                for (; i < size; i++)
                {
                    if (ptr[0] < 0)
                        ptr[0] = 0;
                    if (ptr[1] < 0)
                        ptr[1] = 0;
                    if (ptr[2] < 0)
                        ptr[2] = 0;
                    if (ptr[3] < 0)
                        ptr[3] = 0;
                    if (ptr[4] < 0)
                        ptr[4] = 0;
                    if (ptr[5] < 0)
                        ptr[5] = 0;
                    if (ptr[6] < 0)
                        ptr[6] = 0;
                    if (ptr[7] < 0)
                        ptr[7] = 0;

                    ptr += 8;
                }
            }
        }
        else
        {
            // TODO leakyrelu
        }

        return 0;
    }
#endif // __SSE2__

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            signed char* ptr = bottom_top_blob.channel(q);

            int i = 0;
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr = 0;

                ptr++;
            }
        }
    }
    else
    {
        // TODO leakyrelu
    }

    return 0;
}

#if NCNN_BF16
int ReLU_x86::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    relu_bf16s(bottom_top_blob, slope, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
