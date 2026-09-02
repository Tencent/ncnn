// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "requantize_x86.h"

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

Requantize_x86::Requantize_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

#include "requantize_fp32.h"

int Requantize_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const size_t out_elemsize = elempack * 1u;

    if (dims == 1)
    {
        top_blob.create(w, out_elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;

            const int* intptr = (const int*)bottom_blob + i * elempack;
            signed char* ptr = (signed char*)top_blob + i * elempack;

            // assert scale_in_data_size == 1
            // assert bias_data_size == 0 || bias_data_size == 1
            // assert scale_out_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            requantize(intptr, ptr, scale_in_data, bias_data, scale_out_data, activation_type, activation_params, size, 1);
        }
    }

    if (dims == 2)
    {
        top_blob.create(w, h, out_elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const int* intptr = bottom_blob.row<const int>(i);
            signed char* ptr = top_blob.row<signed char>(i);

            const Mat scale_in_data_i = scale_in_data_size > 1 ? scale_in_data.range(i * elempack, elempack) : scale_in_data;
            const Mat bias_data_i = bias_data_size > 1 ? bias_data.range(i * elempack, elempack) : bias_data;
            const Mat scale_out_data_i = scale_out_data_size > 1 ? scale_out_data.range(i * elempack, elempack) : scale_out_data;

            requantize(intptr, ptr, scale_in_data_i, bias_data_i, scale_out_data_i, activation_type, activation_params, w, elempack);
        }
    }

    if (dims == 3 || dims == 4)
    {
        if (dims == 3)
            top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
        else
            top_blob.create(w, h, d, channels, out_elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
            signed char* ptr = top_blob.channel(q);

            const Mat scale_in_data_q = scale_in_data_size > 1 ? scale_in_data.range(q * elempack, elempack) : scale_in_data;
            const Mat bias_data_q = bias_data_size > 1 ? bias_data.range(q * elempack, elempack) : bias_data;
            const Mat scale_out_data_q = scale_out_data_size > 1 ? scale_out_data.range(q * elempack, elempack) : scale_out_data;

            requantize(intptr, ptr, scale_in_data_q, bias_data_q, scale_out_data_q, activation_type, activation_params, w * h * d, elempack);
        }
    }

    return 0;
}

} // namespace ncnn
