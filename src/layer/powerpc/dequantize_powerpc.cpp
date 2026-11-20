// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "dequantize_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

#include <algorithm>

namespace ncnn {

Dequantize_powerpc::Dequantize_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
}

static void dequantize(const int* intptr, float* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int bias_data_size = bias_data.w;
    const int size = elemcount * elempack;

    const float scale0 = scale_data[0];

    int i = 0;
#if __ALTIVEC__
    // vector path requires the scale / bias layout to be expressible as lane vectors
    const bool vec_ok = (scale_data_size == 1 || elempack == 4 || elempack == 8)
                        && (bias_data_size == 0 || bias_data_size == 1 || elempack == 4 || elempack == 8);

    v4f32 _scale0 = vec_splats(scale0);
    v4f32 _scale1 = vec_splats(scale0);
    if (scale_data_size > 1 && elempack == 4)
    {
        _scale0 = __ncnn_altivec_load(scale_data);
        _scale1 = _scale0;
    }
    if (scale_data_size > 1 && elempack == 8)
    {
        _scale0 = __ncnn_altivec_load(scale_data);
        _scale1 = __ncnn_altivec_load((const float*)scale_data + 4);
    }

    if (vec_ok && bias_data_size == 0)
    {
        for (; i + 7 < size; i += 8)
        {
            v4f32 _v0 = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr));
            v4f32 _v1 = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr + 4));
            _v0 = __ncnn_altivec_mul(_v0, _scale0);
            _v1 = __ncnn_altivec_mul(_v1, _scale1);
            __ncnn_altivec_store(ptr, _v0);
            __ncnn_altivec_store(ptr + 4, _v1);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _v = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr));
            _v = __ncnn_altivec_mul(_v, _scale0);
            __ncnn_altivec_store(ptr, _v);
            intptr += 4;
            ptr += 4;
        }
    }
    else if (vec_ok)
    {
        const float bias0 = bias_data[0];
        v4f32 _bias0 = vec_splats(bias0);
        v4f32 _bias1 = vec_splats(bias0);
        if (bias_data_size > 1 && elempack == 4)
        {
            _bias0 = __ncnn_altivec_load(bias_data);
            _bias1 = _bias0;
        }
        if (bias_data_size > 1 && elempack == 8)
        {
            _bias0 = __ncnn_altivec_load(bias_data);
            _bias1 = __ncnn_altivec_load((const float*)bias_data + 4);
        }

        for (; i + 7 < size; i += 8)
        {
            v4f32 _v0 = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr));
            v4f32 _v1 = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr + 4));
            _v0 = __ncnn_altivec_fmadd(_bias0, _v0, _scale0);
            _v1 = __ncnn_altivec_fmadd(_bias1, _v1, _scale1);
            __ncnn_altivec_store(ptr, _v0);
            __ncnn_altivec_store(ptr + 4, _v1);
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            v4f32 _v = __ncnn_altivec_ffint_s((v4i32)__ncnn_altivec_load((const float*)intptr));
            _v = __ncnn_altivec_fmadd(_bias0, _v, _scale0);
            __ncnn_altivec_store(ptr, _v);
            intptr += 4;
            ptr += 4;
        }
    }
#endif // __ALTIVEC__

    for (; i < size; i++)
    {
        const float scale = scale_data_size == 1 ? scale0 : scale_data[i % elempack];
        const float bias = bias_data_size == 0 ? 0.f : bias_data_size == 1 ? bias_data[0] : bias_data[i % elempack];
        *ptr = *intptr * scale + bias;
        intptr++;
        ptr++;
    }
}

int Dequantize_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // assert bottom_blob.elembits() == 32

    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (dims == 1)
    {
        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;

            const int* intptr = (const int*)bottom_blob + i * elempack;
            float* ptr = (float*)top_blob + i * elempack;

            // assert scale_data_size == 1
            // assert bias_data_size == 0 || bias_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            dequantize(intptr, ptr, scale_data, bias_data, size, 1);
        }
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const int* intptr = bottom_blob.row<const int>(i);
            float* ptr = top_blob.row(i);

            const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;
            const Mat bias_data_i = bias_data_size > 1 ? bias_data.range(i * elempack, elempack) : bias_data;

            dequantize(intptr, ptr, scale_data_i, bias_data_i, w, elempack);
        }
    }

    if (dims == 3 || dims == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
            float* ptr = top_blob.channel(q);

            const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;
            const Mat bias_data_q = bias_data_size > 1 ? bias_data.range(q * elempack, elempack) : bias_data;

            dequantize(intptr, ptr, scale_data_q, bias_data_q, w * h * d, elempack);
        }
    }

    return 0;
}

} // namespace ncnn
