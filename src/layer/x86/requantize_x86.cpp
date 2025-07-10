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

namespace ncnn {

Requantize_x86::Requantize_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

static void requantize(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, int activation_type, const Mat& activation_params, int elemcount, int elempack)
{
    const int scale_in_data_size = scale_in_data.w;
    const int bias_data_size = bias_data.w;
    const int scale_out_data_size = scale_out_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("requantize %d %d %d   %d %d", scale_in_data_size, bias_data_size, scale_out_data_size, elemcount, elempack);

    float scale_in = scale_in_data[0];
#if __SSE2__
    __m128 _scale_in0 = _mm_set1_ps(scale_in);
#if __AVX__
    __m256 _scale_in_avx = _mm256_set1_ps(scale_in);
#if __AVX512F__
    __m512 _scale_in_avx512 = _mm512_set1_ps(scale_in);
#endif // __AVX512F__
#else  // __AVX__
    __m128 _scale_in1 = _scale_in0;
#endif // __AVX__
    if (scale_in_data_size > 1)
    {
#if __AVX512F__
        if (elempack == 16)
        {
            _scale_in_avx512 = _mm512_loadu_ps((const float*)scale_in_data);
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
#if __AVX__
            _scale_in_avx = _mm256_loadu_ps((const float*)scale_in_data);
#if __AVX512F__
            _scale_in_avx512 = combine8x2_ps(_scale_in_avx, _scale_in_avx);
#endif // __AVX512F__
#else  // __AVX__
            _scale_in0 = _mm_loadu_ps((const float*)scale_in_data);
            _scale_in1 = _mm_loadu_ps((const float*)scale_in_data + 4);
#endif // __AVX__
        }
    }
#endif // __SSE2__

    float scale_out = scale_out_data[0];
#if __SSE2__
    __m128 _scale_out0 = _mm_set1_ps(scale_out);
#if __AVX__
    __m256 _scale_out_avx = _mm256_set1_ps(scale_out);
#if __AVX512F__
    __m512 _scale_out_avx512 = _mm512_set1_ps(scale_out);
#endif // __AVX512F__
#else  // __AVX__
    __m128 _scale_out1 = _scale_out0;
#endif // __AVX__
    if (scale_out_data_size > 1)
    {
#if __AVX512F__
        if (elempack == 16)
        {
            _scale_out_avx512 = _mm512_loadu_ps((const float*)scale_out_data);
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
#if __AVX__
            _scale_out_avx = _mm256_loadu_ps((const float*)scale_out_data);
#if __AVX512F__
            _scale_out_avx512 = combine8x2_ps(_scale_out_avx, _scale_out_avx);
#endif // __AVX512F__
#else  // __AVX__
            _scale_out0 = _mm_loadu_ps((const float*)scale_out_data);
            _scale_out1 = _mm_loadu_ps((const float*)scale_out_data + 4);
#endif // __AVX__
        }
    }
#endif // __SSE2__

    if (bias_data_size == 0)
    {
        int i = 0;
#if __SSE2__
#if __AVX__
        for (; i + 15 < size; i += 16)
        {
#if __AVX512F__
            __m512 _v = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)intptr));
            _v = _mm512_mul_ps(_v, _scale_in_avx512);
            _v = activation_avx512(_v, activation_type, activation_params);
            _v = _mm512_mul_ps(_v, _scale_out_avx512);
            _mm_storeu_si128((__m128i*)ptr, float2int8_avx512(_v));
#else  // __AVX512F__
            __m256 _v0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
            __m256 _v1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(intptr + 8)));
            _v0 = _mm256_mul_ps(_v0, _scale_in_avx);
            _v1 = _mm256_mul_ps(_v1, _scale_in_avx);
            _v0 = activation_avx(_v0, activation_type, activation_params);
            _v1 = activation_avx(_v1, activation_type, activation_params);
            _v0 = _mm256_mul_ps(_v0, _scale_out_avx);
            _v1 = _mm256_mul_ps(_v1, _scale_out_avx);
            _mm_storeu_si128((__m128i*)ptr, float2int8_avx(_v0, _v1));
#endif // __AVX512F__
            intptr += 16;
            ptr += 16;
        }
#endif // __AVX__
        for (; i + 7 < size; i += 8)
        {
#if __AVX__
            __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
            _v = _mm256_mul_ps(_v, _scale_in_avx);
            _v = activation_avx(_v, activation_type, activation_params);
            _v = _mm256_mul_ps(_v, _scale_out_avx);
            *(int64_t*)ptr = float2int8_avx(_v);
#else  // __AVX__
            __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)(intptr + 4)));
            _v0 = _mm_mul_ps(_v0, _scale_in0);
            _v1 = _mm_mul_ps(_v1, _scale_in1);
            _v0 = activation_sse(_v0, activation_type, activation_params);
            _v1 = activation_sse(_v1, activation_type, activation_params);
            _v0 = _mm_mul_ps(_v0, _scale_out0);
            _v1 = _mm_mul_ps(_v1, _scale_out1);
            *(int64_t*)ptr = float2int8_sse(_v0, _v1);
#endif // __AVX__
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            _v = _mm_mul_ps(_v, _scale_in0);
            _v = activation_sse(_v, activation_type, activation_params);
            _v = _mm_mul_ps(_v, _scale_out0);
            int32_t v = float2int8_sse(_v);
            ptr[0] = (v >> 0) & 0xff;
            ptr[1] = (v >> 8) & 0xff;
            ptr[2] = (v >> 16) & 0xff;
            ptr[3] = (v >> 24) & 0xff;
            intptr += 4;
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            float v = *intptr * scale_in;
            v = activation_ss(v, activation_type, activation_params);
            *ptr = float2int8(v * scale_out);
            intptr++;
            ptr++;
        }
    }
    else
    {
        float bias = bias_data[0];
#if __SSE2__
        __m128 _bias0 = _mm_set1_ps(bias);
#if __AVX__
        __m256 _bias_avx = _mm256_set1_ps(bias);
#if __AVX512F__
        __m512 _bias_avx512 = _mm512_set1_ps(bias);
#endif // __AVX512F__
#else  // __AVX__
        __m128 _bias1 = _bias0;
#endif // __AVX__
        if (bias_data_size > 1)
        {
#if __AVX512F__
            if (elempack == 16)
            {
                _bias_avx512 = _mm512_loadu_ps((const float*)bias_data);
            }
#endif // __AVX512F__
            if (elempack == 8)
            {
#if __AVX__
                _bias_avx = _mm256_loadu_ps((const float*)bias_data);
#if __AVX512F__
                _bias_avx512 = combine8x2_ps(_bias_avx, _bias_avx);
#endif // __AVX512F__
#else  // __AVX__
                _bias0 = _mm_loadu_ps((const float*)bias_data);
                _bias1 = _mm_loadu_ps((const float*)bias_data + 4);
#endif // __AVX__
            }
        }
#endif // __SSE2__

        int i = 0;
#if __SSE2__
#if __AVX__
        for (; i + 15 < size; i += 16)
        {
#if __AVX512F__
            __m512 _v = _mm512_cvtepi32_ps(_mm512_loadu_si512((const __m512i*)intptr));
            _v = _mm512_fmadd_ps(_v, _scale_in_avx512, _bias_avx512);
            _v = activation_avx512(_v, activation_type, activation_params);
            _v = _mm512_mul_ps(_v, _scale_out_avx512);
            _mm_storeu_si128((__m128i*)ptr, float2int8_avx512(_v));
#else  // __AVX512F__
            __m256 _v0 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
            __m256 _v1 = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(intptr + 8)));
            _v0 = _mm256_comp_fmadd_ps(_v0, _scale_in_avx, _bias_avx);
            _v1 = _mm256_comp_fmadd_ps(_v1, _scale_in_avx, _bias_avx);
            _v0 = activation_avx(_v0, activation_type, activation_params);
            _v1 = activation_avx(_v1, activation_type, activation_params);
            _v0 = _mm256_mul_ps(_v0, _scale_out_avx);
            _v1 = _mm256_mul_ps(_v1, _scale_out_avx);
            _mm_storeu_si128((__m128i*)ptr, float2int8_avx(_v0, _v1));
#endif // __AVX512F__
            intptr += 16;
            ptr += 16;
        }
#endif // __AVX__
        for (; i + 7 < size; i += 8)
        {
#if __AVX__
            __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
            _v = _mm256_comp_fmadd_ps(_v, _scale_in_avx, _bias_avx);
            _v = activation_avx(_v, activation_type, activation_params);
            _v = _mm256_mul_ps(_v, _scale_out_avx);
            *(int64_t*)ptr = float2int8_avx(_v);
#else  // __AVX__
            __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)(intptr + 4)));
            _v0 = _mm_comp_fmadd_ps(_v0, _scale_in0, _bias0);
            _v1 = _mm_comp_fmadd_ps(_v1, _scale_in1, _bias1);
            _v0 = activation_sse(_v0, activation_type, activation_params);
            _v1 = activation_sse(_v1, activation_type, activation_params);
            _v0 = _mm_mul_ps(_v0, _scale_out0);
            _v1 = _mm_mul_ps(_v1, _scale_out1);
            *(int64_t*)ptr = float2int8_sse(_v0, _v1);
#endif // __AVX__
            intptr += 8;
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
            _v = _mm_comp_fmadd_ps(_v, _scale_in0, _bias0);
            _v = activation_sse(_v, activation_type, activation_params);
            _v = _mm_mul_ps(_v, _scale_out0);
            int32_t v = float2int8_sse(_v);
            ptr[0] = (v >> 0) & 0xff;
            ptr[1] = (v >> 8) & 0xff;
            ptr[2] = (v >> 16) & 0xff;
            ptr[3] = (v >> 24) & 0xff;
            intptr += 4;
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            float v = *intptr * scale_in + bias;
            v = activation_ss(v, activation_type, activation_params);
            *ptr = float2int8(v * scale_out);
            intptr++;
            ptr++;
        }
    }
}

int Requantize_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
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

    if (dims == 3)
    {
        top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
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

            requantize(intptr, ptr, scale_in_data_q, bias_data_q, scale_out_data_q, activation_type, activation_params, w * h, elempack);
        }
    }

    return 0;
}

} // namespace ncnn
