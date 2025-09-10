// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "quantize_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

Quantize_x86::Quantize_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

static void quantize(const float* ptr, signed char* s8ptr, const Mat& scale_data, int elemcount, int elempack)
{
    const int scale_data_size = scale_data.w;
    const int size = elemcount * elempack;

    // NCNN_LOGE("quantize %d   %d %d", scale_data_size, elemcount, elempack);

    float scale = scale_data[0];
#if __SSE2__
    __m128 _scale = _mm_set1_ps(scale);
#if __AVX__
    __m256 _scale_avx = _mm256_set1_ps(scale);
#if __AVX512F__
    __m512 _scale_avx512 = _mm512_set1_ps(scale);
#endif // __AVX512F__
#endif // __AVX__
    if (scale_data_size > 1)
    {
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            _scale_avx512 = _mm512_loadu_ps((const float*)scale_data);
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            _scale_avx = _mm256_loadu_ps((const float*)scale_data);
#if __AVX512F__
            _scale_avx512 = combine8x2_ps(_scale_avx, _scale_avx);
#endif // __AVX512F__
        }
#endif // __AVX__
        if (elempack == 4)
        {
            _scale = _mm_loadu_ps((const float*)scale_data);
#if __AVX__
            _scale_avx = combine4x2_ps(_scale, _scale);
#if __AVX512F__
            _scale_avx512 = combine8x2_ps(_scale_avx, _scale_avx);
#endif // __AVX512F__
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
        __m512 _v = _mm512_loadu_ps(ptr);
        _v = _mm512_mul_ps(_v, _scale_avx512);
        _mm_storeu_si128((__m128i*)s8ptr, float2int8_avx512(_v));
#else  // __AVX512F__
        __m256 _v0 = _mm256_loadu_ps(ptr);
        __m256 _v1 = _mm256_loadu_ps(ptr + 8);
        _v0 = _mm256_mul_ps(_v0, _scale_avx);
        _v1 = _mm256_mul_ps(_v1, _scale_avx);
        _mm_storeu_si128((__m128i*)s8ptr, float2int8_avx(_v0, _v1));
#endif // __AVX512F__
        ptr += 16;
        s8ptr += 16;
    }
#endif // __AVX__
    for (; i + 7 < size; i += 8)
    {
#if __AVX__
        __m256 _v = _mm256_loadu_ps(ptr);
        _v = _mm256_mul_ps(_v, _scale_avx);
        *(int64_t*)s8ptr = float2int8_avx(_v);
#else  // __AVX__
        __m128 _v0 = _mm_loadu_ps(ptr);
        __m128 _v1 = _mm_loadu_ps(ptr + 4);
        _v0 = _mm_mul_ps(_v0, _scale);
        _v1 = _mm_mul_ps(_v1, _scale);
        *(int64_t*)s8ptr = float2int8_sse(_v0, _v1);
#endif // __AVX__
        ptr += 8;
        s8ptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        __m128 _v = _mm_loadu_ps(ptr);
        _v = _mm_mul_ps(_v, _scale);
        int32_t v = float2int8_sse(_v);
        s8ptr[0] = (v >> 0) & 0xff;
        s8ptr[1] = (v >> 8) & 0xff;
        s8ptr[2] = (v >> 16) & 0xff;
        s8ptr[3] = (v >> 24) & 0xff;
        ptr += 4;
        s8ptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        float v = *ptr * scale;
        *s8ptr = float2int8(v);
        ptr++;
        s8ptr++;
    }
}

#if __SSE2__
#if __AVX512F__
static void quantize_pack16to8(const float* ptr, signed char* s8ptr0, signed char* s8ptr1, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;

    // NCNN_LOGE("quantize_pack16to8 %d   %d", scale_data_size, elemcount);

    float scale = scale_data[0];
    __m512 _scale = _mm512_set1_ps(scale);
    if (scale_data_size > 1)
    {
        _scale = _mm512_loadu_ps((const float*)scale_data);
    }

    int i = 0;
    for (; i < elemcount; i++)
    {
        __m512 _v = _mm512_loadu_ps(ptr);
        _v = _mm512_mul_ps(_v, _scale);
        __m128i v = float2int8_avx512(_v);
        _mm_storel_pd((double*)s8ptr0, _mm_castsi128_pd(v));
        _mm_storeh_pd((double*)s8ptr1, _mm_castsi128_pd(v));
        ptr += 16;
        s8ptr0 += 8;
        s8ptr1 += 8;
    }
}
#endif // __AVX512F__

#if !__AVX__
static void quantize_pack4to8(const float* ptr0, const float* ptr1, signed char* s8ptr, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;

    // NCNN_LOGE("quantize_pack4to8 %d   %d", scale_data_size, elemcount);

    float scale = scale_data[0];
    __m128 _scale0 = _mm_set1_ps(scale);
    __m128 _scale1 = _scale0;
    if (scale_data_size > 1)
    {
        _scale0 = _mm_loadu_ps((const float*)scale_data);
        _scale1 = _mm_loadu_ps((const float*)scale_data + 4);
    }

    int i = 0;
    for (; i + 1 < elemcount; i += 2)
    {
        __m128 _v0 = _mm_loadu_ps(ptr0);
        __m128 _v1 = _mm_loadu_ps(ptr1);
        __m128 _v2 = _mm_loadu_ps(ptr0 + 4);
        __m128 _v3 = _mm_loadu_ps(ptr1 + 4);
        _v0 = _mm_mul_ps(_v0, _scale0);
        _v1 = _mm_mul_ps(_v1, _scale1);
        _v2 = _mm_mul_ps(_v2, _scale0);
        _v3 = _mm_mul_ps(_v3, _scale1);
        _mm_storeu_si128((__m128i*)s8ptr, float2int8_sse(_v0, _v1, _v2, _v3));
        ptr0 += 8;
        ptr1 += 8;
        s8ptr += 16;
    }
    for (; i < elemcount; i++)
    {
        __m128 _v0 = _mm_loadu_ps(ptr0);
        __m128 _v1 = _mm_loadu_ps(ptr1);
        _v0 = _mm_mul_ps(_v0, _scale0);
        _v1 = _mm_mul_ps(_v1, _scale1);
        *(int64_t*)s8ptr = float2int8_sse(_v0, _v1);
        ptr0 += 4;
        ptr1 += 4;
        s8ptr += 8;
    }
}
#endif // !__AVX__

static void quantize_pack4to1(const float* ptr, signed char* s8ptr0, signed char* s8ptr1, signed char* s8ptr2, signed char* s8ptr3, const Mat& scale_data, int elemcount)
{
    const int scale_data_size = scale_data.w;

    // NCNN_LOGE("quantize_pack4to1 %d   %d", scale_data_size, elemcount);

    float scale = scale_data[0];
    __m128 _scale = _mm_set1_ps(scale);
    if (scale_data_size > 1)
    {
        _scale = _mm_loadu_ps((const float*)scale_data);
    }

    int i = 0;
    for (; i + 7 < elemcount; i += 8)
    {
        __m128 _v0 = _mm_loadu_ps(ptr);
        __m128 _v1 = _mm_loadu_ps(ptr + 4);
        __m128 _v2 = _mm_loadu_ps(ptr + 8);
        __m128 _v3 = _mm_loadu_ps(ptr + 12);
        __m128 _v4 = _mm_loadu_ps(ptr + 16);
        __m128 _v5 = _mm_loadu_ps(ptr + 20);
        __m128 _v6 = _mm_loadu_ps(ptr + 24);
        __m128 _v7 = _mm_loadu_ps(ptr + 28);
        _v0 = _mm_mul_ps(_v0, _scale);
        _v1 = _mm_mul_ps(_v1, _scale);
        _v2 = _mm_mul_ps(_v2, _scale);
        _v3 = _mm_mul_ps(_v3, _scale);
        _v4 = _mm_mul_ps(_v4, _scale);
        _v5 = _mm_mul_ps(_v5, _scale);
        _v6 = _mm_mul_ps(_v6, _scale);
        _v7 = _mm_mul_ps(_v7, _scale);
        __m128i v0426 = float2int8_sse(_v0, _v4, _v2, _v6);
        __m128i v1537 = float2int8_sse(_v1, _v5, _v3, _v7);
        __m128i v0145 = _mm_unpacklo_epi8(v0426, v1537);
        __m128i v2367 = _mm_unpackhi_epi8(v0426, v1537);
        __m128i v0123 = _mm_unpacklo_epi16(v0145, v2367);
        __m128i v4567 = _mm_unpackhi_epi16(v0145, v2367);
        __m128i v01 = _mm_unpacklo_epi32(v0123, v4567);
        __m128i v23 = _mm_unpackhi_epi32(v0123, v4567);
        _mm_storel_pd((double*)s8ptr0, _mm_castsi128_pd(v01));
        _mm_storeh_pd((double*)s8ptr1, _mm_castsi128_pd(v01));
        _mm_storel_pd((double*)s8ptr2, _mm_castsi128_pd(v23));
        _mm_storeh_pd((double*)s8ptr3, _mm_castsi128_pd(v23));
        ptr += 32;
        s8ptr0 += 8;
        s8ptr1 += 8;
        s8ptr2 += 8;
        s8ptr3 += 8;
    }
    for (; i < elemcount; i++)
    {
        __m128 _v = _mm_loadu_ps(ptr);
        _v = _mm_mul_ps(_v, _scale);
        int64_t v = float2int8_sse(_v, _v);
        s8ptr0[0] = (v >> 32) & 0xff;
        s8ptr1[0] = (v >> 40) & 0xff;
        s8ptr2[0] = (v >> 48) & 0xff;
        s8ptr3[0] = (v >> 56) & 0xff;
        ptr += 4;
        s8ptr0 += 1;
        s8ptr1 += 1;
        s8ptr2 += 1;
        s8ptr3 += 1;
    }
}
#endif // __SSE2__

int Quantize_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    if (dims == 1)
    {
        int out_elempack = 1;
#if __SSE2__
        if (opt.use_packing_layout)
        {
            out_elempack = w * elempack % 8 == 0 ? 8 : 1;
        }
#endif
        const int outw = w * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int wp = std::max(1, w / opt.num_threads);
        const int nn_w = (w + wp - 1) / wp;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_w; ii++)
        {
            const int i = ii * wp;

            const float* ptr = (const float*)bottom_blob + i * elempack;
            signed char* s8ptr = (signed char*)top_blob + i * elempack;

            // assert scale_data_size == 1

            const int size = std::min(w - i, wp) * elempack;

            quantize(ptr, s8ptr, scale_data, size, 1);
        }
    }

    if (dims == 2)
    {
        int out_elempack = 1;
#if __SSE2__
        if (opt.use_packing_layout)
        {
            out_elempack = h * elempack % 8 == 0 ? 8 : 1;
        }
#endif
        const int outh = h * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __SSE2__
#if __AVX512F__
        if (elempack == 16 && out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                signed char* s8ptr0 = top_blob.row<signed char>(i * 2);
                signed char* s8ptr1 = top_blob.row<signed char>(i * 2 + 1);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

                quantize_pack16to8(ptr, s8ptr0, s8ptr1, scale_data_i, w);
            }
        }
#endif // __AVX512F__
#if !__AVX__
        if (elempack == 4 && out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* ptr0 = bottom_blob.row(i * 2);
                const float* ptr1 = bottom_blob.row(i * 2 + 1);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * out_elempack, out_elempack) : scale_data;

                quantize_pack4to8(ptr0, ptr1, s8ptr, scale_data_i, w);
            }
        }
#endif // !__AVX__
        if (elempack == 4 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                signed char* s8ptr0 = top_blob.row<signed char>(i * 4);
                signed char* s8ptr1 = top_blob.row<signed char>(i * 4 + 1);
                signed char* s8ptr2 = top_blob.row<signed char>(i * 4 + 2);
                signed char* s8ptr3 = top_blob.row<signed char>(i * 4 + 3);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

                quantize_pack4to1(ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data_i, w);
            }
        }
#endif // __SSE2__
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                signed char* s8ptr = top_blob.row<signed char>(i);

                const Mat scale_data_i = scale_data_size > 1 ? scale_data.range(i * elempack, elempack) : scale_data;

                quantize(ptr, s8ptr, scale_data_i, w, elempack);
            }
        }
    }

    if (dims == 3)
    {
        int out_elempack = 1;
#if __SSE2__
        if (opt.use_packing_layout)
        {
            out_elempack = channels * elempack % 8 == 0 ? 8 : 1;
        }
#endif
        const int outc = channels * elempack / out_elempack;
        const size_t out_elemsize = out_elempack * 1u;

        top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __SSE2__
#if __AVX512F__
        if (elempack == 16 && out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                signed char* s8ptr0 = top_blob.channel(q * 2);
                signed char* s8ptr1 = top_blob.channel(q * 2 + 1);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize_pack16to8(ptr, s8ptr0, s8ptr1, scale_data_q, w * h);
            }
        }
#endif // __AVX512F__
#if !__AVX__
        if (elempack == 4 && out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* ptr0 = bottom_blob.channel(q * 2);
                const float* ptr1 = bottom_blob.channel(q * 2 + 1);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * out_elempack, out_elempack) : scale_data;

                quantize_pack4to8(ptr0, ptr1, s8ptr, scale_data_q, w * h);
            }
        }
#endif // !__AVX__
        if (elempack == 4 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                signed char* s8ptr0 = top_blob.channel(q * 4);
                signed char* s8ptr1 = top_blob.channel(q * 4 + 1);
                signed char* s8ptr2 = top_blob.channel(q * 4 + 2);
                signed char* s8ptr3 = top_blob.channel(q * 4 + 3);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize_pack4to1(ptr, s8ptr0, s8ptr1, s8ptr2, s8ptr3, scale_data_q, w * h);
            }
        }
#endif // __SSE2__
        if (elempack == out_elempack)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                signed char* s8ptr = top_blob.channel(q);

                const Mat scale_data_q = scale_data_size > 1 ? scale_data.range(q * elempack, elempack) : scale_data;

                quantize(ptr, s8ptr, scale_data_q, w * h, elempack);
            }
        }
    }

    return 0;
}

} // namespace ncnn
