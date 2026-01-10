// Copyright 2025 pchar.cn
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#if __AVX512F__
#include <immintrin.h>
#endif // __AVX512F__
#endif // __SSE2__

namespace ncnn {

RotaryEmbed_x86::RotaryEmbed_x86()
{
}

int RotaryEmbed_x86::forward(const std::vector<Mat>& bottom_blobs,
                             std::vector<Mat>& top_blobs,
                             const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& cos_cache = bottom_blobs[1];
    const Mat& sin_cache = bottom_blobs[2];

    const int embed_dim = bottom_blob.w;
    const int seqlen = bottom_blob.h;
    const int num_heads = bottom_blob.c;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat head = bottom_blob.channel(q);
        Mat out_head = top_blob.channel(q);

        for (int i = 0; i < seqlen; i++)
        {
            if (interleaved)
            {
                const float* ptr = head.row(i);
                const float* cos_ptr = cos_cache.row(i);
                const float* sin_ptr = sin_cache.row(i);
                float* outptr = out_head.row(i);

                int j = 0;

#if __SSE2__
#if __AVX512F__
                {
                    const __m512 signmask512 = _mm512_castsi512_ps(_mm512_set_epi32(
                        0, (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000,
                        0, (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000));

                    const __m512i dupidx = _mm512_set_epi32(
                        7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);

                    for (; j + 7 < embed_dim / 2; j += 8)
                    {
                        __m512 a = _mm512_loadu_ps(ptr);

                        __m256 c8 = _mm256_loadu_ps(cos_ptr);
                        __m256 s8 = _mm256_loadu_ps(sin_ptr);

                        __m512 csrc = _mm512_castps256_ps512(c8);
                        __m512 ssrc = _mm512_castps256_ps512(s8);

                        __m512 c = _mm512_permutexvar_ps(dupidx, csrc);
                        __m512 s = _mm512_permutexvar_ps(dupidx, ssrc);

                        __m512 ac = _mm512_mul_ps(a, c);

                        __m512 swap = _mm512_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1));
                        __m512 ss = _mm512_mul_ps(swap, s);

                        ss = _mm512_xor_ps(ss, signmask512);

                        __m512 y = _mm512_add_ps(ac, ss);
                        _mm512_storeu_ps(outptr, y);

                        ptr += 16;
                        outptr += 16;
                        cos_ptr += 8;
                        sin_ptr += 8;
                    }
                }
#endif // __AVX512F__

#if __AVX2__
                {
                    const __m256 signmask256 = _mm256_castsi256_ps(_mm256_set_epi32(
                        0, (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000));

                    const __m256i dupidx256 = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);

                    for (; j + 3 < embed_dim / 2; j += 4)
                    {
                        __m256 a = _mm256_loadu_ps(ptr);

                        __m128 c4 = _mm_loadu_ps(cos_ptr);
                        __m128 s4 = _mm_loadu_ps(sin_ptr);

                        __m256 csrc = _mm256_castps128_ps256(c4);
                        __m256 ssrc = _mm256_castps128_ps256(s4);

                        __m256 c = _mm256_permutevar8x32_ps(csrc, dupidx256);
                        __m256 s = _mm256_permutevar8x32_ps(ssrc, dupidx256);

                        __m256 ac = _mm256_mul_ps(a, c);

                        __m256 swap = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1));
                        __m256 ss = _mm256_mul_ps(swap, s);

                        ss = _mm256_xor_ps(ss, signmask256);

                        __m256 y = _mm256_add_ps(ac, ss);
                        _mm256_storeu_ps(outptr, y);

                        ptr += 8;
                        outptr += 8;
                        cos_ptr += 4;
                        sin_ptr += 4;
                    }
                }
#elif __AVX__
                {
                    const __m256 signmask256 = _mm256_castsi256_ps(_mm256_set_epi32(
                        0, (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000, 0, (int)0x80000000));

                    for (; j + 3 < embed_dim / 2; j += 4)
                    {
                        __m256 a = _mm256_loadu_ps(ptr);

                        __m128 c4 = _mm_loadu_ps(cos_ptr);
                        __m128 s4 = _mm_loadu_ps(sin_ptr);

                        __m128 clo = _mm_unpacklo_ps(c4, c4); // [c0,c0,c1,c1]
                        __m128 chi = _mm_unpackhi_ps(c4, c4); // [c2,c2,c3,c3]
                        __m128 slo = _mm_unpacklo_ps(s4, s4); // [s0,s0,s1,s1]
                        __m128 shi = _mm_unpackhi_ps(s4, s4); // [s2,s2,s3,s3]

                        __m256 c = _mm256_castps128_ps256(clo);
                        c = _mm256_insertf128_ps(c, chi, 1);

                        __m256 s = _mm256_castps128_ps256(slo);
                        s = _mm256_insertf128_ps(s, shi, 1);

                        __m256 ac = _mm256_mul_ps(a, c);

                        __m256 swap = _mm256_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1));
                        __m256 ss = _mm256_mul_ps(swap, s);

                        ss = _mm256_xor_ps(ss, signmask256);

                        __m256 y = _mm256_add_ps(ac, ss);
                        _mm256_storeu_ps(outptr, y);

                        ptr += 8;
                        outptr += 8;
                        cos_ptr += 4;
                        sin_ptr += 4;
                    }
                }
#endif // __AVX__

                {
                    const __m128 signmask128 = _mm_castsi128_ps(_mm_set_epi32(
                        0, (int)0x80000000, 0, (int)0x80000000));

                    for (; j + 1 < embed_dim / 2; j += 2)
                    {
                        __m128 a = _mm_loadu_ps(ptr);

                        __m128 c01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)cos_ptr));
                        __m128 s01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)sin_ptr));

                        __m128 c = _mm_unpacklo_ps(c01, c01); // [c0,c0,c1,c1]
                        __m128 s = _mm_unpacklo_ps(s01, s01); // [s0,s0,s1,s1]

                        __m128 ac = _mm_mul_ps(a, c);

                        __m128 swap = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1));
                        __m128 ss = _mm_mul_ps(swap, s);

                        ss = _mm_xor_ps(ss, signmask128);

                        __m128 y = _mm_add_ps(ac, ss);
                        _mm_storeu_ps(outptr, y);

                        ptr += 4;
                        outptr += 4;
                        cos_ptr += 2;
                        sin_ptr += 2;
                    }
                }
#endif // __SSE2__

                for (; j < embed_dim / 2; j++)
                {
                    const float x0 = ptr[0];
                    const float x1 = ptr[1];
                    const float cos_val = *cos_ptr++;
                    const float sin_val = *sin_ptr++;

                    outptr[0] = x0 * cos_val - x1 * sin_val;
                    outptr[1] = x0 * sin_val + x1 * cos_val;

                    ptr += 2;
                    outptr += 2;
                }
            }
            else
            {
                const float* ptr0 = head.row(i);
                const float* ptr1 = ptr0 + embed_dim / 2;
                const float* cos_ptr = cos_cache.row(i);
                const float* sin_ptr = sin_cache.row(i);

                float* outptr0 = out_head.row(i);
                float* outptr1 = outptr0 + embed_dim / 2;

                int j = 0;

#if __SSE2__
#if __AVX512F__
                for (; j + 15 < embed_dim / 2; j += 16)
                {
                    __m512 x0 = _mm512_loadu_ps(ptr0);
                    __m512 x1 = _mm512_loadu_ps(ptr1);
                    __m512 c = _mm512_loadu_ps(cos_ptr);
                    __m512 s = _mm512_loadu_ps(sin_ptr);

                    __m512 y0 = _mm512_sub_ps(_mm512_mul_ps(x0, c), _mm512_mul_ps(x1, s));
                    __m512 y1 = _mm512_add_ps(_mm512_mul_ps(x0, s), _mm512_mul_ps(x1, c));

                    _mm512_storeu_ps(outptr0, y0);
                    _mm512_storeu_ps(outptr1, y1);

                    ptr0 += 16;
                    ptr1 += 16;
                    cos_ptr += 16;
                    sin_ptr += 16;
                    outptr0 += 16;
                    outptr1 += 16;
                }
#elif __AVX__
                for (; j + 7 < embed_dim / 2; j += 8)
                {
                    __m256 x0 = _mm256_loadu_ps(ptr0);
                    __m256 x1 = _mm256_loadu_ps(ptr1);
                    __m256 c = _mm256_loadu_ps(cos_ptr);
                    __m256 s = _mm256_loadu_ps(sin_ptr);

                    __m256 y0 = _mm256_sub_ps(_mm256_mul_ps(x0, c), _mm256_mul_ps(x1, s));
                    __m256 y1 = _mm256_add_ps(_mm256_mul_ps(x0, s), _mm256_mul_ps(x1, c));

                    _mm256_storeu_ps(outptr0, y0);
                    _mm256_storeu_ps(outptr1, y1);

                    ptr0 += 8;
                    ptr1 += 8;
                    cos_ptr += 8;
                    sin_ptr += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                }
#endif // __AVX__

                for (; j + 3 < embed_dim / 2; j += 4)
                {
                    __m128 x0 = _mm_loadu_ps(ptr0);
                    __m128 x1 = _mm_loadu_ps(ptr1);
                    __m128 c = _mm_loadu_ps(cos_ptr);
                    __m128 s = _mm_loadu_ps(sin_ptr);

                    __m128 y0 = _mm_sub_ps(_mm_mul_ps(x0, c), _mm_mul_ps(x1, s));
                    __m128 y1 = _mm_add_ps(_mm_mul_ps(x0, s), _mm_mul_ps(x1, c));

                    _mm_storeu_ps(outptr0, y0);
                    _mm_storeu_ps(outptr1, y1);

                    ptr0 += 4;
                    ptr1 += 4;
                    cos_ptr += 4;
                    sin_ptr += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                }
#endif // __SSE2__

                for (; j < embed_dim / 2; j++)
                {
                    const float x0 = *ptr0++;
                    const float x1 = *ptr1++;
                    const float cos_val = *cos_ptr++;
                    const float sin_val = *sin_ptr++;

                    *outptr0++ = x0 * cos_val - x1 * sin_val;
                    *outptr1++ = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
