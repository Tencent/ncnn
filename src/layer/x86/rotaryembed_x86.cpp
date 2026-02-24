// Copyright 2025 pchar.cn
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __SSE3__
#include <pmmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE3__
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

RotaryEmbed_x86::RotaryEmbed_x86()
{
}

int RotaryEmbed_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
#if __AVX__
#if __AVX512F__
                const __m512i dupidx = _mm512_set_epi32(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8);
                const __m512i dupidx_lo = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
                for (; j + 15 < embed_dim / 2; j += 16)
                {
                    __m512 a0 = _mm512_loadu_ps(ptr);
                    __m512 a1 = _mm512_loadu_ps(ptr + 16);

                    __m512 cs_src = _mm512_loadu_ps(cos_ptr);
                    __m512 ss_src = _mm512_loadu_ps(sin_ptr);

                    __m512 c0 = _mm512_permutexvar_ps(dupidx_lo, cs_src);
                    __m512 c1 = _mm512_permutexvar_ps(dupidx, cs_src);
                    __m512 s0 = _mm512_permutexvar_ps(dupidx_lo, ss_src);
                    __m512 s1 = _mm512_permutexvar_ps(dupidx, ss_src);

                    __m512 swap0 = _mm512_shuffle_ps(a0, a0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m512 swap1 = _mm512_shuffle_ps(a1, a1, _MM_SHUFFLE(2, 3, 0, 1));

                    __m512 ss0 = _mm512_mul_ps(swap0, s0);
                    __m512 ss1 = _mm512_mul_ps(swap1, s1);

                    __m512 y0 = _mm512_fmaddsub_ps(a0, c0, ss0);
                    __m512 y1 = _mm512_fmaddsub_ps(a1, c1, ss1);

                    _mm512_storeu_ps(outptr, y0);
                    _mm512_storeu_ps(outptr + 16, y1);

                    ptr += 32;
                    outptr += 32;
                    cos_ptr += 16;
                    sin_ptr += 16;
                }
#endif // __AVX512F__
#if __AVX2__
                const __m256i dupidx256 = _mm256_set_epi32(7, 7, 6, 6, 5, 5, 4, 4);
                const __m256i dupidx256_lo = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
                for (; j + 7 < embed_dim / 2; j += 8)
                {
                    __m256 a0 = _mm256_loadu_ps(ptr);
                    __m256 a1 = _mm256_loadu_ps(ptr + 8);

                    __m256 c_src = _mm256_loadu_ps(cos_ptr);
                    __m256 s_src = _mm256_loadu_ps(sin_ptr);

                    __m256 c0 = _mm256_permutevar8x32_ps(c_src, dupidx256_lo);
                    __m256 c1 = _mm256_permutevar8x32_ps(c_src, dupidx256);
                    __m256 s0 = _mm256_permutevar8x32_ps(s_src, dupidx256_lo);
                    __m256 s1 = _mm256_permutevar8x32_ps(s_src, dupidx256);

                    __m256 swap0 = _mm256_shuffle_ps(a0, a0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m256 swap1 = _mm256_shuffle_ps(a1, a1, _MM_SHUFFLE(2, 3, 0, 1));

                    __m256 ss0 = _mm256_mul_ps(swap0, s0);
                    __m256 ss1 = _mm256_mul_ps(swap1, s1);

                    __m256 y0 = _mm256_fmaddsub_ps(a0, c0, ss0);
                    __m256 y1 = _mm256_fmaddsub_ps(a1, c1, ss1);

                    _mm256_storeu_ps(outptr, y0);
                    _mm256_storeu_ps(outptr + 8, y1);

                    ptr += 16;
                    outptr += 16;
                    cos_ptr += 8;
                    sin_ptr += 8;
                }
#else // __AVX2__
                for (; j + 7 < embed_dim / 2; j += 8)
                {
                    __m256 a0 = _mm256_loadu_ps(ptr);
                    __m256 a1 = _mm256_loadu_ps(ptr + 8);

                    __m128 clo4 = _mm_loadu_ps(cos_ptr);
                    __m128 chi4 = _mm_loadu_ps(cos_ptr + 4);
                    __m128 slo4 = _mm_loadu_ps(sin_ptr);
                    __m128 shi4 = _mm_loadu_ps(sin_ptr + 4);

                    __m128 clo_lo = _mm_unpacklo_ps(clo4, clo4); // [c0,c0,c1,c1]
                    __m128 clo_hi = _mm_unpackhi_ps(clo4, clo4); // [c2,c2,c3,c3]
                    __m128 chi_lo = _mm_unpacklo_ps(chi4, chi4); // [c4,c4,c5,c5]
                    __m128 chi_hi = _mm_unpackhi_ps(chi4, chi4); // [c6,c6,c7,c7]

                    __m256 c0 = combine4x2_ps(clo_lo, clo_hi);
                    __m256 c1 = combine4x2_ps(chi_lo, chi_hi);

                    __m128 slo_lo = _mm_unpacklo_ps(slo4, slo4); // [s0,s0,s1,s1]
                    __m128 slo_hi = _mm_unpackhi_ps(slo4, slo4); // [s2,s2,s3,s3]
                    __m128 shi_lo = _mm_unpacklo_ps(shi4, shi4); // [s4,s4,s5,s5]
                    __m128 shi_hi = _mm_unpackhi_ps(shi4, shi4); // [s6,s6,s7,s7]

                    __m256 s0 = combine4x2_ps(slo_lo, slo_hi);
                    __m256 s1 = combine4x2_ps(shi_lo, shi_hi);

                    __m256 swap0 = _mm256_shuffle_ps(a0, a0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m256 swap1 = _mm256_shuffle_ps(a1, a1, _MM_SHUFFLE(2, 3, 0, 1));

                    __m256 ss0 = _mm256_mul_ps(swap0, s0);
                    __m256 ss1 = _mm256_mul_ps(swap1, s1);

#if __FMA__
                    __m256 y0 = _mm256_fmaddsub_ps(a0, c0, ss0);
                    __m256 y1 = _mm256_fmaddsub_ps(a1, c1, ss1);
#else
                    __m256 ac0 = _mm256_mul_ps(a0, c0);
                    __m256 ac1 = _mm256_mul_ps(a1, c1);

                    __m256 y0 = _mm256_addsub_ps(ac0, ss0);
                    __m256 y1 = _mm256_addsub_ps(ac1, ss1);
#endif
                    _mm256_storeu_ps(outptr, y0);
                    _mm256_storeu_ps(outptr + 8, y1);

                    ptr += 16;
                    outptr += 16;
                    cos_ptr += 8;
                    sin_ptr += 8;
                }
#endif // __AVX2__
#endif // __AVX__
#if !__SSE3__
#if defined(__MINGW32__) && !defined(__x86_64__)
                __attribute__((aligned(16)))
                const float signmask128_array[4]
                    = {-0.f, 0.f, -0.f, 0.f};
                const __m128 signmask128 = _mm_load_ps(signmask128_array);
#else
                const __m128 signmask128 = _mm_set_ps(0.f, -0.f, 0.f, -0.f);
#endif
#endif
                for (; j + 3 < embed_dim / 2; j += 4)
                {
                    __m128 a0 = _mm_loadu_ps(ptr);
                    __m128 a1 = _mm_loadu_ps(ptr + 4);

                    __m128 c4 = _mm_loadu_ps(cos_ptr);
                    __m128 s4 = _mm_loadu_ps(sin_ptr);

                    __m128 clo = _mm_unpacklo_ps(c4, c4); // [c0,c0,c1,c1]
                    __m128 chi = _mm_unpackhi_ps(c4, c4); // [c2,c2,c3,c3]
                    __m128 slo = _mm_unpacklo_ps(s4, s4); // [s0,s0,s1,s1]
                    __m128 shi = _mm_unpackhi_ps(s4, s4); // [s2,s2,s3,s3]

                    __m128 swap0 = _mm_shuffle_ps(a0, a0, _MM_SHUFFLE(2, 3, 0, 1));
                    __m128 swap1 = _mm_shuffle_ps(a1, a1, _MM_SHUFFLE(2, 3, 0, 1));

                    __m128 ss0 = _mm_mul_ps(swap0, slo);
                    __m128 ss1 = _mm_mul_ps(swap1, shi);
#if __FMA__
                    __m128 y0 = _mm_fmaddsub_ps(a0, clo, ss0);
                    __m128 y1 = _mm_fmaddsub_ps(a1, chi, ss1);
#else
                    __m128 ac0 = _mm_mul_ps(a0, clo);
                    __m128 ac1 = _mm_mul_ps(a1, chi);
#if __SSE3__
                    __m128 y0 = _mm_addsub_ps(ac0, ss0);
                    __m128 y1 = _mm_addsub_ps(ac1, ss1);
#else
                    ss0 = _mm_xor_ps(ss0, signmask128);
                    ss1 = _mm_xor_ps(ss1, signmask128);
                    __m128 y0 = _mm_add_ps(ac0, ss0);
                    __m128 y1 = _mm_add_ps(ac1, ss1);
#endif
#endif
                    _mm_storeu_ps(outptr, y0);
                    _mm_storeu_ps(outptr + 4, y1);

                    ptr += 8;
                    outptr += 8;
                    cos_ptr += 4;
                    sin_ptr += 4;
                }
                for (; j + 1 < embed_dim / 2; j += 2)
                {
                    __m128 a = _mm_loadu_ps(ptr);

                    __m128 c01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)cos_ptr));
                    __m128 s01 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)sin_ptr));

                    __m128 c = _mm_unpacklo_ps(c01, c01); // [c0,c0,c1,c1]
                    __m128 s = _mm_unpacklo_ps(s01, s01); // [s0,s0,s1,s1]

                    __m128 swap = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1));
                    __m128 ss = _mm_mul_ps(swap, s);

#if __FMA__
                    __m128 y = _mm_fmaddsub_ps(a, c, ss);
#else
                    __m128 ac = _mm_mul_ps(a, c);
#if __SSE3__
                    __m128 y = _mm_addsub_ps(ac, ss);
#else
                    ss = _mm_xor_ps(ss, signmask128);
                    __m128 y = _mm_add_ps(ac, ss);
#endif
#endif
                    _mm_storeu_ps(outptr, y);

                    ptr += 4;
                    outptr += 4;
                    cos_ptr += 2;
                    sin_ptr += 2;
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
#if __AVX__
#if __AVX512F__
                for (; j + 15 < embed_dim / 2; j += 16)
                {
                    __m512 x0 = _mm512_loadu_ps(ptr0);
                    __m512 x1 = _mm512_loadu_ps(ptr1);
                    __m512 c = _mm512_loadu_ps(cos_ptr);
                    __m512 s = _mm512_loadu_ps(sin_ptr);

                    __m512 y0 = _mm512_fnmadd_ps(x1, s, _mm512_mul_ps(x0, c));
                    __m512 y1 = _mm512_fmadd_ps(x0, s, _mm512_mul_ps(x1, c));

                    _mm512_storeu_ps(outptr0, y0);
                    _mm512_storeu_ps(outptr1, y1);

                    ptr0 += 16;
                    ptr1 += 16;
                    cos_ptr += 16;
                    sin_ptr += 16;
                    outptr0 += 16;
                    outptr1 += 16;
                }
#endif // __AVX512F__
                for (; j + 7 < embed_dim / 2; j += 8)
                {
                    __m256 x0 = _mm256_loadu_ps(ptr0);
                    __m256 x1 = _mm256_loadu_ps(ptr1);
                    __m256 c = _mm256_loadu_ps(cos_ptr);
                    __m256 s = _mm256_loadu_ps(sin_ptr);

                    __m256 y0 = _mm256_comp_fnmadd_ps(x1, s, _mm256_mul_ps(x0, c));
                    __m256 y1 = _mm256_comp_fmadd_ps(x0, s, _mm256_mul_ps(x1, c));

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

                    __m128 y0 = _mm_comp_fnmadd_ps(x1, s, _mm_mul_ps(x0, c));
                    __m128 y1 = _mm_comp_fmadd_ps(x0, s, _mm_mul_ps(x1, c));

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
