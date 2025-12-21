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
                // [x0, x1, x2, x3, ...]
                const float* ptr = head.row(i);
                const float* cos_ptr = cos_cache.row(i);
                const float* sin_ptr = sin_cache.row(i);
                float* outptr = out_head.row(i);

                for (int j = 0; j < embed_dim / 2; j++)
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
                //   [x0_0, x0_1, ..., x0_{D/2-1}, x1_0, x1_1, ..., x1_{D/2-1}]
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
