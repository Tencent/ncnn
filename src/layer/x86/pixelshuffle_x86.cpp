// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pixelshuffle_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

PixelShuffle_x86::PixelShuffle_x86()
{
}

int PixelShuffle_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();
    if (elembits != 32)
        return PixelShuffle::forward(bottom_blob, top_blob, opt);

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const size_t elemsize = bottom_blob.elemsize;

    const int r = upscale_factor;
    const int outw = w * r;
    const int outh = h * r;
    const int outc = channels / (r * r);

    if (bottom_blob.elempack != 1)
        return PixelShuffle::forward(bottom_blob, top_blob, opt);

    if (r != 2 && r != 4)
        return PixelShuffle::forward(bottom_blob, top_blob, opt);

    top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int rr = r * r;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outc; p++)
    {
        Mat m = top_blob.channel(p);

        for (int sh = 0; sh < r; sh++)
        {
            // mode 0: q = p*r*r + sh*r + sw ; mode 1: q = (sh*r + sw)*outc + p
            const float* s[4];
            for (int sw = 0; sw < r; sw++)
            {
                int q;
                if (mode == 0)
                    q = p * rr + sh * r + sw;
                else
                    q = (sh * r + sw) * outc + p;
                s[sw] = bottom_blob.channel(q);
            }

            if (r == 2)
            {
                const float* s0 = s[0];
                const float* s1 = s[1];
                for (int i = 0; i < h; i++)
                {
                    float* outptr = m.row(i * 2 + sh);
                    int j = 0;
#if __SSE2__
#if __AVX__
                    for (; j + 8 <= w; j += 8)
                    {
                        __m256 _p0 = _mm256_loadu_ps(s0 + j);
                        __m256 _p1 = _mm256_loadu_ps(s1 + j);
                        __m256 _lo = _mm256_unpacklo_ps(_p0, _p1);
                        __m256 _hi = _mm256_unpackhi_ps(_p0, _p1);
                        __m256 _out0 = _mm256_permute2f128_ps(_lo, _hi, 0x20);
                        __m256 _out1 = _mm256_permute2f128_ps(_lo, _hi, 0x31);
                        _mm256_storeu_ps(outptr + j * 2 + 0, _out0);
                        _mm256_storeu_ps(outptr + j * 2 + 8, _out1);
                    }
#endif // __AVX__
                    for (; j + 4 <= w; j += 4)
                    {
                        __m128 _p0 = _mm_loadu_ps(s0 + j);
                        __m128 _p1 = _mm_loadu_ps(s1 + j);
                        __m128 _lo = _mm_unpacklo_ps(_p0, _p1);
                        __m128 _hi = _mm_unpackhi_ps(_p0, _p1);
                        _mm_storeu_ps(outptr + j * 2 + 0, _lo);
                        _mm_storeu_ps(outptr + j * 2 + 4, _hi);
                    }
#endif // __SSE2__
                    for (; j < w; j++)
                    {
                        outptr[j * 2 + 0] = s0[j];
                        outptr[j * 2 + 1] = s1[j];
                    }
                    s0 += w;
                    s1 += w;
                }
            }
            else // r == 4
            {
                const float* s0 = s[0];
                const float* s1 = s[1];
                const float* s2 = s[2];
                const float* s3 = s[3];
                for (int i = 0; i < h; i++)
                {
                    float* outptr = m.row(i * 4 + sh);
                    int j = 0;
#if __SSE2__
                    for (; j + 4 <= w; j += 4)
                    {
                        __m128 _p0 = _mm_loadu_ps(s0 + j);
                        __m128 _p1 = _mm_loadu_ps(s1 + j);
                        __m128 _p2 = _mm_loadu_ps(s2 + j);
                        __m128 _p3 = _mm_loadu_ps(s3 + j);
                        _MM_TRANSPOSE4_PS(_p0, _p1, _p2, _p3);
                        _mm_storeu_ps(outptr + j * 4 + 0, _p0);
                        _mm_storeu_ps(outptr + j * 4 + 4, _p1);
                        _mm_storeu_ps(outptr + j * 4 + 8, _p2);
                        _mm_storeu_ps(outptr + j * 4 + 12, _p3);
                    }
#endif // __SSE2__
                    for (; j < w; j++)
                    {
                        outptr[j * 4 + 0] = s0[j];
                        outptr[j * 4 + 1] = s1[j];
                        outptr[j * 4 + 2] = s2[j];
                        outptr[j * 4 + 3] = s3[j];
                    }
                    s0 += w;
                    s1 += w;
                    s2 += w;
                    s3 += w;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
