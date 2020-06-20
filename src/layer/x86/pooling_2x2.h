// Tencent is pleased to support the open source community by making ncnn
// available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

static void pooling2x2s2_max_avx(const Mat& bottom_blob, Mat& top_blob,
                                 const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = w - 2 * outw + w;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const float* img0 = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);
        int outcount = 0;
        const float* r0 = img0;
        const float* r1 = img0 + w;
#if __AVX2__
        __m256i permute_mask = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
#endif // __AVX__

        for (int i = 0; i < outh; i++)
        {
#if __AVX2__
            int nn = outw >> 2;
            int remain = outw - (nn << 2);
#else
            int remain = outw;
#endif // __AVX__

#if __AVX2__
            for (; nn > 0; nn--)
            {
                __m256 _r0 = _mm256_loadu_ps(r0);
                __m256 _r1 = _mm256_loadu_ps(r1);
                __m256 _max_r0_r1 = _mm256_max_ps(_r0, _r1);
                _max_r0_r1 = _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(
                                                     _mm256_castps_si256(_max_r0_r1), permute_mask));
                __m128 _max_0 = _mm256_extractf128_ps(_max_r0_r1, 0);
                __m128 _max_1 = _mm256_extractf128_ps(_max_r0_r1, 1);
                __m128 _max = _mm_max_ps(_max_0, _max_1);
                _mm_storeu_ps(outptr, _max);
                r0 += 8;
                r1 += 8;
                outptr += 4;
                outcount += 4;
            }
#endif // __AVX__
            for (; remain > 0; remain--)
            {
                float max0 = std::max(r0[0], r0[1]);
                float max1 = std::max(r1[0], r1[1]);

                *outptr = std::max(max0, max1);

                r0 += 2;
                r1 += 2;
                outptr++;
                outcount++;
            }
            r0 += tailstep;
            r1 += tailstep;
        }
    }
}
