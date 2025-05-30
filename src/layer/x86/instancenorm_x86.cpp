// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "instancenorm_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif // __SSE2__

namespace ncnn {

InstanceNorm_x86::InstanceNorm_x86()
{
    one_blob_only = true;
    support_inplace = true;
}

int InstanceNorm_x86::load_param(const ParamDict& pd)
{
    channels = pd.get(0, 0);
    eps = pd.get(1, 0.001f);
    affine = pd.get(2, 1);

    return 0;
}

int InstanceNorm_x86::load_model(const ModelBin& mb)
{
    if (affine == 0)
        return 0;

    gamma_data = mb.load(channels, 1);
    if (gamma_data.empty())
        return -100;

    beta_data = mb.load(channels, 1);
    if (beta_data.empty())
        return -100;

    return 0;
}

int InstanceNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // x = (x - mean) / (sqrt(var + eps)) * gamma + beta

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int c = bottom_top_blob.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < c; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        // mean and var
        float sum = 0.f;
        float sqsum = 0.f;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _sum_vec_16 = _mm512_setzero_ps();
        for (; i <= size - 16; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr + i);
            _sum_vec_16 = _mm512_add_ps(_sum_vec_16, _p);
        }
        sum = _mm512_reduce_add_ps(_sum_vec_16);
#endif // __AVX512F__
        __m256 _sum_vec_8 = _mm256_setzero_ps();
        for (; i <= size - 8; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr + i);
            _sum_vec_8 = _mm256_add_ps(_sum_vec_8, _p);
        }

        __m128 vlow = _mm256_castps256_ps128(_sum_vec_8);
        __m128 vhigh = _mm256_extractf128_ps(_sum_vec_8, 1);
        __m128 hsum_128 = _mm_add_ps(vlow, vhigh);
        hsum_128 = _mm_hadd_ps(hsum_128, hsum_128);
        hsum_128 = _mm_hadd_ps(hsum_128, hsum_128);
        sum += _mm_cvtss_f32(hsum_128);
#endif // __AVX__
        __m128 _sum_vec_4 = _mm_setzero_ps();

        for (; i <= size - 4; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr + i);
            _sum_vec_4 = _mm_add_ps(_sum_vec_4, _p);
        }

        _sum_vec_4 = _mm_add_ps(_sum_vec_4, _mm_movehl_ps(_sum_vec_4, _sum_vec_4));                           // _sum_vec_4 = [s0+s2, s1+s3, s2+s2, s3+s3]
        _sum_vec_4 = _mm_add_ss(_sum_vec_4, _mm_shuffle_ps(_sum_vec_4, _sum_vec_4, _MM_SHUFFLE(1, 1, 1, 1))); // h_sum_tmp[0] = (s0+s2) + (s1+s3)
        sum += _mm_cvtss_f32(_sum_vec_4);
#endif //__SSE2__
        for (; i < size; i++)
        {
            sum += ptr[i];
        }

        i = 0;

        float mean = sum / size;
        float tmp = 0.f;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _sqsum_vec_16 = _mm512_setzero_ps();
        __m512 _mean_vec_16 = _mm512_set1_ps(mean);

        for (; i <= size - 16; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr + i);
            __m512 _diff = _mm512_sub_ps(_p, _mean_vec_16);
            _sqsum_vec_16 = _mm512_add_ps(_sqsum_vec_16, _mm512_mul_ps(_diff, _diff));
        }
        sqsum += _mm512_reduce_add_ps(_sqsum_vec_16);
#endif // __AVX512F
        __m256 _sqsum_vec_8 = _mm256_setzero_ps();
        __m256 _mean_vec_8 = _mm256_set1_ps(mean);

        for (; i <= size - 8; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr + i);
            __m256 _diff = _mm256_sub_ps(_p, _mean_vec_8);
            _sqsum_vec_8 = _mm256_add_ps(_sqsum_vec_8, _mm256_mul_ps(_diff, _diff));
        }

        __m128 vlow_sq = _mm256_castps256_ps128(_sqsum_vec_8);
        __m128 vhigh_sq = _mm256_extractf128_ps(_sqsum_vec_8, 1);
        __m128 hsqsum_128 = _mm_add_ps(vlow_sq, vhigh_sq);
        hsqsum_128 = _mm_hadd_ps(hsqsum_128, hsqsum_128);
        hsqsum_128 = _mm_hadd_ps(hsqsum_128, hsqsum_128);
        sqsum += _mm_cvtss_f32(hsqsum_128);
#endif // __AVX__
        __m128 _sqsum_vec_4 = _mm_setzero_ps();
        __m128 _mean_vec_4 = _mm_set1_ps(mean);

        for (; i <= size - 4; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr + i);
            __m128 _diff = _mm_sub_ps(_p, _mean_vec_4);
            _sqsum_vec_4 = _mm_add_ps(_sqsum_vec_4, _mm_mul_ps(_diff, _diff));
        }

        __m128 h_sqsum_tmp = _sqsum_vec_4;
        h_sqsum_tmp = _mm_add_ps(h_sqsum_tmp, _mm_movehl_ps(h_sqsum_tmp, h_sqsum_tmp));
        h_sqsum_tmp = _mm_add_ss(h_sqsum_tmp, _mm_shuffle_ps(h_sqsum_tmp, h_sqsum_tmp, _MM_SHUFFLE(1, 1, 1, 1)));
        sqsum += _mm_cvtss_f32(h_sqsum_tmp);
#endif // __SSE2__

        for (; i < size; i++)
        {
            tmp = ptr[i] - mean;
            sqsum += tmp * tmp;
        }
        float var = sqsum / size;
        // the var maybe minus due to accuracy
        //float var = sqsum / size - mean * mean;

        float a;
        float b;
        if (affine)
        {
            float gamma = gamma_data[q];
            float beta = beta_data[q];

            a = gamma / (sqrtf(var + eps));
            b = -mean * a + beta;
        }
        else
        {
            a = 1.f / (sqrtf(var + eps));
            b = -mean * a;
        }

        i = 0;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _va_16 = _mm512_set1_ps(a);
        __m512 _vb_16 = _mm512_set1_ps(b);

        for (; i <= size - 16; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr + i);
            _p = _mm512_fmadd_ps(_p, _va_16, _vb_16); // Fused Multiply-Add
            _mm512_storeu_ps(ptr + i, _p);
        }
#endif // __AVX512F__
        __m256 _va_8 = _mm256_set1_ps(a);
        __m256 _vb_8 = _mm256_set1_ps(b);

        for (; i <= size - 8; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr + i);
#if defined(__FMA__) // Check for FMA support (usually with AVX2)
            _p = _mm256_fmadd_ps(_p, _va_8, _vb_8);
#else
            _p = _mm256_mul_ps(_p, _va_8);
            _p = _mm256_add_ps(_p, _vb_8);
#endif
            _mm256_storeu_ps(ptr + i, _p);
        }
#endif // __AVX__
        __m128 _va_4 = _mm_set1_ps(a);
        __m128 _vb_4 = _mm_set1_ps(b);

        for (; i <= size - 4; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr + i);
            // SSE2 does not have FMA instructions directly; FMA is part of FMA3 extension.
            // Compilers might fuse mul+add if target architecture supports it (e.g. compiling with -mfma).
            _p = _mm_mul_ps(_p, _va_4);
            _p = _mm_add_ps(_p, _vb_4);
            _mm_storeu_ps(ptr + i, _p);
        }
#endif // __SSE2__

        for (; i < size; i++)
        {
            ptr[i] = ptr[i] * a + b;
        }
    }

    return 0;
}

} // namespace ncnn
