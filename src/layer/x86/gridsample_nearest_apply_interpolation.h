// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#if __SSE2__
#if __AVX__
#if __AVX512F__
static void gridsample_nearest_apply_interpolation_p16(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int outd = dst.d;
    const int grid_size = outw * outh * outd;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        const float* offset_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            __m512 _v = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_ptr) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), _mm512_add_epi32(_mm512_set1_epi32(*offset_ptr), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)), srcptr, sizeof(float));

            _mm512_storeu_ps(dstptr, _v);

            offset_ptr++;
            dstptr += 16;
        }
    }
}
#endif // __AVX512F__

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
void gridsample_nearest_apply_interpolation_p8_avx2(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt);
void gridsample_nearest_apply_interpolation_p4_avx2(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt);
void gridsample_nearest_apply_interpolation_p1_avx2(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt);
#endif

static void gridsample_nearest_apply_interpolation_p8(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        gridsample_nearest_apply_interpolation_p8_avx2(src, dst, offset_value, opt);
        return;
    }
#endif

    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int outd = dst.d;
    const int grid_size = outw * outh * outd;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        const float* offset_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            float in_bound = *reinterpret_cast<const int*>(offset_ptr) >= 0 ? -1.0f : 0.0f;
#if __AVX2__
            __m256i _offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
#else
            __m256i _offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_ptr), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
#endif // __AVX2__
            __m256 _v = mask_gather_ps256(srcptr, _offset, _mm256_set1_ps(in_bound));

            _mm256_storeu_ps(dstptr, _v);

            offset_ptr++;
            dstptr += 8;
        }
    }
}
#endif // __AVX__
static void gridsample_nearest_apply_interpolation_p4(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        gridsample_nearest_apply_interpolation_p4_avx2(src, dst, offset_value, opt);
        return;
    }
#endif

    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int outd = dst.d;
    const int grid_size = outw * outh * outd;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        const float* offset_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            float in_bound = *reinterpret_cast<const int*>(offset_ptr) >= 0 ? -1.0f : 0.0f;
            __m128 _v = mask_gather_ps(srcptr, _mm_add_epi32(_mm_set1_epi32(*offset_ptr), _mm_set_epi32(3, 2, 1, 0)), _mm_set1_ps(in_bound));

            _mm_storeu_ps(dstptr, _v);

            offset_ptr++;
            dstptr += 4;
        }
    }
}

#endif // __SSE2__

static void gridsample_nearest_apply_interpolation_p1(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        gridsample_nearest_apply_interpolation_p1_avx2(src, dst, offset_value, opt);
        return;
    }
#endif

    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int outd = dst.d;
    const int grid_size = outw * outh * outd;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        const float* offset_ptr = offset_value.channel(0);

        int x = 0;
#if __SSE2__
#if __AVX__
        for (; x + 7 < grid_size; x += 8)
        {
            __m256 in_bound = _mm256_andnot_ps(_mm256_loadu_ps(offset_ptr), _mm256_set1_ps(-1.0f));
            __m256 _v = mask_gather_ps256(srcptr, _mm256_set_epi32(*(offset_ptr + 7), *(offset_ptr + 6), *(offset_ptr + 5), *(offset_ptr + 4), *(offset_ptr + 3), *(offset_ptr + 2), *(offset_ptr + 1), *offset_ptr), in_bound);

            _mm256_storeu_ps(dstptr, _v);

            offset_ptr += 8;
            dstptr += 8;
        }
#endif // __AVX__
        for (; x + 3 < grid_size; x += 4)
        {
            __m128 in_bound = _mm_andnot_ps(_mm_loadu_ps(offset_ptr), _mm_set1_ps(-1.0f));
            __m128 _v = mask_gather_ps(srcptr, _mm_set_epi32(*(offset_ptr + 3), *(offset_ptr + 2), *(offset_ptr + 1), *offset_ptr), in_bound);

            _mm_storeu_ps(dstptr, _v);

            offset_ptr += 4;
            dstptr += 4;
        }
#endif // __SSE2__
        for (; x < grid_size; x++)
        {
            *dstptr = *reinterpret_cast<const int*>(offset_ptr) >= 0 ? *(srcptr + static_cast<int>(*offset_ptr)) : 0;

            offset_ptr++;
            dstptr++;
        }
    }
}
