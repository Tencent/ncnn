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

        const int* offset_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            __m512 _v = offset_ptr[0] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[0]) : _mm512_set1_ps(0);
            offset_ptr++;

            _mm512_storeu_ps(dstptr, _v);
            dstptr += 16;
        }
    }
}
#endif // __AVX512F__

static void gridsample_nearest_apply_interpolation_p8(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
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

        const int* offset_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            __m256 _v = offset_ptr[0] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[0]) : _mm256_set1_ps(0);
            offset_ptr++;

            _mm256_storeu_ps(dstptr, _v);
            dstptr += 8;
        }
    }
}
#endif // __AVX__
static void gridsample_nearest_apply_interpolation_p4(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
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

        const int* offset_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            __m128 _v = offset_ptr[0] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[0]) : _mm_set1_ps(0);
            offset_ptr++;

            _mm_storeu_ps(dstptr, _v);
            dstptr += 4;
        }
    }
}

#endif // __SSE2__

static void gridsample_nearest_apply_interpolation_p1(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
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

        const int* offset_ptr = offset_value.channel(0);

        for (int x = 0; x < grid_size; x++)
        {
            *dstptr = offset_ptr[0] >= 0 ? *(srcptr + offset_ptr[0]) : 0;

            offset_ptr++;
            dstptr++;
        }
    }
}
