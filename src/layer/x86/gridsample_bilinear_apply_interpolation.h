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
static void gridsample_2d_bilinear_apply_interpolation_p16(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int grid_size = outw * outh;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            const int* offset_ptr = (int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 4;

            __m512 v00_val = offset_ptr[0] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[0]) : _mm512_set1_ps(0);
            __m512 v01_val = offset_ptr[1] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[1]) : _mm512_set1_ps(0);
            __m512 v10_val = offset_ptr[2] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[2]) : _mm512_set1_ps(0);
            __m512 v11_val = offset_ptr[3] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[3]) : _mm512_set1_ps(0);

            __m512 value1 = _mm512_set1_ps(value_ptr[0]);
            __m512 v0 = _mm512_fmadd_ps(v01_val, value1, _mm512_fnmadd_ps(v00_val, value1, v00_val));
            __m512 v1 = _mm512_fmadd_ps(v11_val, value1, _mm512_fnmadd_ps(v10_val, value1, v10_val));

            __m512 value2 = _mm512_set1_ps(value_ptr[1]);
            __m512 _v = _mm512_fmadd_ps(v1, value2, _mm512_fnmadd_ps(v0, value2, v0));
            _mm512_storeu_ps(dstptr, _v);

            dstptr += 16;
            offset_value_ptr += 6;
        }
    }
}

static void gridsample_3d_bilinear_apply_interpolation_p16(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
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

        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            const int* offset_ptr = (int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 8;

            __m512 v000_val = offset_ptr[0] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[0]) : _mm512_set1_ps(0);
            __m512 v001_val = offset_ptr[1] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[1]) : _mm512_set1_ps(0);
            __m512 v010_val = offset_ptr[2] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[2]) : _mm512_set1_ps(0);
            __m512 v011_val = offset_ptr[3] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[3]) : _mm512_set1_ps(0);

            __m512 v100_val = offset_ptr[4] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[4]) : _mm512_set1_ps(0);
            __m512 v101_val = offset_ptr[5] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[5]) : _mm512_set1_ps(0);
            __m512 v110_val = offset_ptr[6] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[6]) : _mm512_set1_ps(0);
            __m512 v111_val = offset_ptr[7] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[7]) : _mm512_set1_ps(0);

            __m512 value = _mm512_set1_ps(value_ptr[0]);
            __m512 v00 = _mm512_fmadd_ps(v001_val, value, _mm512_fnmadd_ps(v000_val, value, v000_val));
            __m512 v01 = _mm512_fmadd_ps(v011_val, value, _mm512_fnmadd_ps(v010_val, value, v010_val));
            __m512 v10 = _mm512_fmadd_ps(v101_val, value, _mm512_fnmadd_ps(v100_val, value, v100_val));
            __m512 v11 = _mm512_fmadd_ps(v111_val, value, _mm512_fnmadd_ps(v110_val, value, v110_val));

            value = _mm512_set1_ps(value_ptr[1]);
            __m512 v0 = _mm512_fmadd_ps(v01, value, _mm512_fnmadd_ps(v00, value, v00));
            __m512 v1 = _mm512_fmadd_ps(v11, value, _mm512_fnmadd_ps(v10, value, v10));

            value = _mm512_set1_ps(value_ptr[2]);
            __m512 _v = _mm512_fmadd_ps(v1, value, _mm512_fnmadd_ps(v0, value, v0));
            _mm512_storeu_ps(dstptr, _v);

            dstptr += 16;
            offset_value_ptr += 11;
        }
    }
}

#endif // __AVX512F__

static void gridsample_2d_bilinear_apply_interpolation_p8(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int grid_size = outw * outh;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            const int* offset_ptr = (int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 4;

            __m256 v00_val = offset_ptr[0] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[0]) : _mm256_set1_ps(0);
            __m256 v01_val = offset_ptr[1] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[1]) : _mm256_set1_ps(0);
            __m256 v10_val = offset_ptr[2] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[2]) : _mm256_set1_ps(0);
            __m256 v11_val = offset_ptr[3] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[3]) : _mm256_set1_ps(0);

            __m256 value1 = _mm256_set1_ps(value_ptr[0]);
            __m256 v0 = _mm256_comp_fmadd_ps(v01_val, value1, _mm256_comp_fnmadd_ps(v00_val, value1, v00_val));
            __m256 v1 = _mm256_comp_fmadd_ps(v11_val, value1, _mm256_comp_fnmadd_ps(v10_val, value1, v10_val));

            __m256 value2 = _mm256_set1_ps(value_ptr[1]);
            __m256 _v = _mm256_comp_fmadd_ps(v1, value2, _mm256_comp_fnmadd_ps(v0, value2, v0));
            _mm256_storeu_ps(dstptr, _v);

            dstptr += 8;
            offset_value_ptr += 6;
        }
    }
}
static void gridsample_3d_bilinear_apply_interpolation_p8(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
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

        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            const int* offset_ptr = (int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 8;

            __m256 v000_val = offset_ptr[0] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[0]) : _mm256_set1_ps(0);
            __m256 v001_val = offset_ptr[1] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[1]) : _mm256_set1_ps(0);
            __m256 v010_val = offset_ptr[2] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[2]) : _mm256_set1_ps(0);
            __m256 v011_val = offset_ptr[3] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[3]) : _mm256_set1_ps(0);

            __m256 v100_val = offset_ptr[4] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[4]) : _mm256_set1_ps(0);
            __m256 v101_val = offset_ptr[5] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[5]) : _mm256_set1_ps(0);
            __m256 v110_val = offset_ptr[6] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[6]) : _mm256_set1_ps(0);
            __m256 v111_val = offset_ptr[7] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[7]) : _mm256_set1_ps(0);

            __m256 value = _mm256_set1_ps(value_ptr[0]);
            __m256 v00 = _mm256_comp_fmadd_ps(v001_val, value, _mm256_comp_fnmadd_ps(v000_val, value, v000_val));
            __m256 v01 = _mm256_comp_fmadd_ps(v011_val, value, _mm256_comp_fnmadd_ps(v010_val, value, v010_val));
            __m256 v10 = _mm256_comp_fmadd_ps(v101_val, value, _mm256_comp_fnmadd_ps(v100_val, value, v100_val));
            __m256 v11 = _mm256_comp_fmadd_ps(v111_val, value, _mm256_comp_fnmadd_ps(v110_val, value, v110_val));

            value = _mm256_set1_ps(value_ptr[1]);
            __m256 v0 = _mm256_comp_fmadd_ps(v01, value, _mm256_comp_fnmadd_ps(v00, value, v00));
            __m256 v1 = _mm256_comp_fmadd_ps(v11, value, _mm256_comp_fnmadd_ps(v10, value, v10));

            value = _mm256_set1_ps(value_ptr[2]);
            __m256 _v = _mm256_comp_fmadd_ps(v1, value, _mm256_comp_fnmadd_ps(v0, value, v0));
            _mm256_storeu_ps(dstptr, _v);

            dstptr += 8;
            offset_value_ptr += 11;
        }
    }
}
#endif // __AVX__
static void gridsample_2d_bilinear_apply_interpolation_p4(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int grid_size = outw * outh;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            const int* offset_ptr = (int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 4;

            __m128 v00_val = offset_ptr[0] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[0]) : _mm_set1_ps(0);
            __m128 v01_val = offset_ptr[1] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[1]) : _mm_set1_ps(0);
            __m128 v10_val = offset_ptr[2] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[2]) : _mm_set1_ps(0);
            __m128 v11_val = offset_ptr[3] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[3]) : _mm_set1_ps(0);

            __m128 value1 = _mm_set1_ps(value_ptr[0]);
            __m128 v0 = _mm_comp_fmadd_ps(v01_val, value1, _mm_comp_fnmadd_ps(v00_val, value1, v00_val));
            __m128 v1 = _mm_comp_fmadd_ps(v11_val, value1, _mm_comp_fnmadd_ps(v10_val, value1, v10_val));

            __m128 value2 = _mm_set1_ps(value_ptr[1]);
            __m128 _v = _mm_comp_fmadd_ps(v1, value2, _mm_comp_fnmadd_ps(v0, value2, v0));
            _mm_storeu_ps(dstptr, _v);

            dstptr += 4;
            offset_value_ptr += 6;
        }
    }
}
static void gridsample_3d_bilinear_apply_interpolation_p4(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
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

        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            const int* offset_ptr = (int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 8;

            __m128 v000_val = offset_ptr[0] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[0]) : _mm_set1_ps(0);
            __m128 v001_val = offset_ptr[1] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[1]) : _mm_set1_ps(0);
            __m128 v010_val = offset_ptr[2] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[2]) : _mm_set1_ps(0);
            __m128 v011_val = offset_ptr[3] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[3]) : _mm_set1_ps(0);

            __m128 v100_val = offset_ptr[4] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[4]) : _mm_set1_ps(0);
            __m128 v101_val = offset_ptr[5] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[5]) : _mm_set1_ps(0);
            __m128 v110_val = offset_ptr[6] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[6]) : _mm_set1_ps(0);
            __m128 v111_val = offset_ptr[7] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[7]) : _mm_set1_ps(0);

            __m128 value = _mm_set1_ps(value_ptr[0]);
            __m128 v00 = _mm_comp_fmadd_ps(v001_val, value, _mm_comp_fnmadd_ps(v000_val, value, v000_val));
            __m128 v01 = _mm_comp_fmadd_ps(v011_val, value, _mm_comp_fnmadd_ps(v010_val, value, v010_val));
            __m128 v10 = _mm_comp_fmadd_ps(v101_val, value, _mm_comp_fnmadd_ps(v100_val, value, v100_val));
            __m128 v11 = _mm_comp_fmadd_ps(v111_val, value, _mm_comp_fnmadd_ps(v110_val, value, v110_val));

            value = _mm_set1_ps(value_ptr[1]);
            __m128 v0 = _mm_comp_fmadd_ps(v01, value, _mm_comp_fnmadd_ps(v00, value, v00));
            __m128 v1 = _mm_comp_fmadd_ps(v11, value, _mm_comp_fnmadd_ps(v10, value, v10));

            value = _mm_set1_ps(value_ptr[2]);
            __m128 _v = _mm_comp_fmadd_ps(v1, value, _mm_comp_fnmadd_ps(v0, value, v0));
            _mm_storeu_ps(dstptr, _v);

            dstptr += 4;
            offset_value_ptr += 11;
        }
    }
}
#endif // __SSE2__

static void gridsample_2d_bilinear_apply_interpolation_p1(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int grid_size = outw * outh;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        const float* offset_value_ptr = offset_value.channel(0);

        for (int x = 0; x < grid_size; x++)
        {
            const int* offset_ptr = (int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 4;

            float v00 = offset_ptr[0] >= 0 ? *(srcptr + offset_ptr[0]) : 0;
            float v01 = offset_ptr[1] >= 0 ? *(srcptr + offset_ptr[1]) : 0;
            float v10 = offset_ptr[2] >= 0 ? *(srcptr + offset_ptr[2]) : 0;
            float v11 = offset_ptr[3] >= 0 ? *(srcptr + offset_ptr[3]) : 0;

            float v0 = v00 * (1 - value_ptr[0]) + v01 * value_ptr[0];
            float v1 = v10 * (1 - value_ptr[0]) + v11 * value_ptr[0];

            *dstptr = v0 * (1 - value_ptr[1]) + v1 * value_ptr[1];

            dstptr++;
            offset_value_ptr += 6;
        }
    }
}

static void gridsample_3d_bilinear_apply_interpolation_p1(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
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

        const float* offset_value_ptr = offset_value.channel(0);

        for (int x = 0; x < grid_size; x++)
        {
            const int* offset_ptr = (int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 8;

            float v000 = offset_ptr[0] >= 0 ? *(srcptr + offset_ptr[0]) : 0;
            float v001 = offset_ptr[1] >= 0 ? *(srcptr + offset_ptr[1]) : 0;
            float v010 = offset_ptr[2] >= 0 ? *(srcptr + offset_ptr[2]) : 0;
            float v011 = offset_ptr[3] >= 0 ? *(srcptr + offset_ptr[3]) : 0;

            float v100 = offset_ptr[4] >= 0 ? *(srcptr + offset_ptr[4]) : 0;
            float v101 = offset_ptr[5] >= 0 ? *(srcptr + offset_ptr[5]) : 0;
            float v110 = offset_ptr[6] >= 0 ? *(srcptr + offset_ptr[6]) : 0;
            float v111 = offset_ptr[7] >= 0 ? *(srcptr + offset_ptr[7]) : 0;

            float v00 = v000 * (1 - value_ptr[0]) + v001 * value_ptr[0];
            float v01 = v010 * (1 - value_ptr[0]) + v011 * value_ptr[0];
            float v10 = v100 * (1 - value_ptr[0]) + v101 * value_ptr[0];
            float v11 = v110 * (1 - value_ptr[0]) + v111 * value_ptr[0];

            float v0 = v00 * (1 - value_ptr[1]) + v01 * value_ptr[1];
            float v1 = v10 * (1 - value_ptr[1]) + v11 * value_ptr[1];

            *dstptr = v0 * (1 - value_ptr[2]) + v1 * value_ptr[2];

            dstptr++;
            offset_value_ptr += 11;
        }
    }
}
