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
            __mmask16 in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? 0xFFFF : 0;
            __m512 v00_val = _mm512_maskz_load_ps(in_bound, srcptr + static_cast<int>(*offset_value_ptr));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? 0xFFFF : 0;
            __m512 v01_val = _mm512_maskz_load_ps(in_bound, srcptr + static_cast<int>(*offset_value_ptr));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? 0xFFFF : 0;
            __m512 v10_val = _mm512_maskz_load_ps(in_bound, srcptr + static_cast<int>(*offset_value_ptr));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? 0xFFFF : 0;
            __m512 v11_val = _mm512_maskz_load_ps(in_bound, srcptr + static_cast<int>(*offset_value_ptr));
            offset_value_ptr++;

            __m512 value = _mm512_set1_ps(*offset_value_ptr++);
            __m512 v0 = _mm512_fmadd_ps(v01_val, value, _mm512_fnmadd_ps(v00_val, value, v00_val));
            __m512 v1 = _mm512_fmadd_ps(v11_val, value, _mm512_fnmadd_ps(v10_val, value, v10_val));

            value = _mm512_set1_ps(*offset_value_ptr++);
            __m512 _v = _mm512_fmadd_ps(v1, value, _mm512_fnmadd_ps(v0, value, v0));
            _mm512_storeu_ps(dstptr, _v);

            dstptr += 16;
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
            __mmask16 in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? 0xFFFF : 0;
            __m512 v000_val = _mm512_maskz_load_ps(in_bound, srcptr + static_cast<int>(*offset_value_ptr));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? 0xFFFF : 0;
            __m512 v001_val = _mm512_maskz_load_ps(in_bound, srcptr + static_cast<int>(*offset_value_ptr));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? 0xFFFF : 0;
            __m512 v010_val = _mm512_maskz_load_ps(in_bound, srcptr + static_cast<int>(*offset_value_ptr));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? 0xFFFF : 0;
            __m512 v011_val = _mm512_maskz_load_ps(in_bound, srcptr + static_cast<int>(*offset_value_ptr));
            offset_value_ptr++;

            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? 0xFFFF : 0;
            __m512 v100_val = _mm512_maskz_load_ps(in_bound, srcptr + static_cast<int>(*offset_value_ptr));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? 0xFFFF : 0;
            __m512 v101_val = _mm512_maskz_load_ps(in_bound, srcptr + static_cast<int>(*offset_value_ptr));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? 0xFFFF : 0;
            __m512 v110_val = _mm512_maskz_load_ps(in_bound, srcptr + static_cast<int>(*offset_value_ptr));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? 0xFFFF : 0;
            __m512 v111_val = _mm512_maskz_load_ps(in_bound, srcptr + static_cast<int>(*offset_value_ptr));
            offset_value_ptr++;

            __m512 value = _mm512_set1_ps(*offset_value_ptr++);
            __m512 v00 = _mm512_fmadd_ps(v001_val, value, _mm512_fnmadd_ps(v000_val, value, v000_val));
            __m512 v01 = _mm512_fmadd_ps(v011_val, value, _mm512_fnmadd_ps(v010_val, value, v010_val));
            __m512 v10 = _mm512_fmadd_ps(v101_val, value, _mm512_fnmadd_ps(v100_val, value, v100_val));
            __m512 v11 = _mm512_fmadd_ps(v111_val, value, _mm512_fnmadd_ps(v110_val, value, v110_val));

            value = _mm512_set1_ps(*offset_value_ptr++);
            __m512 v0 = _mm512_fmadd_ps(v01, value, _mm512_fnmadd_ps(v00, value, v00));
            __m512 v1 = _mm512_fmadd_ps(v11, value, _mm512_fnmadd_ps(v10, value, v10));

            value = _mm512_set1_ps(*offset_value_ptr++);
            __m512 _v = _mm512_fmadd_ps(v1, value, _mm512_fnmadd_ps(v0, value, v0));
            _mm512_storeu_ps(dstptr, _v);

            dstptr += 16;
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
            int in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? -1 : 0;
            __m256 v00_val = _mm256_maskload_ps(srcptr + static_cast<int>(*offset_value_ptr), _mm256_set1_epi32(in_bound));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? -1 : 0;
            __m256 v01_val = _mm256_maskload_ps(srcptr + static_cast<int>(*offset_value_ptr), _mm256_set1_epi32(in_bound));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? -1 : 0;
            __m256 v10_val = _mm256_maskload_ps(srcptr + static_cast<int>(*offset_value_ptr), _mm256_set1_epi32(in_bound));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? -1 : 0;
            __m256 v11_val = _mm256_maskload_ps(srcptr + static_cast<int>(*offset_value_ptr), _mm256_set1_epi32(in_bound));
            offset_value_ptr++;

            __m256 value = _mm256_set1_ps(*offset_value_ptr++);
            __m256 v0 = _mm256_comp_fmadd_ps(v01_val, value, _mm256_comp_fnmadd_ps(v00_val, value, v00_val));
            __m256 v1 = _mm256_comp_fmadd_ps(v11_val, value, _mm256_comp_fnmadd_ps(v10_val, value, v10_val));

            value = _mm256_set1_ps(*offset_value_ptr++);
            __m256 _v = _mm256_comp_fmadd_ps(v1, value, _mm256_comp_fnmadd_ps(v0, value, v0));
            _mm256_storeu_ps(dstptr, _v);

            dstptr += 8;
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
            int in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? -1 : 0;
            __m256 v000_val = _mm256_maskload_ps(srcptr + static_cast<int>(*offset_value_ptr), _mm256_set1_epi32(in_bound));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? -1 : 0;
            __m256 v001_val = _mm256_maskload_ps(srcptr + static_cast<int>(*offset_value_ptr), _mm256_set1_epi32(in_bound));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? -1 : 0;
            __m256 v010_val = _mm256_maskload_ps(srcptr + static_cast<int>(*offset_value_ptr), _mm256_set1_epi32(in_bound));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? -1 : 0;
            __m256 v011_val = _mm256_maskload_ps(srcptr + static_cast<int>(*offset_value_ptr), _mm256_set1_epi32(in_bound));
            offset_value_ptr++;

            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? -1 : 0;
            __m256 v100_val = _mm256_maskload_ps(srcptr + static_cast<int>(*offset_value_ptr), _mm256_set1_epi32(in_bound));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? -1 : 0;
            __m256 v101_val = _mm256_maskload_ps(srcptr + static_cast<int>(*offset_value_ptr), _mm256_set1_epi32(in_bound));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? -1 : 0;
            __m256 v110_val = _mm256_maskload_ps(srcptr + static_cast<int>(*offset_value_ptr), _mm256_set1_epi32(in_bound));
            offset_value_ptr++;
            in_bound = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? -1 : 0;
            __m256 v111_val = _mm256_maskload_ps(srcptr + static_cast<int>(*offset_value_ptr), _mm256_set1_epi32(in_bound));
            offset_value_ptr++;

            __m256 value = _mm256_set1_ps(*offset_value_ptr++);
            __m256 v00 = _mm256_comp_fmadd_ps(v001_val, value, _mm256_comp_fnmadd_ps(v000_val, value, v000_val));
            __m256 v01 = _mm256_comp_fmadd_ps(v011_val, value, _mm256_comp_fnmadd_ps(v010_val, value, v010_val));
            __m256 v10 = _mm256_comp_fmadd_ps(v101_val, value, _mm256_comp_fnmadd_ps(v100_val, value, v100_val));
            __m256 v11 = _mm256_comp_fmadd_ps(v111_val, value, _mm256_comp_fnmadd_ps(v110_val, value, v110_val));

            value = _mm256_set1_ps(*offset_value_ptr++);
            __m256 v0 = _mm256_comp_fmadd_ps(v01, value, _mm256_comp_fnmadd_ps(v00, value, v00));
            __m256 v1 = _mm256_comp_fmadd_ps(v11, value, _mm256_comp_fnmadd_ps(v10, value, v10));

            value = _mm256_set1_ps(*offset_value_ptr++);
            __m256 _v = _mm256_comp_fmadd_ps(v1, value, _mm256_comp_fnmadd_ps(v0, value, v0));
            _mm256_storeu_ps(dstptr, _v);

            dstptr += 8;
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
            __m128 v00_val = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? _mm_load_ps(srcptr + static_cast<int>(*offset_value_ptr)) : _mm_set1_ps(0);
            offset_value_ptr++;
            __m128 v01_val = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? _mm_load_ps(srcptr + static_cast<int>(*offset_value_ptr)) : _mm_set1_ps(0);
            offset_value_ptr++;
            __m128 v10_val = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? _mm_load_ps(srcptr + static_cast<int>(*offset_value_ptr)) : _mm_set1_ps(0);
            offset_value_ptr++;
            __m128 v11_val = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? _mm_load_ps(srcptr + static_cast<int>(*offset_value_ptr)) : _mm_set1_ps(0);
            offset_value_ptr++;

            __m128 value = _mm_set1_ps(*offset_value_ptr++);
            __m128 v0 = _mm_comp_fmadd_ps(v01_val, value, _mm_comp_fnmadd_ps(v00_val, value, v00_val));
            __m128 v1 = _mm_comp_fmadd_ps(v11_val, value, _mm_comp_fnmadd_ps(v10_val, value, v10_val));

            value = _mm_set1_ps(*offset_value_ptr++);
            __m128 _v = _mm_comp_fmadd_ps(v1, value, _mm_comp_fnmadd_ps(v0, value, v0));
            _mm_storeu_ps(dstptr, _v);

            dstptr += 4;
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
            __m128 v000_val = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? _mm_load_ps(srcptr + static_cast<int>(*offset_value_ptr)) : _mm_set1_ps(0);
            offset_value_ptr++;
            __m128 v001_val = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? _mm_load_ps(srcptr + static_cast<int>(*offset_value_ptr)) : _mm_set1_ps(0);
            offset_value_ptr++;
            __m128 v010_val = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? _mm_load_ps(srcptr + static_cast<int>(*offset_value_ptr)) : _mm_set1_ps(0);
            offset_value_ptr++;
            __m128 v011_val = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? _mm_load_ps(srcptr + static_cast<int>(*offset_value_ptr)) : _mm_set1_ps(0);
            offset_value_ptr++;

            __m128 v100_val = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? _mm_load_ps(srcptr + static_cast<int>(*offset_value_ptr)) : _mm_set1_ps(0);
            offset_value_ptr++;
            __m128 v101_val = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? _mm_load_ps(srcptr + static_cast<int>(*offset_value_ptr)) : _mm_set1_ps(0);
            offset_value_ptr++;
            __m128 v110_val = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? _mm_load_ps(srcptr + static_cast<int>(*offset_value_ptr)) : _mm_set1_ps(0);
            offset_value_ptr++;
            __m128 v111_val = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? _mm_load_ps(srcptr + static_cast<int>(*offset_value_ptr)) : _mm_set1_ps(0);
            offset_value_ptr++;

            __m128 value = _mm_set1_ps(*offset_value_ptr++);
            __m128 v00 = _mm_comp_fmadd_ps(v001_val, value, _mm_comp_fnmadd_ps(v000_val, value, v000_val));
            __m128 v01 = _mm_comp_fmadd_ps(v011_val, value, _mm_comp_fnmadd_ps(v010_val, value, v010_val));
            __m128 v10 = _mm_comp_fmadd_ps(v101_val, value, _mm_comp_fnmadd_ps(v100_val, value, v100_val));
            __m128 v11 = _mm_comp_fmadd_ps(v111_val, value, _mm_comp_fnmadd_ps(v110_val, value, v110_val));

            value = _mm_set1_ps(*offset_value_ptr++);
            __m128 v0 = _mm_comp_fmadd_ps(v01, value, _mm_comp_fnmadd_ps(v00, value, v00));
            __m128 v1 = _mm_comp_fmadd_ps(v11, value, _mm_comp_fnmadd_ps(v10, value, v10));

            value = _mm_set1_ps(*offset_value_ptr++);
            __m128 _v = _mm_comp_fmadd_ps(v1, value, _mm_comp_fnmadd_ps(v0, value, v0));
            _mm_storeu_ps(dstptr, _v);

            dstptr += 4;
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
            float v00 = *offset_value_ptr >= 0 ? *(srcptr + static_cast<int>(*offset_value_ptr)) : 0;
            offset_value_ptr++;
            float v01 = *offset_value_ptr >= 0 ? *(srcptr + static_cast<int>(*offset_value_ptr)) : 0;
            offset_value_ptr++;
            float v10 = *offset_value_ptr >= 0 ? *(srcptr + static_cast<int>(*offset_value_ptr)) : 0;
            offset_value_ptr++;
            float v11 = *offset_value_ptr >= 0 ? *(srcptr + static_cast<int>(*offset_value_ptr)) : 0;
            offset_value_ptr++;

            float v0 = v00 * (1 - *offset_value_ptr) + v01 * *offset_value_ptr;
            float v1 = v10 * (1 - *offset_value_ptr) + v11 * *offset_value_ptr;
            offset_value_ptr++;

            *dstptr = v0 * (1 - *offset_value_ptr) + v1 * *offset_value_ptr;
            offset_value_ptr++;

            dstptr++;
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
            float v000 = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? *(srcptr + static_cast<int>(*offset_value_ptr)) : 0;
            offset_value_ptr++;
            float v001 = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? *(srcptr + static_cast<int>(*offset_value_ptr)) : 0;
            offset_value_ptr++;
            float v010 = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? *(srcptr + static_cast<int>(*offset_value_ptr)) : 0;
            offset_value_ptr++;
            float v011 = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? *(srcptr + static_cast<int>(*offset_value_ptr)) : 0;
            offset_value_ptr++;

            float v100 = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? *(srcptr + static_cast<int>(*offset_value_ptr)) : 0;
            offset_value_ptr++;
            float v101 = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? *(srcptr + static_cast<int>(*offset_value_ptr)) : 0;
            offset_value_ptr++;
            float v110 = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? *(srcptr + static_cast<int>(*offset_value_ptr)) : 0;
            offset_value_ptr++;
            float v111 = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? *(srcptr + static_cast<int>(*offset_value_ptr)) : 0;
            offset_value_ptr++;

            float v00 = v000 * (1 - *offset_value_ptr) + v001 * *offset_value_ptr;
            float v01 = v010 * (1 - *offset_value_ptr) + v011 * *offset_value_ptr;
            float v10 = v100 * (1 - *offset_value_ptr) + v101 * *offset_value_ptr;
            float v11 = v110 * (1 - *offset_value_ptr) + v111 * *offset_value_ptr;
            offset_value_ptr++;

            float v0 = v00 * (1 - *offset_value_ptr) + v01 * *offset_value_ptr;
            float v1 = v10 * (1 - *offset_value_ptr) + v11 * *offset_value_ptr;
            offset_value_ptr++;

            *dstptr = v0 * (1 - *offset_value_ptr) + v1 * *offset_value_ptr;
            offset_value_ptr++;

            dstptr++;
        }
    }
}