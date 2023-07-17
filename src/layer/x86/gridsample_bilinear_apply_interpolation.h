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
            __m512i v00_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_value_ptr), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __mmask16 mask00 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0);

            __m512i v01_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_value_ptr), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __mmask16 mask01 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0);

            __m512i v10_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_value_ptr), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __mmask16 mask10 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0);

            __m512i v11_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_value_ptr), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __mmask16 mask11 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0);

            __m512 v00_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask00, v00_offset, srcptr, sizeof(float));
            __m512 v01_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask01, v01_offset, srcptr, sizeof(float));
            __m512 v10_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask10, v10_offset, srcptr, sizeof(float));
            __m512 v11_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask11, v11_offset, srcptr, sizeof(float));

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
            __m512i v000_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_value_ptr), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512 v000_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), v000_offset, srcptr, sizeof(float));

            __m512i v001_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_value_ptr), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512 v001_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), v001_offset, srcptr, sizeof(float));

            __m512i v010_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_value_ptr), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512 v010_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), v010_offset, srcptr, sizeof(float));

            __m512i v011_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_value_ptr), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512 v011_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), v011_offset, srcptr, sizeof(float));

            __m512i v100_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_value_ptr), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512 v100_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), v100_offset, srcptr, sizeof(float));

            __m512i v101_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_value_ptr), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512 v101_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), v101_offset, srcptr, sizeof(float));

            __m512i v110_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_value_ptr), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512 v110_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), v110_offset, srcptr, sizeof(float));

            __m512i v111_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_value_ptr), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512 v111_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), v111_offset, srcptr, sizeof(float));

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

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
void gridsample_2d_bilinear_apply_interpolation_p8_avx2(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt);
void gridsample_2d_bilinear_apply_interpolation_p4_avx2(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt);

void gridsample_3d_bilinear_apply_interpolation_p8_avx2(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt);
void gridsample_3d_bilinear_apply_interpolation_p4_avx2(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt);
#endif

static void gridsample_2d_bilinear_apply_interpolation_p8(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        gridsample_2d_bilinear_apply_interpolation_p8_avx2(src, dst, offset_value, opt);
        return;
    }
#endif

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
            float in_bound_00 = *reinterpret_cast<const int*>(offset_value_ptr) >= 0 ? -1.0f : 0.0f;
            float in_bound_01 = *reinterpret_cast<const int*>(offset_value_ptr + 1) >= 0 ? -1.0f : 0.0f;
            float in_bound_10 = *reinterpret_cast<const int*>(offset_value_ptr + 2) >= 0 ? -1.0f : 0.0f;
            float in_bound_11 = *reinterpret_cast<const int*>(offset_value_ptr + 3) >= 0 ? -1.0f : 0.0f;
#if __AVX2__
            __m256i v00_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_value_ptr++), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v01_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_value_ptr++), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v10_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_value_ptr++), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v11_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_value_ptr++), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
#else
            __m256i v00_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_value_ptr++), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v01_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_value_ptr++), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v10_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_value_ptr++), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v11_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_value_ptr++), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
#endif // __AVX2__

            __m256 v00_val = mask_gather_ps256(srcptr, v00_offset, _mm256_set1_ps(in_bound_00));
            __m256 v01_val = mask_gather_ps256(srcptr, v01_offset, _mm256_set1_ps(in_bound_01));
            __m256 v10_val = mask_gather_ps256(srcptr, v10_offset, _mm256_set1_ps(in_bound_10));
            __m256 v11_val = mask_gather_ps256(srcptr, v11_offset, _mm256_set1_ps(in_bound_11));

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
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        gridsample_3d_bilinear_apply_interpolation_p8_avx2(src, dst, offset_value, opt);
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

        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
#if __AVX2__
            __m256i v000_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_value_ptr), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v001_offset = _mm256_add_epi32(_mm256_set1_epi32(*(offset_value_ptr + 1)), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v010_offset = _mm256_add_epi32(_mm256_set1_epi32(*(offset_value_ptr + 2)), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v011_offset = _mm256_add_epi32(_mm256_set1_epi32(*(offset_value_ptr + 3)), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v100_offset = _mm256_add_epi32(_mm256_set1_epi32(*(offset_value_ptr + 4)), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v101_offset = _mm256_add_epi32(_mm256_set1_epi32(*(offset_value_ptr + 5)), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v110_offset = _mm256_add_epi32(_mm256_set1_epi32(*(offset_value_ptr + 6)), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v111_offset = _mm256_add_epi32(_mm256_set1_epi32(*(offset_value_ptr + 7)), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
#else
            __m256i v000_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_value_ptr), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v001_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*(offset_value_ptr + 1)), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v010_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*(offset_value_ptr + 2)), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v011_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*(offset_value_ptr + 3)), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v100_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*(offset_value_ptr + 4)), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v101_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*(offset_value_ptr + 5)), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v110_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*(offset_value_ptr + 6)), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v111_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*(offset_value_ptr + 7)), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
#endif // __AVX2__

            float in_bound_000 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;
            float in_bound_001 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;
            float in_bound_010 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;
            float in_bound_011 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;
            float in_bound_100 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;
            float in_bound_101 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;
            float in_bound_110 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;
            float in_bound_111 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;

            __m256 v000_val = mask_gather_ps256(srcptr, v000_offset, _mm256_set1_ps(in_bound_000));
            __m256 v001_val = mask_gather_ps256(srcptr, v001_offset, _mm256_set1_ps(in_bound_001));
            __m256 v010_val = mask_gather_ps256(srcptr, v010_offset, _mm256_set1_ps(in_bound_010));
            __m256 v011_val = mask_gather_ps256(srcptr, v011_offset, _mm256_set1_ps(in_bound_011));
            __m256 v100_val = mask_gather_ps256(srcptr, v100_offset, _mm256_set1_ps(in_bound_100));
            __m256 v101_val = mask_gather_ps256(srcptr, v101_offset, _mm256_set1_ps(in_bound_101));
            __m256 v110_val = mask_gather_ps256(srcptr, v110_offset, _mm256_set1_ps(in_bound_110));
            __m256 v111_val = mask_gather_ps256(srcptr, v111_offset, _mm256_set1_ps(in_bound_111));

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
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        gridsample_2d_bilinear_apply_interpolation_p4_avx2(src, dst, offset_value, opt);
        return;
    }
#endif

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
            __m128i v00_offset = _mm_add_epi32(_mm_set1_epi32(*offset_value_ptr), _mm_set_epi32(3, 2, 1, 0));
            float in_bound_00 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;

            __m128i v01_offset = _mm_add_epi32(_mm_set1_epi32(*offset_value_ptr), _mm_set_epi32(3, 2, 1, 0));
            float in_bound_01 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;

            __m128i v10_offset = _mm_add_epi32(_mm_set1_epi32(*offset_value_ptr), _mm_set_epi32(3, 2, 1, 0));
            float in_bound_10 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;

            __m128i v11_offset = _mm_add_epi32(_mm_set1_epi32(*offset_value_ptr), _mm_set_epi32(3, 2, 1, 0));
            float in_bound_11 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;

            __m128 v00_val = mask_gather_ps(srcptr, v00_offset, _mm_set1_ps(in_bound_00));
            __m128 v01_val = mask_gather_ps(srcptr, v01_offset, _mm_set1_ps(in_bound_01));
            __m128 v10_val = mask_gather_ps(srcptr, v10_offset, _mm_set1_ps(in_bound_10));
            __m128 v11_val = mask_gather_ps(srcptr, v11_offset, _mm_set1_ps(in_bound_11));

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
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        gridsample_3d_bilinear_apply_interpolation_p4_avx2(src, dst, offset_value, opt);
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

        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            __m128i v000_offset = _mm_add_epi32(_mm_set1_epi32(*offset_value_ptr), _mm_set_epi32(3, 2, 1, 0));
            float in_bound_000 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;

            __m128i v001_offset = _mm_add_epi32(_mm_set1_epi32(*offset_value_ptr), _mm_set_epi32(3, 2, 1, 0));
            float in_bound_001 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;

            __m128i v010_offset = _mm_add_epi32(_mm_set1_epi32(*offset_value_ptr), _mm_set_epi32(3, 2, 1, 0));
            float in_bound_010 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;

            __m128i v011_offset = _mm_add_epi32(_mm_set1_epi32(*offset_value_ptr), _mm_set_epi32(3, 2, 1, 0));
            float in_bound_011 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;

            __m128i v100_offset = _mm_add_epi32(_mm_set1_epi32(*offset_value_ptr), _mm_set_epi32(3, 2, 1, 0));
            float in_bound_100 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;

            __m128i v101_offset = _mm_add_epi32(_mm_set1_epi32(*offset_value_ptr), _mm_set_epi32(3, 2, 1, 0));
            float in_bound_101 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;

            __m128i v110_offset = _mm_add_epi32(_mm_set1_epi32(*offset_value_ptr), _mm_set_epi32(3, 2, 1, 0));
            float in_bound_110 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;

            __m128i v111_offset = _mm_add_epi32(_mm_set1_epi32(*offset_value_ptr), _mm_set_epi32(3, 2, 1, 0));
            float in_bound_111 = *reinterpret_cast<const int*>(offset_value_ptr++) >= 0 ? -1.0f : 0.0f;

            __m128 v000_val = mask_gather_ps(srcptr, v000_offset, _mm_set1_ps(in_bound_000));
            __m128 v001_val = mask_gather_ps(srcptr, v001_offset, _mm_set1_ps(in_bound_001));
            __m128 v010_val = mask_gather_ps(srcptr, v010_offset, _mm_set1_ps(in_bound_010));
            __m128 v011_val = mask_gather_ps(srcptr, v011_offset, _mm_set1_ps(in_bound_011));
            __m128 v100_val = mask_gather_ps(srcptr, v100_offset, _mm_set1_ps(in_bound_100));
            __m128 v101_val = mask_gather_ps(srcptr, v101_offset, _mm_set1_ps(in_bound_101));
            __m128 v110_val = mask_gather_ps(srcptr, v110_offset, _mm_set1_ps(in_bound_110));
            __m128 v111_val = mask_gather_ps(srcptr, v111_offset, _mm_set1_ps(in_bound_111));

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