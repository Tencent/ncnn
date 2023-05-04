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
static void gridsample_2d_bilinear_apply_interpolation_p16(const Mat& src, Mat& dst, const Mat& offset, const Mat& value, const Option& opt)
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

        const float* offset_ptr_00 = offset.channel(0);
        const float* offset_ptr_01 = offset.channel(1);
        const float* offset_ptr_10 = offset.channel(2);
        const float* offset_ptr_11 = offset.channel(3);

        const float* value_ptr_alpha = value.channel(0);
        const float* value_ptr_beta = value.channel(1);

        for (int i = 0; i < grid_size; i++)
        {
            __m512i v00_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_ptr_00), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512i v01_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_ptr_01), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512i v10_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_ptr_10), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512i v11_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_ptr_11), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

            __mmask16 mask00 = *reinterpret_cast<const int*>(offset_ptr_00) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0);
            __mmask16 mask01 = *reinterpret_cast<const int*>(offset_ptr_01) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0);
            __mmask16 mask10 = *reinterpret_cast<const int*>(offset_ptr_10) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0);
            __mmask16 mask11 = *reinterpret_cast<const int*>(offset_ptr_11) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0);

            __m512 v00_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask00, v00_offset, srcptr, sizeof(float));
            __m512 v01_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask01, v01_offset, srcptr, sizeof(float));
            __m512 v10_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask10, v10_offset, srcptr, sizeof(float));
            __m512 v11_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask11, v11_offset, srcptr, sizeof(float));

            __m512 alpha = _mm512_set1_ps(*value_ptr_alpha);
            __m512 beta = _mm512_set1_ps(*value_ptr_beta);

            __m512 v0 = _mm512_fmadd_ps(v01_val, alpha, _mm512_fnmadd_ps(v00_val, alpha, v00_val));
            __m512 v1 = _mm512_fmadd_ps(v11_val, alpha, _mm512_fnmadd_ps(v10_val, alpha, v10_val));

            __m512 _v = _mm512_fmadd_ps(v1, beta, _mm512_fnmadd_ps(v0, beta, v0));
            _mm512_storeu_ps(dstptr, _v);

            offset_ptr_00++;
            offset_ptr_01++;
            offset_ptr_10++;
            offset_ptr_11++;

            value_ptr_alpha++;
            value_ptr_beta++;

            dstptr += 16;
        }
    }
}
static void gridsample_3d_bilinear_apply_interpolation_p16(const Mat& src, Mat& dst, const Mat& offset, const Mat& value, const Option& opt)
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

        const float* offset_ptr_000 = offset.channel(0);
        const float* offset_ptr_001 = offset.channel(1);
        const float* offset_ptr_010 = offset.channel(2);
        const float* offset_ptr_011 = offset.channel(3);
        const float* offset_ptr_100 = offset.channel(4);
        const float* offset_ptr_101 = offset.channel(5);
        const float* offset_ptr_110 = offset.channel(6);
        const float* offset_ptr_111 = offset.channel(7);

        const float* value_ptr_alpha = value.channel(0);
        const float* value_ptr_beta = value.channel(1);
        const float* value_ptr_gamma = value.channel(2);

        for (int i = 0; i < grid_size; i++)
        {
            __m512i v000_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_ptr_000), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512i v001_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_ptr_001), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512i v010_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_ptr_010), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512i v011_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_ptr_011), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512i v100_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_ptr_100), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512i v101_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_ptr_101), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512i v110_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_ptr_110), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
            __m512i v111_offset = _mm512_add_epi32(_mm512_set1_epi32(*offset_ptr_111), _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

            __m512 v000_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_ptr_000) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), v000_offset, srcptr, sizeof(float));
            __m512 v001_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_ptr_001) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), v001_offset, srcptr, sizeof(float));
            __m512 v010_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_ptr_010) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), v010_offset, srcptr, sizeof(float));
            __m512 v011_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_ptr_011) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), v011_offset, srcptr, sizeof(float));
            __m512 v100_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_ptr_100) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), v100_offset, srcptr, sizeof(float));
            __m512 v101_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_ptr_101) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), v101_offset, srcptr, sizeof(float));
            __m512 v110_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_ptr_110) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), v110_offset, srcptr, sizeof(float));
            __m512 v111_val = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), *reinterpret_cast<const int*>(offset_ptr_111) >= 0 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>(0x0), v111_offset, srcptr, sizeof(float));

            __m512 alpha = _mm512_set1_ps(*value_ptr_alpha);
            __m512 beta = _mm512_set1_ps(*value_ptr_beta);
            __m512 gamma = _mm512_set1_ps(*value_ptr_gamma);

            __m512 v00 = _mm512_fmadd_ps(v001_val, alpha, _mm512_fnmadd_ps(v000_val, alpha, v000_val));
            __m512 v01 = _mm512_fmadd_ps(v011_val, alpha, _mm512_fnmadd_ps(v010_val, alpha, v010_val));
            __m512 v10 = _mm512_fmadd_ps(v101_val, alpha, _mm512_fnmadd_ps(v100_val, alpha, v100_val));
            __m512 v11 = _mm512_fmadd_ps(v111_val, alpha, _mm512_fnmadd_ps(v110_val, alpha, v110_val));

            __m512 v0 = _mm512_fmadd_ps(v01, beta, _mm512_fnmadd_ps(v00, beta, v00));
            __m512 v1 = _mm512_fmadd_ps(v11, beta, _mm512_fnmadd_ps(v10, beta, v10));

            __m512 _v = _mm512_fmadd_ps(v1, gamma, _mm512_fnmadd_ps(v0, gamma, v0));
            _mm512_storeu_ps(dstptr, _v);

            offset_ptr_000++;
            offset_ptr_001++;
            offset_ptr_010++;
            offset_ptr_011++;

            offset_ptr_100++;
            offset_ptr_101++;
            offset_ptr_110++;
            offset_ptr_111++;

            value_ptr_alpha++;
            value_ptr_beta++;
            value_ptr_gamma++;

            dstptr += 16;
        }
    }
}

#endif // __AVX512F__

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
void gridsample_2d_bilinear_apply_interpolation_p8_avx2(const Mat& src, Mat& dst, const Mat& offset, const Mat& value, const Option& opt);
void gridsample_2d_bilinear_apply_interpolation_p4_avx2(const Mat& src, Mat& dst, const Mat& offset, const Mat& value, const Option& opt);
void gridsample_2d_bilinear_apply_interpolation_p1_avx2(const Mat& src, Mat& dst, const Mat& offset, const Mat& value, const Option& opt);

void gridsample_3d_bilinear_apply_interpolation_p8_avx2(const Mat& src, Mat& dst, const Mat& offset, const Mat& value, const Option& opt);
void gridsample_3d_bilinear_apply_interpolation_p4_avx2(const Mat& src, Mat& dst, const Mat& offset, const Mat& value, const Option& opt);
void gridsample_3d_bilinear_apply_interpolation_p1_avx2(const Mat& src, Mat& dst, const Mat& offset, const Mat& value, const Option& opt);
#endif

static void gridsample_2d_bilinear_apply_interpolation_p8(const Mat& src, Mat& dst, const Mat& offset, const Mat& value, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        gridsample_2d_bilinear_apply_interpolation_p8_avx2(src, dst, offset, value, opt);
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

        const float* offset_ptr_00 = offset.channel(0);
        const float* offset_ptr_01 = offset.channel(1);
        const float* offset_ptr_10 = offset.channel(2);
        const float* offset_ptr_11 = offset.channel(3);

        const float* value_ptr_alpha = value.channel(0);
        const float* value_ptr_beta = value.channel(1);

        for (int i = 0; i < grid_size; i++)
        {
#if __AVX2__
            __m256i v00_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_00), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v01_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_01), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v10_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_10), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v11_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_11), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
#else
            __m256i v00_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_ptr_00), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v01_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_ptr_01), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v10_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_ptr_10), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v11_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_ptr_11), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
#endif // __AVX2__

            float in_bound_00 = *reinterpret_cast<const int*>(offset_ptr_00) >= 0 ? -1.0f : 0.0f;
            float in_bound_01 = *reinterpret_cast<const int*>(offset_ptr_01) >= 0 ? -1.0f : 0.0f;
            float in_bound_10 = *reinterpret_cast<const int*>(offset_ptr_10) >= 0 ? -1.0f : 0.0f;
            float in_bound_11 = *reinterpret_cast<const int*>(offset_ptr_11) >= 0 ? -1.0f : 0.0f;

            __m256 v00_val = mask_gather_ps256(srcptr, v00_offset, _mm256_set1_ps(in_bound_00));
            __m256 v01_val = mask_gather_ps256(srcptr, v01_offset, _mm256_set1_ps(in_bound_01));
            __m256 v10_val = mask_gather_ps256(srcptr, v10_offset, _mm256_set1_ps(in_bound_10));
            __m256 v11_val = mask_gather_ps256(srcptr, v11_offset, _mm256_set1_ps(in_bound_11));

            __m256 alpha = _mm256_set1_ps(*value_ptr_alpha);
            __m256 beta = _mm256_set1_ps(*value_ptr_beta);

            __m256 v0 = _mm256_comp_fmadd_ps(v01_val, alpha, _mm256_comp_fnmadd_ps(v00_val, alpha, v00_val));
            __m256 v1 = _mm256_comp_fmadd_ps(v11_val, alpha, _mm256_comp_fnmadd_ps(v10_val, alpha, v10_val));

            __m256 _v = _mm256_comp_fmadd_ps(v1, beta, _mm256_comp_fnmadd_ps(v0, beta, v0));
            _mm256_storeu_ps(dstptr, _v);

            offset_ptr_00++;
            offset_ptr_01++;
            offset_ptr_10++;
            offset_ptr_11++;

            value_ptr_alpha++;
            value_ptr_beta++;

            dstptr += 8;
        }
    }
}
static void gridsample_3d_bilinear_apply_interpolation_p8(const Mat& src, Mat& dst, const Mat& offset, const Mat& value, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        gridsample_3d_bilinear_apply_interpolation_p8_avx2(src, dst, offset, value, opt);
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

        const float* offset_ptr_000 = offset.channel(0);
        const float* offset_ptr_001 = offset.channel(1);
        const float* offset_ptr_010 = offset.channel(2);
        const float* offset_ptr_011 = offset.channel(3);
        const float* offset_ptr_100 = offset.channel(4);
        const float* offset_ptr_101 = offset.channel(5);
        const float* offset_ptr_110 = offset.channel(6);
        const float* offset_ptr_111 = offset.channel(7);

        const float* value_ptr_alpha = value.channel(0);
        const float* value_ptr_beta = value.channel(1);
        const float* value_ptr_gamma = value.channel(2);

        for (int i = 0; i < grid_size; i++)
        {
#if __AVX2__
            __m256i v000_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_000), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v001_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_001), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v010_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_010), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v011_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_011), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v100_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_100), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v101_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_101), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v110_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_110), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            __m256i v111_offset = _mm256_add_epi32(_mm256_set1_epi32(*offset_ptr_111), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
#else
            __m256i v000_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_ptr_000), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v001_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_ptr_001), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v010_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_ptr_010), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v011_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_ptr_011), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v100_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_ptr_100), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v101_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_ptr_101), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v110_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_ptr_110), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
            __m256i v111_offset = _mm256_cvtps_epi32(_mm256_add_ps(_mm256_set1_ps(*offset_ptr_111), _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0)));
#endif // __AVX2__

            float in_bound_000 = *reinterpret_cast<const int*>(offset_ptr_000) >= 0 ? -1.0f : 0.0f;
            float in_bound_001 = *reinterpret_cast<const int*>(offset_ptr_001) >= 0 ? -1.0f : 0.0f;
            float in_bound_010 = *reinterpret_cast<const int*>(offset_ptr_010) >= 0 ? -1.0f : 0.0f;
            float in_bound_011 = *reinterpret_cast<const int*>(offset_ptr_011) >= 0 ? -1.0f : 0.0f;
            float in_bound_100 = *reinterpret_cast<const int*>(offset_ptr_100) >= 0 ? -1.0f : 0.0f;
            float in_bound_101 = *reinterpret_cast<const int*>(offset_ptr_101) >= 0 ? -1.0f : 0.0f;
            float in_bound_110 = *reinterpret_cast<const int*>(offset_ptr_110) >= 0 ? -1.0f : 0.0f;
            float in_bound_111 = *reinterpret_cast<const int*>(offset_ptr_111) >= 0 ? -1.0f : 0.0f;

            __m256 v000_val = mask_gather_ps256(srcptr, v000_offset, _mm256_set1_ps(in_bound_000));
            __m256 v001_val = mask_gather_ps256(srcptr, v001_offset, _mm256_set1_ps(in_bound_001));
            __m256 v010_val = mask_gather_ps256(srcptr, v010_offset, _mm256_set1_ps(in_bound_010));
            __m256 v011_val = mask_gather_ps256(srcptr, v011_offset, _mm256_set1_ps(in_bound_011));
            __m256 v100_val = mask_gather_ps256(srcptr, v100_offset, _mm256_set1_ps(in_bound_100));
            __m256 v101_val = mask_gather_ps256(srcptr, v101_offset, _mm256_set1_ps(in_bound_101));
            __m256 v110_val = mask_gather_ps256(srcptr, v110_offset, _mm256_set1_ps(in_bound_110));
            __m256 v111_val = mask_gather_ps256(srcptr, v111_offset, _mm256_set1_ps(in_bound_111));

            __m256 alpha = _mm256_set1_ps(*value_ptr_alpha);
            __m256 beta = _mm256_set1_ps(*value_ptr_beta);
            __m256 gamma = _mm256_set1_ps(*value_ptr_gamma);

            __m256 v00 = _mm256_comp_fmadd_ps(v001_val, alpha, _mm256_comp_fnmadd_ps(v000_val, alpha, v000_val));
            __m256 v01 = _mm256_comp_fmadd_ps(v011_val, alpha, _mm256_comp_fnmadd_ps(v010_val, alpha, v010_val));
            __m256 v10 = _mm256_comp_fmadd_ps(v101_val, alpha, _mm256_comp_fnmadd_ps(v100_val, alpha, v100_val));
            __m256 v11 = _mm256_comp_fmadd_ps(v111_val, alpha, _mm256_comp_fnmadd_ps(v110_val, alpha, v110_val));

            __m256 v0 = _mm256_comp_fmadd_ps(v01, beta, _mm256_comp_fnmadd_ps(v00, beta, v00));
            __m256 v1 = _mm256_comp_fmadd_ps(v11, beta, _mm256_comp_fnmadd_ps(v10, beta, v10));

            __m256 _v = _mm256_comp_fmadd_ps(v1, gamma, _mm256_comp_fnmadd_ps(v0, gamma, v0));
            _mm256_storeu_ps(dstptr, _v);

            offset_ptr_000++;
            offset_ptr_001++;
            offset_ptr_010++;
            offset_ptr_011++;

            offset_ptr_100++;
            offset_ptr_101++;
            offset_ptr_110++;
            offset_ptr_111++;

            value_ptr_alpha++;
            value_ptr_beta++;
            value_ptr_gamma++;

            dstptr += 8;
        }
    }
}
#endif // __AVX__
static void gridsample_2d_bilinear_apply_interpolation_p4(const Mat& src, Mat& dst, const Mat& offset, const Mat& value, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        gridsample_2d_bilinear_apply_interpolation_p4_avx2(src, dst, offset, value, opt);
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

        const float* offset_ptr_00 = offset.channel(0);
        const float* offset_ptr_01 = offset.channel(1);
        const float* offset_ptr_10 = offset.channel(2);
        const float* offset_ptr_11 = offset.channel(3);

        const float* value_ptr_alpha = value.channel(0);
        const float* value_ptr_beta = value.channel(1);

        for (int i = 0; i < grid_size; i++)
        {
            __m128i v00_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_00), _mm_set_epi32(3, 2, 1, 0));
            __m128i v01_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_01), _mm_set_epi32(3, 2, 1, 0));
            __m128i v10_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_10), _mm_set_epi32(3, 2, 1, 0));
            __m128i v11_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_11), _mm_set_epi32(3, 2, 1, 0));

            float in_bound_00 = *reinterpret_cast<const int*>(offset_ptr_00) >= 0 ? -1.0f : 0.0f;
            float in_bound_01 = *reinterpret_cast<const int*>(offset_ptr_01) >= 0 ? -1.0f : 0.0f;
            float in_bound_10 = *reinterpret_cast<const int*>(offset_ptr_10) >= 0 ? -1.0f : 0.0f;
            float in_bound_11 = *reinterpret_cast<const int*>(offset_ptr_11) >= 0 ? -1.0f : 0.0f;

            __m128 v00_val = mask_gather_ps(srcptr, v00_offset, _mm_set1_ps(in_bound_00));
            __m128 v01_val = mask_gather_ps(srcptr, v01_offset, _mm_set1_ps(in_bound_01));
            __m128 v10_val = mask_gather_ps(srcptr, v10_offset, _mm_set1_ps(in_bound_10));
            __m128 v11_val = mask_gather_ps(srcptr, v11_offset, _mm_set1_ps(in_bound_11));

            __m128 alpha = _mm_set1_ps(*value_ptr_alpha);
            __m128 beta = _mm_set1_ps(*value_ptr_beta);

            __m128 v0 = _mm_comp_fmadd_ps(v01_val, alpha, _mm_comp_fnmadd_ps(v00_val, alpha, v00_val));
            __m128 v1 = _mm_comp_fmadd_ps(v11_val, alpha, _mm_comp_fnmadd_ps(v10_val, alpha, v10_val));

            __m128 _v = _mm_comp_fmadd_ps(v1, beta, _mm_comp_fnmadd_ps(v0, beta, v0));
            _mm_storeu_ps(dstptr, _v);

            offset_ptr_00++;
            offset_ptr_01++;
            offset_ptr_10++;
            offset_ptr_11++;

            value_ptr_alpha++;
            value_ptr_beta++;

            dstptr += 4;
        }
    }
}
static void gridsample_3d_bilinear_apply_interpolation_p4(const Mat& src, Mat& dst, const Mat& offset, const Mat& value, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        gridsample_3d_bilinear_apply_interpolation_p4_avx2(src, dst, offset, value, opt);
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

        const float* offset_ptr_000 = offset.channel(0);
        const float* offset_ptr_001 = offset.channel(1);
        const float* offset_ptr_010 = offset.channel(2);
        const float* offset_ptr_011 = offset.channel(3);
        const float* offset_ptr_100 = offset.channel(4);
        const float* offset_ptr_101 = offset.channel(5);
        const float* offset_ptr_110 = offset.channel(6);
        const float* offset_ptr_111 = offset.channel(7);

        const float* value_ptr_alpha = value.channel(0);
        const float* value_ptr_beta = value.channel(1);
        const float* value_ptr_gamma = value.channel(2);

        for (int i = 0; i < grid_size; i++)
        {
            __m128i v000_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_000), _mm_set_epi32(3, 2, 1, 0));
            __m128i v001_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_001), _mm_set_epi32(3, 2, 1, 0));
            __m128i v010_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_010), _mm_set_epi32(3, 2, 1, 0));
            __m128i v011_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_011), _mm_set_epi32(3, 2, 1, 0));
            __m128i v100_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_100), _mm_set_epi32(3, 2, 1, 0));
            __m128i v101_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_101), _mm_set_epi32(3, 2, 1, 0));
            __m128i v110_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_110), _mm_set_epi32(3, 2, 1, 0));
            __m128i v111_offset = _mm_add_epi32(_mm_set1_epi32(*offset_ptr_111), _mm_set_epi32(3, 2, 1, 0));

            float in_bound_000 = *reinterpret_cast<const int*>(offset_ptr_000) >= 0 ? -1.0f : 0.0f;
            float in_bound_001 = *reinterpret_cast<const int*>(offset_ptr_001) >= 0 ? -1.0f : 0.0f;
            float in_bound_010 = *reinterpret_cast<const int*>(offset_ptr_010) >= 0 ? -1.0f : 0.0f;
            float in_bound_011 = *reinterpret_cast<const int*>(offset_ptr_011) >= 0 ? -1.0f : 0.0f;
            float in_bound_100 = *reinterpret_cast<const int*>(offset_ptr_100) >= 0 ? -1.0f : 0.0f;
            float in_bound_101 = *reinterpret_cast<const int*>(offset_ptr_101) >= 0 ? -1.0f : 0.0f;
            float in_bound_110 = *reinterpret_cast<const int*>(offset_ptr_110) >= 0 ? -1.0f : 0.0f;
            float in_bound_111 = *reinterpret_cast<const int*>(offset_ptr_111) >= 0 ? -1.0f : 0.0f;

            __m128 v000_val = mask_gather_ps(srcptr, v000_offset, _mm_set1_ps(in_bound_000));
            __m128 v001_val = mask_gather_ps(srcptr, v001_offset, _mm_set1_ps(in_bound_001));
            __m128 v010_val = mask_gather_ps(srcptr, v010_offset, _mm_set1_ps(in_bound_010));
            __m128 v011_val = mask_gather_ps(srcptr, v011_offset, _mm_set1_ps(in_bound_011));
            __m128 v100_val = mask_gather_ps(srcptr, v100_offset, _mm_set1_ps(in_bound_100));
            __m128 v101_val = mask_gather_ps(srcptr, v101_offset, _mm_set1_ps(in_bound_101));
            __m128 v110_val = mask_gather_ps(srcptr, v110_offset, _mm_set1_ps(in_bound_110));
            __m128 v111_val = mask_gather_ps(srcptr, v111_offset, _mm_set1_ps(in_bound_111));

            __m128 alpha = _mm_set1_ps(*value_ptr_alpha);
            __m128 beta = _mm_set1_ps(*value_ptr_beta);
            __m128 gamma = _mm_set1_ps(*value_ptr_gamma);

            __m128 v00 = _mm_comp_fmadd_ps(v001_val, alpha, _mm_comp_fnmadd_ps(v000_val, alpha, v000_val));
            __m128 v01 = _mm_comp_fmadd_ps(v011_val, alpha, _mm_comp_fnmadd_ps(v010_val, alpha, v010_val));
            __m128 v10 = _mm_comp_fmadd_ps(v101_val, alpha, _mm_comp_fnmadd_ps(v100_val, alpha, v100_val));
            __m128 v11 = _mm_comp_fmadd_ps(v111_val, alpha, _mm_comp_fnmadd_ps(v110_val, alpha, v110_val));

            __m128 v0 = _mm_comp_fmadd_ps(v01, beta, _mm_comp_fnmadd_ps(v00, beta, v00));
            __m128 v1 = _mm_comp_fmadd_ps(v11, beta, _mm_comp_fnmadd_ps(v10, beta, v10));

            __m128 _v = _mm_comp_fmadd_ps(v1, gamma, _mm_comp_fnmadd_ps(v0, gamma, v0));
            _mm_storeu_ps(dstptr, _v);

            offset_ptr_000++;
            offset_ptr_001++;
            offset_ptr_010++;
            offset_ptr_011++;

            offset_ptr_100++;
            offset_ptr_101++;
            offset_ptr_110++;
            offset_ptr_111++;

            value_ptr_alpha++;
            value_ptr_beta++;
            value_ptr_gamma++;

            dstptr += 4;
        }
    }
}
#endif // __SSE2__

static void gridsample_2d_bilinear_apply_interpolation_p1(const Mat& src, Mat& dst, const Mat& offset, const Mat& value, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        gridsample_2d_bilinear_apply_interpolation_p1_avx2(src, dst, offset, value, opt);
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

        const float* offset_ptr_00 = offset.channel(0);
        const float* offset_ptr_01 = offset.channel(1);
        const float* offset_ptr_10 = offset.channel(2);
        const float* offset_ptr_11 = offset.channel(3);

        const float* value_ptr_alpha = value.channel(0);
        const float* value_ptr_beta = value.channel(1);

        int x = 0;
#if __SSE2__
#if __AVX__

        for (; x + 7 < grid_size; x += 8)
        {
            __m256i v00_offset = _mm256_set_epi32(*(offset_ptr_00 + 7), *(offset_ptr_00 + 6), *(offset_ptr_00 + 5), *(offset_ptr_00 + 4), *(offset_ptr_00 + 3), *(offset_ptr_00 + 2), *(offset_ptr_00 + 1), *offset_ptr_00);
            __m256i v01_offset = _mm256_set_epi32(*(offset_ptr_01 + 7), *(offset_ptr_01 + 6), *(offset_ptr_01 + 5), *(offset_ptr_01 + 4), *(offset_ptr_01 + 3), *(offset_ptr_01 + 2), *(offset_ptr_01 + 1), *offset_ptr_01);
            __m256i v10_offset = _mm256_set_epi32(*(offset_ptr_10 + 7), *(offset_ptr_10 + 6), *(offset_ptr_10 + 5), *(offset_ptr_10 + 4), *(offset_ptr_10 + 3), *(offset_ptr_10 + 2), *(offset_ptr_10 + 1), *offset_ptr_10);
            __m256i v11_offset = _mm256_set_epi32(*(offset_ptr_11 + 7), *(offset_ptr_11 + 6), *(offset_ptr_11 + 5), *(offset_ptr_11 + 4), *(offset_ptr_11 + 3), *(offset_ptr_11 + 2), *(offset_ptr_11 + 1), *offset_ptr_11);

            __m256 v00_in_bound = _mm256_andnot_ps(_mm256_loadu_ps(offset_ptr_00), _mm256_set1_ps(-1.0f));
            __m256 v01_in_bound = _mm256_andnot_ps(_mm256_loadu_ps(offset_ptr_01), _mm256_set1_ps(-1.0f));
            __m256 v10_in_bound = _mm256_andnot_ps(_mm256_loadu_ps(offset_ptr_10), _mm256_set1_ps(-1.0f));
            __m256 v11_in_bound = _mm256_andnot_ps(_mm256_loadu_ps(offset_ptr_11), _mm256_set1_ps(-1.0f));

            __m256 v00_val = mask_gather_ps256(srcptr, v00_offset, v00_in_bound);
            __m256 v01_val = mask_gather_ps256(srcptr, v01_offset, v01_in_bound);
            __m256 v10_val = mask_gather_ps256(srcptr, v10_offset, v10_in_bound);
            __m256 v11_val = mask_gather_ps256(srcptr, v11_offset, v11_in_bound);

            __m256 alpha = _mm256_loadu_ps(value_ptr_alpha);
            __m256 beta = _mm256_loadu_ps(value_ptr_beta);

            __m256 v0 = _mm256_comp_fmadd_ps(v01_val, alpha, _mm256_comp_fnmadd_ps(v00_val, alpha, v00_val));
            __m256 v1 = _mm256_comp_fmadd_ps(v11_val, alpha, _mm256_comp_fnmadd_ps(v10_val, alpha, v10_val));

            __m256 _v = _mm256_comp_fmadd_ps(v1, beta, _mm256_comp_fnmadd_ps(v0, beta, v0));
            _mm256_storeu_ps(dstptr, _v);

            offset_ptr_00 += 8;
            offset_ptr_01 += 8;
            offset_ptr_10 += 8;
            offset_ptr_11 += 8;

            value_ptr_alpha += 8;
            value_ptr_beta += 8;

            dstptr += 8;
        }
#endif // __AVX__
        for (; x + 3 < grid_size; x += 4)
        {
            __m128i v00_offset = _mm_set_epi32(*(offset_ptr_00 + 3), *(offset_ptr_00 + 2), *(offset_ptr_00 + 1), *offset_ptr_00);
            __m128i v01_offset = _mm_set_epi32(*(offset_ptr_01 + 3), *(offset_ptr_01 + 2), *(offset_ptr_01 + 1), *offset_ptr_01);
            __m128i v10_offset = _mm_set_epi32(*(offset_ptr_10 + 3), *(offset_ptr_10 + 2), *(offset_ptr_10 + 1), *offset_ptr_10);
            __m128i v11_offset = _mm_set_epi32(*(offset_ptr_11 + 3), *(offset_ptr_11 + 2), *(offset_ptr_11 + 1), *offset_ptr_11);

            __m128 v00_in_bound = _mm_andnot_ps(_mm_loadu_ps(offset_ptr_00), _mm_set1_ps(-1.0f));
            __m128 v01_in_bound = _mm_andnot_ps(_mm_loadu_ps(offset_ptr_01), _mm_set1_ps(-1.0f));
            __m128 v10_in_bound = _mm_andnot_ps(_mm_loadu_ps(offset_ptr_10), _mm_set1_ps(-1.0f));
            __m128 v11_in_bound = _mm_andnot_ps(_mm_loadu_ps(offset_ptr_11), _mm_set1_ps(-1.0f));

            __m128 v00_val = mask_gather_ps(srcptr, v00_offset, v00_in_bound);
            __m128 v01_val = mask_gather_ps(srcptr, v01_offset, v01_in_bound);
            __m128 v10_val = mask_gather_ps(srcptr, v10_offset, v10_in_bound);
            __m128 v11_val = mask_gather_ps(srcptr, v11_offset, v11_in_bound);

            __m128 alpha = _mm_loadu_ps(value_ptr_alpha);
            __m128 beta = _mm_loadu_ps(value_ptr_beta);

            __m128 v0 = _mm_comp_fmadd_ps(v01_val, alpha, _mm_comp_fnmadd_ps(v00_val, alpha, v00_val));
            __m128 v1 = _mm_comp_fmadd_ps(v11_val, alpha, _mm_comp_fnmadd_ps(v10_val, alpha, v10_val));

            __m128 _v = _mm_comp_fmadd_ps(v1, beta, _mm_comp_fnmadd_ps(v0, beta, v0));
            _mm_storeu_ps(dstptr, _v);

            offset_ptr_00 += 4;
            offset_ptr_01 += 4;
            offset_ptr_10 += 4;
            offset_ptr_11 += 4;

            value_ptr_alpha += 4;
            value_ptr_beta += 4;

            dstptr += 4;
        }
#endif // __SSE2__
        for (; x < grid_size; x++)
        {
            float v00 = *offset_ptr_00 >= 0 ? *(srcptr + static_cast<int>(*offset_ptr_00)) : 0;
            float v01 = *offset_ptr_01 >= 0 ? *(srcptr + static_cast<int>(*offset_ptr_01)) : 0;
            float v10 = *offset_ptr_10 >= 0 ? *(srcptr + static_cast<int>(*offset_ptr_10)) : 0;
            float v11 = *offset_ptr_11 >= 0 ? *(srcptr + static_cast<int>(*offset_ptr_11)) : 0;

            float v0 = v00 * (1 - *value_ptr_alpha) + v01 * *value_ptr_alpha;
            float v1 = v10 * (1 - *value_ptr_alpha) + v11 * *value_ptr_alpha;

            *dstptr = v0 * (1 - *value_ptr_beta) + v1 * *value_ptr_beta;

            offset_ptr_00++;
            offset_ptr_01++;
            offset_ptr_10++;
            offset_ptr_11++;

            value_ptr_alpha++;
            value_ptr_beta++;
            dstptr++;
        }
    }
}
static void gridsample_3d_bilinear_apply_interpolation_p1(const Mat& src, Mat& dst, const Mat& offset, const Mat& value, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        gridsample_3d_bilinear_apply_interpolation_p1_avx2(src, dst, offset, value, opt);
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

        const float* offset_ptr_000 = offset.channel(0);
        const float* offset_ptr_001 = offset.channel(1);
        const float* offset_ptr_010 = offset.channel(2);
        const float* offset_ptr_011 = offset.channel(3);
        const float* offset_ptr_100 = offset.channel(4);
        const float* offset_ptr_101 = offset.channel(5);
        const float* offset_ptr_110 = offset.channel(6);
        const float* offset_ptr_111 = offset.channel(7);

        const float* value_ptr_alpha = value.channel(0);
        const float* value_ptr_beta = value.channel(1);
        const float* value_ptr_gamma = value.channel(2);

        int x = 0;
#if __SSE2__
#if __AVX__
        for (; x + 7 < grid_size; x += 8)
        {
            __m256i v000_offset = _mm256_set_epi32(*(offset_ptr_000 + 7), *(offset_ptr_000 + 6), *(offset_ptr_000 + 5), *(offset_ptr_000 + 4), *(offset_ptr_000 + 3), *(offset_ptr_000 + 2), *(offset_ptr_000 + 1), *offset_ptr_000);
            __m256i v001_offset = _mm256_set_epi32(*(offset_ptr_001 + 7), *(offset_ptr_001 + 6), *(offset_ptr_001 + 5), *(offset_ptr_001 + 4), *(offset_ptr_001 + 3), *(offset_ptr_001 + 2), *(offset_ptr_001 + 1), *offset_ptr_001);
            __m256i v010_offset = _mm256_set_epi32(*(offset_ptr_010 + 7), *(offset_ptr_010 + 6), *(offset_ptr_010 + 5), *(offset_ptr_010 + 4), *(offset_ptr_010 + 3), *(offset_ptr_010 + 2), *(offset_ptr_010 + 1), *offset_ptr_010);
            __m256i v011_offset = _mm256_set_epi32(*(offset_ptr_011 + 7), *(offset_ptr_011 + 6), *(offset_ptr_011 + 5), *(offset_ptr_011 + 4), *(offset_ptr_011 + 3), *(offset_ptr_011 + 2), *(offset_ptr_011 + 1), *offset_ptr_011);
            __m256i v100_offset = _mm256_set_epi32(*(offset_ptr_100 + 7), *(offset_ptr_100 + 6), *(offset_ptr_100 + 5), *(offset_ptr_100 + 4), *(offset_ptr_100 + 3), *(offset_ptr_100 + 2), *(offset_ptr_100 + 1), *offset_ptr_100);
            __m256i v101_offset = _mm256_set_epi32(*(offset_ptr_101 + 7), *(offset_ptr_101 + 6), *(offset_ptr_101 + 5), *(offset_ptr_101 + 4), *(offset_ptr_101 + 3), *(offset_ptr_101 + 2), *(offset_ptr_101 + 1), *offset_ptr_101);
            __m256i v110_offset = _mm256_set_epi32(*(offset_ptr_110 + 7), *(offset_ptr_110 + 6), *(offset_ptr_110 + 5), *(offset_ptr_110 + 4), *(offset_ptr_110 + 3), *(offset_ptr_110 + 2), *(offset_ptr_110 + 1), *offset_ptr_110);
            __m256i v111_offset = _mm256_set_epi32(*(offset_ptr_111 + 7), *(offset_ptr_111 + 6), *(offset_ptr_111 + 5), *(offset_ptr_111 + 4), *(offset_ptr_111 + 3), *(offset_ptr_111 + 2), *(offset_ptr_111 + 1), *offset_ptr_111);

            __m256 v000_in_bound = _mm256_andnot_ps(_mm256_loadu_ps(offset_ptr_000), _mm256_set1_ps(-1.0f));
            __m256 v001_in_bound = _mm256_andnot_ps(_mm256_loadu_ps(offset_ptr_001), _mm256_set1_ps(-1.0f));
            __m256 v010_in_bound = _mm256_andnot_ps(_mm256_loadu_ps(offset_ptr_010), _mm256_set1_ps(-1.0f));
            __m256 v011_in_bound = _mm256_andnot_ps(_mm256_loadu_ps(offset_ptr_011), _mm256_set1_ps(-1.0f));
            __m256 v100_in_bound = _mm256_andnot_ps(_mm256_loadu_ps(offset_ptr_100), _mm256_set1_ps(-1.0f));
            __m256 v101_in_bound = _mm256_andnot_ps(_mm256_loadu_ps(offset_ptr_101), _mm256_set1_ps(-1.0f));
            __m256 v110_in_bound = _mm256_andnot_ps(_mm256_loadu_ps(offset_ptr_110), _mm256_set1_ps(-1.0f));
            __m256 v111_in_bound = _mm256_andnot_ps(_mm256_loadu_ps(offset_ptr_111), _mm256_set1_ps(-1.0f));

            __m256 v000_val = mask_gather_ps256(srcptr, v000_offset, v000_in_bound);
            __m256 v001_val = mask_gather_ps256(srcptr, v001_offset, v001_in_bound);
            __m256 v010_val = mask_gather_ps256(srcptr, v010_offset, v010_in_bound);
            __m256 v011_val = mask_gather_ps256(srcptr, v011_offset, v011_in_bound);
            __m256 v100_val = mask_gather_ps256(srcptr, v100_offset, v100_in_bound);
            __m256 v101_val = mask_gather_ps256(srcptr, v101_offset, v101_in_bound);
            __m256 v110_val = mask_gather_ps256(srcptr, v110_offset, v110_in_bound);
            __m256 v111_val = mask_gather_ps256(srcptr, v111_offset, v111_in_bound);

            __m256 alpha = _mm256_loadu_ps(value_ptr_alpha);
            __m256 beta = _mm256_loadu_ps(value_ptr_beta);
            __m256 gamma = _mm256_loadu_ps(value_ptr_gamma);

            __m256 v00 = _mm256_comp_fmadd_ps(v001_val, alpha, _mm256_comp_fnmadd_ps(v000_val, alpha, v000_val));
            __m256 v01 = _mm256_comp_fmadd_ps(v011_val, alpha, _mm256_comp_fnmadd_ps(v010_val, alpha, v010_val));
            __m256 v10 = _mm256_comp_fmadd_ps(v101_val, alpha, _mm256_comp_fnmadd_ps(v100_val, alpha, v100_val));
            __m256 v11 = _mm256_comp_fmadd_ps(v111_val, alpha, _mm256_comp_fnmadd_ps(v110_val, alpha, v110_val));

            __m256 v0 = _mm256_comp_fmadd_ps(v01, beta, _mm256_comp_fnmadd_ps(v00, beta, v00));
            __m256 v1 = _mm256_comp_fmadd_ps(v11, beta, _mm256_comp_fnmadd_ps(v10, beta, v10));

            __m256 _v = _mm256_comp_fmadd_ps(v1, gamma, _mm256_comp_fnmadd_ps(v0, gamma, v0));
            _mm256_storeu_ps(dstptr, _v);

            offset_ptr_000 += 8;
            offset_ptr_001 += 8;
            offset_ptr_010 += 8;
            offset_ptr_011 += 8;

            offset_ptr_100 += 8;
            offset_ptr_101 += 8;
            offset_ptr_110 += 8;
            offset_ptr_111 += 8;

            value_ptr_alpha += 8;
            value_ptr_beta += 8;
            value_ptr_gamma += 8;

            dstptr += 8;
        }

#endif // __AVX__
        for (; x + 3 < grid_size; x += 4)
        {
            __m128i v000_offset = _mm_set_epi32(*(offset_ptr_000 + 3), *(offset_ptr_000 + 2), *(offset_ptr_000 + 1), *offset_ptr_000);
            __m128i v001_offset = _mm_set_epi32(*(offset_ptr_001 + 3), *(offset_ptr_001 + 2), *(offset_ptr_001 + 1), *offset_ptr_001);
            __m128i v010_offset = _mm_set_epi32(*(offset_ptr_010 + 3), *(offset_ptr_010 + 2), *(offset_ptr_010 + 1), *offset_ptr_010);
            __m128i v011_offset = _mm_set_epi32(*(offset_ptr_011 + 3), *(offset_ptr_011 + 2), *(offset_ptr_011 + 1), *offset_ptr_011);
            __m128i v100_offset = _mm_set_epi32(*(offset_ptr_100 + 3), *(offset_ptr_100 + 2), *(offset_ptr_100 + 1), *offset_ptr_100);
            __m128i v101_offset = _mm_set_epi32(*(offset_ptr_101 + 3), *(offset_ptr_101 + 2), *(offset_ptr_101 + 1), *offset_ptr_101);
            __m128i v110_offset = _mm_set_epi32(*(offset_ptr_110 + 3), *(offset_ptr_110 + 2), *(offset_ptr_110 + 1), *offset_ptr_110);
            __m128i v111_offset = _mm_set_epi32(*(offset_ptr_111 + 3), *(offset_ptr_111 + 2), *(offset_ptr_111 + 1), *offset_ptr_111);

            __m128 v000_in_bound = _mm_andnot_ps(_mm_loadu_ps(offset_ptr_000), _mm_set1_ps(-1.0f));
            __m128 v001_in_bound = _mm_andnot_ps(_mm_loadu_ps(offset_ptr_001), _mm_set1_ps(-1.0f));
            __m128 v010_in_bound = _mm_andnot_ps(_mm_loadu_ps(offset_ptr_010), _mm_set1_ps(-1.0f));
            __m128 v011_in_bound = _mm_andnot_ps(_mm_loadu_ps(offset_ptr_011), _mm_set1_ps(-1.0f));
            __m128 v100_in_bound = _mm_andnot_ps(_mm_loadu_ps(offset_ptr_100), _mm_set1_ps(-1.0f));
            __m128 v101_in_bound = _mm_andnot_ps(_mm_loadu_ps(offset_ptr_101), _mm_set1_ps(-1.0f));
            __m128 v110_in_bound = _mm_andnot_ps(_mm_loadu_ps(offset_ptr_110), _mm_set1_ps(-1.0f));
            __m128 v111_in_bound = _mm_andnot_ps(_mm_loadu_ps(offset_ptr_111), _mm_set1_ps(-1.0f));

            __m128 v000_val = mask_gather_ps(srcptr, v000_offset, v000_in_bound);
            __m128 v001_val = mask_gather_ps(srcptr, v001_offset, v001_in_bound);
            __m128 v010_val = mask_gather_ps(srcptr, v010_offset, v010_in_bound);
            __m128 v011_val = mask_gather_ps(srcptr, v011_offset, v011_in_bound);
            __m128 v100_val = mask_gather_ps(srcptr, v100_offset, v100_in_bound);
            __m128 v101_val = mask_gather_ps(srcptr, v101_offset, v101_in_bound);
            __m128 v110_val = mask_gather_ps(srcptr, v110_offset, v110_in_bound);
            __m128 v111_val = mask_gather_ps(srcptr, v111_offset, v111_in_bound);

            __m128 alpha = _mm_loadu_ps(value_ptr_alpha);
            __m128 beta = _mm_loadu_ps(value_ptr_beta);
            __m128 gamma = _mm_loadu_ps(value_ptr_gamma);

            __m128 v00 = _mm_comp_fmadd_ps(v001_val, alpha, _mm_comp_fnmadd_ps(v000_val, alpha, v000_val));
            __m128 v01 = _mm_comp_fmadd_ps(v011_val, alpha, _mm_comp_fnmadd_ps(v010_val, alpha, v010_val));
            __m128 v10 = _mm_comp_fmadd_ps(v101_val, alpha, _mm_comp_fnmadd_ps(v100_val, alpha, v100_val));
            __m128 v11 = _mm_comp_fmadd_ps(v111_val, alpha, _mm_comp_fnmadd_ps(v110_val, alpha, v110_val));

            __m128 v0 = _mm_comp_fmadd_ps(v01, beta, _mm_comp_fnmadd_ps(v00, beta, v00));
            __m128 v1 = _mm_comp_fmadd_ps(v11, beta, _mm_comp_fnmadd_ps(v10, beta, v10));

            __m128 _v = _mm_comp_fmadd_ps(v1, gamma, _mm_comp_fnmadd_ps(v0, gamma, v0));
            _mm_storeu_ps(dstptr, _v);

            offset_ptr_000 += 4;
            offset_ptr_001 += 4;
            offset_ptr_010 += 4;
            offset_ptr_011 += 4;

            offset_ptr_100 += 4;
            offset_ptr_101 += 4;
            offset_ptr_110 += 4;
            offset_ptr_111 += 4;

            value_ptr_alpha += 4;
            value_ptr_beta += 4;
            value_ptr_gamma += 4;

            dstptr += 4;
        }
#endif // __SSE2__
        for (; x < grid_size; x++)
        {
            float v000 = *reinterpret_cast<const int*>(offset_ptr_000) >= 0 ? *(srcptr + static_cast<int>(*offset_ptr_000)) : 0;
            float v001 = *reinterpret_cast<const int*>(offset_ptr_001) >= 0 ? *(srcptr + static_cast<int>(*offset_ptr_001)) : 0;
            float v010 = *reinterpret_cast<const int*>(offset_ptr_010) >= 0 ? *(srcptr + static_cast<int>(*offset_ptr_010)) : 0;
            float v011 = *reinterpret_cast<const int*>(offset_ptr_011) >= 0 ? *(srcptr + static_cast<int>(*offset_ptr_011)) : 0;
                                                                       
            float v100 = *reinterpret_cast<const int*>(offset_ptr_100) >= 0 ? *(srcptr + static_cast<int>(*offset_ptr_100)) : 0;
            float v101 = *reinterpret_cast<const int*>(offset_ptr_101) >= 0 ? *(srcptr + static_cast<int>(*offset_ptr_101)) : 0;
            float v110 = *reinterpret_cast<const int*>(offset_ptr_110) >= 0 ? *(srcptr + static_cast<int>(*offset_ptr_110)) : 0;
            float v111 = *reinterpret_cast<const int*>(offset_ptr_111) >= 0 ? *(srcptr + static_cast<int>(*offset_ptr_111)) : 0;

            float v00 = v000 * (1 - *value_ptr_alpha) + v001 * *value_ptr_alpha;
            float v01 = v010 * (1 - *value_ptr_alpha) + v011 * *value_ptr_alpha;
            float v10 = v100 * (1 - *value_ptr_alpha) + v101 * *value_ptr_alpha;
            float v11 = v110 * (1 - *value_ptr_alpha) + v111 * *value_ptr_alpha;

            float v0 = v00 * (1 - *value_ptr_beta) + v01 * *value_ptr_beta;
            float v1 = v10 * (1 - *value_ptr_beta) + v11 * *value_ptr_beta;

            *dstptr = v0 * (1 - *value_ptr_gamma) + v1 * *value_ptr_gamma;

            offset_ptr_000++;
            offset_ptr_001++;
            offset_ptr_010++;
            offset_ptr_011++;

            offset_ptr_100++;
            offset_ptr_101++;
            offset_ptr_110++;
            offset_ptr_111++;

            value_ptr_alpha++;
            value_ptr_beta++;
            value_ptr_gamma++;
            dstptr++;
        }
    }
}