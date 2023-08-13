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
static void cubic_interp1d_p16(__m512& coeffs0, __m512& coeffs1, __m512& coeffs2, __m512& coeffs3, const __m512& tx)
{
    const __m512 A = _mm512_set1_ps(-0.75f);

    const __m512 x0 = _mm512_add_ps(tx, _mm512_set1_ps(1.0f));
    const __m512& x1 = tx;
    const __m512 x2 = _mm512_sub_ps(_mm512_set1_ps(1.0f), tx);

    coeffs0 = _mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(_mm512_mul_ps(_mm512_sub_ps(_mm512_mul_ps(A, x0), _mm512_mul_ps(_mm512_set1_ps(5.0f), A)), x0), _mm512_mul_ps(_mm512_set1_ps(8.0f), A)), x0), _mm512_mul_ps(_mm512_set1_ps(4), A));
    coeffs1 = _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(A, _mm512_set1_ps(2.0f)), x1), _mm512_add_ps(A, _mm512_set1_ps(3.0f))), x1), x1), _mm512_set1_ps(1.0f));
    coeffs2 = _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(_mm512_mul_ps(_mm512_add_ps(A, _mm512_set1_ps(2.0f)), x2), _mm512_add_ps(A, _mm512_set1_ps(3.0f))), x2), x2), _mm512_set1_ps(1.0f));
    coeffs3 = _mm512_sub_ps(_mm512_sub_ps(_mm512_sub_ps(_mm512_set1_ps(1.0f), coeffs0), coeffs1), coeffs2);
}

static void gridsample_2d_bicubic_apply_interpolation_p16(const Mat& src, Mat& dst, Mat& offset_value, const Option& opt)
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
            __m512 x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3;
            __m512 y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3;
            __m512 value_f[4];
            cubic_interp1d_p16(x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3, _mm512_set1_ps(offset_value_ptr[0]));
            cubic_interp1d_p16(y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3, _mm512_set1_ps(offset_value_ptr[1]));

            const int* offset_ptr = (int*)offset_value_ptr + 2;

            for (int ii = 0; ii < 4; ii++)
            {
                __m512 x0_val = offset_ptr[0] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[0]) : _mm512_set1_ps(0);
                __m512 x1_val = offset_ptr[1] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[1]) : _mm512_set1_ps(0);
                __m512 x2_val = offset_ptr[2] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[2]) : _mm512_set1_ps(0);
                __m512 x3_val = offset_ptr[3] >= 0 ? _mm512_loadu_ps(srcptr + offset_ptr[3]) : _mm512_set1_ps(0);

                value_f[ii] = _mm512_mul_ps(x_coeffs0, x0_val);
                value_f[ii] = _mm512_fmadd_ps(x_coeffs1, x1_val, value_f[ii]);
                value_f[ii] = _mm512_fmadd_ps(x_coeffs2, x2_val, value_f[ii]);
                value_f[ii] = _mm512_fmadd_ps(x_coeffs3, x3_val, value_f[ii]);

                offset_ptr += 4;
            }

            __m512 _v = _mm512_mul_ps(y_coeffs0, value_f[0]);
            _v = _mm512_fmadd_ps(y_coeffs1, value_f[1], _v);
            _v = _mm512_fmadd_ps(y_coeffs2, value_f[2], _v);
            _v = _mm512_fmadd_ps(y_coeffs3, value_f[3], _v);
            _mm512_storeu_ps(dstptr, _v);

            dstptr += 16;
            offset_value_ptr += 18;
        }
    }
}

#endif // __AVX512F__
static void cubic_interp1d_p8(__m256& coeffs0, __m256& coeffs1, __m256& coeffs2, __m256& coeffs3, const __m256& tx)
{
    const __m256 A = _mm256_set1_ps(-0.75f);

    const __m256 x0 = _mm256_add_ps(tx, _mm256_set1_ps(1));
    const __m256& x1 = tx;
    const __m256 x2 = _mm256_sub_ps(_mm256_set1_ps(1), tx);
    //const __m256 x3 = _mm256_add_ps(x2, _mm256_set1_ps(1));

    coeffs0 = _mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(A, x0), _mm256_mul_ps(_mm256_set1_ps(5.0f), A)), x0), _mm256_mul_ps(_mm256_set1_ps(8.0f), A)), x0), _mm256_mul_ps(_mm256_set1_ps(4), A));
    coeffs1 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(A, _mm256_set1_ps(2.0f)), x1), _mm256_add_ps(A, _mm256_set1_ps(3.0f))), x1), x1), _mm256_set1_ps(1));
    coeffs2 = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_add_ps(A, _mm256_set1_ps(2.0f)), x2), _mm256_add_ps(A, _mm256_set1_ps(3.0f))), x2), x2), _mm256_set1_ps(1));
    coeffs3 = _mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_set1_ps(1), coeffs0), coeffs1), coeffs2);
}

static void gridsample_2d_bicubic_apply_interpolation_p8(const Mat& src, Mat& dst, Mat& offset_value, const Option& opt)
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
            __m256 x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3;
            __m256 y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3;
            __m256 value_f[4];
            cubic_interp1d_p8(x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3, _mm256_set1_ps(offset_value_ptr[0]));
            cubic_interp1d_p8(y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3, _mm256_set1_ps(offset_value_ptr[1]));

            const int* offset_ptr = (int*)offset_value_ptr + 2;

            for (int ii = 0; ii < 4; ii++)
            {
                __m256 x0_val = offset_ptr[0] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[0]) : _mm256_set1_ps(0);
                __m256 x1_val = offset_ptr[1] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[1]) : _mm256_set1_ps(0);
                __m256 x2_val = offset_ptr[2] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[2]) : _mm256_set1_ps(0);
                __m256 x3_val = offset_ptr[3] >= 0 ? _mm256_loadu_ps(srcptr + offset_ptr[3]) : _mm256_set1_ps(0);

                value_f[ii] = _mm256_mul_ps(x_coeffs0, x0_val);
                value_f[ii] = _mm256_comp_fmadd_ps(x_coeffs1, x1_val, value_f[ii]);
                value_f[ii] = _mm256_comp_fmadd_ps(x_coeffs2, x2_val, value_f[ii]);
                value_f[ii] = _mm256_comp_fmadd_ps(x_coeffs3, x3_val, value_f[ii]);

                offset_ptr += 4;
            }

            __m256 _v = _mm256_mul_ps(y_coeffs0, value_f[0]);
            _v = _mm256_comp_fmadd_ps(y_coeffs1, value_f[1], _v);
            _v = _mm256_comp_fmadd_ps(y_coeffs2, value_f[2], _v);
            _v = _mm256_comp_fmadd_ps(y_coeffs3, value_f[3], _v);
            _mm256_storeu_ps(dstptr, _v);

            dstptr += 8;
            offset_value_ptr += 18;
        }
    }
}

#endif // __AVX__
static void cubic_interp1d_p4(__m128& coeffs0, __m128& coeffs1, __m128& coeffs2, __m128& coeffs3, const __m128& tx)
{
    const __m128 A = _mm_set_ps1(-0.75f);

    const __m128 x0 = _mm_add_ps(tx, _mm_set_ps1(1.0f));
    const __m128& x1 = tx;
    const __m128 x2 = _mm_sub_ps(_mm_set_ps1(1.0f), tx);
    //const __m128 x3 = _mm_add_ps(x2, _mm_set_ps1(1.0f));

    coeffs0 = _mm_sub_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(A, x0), _mm_mul_ps(_mm_set_ps1(5.0f), A)), x0), _mm_mul_ps(_mm_set_ps1(8.0f), A)), x0), _mm_mul_ps(_mm_set_ps1(4), A));
    coeffs1 = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(_mm_add_ps(A, _mm_set_ps1(2.0f)), x1), _mm_add_ps(A, _mm_set_ps1(3.0f))), x1), x1), _mm_set_ps1(1.0f));
    coeffs2 = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_mm_mul_ps(_mm_add_ps(A, _mm_set_ps1(2.0f)), x2), _mm_add_ps(A, _mm_set_ps1(3.0f))), x2), x2), _mm_set_ps1(1.0f));
    coeffs3 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(_mm_set_ps1(1.0f), coeffs0), coeffs1), coeffs2);
}

static void gridsample_2d_bicubic_apply_interpolation_p4(const Mat& src, Mat& dst, Mat& offset_value, const Option& opt)
{
    const int channels = dst.c;
    const int outw = dst.w;
    const int outh = dst.h;
    const int grid_size = outw * outh;

    __m128 x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3;
    __m128 y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3;
    __m128 value_f[4];

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);

        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            cubic_interp1d_p4(x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3, _mm_set_ps1(offset_value_ptr[0]));
            cubic_interp1d_p4(y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3, _mm_set_ps1(offset_value_ptr[1]));

            const int* offset_ptr = (int*)offset_value_ptr + 2;

            for (int ii = 0; ii < 4; ii++)
            {
                __m128 x0_val = offset_ptr[0] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[0]) : _mm_set1_ps(0);
                __m128 x1_val = offset_ptr[1] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[1]) : _mm_set1_ps(0);
                __m128 x2_val = offset_ptr[2] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[2]) : _mm_set1_ps(0);
                __m128 x3_val = offset_ptr[3] >= 0 ? _mm_loadu_ps(srcptr + offset_ptr[3]) : _mm_set1_ps(0);

                value_f[ii] = _mm_mul_ps(x_coeffs0, x0_val);
                value_f[ii] = _mm_comp_fmadd_ps(x_coeffs1, x1_val, value_f[ii]);
                value_f[ii] = _mm_comp_fmadd_ps(x_coeffs2, x2_val, value_f[ii]);
                value_f[ii] = _mm_comp_fmadd_ps(x_coeffs3, x3_val, value_f[ii]);

                offset_ptr += 4;
            }

            __m128 _v = _mm_mul_ps(y_coeffs0, value_f[0]);
            _v = _mm_comp_fmadd_ps(y_coeffs1, value_f[1], _v);
            _v = _mm_comp_fmadd_ps(y_coeffs2, value_f[2], _v);
            _v = _mm_comp_fmadd_ps(y_coeffs3, value_f[3], _v);
            _mm_storeu_ps(dstptr, _v);

            dstptr += 4;
            offset_value_ptr += 18;
        }
    }
}

#endif // __SSE2__

static inline void cubic_interp1d(float& coeffs0, float& coeffs1, float& coeffs2, float& coeffs3, float fx)
{
    const float A = -0.75f;

    float fx0 = fx + 1;
    float fx1 = fx;
    float fx2 = 1 - fx;
    // float fx3 = 2 - fx;

    coeffs0 = A * fx0 * fx0 * fx0 - 5 * A * fx0 * fx0 + 8 * A * fx0 - 4 * A;
    coeffs1 = (A + 2) * fx1 * fx1 * fx1 - (A + 3) * fx1 * fx1 + 1;
    coeffs2 = (A + 2) * fx2 * fx2 * fx2 - (A + 3) * fx2 * fx2 + 1;
    coeffs3 = 1.f - coeffs0 - coeffs1 - coeffs2;
}

static void gridsample_2d_bicubic_apply_interpolation_p1(const Mat& src, Mat& dst, Mat& offset_value, const Option& opt)
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
            float x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3;
            float y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3;
            float value_f[4];
            cubic_interp1d(x_coeffs0, x_coeffs1, x_coeffs2, x_coeffs3, offset_value_ptr[0]);
            cubic_interp1d(y_coeffs0, y_coeffs1, y_coeffs2, y_coeffs3, offset_value_ptr[1]);

            const int* offset_ptr = (int*)offset_value_ptr + 2;

            for (int ii = 0; ii < 4; ii++)
            {
                float x0_val = offset_ptr[0] >= 0 ? *(srcptr + offset_ptr[0]) : 0;
                float x1_val = offset_ptr[1] >= 0 ? *(srcptr + offset_ptr[1]) : 0;
                float x2_val = offset_ptr[2] >= 0 ? *(srcptr + offset_ptr[2]) : 0;
                float x3_val = offset_ptr[3] >= 0 ? *(srcptr + offset_ptr[3]) : 0;

                value_f[ii] = x_coeffs0 * x0_val;
                value_f[ii] = x_coeffs1 * x1_val + value_f[ii];
                value_f[ii] = x_coeffs2 * x2_val + value_f[ii];
                value_f[ii] = x_coeffs3 * x3_val + value_f[ii];

                offset_ptr += 4;
            }

            float _v = y_coeffs0 * value_f[0];
            _v = y_coeffs1 * value_f[1] + _v;
            _v = y_coeffs2 * value_f[2] + _v;
            _v = y_coeffs3 * value_f[3] + _v;
            *dstptr = _v;

            dstptr++;
            offset_value_ptr += 18;
        }
    }
}