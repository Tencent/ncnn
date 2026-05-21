// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if __loongarch_sx
static __m128 gridsample_load_pack4_lsx(const float* ptr, int offset, __m128 _zero)
{
    return offset >= 0 ? (__m128)__lsx_vld(ptr + offset, 0) : _zero;
}
#endif // __loongarch_sx

static float gridsample_load_p1_loongarch(const float* ptr, int offset)
{
    return offset >= 0 ? ptr[offset] : 0.f;
}

#if __loongarch_sx
#if __loongarch_asx
static __m256 gridsample_load_pack4x2_lasx(const float* ptr, int offset0, int offset1, __m128 _zero)
{
    __m128 _v0 = offset0 >= 0 ? (__m128)__lsx_vld(ptr + offset0, 0) : _zero;
    __m128 _v1 = offset1 >= 0 ? (__m128)__lsx_vld(ptr + offset1, 0) : _zero;
    return __lasx_concat_128_s(_v0, _v1);
}

static __m256 gridsample_set2_pack4_ps_lasx(float v0, float v1)
{
    return __lasx_concat_128_s((__m128)__lsx_vreplfr2vr_s(v0), (__m128)__lsx_vreplfr2vr_s(v1));
}
#endif // __loongarch_asx

static void gridsample_2d_bilinear_apply_interpolation_pack4_lsx(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h;
    const __m128 _zero = (__m128)__lsx_vldi(0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            const int* offset_ptr = (const int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 4;

            __m128 _v00 = gridsample_load_pack4_lsx(srcptr, offset_ptr[0], _zero);
            __m128 _v01 = gridsample_load_pack4_lsx(srcptr, offset_ptr[1], _zero);
            __m128 _v10 = gridsample_load_pack4_lsx(srcptr, offset_ptr[2], _zero);
            __m128 _v11 = gridsample_load_pack4_lsx(srcptr, offset_ptr[3], _zero);

            __m128 _alpha = (__m128)__lsx_vreplfr2vr_s(value_ptr[0]);
            __m128 _beta = (__m128)__lsx_vreplfr2vr_s(value_ptr[1]);
            __m128 _v0 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v01, _v00), _v00);
            __m128 _v1 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v11, _v10), _v10);
            __m128 _v = __lsx_vfmadd_s(_beta, __lsx_vfsub_s(_v1, _v0), _v0);

            __lsx_vst(_v, dstptr, 0);

            dstptr += 4;
            offset_value_ptr += 6;
        }
    }
}

static void gridsample_3d_bilinear_apply_interpolation_pack4_lsx(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h * dst.d;
    const __m128 _zero = (__m128)__lsx_vldi(0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            const int* offset_ptr = (const int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 8;

            __m128 _v000 = gridsample_load_pack4_lsx(srcptr, offset_ptr[0], _zero);
            __m128 _v001 = gridsample_load_pack4_lsx(srcptr, offset_ptr[1], _zero);
            __m128 _v010 = gridsample_load_pack4_lsx(srcptr, offset_ptr[2], _zero);
            __m128 _v011 = gridsample_load_pack4_lsx(srcptr, offset_ptr[3], _zero);
            __m128 _v100 = gridsample_load_pack4_lsx(srcptr, offset_ptr[4], _zero);
            __m128 _v101 = gridsample_load_pack4_lsx(srcptr, offset_ptr[5], _zero);
            __m128 _v110 = gridsample_load_pack4_lsx(srcptr, offset_ptr[6], _zero);
            __m128 _v111 = gridsample_load_pack4_lsx(srcptr, offset_ptr[7], _zero);

            __m128 _alpha = (__m128)__lsx_vreplfr2vr_s(value_ptr[0]);
            __m128 _beta = (__m128)__lsx_vreplfr2vr_s(value_ptr[1]);
            __m128 _gamma = (__m128)__lsx_vreplfr2vr_s(value_ptr[2]);

            __m128 _v00 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v001, _v000), _v000);
            __m128 _v01 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v011, _v010), _v010);
            __m128 _v10 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v101, _v100), _v100);
            __m128 _v11 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v111, _v110), _v110);
            __m128 _v0 = __lsx_vfmadd_s(_beta, __lsx_vfsub_s(_v01, _v00), _v00);
            __m128 _v1 = __lsx_vfmadd_s(_beta, __lsx_vfsub_s(_v11, _v10), _v10);
            __m128 _v = __lsx_vfmadd_s(_gamma, __lsx_vfsub_s(_v1, _v0), _v0);

            __lsx_vst(_v, dstptr, 0);

            dstptr += 4;
            offset_value_ptr += 11;
        }
    }
}

static void gridsample_nearest_apply_interpolation_pack4_lsx(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h * dst.d;
    const __m128 _zero = (__m128)__lsx_vldi(0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const int* offset_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            __m128 _v = gridsample_load_pack4_lsx(srcptr, offset_ptr[0], _zero);
            __lsx_vst(_v, dstptr, 0);

            offset_ptr++;
            dstptr += 4;
        }
    }
}

#endif // __loongarch_sx

static void cubic_interp1d_loongarch(float fx, float* coeffs)
{
    const float A = -0.75f;

    float fx0 = fx + 1.f;
    float fx1 = fx;
    float fx2 = 1.f - fx;

    coeffs[0] = A * fx0 * fx0 * fx0 - 5.f * A * fx0 * fx0 + 8.f * A * fx0 - 4.f * A;
    coeffs[1] = (A + 2.f) * fx1 * fx1 * fx1 - (A + 3.f) * fx1 * fx1 + 1.f;
    coeffs[2] = (A + 2.f) * fx2 * fx2 * fx2 - (A + 3.f) * fx2 * fx2 + 1.f;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

#if __loongarch_sx
static void gridsample_2d_bicubic_apply_interpolation_pack4_lsx(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h;
    const __m128 _zero = (__m128)__lsx_vldi(0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            float x_coeffs[4];
            float y_coeffs[4];
            cubic_interp1d_loongarch(offset_value_ptr[0], x_coeffs);
            cubic_interp1d_loongarch(offset_value_ptr[1], y_coeffs);

            const int* offset_ptr = (const int*)offset_value_ptr + 2;
            __m128 _rows[4];
            for (int j = 0; j < 4; j++)
            {
                __m128 _v0 = gridsample_load_pack4_lsx(srcptr, offset_ptr[0], _zero);
                __m128 _v1 = gridsample_load_pack4_lsx(srcptr, offset_ptr[1], _zero);
                __m128 _v2 = gridsample_load_pack4_lsx(srcptr, offset_ptr[2], _zero);
                __m128 _v3 = gridsample_load_pack4_lsx(srcptr, offset_ptr[3], _zero);

                _rows[j] = __lsx_vfmul_s(_v0, (__m128)__lsx_vreplfr2vr_s(x_coeffs[0]));
                _rows[j] = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(x_coeffs[1]), _v1, _rows[j]);
                _rows[j] = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(x_coeffs[2]), _v2, _rows[j]);
                _rows[j] = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(x_coeffs[3]), _v3, _rows[j]);

                offset_ptr += 4;
            }

            __m128 _v = __lsx_vfmul_s(_rows[0], (__m128)__lsx_vreplfr2vr_s(y_coeffs[0]));
            _v = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(y_coeffs[1]), _rows[1], _v);
            _v = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(y_coeffs[2]), _rows[2], _v);
            _v = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(y_coeffs[3]), _rows[3], _v);

            __lsx_vst(_v, dstptr, 0);

            dstptr += 4;
            offset_value_ptr += 18;
        }
    }
}

#if __loongarch_asx
static __m256 gridsample_load_pack8_lasx(const float* ptr, int offset, __m256 _zero)
{
    return offset >= 0 ? (__m256)__lasx_xvld(ptr + offset, 0) : _zero;
}

static void gridsample_2d_bilinear_apply_interpolation_pack8_lasx(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h;
    const __m256 _zero = (__m256)__lasx_xvldi(0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            const int* offset_ptr = (const int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 4;

            __m256 _v00 = gridsample_load_pack8_lasx(srcptr, offset_ptr[0], _zero);
            __m256 _v01 = gridsample_load_pack8_lasx(srcptr, offset_ptr[1], _zero);
            __m256 _v10 = gridsample_load_pack8_lasx(srcptr, offset_ptr[2], _zero);
            __m256 _v11 = gridsample_load_pack8_lasx(srcptr, offset_ptr[3], _zero);

            __m256 _alpha = (__m256)__lasx_xvreplfr2vr_s(value_ptr[0]);
            __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(value_ptr[1]);
            __m256 _v0 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v01, _v00), _v00);
            __m256 _v1 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v11, _v10), _v10);
            __m256 _v = __lasx_xvfmadd_s(_beta, __lasx_xvfsub_s(_v1, _v0), _v0);

            __lasx_xvst(_v, dstptr, 0);

            dstptr += 8;
            offset_value_ptr += 6;
        }
    }
}

static void gridsample_3d_bilinear_apply_interpolation_pack8_lasx(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h * dst.d;
    const __m256 _zero = (__m256)__lasx_xvldi(0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            const int* offset_ptr = (const int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 8;

            __m256 _v000 = gridsample_load_pack8_lasx(srcptr, offset_ptr[0], _zero);
            __m256 _v001 = gridsample_load_pack8_lasx(srcptr, offset_ptr[1], _zero);
            __m256 _v010 = gridsample_load_pack8_lasx(srcptr, offset_ptr[2], _zero);
            __m256 _v011 = gridsample_load_pack8_lasx(srcptr, offset_ptr[3], _zero);
            __m256 _v100 = gridsample_load_pack8_lasx(srcptr, offset_ptr[4], _zero);
            __m256 _v101 = gridsample_load_pack8_lasx(srcptr, offset_ptr[5], _zero);
            __m256 _v110 = gridsample_load_pack8_lasx(srcptr, offset_ptr[6], _zero);
            __m256 _v111 = gridsample_load_pack8_lasx(srcptr, offset_ptr[7], _zero);

            __m256 _alpha = (__m256)__lasx_xvreplfr2vr_s(value_ptr[0]);
            __m256 _beta = (__m256)__lasx_xvreplfr2vr_s(value_ptr[1]);
            __m256 _gamma = (__m256)__lasx_xvreplfr2vr_s(value_ptr[2]);

            __m256 _v00 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v001, _v000), _v000);
            __m256 _v01 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v011, _v010), _v010);
            __m256 _v10 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v101, _v100), _v100);
            __m256 _v11 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v111, _v110), _v110);
            __m256 _v0 = __lasx_xvfmadd_s(_beta, __lasx_xvfsub_s(_v01, _v00), _v00);
            __m256 _v1 = __lasx_xvfmadd_s(_beta, __lasx_xvfsub_s(_v11, _v10), _v10);
            __m256 _v = __lasx_xvfmadd_s(_gamma, __lasx_xvfsub_s(_v1, _v0), _v0);

            __lasx_xvst(_v, dstptr, 0);

            dstptr += 8;
            offset_value_ptr += 11;
        }
    }
}

static void gridsample_nearest_apply_interpolation_pack8_lasx(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h * dst.d;
    const __m256 _zero = (__m256)__lasx_xvldi(0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const int* offset_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            __m256 _v = gridsample_load_pack8_lasx(srcptr, offset_ptr[0], _zero);
            __lasx_xvst(_v, dstptr, 0);

            offset_ptr++;
            dstptr += 8;
        }
    }
}

static void gridsample_2d_bicubic_apply_interpolation_pack8_lasx(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h;
    const __m256 _zero = (__m256)__lasx_xvldi(0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            float x_coeffs[4];
            float y_coeffs[4];
            cubic_interp1d_loongarch(offset_value_ptr[0], x_coeffs);
            cubic_interp1d_loongarch(offset_value_ptr[1], y_coeffs);

            const int* offset_ptr = (const int*)offset_value_ptr + 2;
            __m256 _rows[4];
            for (int j = 0; j < 4; j++)
            {
                __m256 _v0 = gridsample_load_pack8_lasx(srcptr, offset_ptr[0], _zero);
                __m256 _v1 = gridsample_load_pack8_lasx(srcptr, offset_ptr[1], _zero);
                __m256 _v2 = gridsample_load_pack8_lasx(srcptr, offset_ptr[2], _zero);
                __m256 _v3 = gridsample_load_pack8_lasx(srcptr, offset_ptr[3], _zero);

                _rows[j] = __lasx_xvfmul_s(_v0, (__m256)__lasx_xvreplfr2vr_s(x_coeffs[0]));
                _rows[j] = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(x_coeffs[1]), _v1, _rows[j]);
                _rows[j] = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(x_coeffs[2]), _v2, _rows[j]);
                _rows[j] = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(x_coeffs[3]), _v3, _rows[j]);

                offset_ptr += 4;
            }

            __m256 _v = __lasx_xvfmul_s(_rows[0], (__m256)__lasx_xvreplfr2vr_s(y_coeffs[0]));
            _v = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(y_coeffs[1]), _rows[1], _v);
            _v = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(y_coeffs[2]), _rows[2], _v);
            _v = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(y_coeffs[3]), _rows[3], _v);

            __lasx_xvst(_v, dstptr, 0);

            dstptr += 8;
            offset_value_ptr += 18;
        }
    }
}

static void gridsample_2d_bilinear_apply_interpolation_pack4_lasx(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h;
    const __m128 _zero = (__m128)__lsx_vldi(0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value.channel(0);

        int i = 0;
        for (; i + 1 < grid_size; i += 2)
        {
            const float* offset_value_ptr0 = offset_value_ptr;
            const float* offset_value_ptr1 = offset_value_ptr + 6;
            const int* offset_ptr0 = (const int*)offset_value_ptr0;
            const int* offset_ptr1 = (const int*)offset_value_ptr1;
            const float* value_ptr0 = offset_value_ptr0 + 4;
            const float* value_ptr1 = offset_value_ptr1 + 4;

            __m256 _v00 = gridsample_load_pack4x2_lasx(srcptr, offset_ptr0[0], offset_ptr1[0], _zero);
            __m256 _v01 = gridsample_load_pack4x2_lasx(srcptr, offset_ptr0[1], offset_ptr1[1], _zero);
            __m256 _v10 = gridsample_load_pack4x2_lasx(srcptr, offset_ptr0[2], offset_ptr1[2], _zero);
            __m256 _v11 = gridsample_load_pack4x2_lasx(srcptr, offset_ptr0[3], offset_ptr1[3], _zero);

            __m256 _alpha = gridsample_set2_pack4_ps_lasx(value_ptr0[0], value_ptr1[0]);
            __m256 _beta = gridsample_set2_pack4_ps_lasx(value_ptr0[1], value_ptr1[1]);
            __m256 _v0 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v01, _v00), _v00);
            __m256 _v1 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v11, _v10), _v10);
            __m256 _v = __lasx_xvfmadd_s(_beta, __lasx_xvfsub_s(_v1, _v0), _v0);

            __lasx_xvst(_v, dstptr, 0);

            dstptr += 8;
            offset_value_ptr += 12;
        }

        for (; i < grid_size; i++)
        {
            const int* offset_ptr = (const int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 4;

            __m128 _v00 = gridsample_load_pack4_lsx(srcptr, offset_ptr[0], _zero);
            __m128 _v01 = gridsample_load_pack4_lsx(srcptr, offset_ptr[1], _zero);
            __m128 _v10 = gridsample_load_pack4_lsx(srcptr, offset_ptr[2], _zero);
            __m128 _v11 = gridsample_load_pack4_lsx(srcptr, offset_ptr[3], _zero);

            __m128 _alpha = (__m128)__lsx_vreplfr2vr_s(value_ptr[0]);
            __m128 _beta = (__m128)__lsx_vreplfr2vr_s(value_ptr[1]);
            __m128 _v0 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v01, _v00), _v00);
            __m128 _v1 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v11, _v10), _v10);
            __m128 _v = __lsx_vfmadd_s(_beta, __lsx_vfsub_s(_v1, _v0), _v0);

            __lsx_vst(_v, dstptr, 0);

            dstptr += 4;
            offset_value_ptr += 6;
        }
    }
}

static void gridsample_3d_bilinear_apply_interpolation_pack4_lasx(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h * dst.d;
    const __m128 _zero = (__m128)__lsx_vldi(0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value.channel(0);

        int i = 0;
        for (; i + 1 < grid_size; i += 2)
        {
            const float* offset_value_ptr0 = offset_value_ptr;
            const float* offset_value_ptr1 = offset_value_ptr + 11;
            const int* offset_ptr0 = (const int*)offset_value_ptr0;
            const int* offset_ptr1 = (const int*)offset_value_ptr1;
            const float* value_ptr0 = offset_value_ptr0 + 8;
            const float* value_ptr1 = offset_value_ptr1 + 8;

            __m256 _v000 = gridsample_load_pack4x2_lasx(srcptr, offset_ptr0[0], offset_ptr1[0], _zero);
            __m256 _v001 = gridsample_load_pack4x2_lasx(srcptr, offset_ptr0[1], offset_ptr1[1], _zero);
            __m256 _v010 = gridsample_load_pack4x2_lasx(srcptr, offset_ptr0[2], offset_ptr1[2], _zero);
            __m256 _v011 = gridsample_load_pack4x2_lasx(srcptr, offset_ptr0[3], offset_ptr1[3], _zero);
            __m256 _v100 = gridsample_load_pack4x2_lasx(srcptr, offset_ptr0[4], offset_ptr1[4], _zero);
            __m256 _v101 = gridsample_load_pack4x2_lasx(srcptr, offset_ptr0[5], offset_ptr1[5], _zero);
            __m256 _v110 = gridsample_load_pack4x2_lasx(srcptr, offset_ptr0[6], offset_ptr1[6], _zero);
            __m256 _v111 = gridsample_load_pack4x2_lasx(srcptr, offset_ptr0[7], offset_ptr1[7], _zero);

            __m256 _alpha = gridsample_set2_pack4_ps_lasx(value_ptr0[0], value_ptr1[0]);
            __m256 _beta = gridsample_set2_pack4_ps_lasx(value_ptr0[1], value_ptr1[1]);
            __m256 _gamma = gridsample_set2_pack4_ps_lasx(value_ptr0[2], value_ptr1[2]);

            __m256 _v00 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v001, _v000), _v000);
            __m256 _v01 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v011, _v010), _v010);
            __m256 _v10 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v101, _v100), _v100);
            __m256 _v11 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v111, _v110), _v110);
            __m256 _v0 = __lasx_xvfmadd_s(_beta, __lasx_xvfsub_s(_v01, _v00), _v00);
            __m256 _v1 = __lasx_xvfmadd_s(_beta, __lasx_xvfsub_s(_v11, _v10), _v10);
            __m256 _v = __lasx_xvfmadd_s(_gamma, __lasx_xvfsub_s(_v1, _v0), _v0);

            __lasx_xvst(_v, dstptr, 0);

            dstptr += 8;
            offset_value_ptr += 22;
        }

        for (; i < grid_size; i++)
        {
            const int* offset_ptr = (const int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 8;

            __m128 _v000 = gridsample_load_pack4_lsx(srcptr, offset_ptr[0], _zero);
            __m128 _v001 = gridsample_load_pack4_lsx(srcptr, offset_ptr[1], _zero);
            __m128 _v010 = gridsample_load_pack4_lsx(srcptr, offset_ptr[2], _zero);
            __m128 _v011 = gridsample_load_pack4_lsx(srcptr, offset_ptr[3], _zero);
            __m128 _v100 = gridsample_load_pack4_lsx(srcptr, offset_ptr[4], _zero);
            __m128 _v101 = gridsample_load_pack4_lsx(srcptr, offset_ptr[5], _zero);
            __m128 _v110 = gridsample_load_pack4_lsx(srcptr, offset_ptr[6], _zero);
            __m128 _v111 = gridsample_load_pack4_lsx(srcptr, offset_ptr[7], _zero);

            __m128 _alpha = (__m128)__lsx_vreplfr2vr_s(value_ptr[0]);
            __m128 _beta = (__m128)__lsx_vreplfr2vr_s(value_ptr[1]);
            __m128 _gamma = (__m128)__lsx_vreplfr2vr_s(value_ptr[2]);

            __m128 _v00 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v001, _v000), _v000);
            __m128 _v01 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v011, _v010), _v010);
            __m128 _v10 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v101, _v100), _v100);
            __m128 _v11 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v111, _v110), _v110);
            __m128 _v0 = __lsx_vfmadd_s(_beta, __lsx_vfsub_s(_v01, _v00), _v00);
            __m128 _v1 = __lsx_vfmadd_s(_beta, __lsx_vfsub_s(_v11, _v10), _v10);
            __m128 _v = __lsx_vfmadd_s(_gamma, __lsx_vfsub_s(_v1, _v0), _v0);

            __lsx_vst(_v, dstptr, 0);

            dstptr += 4;
            offset_value_ptr += 11;
        }
    }
}

static void gridsample_nearest_apply_interpolation_pack4_lasx(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h * dst.d;
    const __m128 _zero = (__m128)__lsx_vldi(0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const int* offset_ptr = offset_value.channel(0);

        int i = 0;
        for (; i + 1 < grid_size; i += 2)
        {
            __m256 _v = gridsample_load_pack4x2_lasx(srcptr, offset_ptr[0], offset_ptr[1], _zero);
            __lasx_xvst(_v, dstptr, 0);

            offset_ptr += 2;
            dstptr += 8;
        }

        for (; i < grid_size; i++)
        {
            __m128 _v = gridsample_load_pack4_lsx(srcptr, offset_ptr[0], _zero);
            __lsx_vst(_v, dstptr, 0);

            offset_ptr++;
            dstptr += 4;
        }
    }
}

static void gridsample_2d_bicubic_apply_interpolation_pack4_lasx(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h;
    const __m128 _zero = (__m128)__lsx_vldi(0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value.channel(0);

        int i = 0;
        for (; i + 1 < grid_size; i += 2)
        {
            const float* offset_value_ptr0 = offset_value_ptr;
            const float* offset_value_ptr1 = offset_value_ptr + 18;

            float x_coeffs0[4];
            float y_coeffs0[4];
            float x_coeffs1[4];
            float y_coeffs1[4];
            cubic_interp1d_loongarch(offset_value_ptr0[0], x_coeffs0);
            cubic_interp1d_loongarch(offset_value_ptr0[1], y_coeffs0);
            cubic_interp1d_loongarch(offset_value_ptr1[0], x_coeffs1);
            cubic_interp1d_loongarch(offset_value_ptr1[1], y_coeffs1);

            const int* offset_ptr0 = (const int*)offset_value_ptr0 + 2;
            const int* offset_ptr1 = (const int*)offset_value_ptr1 + 2;
            __m256 _rows[4];
            for (int j = 0; j < 4; j++)
            {
                __m256 _v0 = gridsample_load_pack4x2_lasx(srcptr, offset_ptr0[0], offset_ptr1[0], _zero);
                __m256 _v1 = gridsample_load_pack4x2_lasx(srcptr, offset_ptr0[1], offset_ptr1[1], _zero);
                __m256 _v2 = gridsample_load_pack4x2_lasx(srcptr, offset_ptr0[2], offset_ptr1[2], _zero);
                __m256 _v3 = gridsample_load_pack4x2_lasx(srcptr, offset_ptr0[3], offset_ptr1[3], _zero);

                _rows[j] = __lasx_xvfmul_s(_v0, gridsample_set2_pack4_ps_lasx(x_coeffs0[0], x_coeffs1[0]));
                _rows[j] = __lasx_xvfmadd_s(gridsample_set2_pack4_ps_lasx(x_coeffs0[1], x_coeffs1[1]), _v1, _rows[j]);
                _rows[j] = __lasx_xvfmadd_s(gridsample_set2_pack4_ps_lasx(x_coeffs0[2], x_coeffs1[2]), _v2, _rows[j]);
                _rows[j] = __lasx_xvfmadd_s(gridsample_set2_pack4_ps_lasx(x_coeffs0[3], x_coeffs1[3]), _v3, _rows[j]);

                offset_ptr0 += 4;
                offset_ptr1 += 4;
            }

            __m256 _v = __lasx_xvfmul_s(_rows[0], gridsample_set2_pack4_ps_lasx(y_coeffs0[0], y_coeffs1[0]));
            _v = __lasx_xvfmadd_s(gridsample_set2_pack4_ps_lasx(y_coeffs0[1], y_coeffs1[1]), _rows[1], _v);
            _v = __lasx_xvfmadd_s(gridsample_set2_pack4_ps_lasx(y_coeffs0[2], y_coeffs1[2]), _rows[2], _v);
            _v = __lasx_xvfmadd_s(gridsample_set2_pack4_ps_lasx(y_coeffs0[3], y_coeffs1[3]), _rows[3], _v);

            __lasx_xvst(_v, dstptr, 0);

            dstptr += 8;
            offset_value_ptr += 36;
        }

        for (; i < grid_size; i++)
        {
            float x_coeffs[4];
            float y_coeffs[4];
            cubic_interp1d_loongarch(offset_value_ptr[0], x_coeffs);
            cubic_interp1d_loongarch(offset_value_ptr[1], y_coeffs);

            const int* offset_ptr = (const int*)offset_value_ptr + 2;
            __m128 _rows[4];
            for (int j = 0; j < 4; j++)
            {
                __m128 _v0 = gridsample_load_pack4_lsx(srcptr, offset_ptr[0], _zero);
                __m128 _v1 = gridsample_load_pack4_lsx(srcptr, offset_ptr[1], _zero);
                __m128 _v2 = gridsample_load_pack4_lsx(srcptr, offset_ptr[2], _zero);
                __m128 _v3 = gridsample_load_pack4_lsx(srcptr, offset_ptr[3], _zero);

                _rows[j] = __lsx_vfmul_s(_v0, (__m128)__lsx_vreplfr2vr_s(x_coeffs[0]));
                _rows[j] = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(x_coeffs[1]), _v1, _rows[j]);
                _rows[j] = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(x_coeffs[2]), _v2, _rows[j]);
                _rows[j] = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(x_coeffs[3]), _v3, _rows[j]);

                offset_ptr += 4;
            }

            __m128 _v = __lsx_vfmul_s(_rows[0], (__m128)__lsx_vreplfr2vr_s(y_coeffs[0]));
            _v = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(y_coeffs[1]), _rows[1], _v);
            _v = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(y_coeffs[2]), _rows[2], _v);
            _v = __lsx_vfmadd_s((__m128)__lsx_vreplfr2vr_s(y_coeffs[3]), _rows[3], _v);

            __lsx_vst(_v, dstptr, 0);

            dstptr += 4;
            offset_value_ptr += 18;
        }
    }
}
#endif // __loongarch_asx
#endif // __loongarch_sx

static void gridsample_2d_bilinear_apply_interpolation_p1_loongarch(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value.channel(0);

        int i = 0;
#if __loongarch_asx
        for (; i + 7 < grid_size; i += 8)
        {
            const float* ov0 = offset_value_ptr;
            const float* ov1 = offset_value_ptr + 6;
            const float* ov2 = offset_value_ptr + 12;
            const float* ov3 = offset_value_ptr + 18;
            const float* ov4 = offset_value_ptr + 24;
            const float* ov5 = offset_value_ptr + 30;
            const float* ov6 = offset_value_ptr + 36;
            const float* ov7 = offset_value_ptr + 42;
            const int* off0 = (const int*)ov0;
            const int* off1 = (const int*)ov1;
            const int* off2 = (const int*)ov2;
            const int* off3 = (const int*)ov3;
            const int* off4 = (const int*)ov4;
            const int* off5 = (const int*)ov5;
            const int* off6 = (const int*)ov6;
            const int* off7 = (const int*)ov7;
            const float* val0 = ov0 + 4;
            const float* val1 = ov1 + 4;
            const float* val2 = ov2 + 4;
            const float* val3 = ov3 + 4;
            const float* val4 = ov4 + 4;
            const float* val5 = ov5 + 4;
            const float* val6 = ov6 + 4;
            const float* val7 = ov7 + 4;

            __m256 _v00 = gridsample_set8_ps_lasx(gridsample_load_p1_loongarch(srcptr, off0[0]), gridsample_load_p1_loongarch(srcptr, off1[0]), gridsample_load_p1_loongarch(srcptr, off2[0]), gridsample_load_p1_loongarch(srcptr, off3[0]), gridsample_load_p1_loongarch(srcptr, off4[0]), gridsample_load_p1_loongarch(srcptr, off5[0]), gridsample_load_p1_loongarch(srcptr, off6[0]), gridsample_load_p1_loongarch(srcptr, off7[0]));
            __m256 _v01 = gridsample_set8_ps_lasx(gridsample_load_p1_loongarch(srcptr, off0[1]), gridsample_load_p1_loongarch(srcptr, off1[1]), gridsample_load_p1_loongarch(srcptr, off2[1]), gridsample_load_p1_loongarch(srcptr, off3[1]), gridsample_load_p1_loongarch(srcptr, off4[1]), gridsample_load_p1_loongarch(srcptr, off5[1]), gridsample_load_p1_loongarch(srcptr, off6[1]), gridsample_load_p1_loongarch(srcptr, off7[1]));
            __m256 _v10 = gridsample_set8_ps_lasx(gridsample_load_p1_loongarch(srcptr, off0[2]), gridsample_load_p1_loongarch(srcptr, off1[2]), gridsample_load_p1_loongarch(srcptr, off2[2]), gridsample_load_p1_loongarch(srcptr, off3[2]), gridsample_load_p1_loongarch(srcptr, off4[2]), gridsample_load_p1_loongarch(srcptr, off5[2]), gridsample_load_p1_loongarch(srcptr, off6[2]), gridsample_load_p1_loongarch(srcptr, off7[2]));
            __m256 _v11 = gridsample_set8_ps_lasx(gridsample_load_p1_loongarch(srcptr, off0[3]), gridsample_load_p1_loongarch(srcptr, off1[3]), gridsample_load_p1_loongarch(srcptr, off2[3]), gridsample_load_p1_loongarch(srcptr, off3[3]), gridsample_load_p1_loongarch(srcptr, off4[3]), gridsample_load_p1_loongarch(srcptr, off5[3]), gridsample_load_p1_loongarch(srcptr, off6[3]), gridsample_load_p1_loongarch(srcptr, off7[3]));

            __m256 _alpha = gridsample_set8_ps_lasx(val0[0], val1[0], val2[0], val3[0], val4[0], val5[0], val6[0], val7[0]);
            __m256 _beta = gridsample_set8_ps_lasx(val0[1], val1[1], val2[1], val3[1], val4[1], val5[1], val6[1], val7[1]);
            __m256 _v0 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v01, _v00), _v00);
            __m256 _v1 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v11, _v10), _v10);
            __m256 _v = __lasx_xvfmadd_s(_beta, __lasx_xvfsub_s(_v1, _v0), _v0);

            __lasx_xvst(_v, dstptr, 0);

            dstptr += 8;
            offset_value_ptr += 48;
        }
#endif // __loongarch_asx
#if __loongarch_sx
        for (; i + 3 < grid_size; i += 4)
        {
            const float* ov0 = offset_value_ptr;
            const float* ov1 = offset_value_ptr + 6;
            const float* ov2 = offset_value_ptr + 12;
            const float* ov3 = offset_value_ptr + 18;
            const int* off0 = (const int*)ov0;
            const int* off1 = (const int*)ov1;
            const int* off2 = (const int*)ov2;
            const int* off3 = (const int*)ov3;
            const float* val0 = ov0 + 4;
            const float* val1 = ov1 + 4;
            const float* val2 = ov2 + 4;
            const float* val3 = ov3 + 4;

            __m128 _v00 = gridsample_set4_ps_lsx(gridsample_load_p1_loongarch(srcptr, off0[0]), gridsample_load_p1_loongarch(srcptr, off1[0]), gridsample_load_p1_loongarch(srcptr, off2[0]), gridsample_load_p1_loongarch(srcptr, off3[0]));
            __m128 _v01 = gridsample_set4_ps_lsx(gridsample_load_p1_loongarch(srcptr, off0[1]), gridsample_load_p1_loongarch(srcptr, off1[1]), gridsample_load_p1_loongarch(srcptr, off2[1]), gridsample_load_p1_loongarch(srcptr, off3[1]));
            __m128 _v10 = gridsample_set4_ps_lsx(gridsample_load_p1_loongarch(srcptr, off0[2]), gridsample_load_p1_loongarch(srcptr, off1[2]), gridsample_load_p1_loongarch(srcptr, off2[2]), gridsample_load_p1_loongarch(srcptr, off3[2]));
            __m128 _v11 = gridsample_set4_ps_lsx(gridsample_load_p1_loongarch(srcptr, off0[3]), gridsample_load_p1_loongarch(srcptr, off1[3]), gridsample_load_p1_loongarch(srcptr, off2[3]), gridsample_load_p1_loongarch(srcptr, off3[3]));

            __m128 _alpha = gridsample_set4_ps_lsx(val0[0], val1[0], val2[0], val3[0]);
            __m128 _beta = gridsample_set4_ps_lsx(val0[1], val1[1], val2[1], val3[1]);
            __m128 _v0 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v01, _v00), _v00);
            __m128 _v1 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v11, _v10), _v10);
            __m128 _v = __lsx_vfmadd_s(_beta, __lsx_vfsub_s(_v1, _v0), _v0);

            __lsx_vst(_v, dstptr, 0);

            dstptr += 4;
            offset_value_ptr += 24;
        }
#endif // __loongarch_sx

        for (; i < grid_size; i++)
        {
            const int* offset_ptr = (const int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 4;

            float v00 = offset_ptr[0] >= 0 ? srcptr[offset_ptr[0]] : 0.f;
            float v01 = offset_ptr[1] >= 0 ? srcptr[offset_ptr[1]] : 0.f;
            float v10 = offset_ptr[2] >= 0 ? srcptr[offset_ptr[2]] : 0.f;
            float v11 = offset_ptr[3] >= 0 ? srcptr[offset_ptr[3]] : 0.f;

            float v0 = v00 * (1.f - value_ptr[0]) + v01 * value_ptr[0];
            float v1 = v10 * (1.f - value_ptr[0]) + v11 * value_ptr[0];

            dstptr[0] = v0 * (1.f - value_ptr[1]) + v1 * value_ptr[1];

            dstptr++;
            offset_value_ptr += 6;
        }
    }
}

static void gridsample_3d_bilinear_apply_interpolation_p1_loongarch(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h * dst.d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value.channel(0);

        int i = 0;
#if __loongarch_asx
        for (; i + 7 < grid_size; i += 8)
        {
            const float* ov0 = offset_value_ptr;
            const float* ov1 = offset_value_ptr + 11;
            const float* ov2 = offset_value_ptr + 22;
            const float* ov3 = offset_value_ptr + 33;
            const float* ov4 = offset_value_ptr + 44;
            const float* ov5 = offset_value_ptr + 55;
            const float* ov6 = offset_value_ptr + 66;
            const float* ov7 = offset_value_ptr + 77;
            const int* off0 = (const int*)ov0;
            const int* off1 = (const int*)ov1;
            const int* off2 = (const int*)ov2;
            const int* off3 = (const int*)ov3;
            const int* off4 = (const int*)ov4;
            const int* off5 = (const int*)ov5;
            const int* off6 = (const int*)ov6;
            const int* off7 = (const int*)ov7;
            const float* val0 = ov0 + 8;
            const float* val1 = ov1 + 8;
            const float* val2 = ov2 + 8;
            const float* val3 = ov3 + 8;
            const float* val4 = ov4 + 8;
            const float* val5 = ov5 + 8;
            const float* val6 = ov6 + 8;
            const float* val7 = ov7 + 8;

            __m256 _v000 = gridsample_set8_ps_lasx(gridsample_load_p1_loongarch(srcptr, off0[0]), gridsample_load_p1_loongarch(srcptr, off1[0]), gridsample_load_p1_loongarch(srcptr, off2[0]), gridsample_load_p1_loongarch(srcptr, off3[0]), gridsample_load_p1_loongarch(srcptr, off4[0]), gridsample_load_p1_loongarch(srcptr, off5[0]), gridsample_load_p1_loongarch(srcptr, off6[0]), gridsample_load_p1_loongarch(srcptr, off7[0]));
            __m256 _v001 = gridsample_set8_ps_lasx(gridsample_load_p1_loongarch(srcptr, off0[1]), gridsample_load_p1_loongarch(srcptr, off1[1]), gridsample_load_p1_loongarch(srcptr, off2[1]), gridsample_load_p1_loongarch(srcptr, off3[1]), gridsample_load_p1_loongarch(srcptr, off4[1]), gridsample_load_p1_loongarch(srcptr, off5[1]), gridsample_load_p1_loongarch(srcptr, off6[1]), gridsample_load_p1_loongarch(srcptr, off7[1]));
            __m256 _v010 = gridsample_set8_ps_lasx(gridsample_load_p1_loongarch(srcptr, off0[2]), gridsample_load_p1_loongarch(srcptr, off1[2]), gridsample_load_p1_loongarch(srcptr, off2[2]), gridsample_load_p1_loongarch(srcptr, off3[2]), gridsample_load_p1_loongarch(srcptr, off4[2]), gridsample_load_p1_loongarch(srcptr, off5[2]), gridsample_load_p1_loongarch(srcptr, off6[2]), gridsample_load_p1_loongarch(srcptr, off7[2]));
            __m256 _v011 = gridsample_set8_ps_lasx(gridsample_load_p1_loongarch(srcptr, off0[3]), gridsample_load_p1_loongarch(srcptr, off1[3]), gridsample_load_p1_loongarch(srcptr, off2[3]), gridsample_load_p1_loongarch(srcptr, off3[3]), gridsample_load_p1_loongarch(srcptr, off4[3]), gridsample_load_p1_loongarch(srcptr, off5[3]), gridsample_load_p1_loongarch(srcptr, off6[3]), gridsample_load_p1_loongarch(srcptr, off7[3]));
            __m256 _v100 = gridsample_set8_ps_lasx(gridsample_load_p1_loongarch(srcptr, off0[4]), gridsample_load_p1_loongarch(srcptr, off1[4]), gridsample_load_p1_loongarch(srcptr, off2[4]), gridsample_load_p1_loongarch(srcptr, off3[4]), gridsample_load_p1_loongarch(srcptr, off4[4]), gridsample_load_p1_loongarch(srcptr, off5[4]), gridsample_load_p1_loongarch(srcptr, off6[4]), gridsample_load_p1_loongarch(srcptr, off7[4]));
            __m256 _v101 = gridsample_set8_ps_lasx(gridsample_load_p1_loongarch(srcptr, off0[5]), gridsample_load_p1_loongarch(srcptr, off1[5]), gridsample_load_p1_loongarch(srcptr, off2[5]), gridsample_load_p1_loongarch(srcptr, off3[5]), gridsample_load_p1_loongarch(srcptr, off4[5]), gridsample_load_p1_loongarch(srcptr, off5[5]), gridsample_load_p1_loongarch(srcptr, off6[5]), gridsample_load_p1_loongarch(srcptr, off7[5]));
            __m256 _v110 = gridsample_set8_ps_lasx(gridsample_load_p1_loongarch(srcptr, off0[6]), gridsample_load_p1_loongarch(srcptr, off1[6]), gridsample_load_p1_loongarch(srcptr, off2[6]), gridsample_load_p1_loongarch(srcptr, off3[6]), gridsample_load_p1_loongarch(srcptr, off4[6]), gridsample_load_p1_loongarch(srcptr, off5[6]), gridsample_load_p1_loongarch(srcptr, off6[6]), gridsample_load_p1_loongarch(srcptr, off7[6]));
            __m256 _v111 = gridsample_set8_ps_lasx(gridsample_load_p1_loongarch(srcptr, off0[7]), gridsample_load_p1_loongarch(srcptr, off1[7]), gridsample_load_p1_loongarch(srcptr, off2[7]), gridsample_load_p1_loongarch(srcptr, off3[7]), gridsample_load_p1_loongarch(srcptr, off4[7]), gridsample_load_p1_loongarch(srcptr, off5[7]), gridsample_load_p1_loongarch(srcptr, off6[7]), gridsample_load_p1_loongarch(srcptr, off7[7]));

            __m256 _alpha = gridsample_set8_ps_lasx(val0[0], val1[0], val2[0], val3[0], val4[0], val5[0], val6[0], val7[0]);
            __m256 _beta = gridsample_set8_ps_lasx(val0[1], val1[1], val2[1], val3[1], val4[1], val5[1], val6[1], val7[1]);
            __m256 _gamma = gridsample_set8_ps_lasx(val0[2], val1[2], val2[2], val3[2], val4[2], val5[2], val6[2], val7[2]);
            __m256 _v00 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v001, _v000), _v000);
            __m256 _v01 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v011, _v010), _v010);
            __m256 _v10 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v101, _v100), _v100);
            __m256 _v11 = __lasx_xvfmadd_s(_alpha, __lasx_xvfsub_s(_v111, _v110), _v110);
            __m256 _v0 = __lasx_xvfmadd_s(_beta, __lasx_xvfsub_s(_v01, _v00), _v00);
            __m256 _v1 = __lasx_xvfmadd_s(_beta, __lasx_xvfsub_s(_v11, _v10), _v10);
            __m256 _v = __lasx_xvfmadd_s(_gamma, __lasx_xvfsub_s(_v1, _v0), _v0);

            __lasx_xvst(_v, dstptr, 0);

            dstptr += 8;
            offset_value_ptr += 88;
        }
#endif // __loongarch_asx
#if __loongarch_sx
        for (; i + 3 < grid_size; i += 4)
        {
            const float* ov0 = offset_value_ptr;
            const float* ov1 = offset_value_ptr + 11;
            const float* ov2 = offset_value_ptr + 22;
            const float* ov3 = offset_value_ptr + 33;
            const int* off0 = (const int*)ov0;
            const int* off1 = (const int*)ov1;
            const int* off2 = (const int*)ov2;
            const int* off3 = (const int*)ov3;
            const float* val0 = ov0 + 8;
            const float* val1 = ov1 + 8;
            const float* val2 = ov2 + 8;
            const float* val3 = ov3 + 8;

            __m128 _v000 = gridsample_set4_ps_lsx(gridsample_load_p1_loongarch(srcptr, off0[0]), gridsample_load_p1_loongarch(srcptr, off1[0]), gridsample_load_p1_loongarch(srcptr, off2[0]), gridsample_load_p1_loongarch(srcptr, off3[0]));
            __m128 _v001 = gridsample_set4_ps_lsx(gridsample_load_p1_loongarch(srcptr, off0[1]), gridsample_load_p1_loongarch(srcptr, off1[1]), gridsample_load_p1_loongarch(srcptr, off2[1]), gridsample_load_p1_loongarch(srcptr, off3[1]));
            __m128 _v010 = gridsample_set4_ps_lsx(gridsample_load_p1_loongarch(srcptr, off0[2]), gridsample_load_p1_loongarch(srcptr, off1[2]), gridsample_load_p1_loongarch(srcptr, off2[2]), gridsample_load_p1_loongarch(srcptr, off3[2]));
            __m128 _v011 = gridsample_set4_ps_lsx(gridsample_load_p1_loongarch(srcptr, off0[3]), gridsample_load_p1_loongarch(srcptr, off1[3]), gridsample_load_p1_loongarch(srcptr, off2[3]), gridsample_load_p1_loongarch(srcptr, off3[3]));
            __m128 _v100 = gridsample_set4_ps_lsx(gridsample_load_p1_loongarch(srcptr, off0[4]), gridsample_load_p1_loongarch(srcptr, off1[4]), gridsample_load_p1_loongarch(srcptr, off2[4]), gridsample_load_p1_loongarch(srcptr, off3[4]));
            __m128 _v101 = gridsample_set4_ps_lsx(gridsample_load_p1_loongarch(srcptr, off0[5]), gridsample_load_p1_loongarch(srcptr, off1[5]), gridsample_load_p1_loongarch(srcptr, off2[5]), gridsample_load_p1_loongarch(srcptr, off3[5]));
            __m128 _v110 = gridsample_set4_ps_lsx(gridsample_load_p1_loongarch(srcptr, off0[6]), gridsample_load_p1_loongarch(srcptr, off1[6]), gridsample_load_p1_loongarch(srcptr, off2[6]), gridsample_load_p1_loongarch(srcptr, off3[6]));
            __m128 _v111 = gridsample_set4_ps_lsx(gridsample_load_p1_loongarch(srcptr, off0[7]), gridsample_load_p1_loongarch(srcptr, off1[7]), gridsample_load_p1_loongarch(srcptr, off2[7]), gridsample_load_p1_loongarch(srcptr, off3[7]));

            __m128 _alpha = gridsample_set4_ps_lsx(val0[0], val1[0], val2[0], val3[0]);
            __m128 _beta = gridsample_set4_ps_lsx(val0[1], val1[1], val2[1], val3[1]);
            __m128 _gamma = gridsample_set4_ps_lsx(val0[2], val1[2], val2[2], val3[2]);
            __m128 _v00 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v001, _v000), _v000);
            __m128 _v01 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v011, _v010), _v010);
            __m128 _v10 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v101, _v100), _v100);
            __m128 _v11 = __lsx_vfmadd_s(_alpha, __lsx_vfsub_s(_v111, _v110), _v110);
            __m128 _v0 = __lsx_vfmadd_s(_beta, __lsx_vfsub_s(_v01, _v00), _v00);
            __m128 _v1 = __lsx_vfmadd_s(_beta, __lsx_vfsub_s(_v11, _v10), _v10);
            __m128 _v = __lsx_vfmadd_s(_gamma, __lsx_vfsub_s(_v1, _v0), _v0);

            __lsx_vst(_v, dstptr, 0);

            dstptr += 4;
            offset_value_ptr += 44;
        }
#endif // __loongarch_sx

        for (; i < grid_size; i++)
        {
            const int* offset_ptr = (const int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 8;

            float v000 = offset_ptr[0] >= 0 ? srcptr[offset_ptr[0]] : 0.f;
            float v001 = offset_ptr[1] >= 0 ? srcptr[offset_ptr[1]] : 0.f;
            float v010 = offset_ptr[2] >= 0 ? srcptr[offset_ptr[2]] : 0.f;
            float v011 = offset_ptr[3] >= 0 ? srcptr[offset_ptr[3]] : 0.f;
            float v100 = offset_ptr[4] >= 0 ? srcptr[offset_ptr[4]] : 0.f;
            float v101 = offset_ptr[5] >= 0 ? srcptr[offset_ptr[5]] : 0.f;
            float v110 = offset_ptr[6] >= 0 ? srcptr[offset_ptr[6]] : 0.f;
            float v111 = offset_ptr[7] >= 0 ? srcptr[offset_ptr[7]] : 0.f;

            float v00 = v000 * (1.f - value_ptr[0]) + v001 * value_ptr[0];
            float v01 = v010 * (1.f - value_ptr[0]) + v011 * value_ptr[0];
            float v10 = v100 * (1.f - value_ptr[0]) + v101 * value_ptr[0];
            float v11 = v110 * (1.f - value_ptr[0]) + v111 * value_ptr[0];
            float v0 = v00 * (1.f - value_ptr[1]) + v01 * value_ptr[1];
            float v1 = v10 * (1.f - value_ptr[1]) + v11 * value_ptr[1];

            dstptr[0] = v0 * (1.f - value_ptr[2]) + v1 * value_ptr[2];

            dstptr++;
            offset_value_ptr += 11;
        }
    }
}

static void gridsample_nearest_apply_interpolation_p1_loongarch(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h * dst.d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const int* offset_ptr = offset_value.channel(0);

        int i = 0;
#if __loongarch_asx
        for (; i + 7 < grid_size; i += 8)
        {
            __m256 _v = gridsample_set8_ps_lasx(gridsample_load_p1_loongarch(srcptr, offset_ptr[0]), gridsample_load_p1_loongarch(srcptr, offset_ptr[1]), gridsample_load_p1_loongarch(srcptr, offset_ptr[2]), gridsample_load_p1_loongarch(srcptr, offset_ptr[3]), gridsample_load_p1_loongarch(srcptr, offset_ptr[4]), gridsample_load_p1_loongarch(srcptr, offset_ptr[5]), gridsample_load_p1_loongarch(srcptr, offset_ptr[6]), gridsample_load_p1_loongarch(srcptr, offset_ptr[7]));
            __lasx_xvst(_v, dstptr, 0);

            offset_ptr += 8;
            dstptr += 8;
        }
#endif // __loongarch_asx
#if __loongarch_sx
        for (; i + 3 < grid_size; i += 4)
        {
            __m128 _v = gridsample_set4_ps_lsx(gridsample_load_p1_loongarch(srcptr, offset_ptr[0]), gridsample_load_p1_loongarch(srcptr, offset_ptr[1]), gridsample_load_p1_loongarch(srcptr, offset_ptr[2]), gridsample_load_p1_loongarch(srcptr, offset_ptr[3]));
            __lsx_vst(_v, dstptr, 0);

            offset_ptr += 4;
            dstptr += 4;
        }
#endif // __loongarch_sx

        for (; i < grid_size; i++)
        {
            dstptr[0] = offset_ptr[0] >= 0 ? srcptr[offset_ptr[0]] : 0.f;

            offset_ptr++;
            dstptr++;
        }
    }
}

static void gridsample_2d_bicubic_apply_interpolation_p1_loongarch(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            float x_coeffs[4];
            float y_coeffs[4];
            cubic_interp1d_loongarch(offset_value_ptr[0], x_coeffs);
            cubic_interp1d_loongarch(offset_value_ptr[1], y_coeffs);

            const int* offset_ptr = (const int*)offset_value_ptr + 2;
            float rows[4];
            for (int j = 0; j < 4; j++)
            {
                float v0 = offset_ptr[0] >= 0 ? srcptr[offset_ptr[0]] : 0.f;
                float v1 = offset_ptr[1] >= 0 ? srcptr[offset_ptr[1]] : 0.f;
                float v2 = offset_ptr[2] >= 0 ? srcptr[offset_ptr[2]] : 0.f;
                float v3 = offset_ptr[3] >= 0 ? srcptr[offset_ptr[3]] : 0.f;

                rows[j] = v0 * x_coeffs[0];
                rows[j] += v1 * x_coeffs[1];
                rows[j] += v2 * x_coeffs[2];
                rows[j] += v3 * x_coeffs[3];

                offset_ptr += 4;
            }

            float v = rows[0] * y_coeffs[0];
            v += rows[1] * y_coeffs[1];
            v += rows[2] * y_coeffs[2];
            v += rows[3] * y_coeffs[3];

            dstptr[0] = v;

            dstptr++;
            offset_value_ptr += 18;
        }
    }
}
