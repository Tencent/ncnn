// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if __ARM_NEON
static float32x4_t gridsample_load_pack4_neon(const float* ptr, int offset)
{
    return offset >= 0 ? vld1q_f32(ptr + offset) : vdupq_n_f32(0.f);
}

static float32x4_t gridsample_set4_ps_neon(float v0, float v1, float v2, float v3)
{
    float32x4_t _v = vdupq_n_f32(v0);
    _v = vsetq_lane_f32(v1, _v, 1);
    _v = vsetq_lane_f32(v2, _v, 2);
    _v = vsetq_lane_f32(v3, _v, 3);
    return _v;
}

// a + b * c
static float32x4_t gridsample_fmadd_neon(float32x4_t _a, float32x4_t _b, float32x4_t _c)
{
#if __aarch64__
    return vfmaq_f32(_a, _b, _c);
#else
    return vmlaq_f32(_a, _b, _c);
#endif
}
#endif // __ARM_NEON

static float gridsample_load_p1_arm(const float* ptr, int offset)
{
    return offset >= 0 ? ptr[offset] : 0.f;
}

static void cubic_interp1d_arm(float fx, float* coeffs)
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

#if __ARM_NEON
static void gridsample_2d_bilinear_apply_interpolation_pack4_neon(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value;

        for (int i = 0; i < grid_size; i++)
        {
            const int* offset_ptr = (const int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 4;

            float32x4_t _v00 = gridsample_load_pack4_neon(srcptr, offset_ptr[0]);
            float32x4_t _v01 = gridsample_load_pack4_neon(srcptr, offset_ptr[1]);
            float32x4_t _v10 = gridsample_load_pack4_neon(srcptr, offset_ptr[2]);
            float32x4_t _v11 = gridsample_load_pack4_neon(srcptr, offset_ptr[3]);

            float32x4_t _alpha = vdupq_n_f32(value_ptr[0]);
            float32x4_t _beta = vdupq_n_f32(value_ptr[1]);
            float32x4_t _v0 = gridsample_fmadd_neon(_v00, vsubq_f32(_v01, _v00), _alpha);
            float32x4_t _v1 = gridsample_fmadd_neon(_v10, vsubq_f32(_v11, _v10), _alpha);
            float32x4_t _v = gridsample_fmadd_neon(_v0, vsubq_f32(_v1, _v0), _beta);

            vst1q_f32(dstptr, _v);

            dstptr += 4;
            offset_value_ptr += 6;
        }
    }
}

static void gridsample_3d_bilinear_apply_interpolation_pack4_neon(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h * dst.d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value;

        for (int i = 0; i < grid_size; i++)
        {
            const int* offset_ptr = (const int*)offset_value_ptr;
            const float* value_ptr = offset_value_ptr + 8;

            float32x4_t _v000 = gridsample_load_pack4_neon(srcptr, offset_ptr[0]);
            float32x4_t _v001 = gridsample_load_pack4_neon(srcptr, offset_ptr[1]);
            float32x4_t _v010 = gridsample_load_pack4_neon(srcptr, offset_ptr[2]);
            float32x4_t _v011 = gridsample_load_pack4_neon(srcptr, offset_ptr[3]);
            float32x4_t _v100 = gridsample_load_pack4_neon(srcptr, offset_ptr[4]);
            float32x4_t _v101 = gridsample_load_pack4_neon(srcptr, offset_ptr[5]);
            float32x4_t _v110 = gridsample_load_pack4_neon(srcptr, offset_ptr[6]);
            float32x4_t _v111 = gridsample_load_pack4_neon(srcptr, offset_ptr[7]);

            float32x4_t _alpha = vdupq_n_f32(value_ptr[0]);
            float32x4_t _beta = vdupq_n_f32(value_ptr[1]);
            float32x4_t _gamma = vdupq_n_f32(value_ptr[2]);

            float32x4_t _v00 = gridsample_fmadd_neon(_v000, vsubq_f32(_v001, _v000), _alpha);
            float32x4_t _v01 = gridsample_fmadd_neon(_v010, vsubq_f32(_v011, _v010), _alpha);
            float32x4_t _v10 = gridsample_fmadd_neon(_v100, vsubq_f32(_v101, _v100), _alpha);
            float32x4_t _v11 = gridsample_fmadd_neon(_v110, vsubq_f32(_v111, _v110), _alpha);
            float32x4_t _v0 = gridsample_fmadd_neon(_v00, vsubq_f32(_v01, _v00), _beta);
            float32x4_t _v1 = gridsample_fmadd_neon(_v10, vsubq_f32(_v11, _v10), _beta);
            float32x4_t _v = gridsample_fmadd_neon(_v0, vsubq_f32(_v1, _v0), _gamma);

            vst1q_f32(dstptr, _v);

            dstptr += 4;
            offset_value_ptr += 11;
        }
    }
}

static void gridsample_nearest_apply_interpolation_pack4_neon(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h * dst.d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const int* offset_ptr = offset_value;

        for (int i = 0; i < grid_size; i++)
        {
            vst1q_f32(dstptr, gridsample_load_pack4_neon(srcptr, offset_ptr[0]));

            offset_ptr++;
            dstptr += 4;
        }
    }
}

static void gridsample_bicubic_apply_interpolation_pack4_neon(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value;

        for (int i = 0; i < grid_size; i++)
        {
            float x_coeffs[4];
            float y_coeffs[4];
            cubic_interp1d_arm(offset_value_ptr[0], x_coeffs);
            cubic_interp1d_arm(offset_value_ptr[1], y_coeffs);

            const int* offset_ptr = (const int*)offset_value_ptr + 2;
            float32x4_t _rows[4];
            for (int j = 0; j < 4; j++)
            {
                float32x4_t _v0 = gridsample_load_pack4_neon(srcptr, offset_ptr[0]);
                float32x4_t _v1 = gridsample_load_pack4_neon(srcptr, offset_ptr[1]);
                float32x4_t _v2 = gridsample_load_pack4_neon(srcptr, offset_ptr[2]);
                float32x4_t _v3 = gridsample_load_pack4_neon(srcptr, offset_ptr[3]);

                _rows[j] = vmulq_n_f32(_v0, x_coeffs[0]);
                _rows[j] = gridsample_fmadd_neon(_rows[j], _v1, vdupq_n_f32(x_coeffs[1]));
                _rows[j] = gridsample_fmadd_neon(_rows[j], _v2, vdupq_n_f32(x_coeffs[2]));
                _rows[j] = gridsample_fmadd_neon(_rows[j], _v3, vdupq_n_f32(x_coeffs[3]));

                offset_ptr += 4;
            }

            float32x4_t _v = vmulq_n_f32(_rows[0], y_coeffs[0]);
            _v = gridsample_fmadd_neon(_v, _rows[1], vdupq_n_f32(y_coeffs[1]));
            _v = gridsample_fmadd_neon(_v, _rows[2], vdupq_n_f32(y_coeffs[2]));
            _v = gridsample_fmadd_neon(_v, _rows[3], vdupq_n_f32(y_coeffs[3]));

            vst1q_f32(dstptr, _v);

            dstptr += 4;
            offset_value_ptr += 18;
        }
    }
}
#endif // __ARM_NEON

static void gridsample_2d_bilinear_apply_interpolation_p1_arm(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < grid_size; i += 4)
        {
            const int* offset_ptr0 = (const int*)offset_value_ptr;
            const int* offset_ptr1 = (const int*)offset_value_ptr + 6;
            const int* offset_ptr2 = (const int*)offset_value_ptr + 12;
            const int* offset_ptr3 = (const int*)offset_value_ptr + 18;

            float32x4_t _v00 = gridsample_set4_ps_neon(gridsample_load_p1_arm(srcptr, offset_ptr0[0]), gridsample_load_p1_arm(srcptr, offset_ptr1[0]), gridsample_load_p1_arm(srcptr, offset_ptr2[0]), gridsample_load_p1_arm(srcptr, offset_ptr3[0]));
            float32x4_t _v01 = gridsample_set4_ps_neon(gridsample_load_p1_arm(srcptr, offset_ptr0[1]), gridsample_load_p1_arm(srcptr, offset_ptr1[1]), gridsample_load_p1_arm(srcptr, offset_ptr2[1]), gridsample_load_p1_arm(srcptr, offset_ptr3[1]));
            float32x4_t _v10 = gridsample_set4_ps_neon(gridsample_load_p1_arm(srcptr, offset_ptr0[2]), gridsample_load_p1_arm(srcptr, offset_ptr1[2]), gridsample_load_p1_arm(srcptr, offset_ptr2[2]), gridsample_load_p1_arm(srcptr, offset_ptr3[2]));
            float32x4_t _v11 = gridsample_set4_ps_neon(gridsample_load_p1_arm(srcptr, offset_ptr0[3]), gridsample_load_p1_arm(srcptr, offset_ptr1[3]), gridsample_load_p1_arm(srcptr, offset_ptr2[3]), gridsample_load_p1_arm(srcptr, offset_ptr3[3]));

            float32x4_t _alpha = gridsample_set4_ps_neon(offset_value_ptr[4], offset_value_ptr[10], offset_value_ptr[16], offset_value_ptr[22]);
            float32x4_t _beta = gridsample_set4_ps_neon(offset_value_ptr[5], offset_value_ptr[11], offset_value_ptr[17], offset_value_ptr[23]);
            float32x4_t _v0 = gridsample_fmadd_neon(_v00, vsubq_f32(_v01, _v00), _alpha);
            float32x4_t _v1 = gridsample_fmadd_neon(_v10, vsubq_f32(_v11, _v10), _alpha);
            vst1q_f32(dstptr, gridsample_fmadd_neon(_v0, vsubq_f32(_v1, _v0), _beta));

            dstptr += 4;
            offset_value_ptr += 24;
        }
#endif // __ARM_NEON
        for (; i < grid_size; i++)
        {
            const int* offset_ptr = (const int*)offset_value_ptr;
            const float alpha = offset_value_ptr[4];
            const float beta = offset_value_ptr[5];

            float v00 = gridsample_load_p1_arm(srcptr, offset_ptr[0]);
            float v01 = gridsample_load_p1_arm(srcptr, offset_ptr[1]);
            float v10 = gridsample_load_p1_arm(srcptr, offset_ptr[2]);
            float v11 = gridsample_load_p1_arm(srcptr, offset_ptr[3]);

            float v0 = v00 * (1.f - alpha) + v01 * alpha;
            float v1 = v10 * (1.f - alpha) + v11 * alpha;

            dstptr[0] = v0 * (1.f - beta) + v1 * beta;

            dstptr++;
            offset_value_ptr += 6;
        }
    }
}

static void gridsample_3d_bilinear_apply_interpolation_p1_arm(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h * dst.d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < grid_size; i += 4)
        {
            const int* offset_ptr0 = (const int*)offset_value_ptr;
            const int* offset_ptr1 = (const int*)offset_value_ptr + 11;
            const int* offset_ptr2 = (const int*)offset_value_ptr + 22;
            const int* offset_ptr3 = (const int*)offset_value_ptr + 33;

            float32x4_t _v000 = gridsample_set4_ps_neon(gridsample_load_p1_arm(srcptr, offset_ptr0[0]), gridsample_load_p1_arm(srcptr, offset_ptr1[0]), gridsample_load_p1_arm(srcptr, offset_ptr2[0]), gridsample_load_p1_arm(srcptr, offset_ptr3[0]));
            float32x4_t _v001 = gridsample_set4_ps_neon(gridsample_load_p1_arm(srcptr, offset_ptr0[1]), gridsample_load_p1_arm(srcptr, offset_ptr1[1]), gridsample_load_p1_arm(srcptr, offset_ptr2[1]), gridsample_load_p1_arm(srcptr, offset_ptr3[1]));
            float32x4_t _v010 = gridsample_set4_ps_neon(gridsample_load_p1_arm(srcptr, offset_ptr0[2]), gridsample_load_p1_arm(srcptr, offset_ptr1[2]), gridsample_load_p1_arm(srcptr, offset_ptr2[2]), gridsample_load_p1_arm(srcptr, offset_ptr3[2]));
            float32x4_t _v011 = gridsample_set4_ps_neon(gridsample_load_p1_arm(srcptr, offset_ptr0[3]), gridsample_load_p1_arm(srcptr, offset_ptr1[3]), gridsample_load_p1_arm(srcptr, offset_ptr2[3]), gridsample_load_p1_arm(srcptr, offset_ptr3[3]));
            float32x4_t _v100 = gridsample_set4_ps_neon(gridsample_load_p1_arm(srcptr, offset_ptr0[4]), gridsample_load_p1_arm(srcptr, offset_ptr1[4]), gridsample_load_p1_arm(srcptr, offset_ptr2[4]), gridsample_load_p1_arm(srcptr, offset_ptr3[4]));
            float32x4_t _v101 = gridsample_set4_ps_neon(gridsample_load_p1_arm(srcptr, offset_ptr0[5]), gridsample_load_p1_arm(srcptr, offset_ptr1[5]), gridsample_load_p1_arm(srcptr, offset_ptr2[5]), gridsample_load_p1_arm(srcptr, offset_ptr3[5]));
            float32x4_t _v110 = gridsample_set4_ps_neon(gridsample_load_p1_arm(srcptr, offset_ptr0[6]), gridsample_load_p1_arm(srcptr, offset_ptr1[6]), gridsample_load_p1_arm(srcptr, offset_ptr2[6]), gridsample_load_p1_arm(srcptr, offset_ptr3[6]));
            float32x4_t _v111 = gridsample_set4_ps_neon(gridsample_load_p1_arm(srcptr, offset_ptr0[7]), gridsample_load_p1_arm(srcptr, offset_ptr1[7]), gridsample_load_p1_arm(srcptr, offset_ptr2[7]), gridsample_load_p1_arm(srcptr, offset_ptr3[7]));

            float32x4_t _alpha = gridsample_set4_ps_neon(offset_value_ptr[8], offset_value_ptr[19], offset_value_ptr[30], offset_value_ptr[41]);
            float32x4_t _beta = gridsample_set4_ps_neon(offset_value_ptr[9], offset_value_ptr[20], offset_value_ptr[31], offset_value_ptr[42]);
            float32x4_t _gamma = gridsample_set4_ps_neon(offset_value_ptr[10], offset_value_ptr[21], offset_value_ptr[32], offset_value_ptr[43]);

            float32x4_t _v00 = gridsample_fmadd_neon(_v000, vsubq_f32(_v001, _v000), _alpha);
            float32x4_t _v01 = gridsample_fmadd_neon(_v010, vsubq_f32(_v011, _v010), _alpha);
            float32x4_t _v10 = gridsample_fmadd_neon(_v100, vsubq_f32(_v101, _v100), _alpha);
            float32x4_t _v11 = gridsample_fmadd_neon(_v110, vsubq_f32(_v111, _v110), _alpha);
            float32x4_t _v0 = gridsample_fmadd_neon(_v00, vsubq_f32(_v01, _v00), _beta);
            float32x4_t _v1 = gridsample_fmadd_neon(_v10, vsubq_f32(_v11, _v10), _beta);
            vst1q_f32(dstptr, gridsample_fmadd_neon(_v0, vsubq_f32(_v1, _v0), _gamma));

            dstptr += 4;
            offset_value_ptr += 44;
        }
#endif // __ARM_NEON
        for (; i < grid_size; i++)
        {
            const int* offset_ptr = (const int*)offset_value_ptr;
            const float alpha = offset_value_ptr[8];
            const float beta = offset_value_ptr[9];
            const float gamma = offset_value_ptr[10];

            float v000 = gridsample_load_p1_arm(srcptr, offset_ptr[0]);
            float v001 = gridsample_load_p1_arm(srcptr, offset_ptr[1]);
            float v010 = gridsample_load_p1_arm(srcptr, offset_ptr[2]);
            float v011 = gridsample_load_p1_arm(srcptr, offset_ptr[3]);
            float v100 = gridsample_load_p1_arm(srcptr, offset_ptr[4]);
            float v101 = gridsample_load_p1_arm(srcptr, offset_ptr[5]);
            float v110 = gridsample_load_p1_arm(srcptr, offset_ptr[6]);
            float v111 = gridsample_load_p1_arm(srcptr, offset_ptr[7]);

            float v00 = v000 * (1.f - alpha) + v001 * alpha;
            float v01 = v010 * (1.f - alpha) + v011 * alpha;
            float v10 = v100 * (1.f - alpha) + v101 * alpha;
            float v11 = v110 * (1.f - alpha) + v111 * alpha;
            float v0 = v00 * (1.f - beta) + v01 * beta;
            float v1 = v10 * (1.f - beta) + v11 * beta;

            dstptr[0] = v0 * (1.f - gamma) + v1 * gamma;

            dstptr++;
            offset_value_ptr += 11;
        }
    }
}

static void gridsample_nearest_apply_interpolation_p1_arm(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h * dst.d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const int* offset_ptr = offset_value;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < grid_size; i += 4)
        {
            vst1q_f32(dstptr + i, gridsample_set4_ps_neon(gridsample_load_p1_arm(srcptr, offset_ptr[i]), gridsample_load_p1_arm(srcptr, offset_ptr[i + 1]), gridsample_load_p1_arm(srcptr, offset_ptr[i + 2]), gridsample_load_p1_arm(srcptr, offset_ptr[i + 3])));
        }
#endif // __ARM_NEON
        for (; i < grid_size; i++)
        {
            dstptr[i] = gridsample_load_p1_arm(srcptr, offset_ptr[i]);
        }
    }
}

static void gridsample_bicubic_apply_interpolation_p1_arm(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const float* offset_value_ptr = offset_value;

        for (int i = 0; i < grid_size; i++)
        {
            float x_coeffs[4];
            float y_coeffs[4];
            cubic_interp1d_arm(offset_value_ptr[0], x_coeffs);
            cubic_interp1d_arm(offset_value_ptr[1], y_coeffs);

            const int* offset_ptr = (const int*)offset_value_ptr + 2;

            float rows[4];
            for (int j = 0; j < 4; j++)
            {
                float v0 = gridsample_load_p1_arm(srcptr, offset_ptr[0]);
                float v1 = gridsample_load_p1_arm(srcptr, offset_ptr[1]);
                float v2 = gridsample_load_p1_arm(srcptr, offset_ptr[2]);
                float v3 = gridsample_load_p1_arm(srcptr, offset_ptr[3]);

                rows[j] = v0 * x_coeffs[0] + v1 * x_coeffs[1] + v2 * x_coeffs[2] + v3 * x_coeffs[3];

                offset_ptr += 4;
            }

            dstptr[0] = rows[0] * y_coeffs[0] + rows[1] * y_coeffs[1] + rows[2] * y_coeffs[2] + rows[3] * y_coeffs[3];

            dstptr++;
            offset_value_ptr += 18;
        }
    }
}
