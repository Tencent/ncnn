// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if __mips_msa
static v4f32 gridsample_load_pack4_msa(const float* ptr, int offset, v4f32 _zero)
{
    return offset >= 0 ? (v4f32)__msa_ld_w(ptr + offset, 0) : _zero;
}
#endif // __mips_msa

static float gridsample_load_p1_mips(const float* ptr, int offset)
{
    return offset >= 0 ? ptr[offset] : 0.f;
}

#if __mips_msa
static void gridsample_2d_bilinear_apply_interpolation_pack4_msa(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h;
    const v4f32 _zero = (v4f32)__msa_fill_w(0);

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

            v4f32 _v00 = gridsample_load_pack4_msa(srcptr, offset_ptr[0], _zero);
            v4f32 _v01 = gridsample_load_pack4_msa(srcptr, offset_ptr[1], _zero);
            v4f32 _v10 = gridsample_load_pack4_msa(srcptr, offset_ptr[2], _zero);
            v4f32 _v11 = gridsample_load_pack4_msa(srcptr, offset_ptr[3], _zero);

            v4f32 _alpha = __msa_fill_w_f32(value_ptr[0]);
            v4f32 _beta = __msa_fill_w_f32(value_ptr[1]);
            v4f32 _v0 = __ncnn_msa_fmadd_w(_v00, __msa_fsub_w(_v01, _v00), _alpha);
            v4f32 _v1 = __ncnn_msa_fmadd_w(_v10, __msa_fsub_w(_v11, _v10), _alpha);
            v4f32 _v = __ncnn_msa_fmadd_w(_v0, __msa_fsub_w(_v1, _v0), _beta);

            __msa_st_w((v4i32)_v, dstptr, 0);

            dstptr += 4;
            offset_value_ptr += 6;
        }
    }
}

static void gridsample_3d_bilinear_apply_interpolation_pack4_msa(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h * dst.d;
    const v4f32 _zero = (v4f32)__msa_fill_w(0);

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

            v4f32 _v000 = gridsample_load_pack4_msa(srcptr, offset_ptr[0], _zero);
            v4f32 _v001 = gridsample_load_pack4_msa(srcptr, offset_ptr[1], _zero);
            v4f32 _v010 = gridsample_load_pack4_msa(srcptr, offset_ptr[2], _zero);
            v4f32 _v011 = gridsample_load_pack4_msa(srcptr, offset_ptr[3], _zero);
            v4f32 _v100 = gridsample_load_pack4_msa(srcptr, offset_ptr[4], _zero);
            v4f32 _v101 = gridsample_load_pack4_msa(srcptr, offset_ptr[5], _zero);
            v4f32 _v110 = gridsample_load_pack4_msa(srcptr, offset_ptr[6], _zero);
            v4f32 _v111 = gridsample_load_pack4_msa(srcptr, offset_ptr[7], _zero);

            v4f32 _alpha = __msa_fill_w_f32(value_ptr[0]);
            v4f32 _beta = __msa_fill_w_f32(value_ptr[1]);
            v4f32 _gamma = __msa_fill_w_f32(value_ptr[2]);

            v4f32 _v00 = __ncnn_msa_fmadd_w(_v000, __msa_fsub_w(_v001, _v000), _alpha);
            v4f32 _v01 = __ncnn_msa_fmadd_w(_v010, __msa_fsub_w(_v011, _v010), _alpha);
            v4f32 _v10 = __ncnn_msa_fmadd_w(_v100, __msa_fsub_w(_v101, _v100), _alpha);
            v4f32 _v11 = __ncnn_msa_fmadd_w(_v110, __msa_fsub_w(_v111, _v110), _alpha);
            v4f32 _v0 = __ncnn_msa_fmadd_w(_v00, __msa_fsub_w(_v01, _v00), _beta);
            v4f32 _v1 = __ncnn_msa_fmadd_w(_v10, __msa_fsub_w(_v11, _v10), _beta);
            v4f32 _v = __ncnn_msa_fmadd_w(_v0, __msa_fsub_w(_v1, _v0), _gamma);

            __msa_st_w((v4i32)_v, dstptr, 0);

            dstptr += 4;
            offset_value_ptr += 11;
        }
    }
}

static void gridsample_nearest_apply_interpolation_pack4_msa(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h * dst.d;
    const v4f32 _zero = (v4f32)__msa_fill_w(0);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* srcptr = src.channel(q);
        float* dstptr = dst.channel(q);
        const int* offset_ptr = offset_value.channel(0);

        for (int i = 0; i < grid_size; i++)
        {
            v4f32 _v = gridsample_load_pack4_msa(srcptr, offset_ptr[0], _zero);
            __msa_st_w((v4i32)_v, dstptr, 0);

            offset_ptr++;
            dstptr += 4;
        }
    }
}

#endif // __mips_msa

static void cubic_interp1d_mips(float fx, float* coeffs)
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

#if __mips_msa
static void gridsample_2d_bicubic_apply_interpolation_pack4_msa(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
{
    (void)opt;

    const int channels = dst.c;
    const int grid_size = dst.w * dst.h;
    const v4f32 _zero = (v4f32)__msa_fill_w(0);

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
            cubic_interp1d_mips(offset_value_ptr[0], x_coeffs);
            cubic_interp1d_mips(offset_value_ptr[1], y_coeffs);

            const int* offset_ptr = (const int*)offset_value_ptr + 2;
            v4f32 _rows[4];
            for (int j = 0; j < 4; j++)
            {
                v4f32 _v0 = gridsample_load_pack4_msa(srcptr, offset_ptr[0], _zero);
                v4f32 _v1 = gridsample_load_pack4_msa(srcptr, offset_ptr[1], _zero);
                v4f32 _v2 = gridsample_load_pack4_msa(srcptr, offset_ptr[2], _zero);
                v4f32 _v3 = gridsample_load_pack4_msa(srcptr, offset_ptr[3], _zero);

                _rows[j] = __msa_fmul_w(_v0, __msa_fill_w_f32(x_coeffs[0]));
                _rows[j] = __ncnn_msa_fmadd_w(_rows[j], _v1, __msa_fill_w_f32(x_coeffs[1]));
                _rows[j] = __ncnn_msa_fmadd_w(_rows[j], _v2, __msa_fill_w_f32(x_coeffs[2]));
                _rows[j] = __ncnn_msa_fmadd_w(_rows[j], _v3, __msa_fill_w_f32(x_coeffs[3]));

                offset_ptr += 4;
            }

            v4f32 _v = __msa_fmul_w(_rows[0], __msa_fill_w_f32(y_coeffs[0]));
            _v = __ncnn_msa_fmadd_w(_v, _rows[1], __msa_fill_w_f32(y_coeffs[1]));
            _v = __ncnn_msa_fmadd_w(_v, _rows[2], __msa_fill_w_f32(y_coeffs[2]));
            _v = __ncnn_msa_fmadd_w(_v, _rows[3], __msa_fill_w_f32(y_coeffs[3]));

            __msa_st_w((v4i32)_v, dstptr, 0);

            dstptr += 4;
            offset_value_ptr += 18;
        }
    }
}
#endif // __mips_msa

static void gridsample_2d_bilinear_apply_interpolation_p1_mips(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
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
#if __mips_msa
        for (; i + 3 < grid_size; i += 4)
        {
            const float* offset_value_ptr0 = offset_value_ptr;
            const float* offset_value_ptr1 = offset_value_ptr + 6;
            const float* offset_value_ptr2 = offset_value_ptr + 12;
            const float* offset_value_ptr3 = offset_value_ptr + 18;
            const int* offset_ptr0 = (const int*)offset_value_ptr0;
            const int* offset_ptr1 = (const int*)offset_value_ptr1;
            const int* offset_ptr2 = (const int*)offset_value_ptr2;
            const int* offset_ptr3 = (const int*)offset_value_ptr3;
            const float* value_ptr0 = offset_value_ptr0 + 4;
            const float* value_ptr1 = offset_value_ptr1 + 4;
            const float* value_ptr2 = offset_value_ptr2 + 4;
            const float* value_ptr3 = offset_value_ptr3 + 4;

            v4f32 _v00 = gridsample_set4_ps_msa(gridsample_load_p1_mips(srcptr, offset_ptr0[0]), gridsample_load_p1_mips(srcptr, offset_ptr1[0]), gridsample_load_p1_mips(srcptr, offset_ptr2[0]), gridsample_load_p1_mips(srcptr, offset_ptr3[0]));
            v4f32 _v01 = gridsample_set4_ps_msa(gridsample_load_p1_mips(srcptr, offset_ptr0[1]), gridsample_load_p1_mips(srcptr, offset_ptr1[1]), gridsample_load_p1_mips(srcptr, offset_ptr2[1]), gridsample_load_p1_mips(srcptr, offset_ptr3[1]));
            v4f32 _v10 = gridsample_set4_ps_msa(gridsample_load_p1_mips(srcptr, offset_ptr0[2]), gridsample_load_p1_mips(srcptr, offset_ptr1[2]), gridsample_load_p1_mips(srcptr, offset_ptr2[2]), gridsample_load_p1_mips(srcptr, offset_ptr3[2]));
            v4f32 _v11 = gridsample_set4_ps_msa(gridsample_load_p1_mips(srcptr, offset_ptr0[3]), gridsample_load_p1_mips(srcptr, offset_ptr1[3]), gridsample_load_p1_mips(srcptr, offset_ptr2[3]), gridsample_load_p1_mips(srcptr, offset_ptr3[3]));

            v4f32 _alpha = gridsample_set4_ps_msa(value_ptr0[0], value_ptr1[0], value_ptr2[0], value_ptr3[0]);
            v4f32 _beta = gridsample_set4_ps_msa(value_ptr0[1], value_ptr1[1], value_ptr2[1], value_ptr3[1]);
            v4f32 _v0 = __ncnn_msa_fmadd_w(_v00, __msa_fsub_w(_v01, _v00), _alpha);
            v4f32 _v1 = __ncnn_msa_fmadd_w(_v10, __msa_fsub_w(_v11, _v10), _alpha);
            v4f32 _v = __ncnn_msa_fmadd_w(_v0, __msa_fsub_w(_v1, _v0), _beta);

            __msa_st_w((v4i32)_v, dstptr, 0);

            dstptr += 4;
            offset_value_ptr += 24;
        }
#endif // __mips_msa

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

static void gridsample_3d_bilinear_apply_interpolation_p1_mips(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
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
#if __mips_msa
        for (; i + 3 < grid_size; i += 4)
        {
            const float* offset_value_ptr0 = offset_value_ptr;
            const float* offset_value_ptr1 = offset_value_ptr + 11;
            const float* offset_value_ptr2 = offset_value_ptr + 22;
            const float* offset_value_ptr3 = offset_value_ptr + 33;
            const int* offset_ptr0 = (const int*)offset_value_ptr0;
            const int* offset_ptr1 = (const int*)offset_value_ptr1;
            const int* offset_ptr2 = (const int*)offset_value_ptr2;
            const int* offset_ptr3 = (const int*)offset_value_ptr3;
            const float* value_ptr0 = offset_value_ptr0 + 8;
            const float* value_ptr1 = offset_value_ptr1 + 8;
            const float* value_ptr2 = offset_value_ptr2 + 8;
            const float* value_ptr3 = offset_value_ptr3 + 8;

            v4f32 _v000 = gridsample_set4_ps_msa(gridsample_load_p1_mips(srcptr, offset_ptr0[0]), gridsample_load_p1_mips(srcptr, offset_ptr1[0]), gridsample_load_p1_mips(srcptr, offset_ptr2[0]), gridsample_load_p1_mips(srcptr, offset_ptr3[0]));
            v4f32 _v001 = gridsample_set4_ps_msa(gridsample_load_p1_mips(srcptr, offset_ptr0[1]), gridsample_load_p1_mips(srcptr, offset_ptr1[1]), gridsample_load_p1_mips(srcptr, offset_ptr2[1]), gridsample_load_p1_mips(srcptr, offset_ptr3[1]));
            v4f32 _v010 = gridsample_set4_ps_msa(gridsample_load_p1_mips(srcptr, offset_ptr0[2]), gridsample_load_p1_mips(srcptr, offset_ptr1[2]), gridsample_load_p1_mips(srcptr, offset_ptr2[2]), gridsample_load_p1_mips(srcptr, offset_ptr3[2]));
            v4f32 _v011 = gridsample_set4_ps_msa(gridsample_load_p1_mips(srcptr, offset_ptr0[3]), gridsample_load_p1_mips(srcptr, offset_ptr1[3]), gridsample_load_p1_mips(srcptr, offset_ptr2[3]), gridsample_load_p1_mips(srcptr, offset_ptr3[3]));
            v4f32 _v100 = gridsample_set4_ps_msa(gridsample_load_p1_mips(srcptr, offset_ptr0[4]), gridsample_load_p1_mips(srcptr, offset_ptr1[4]), gridsample_load_p1_mips(srcptr, offset_ptr2[4]), gridsample_load_p1_mips(srcptr, offset_ptr3[4]));
            v4f32 _v101 = gridsample_set4_ps_msa(gridsample_load_p1_mips(srcptr, offset_ptr0[5]), gridsample_load_p1_mips(srcptr, offset_ptr1[5]), gridsample_load_p1_mips(srcptr, offset_ptr2[5]), gridsample_load_p1_mips(srcptr, offset_ptr3[5]));
            v4f32 _v110 = gridsample_set4_ps_msa(gridsample_load_p1_mips(srcptr, offset_ptr0[6]), gridsample_load_p1_mips(srcptr, offset_ptr1[6]), gridsample_load_p1_mips(srcptr, offset_ptr2[6]), gridsample_load_p1_mips(srcptr, offset_ptr3[6]));
            v4f32 _v111 = gridsample_set4_ps_msa(gridsample_load_p1_mips(srcptr, offset_ptr0[7]), gridsample_load_p1_mips(srcptr, offset_ptr1[7]), gridsample_load_p1_mips(srcptr, offset_ptr2[7]), gridsample_load_p1_mips(srcptr, offset_ptr3[7]));

            v4f32 _alpha = gridsample_set4_ps_msa(value_ptr0[0], value_ptr1[0], value_ptr2[0], value_ptr3[0]);
            v4f32 _beta = gridsample_set4_ps_msa(value_ptr0[1], value_ptr1[1], value_ptr2[1], value_ptr3[1]);
            v4f32 _gamma = gridsample_set4_ps_msa(value_ptr0[2], value_ptr1[2], value_ptr2[2], value_ptr3[2]);

            v4f32 _v00 = __ncnn_msa_fmadd_w(_v000, __msa_fsub_w(_v001, _v000), _alpha);
            v4f32 _v01 = __ncnn_msa_fmadd_w(_v010, __msa_fsub_w(_v011, _v010), _alpha);
            v4f32 _v10 = __ncnn_msa_fmadd_w(_v100, __msa_fsub_w(_v101, _v100), _alpha);
            v4f32 _v11 = __ncnn_msa_fmadd_w(_v110, __msa_fsub_w(_v111, _v110), _alpha);
            v4f32 _v0 = __ncnn_msa_fmadd_w(_v00, __msa_fsub_w(_v01, _v00), _beta);
            v4f32 _v1 = __ncnn_msa_fmadd_w(_v10, __msa_fsub_w(_v11, _v10), _beta);
            v4f32 _v = __ncnn_msa_fmadd_w(_v0, __msa_fsub_w(_v1, _v0), _gamma);

            __msa_st_w((v4i32)_v, dstptr, 0);

            dstptr += 4;
            offset_value_ptr += 44;
        }
#endif // __mips_msa

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

static void gridsample_nearest_apply_interpolation_p1_mips(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
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
#if __mips_msa
        for (; i + 3 < grid_size; i += 4)
        {
            v4f32 _v = gridsample_set4_ps_msa(gridsample_load_p1_mips(srcptr, offset_ptr[0]), gridsample_load_p1_mips(srcptr, offset_ptr[1]), gridsample_load_p1_mips(srcptr, offset_ptr[2]), gridsample_load_p1_mips(srcptr, offset_ptr[3]));
            __msa_st_w((v4i32)_v, dstptr, 0);

            offset_ptr += 4;
            dstptr += 4;
        }
#endif // __mips_msa

        for (; i < grid_size; i++)
        {
            dstptr[0] = offset_ptr[0] >= 0 ? srcptr[offset_ptr[0]] : 0.f;

            offset_ptr++;
            dstptr++;
        }
    }
}

static void gridsample_2d_bicubic_apply_interpolation_p1_mips(const Mat& src, Mat& dst, const Mat& offset_value, const Option& opt)
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
            cubic_interp1d_mips(offset_value_ptr[0], x_coeffs);
            cubic_interp1d_mips(offset_value_ptr[1], y_coeffs);

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
