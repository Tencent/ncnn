// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void innerproduct_transform_kernel_bf16s_lsx(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt)
{
    int out_elempack = 1;
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
#if __loongarch_asx
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif

    // src = inch-outch
    // dst = pb-inch-outch/pb
#if __loongarch_sx
#if __loongarch_asx
    if (out_elempack == 8)
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_tm.create(num_input, num_output / 8, (size_t)16u, 8);

        for (int q = 0; q + 7 < num_output; q += 8)
        {
            unsigned short* g0 = weight_data_tm.row<unsigned short>(q / 8);

            const float* k0 = weight_data_r2.row(q);
            const float* k1 = weight_data_r2.row(q + 1);
            const float* k2 = weight_data_r2.row(q + 2);
            const float* k3 = weight_data_r2.row(q + 3);
            const float* k4 = weight_data_r2.row(q + 4);
            const float* k5 = weight_data_r2.row(q + 5);
            const float* k6 = weight_data_r2.row(q + 6);
            const float* k7 = weight_data_r2.row(q + 7);

            int p = 0;
            for (; p + 7 < num_input; p += 8)
            {
                // transpose 8x8
                __m256 _r0 = (__m256)__lasx_xvld(k0, 0);
                __m256 _r1 = (__m256)__lasx_xvld(k1, 0);
                __m256 _r2 = (__m256)__lasx_xvld(k2, 0);
                __m256 _r3 = (__m256)__lasx_xvld(k3, 0);
                __m256 _r4 = (__m256)__lasx_xvld(k4, 0);
                __m256 _r5 = (__m256)__lasx_xvld(k5, 0);
                __m256 _r6 = (__m256)__lasx_xvld(k6, 0);
                __m256 _r7 = (__m256)__lasx_xvld(k7, 0);

                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                __m128i _bf16_0 = float2bfloat_lasx(_r0);
                __m128i _bf16_1 = float2bfloat_lasx(_r1);
                __m128i _bf16_2 = float2bfloat_lasx(_r2);
                __m128i _bf16_3 = float2bfloat_lasx(_r3);
                __m128i _bf16_4 = float2bfloat_lasx(_r4);
                __m128i _bf16_5 = float2bfloat_lasx(_r5);
                __m128i _bf16_6 = float2bfloat_lasx(_r6);
                __m128i _bf16_7 = float2bfloat_lasx(_r7);

                __lsx_vst(_bf16_0, g0, 0);
                __lsx_vst(_bf16_1, g0 + 8, 0);
                __lsx_vst(_bf16_2, g0 + 16, 0);
                __lsx_vst(_bf16_3, g0 + 24, 0);
                __lsx_vst(_bf16_4, g0 + 32, 0);
                __lsx_vst(_bf16_5, g0 + 40, 0);
                __lsx_vst(_bf16_6, g0 + 48, 0);
                __lsx_vst(_bf16_7, g0 + 56, 0);

                k0 += 8;
                k1 += 8;
                k2 += 8;
                k3 += 8;
                k4 += 8;
                k5 += 8;
                k6 += 8;
                k7 += 8;
                g0 += 64;
            }
            for (; p < num_input; p++)
            {
                g0[0] = float32_to_bfloat16(*k0++);
                g0[1] = float32_to_bfloat16(*k1++);
                g0[2] = float32_to_bfloat16(*k2++);
                g0[3] = float32_to_bfloat16(*k3++);
                g0[4] = float32_to_bfloat16(*k4++);
                g0[5] = float32_to_bfloat16(*k5++);
                g0[6] = float32_to_bfloat16(*k6++);
                g0[7] = float32_to_bfloat16(*k7++);
                g0 += 8;
            }
        }
    }
#endif // __loongarch_asx

    if (out_elempack == 4)
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_tm.create(num_input, num_output / 4, (size_t)8u, 4);

        for (int q = 0; q + 3 < num_output; q += 4)
        {
            unsigned short* g0 = weight_data_tm.row<unsigned short>(q / 4);

            const float* k0 = weight_data_r2.row(q);
            const float* k1 = weight_data_r2.row(q + 1);
            const float* k2 = weight_data_r2.row(q + 2);
            const float* k3 = weight_data_r2.row(q + 3);

            int p = 0;
            for (; p + 3 < num_input; p += 4)
            {
                __m128 _r0 = (__m128)__lsx_vld(k0, 0);
                __m128 _r1 = (__m128)__lsx_vld(k1, 0);
                __m128 _r2 = (__m128)__lsx_vld(k2, 0);
                __m128 _r3 = (__m128)__lsx_vld(k3, 0);

                transpose4x4_ps(_r0, _r1, _r2, _r3);

                __m128i _bf16_01 = float2bfloat_lsx(_r0, _r1);
                __m128i _bf16_23 = float2bfloat_lsx(_r2, _r3);
                __lsx_vst(_bf16_01, g0, 0);
                __lsx_vst(_bf16_23, g0 + 8, 0);

                k0 += 4;
                k1 += 4;
                k2 += 4;
                k3 += 4;
                g0 += 16;
            }
            for (; p < num_input; p++)
            {
                g0[0] = float32_to_bfloat16(*k0++);
                g0[1] = float32_to_bfloat16(*k1++);
                g0[2] = float32_to_bfloat16(*k2++);
                g0[3] = float32_to_bfloat16(*k3++);
                g0 += 4;
            }
        }
    }
#endif // __loongarch_sx

    if (out_elempack == 1)
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);
        ncnn::cast_float32_to_bfloat16(weight_data_r2, weight_data_tm, opt);
    }
}

static void innerproduct_gemm_bf16s_lsx(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int num_input = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int num_output = top_blob.w;
    const int h = bottom_blob.h;

    const float* bias_data_ptr = bias_data;

    int num_output_elempack = 1;
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
#if __loongarch_asx
        num_output_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        num_output_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int j = 0; j < h; j++)
    {
#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8 && num_output_elempack == 8)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m256 _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
                __m256 _sum1 = _sum0;
                __m256 _sum2 = _sum0;
                __m256 _sum3 = _sum0;
                __m256 _sum4 = _sum0;
                __m256 _sum5 = _sum0;
                __m256 _sum6 = _sum0;
                __m256 _sum7 = _sum0;

                if (bias_data_ptr)
                {
                    _sum0 = (__m256)__lasx_xvld(bias_data_ptr + p * 8, 0);
                }
                _sum1 = _sum0;
                _sum2 = _sum0;
                _sum3 = _sum0;
                _sum4 = _sum0;
                _sum5 = _sum0;
                _sum6 = _sum0;
                _sum7 = _sum0;

                int i = 0;
                for (; i < num_input; i++)
                {
                    __m256 _val0 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(m[0]));
                    __m256 _val1 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(m[1]));
                    __m256 _val2 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(m[2]));
                    __m256 _val3 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(m[3]));
                    __m256 _val4 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(m[4]));
                    __m256 _val5 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(m[5]));
                    __m256 _val6 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(m[6]));
                    __m256 _val7 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(m[7]));
                    __m256 _w = bfloat2float_lasx(__lsx_vld(kptr, 0));
                    _sum0 = __lasx_xvfmadd_s(_val0, _w, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_val1, _w, _sum1);
                    _sum2 = __lasx_xvfmadd_s(_val2, _w, _sum2);
                    _sum3 = __lasx_xvfmadd_s(_val3, _w, _sum3);
                    _sum4 = __lasx_xvfmadd_s(_val4, _w, _sum4);
                    _sum5 = __lasx_xvfmadd_s(_val5, _w, _sum5);
                    _sum6 = __lasx_xvfmadd_s(_val6, _w, _sum6);
                    _sum7 = __lasx_xvfmadd_s(_val7, _w, _sum7);

                    m += 8;
                    kptr += 8;
                }

                _sum0 = activation_lasx(_sum0, activation_type, activation_params);
                _sum1 = activation_lasx(_sum1, activation_type, activation_params);
                _sum2 = activation_lasx(_sum2, activation_type, activation_params);
                _sum3 = activation_lasx(_sum3, activation_type, activation_params);
                _sum4 = activation_lasx(_sum4, activation_type, activation_params);
                _sum5 = activation_lasx(_sum5, activation_type, activation_params);
                _sum6 = activation_lasx(_sum6, activation_type, activation_params);
                _sum7 = activation_lasx(_sum7, activation_type, activation_params);

                // transpose 8x8 and store as bf16
                transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                __lsx_vst(float2bfloat_lasx(_sum0), outptr, 0);
                __lsx_vst(float2bfloat_lasx(_sum1), outptr + 8, 0);
                __lsx_vst(float2bfloat_lasx(_sum2), outptr + 16, 0);
                __lsx_vst(float2bfloat_lasx(_sum3), outptr + 24, 0);
                __lsx_vst(float2bfloat_lasx(_sum4), outptr + 32, 0);
                __lsx_vst(float2bfloat_lasx(_sum5), outptr + 40, 0);
                __lsx_vst(float2bfloat_lasx(_sum6), outptr + 48, 0);
                __lsx_vst(float2bfloat_lasx(_sum7), outptr + 56, 0);
                outptr += 64;
            }
        }

        if (elempack == 1 && num_output_elempack == 8)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m256 _sum = (__m256)__lasx_xvreplgr2vr_w(0);

                if (bias_data_ptr)
                {
                    _sum = (__m256)__lasx_xvld(bias_data_ptr + p * 8, 0);
                }

                int i = 0;
                for (; i < num_input; i++)
                {
                    __m256 _val = __lasx_xvreplfr2vr_s(bfloat16_to_float32(m[0]));
                    __m256 _w = bfloat2float_lasx(__lsx_vld(kptr, 0));
                    _sum = __lasx_xvfmadd_s(_val, _w, _sum);

                    m += 1;
                    kptr += 8;
                }

                _sum = activation_lasx(_sum, activation_type, activation_params);

                __lsx_vst(float2bfloat_lasx(_sum), outptr, 0);
                outptr += 8;
            }
        }

        if (elempack == 4 && num_output_elempack == 8)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m256 _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);

                if (bias_data_ptr)
                {
                    _sum0 = (__m256)__lasx_xvld(bias_data_ptr + p * 8, 0);
                }

                __m256 _sum1 = _sum0;
                __m256 _sum2 = _sum0;
                __m256 _sum3 = _sum0;

                int i = 0;
                for (; i < num_input; i++)
                {
                    __m256 _val0 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(m[0]));
                    __m256 _val1 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(m[1]));
                    __m256 _val2 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(m[2]));
                    __m256 _val3 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(m[3]));
                    __m256 _w = bfloat2float_lasx(__lsx_vld(kptr, 0));
                    _sum0 = __lasx_xvfmadd_s(_val0, _w, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_val1, _w, _sum1);
                    _sum2 = __lasx_xvfmadd_s(_val2, _w, _sum2);
                    _sum3 = __lasx_xvfmadd_s(_val3, _w, _sum3);

                    m += 4;
                    kptr += 8;
                }

                _sum0 = activation_lasx(_sum0, activation_type, activation_params);
                _sum1 = activation_lasx(_sum1, activation_type, activation_params);
                _sum2 = activation_lasx(_sum2, activation_type, activation_params);
                _sum3 = activation_lasx(_sum3, activation_type, activation_params);

                // transpose 8x4 and store
                transpose8x4_ps(_sum0, _sum1, _sum2, _sum3);

                __lsx_vst(float2bfloat_lasx(_sum0), outptr, 0);
                __lsx_vst(float2bfloat_lasx(_sum1), outptr + 8, 0);
                __lsx_vst(float2bfloat_lasx(_sum2), outptr + 16, 0);
                __lsx_vst(float2bfloat_lasx(_sum3), outptr + 24, 0);
                outptr += 32;
            }
        }

        if (elempack == 8 && num_output_elempack == 4)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);

                if (bias_data_ptr)
                {
                    _sum0 = (__m128)__lsx_vld(bias_data_ptr + p * 4, 0);
                }

                __m128 _sum1 = _sum0;
                __m128 _sum2 = _sum0;
                __m128 _sum3 = _sum0;
                __m128 _sum4 = _sum0;
                __m128 _sum5 = _sum0;
                __m128 _sum6 = _sum0;
                __m128 _sum7 = _sum0;

                int i = 0;
                for (; i < num_input; i++)
                {
                    __m128 _val0 = __lsx_vreplfr2vr_s(bfloat16_to_float32(m[0]));
                    __m128 _val1 = __lsx_vreplfr2vr_s(bfloat16_to_float32(m[1]));
                    __m128 _val2 = __lsx_vreplfr2vr_s(bfloat16_to_float32(m[2]));
                    __m128 _val3 = __lsx_vreplfr2vr_s(bfloat16_to_float32(m[3]));
                    __m128 _val4 = __lsx_vreplfr2vr_s(bfloat16_to_float32(m[4]));
                    __m128 _val5 = __lsx_vreplfr2vr_s(bfloat16_to_float32(m[5]));
                    __m128 _val6 = __lsx_vreplfr2vr_s(bfloat16_to_float32(m[6]));
                    __m128 _val7 = __lsx_vreplfr2vr_s(bfloat16_to_float32(m[7]));
                    __m128 _w = bfloat2float_lsx(kptr);
                    _sum0 = __lsx_vfmadd_s(_val0, _w, _sum0);
                    _sum1 = __lsx_vfmadd_s(_val1, _w, _sum1);
                    _sum2 = __lsx_vfmadd_s(_val2, _w, _sum2);
                    _sum3 = __lsx_vfmadd_s(_val3, _w, _sum3);
                    _sum4 = __lsx_vfmadd_s(_val4, _w, _sum4);
                    _sum5 = __lsx_vfmadd_s(_val5, _w, _sum5);
                    _sum6 = __lsx_vfmadd_s(_val6, _w, _sum6);
                    _sum7 = __lsx_vfmadd_s(_val7, _w, _sum7);

                    m += 8;
                    kptr += 4;
                }

                _sum0 = activation_lsx(_sum0, activation_type, activation_params);
                _sum1 = activation_lsx(_sum1, activation_type, activation_params);
                _sum2 = activation_lsx(_sum2, activation_type, activation_params);
                _sum3 = activation_lsx(_sum3, activation_type, activation_params);
                _sum4 = activation_lsx(_sum4, activation_type, activation_params);
                _sum5 = activation_lsx(_sum5, activation_type, activation_params);
                _sum6 = activation_lsx(_sum6, activation_type, activation_params);
                _sum7 = activation_lsx(_sum7, activation_type, activation_params);

                // transpose 4x8 and store as bf16
                // _sum0..3 hold batch 0..3, _sum4..7 hold batch 4..7
                // each has [out0, out1, out2, out3]
                transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);
                // after transpose: _sum0=[out0_b0..3], _sum4=[out0_b4..7] etc.

                __m128i _bf16_0 = float2bfloat_lsx(_sum0, _sum4);
                __m128i _bf16_1 = float2bfloat_lsx(_sum1, _sum5);
                __m128i _bf16_2 = float2bfloat_lsx(_sum2, _sum6);
                __m128i _bf16_3 = float2bfloat_lsx(_sum3, _sum7);
                __lsx_vst(_bf16_0, outptr, 0);
                __lsx_vst(_bf16_1, outptr + 8, 0);
                __lsx_vst(_bf16_2, outptr + 16, 0);
                __lsx_vst(_bf16_3, outptr + 24, 0);
                outptr += 32;
            }
        }

        if (elempack == 8 && num_output_elempack == 1)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m256 _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
                __m256 _sum1 = _sum0;
                __m256 _sum2 = _sum0;
                __m256 _sum3 = _sum0;

                if (bias_data_ptr)
                {
                    _sum0 = __lasx_xvreplfr2vr_s(bias_data_ptr[p]);
                }

                int i = 0;
                for (; i + 3 < num_input; i += 4)
                {
                    __m256 _val0 = bfloat2float_lasx(__lsx_vld(m, 0));
                    __m256 _val1 = bfloat2float_lasx(__lsx_vld(m + 8, 0));
                    __m256 _val2 = bfloat2float_lasx(__lsx_vld(m + 16, 0));
                    __m256 _val3 = bfloat2float_lasx(__lsx_vld(m + 24, 0));

                    __m256 _w0 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(kptr[0]));
                    __m256 _w1 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(kptr[1]));
                    __m256 _w2 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(kptr[2]));
                    __m256 _w3 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(kptr[3]));

                    _sum0 = __lasx_xvfmadd_s(_val0, _w0, _sum0);
                    _sum1 = __lasx_xvfmadd_s(_val1, _w1, _sum1);
                    _sum2 = __lasx_xvfmadd_s(_val2, _w2, _sum2);
                    _sum3 = __lasx_xvfmadd_s(_val3, _w3, _sum3);

                    m += 32;
                    kptr += 4;
                }
                for (; i < num_input; i++)
                {
                    __m256 _val = bfloat2float_lasx(__lsx_vld(m, 0));
                    __m256 _w = __lasx_xvreplfr2vr_s(bfloat16_to_float32(kptr[0]));
                    _sum0 = __lasx_xvfmadd_s(_val, _w, _sum0);

                    m += 8;
                    kptr += 1;
                }

                _sum0 = __lasx_xvfadd_s(_sum0, _sum1);
                _sum2 = __lasx_xvfadd_s(_sum2, _sum3);
                _sum0 = __lasx_xvfadd_s(_sum0, _sum2);

                _sum0 = activation_lasx(_sum0, activation_type, activation_params);

                __lsx_vst(float2bfloat_lasx(_sum0), outptr, 0);
                outptr += 8;
            }
        }
#endif // __loongarch_asx

        if (elempack == 4 && num_output_elempack == 4)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);

                if (bias_data_ptr)
                {
                    _sum0 = (__m128)__lsx_vld(bias_data_ptr + p * 4, 0);
                }

                __m128 _sum1 = _sum0;
                __m128 _sum2 = _sum0;
                __m128 _sum3 = _sum0;

                int i = 0;
                for (; i < num_input; i++)
                {
                    __m128 _val0 = __lsx_vreplfr2vr_s(bfloat16_to_float32(m[0]));
                    __m128 _val1 = __lsx_vreplfr2vr_s(bfloat16_to_float32(m[1]));
                    __m128 _val2 = __lsx_vreplfr2vr_s(bfloat16_to_float32(m[2]));
                    __m128 _val3 = __lsx_vreplfr2vr_s(bfloat16_to_float32(m[3]));
                    __m128 _w = bfloat2float_lsx(kptr);
                    _sum0 = __lsx_vfmadd_s(_val0, _w, _sum0);
                    _sum1 = __lsx_vfmadd_s(_val1, _w, _sum1);
                    _sum2 = __lsx_vfmadd_s(_val2, _w, _sum2);
                    _sum3 = __lsx_vfmadd_s(_val3, _w, _sum3);

                    m += 4;
                    kptr += 4;
                }

                _sum0 = activation_lsx(_sum0, activation_type, activation_params);
                _sum1 = activation_lsx(_sum1, activation_type, activation_params);
                _sum2 = activation_lsx(_sum2, activation_type, activation_params);
                _sum3 = activation_lsx(_sum3, activation_type, activation_params);

                transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);

                __m128i _bf16_01 = float2bfloat_lsx(_sum0, _sum1);
                __m128i _bf16_23 = float2bfloat_lsx(_sum2, _sum3);
                __lsx_vst(_bf16_01, outptr, 0);
                __lsx_vst(_bf16_23, outptr + 8, 0);
                outptr += 16;
            }
        }

        if (elempack == 1 && num_output_elempack == 4)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m128 _sum = (__m128)__lsx_vreplgr2vr_w(0);

                if (bias_data_ptr)
                {
                    _sum = (__m128)__lsx_vld(bias_data_ptr + p * 4, 0);
                }

                int i = 0;
                for (; i < num_input; i++)
                {
                    __m128 _val = __lsx_vreplfr2vr_s(bfloat16_to_float32(m[0]));
                    __m128 _w = bfloat2float_lsx(kptr);
                    _sum = __lsx_vfmadd_s(_val, _w, _sum);

                    m += 1;
                    kptr += 4;
                }

                _sum = activation_lsx(_sum, activation_type, activation_params);

                __lsx_vstelm_d(float2bfloat_lsx(_sum), outptr, 0, 0);
                outptr += 4;
            }
        }

        if (elempack == 4 && num_output_elempack == 1)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _sum1 = _sum0;
                __m128 _sum2 = _sum0;
                __m128 _sum3 = _sum0;

                if (bias_data_ptr)
                {
                    _sum0 = __lsx_vreplfr2vr_s(bias_data_ptr[p]);
                }

                int i = 0;
                for (; i + 3 < num_input; i += 4)
                {
                    __m128 _val0 = bfloat2float_lsx(m);
                    __m128 _val1 = bfloat2float_lsx(m + 4);
                    __m128 _val2 = bfloat2float_lsx(m + 8);
                    __m128 _val3 = bfloat2float_lsx(m + 12);

                    __m128 _w = bfloat2float_lsx(kptr);

                    __m128 _w0 = (__m128)__lsx_vreplvei_w((__m128i)_w, 0);
                    __m128 _w1 = (__m128)__lsx_vreplvei_w((__m128i)_w, 1);
                    __m128 _w2 = (__m128)__lsx_vreplvei_w((__m128i)_w, 2);
                    __m128 _w3 = (__m128)__lsx_vreplvei_w((__m128i)_w, 3);

                    _sum0 = __lsx_vfmadd_s(_val0, _w0, _sum0);
                    _sum1 = __lsx_vfmadd_s(_val1, _w1, _sum1);
                    _sum2 = __lsx_vfmadd_s(_val2, _w2, _sum2);
                    _sum3 = __lsx_vfmadd_s(_val3, _w3, _sum3);

                    m += 16;
                    kptr += 4;
                }
                for (; i < num_input; i++)
                {
                    __m128 _val = bfloat2float_lsx(m);
                    __m128 _w = __lsx_vreplfr2vr_s(bfloat16_to_float32(kptr[0]));
                    _sum0 = __lsx_vfmadd_s(_val, _w, _sum0);

                    m += 4;
                    kptr += 1;
                }

                _sum0 = __lsx_vfadd_s(_sum0, _sum1);
                _sum2 = __lsx_vfadd_s(_sum2, _sum3);
                _sum0 = __lsx_vfadd_s(_sum0, _sum2);

                _sum0 = activation_lsx(_sum0, activation_type, activation_params);

                __lsx_vstelm_d(float2bfloat_lsx(_sum0), outptr, 0, 0);
                outptr += 4;
            }
        }
#endif // __loongarch_sx

        if (elempack == 1 && num_output_elempack == 1)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                float sum = 0.f;

                if (bias_data_ptr)
                {
                    sum = bias_data_ptr[p];
                }

                int i = 0;
#if __loongarch_sx
#if __loongarch_asx
                __m256 _sum256 = (__m256)__lasx_xvreplgr2vr_w(0);
                for (; i + 7 < num_input; i += 8)
                {
                    __m256 _m = bfloat2float_lasx(__lsx_vld(m, 0));
                    __m256 _w = bfloat2float_lasx(__lsx_vld(kptr, 0));
                    _sum256 = __lasx_xvfmadd_s(_m, _w, _sum256);

                    m += 8;
                    kptr += 8;
                }
#endif // __loongarch_asx
                __m128 _suml = (__m128)__lsx_vreplgr2vr_w(0);
                for (; i + 3 < num_input; i += 4)
                {
                    __m128 _val = bfloat2float_lsx(m);
                    __m128 _w = bfloat2float_lsx(kptr);
                    _suml = __lsx_vfmadd_s(_val, _w, _suml);

                    m += 4;
                    kptr += 4;
                }
#endif // __loongarch_sx
                for (; i < num_input; i++)
                {
                    sum += bfloat16_to_float32(*m++) * bfloat16_to_float32(*kptr++);
                }

#if __loongarch_sx
#if __loongarch_asx
                __m128 _lo = __lasx_extract_128_lo_s(_sum256);
                __m128 _hi = __lasx_extract_128_hi_s(_sum256);
                _suml = __lsx_vfadd_s(_suml, _lo);
                _suml = __lsx_vfadd_s(_suml, _hi);
#endif // __loongarch_asx
                sum += __lsx_reduce_fadd_s(_suml);
#endif // __loongarch_sx

                sum = activation_ss(sum, activation_type, activation_params);

                outptr[0] = float32_to_bfloat16(sum);
                outptr += 1;
            }
        }
    }
}

static void innerproduct_bf16s_lsx(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int num_input = bottom_blob.w * bottom_blob.elempack;
    const int outw = top_blob.w;
    const int out_elempack = top_blob.elempack;

    const float* bias_data_ptr = bias_data;

#if __loongarch_sx
#if __loongarch_asx
    if (out_elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outw; p++)
        {
            __m256 _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
            __m256 _sum1 = _sum0;
            __m256 _sum2 = _sum0;
            __m256 _sum3 = _sum0;
            __m256 _sum4 = _sum0;
            __m256 _sum5 = _sum0;
            __m256 _sum6 = _sum0;
            __m256 _sum7 = _sum0;

            if (bias_data_ptr)
            {
                _sum0 = (__m256)__lasx_xvld(bias_data_ptr + p * 8, 0);
            }

            const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* sptr = bottom_blob;

            int i = 0;
            for (; i + 7 < num_input; i += 8)
            {
                __m256 _val0 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(sptr[0]));
                __m256 _val1 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(sptr[1]));
                __m256 _val2 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(sptr[2]));
                __m256 _val3 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(sptr[3]));

                __m256 _w0 = bfloat2float_lasx(__lsx_vld(kptr, 0));
                __m256 _w1 = bfloat2float_lasx(__lsx_vld(kptr + 8, 0));
                __m256 _w2 = bfloat2float_lasx(__lsx_vld(kptr + 16, 0));
                __m256 _w3 = bfloat2float_lasx(__lsx_vld(kptr + 24, 0));

                _sum0 = __lasx_xvfmadd_s(_val0, _w0, _sum0);
                _sum1 = __lasx_xvfmadd_s(_val1, _w1, _sum1);
                _sum2 = __lasx_xvfmadd_s(_val2, _w2, _sum2);
                _sum3 = __lasx_xvfmadd_s(_val3, _w3, _sum3);

                __m256 _val4 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(sptr[4]));
                __m256 _val5 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(sptr[5]));
                __m256 _val6 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(sptr[6]));
                __m256 _val7 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(sptr[7]));

                __m256 _w4 = bfloat2float_lasx(__lsx_vld(kptr + 32, 0));
                __m256 _w5 = bfloat2float_lasx(__lsx_vld(kptr + 40, 0));
                __m256 _w6 = bfloat2float_lasx(__lsx_vld(kptr + 48, 0));
                __m256 _w7 = bfloat2float_lasx(__lsx_vld(kptr + 56, 0));

                _sum4 = __lasx_xvfmadd_s(_val4, _w4, _sum4);
                _sum5 = __lasx_xvfmadd_s(_val5, _w5, _sum5);
                _sum6 = __lasx_xvfmadd_s(_val6, _w6, _sum6);
                _sum7 = __lasx_xvfmadd_s(_val7, _w7, _sum7);

                sptr += 8;
                kptr += 64;
            }
            for (; i + 3 < num_input; i += 4)
            {
                __m256 _val0 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(sptr[0]));
                __m256 _val1 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(sptr[1]));
                __m256 _val2 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(sptr[2]));
                __m256 _val3 = __lasx_xvreplfr2vr_s(bfloat16_to_float32(sptr[3]));

                __m256 _w0 = bfloat2float_lasx(__lsx_vld(kptr, 0));
                __m256 _w1 = bfloat2float_lasx(__lsx_vld(kptr + 8, 0));
                __m256 _w2 = bfloat2float_lasx(__lsx_vld(kptr + 16, 0));
                __m256 _w3 = bfloat2float_lasx(__lsx_vld(kptr + 24, 0));

                _sum0 = __lasx_xvfmadd_s(_val0, _w0, _sum0);
                _sum1 = __lasx_xvfmadd_s(_val1, _w1, _sum1);
                _sum2 = __lasx_xvfmadd_s(_val2, _w2, _sum2);
                _sum3 = __lasx_xvfmadd_s(_val3, _w3, _sum3);

                sptr += 4;
                kptr += 32;
            }
            for (; i < num_input; i++)
            {
                __m256 _val = __lasx_xvreplfr2vr_s(bfloat16_to_float32(sptr[0]));
                __m256 _w = bfloat2float_lasx(__lsx_vld(kptr, 0));
                _sum0 = __lasx_xvfmadd_s(_val, _w, _sum0);

                sptr += 1;
                kptr += 8;
            }

            _sum0 = __lasx_xvfadd_s(_sum0, _sum1);
            _sum2 = __lasx_xvfadd_s(_sum2, _sum3);
            _sum4 = __lasx_xvfadd_s(_sum4, _sum5);
            _sum6 = __lasx_xvfadd_s(_sum6, _sum7);
            _sum0 = __lasx_xvfadd_s(_sum0, _sum2);
            _sum4 = __lasx_xvfadd_s(_sum4, _sum6);
            _sum0 = __lasx_xvfadd_s(_sum0, _sum4);

            _sum0 = activation_lasx(_sum0, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            __lsx_vst(float2bfloat_lasx(_sum0), outptr + p * 8, 0);
        }
    }
#endif // __loongarch_asx

    if (out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outw; p++)
        {
            __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum1 = _sum0;
            __m128 _sum2 = _sum0;
            __m128 _sum3 = _sum0;

            if (bias_data_ptr)
            {
                _sum0 = (__m128)__lsx_vld(bias_data_ptr + p * 4, 0);
            }

            const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* sptr = bottom_blob;

            int i = 0;
            for (; i + 3 < num_input; i += 4)
            {
                __m128 _val0 = __lsx_vreplfr2vr_s(bfloat16_to_float32(sptr[0]));
                __m128 _val1 = __lsx_vreplfr2vr_s(bfloat16_to_float32(sptr[1]));
                __m128 _val2 = __lsx_vreplfr2vr_s(bfloat16_to_float32(sptr[2]));
                __m128 _val3 = __lsx_vreplfr2vr_s(bfloat16_to_float32(sptr[3]));

                __m128 _w0 = bfloat2float_lsx(kptr);
                __m128 _w1 = bfloat2float_lsx(kptr + 4);
                __m128 _w2 = bfloat2float_lsx(kptr + 8);
                __m128 _w3 = bfloat2float_lsx(kptr + 12);

                _sum0 = __lsx_vfmadd_s(_val0, _w0, _sum0);
                _sum1 = __lsx_vfmadd_s(_val1, _w1, _sum1);
                _sum2 = __lsx_vfmadd_s(_val2, _w2, _sum2);
                _sum3 = __lsx_vfmadd_s(_val3, _w3, _sum3);

                sptr += 4;
                kptr += 16;
            }
            for (; i < num_input; i++)
            {
                __m128 _val = __lsx_vreplfr2vr_s(bfloat16_to_float32(sptr[0]));
                __m128 _w = bfloat2float_lsx(kptr);
                _sum0 = __lsx_vfmadd_s(_val, _w, _sum0);

                sptr += 1;
                kptr += 4;
            }

            _sum0 = __lsx_vfadd_s(_sum0, _sum1);
            _sum2 = __lsx_vfadd_s(_sum2, _sum3);
            _sum0 = __lsx_vfadd_s(_sum0, _sum2);

            _sum0 = activation_lsx(_sum0, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            __lsx_vstelm_d(float2bfloat_lsx(_sum0), outptr + p * 4, 0, 0);
        }
    }
#else  // !__loongarch_sx
    (void)out_elempack;
#endif // __loongarch_sx

    if (out_elempack == 1)
    {
#if __loongarch_sx
#if __loongarch_asx
        int remain_outw_start = 0;
        int nn_outw = outw >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outw; pp++)
        {
            int p = remain_outw_start + (pp * 8);

            float sums[8] = {0.0f};
            if (bias_data_ptr)
            {
                for (int k = 0; k < 8; k++)
                    sums[k] = bias_data_ptr[p + k];
            }

            const unsigned short* w0 = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* w1 = weight_data_tm.row<const unsigned short>(p + 1);
            const unsigned short* w2 = weight_data_tm.row<const unsigned short>(p + 2);
            const unsigned short* w3 = weight_data_tm.row<const unsigned short>(p + 3);
            const unsigned short* w4 = weight_data_tm.row<const unsigned short>(p + 4);
            const unsigned short* w5 = weight_data_tm.row<const unsigned short>(p + 5);
            const unsigned short* w6 = weight_data_tm.row<const unsigned short>(p + 6);
            const unsigned short* w7 = weight_data_tm.row<const unsigned short>(p + 7);
            const unsigned short* m = bottom_blob;

            int i = 0;
            __m256 _sum0 = (__m256)__lasx_xvreplgr2vr_w(0);
            __m256 _sum1 = _sum0;
            __m256 _sum2 = _sum0;
            __m256 _sum3 = _sum0;
            __m256 _sum4 = _sum0;
            __m256 _sum5 = _sum0;
            __m256 _sum6 = _sum0;
            __m256 _sum7 = _sum0;
            for (; i + 7 < num_input; i += 8)
            {
                __m256 _m = bfloat2float_lasx(__lsx_vld(m, 0));

                _sum0 = __lasx_xvfmadd_s(_m, bfloat2float_lasx(__lsx_vld(w0, 0)), _sum0);
                _sum1 = __lasx_xvfmadd_s(_m, bfloat2float_lasx(__lsx_vld(w1, 0)), _sum1);
                _sum2 = __lasx_xvfmadd_s(_m, bfloat2float_lasx(__lsx_vld(w2, 0)), _sum2);
                _sum3 = __lasx_xvfmadd_s(_m, bfloat2float_lasx(__lsx_vld(w3, 0)), _sum3);
                _sum4 = __lasx_xvfmadd_s(_m, bfloat2float_lasx(__lsx_vld(w4, 0)), _sum4);
                _sum5 = __lasx_xvfmadd_s(_m, bfloat2float_lasx(__lsx_vld(w5, 0)), _sum5);
                _sum6 = __lasx_xvfmadd_s(_m, bfloat2float_lasx(__lsx_vld(w6, 0)), _sum6);
                _sum7 = __lasx_xvfmadd_s(_m, bfloat2float_lasx(__lsx_vld(w7, 0)), _sum7);

                m += 8;
                w0 += 8;
                w1 += 8;
                w2 += 8;
                w3 += 8;
                w4 += 8;
                w5 += 8;
                w6 += 8;
                w7 += 8;
            }
            for (; i < num_input; i++)
            {
                sums[0] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w0);
                sums[1] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w1);
                sums[2] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w2);
                sums[3] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w3);
                sums[4] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w4);
                sums[5] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w5);
                sums[6] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w6);
                sums[7] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w7);
                m++;
                w0++;
                w1++;
                w2++;
                w3++;
                w4++;
                w5++;
                w6++;
                w7++;
            }

            __m256 _sums = (__m256)__lasx_xvld(sums, 0);
            __m256 _hsums = HorizontalSums(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
            _sums = __lasx_xvfadd_s(_sums, _hsums);
            _sums = activation_lasx(_sums, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            __lsx_vst(float2bfloat_lasx(_sums), outptr + p, 0);
        }

        remain_outw_start += (nn_outw << 3);
        nn_outw = (outw - remain_outw_start) >> 2;
#else
        int remain_outw_start = 0;
        int nn_outw = outw >> 2;
#endif // __loongarch_asx

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outw; pp++)
        {
            int p = remain_outw_start + (pp * 4);

            float sums[4] = {0.0f};
            if (bias_data_ptr)
            {
                sums[0] = bias_data_ptr[p];
                sums[1] = bias_data_ptr[p + 1];
                sums[2] = bias_data_ptr[p + 2];
                sums[3] = bias_data_ptr[p + 3];
            }

            const unsigned short* w0 = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* w1 = weight_data_tm.row<const unsigned short>(p + 1);
            const unsigned short* w2 = weight_data_tm.row<const unsigned short>(p + 2);
            const unsigned short* w3 = weight_data_tm.row<const unsigned short>(p + 3);
            const unsigned short* m = bottom_blob;

            int i = 0;
#if __loongarch_asx
            __m256 _sum0a = (__m256)__lasx_xvreplgr2vr_w(0);
            __m256 _sum1a = _sum0a;
            __m256 _sum2a = _sum0a;
            __m256 _sum3a = _sum0a;
            for (; i + 7 < num_input; i += 8)
            {
                __m256 _m = bfloat2float_lasx(__lsx_vld(m, 0));

                _sum0a = __lasx_xvfmadd_s(_m, bfloat2float_lasx(__lsx_vld(w0, 0)), _sum0a);
                _sum1a = __lasx_xvfmadd_s(_m, bfloat2float_lasx(__lsx_vld(w1, 0)), _sum1a);
                _sum2a = __lasx_xvfmadd_s(_m, bfloat2float_lasx(__lsx_vld(w2, 0)), _sum2a);
                _sum3a = __lasx_xvfmadd_s(_m, bfloat2float_lasx(__lsx_vld(w3, 0)), _sum3a);

                m += 8;
                w0 += 8;
                w1 += 8;
                w2 += 8;
                w3 += 8;
            }
#endif // __loongarch_asx

            __m128 _sum0l = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum1l = _sum0l;
            __m128 _sum2l = _sum0l;
            __m128 _sum3l = _sum0l;
            for (; i + 3 < num_input; i += 4)
            {
                __m128 _m = bfloat2float_lsx(m);

                _sum0l = __lsx_vfmadd_s(_m, bfloat2float_lsx(w0), _sum0l);
                _sum1l = __lsx_vfmadd_s(_m, bfloat2float_lsx(w1), _sum1l);
                _sum2l = __lsx_vfmadd_s(_m, bfloat2float_lsx(w2), _sum2l);
                _sum3l = __lsx_vfmadd_s(_m, bfloat2float_lsx(w3), _sum3l);

                m += 4;
                w0 += 4;
                w1 += 4;
                w2 += 4;
                w3 += 4;
            }
            for (; i < num_input; i++)
            {
                sums[0] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w0);
                sums[1] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w1);
                sums[2] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w2);
                sums[3] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w3);

                m++;
                w0++;
                w1++;
                w2++;
                w3++;
            }

            __m128 _sums = (__m128)__lsx_vld(sums, 0);
#if __loongarch_asx
            _sums = __lsx_vfadd_s(HorizontalSums(_sum0a, _sum1a, _sum2a, _sum3a), _sums);
#endif
            transpose4x4_ps(_sum0l, _sum1l, _sum2l, _sum3l);
            _sums = __lsx_vfadd_s(_sum0l, _sums);
            _sums = __lsx_vfadd_s(_sum1l, _sums);
            _sums = __lsx_vfadd_s(_sum2l, _sums);
            _sums = __lsx_vfadd_s(_sum3l, _sums);
            _sums = activation_lsx(_sums, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            __lsx_vstelm_d(float2bfloat_lsx(_sums), outptr + p, 0, 0);
        }

        remain_outw_start += (nn_outw << 2);
#else
        int remain_outw_start = 0;
#endif // __loongarch_sx

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outw_start; p < outw; p++)
        {
            float sum = 0.f;

            if (bias_data_ptr)
                sum = bias_data_ptr[p];

            const unsigned short* w = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* m = bottom_blob;

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            __m256 _sum256 = (__m256)__lasx_xvreplgr2vr_w(0);
            for (; i + 7 < num_input; i += 8)
            {
                __m256 _m = bfloat2float_lasx(__lsx_vld(m, 0));
                __m256 _w = bfloat2float_lasx(__lsx_vld(w, 0));
                _sum256 = __lasx_xvfmadd_s(_m, _w, _sum256);

                m += 8;
                w += 8;
            }
#endif // __loongarch_asx
            __m128 _suml = (__m128)__lsx_vreplgr2vr_w(0);
            for (; i + 3 < num_input; i += 4)
            {
                __m128 _m = bfloat2float_lsx(m);
                __m128 _w = bfloat2float_lsx(w);
                _suml = __lsx_vfmadd_s(_m, _w, _suml);

                m += 4;
                w += 4;
            }
#endif // __loongarch_sx
            for (; i < num_input; i++)
            {
                sum += bfloat16_to_float32(*m) * bfloat16_to_float32(*w);
                m++;
                w++;
            }

#if __loongarch_sx
#if __loongarch_asx
            __m128 _lo = __lasx_extract_128_lo_s(_sum256);
            __m128 _hi = __lasx_extract_128_hi_s(_sum256);
            _suml = __lsx_vfadd_s(_suml, _lo);
            _suml = __lsx_vfadd_s(_suml, _hi);
#endif // __loongarch_asx
            sum += __lsx_reduce_fadd_s(_suml);
#endif // __loongarch_sx

            sum = activation_ss(sum, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            outptr[p] = float32_to_bfloat16(sum);
        }
    } // out_elempack == 1
}
