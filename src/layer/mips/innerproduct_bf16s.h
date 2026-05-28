// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void innerproduct_transform_kernel_bf16s_msa(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt)
{
    int out_elempack = 1;
#if __mips_msa
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    }
#endif

    // src = inch-outch
    // dst = pb-inch-outch/pb
#if __mips_msa
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
            for (; p + 3 < num_input; p += 4)
            {
                __builtin_prefetch(k0 + 16);
                __builtin_prefetch(k1 + 16);
                __builtin_prefetch(k2 + 16);
                __builtin_prefetch(k3 + 16);
                __builtin_prefetch(k4 + 16);
                __builtin_prefetch(k5 + 16);
                __builtin_prefetch(k6 + 16);
                __builtin_prefetch(k7 + 16);
                v4f32 _r0 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _r1 = (v4f32)__msa_ld_w(k1, 0);
                v4f32 _r2 = (v4f32)__msa_ld_w(k2, 0);
                v4f32 _r3 = (v4f32)__msa_ld_w(k3, 0);
                v4f32 _r4 = (v4f32)__msa_ld_w(k4, 0);
                v4f32 _r5 = (v4f32)__msa_ld_w(k5, 0);
                v4f32 _r6 = (v4f32)__msa_ld_w(k6, 0);
                v4f32 _r7 = (v4f32)__msa_ld_w(k7, 0);

                // transpose 4x4
                v4i32 _t0 = __msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                v4i32 _t1 = __msa_ilvr_w((v4i32)_r3, (v4i32)_r2);
                v4i32 _t2 = __msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
                v4i32 _t3 = __msa_ilvl_w((v4i32)_r3, (v4i32)_r2);
                _r0 = (v4f32)__msa_ilvr_d((v2i64)_t1, (v2i64)_t0);
                _r1 = (v4f32)__msa_ilvl_d((v2i64)_t1, (v2i64)_t0);
                _r2 = (v4f32)__msa_ilvr_d((v2i64)_t3, (v2i64)_t2);
                _r3 = (v4f32)__msa_ilvl_d((v2i64)_t3, (v2i64)_t2);

                _t0 = __msa_ilvr_w((v4i32)_r5, (v4i32)_r4);
                _t1 = __msa_ilvr_w((v4i32)_r7, (v4i32)_r6);
                _t2 = __msa_ilvl_w((v4i32)_r5, (v4i32)_r4);
                _t3 = __msa_ilvl_w((v4i32)_r7, (v4i32)_r6);
                _r4 = (v4f32)__msa_ilvr_d((v2i64)_t1, (v2i64)_t0);
                _r5 = (v4f32)__msa_ilvl_d((v2i64)_t1, (v2i64)_t0);
                _r6 = (v4f32)__msa_ilvr_d((v2i64)_t3, (v2i64)_t2);
                _r7 = (v4f32)__msa_ilvl_d((v2i64)_t3, (v2i64)_t2);

                __msa_st_w(float2bfloat_msa(_r0, _r4), g0, 0);
                __msa_st_w(float2bfloat_msa(_r1, _r5), g0 + 8, 0);
                __msa_st_w(float2bfloat_msa(_r2, _r6), g0 + 16, 0);
                __msa_st_w(float2bfloat_msa(_r3, _r7), g0 + 24, 0);

                k0 += 4;
                k1 += 4;
                k2 += 4;
                k3 += 4;
                k4 += 4;
                k5 += 4;
                k6 += 4;
                k7 += 4;
                g0 += 32;
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
                __builtin_prefetch(k0 + 16);
                __builtin_prefetch(k1 + 16);
                __builtin_prefetch(k2 + 16);
                __builtin_prefetch(k3 + 16);
                v4f32 _r0 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _r1 = (v4f32)__msa_ld_w(k1, 0);
                v4f32 _r2 = (v4f32)__msa_ld_w(k2, 0);
                v4f32 _r3 = (v4f32)__msa_ld_w(k3, 0);

                // transpose 4x4
                v4i32 _t0 = __msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                v4i32 _t1 = __msa_ilvr_w((v4i32)_r3, (v4i32)_r2);
                v4i32 _t2 = __msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
                v4i32 _t3 = __msa_ilvl_w((v4i32)_r3, (v4i32)_r2);
                _r0 = (v4f32)__msa_ilvr_d((v2i64)_t1, (v2i64)_t0);
                _r1 = (v4f32)__msa_ilvl_d((v2i64)_t1, (v2i64)_t0);
                _r2 = (v4f32)__msa_ilvr_d((v2i64)_t3, (v2i64)_t2);
                _r3 = (v4f32)__msa_ilvl_d((v2i64)_t3, (v2i64)_t2);

                v4i32 _bf16_01 = float2bfloat_msa(_r0, _r1);
                v4i32 _bf16_23 = float2bfloat_msa(_r2, _r3);
                __msa_st_w(_bf16_01, g0, 0);
                __msa_st_w(_bf16_23, g0 + 8, 0);

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
#endif // __mips_msa

    if (out_elempack == 1)
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);
        ncnn::cast_float32_to_bfloat16(weight_data_r2, weight_data_tm, opt);
    }
}

static void innerproduct_gemm_bf16s_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int num_input = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int num_output = top_blob.w;
    const int h = bottom_blob.h;

    const float* bias_data_ptr = bias_data;

    int num_output_elempack = 1;
#if __mips_msa
    if (opt.use_packing_layout)
    {
        num_output_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    }
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int j = 0; j < h; j++)
    {
#if __mips_msa
        if (elempack == 8 && num_output_elempack == 8)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                v4f32 _sum1 = _sum0;
                v4f32 _sum2 = _sum0;
                v4f32 _sum3 = _sum0;
                v4f32 _sum4 = _sum0;
                v4f32 _sum5 = _sum0;
                v4f32 _sum6 = _sum0;
                v4f32 _sum7 = _sum0;
                v4f32 _sum8 = _sum0;
                v4f32 _sum9 = _sum0;
                v4f32 _suma = _sum0;
                v4f32 _sumb = _sum0;
                v4f32 _sumc = _sum0;
                v4f32 _sumd = _sum0;
                v4f32 _sume = _sum0;
                v4f32 _sumf = _sum0;

                if (bias_data_ptr)
                {
                    _sum0 = (v4f32)__msa_ld_w(bias_data_ptr + p * 8, 0);
                    _sum8 = (v4f32)__msa_ld_w(bias_data_ptr + p * 8 + 4, 0);
                }
                _sum1 = _sum0;
                _sum2 = _sum0;
                _sum3 = _sum0;
                _sum4 = _sum0;
                _sum5 = _sum0;
                _sum6 = _sum0;
                _sum7 = _sum0;
                _sum9 = _sum8;
                _suma = _sum8;
                _sumb = _sum8;
                _sumc = _sum8;
                _sumd = _sum8;
                _sume = _sum8;
                _sumf = _sum8;

                v8i16 _zero_bf16 = __msa_fill_h(0);
                for (int i = 0; i < num_input; i++)
                {
                    __builtin_prefetch(m + 16);
                    __builtin_prefetch(kptr + 16);

                    v8i16 _m01 = __msa_ld_h(m, 0);
                    v4f32 _m0 = (v4f32)__msa_ilvr_h(_m01, _zero_bf16);
                    v4f32 _m1 = (v4f32)__msa_ilvl_h(_m01, _zero_bf16);
                    v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                    v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                    v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);

                    v4f32 _val0 = (v4f32)__msa_splati_w((v4i32)_m0, 0);
                    v4f32 _val1 = (v4f32)__msa_splati_w((v4i32)_m0, 1);
                    v4f32 _val2 = (v4f32)__msa_splati_w((v4i32)_m0, 2);
                    v4f32 _val3 = (v4f32)__msa_splati_w((v4i32)_m0, 3);
                    v4f32 _val4 = (v4f32)__msa_splati_w((v4i32)_m1, 0);
                    v4f32 _val5 = (v4f32)__msa_splati_w((v4i32)_m1, 1);
                    v4f32 _val6 = (v4f32)__msa_splati_w((v4i32)_m1, 2);
                    v4f32 _val7 = (v4f32)__msa_splati_w((v4i32)_m1, 3);

                    _sum0 = __ncnn_msa_fmadd_w(_sum0, _val0, _w0);
                    _sum1 = __ncnn_msa_fmadd_w(_sum1, _val1, _w0);
                    _sum2 = __ncnn_msa_fmadd_w(_sum2, _val2, _w0);
                    _sum3 = __ncnn_msa_fmadd_w(_sum3, _val3, _w0);
                    _sum4 = __ncnn_msa_fmadd_w(_sum4, _val4, _w0);
                    _sum5 = __ncnn_msa_fmadd_w(_sum5, _val5, _w0);
                    _sum6 = __ncnn_msa_fmadd_w(_sum6, _val6, _w0);
                    _sum7 = __ncnn_msa_fmadd_w(_sum7, _val7, _w0);
                    _sum8 = __ncnn_msa_fmadd_w(_sum8, _val0, _w1);
                    _sum9 = __ncnn_msa_fmadd_w(_sum9, _val1, _w1);
                    _suma = __ncnn_msa_fmadd_w(_suma, _val2, _w1);
                    _sumb = __ncnn_msa_fmadd_w(_sumb, _val3, _w1);
                    _sumc = __ncnn_msa_fmadd_w(_sumc, _val4, _w1);
                    _sumd = __ncnn_msa_fmadd_w(_sumd, _val5, _w1);
                    _sume = __ncnn_msa_fmadd_w(_sume, _val6, _w1);
                    _sumf = __ncnn_msa_fmadd_w(_sumf, _val7, _w1);

                    m += 8;
                    kptr += 8;
                }

                _sum0 = activation_msa(_sum0, activation_type, activation_params);
                _sum1 = activation_msa(_sum1, activation_type, activation_params);
                _sum2 = activation_msa(_sum2, activation_type, activation_params);
                _sum3 = activation_msa(_sum3, activation_type, activation_params);
                _sum4 = activation_msa(_sum4, activation_type, activation_params);
                _sum5 = activation_msa(_sum5, activation_type, activation_params);
                _sum6 = activation_msa(_sum6, activation_type, activation_params);
                _sum7 = activation_msa(_sum7, activation_type, activation_params);
                _sum8 = activation_msa(_sum8, activation_type, activation_params);
                _sum9 = activation_msa(_sum9, activation_type, activation_params);
                _suma = activation_msa(_suma, activation_type, activation_params);
                _sumb = activation_msa(_sumb, activation_type, activation_params);
                _sumc = activation_msa(_sumc, activation_type, activation_params);
                _sumd = activation_msa(_sumd, activation_type, activation_params);
                _sume = activation_msa(_sume, activation_type, activation_params);
                _sumf = activation_msa(_sumf, activation_type, activation_params);

                // transpose 4x4
                v4i32 _t0 = __msa_ilvr_w((v4i32)_sum1, (v4i32)_sum0);
                v4i32 _t1 = __msa_ilvr_w((v4i32)_sum3, (v4i32)_sum2);
                v4i32 _t2 = __msa_ilvl_w((v4i32)_sum1, (v4i32)_sum0);
                v4i32 _t3 = __msa_ilvl_w((v4i32)_sum3, (v4i32)_sum2);
                _sum0 = (v4f32)__msa_ilvr_d((v2i64)_t1, (v2i64)_t0);
                _sum1 = (v4f32)__msa_ilvl_d((v2i64)_t1, (v2i64)_t0);
                _sum2 = (v4f32)__msa_ilvr_d((v2i64)_t3, (v2i64)_t2);
                _sum3 = (v4f32)__msa_ilvl_d((v2i64)_t3, (v2i64)_t2);

                _t0 = __msa_ilvr_w((v4i32)_sum5, (v4i32)_sum4);
                _t1 = __msa_ilvr_w((v4i32)_sum7, (v4i32)_sum6);
                _t2 = __msa_ilvl_w((v4i32)_sum5, (v4i32)_sum4);
                _t3 = __msa_ilvl_w((v4i32)_sum7, (v4i32)_sum6);
                _sum4 = (v4f32)__msa_ilvr_d((v2i64)_t1, (v2i64)_t0);
                _sum5 = (v4f32)__msa_ilvl_d((v2i64)_t1, (v2i64)_t0);
                _sum6 = (v4f32)__msa_ilvr_d((v2i64)_t3, (v2i64)_t2);
                _sum7 = (v4f32)__msa_ilvl_d((v2i64)_t3, (v2i64)_t2);

                _t0 = __msa_ilvr_w((v4i32)_sum9, (v4i32)_sum8);
                _t1 = __msa_ilvr_w((v4i32)_sumb, (v4i32)_suma);
                _t2 = __msa_ilvl_w((v4i32)_sum9, (v4i32)_sum8);
                _t3 = __msa_ilvl_w((v4i32)_sumb, (v4i32)_suma);
                _sum8 = (v4f32)__msa_ilvr_d((v2i64)_t1, (v2i64)_t0);
                _sum9 = (v4f32)__msa_ilvl_d((v2i64)_t1, (v2i64)_t0);
                _suma = (v4f32)__msa_ilvr_d((v2i64)_t3, (v2i64)_t2);
                _sumb = (v4f32)__msa_ilvl_d((v2i64)_t3, (v2i64)_t2);

                _t0 = __msa_ilvr_w((v4i32)_sumd, (v4i32)_sumc);
                _t1 = __msa_ilvr_w((v4i32)_sumf, (v4i32)_sume);
                _t2 = __msa_ilvl_w((v4i32)_sumd, (v4i32)_sumc);
                _t3 = __msa_ilvl_w((v4i32)_sumf, (v4i32)_sume);
                _sumc = (v4f32)__msa_ilvr_d((v2i64)_t1, (v2i64)_t0);
                _sumd = (v4f32)__msa_ilvl_d((v2i64)_t1, (v2i64)_t0);
                _sume = (v4f32)__msa_ilvr_d((v2i64)_t3, (v2i64)_t2);
                _sumf = (v4f32)__msa_ilvl_d((v2i64)_t3, (v2i64)_t2);

                __msa_st_w(float2bfloat_msa(_sum0, _sum4), outptr, 0);
                __msa_st_w(float2bfloat_msa(_sum1, _sum5), outptr + 8, 0);
                __msa_st_w(float2bfloat_msa(_sum2, _sum6), outptr + 16, 0);
                __msa_st_w(float2bfloat_msa(_sum3, _sum7), outptr + 24, 0);
                __msa_st_w(float2bfloat_msa(_sum8, _sumc), outptr + 32, 0);
                __msa_st_w(float2bfloat_msa(_sum9, _sumd), outptr + 40, 0);
                __msa_st_w(float2bfloat_msa(_suma, _sume), outptr + 48, 0);
                __msa_st_w(float2bfloat_msa(_sumb, _sumf), outptr + 56, 0);
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

                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                v4f32 _sum1 = _sum0;
                v4f32 _sum2 = _sum0;
                v4f32 _sum3 = _sum0;
                v4f32 _sum4 = _sum0;
                v4f32 _sum5 = _sum0;
                v4f32 _sum6 = _sum0;
                v4f32 _sum7 = _sum0;

                if (bias_data_ptr)
                {
                    _sum0 = (v4f32)__msa_ld_w(bias_data_ptr + p * 8, 0);
                    _sum4 = (v4f32)__msa_ld_w(bias_data_ptr + p * 8 + 4, 0);
                }

                int i = 0;
                v8i16 _zero_bf16 = __msa_fill_h(0);
                for (; i + 3 < num_input; i += 4)
                {
                    __builtin_prefetch(m + 16);
                    __builtin_prefetch(kptr + 64);

                    v4f32 _m0 = bfloat2float_msa(m);
                    v4f32 _val0 = (v4f32)__msa_splati_w((v4i32)_m0, 0);
                    v4f32 _val1 = (v4f32)__msa_splati_w((v4i32)_m0, 1);
                    v4f32 _val2 = (v4f32)__msa_splati_w((v4i32)_m0, 2);
                    v4f32 _val3 = (v4f32)__msa_splati_w((v4i32)_m0, 3);

                    v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                    v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                    v4f32 _w4 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                    v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                    v4f32 _w1 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                    v4f32 _w5 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                    v8i16 _w45_bf16 = __msa_ld_h(kptr + 16, 0);
                    v4f32 _w2 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                    v4f32 _w6 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                    v8i16 _w67_bf16 = __msa_ld_h(kptr + 24, 0);
                    v4f32 _w3 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                    v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);

                    _sum0 = __ncnn_msa_fmadd_w(_sum0, _val0, _w0);
                    _sum1 = __ncnn_msa_fmadd_w(_sum1, _val1, _w1);
                    _sum2 = __ncnn_msa_fmadd_w(_sum2, _val2, _w2);
                    _sum3 = __ncnn_msa_fmadd_w(_sum3, _val3, _w3);
                    _sum4 = __ncnn_msa_fmadd_w(_sum4, _val0, _w4);
                    _sum5 = __ncnn_msa_fmadd_w(_sum5, _val1, _w5);
                    _sum6 = __ncnn_msa_fmadd_w(_sum6, _val2, _w6);
                    _sum7 = __ncnn_msa_fmadd_w(_sum7, _val3, _w7);

                    m += 4;
                    kptr += 32;
                }
                for (; i < num_input; i++)
                {
                    v4f32 _val = __msa_fill_w_f32(bfloat16_to_float32(m[0]));
                    v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                    v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                    v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                    _sum0 = __ncnn_msa_fmadd_w(_sum0, _val, _w0);
                    _sum4 = __ncnn_msa_fmadd_w(_sum4, _val, _w1);

                    m += 1;
                    kptr += 8;
                }

                _sum0 = __msa_fadd_w(_sum0, _sum1);
                _sum2 = __msa_fadd_w(_sum2, _sum3);
                _sum4 = __msa_fadd_w(_sum4, _sum5);
                _sum6 = __msa_fadd_w(_sum6, _sum7);
                _sum0 = __msa_fadd_w(_sum0, _sum2);
                _sum4 = __msa_fadd_w(_sum4, _sum6);

                _sum0 = activation_msa(_sum0, activation_type, activation_params);
                _sum4 = activation_msa(_sum4, activation_type, activation_params);

                __msa_st_w(float2bfloat_msa(_sum0, _sum4), outptr, 0);
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

                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                v4f32 _sum1 = _sum0;
                v4f32 _sum2 = _sum0;
                v4f32 _sum3 = _sum0;
                v4f32 _sum4 = _sum0;
                v4f32 _sum5 = _sum0;
                v4f32 _sum6 = _sum0;
                v4f32 _sum7 = _sum0;

                if (bias_data_ptr)
                {
                    _sum0 = (v4f32)__msa_ld_w(bias_data_ptr + p * 8, 0);
                    _sum4 = (v4f32)__msa_ld_w(bias_data_ptr + p * 8 + 4, 0);
                }
                _sum1 = _sum0;
                _sum2 = _sum0;
                _sum3 = _sum0;
                _sum5 = _sum4;
                _sum6 = _sum4;
                _sum7 = _sum4;

                int i = 0;
                for (; i < num_input; i++)
                {
                    __builtin_prefetch(m + 16);
                    __builtin_prefetch(kptr + 16);

                    v4f32 _m0 = bfloat2float_msa(m);
                    v4f32 _val0 = (v4f32)__msa_splati_w((v4i32)_m0, 0);
                    v4f32 _val1 = (v4f32)__msa_splati_w((v4i32)_m0, 1);
                    v4f32 _val2 = (v4f32)__msa_splati_w((v4i32)_m0, 2);
                    v4f32 _val3 = (v4f32)__msa_splati_w((v4i32)_m0, 3);
                    v8i16 _zero_bf16 = __msa_fill_h(0);
                    v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                    v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                    v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);

                    _sum0 = __ncnn_msa_fmadd_w(_sum0, _val0, _w0);
                    _sum1 = __ncnn_msa_fmadd_w(_sum1, _val1, _w0);
                    _sum2 = __ncnn_msa_fmadd_w(_sum2, _val2, _w0);
                    _sum3 = __ncnn_msa_fmadd_w(_sum3, _val3, _w0);
                    _sum4 = __ncnn_msa_fmadd_w(_sum4, _val0, _w1);
                    _sum5 = __ncnn_msa_fmadd_w(_sum5, _val1, _w1);
                    _sum6 = __ncnn_msa_fmadd_w(_sum6, _val2, _w1);
                    _sum7 = __ncnn_msa_fmadd_w(_sum7, _val3, _w1);

                    m += 4;
                    kptr += 8;
                }

                _sum0 = activation_msa(_sum0, activation_type, activation_params);
                _sum1 = activation_msa(_sum1, activation_type, activation_params);
                _sum2 = activation_msa(_sum2, activation_type, activation_params);
                _sum3 = activation_msa(_sum3, activation_type, activation_params);
                _sum4 = activation_msa(_sum4, activation_type, activation_params);
                _sum5 = activation_msa(_sum5, activation_type, activation_params);
                _sum6 = activation_msa(_sum6, activation_type, activation_params);
                _sum7 = activation_msa(_sum7, activation_type, activation_params);

                // transpose 4x4
                v4i32 _t0 = __msa_ilvr_w((v4i32)_sum1, (v4i32)_sum0);
                v4i32 _t1 = __msa_ilvr_w((v4i32)_sum3, (v4i32)_sum2);
                v4i32 _t2 = __msa_ilvl_w((v4i32)_sum1, (v4i32)_sum0);
                v4i32 _t3 = __msa_ilvl_w((v4i32)_sum3, (v4i32)_sum2);
                _sum0 = (v4f32)__msa_ilvr_d((v2i64)_t1, (v2i64)_t0);
                _sum1 = (v4f32)__msa_ilvl_d((v2i64)_t1, (v2i64)_t0);
                _sum2 = (v4f32)__msa_ilvr_d((v2i64)_t3, (v2i64)_t2);
                _sum3 = (v4f32)__msa_ilvl_d((v2i64)_t3, (v2i64)_t2);

                _t0 = __msa_ilvr_w((v4i32)_sum5, (v4i32)_sum4);
                _t1 = __msa_ilvr_w((v4i32)_sum7, (v4i32)_sum6);
                _t2 = __msa_ilvl_w((v4i32)_sum5, (v4i32)_sum4);
                _t3 = __msa_ilvl_w((v4i32)_sum7, (v4i32)_sum6);
                _sum4 = (v4f32)__msa_ilvr_d((v2i64)_t1, (v2i64)_t0);
                _sum5 = (v4f32)__msa_ilvl_d((v2i64)_t1, (v2i64)_t0);
                _sum6 = (v4f32)__msa_ilvr_d((v2i64)_t3, (v2i64)_t2);
                _sum7 = (v4f32)__msa_ilvl_d((v2i64)_t3, (v2i64)_t2);

                __msa_st_w(float2bfloat_msa(_sum0, _sum1), outptr, 0);
                __msa_st_w(float2bfloat_msa(_sum2, _sum3), outptr + 8, 0);
                __msa_st_w(float2bfloat_msa(_sum4, _sum5), outptr + 16, 0);
                __msa_st_w(float2bfloat_msa(_sum6, _sum7), outptr + 24, 0);
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

                v4f32 _sum0 = (v4f32)__msa_fill_w(0);

                if (bias_data_ptr)
                {
                    _sum0 = (v4f32)__msa_ld_w(bias_data_ptr + p * 4, 0);
                }

                v4f32 _sum1 = _sum0;
                v4f32 _sum2 = _sum0;
                v4f32 _sum3 = _sum0;
                v4f32 _sum4 = _sum0;
                v4f32 _sum5 = _sum0;
                v4f32 _sum6 = _sum0;
                v4f32 _sum7 = _sum0;

                v8i16 _zero_bf16 = __msa_fill_h(0);
                for (int i = 0; i < num_input; i++)
                {
                    __builtin_prefetch(m + 16);
                    __builtin_prefetch(kptr + 16);

                    v8i16 _m01 = __msa_ld_h(m, 0);
                    v4f32 _m0 = (v4f32)__msa_ilvr_h(_m01, _zero_bf16);
                    v4f32 _m1 = (v4f32)__msa_ilvl_h(_m01, _zero_bf16);
                    v4f32 _w = bfloat2float_msa(kptr);

                    v4f32 _val0 = (v4f32)__msa_splati_w((v4i32)_m0, 0);
                    v4f32 _val1 = (v4f32)__msa_splati_w((v4i32)_m0, 1);
                    v4f32 _val2 = (v4f32)__msa_splati_w((v4i32)_m0, 2);
                    v4f32 _val3 = (v4f32)__msa_splati_w((v4i32)_m0, 3);
                    v4f32 _val4 = (v4f32)__msa_splati_w((v4i32)_m1, 0);
                    v4f32 _val5 = (v4f32)__msa_splati_w((v4i32)_m1, 1);
                    v4f32 _val6 = (v4f32)__msa_splati_w((v4i32)_m1, 2);
                    v4f32 _val7 = (v4f32)__msa_splati_w((v4i32)_m1, 3);

                    _sum0 = __ncnn_msa_fmadd_w(_sum0, _val0, _w);
                    _sum1 = __ncnn_msa_fmadd_w(_sum1, _val1, _w);
                    _sum2 = __ncnn_msa_fmadd_w(_sum2, _val2, _w);
                    _sum3 = __ncnn_msa_fmadd_w(_sum3, _val3, _w);
                    _sum4 = __ncnn_msa_fmadd_w(_sum4, _val4, _w);
                    _sum5 = __ncnn_msa_fmadd_w(_sum5, _val5, _w);
                    _sum6 = __ncnn_msa_fmadd_w(_sum6, _val6, _w);
                    _sum7 = __ncnn_msa_fmadd_w(_sum7, _val7, _w);

                    m += 8;
                    kptr += 4;
                }

                _sum0 = activation_msa(_sum0, activation_type, activation_params);
                _sum1 = activation_msa(_sum1, activation_type, activation_params);
                _sum2 = activation_msa(_sum2, activation_type, activation_params);
                _sum3 = activation_msa(_sum3, activation_type, activation_params);
                _sum4 = activation_msa(_sum4, activation_type, activation_params);
                _sum5 = activation_msa(_sum5, activation_type, activation_params);
                _sum6 = activation_msa(_sum6, activation_type, activation_params);
                _sum7 = activation_msa(_sum7, activation_type, activation_params);

                transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);
                transpose4x4_ps(_sum4, _sum5, _sum6, _sum7);

                __msa_st_w(float2bfloat_msa(_sum0, _sum4), outptr, 0);
                __msa_st_w(float2bfloat_msa(_sum1, _sum5), outptr + 8, 0);
                __msa_st_w(float2bfloat_msa(_sum2, _sum6), outptr + 16, 0);
                __msa_st_w(float2bfloat_msa(_sum3, _sum7), outptr + 24, 0);
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

                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                if (bias_data_ptr)
                {
                    _sum0 = __msa_fill_w_f32(bias_data_ptr[p]);
                }
                v4f32 _sum1 = _sum0;

                v8i16 _zero_bf16 = __msa_fill_h(0);
                for (int i = 0; i < num_input; i++)
                {
                    __builtin_prefetch(m + 16);
                    __builtin_prefetch(kptr + 16);

                    v8i16 _m01 = __msa_ld_h(m, 0);
                    v4f32 _m0 = (v4f32)__msa_ilvr_h(_m01, _zero_bf16);
                    v4f32 _m1 = (v4f32)__msa_ilvl_h(_m01, _zero_bf16);
                    v4f32 _w = __msa_fill_w_f32(bfloat16_to_float32(kptr[0]));
                    _sum0 = __ncnn_msa_fmadd_w(_sum0, _m0, _w);
                    _sum1 = __ncnn_msa_fmadd_w(_sum1, _m1, _w);

                    m += 8;
                    kptr += 1;
                }

                _sum0 = activation_msa(_sum0, activation_type, activation_params);
                _sum1 = activation_msa(_sum1, activation_type, activation_params);

                __msa_st_w(float2bfloat_msa(_sum0, _sum1), outptr, 0);
                outptr += 8;
            }
        }

        if (elempack == 4 && num_output_elempack == 4)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                v4f32 _sum0 = (v4f32)__msa_fill_w(0);

                if (bias_data_ptr)
                {
                    _sum0 = (v4f32)__msa_ld_w(bias_data_ptr + p * 4, 0);
                }

                v4f32 _sum1 = _sum0;
                v4f32 _sum2 = _sum0;
                v4f32 _sum3 = _sum0;

                int i = 0;
                for (; i < num_input; i++)
                {
                    __builtin_prefetch(m + 16);
                    __builtin_prefetch(kptr + 16);
                    v4f32 _m0 = bfloat2float_msa(m);
                    v4f32 _val0 = (v4f32)__msa_splati_w((v4i32)_m0, 0);
                    v4f32 _val1 = (v4f32)__msa_splati_w((v4i32)_m0, 1);
                    v4f32 _val2 = (v4f32)__msa_splati_w((v4i32)_m0, 2);
                    v4f32 _val3 = (v4f32)__msa_splati_w((v4i32)_m0, 3);
                    v4f32 _w = bfloat2float_msa(kptr);
                    _sum0 = __ncnn_msa_fmadd_w(_sum0, _val0, _w);
                    _sum1 = __ncnn_msa_fmadd_w(_sum1, _val1, _w);
                    _sum2 = __ncnn_msa_fmadd_w(_sum2, _val2, _w);
                    _sum3 = __ncnn_msa_fmadd_w(_sum3, _val3, _w);

                    m += 4;
                    kptr += 4;
                }

                _sum0 = activation_msa(_sum0, activation_type, activation_params);
                _sum1 = activation_msa(_sum1, activation_type, activation_params);
                _sum2 = activation_msa(_sum2, activation_type, activation_params);
                _sum3 = activation_msa(_sum3, activation_type, activation_params);

                // transpose 4x4
                v4i32 _t0 = __msa_ilvr_w((v4i32)_sum1, (v4i32)_sum0);
                v4i32 _t1 = __msa_ilvr_w((v4i32)_sum3, (v4i32)_sum2);
                v4i32 _t2 = __msa_ilvl_w((v4i32)_sum1, (v4i32)_sum0);
                v4i32 _t3 = __msa_ilvl_w((v4i32)_sum3, (v4i32)_sum2);
                _sum0 = (v4f32)__msa_ilvr_d((v2i64)_t1, (v2i64)_t0);
                _sum1 = (v4f32)__msa_ilvl_d((v2i64)_t1, (v2i64)_t0);
                _sum2 = (v4f32)__msa_ilvr_d((v2i64)_t3, (v2i64)_t2);
                _sum3 = (v4f32)__msa_ilvl_d((v2i64)_t3, (v2i64)_t2);

                v4i32 _bf16_01 = float2bfloat_msa(_sum0, _sum1);
                v4i32 _bf16_23 = float2bfloat_msa(_sum2, _sum3);
                __msa_st_w(_bf16_01, outptr, 0);
                __msa_st_w(_bf16_23, outptr + 8, 0);
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

                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                v4f32 _sum1 = _sum0;
                v4f32 _sum2 = _sum0;
                v4f32 _sum3 = _sum0;

                if (bias_data_ptr)
                {
                    _sum0 = (v4f32)__msa_ld_w(bias_data_ptr + p * 4, 0);
                }

                int i = 0;
                for (; i + 3 < num_input; i += 4)
                {
                    __builtin_prefetch(m + 16);
                    __builtin_prefetch(kptr + 32);

                    v4f32 _m0 = bfloat2float_msa(m);
                    v4f32 _val0 = (v4f32)__msa_splati_w((v4i32)_m0, 0);
                    v4f32 _val1 = (v4f32)__msa_splati_w((v4i32)_m0, 1);
                    v4f32 _val2 = (v4f32)__msa_splati_w((v4i32)_m0, 2);
                    v4f32 _val3 = (v4f32)__msa_splati_w((v4i32)_m0, 3);

                    v4f32 _w0 = bfloat2float_msa(kptr);
                    v4f32 _w1 = bfloat2float_msa(kptr + 4);
                    v4f32 _w2 = bfloat2float_msa(kptr + 8);
                    v4f32 _w3 = bfloat2float_msa(kptr + 12);

                    _sum0 = __ncnn_msa_fmadd_w(_sum0, _val0, _w0);
                    _sum1 = __ncnn_msa_fmadd_w(_sum1, _val1, _w1);
                    _sum2 = __ncnn_msa_fmadd_w(_sum2, _val2, _w2);
                    _sum3 = __ncnn_msa_fmadd_w(_sum3, _val3, _w3);

                    m += 4;
                    kptr += 16;
                }
                for (; i < num_input; i++)
                {
                    v4f32 _val = __msa_fill_w_f32(bfloat16_to_float32(m[0]));
                    v4f32 _w = bfloat2float_msa(kptr);
                    _sum0 = __ncnn_msa_fmadd_w(_sum0, _val, _w);

                    m += 1;
                    kptr += 4;
                }

                _sum0 = __msa_fadd_w(_sum0, _sum1);
                _sum2 = __msa_fadd_w(_sum2, _sum3);
                _sum0 = __msa_fadd_w(_sum0, _sum2);

                _sum0 = activation_msa(_sum0, activation_type, activation_params);

                __msa_storel_d(float2bfloat_msa(_sum0), outptr);
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

                v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                v4f32 _sum1 = (v4f32)__msa_fill_w(0);
                v4f32 _sum2 = (v4f32)__msa_fill_w(0);
                v4f32 _sum3 = (v4f32)__msa_fill_w(0);

                if (bias_data_ptr)
                {
                    _sum0 = __msa_fill_w_f32(bias_data_ptr[p]);
                }

                int i = 0;
                for (; i + 3 < num_input; i += 4)
                {
                    __builtin_prefetch(m + 16);
                    __builtin_prefetch(kptr + 16);
                    v8i16 _zero_bf16 = __msa_fill_h(0);
                    v8i16 _val01_bf16 = __msa_ld_h(m, 0);
                    v4f32 _val0 = (v4f32)__msa_ilvr_h(_val01_bf16, _zero_bf16);
                    v4f32 _val1 = (v4f32)__msa_ilvl_h(_val01_bf16, _zero_bf16);
                    v8i16 _val23_bf16 = __msa_ld_h(m + 8, 0);
                    v4f32 _val2 = (v4f32)__msa_ilvr_h(_val23_bf16, _zero_bf16);
                    v4f32 _val3 = (v4f32)__msa_ilvl_h(_val23_bf16, _zero_bf16);

                    v4f32 _w = bfloat2float_msa(kptr);

                    v4f32 _w0 = (v4f32)__msa_splati_w((v4i32)_w, 0);
                    v4f32 _w1 = (v4f32)__msa_splati_w((v4i32)_w, 1);
                    v4f32 _w2 = (v4f32)__msa_splati_w((v4i32)_w, 2);
                    v4f32 _w3 = (v4f32)__msa_splati_w((v4i32)_w, 3);

                    _sum0 = __ncnn_msa_fmadd_w(_sum0, _val0, _w0);
                    _sum1 = __ncnn_msa_fmadd_w(_sum1, _val1, _w1);
                    _sum2 = __ncnn_msa_fmadd_w(_sum2, _val2, _w2);
                    _sum3 = __ncnn_msa_fmadd_w(_sum3, _val3, _w3);

                    m += 16;
                    kptr += 4;
                }
                for (; i < num_input; i++)
                {
                    v4f32 _val = bfloat2float_msa(m);
                    v4f32 _w = __msa_fill_w_f32(bfloat16_to_float32(kptr[0]));
                    _sum0 = __ncnn_msa_fmadd_w(_sum0, _val, _w);

                    m += 4;
                    kptr += 1;
                }

                _sum0 = __msa_fadd_w(_sum0, _sum1);
                _sum2 = __msa_fadd_w(_sum2, _sum3);
                _sum0 = __msa_fadd_w(_sum0, _sum2);

                _sum0 = activation_msa(_sum0, activation_type, activation_params);

                *(int64_t*)outptr = __msa_copy_s_d((v2i64)float2bfloat_msa(_sum0), 0);
                outptr += 4;
            }
        }
#endif // __mips_msa

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
#if __mips_msa
                v4f32 _sum = (v4f32)__msa_fill_w(0);
                for (; i + 3 < num_input; i += 4)
                {
                    __builtin_prefetch(m + 16);
                    __builtin_prefetch(kptr + 16);
                    v4f32 _val = bfloat2float_msa(m);
                    v4f32 _w = bfloat2float_msa(kptr);
                    _sum = __ncnn_msa_fmadd_w(_sum, _val, _w);

                    m += 4;
                    kptr += 4;
                }
#endif // __mips_msa
                for (; i < num_input; i++)
                {
                    sum += bfloat16_to_float32(*m++) * bfloat16_to_float32(*kptr++);
                }

#if __mips_msa
                sum += __msa_reduce_fadd_w(_sum);
#endif // __mips_msa

                sum = activation_ss(sum, activation_type, activation_params);

                outptr[0] = float32_to_bfloat16(sum);
                outptr += 1;
            }
        }
    }
}

static void innerproduct_bf16s_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int num_input = bottom_blob.w * bottom_blob.elempack;
    const int outw = top_blob.w;
    const int out_elempack = top_blob.elempack;

    const float* bias_data_ptr = bias_data;

#if __mips_msa
    if (out_elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outw; p++)
        {
            v4f32 _sum0 = (v4f32)__msa_fill_w(0);
            v4f32 _sum1 = (v4f32)__msa_fill_w(0);
            v4f32 _sum2 = (v4f32)__msa_fill_w(0);
            v4f32 _sum3 = (v4f32)__msa_fill_w(0);
            v4f32 _sum4 = (v4f32)__msa_fill_w(0);
            v4f32 _sum5 = (v4f32)__msa_fill_w(0);
            v4f32 _sum6 = (v4f32)__msa_fill_w(0);
            v4f32 _sum7 = (v4f32)__msa_fill_w(0);

            if (bias_data_ptr)
            {
                _sum0 = (v4f32)__msa_ld_w(bias_data_ptr + p * 8, 0);
                _sum4 = (v4f32)__msa_ld_w(bias_data_ptr + p * 8 + 4, 0);
            }

            const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* sptr = bottom_blob;

            int i = 0;
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(sptr + 16);
                __builtin_prefetch(kptr + 64);
                v4f32 _m = bfloat2float_msa(sptr);
                v4f32 _val0 = (v4f32)__msa_splati_w((v4i32)_m, 0);
                v4f32 _val1 = (v4f32)__msa_splati_w((v4i32)_m, 1);
                v4f32 _val2 = (v4f32)__msa_splati_w((v4i32)_m, 2);
                v4f32 _val3 = (v4f32)__msa_splati_w((v4i32)_m, 3);

                v8i16 _zero_bf16 = __msa_fill_h(0);
                v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);
                v8i16 _w45_bf16 = __msa_ld_h(kptr + 16, 0);
                v4f32 _w4 = (v4f32)__msa_ilvr_h(_w45_bf16, _zero_bf16);
                v4f32 _w5 = (v4f32)__msa_ilvl_h(_w45_bf16, _zero_bf16);
                v8i16 _w67_bf16 = __msa_ld_h(kptr + 24, 0);
                v4f32 _w6 = (v4f32)__msa_ilvr_h(_w67_bf16, _zero_bf16);
                v4f32 _w7 = (v4f32)__msa_ilvl_h(_w67_bf16, _zero_bf16);

                _sum0 = __ncnn_msa_fmadd_w(_sum0, _val0, _w0);
                _sum1 = __ncnn_msa_fmadd_w(_sum1, _val1, _w2);
                _sum2 = __ncnn_msa_fmadd_w(_sum2, _val2, _w4);
                _sum3 = __ncnn_msa_fmadd_w(_sum3, _val3, _w6);
                _sum4 = __ncnn_msa_fmadd_w(_sum4, _val0, _w1);
                _sum5 = __ncnn_msa_fmadd_w(_sum5, _val1, _w3);
                _sum6 = __ncnn_msa_fmadd_w(_sum6, _val2, _w5);
                _sum7 = __ncnn_msa_fmadd_w(_sum7, _val3, _w7);

                sptr += 4;
                kptr += 32;
            }
            for (; i < num_input; i++)
            {
                v4f32 _val = __msa_fill_w_f32(bfloat16_to_float32(sptr[0]));
                v8i16 _zero_bf16 = __msa_fill_h(0);
                v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                _sum0 = __ncnn_msa_fmadd_w(_sum0, _val, _w0);
                _sum4 = __ncnn_msa_fmadd_w(_sum4, _val, _w1);

                sptr += 1;
                kptr += 8;
            }

            _sum0 = __msa_fadd_w(_sum0, _sum1);
            _sum2 = __msa_fadd_w(_sum2, _sum3);
            _sum4 = __msa_fadd_w(_sum4, _sum5);
            _sum6 = __msa_fadd_w(_sum6, _sum7);
            _sum0 = __msa_fadd_w(_sum0, _sum2);
            _sum4 = __msa_fadd_w(_sum4, _sum6);

            _sum0 = activation_msa(_sum0, activation_type, activation_params);
            _sum4 = activation_msa(_sum4, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            __msa_st_w(float2bfloat_msa(_sum0, _sum4), outptr + p * 8, 0);
        }
    }

    if (out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outw; p++)
        {
            v4f32 _sum0 = (v4f32)__msa_fill_w(0);
            v4f32 _sum1 = (v4f32)__msa_fill_w(0);
            v4f32 _sum2 = (v4f32)__msa_fill_w(0);
            v4f32 _sum3 = (v4f32)__msa_fill_w(0);

            if (bias_data_ptr)
            {
                _sum0 = (v4f32)__msa_ld_w(bias_data_ptr + p * 4, 0);
            }

            const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* sptr = bottom_blob;

            int i = 0;
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(sptr + 16);
                __builtin_prefetch(kptr + 64);
                v4f32 _m = bfloat2float_msa(sptr);
                v4f32 _val0 = (v4f32)__msa_splati_w((v4i32)_m, 0);
                v4f32 _val1 = (v4f32)__msa_splati_w((v4i32)_m, 1);
                v4f32 _val2 = (v4f32)__msa_splati_w((v4i32)_m, 2);
                v4f32 _val3 = (v4f32)__msa_splati_w((v4i32)_m, 3);

                v8i16 _zero_bf16 = __msa_fill_h(0);
                v8i16 _w01_bf16 = __msa_ld_h(kptr, 0);
                v4f32 _w0 = (v4f32)__msa_ilvr_h(_w01_bf16, _zero_bf16);
                v4f32 _w1 = (v4f32)__msa_ilvl_h(_w01_bf16, _zero_bf16);
                v8i16 _w23_bf16 = __msa_ld_h(kptr + 8, 0);
                v4f32 _w2 = (v4f32)__msa_ilvr_h(_w23_bf16, _zero_bf16);
                v4f32 _w3 = (v4f32)__msa_ilvl_h(_w23_bf16, _zero_bf16);

                _sum0 = __ncnn_msa_fmadd_w(_sum0, _val0, _w0);
                _sum1 = __ncnn_msa_fmadd_w(_sum1, _val1, _w1);
                _sum2 = __ncnn_msa_fmadd_w(_sum2, _val2, _w2);
                _sum3 = __ncnn_msa_fmadd_w(_sum3, _val3, _w3);

                sptr += 4;
                kptr += 16;
            }
            for (; i < num_input; i++)
            {
                v4f32 _val = __msa_fill_w_f32(bfloat16_to_float32(sptr[0]));
                v4f32 _w = bfloat2float_msa(kptr);
                _sum0 = __ncnn_msa_fmadd_w(_sum0, _val, _w);

                sptr += 1;
                kptr += 4;
            }

            _sum0 = __msa_fadd_w(_sum0, _sum1);
            _sum2 = __msa_fadd_w(_sum2, _sum3);
            _sum0 = __msa_fadd_w(_sum0, _sum2);

            _sum0 = activation_msa(_sum0, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            *(int64_t*)(outptr + p * 4) = __msa_copy_s_d((v2i64)float2bfloat_msa(_sum0), 0);
        }
    }

#endif // __mips_msa

    if (out_elempack == 1)
    {
        int remain_outw_start = 0;
#if __mips_msa
        int nn_outw = outw >> 2;

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
            v4f32 _sum0l = (v4f32)__msa_fill_w(0);
            v4f32 _sum1l = (v4f32)__msa_fill_w(0);
            v4f32 _sum2l = (v4f32)__msa_fill_w(0);
            v4f32 _sum3l = (v4f32)__msa_fill_w(0);
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(m + 16);
                __builtin_prefetch(w0 + 16);
                __builtin_prefetch(w1 + 16);
                __builtin_prefetch(w2 + 16);
                __builtin_prefetch(w3 + 16);
                v4f32 _m = bfloat2float_msa(m);

                v4f32 _w0 = bfloat2float_msa(w0);
                v4f32 _w1 = bfloat2float_msa(w1);
                v4f32 _w2 = bfloat2float_msa(w2);
                v4f32 _w3 = bfloat2float_msa(w3);

                _sum0l = __ncnn_msa_fmadd_w(_sum0l, _m, _w0);
                _sum1l = __ncnn_msa_fmadd_w(_sum1l, _m, _w1);
                _sum2l = __ncnn_msa_fmadd_w(_sum2l, _m, _w2);
                _sum3l = __ncnn_msa_fmadd_w(_sum3l, _m, _w3);

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

            v4f32 _sums = (v4f32)__msa_ld_w(sums, 0);

            // transpose and reduce
            v4i32 _t0 = __msa_ilvr_w((v4i32)_sum1l, (v4i32)_sum0l);
            v4i32 _t1 = __msa_ilvr_w((v4i32)_sum3l, (v4i32)_sum2l);
            v4i32 _t2 = __msa_ilvl_w((v4i32)_sum1l, (v4i32)_sum0l);
            v4i32 _t3 = __msa_ilvl_w((v4i32)_sum3l, (v4i32)_sum2l);
            v4f32 _r0 = (v4f32)__msa_ilvr_d((v2i64)_t1, (v2i64)_t0);
            v4f32 _r1 = (v4f32)__msa_ilvl_d((v2i64)_t1, (v2i64)_t0);
            v4f32 _r2 = (v4f32)__msa_ilvr_d((v2i64)_t3, (v2i64)_t2);
            v4f32 _r3 = (v4f32)__msa_ilvl_d((v2i64)_t3, (v2i64)_t2);
            _sums = __msa_fadd_w(_sums, _r0);
            _sums = __msa_fadd_w(_sums, _r1);
            _sums = __msa_fadd_w(_sums, _r2);
            _sums = __msa_fadd_w(_sums, _r3);

            _sums = activation_msa(_sums, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            *(int64_t*)(outptr + p) = __msa_copy_s_d((v2i64)float2bfloat_msa(_sums), 0);
        }

        remain_outw_start += (nn_outw << 2);
#endif // __mips_msa

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outw_start; p < outw; p++)
        {
            float sum = 0.f;

            if (bias_data_ptr)
                sum = bias_data_ptr[p];

            const unsigned short* w = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* m = bottom_blob;

            int i = 0;
#if __mips_msa
            v4f32 _sum = (v4f32)__msa_fill_w(0);
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(m + 16);
                __builtin_prefetch(w + 16);
                v4f32 _m = bfloat2float_msa(m);
                v4f32 _w = bfloat2float_msa(w);
                _sum = __ncnn_msa_fmadd_w(_sum, _m, _w);

                m += 4;
                w += 4;
            }
#endif // __mips_msa
            for (; i < num_input; i++)
            {
                sum += bfloat16_to_float32(*m) * bfloat16_to_float32(*w);
                m++;
                w++;
            }

#if __mips_msa
            sum += __msa_reduce_fadd_w(_sum);
#endif // __mips_msa

            sum = activation_ss(sum, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            outptr[p] = float32_to_bfloat16(sum);
        }
    } // out_elempack == 1
}
