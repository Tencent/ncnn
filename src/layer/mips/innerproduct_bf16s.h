// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void innerproduct_transform_kernel_bf16s_msa(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt)
{
    int out_elempack = 1;
#if __mips_msa
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif

    // src = inch-outch
    // dst = pb-inch-outch/pb
#if __mips_msa
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
        num_output_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int j = 0; j < h; j++)
    {
#if __mips_msa
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
                    v4f32 _val0 = __msa_fill_w_f32(bfloat16_to_float32(m[0]));
                    v4f32 _val1 = __msa_fill_w_f32(bfloat16_to_float32(m[1]));
                    v4f32 _val2 = __msa_fill_w_f32(bfloat16_to_float32(m[2]));
                    v4f32 _val3 = __msa_fill_w_f32(bfloat16_to_float32(m[3]));
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

                v4f32 _sum = (v4f32)__msa_fill_w(0);

                if (bias_data_ptr)
                {
                    _sum = (v4f32)__msa_ld_w(bias_data_ptr + p * 4, 0);
                }

                int i = 0;
                for (; i < num_input; i++)
                {
                    __builtin_prefetch(m + 16);
                    __builtin_prefetch(kptr + 16);
                    v4f32 _val = __msa_fill_w_f32(bfloat16_to_float32(m[0]));
                    v4f32 _w = bfloat2float_msa(kptr);
                    _sum = __ncnn_msa_fmadd_w(_sum, _val, _w);

                    m += 1;
                    kptr += 4;
                }

                _sum = activation_msa(_sum, activation_type, activation_params);

                __msa_storel_d(float2bfloat_msa(_sum), outptr);
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
                    v4f32 _val0 = bfloat2float_msa(m);
                    v4f32 _val1 = bfloat2float_msa(m + 4);
                    v4f32 _val2 = bfloat2float_msa(m + 8);
                    v4f32 _val3 = bfloat2float_msa(m + 12);

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

                __msa_storel_d(float2bfloat_msa(_sum0), outptr);
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
                v4f32 _val0 = __msa_fill_w_f32(bfloat16_to_float32(sptr[0]));
                v4f32 _val1 = __msa_fill_w_f32(bfloat16_to_float32(sptr[1]));
                v4f32 _val2 = __msa_fill_w_f32(bfloat16_to_float32(sptr[2]));
                v4f32 _val3 = __msa_fill_w_f32(bfloat16_to_float32(sptr[3]));

                v4f32 _w0 = bfloat2float_msa(kptr);
                v4f32 _w1 = bfloat2float_msa(kptr + 4);
                v4f32 _w2 = bfloat2float_msa(kptr + 8);
                v4f32 _w3 = bfloat2float_msa(kptr + 12);

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
            __msa_storel_d(float2bfloat_msa(_sum0), outptr + p * 4);
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
            __msa_storel_d(float2bfloat_msa(_sums), outptr + p);
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
