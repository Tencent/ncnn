// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
void innerproduct_pack4_bf16s_neon_bf16(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_bf16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
void innerproduct_bf16s_neon_bf16(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_bf16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
void innerproduct_transform_kernel_bf16s_neon_bf16(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt);
#endif

#if __ARM_NEON
static void innerproduct_pack4_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_bf16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        innerproduct_pack4_bf16s_neon_bf16(bottom_blob, top_blob, weight_data_bf16, bias_data, activation_type, activation_params, opt);
        return;
    }
#endif

    const int num_input = bottom_blob.w * bottom_blob.elempack;
    const int num_output = top_blob.w;

    const float* bias_data_ptr = bias_data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < num_output; p++)
    {
        float32x4_t _sum0 = vdupq_n_f32(0.f);

        if (bias_data_ptr)
        {
            _sum0 = vld1q_f32(bias_data_ptr + p * 4);
        }

        float32x4_t _sum1 = vdupq_n_f32(0.f);
        float32x4_t _sum2 = vdupq_n_f32(0.f);
        float32x4_t _sum3 = vdupq_n_f32(0.f);

        const unsigned short* sptr = bottom_blob;
        const unsigned short* kptr = weight_data_bf16.row<const unsigned short>(p);

        int i = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        float32x4_t _sum01 = vdupq_n_f32(0.f);
        float32x4_t _sum23 = vdupq_n_f32(0.f);
        for (; i + 3 < num_input; i += 4)
        {
            uint16x4_t _val = vld1_u16(sptr);
            uint16x8_t _val01 = vcombine_u16(_val, _val);

            uint16x8_t _w01 = vld1q_u16(kptr);
            uint16x8_t _w23 = vld1q_u16(kptr + 8);

            _sum01 = vbfdotq_f32(_sum01, (bfloat16x8_t)_val01, (bfloat16x8_t)_w01);
            _sum23 = vbfdotq_f32(_sum23, (bfloat16x8_t)_val01, (bfloat16x8_t)_w23);

            sptr += 4;
            kptr += 16;
        }
        float32x2_t _sum01p = vpadd_f32(vget_low_f32(_sum01), vget_high_f32(_sum01));
        float32x2_t _sum23p = vpadd_f32(vget_low_f32(_sum23), vget_high_f32(_sum23));
        _sum0 = vaddq_f32(_sum0, vcombine_f32(_sum01p, _sum23p));
        for (; i + 1 < num_input; i += 2)
        {
            uint32x2_t _val01_32 = vld1_dup_u32((const uint32_t*)sptr);
            uint16x8_t _val01 = vreinterpretq_u16_u32(vcombine_u32(_val01_32, _val01_32));
            uint16x8_t _w = vld1q_u16(kptr);

            _sum0 = vbfdotq_f32(_sum0, (bfloat16x8_t)_val01, (bfloat16x8_t)_w);

            sptr += 2;
            kptr += 8;
        }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; i + 3 < num_input; i += 4)
        {
            float32x4_t _val = bfloat2float(vld1_u16(sptr));

            float32x4_t _w0 = bfloat2float(vld1_u16(kptr));
            float32x4_t _w1 = bfloat2float(vld1_u16(kptr + 4));
            float32x4_t _w2 = bfloat2float(vld1_u16(kptr + 8));
            float32x4_t _w3 = bfloat2float(vld1_u16(kptr + 12));

#if __aarch64__
            _sum0 = vmlaq_laneq_f32(_sum0, _w0, _val, 0);
            _sum1 = vmlaq_laneq_f32(_sum1, _w1, _val, 1);
            _sum2 = vmlaq_laneq_f32(_sum2, _w2, _val, 2);
            _sum3 = vmlaq_laneq_f32(_sum3, _w3, _val, 3);
#else
            _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_val), 0);
            _sum1 = vmlaq_lane_f32(_sum1, _w1, vget_low_f32(_val), 1);
            _sum2 = vmlaq_lane_f32(_sum2, _w2, vget_high_f32(_val), 0);
            _sum3 = vmlaq_lane_f32(_sum3, _w3, vget_high_f32(_val), 1);
#endif

            sptr += 4;
            kptr += 16;
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        for (; i < num_input; i++)
        {
            float32x4_t _val = vdupq_n_f32(bfloat16_to_float32(sptr[0]));

            float32x4_t _w = bfloat2float(vld1_u16(kptr));

            _sum0 = vmlaq_f32(_sum0, _val, _w);

            sptr += 1;
            kptr += 4;
        }

        _sum0 = vaddq_f32(_sum0, _sum1);
        _sum2 = vaddq_f32(_sum2, _sum3);
        _sum0 = vaddq_f32(_sum0, _sum2);

        _sum0 = activation_ps(_sum0, activation_type, activation_params);

        unsigned short* outptr = (unsigned short*)top_blob;
        vst1_u16(outptr + p * 4, float2bfloat(_sum0));
    }
}
#endif // __ARM_NEON

static void innerproduct_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_bf16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        innerproduct_bf16s_neon_bf16(bottom_blob, top_blob, weight_data_bf16, bias_data, activation_type, activation_params, opt);
        return;
    }
#endif

    const int num_input = bottom_blob.w * bottom_blob.elempack;
    const int num_output = top_blob.w;

    const float* bias_data_ptr = bias_data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < num_output; p++)
    {
        float sum = 0.f;

        if (bias_data_ptr)
            sum = bias_data_ptr[p];

        const unsigned short* kptr = weight_data_bf16.row<unsigned short>(p);

        const unsigned short* sptr = bottom_blob;

        int i = 0;
#if __ARM_NEON
        float32x4_t _sum = vdupq_n_f32(0.f);
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        float32x2_t _sum2 = vdup_n_f32(0.f);
        for (; i + 7 < num_input; i += 8)
        {
            uint16x8_t _m = vld1q_u16(sptr);
            uint16x8_t _w = vld1q_u16(kptr);

            _sum = vbfdotq_f32(_sum, (bfloat16x8_t)_m, (bfloat16x8_t)_w);

            sptr += 8;
            kptr += 8;
        }
        for (; i + 3 < num_input; i += 4)
        {
            uint16x4_t _m = vld1_u16(sptr);
            uint16x4_t _w = vld1_u16(kptr);

            _sum2 = vbfdot_f32(_sum2, (bfloat16x4_t)_m, (bfloat16x4_t)_w);

            sptr += 4;
            kptr += 4;
        }
#else
        for (; i + 3 < num_input; i += 4)
        {
            float32x4_t _m = bfloat2float(vld1_u16(sptr));
            float32x4_t _w = bfloat2float(vld1_u16(kptr));

            _sum = vmlaq_f32(_sum, _m, _w);

            sptr += 4;
            kptr += 4;
        }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#endif // __ARM_NEON
        for (; i < num_input; i++)
        {
            float v = bfloat16_to_float32(*sptr);
            float k = bfloat16_to_float32(*kptr);

            sum += v * k;

            sptr++;
            kptr++;
        }

#if __ARM_NEON
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        float32x2_t _sum2p = vpadd_f32(_sum2, _sum2);
        sum += vget_lane_f32(_sum2p, 0);
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#if __aarch64__
        sum += vaddvq_f32(_sum);
#else
        float32x2_t _sumss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
        _sumss = vpadd_f32(_sumss, _sumss);
        sum += vget_lane_f32(_sumss, 0);
#endif // __aarch64__
#endif // __ARM_NEON

        sum = activation_ss(sum, activation_type, activation_params);

        unsigned short* outptr = (unsigned short*)top_blob;
        outptr[p] = float32_to_bfloat16(sum);
    }
}

static void innerproduct_transform_kernel_bf16s_neon(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        innerproduct_transform_kernel_bf16s_neon_bf16(weight_data, weight_data_tm, num_input, num_output, opt);
        return;
    }
#endif

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
#if __ARM_NEON
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif // __ARM_NEON
    }

    // src = inch-outch
    // dst = pb-inch-outch/pb
    Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

    weight_data_tm.create(num_input, num_output / out_elempack, (size_t)2u * out_elempack, out_elempack);

    if (out_elempack == 4)
    {
        for (int q = 0; q + 3 < num_output; q += 4)
        {
            unsigned short* g0 = weight_data_tm.row<unsigned short>(q / 4);

            const float* k0 = weight_data_r2.row(q);
            const float* k1 = weight_data_r2.row(q + 1);
            const float* k2 = weight_data_r2.row(q + 2);
            const float* k3 = weight_data_r2.row(q + 3);

            int p = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; p + 3 < num_input; p += 4)
            {
                uint16x4_t _r0 = float2bfloat(vld1q_f32(k0 + p));
                uint16x4_t _r1 = float2bfloat(vld1q_f32(k1 + p));
                uint16x4_t _r2 = float2bfloat(vld1q_f32(k2 + p));
                uint16x4_t _r3 = float2bfloat(vld1q_f32(k3 + p));
                vst1q_u16(g0, vcombine_u16(_r0, _r1));
                vst1q_u16(g0 + 8, vcombine_u16(_r2, _r3));
                g0 += 16;
            }
            for (; p + 1 < num_input; p += 2)
            {
                float32x4_t _k01 = vcombine_f32(vld1_f32(k0 + p), vld1_f32(k1 + p));
                float32x4_t _k23 = vcombine_f32(vld1_f32(k2 + p), vld1_f32(k3 + p));
                uint16x4_t _r01 = float2bfloat(_k01);
                uint16x4_t _r23 = float2bfloat(_k23);
                vst1q_u16(g0, vcombine_u16(_r01, _r23));
                g0 += 8;
            }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            for (; p < num_input; p++)
            {
                g0[0] = float32_to_bfloat16(k0[p]);
                g0[1] = float32_to_bfloat16(k1[p]);
                g0[2] = float32_to_bfloat16(k2[p]);
                g0[3] = float32_to_bfloat16(k3[p]);
                g0 += 4;
            }
        }
    }
    else
    {
        for (int q = 0; q < num_output; q++)
        {
            unsigned short* g0 = weight_data_tm.row<unsigned short>(q);
            const float* k0 = weight_data_r2.row(q);

            int p = 0;
#if __ARM_NEON
            for (; p + 3 < num_input; p += 4)
            {
                vst1_u16(g0, float2bfloat(vld1q_f32(k0 + p)));
                g0 += 4;
            }
#endif // __ARM_NEON
            for (; p < num_input; p++)
            {
                *g0++ = float32_to_bfloat16(k0[p]);
            }
        }
    }
}
