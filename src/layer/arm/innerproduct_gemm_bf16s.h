// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
void innerproduct_gemm_bf16s_neon_bf16(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_bf16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
#endif

static void innerproduct_gemm_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_bf16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        innerproduct_gemm_bf16s_neon_bf16(bottom_blob, top_blob, weight_data_bf16, bias_data, activation_type, activation_params, opt);
        return;
    }
#endif

    const int num_input = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int num_output = top_blob.w;
    const int h = bottom_blob.h;

    const float* bias_data_ptr = bias_data;

    int num_output_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        num_output_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __ARM_NEON

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int j = 0; j < h; j++)
    {
#if __ARM_NEON
        if (elempack == 4 && num_output_elempack == 4)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_bf16.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                float32x4_t _sum0 = vdupq_n_f32(0.f);
                float32x4_t _sum1 = vdupq_n_f32(0.f);
                float32x4_t _sum2 = vdupq_n_f32(0.f);
                float32x4_t _sum3 = vdupq_n_f32(0.f);

                if (bias_data_ptr)
                {
                    _sum0 = vdupq_n_f32(bias_data_ptr[p * 4 + 0]);
                    _sum1 = vdupq_n_f32(bias_data_ptr[p * 4 + 1]);
                    _sum2 = vdupq_n_f32(bias_data_ptr[p * 4 + 2]);
                    _sum3 = vdupq_n_f32(bias_data_ptr[p * 4 + 3]);
                }

                int i = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                float32x4_t _sum01a;
                float32x4_t _sum01b;
                float32x4_t _sum23a;
                float32x4_t _sum23b;

                if (bias_data_ptr)
                {
                    float32x4_t _bias = vld1q_f32(bias_data_ptr + p * 4);
                    _sum01a = vcombine_f32(vget_low_f32(_bias), vget_low_f32(_bias));
                    _sum01b = _sum01a;
                    _sum23a = vcombine_f32(vget_high_f32(_bias), vget_high_f32(_bias));
                    _sum23b = _sum23a;
                }
                else
                {
                    _sum01a = vdupq_n_f32(0.f);
                    _sum01b = vdupq_n_f32(0.f);
                    _sum23a = vdupq_n_f32(0.f);
                    _sum23b = vdupq_n_f32(0.f);
                }

                for (; i + 3 < num_input; i += 4)
                {
                    uint16x4x4_t _val = vld4_u16(m);
                    uint16x8_t _val01 = vcombine_u16(_val.val[0], _val.val[1]);
                    uint16x8_t _val23 = vcombine_u16(_val.val[2], _val.val[3]);

                    uint16x8_t _w01 = vld1q_u16(kptr);
                    uint16x8_t _w23 = vld1q_u16(kptr + 8);

                    _sum01a = vbfmmlaq_f32(_sum01a, (bfloat16x8_t)_val01, (bfloat16x8_t)_w01);
                    _sum01b = vbfmmlaq_f32(_sum01b, (bfloat16x8_t)_val23, (bfloat16x8_t)_w01);
                    _sum23a = vbfmmlaq_f32(_sum23a, (bfloat16x8_t)_val01, (bfloat16x8_t)_w23);
                    _sum23b = vbfmmlaq_f32(_sum23b, (bfloat16x8_t)_val23, (bfloat16x8_t)_w23);

                    m += 16;
                    kptr += 16;
                }

                float32x4x2_t _sum01 = vuzpq_f32(_sum01a, _sum01b);
                float32x4x2_t _sum23 = vuzpq_f32(_sum23a, _sum23b);
                _sum0 = _sum01.val[0];
                _sum1 = _sum01.val[1];
                _sum2 = _sum23.val[0];
                _sum3 = _sum23.val[1];

                for (; i + 1 < num_input; i += 2)
                {
                    uint16x4_t _val0 = vld1_u16(m);
                    uint16x4_t _val1 = vld1_u16(m + 4);
                    uint16x4x2_t _val01 = vzip_u16(_val0, _val1);
                    uint16x8_t _val = vcombine_u16(_val01.val[0], _val01.val[1]);

                    uint16x8_t _w = vld1q_u16(kptr);
                    uint32x4_t _w_32 = vreinterpretq_u32_u16(_w);
                    uint32x2_t _w01_32 = vget_low_u32(_w_32);
                    uint32x2_t _w23_32 = vget_high_u32(_w_32);
                    uint32x2_t _w0_32 = vdup_lane_u32(_w01_32, 0);
                    uint32x2_t _w1_32 = vdup_lane_u32(_w01_32, 1);
                    uint32x2_t _w2_32 = vdup_lane_u32(_w23_32, 0);
                    uint32x2_t _w3_32 = vdup_lane_u32(_w23_32, 1);
                    uint16x8_t _w00 = vreinterpretq_u16_u32(vcombine_u32(_w0_32, _w0_32));
                    uint16x8_t _w11 = vreinterpretq_u16_u32(vcombine_u32(_w1_32, _w1_32));
                    uint16x8_t _w22 = vreinterpretq_u16_u32(vcombine_u32(_w2_32, _w2_32));
                    uint16x8_t _w33 = vreinterpretq_u16_u32(vcombine_u32(_w3_32, _w3_32));

                    _sum0 = vbfdotq_f32(_sum0, (bfloat16x8_t)_val, (bfloat16x8_t)_w00);
                    _sum1 = vbfdotq_f32(_sum1, (bfloat16x8_t)_val, (bfloat16x8_t)_w11);
                    _sum2 = vbfdotq_f32(_sum2, (bfloat16x8_t)_val, (bfloat16x8_t)_w22);
                    _sum3 = vbfdotq_f32(_sum3, (bfloat16x8_t)_val, (bfloat16x8_t)_w33);

                    m += 8;
                    kptr += 8;
                }
#else // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                for (; i < num_input; i++)
                {
                    float32x4_t _val = bfloat2float(vld1_u16(m));
                    float32x4_t _k = bfloat2float(vld1_u16(kptr));
#if __aarch64__
                    _sum0 = vfmaq_laneq_f32(_sum0, _val, _k, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _val, _k, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _val, _k, 2);
                    _sum3 = vfmaq_laneq_f32(_sum3, _val, _k, 3);
#else
                    _sum0 = vmlaq_lane_f32(_sum0, _val, vget_low_f32(_k), 0);
                    _sum1 = vmlaq_lane_f32(_sum1, _val, vget_low_f32(_k), 1);
                    _sum2 = vmlaq_lane_f32(_sum2, _val, vget_high_f32(_k), 0);
                    _sum3 = vmlaq_lane_f32(_sum3, _val, vget_high_f32(_k), 1);
#endif

                    m += 4;
                    kptr += 4;
                }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                for (; i < num_input; i++)
                {
                    float32x4_t _val = bfloat2float(vld1_u16(m));
                    float32x4_t _k = bfloat2float(vld1_u16(kptr));
#if __aarch64__
                    _sum0 = vfmaq_laneq_f32(_sum0, _val, _k, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _val, _k, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _val, _k, 2);
                    _sum3 = vfmaq_laneq_f32(_sum3, _val, _k, 3);
#else
                    _sum0 = vmlaq_lane_f32(_sum0, _val, vget_low_f32(_k), 0);
                    _sum1 = vmlaq_lane_f32(_sum1, _val, vget_low_f32(_k), 1);
                    _sum2 = vmlaq_lane_f32(_sum2, _val, vget_high_f32(_k), 0);
                    _sum3 = vmlaq_lane_f32(_sum3, _val, vget_high_f32(_k), 1);
#endif

                    m += 4;
                    kptr += 4;
                }

                _sum0 = activation_ps(_sum0, activation_type, activation_params);
                _sum1 = activation_ps(_sum1, activation_type, activation_params);
                _sum2 = activation_ps(_sum2, activation_type, activation_params);
                _sum3 = activation_ps(_sum3, activation_type, activation_params);

                vst1_u16(outptr, float2bfloat(_sum0));
                vst1_u16(outptr + 4, float2bfloat(_sum1));
                vst1_u16(outptr + 8, float2bfloat(_sum2));
                vst1_u16(outptr + 12, float2bfloat(_sum3));
                outptr += 16;
            }
        }

        if (elempack == 1 && num_output_elempack == 4)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_bf16.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                float32x4_t _sum = vdupq_n_f32(0.f);

                if (bias_data_ptr)
                {
                    _sum = vld1q_f32(bias_data_ptr + p * 4);
                }

                int i = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                float32x4_t _sum01 = vdupq_n_f32(0.f);
                float32x4_t _sum23 = vdupq_n_f32(0.f);
                for (; i + 3 < num_input; i += 4)
                {
                    uint16x4_t _val = vld1_u16(m);
                    uint16x8_t _val01 = vcombine_u16(_val, _val);

                    uint16x8_t _w01 = vld1q_u16(kptr);
                    uint16x8_t _w23 = vld1q_u16(kptr + 8);

                    _sum01 = vbfdotq_f32(_sum01, (bfloat16x8_t)_val01, (bfloat16x8_t)_w01);
                    _sum23 = vbfdotq_f32(_sum23, (bfloat16x8_t)_val01, (bfloat16x8_t)_w23);

                    m += 4;
                    kptr += 16;
                }
                float32x2_t _sum01p = vpadd_f32(vget_low_f32(_sum01), vget_high_f32(_sum01));
                float32x2_t _sum23p = vpadd_f32(vget_low_f32(_sum23), vget_high_f32(_sum23));
                _sum = vaddq_f32(_sum, vcombine_f32(_sum01p, _sum23p));
                for (; i + 1 < num_input; i += 2)
                {
                    uint32x2_t _val01_32 = vld1_dup_u32((const uint32_t*)m);
                    uint16x8_t _val01 = vreinterpretq_u16_u32(vcombine_u32(_val01_32, _val01_32));

                    uint16x8_t _w0123 = vld1q_u16(kptr);

                    _sum = vbfdotq_f32(_sum, (bfloat16x8_t)_val01, (bfloat16x8_t)_w0123);

                    m += 2;
                    kptr += 8;
                }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                for (; i < num_input; i++)
                {
                    float32x4_t _val = vdupq_n_f32(bfloat16_to_float32(m[0]));
                    float32x4_t _k = bfloat2float(vld1_u16(kptr));
                    _sum = vmlaq_f32(_sum, _val, _k);

                    m += 1;
                    kptr += 4;
                }

                _sum = activation_ps(_sum, activation_type, activation_params);

                vst1_u16(outptr, float2bfloat(_sum));
                outptr += 4;
            }
        }

        if (elempack == 4 && num_output_elempack == 1)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output; p++)
            {
                const unsigned short* kptr = weight_data_bf16.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                float32x4_t _sum = vdupq_n_f32(0.f);

                if (bias_data_ptr)
                {
                    _sum = vdupq_n_f32(bias_data_ptr[p]);
                }

                int i = 0;
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                float32x4_t _sum01 = vdupq_n_f32(0.f);
                float32x4_t _sum23 = vdupq_n_f32(0.f);

                for (; i + 3 < num_input; i += 4)
                {
                    uint16x4x4_t _val = vld4_u16(m);
                    uint16x8_t _val01 = vcombine_u16(_val.val[0], _val.val[1]);
                    uint16x8_t _val23 = vcombine_u16(_val.val[2], _val.val[3]);
                    uint16x4_t _w = vld1_u16(kptr);
                    uint16x8_t _w01 = vcombine_u16(_w, _w);

                    _sum01 = vbfdotq_f32(_sum01, (bfloat16x8_t)_val01, (bfloat16x8_t)_w01);
                    _sum23 = vbfdotq_f32(_sum23, (bfloat16x8_t)_val23, (bfloat16x8_t)_w01);

                    m += 16;
                    kptr += 4;
                }

                float32x2_t _sum01p = vpadd_f32(vget_low_f32(_sum01), vget_high_f32(_sum01));
                float32x2_t _sum23p = vpadd_f32(vget_low_f32(_sum23), vget_high_f32(_sum23));
                _sum = vaddq_f32(_sum, vcombine_f32(_sum01p, _sum23p));

                for (; i + 1 < num_input; i += 2)
                {
                    uint16x4_t _val0 = vld1_u16(m);
                    uint16x4_t _val1 = vld1_u16(m + 4);
                    uint16x4x2_t _val01 = vzip_u16(_val0, _val1);
                    uint16x8_t _val = vcombine_u16(_val01.val[0], _val01.val[1]);
                    uint32x2_t _w_32 = vld1_dup_u32((const uint32_t*)kptr);
                    uint16x8_t _w = vreinterpretq_u16_u32(vcombine_u32(_w_32, _w_32));

                    _sum = vbfdotq_f32(_sum, (bfloat16x8_t)_val, (bfloat16x8_t)_w);

                    m += 8;
                    kptr += 2;
                }
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                for (; i < num_input; i++)
                {
                    float32x4_t _val = bfloat2float(vld1_u16(m));
                    float32x4_t _k = vdupq_n_f32(bfloat16_to_float32(kptr[0]));
                    _sum = vmlaq_f32(_sum, _val, _k);

                    m += 4;
                    kptr += 1;
                }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                for (; i < num_input; i++)
                {
                    float32x4_t _val = bfloat2float(vld1_u16(m));
                    float32x4_t _k = vdupq_n_f32(bfloat16_to_float32(kptr[0]));
                    _sum = vmlaq_f32(_sum, _val, _k);

                    m += 4;
                    kptr += 1;
                }

                _sum = activation_ps(_sum, activation_type, activation_params);

                vst1_u16(outptr, float2bfloat(_sum));
                outptr += 4;
            }
        }
#endif // __ARM_NEON

        if (elempack == 1 && num_output_elempack == 1)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output; p++)
            {
                const unsigned short* kptr = weight_data_bf16.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                float sum = 0.f;

                if (bias_data_ptr)
                {
                    sum = bias_data_ptr[p];
                }

                int i = 0;
#if __ARM_NEON
                float32x4_t _sum = vdupq_n_f32(0.f);
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
                float32x2_t _sum2 = vdup_n_f32(0.f);
                for (; i + 7 < num_input; i += 8)
                {
                    uint16x8_t _m = vld1q_u16(m);
                    uint16x8_t _w = vld1q_u16(kptr);

                    _sum = vbfdotq_f32(_sum, (bfloat16x8_t)_m, (bfloat16x8_t)_w);

                    m += 8;
                    kptr += 8;
                }
                for (; i + 3 < num_input; i += 4)
                {
                    uint16x4_t _m = vld1_u16(m);
                    uint16x4_t _w = vld1_u16(kptr);

                    _sum2 = vbfdot_f32(_sum2, (bfloat16x4_t)_m, (bfloat16x4_t)_w);

                    m += 4;
                    kptr += 4;
                }
#else
                for (; i + 3 < num_input; i += 4)
                {
                    float32x4_t _m = bfloat2float(vld1_u16(m));
                    float32x4_t _w = bfloat2float(vld1_u16(kptr));

                    _sum = vmlaq_f32(_sum, _m, _w);

                    m += 4;
                    kptr += 4;
                }
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#endif // __ARM_NEON
                for (; i < num_input; i++)
                {
                    sum += bfloat16_to_float32(*m) * bfloat16_to_float32(*kptr);

                    m += 1;
                    kptr += 1;
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

                outptr[0] = float32_to_bfloat16(sum);
                outptr += 1;
            }
        }
    }
}
