// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void innerproduct_fp16s_pack4_neon_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
void innerproduct_fp16s_neon_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
void innerproduct_transform_kernel_fp16s_neon_asimdhp(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_VFPV4 && __ARM_NEON && !(__ARM_FP & 2)
void innerproduct_fp16s_pack4_neon_vfpv4(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
void innerproduct_fp16s_neon_vfpv4(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
void innerproduct_transform_kernel_fp16s_neon_vfpv4(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt);
#endif

static void innerproduct_fp16s_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_asimdhp())
    {
        innerproduct_fp16s_pack4_neon_asimdhp(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_VFPV4 && __ARM_NEON && !(__ARM_FP & 2)
    if (ncnn::cpu_support_arm_vfpv4())
    {
        innerproduct_fp16s_pack4_neon_vfpv4(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
        return;
    }
#endif

#if (__ARM_FP & 2)
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

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        const __fp16* sptr = bottom_blob;
        const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);
#else
        const float* sptr = bottom_blob;
        const unsigned short* kptr = weight_data_fp16.row<const unsigned short>(p);
#endif

        int i = 0;
        for (; i + 7 < num_input; i += 8)
        {
#if __aarch64__
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            asm volatile(
                "prfm   pldl1keep, [%0, #128]       \n"
                "ld1    {v1.8h}, [%0], #16          \n"
                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v6.8h, v7.8h, v8.8h, v9.8h}, [%1], #64 \n"
                "fcvtl  v0.4s, v1.4h                \n"
                "fcvtl2 v1.4s, v1.8h                \n"
                "fcvtl  v2.4s, v6.4h                \n"
                "fcvtl2 v3.4s, v6.8h                \n"
                "fcvtl  v4.4s, v7.4h                \n"
                "fcvtl2 v5.4s, v7.8h                \n"
                "fcvtl  v6.4s, v8.4h                \n"
                "fcvtl2 v7.4s, v8.8h                \n"
                "fcvtl  v8.4s, v9.4h                \n"
                "fcvtl2 v9.4s, v9.8h                \n"
                "fmla   %2.4s, v2.4s, v0.s[0]       \n"
                "fmla   %3.4s, v3.4s, v0.s[1]       \n"
                "fmla   %4.4s, v4.4s, v0.s[2]       \n"
                "fmla   %5.4s, v5.4s, v0.s[3]       \n"
                "fmla   %2.4s, v6.4s, v1.s[0]       \n"
                "fmla   %3.4s, v7.4s, v1.s[1]       \n"
                "fmla   %4.4s, v8.4s, v1.s[2]       \n"
                "fmla   %5.4s, v9.4s, v1.s[3]       \n"
                : "=r"(sptr),  // %0
                "=r"(kptr),  // %1
                "=w"(_sum0), // %2
                "=w"(_sum1), // %3
                "=w"(_sum2), // %4
                "=w"(_sum3)  // %5
                : "0"(sptr),
                "1"(kptr),
                "2"(_sum0),
                "3"(_sum1),
                "4"(_sum2),
                "5"(_sum3)
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9");
#else  // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            asm volatile(
                "prfm   pldl1keep, [%0, #256]       \n"
                "ld1    {v0.4s, v1.4s}, [%0], #32   \n"
                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v6.8h, v7.8h, v8.8h, v9.8h}, [%1], #64 \n"
                "fcvtl  v2.4s, v6.4h                \n"
                "fcvtl2 v3.4s, v6.8h                \n"
                "fcvtl  v4.4s, v7.4h                \n"
                "fcvtl2 v5.4s, v7.8h                \n"
                "fcvtl  v6.4s, v8.4h                \n"
                "fcvtl2 v7.4s, v8.8h                \n"
                "fcvtl  v8.4s, v9.4h                \n"
                "fcvtl2 v9.4s, v9.8h                \n"
                "fmla   %2.4s, v2.4s, v0.s[0]       \n"
                "fmla   %3.4s, v3.4s, v0.s[1]       \n"
                "fmla   %4.4s, v4.4s, v0.s[2]       \n"
                "fmla   %5.4s, v5.4s, v0.s[3]       \n"
                "fmla   %2.4s, v6.4s, v1.s[0]       \n"
                "fmla   %3.4s, v7.4s, v1.s[1]       \n"
                "fmla   %4.4s, v8.4s, v1.s[2]       \n"
                "fmla   %5.4s, v9.4s, v1.s[3]       \n"
                : "=r"(sptr),  // %0
                "=r"(kptr),  // %1
                "=w"(_sum0), // %2
                "=w"(_sum1), // %3
                "=w"(_sum2), // %4
                "=w"(_sum3)  // %5
                : "0"(sptr),
                "1"(kptr),
                "2"(_sum0),
                "3"(_sum1),
                "4"(_sum2),
                "5"(_sum3)
                : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9");
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#else  // __aarch64__
            asm volatile(
                "pld        [%0, #256]          \n"
                "vld1.f32   {d0-d3}, [%0 :128]! \n"
                "pld        [%1, #512]          \n"
                "vldm       %1!, {d12-d19}      \n"
                "vcvt.f32.f16 q2, d12           \n"
                "vcvt.f32.f16 q3, d13           \n"
                "vcvt.f32.f16 q4, d14           \n"
                "vcvt.f32.f16 q5, d15           \n"
                "vcvt.f32.f16 q6, d16           \n"
                "vcvt.f32.f16 q7, d17           \n"
                "vcvt.f32.f16 q8, d18           \n"
                "vcvt.f32.f16 q9, d19           \n"
                "vmla.f32   %q2, q2, d0[0]      \n"
                "vmla.f32   %q3, q3, d0[1]      \n"
                "vmla.f32   %q4, q4, d1[0]      \n"
                "vmla.f32   %q5, q5, d1[1]      \n"
                "vmla.f32   %q2, q6, d2[0]      \n"
                "vmla.f32   %q3, q7, d2[1]      \n"
                "vmla.f32   %q4, q8, d3[0]      \n"
                "vmla.f32   %q5, q9, d3[1]      \n"
                : "=r"(sptr),  // %0
                "=r"(kptr),  // %1
                "=w"(_sum0), // %2
                "=w"(_sum1), // %3
                "=w"(_sum2), // %4
                "=w"(_sum3)  // %5
                : "0"(sptr),
                "1"(kptr),
                "2"(_sum0),
                "3"(_sum1),
                "4"(_sum2),
                "5"(_sum3)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9");
#endif // __aarch64__
        }
        for (; i + 3 < num_input; i += 4)
        {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr));
            float16x8_t _w01 = vld1q_f16(kptr);
            float16x8_t _w23 = vld1q_f16(kptr + 8);
            float32x4_t _w0 = vcvt_f32_f16(vget_low_f16(_w01));
            float32x4_t _w1 = vcvt_f32_f16(vget_high_f16(_w01));
            float32x4_t _w2 = vcvt_f32_f16(vget_low_f16(_w23));
            float32x4_t _w3 = vcvt_f32_f16(vget_high_f16(_w23));
#else
            float32x4_t _val = vld1q_f32(sptr);
            uint16x8_t _w01 = vld1q_u16(kptr);
            uint16x8_t _w23 = vld1q_u16(kptr + 8);
            float32x4_t _w0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(_w01)));
            float32x4_t _w1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(_w01)));
            float32x4_t _w2 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(_w23)));
            float32x4_t _w3 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(_w23)));
#endif

#if __aarch64__
            _sum0 = vfmaq_laneq_f32(_sum0, _w0, _val, 0);
            _sum1 = vfmaq_laneq_f32(_sum1, _w1, _val, 1);
            _sum2 = vfmaq_laneq_f32(_sum2, _w2, _val, 2);
            _sum3 = vfmaq_laneq_f32(_sum3, _w3, _val, 3);
#else
            _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_val), 0);
            _sum1 = vmlaq_lane_f32(_sum1, _w1, vget_low_f32(_val), 1);
            _sum2 = vmlaq_lane_f32(_sum2, _w2, vget_high_f32(_val), 0);
            _sum3 = vmlaq_lane_f32(_sum3, _w3, vget_high_f32(_val), 1);
#endif

            sptr += 4;
            kptr += 16;
        }
        for (; i < num_input; i++)
        {
            float32x4_t _val = vdupq_n_f32((float)sptr[0]);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr));
#else
            float32x4_t _w = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kptr)));
#endif
            _sum0 = vfmaq_f32(_sum0, _val, _w);

            sptr += 1;
            kptr += 4;
        }

        _sum0 = vaddq_f32(_sum0, _sum1);
        _sum2 = vaddq_f32(_sum2, _sum3);
        _sum0 = vaddq_f32(_sum0, _sum2);

        _sum0 = activation_ps(_sum0, activation_type, activation_params);

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        __fp16* outptr = (__fp16*)top_blob;
        vst1_f16(outptr + p * 4, vcvt_f16_f32(_sum0));
#else
        float* outptr = top_blob;
        vst1q_f32(outptr + p * 4, _sum0);
#endif
    }
#else  // (__ARM_FP & 2)
    (void)bottom_blob;
    (void)top_blob;
    (void)weight_data_fp16;
    (void)bias_data;
    (void)activation_type;
    (void)activation_params;
    (void)opt;
#endif // (__ARM_FP & 2)
}

static void innerproduct_fp16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_asimdhp())
    {
        innerproduct_fp16s_neon_asimdhp(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_VFPV4 && __ARM_NEON && !(__ARM_FP & 2)
    if (ncnn::cpu_support_arm_vfpv4())
    {
        innerproduct_fp16s_neon_vfpv4(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
        return;
    }
#endif

#if (__ARM_FP & 2)
    const int num_input = bottom_blob.w * bottom_blob.elempack;
    const int num_output = top_blob.w;

    const float* bias_data_ptr = bias_data;

    int nn_num_output = num_output >> 2;
    int remain_num_output_start = nn_num_output << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_num_output; pp++)
    {
        int p = pp * 4;

        float sums[4] = {0.0f};
        if (bias_data_ptr)
        {
            sums[0] = bias_data_ptr[p];
            sums[1] = bias_data_ptr[p + 1];
            sums[2] = bias_data_ptr[p + 2];
            sums[3] = bias_data_ptr[p + 3];
        }

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        const __fp16* sptr = bottom_blob;
        const __fp16* kptr0 = weight_data_fp16.row<const __fp16>(p);
        const __fp16* kptr1 = weight_data_fp16.row<const __fp16>(p + 1);
        const __fp16* kptr2 = weight_data_fp16.row<const __fp16>(p + 2);
        const __fp16* kptr3 = weight_data_fp16.row<const __fp16>(p + 3);
#else
        const float* sptr = bottom_blob;
        const unsigned short* kptr0 = weight_data_fp16.row<const unsigned short>(p);
        const unsigned short* kptr1 = weight_data_fp16.row<const unsigned short>(p + 1);
        const unsigned short* kptr2 = weight_data_fp16.row<const unsigned short>(p + 2);
        const unsigned short* kptr3 = weight_data_fp16.row<const unsigned short>(p + 3);
#endif

        int i = 0;

        float32x4_t _sum0 = vdupq_n_f32(0.f);
        float32x4_t _sum1 = vdupq_n_f32(0.f);
        float32x4_t _sum2 = vdupq_n_f32(0.f);
        float32x4_t _sum3 = vdupq_n_f32(0.f);
        for (; i + 3 < num_input; i += 4)
        {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr));
            float32x4_t _w0 = vcvt_f32_f16(vld1_f16(kptr0));
            float32x4_t _w1 = vcvt_f32_f16(vld1_f16(kptr1));
            float32x4_t _w2 = vcvt_f32_f16(vld1_f16(kptr2));
            float32x4_t _w3 = vcvt_f32_f16(vld1_f16(kptr3));
#else
            float32x4_t _val = vld1q_f32(sptr);
            float32x4_t _w0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kptr0)));
            float32x4_t _w1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kptr1)));
            float32x4_t _w2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kptr2)));
            float32x4_t _w3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kptr3)));
#endif

            _sum0 = vfmaq_f32(_sum0, _val, _w0);
            _sum1 = vfmaq_f32(_sum1, _val, _w1);
            _sum2 = vfmaq_f32(_sum2, _val, _w2);
            _sum3 = vfmaq_f32(_sum3, _val, _w3);

            sptr += 4;
            kptr0 += 4;
            kptr1 += 4;
            kptr2 += 4;
            kptr3 += 4;
        }

#if __aarch64__
        sums[0] += vaddvq_f32(_sum0);
        sums[1] += vaddvq_f32(_sum1);
        sums[2] += vaddvq_f32(_sum2);
        sums[3] += vaddvq_f32(_sum3);
#else
        float32x2_t _sum0ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
        float32x2_t _sum1ss = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
        float32x2_t _sum2ss = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
        float32x2_t _sum3ss = vadd_f32(vget_low_f32(_sum3), vget_high_f32(_sum3));
        float32x2_t _sum01ss = vpadd_f32(_sum0ss, _sum1ss);
        float32x2_t _sum23ss = vpadd_f32(_sum2ss, _sum3ss);
        sums[0] += vget_lane_f32(_sum01ss, 0);
        sums[1] += vget_lane_f32(_sum01ss, 1);
        sums[2] += vget_lane_f32(_sum23ss, 0);
        sums[3] += vget_lane_f32(_sum23ss, 1);
#endif

        for (; i < num_input; i++)
        {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            sums[0] += (float)(*sptr) * (float)(*kptr0);
            sums[1] += (float)(*sptr) * (float)(*kptr1);
            sums[2] += (float)(*sptr) * (float)(*kptr2);
            sums[3] += (float)(*sptr) * (float)(*kptr3);
#else
            sums[0] += *sptr * float16_to_float32(*kptr0);
            sums[1] += *sptr * float16_to_float32(*kptr1);
            sums[2] += *sptr * float16_to_float32(*kptr2);
            sums[3] += *sptr * float16_to_float32(*kptr3);
#endif

            sptr++;
            kptr0++;
            kptr1++;
            kptr2++;
            kptr3++;
        }

        float32x4_t _sum = vld1q_f32(sums);

        _sum = activation_ps(_sum, activation_type, activation_params);

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        __fp16* outptr = (__fp16*)top_blob;
        vst1_f16(outptr + p, vcvt_f16_f32(_sum));
#else
        float* outptr = top_blob;
        vst1q_f32(outptr + p, _sum);
#endif
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_num_output_start; p < num_output; p++)
    {
        float sum = 0.f;

        if (bias_data_ptr)
            sum = bias_data_ptr[p];

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        const __fp16* sptr = bottom_blob;
        const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);
#else
        const float* sptr = bottom_blob;
        const unsigned short* kptr = weight_data_fp16.row<const unsigned short>(p);
#endif

        int i = 0;

        float32x4_t _sum = vdupq_n_f32(0.f);
        for (; i + 3 < num_input; i += 4)
        {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr));
            float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr));
#else
            float32x4_t _val = vld1q_f32(sptr);
            float32x4_t _w = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(kptr)));
#endif
            _sum = vfmaq_f32(_sum, _val, _w);

            sptr += 4;
            kptr += 4;
        }
        for (; i < num_input; i++)
        {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            sum += (float)(*sptr) * (float)(*kptr);
#else
            sum += *sptr * float16_to_float32(*kptr);
#endif
            sptr++;
            kptr++;
        }

#if __aarch64__
        sum += vaddvq_f32(_sum);
#else
        float32x2_t _sumss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
        _sumss = vpadd_f32(_sumss, _sumss);
        sum += vget_lane_f32(_sumss, 0);
#endif // __aarch64__

        sum = activation_ss(sum, activation_type, activation_params);

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        __fp16* outptr = (__fp16*)top_blob;
        outptr[p] = (__fp16)sum;
#else
        float* outptr = top_blob;
        outptr[p] = sum;
#endif
    }
#else  // (__ARM_FP & 2)
    (void)bottom_blob;
    (void)top_blob;
    (void)weight_data_fp16;
    (void)bias_data;
    (void)activation_type;
    (void)activation_params;
    (void)opt;
#endif // (__ARM_FP & 2)
}

static void innerproduct_transform_kernel_fp16s_neon(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_asimdhp())
    {
        innerproduct_transform_kernel_fp16s_neon_asimdhp(weight_data, weight_data_tm, num_input, num_output, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_VFPV4 && __ARM_NEON && !(__ARM_FP & 2)
    if (ncnn::cpu_support_arm_vfpv4())
    {
        innerproduct_transform_kernel_fp16s_neon_vfpv4(weight_data, weight_data_tm, num_input, num_output, opt);
        return;
    }
#endif

#if (__ARM_FP & 2)
    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        out_elempack = opt.use_fp16_arithmetic && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }

    Mat weight_data_fp16;
    ncnn::cast_float32_to_float16(weight_data, weight_data_fp16, opt);

    // src = inch-outch
    // dst = pb-inch-outch/pb
    {
        Mat weight_data_r2 = weight_data_fp16.reshape(num_input, num_output);

        weight_data_tm.create(num_input, num_output / out_elempack, (size_t)2u * out_elempack, out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            unsigned short* g0 = weight_data_tm.row<unsigned short>(q / out_elempack);

            for (int p = 0; p < num_input; p++)
            {
                for (int j = 0; j < out_elempack; j++)
                {
                    *g0++ = weight_data_r2.row<const unsigned short>(q + j)[p];
                }
            }
        }
    }
#else  // (__ARM_FP & 2)
    (void)weight_data;
    (void)weight_data_tm;
    (void)num_input;
    (void)num_output;
    (void)opt;
#endif // (__ARM_FP & 2)
}
