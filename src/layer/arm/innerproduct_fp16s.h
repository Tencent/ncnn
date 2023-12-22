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

#if !(__ARM_FEATURE_FP16_FML || __ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#if NCNN_RUNTIME_CPU && NCNN_ARM82FP16FML && __aarch64__ && !__ARM_FEATURE_FP16_FML
void innerproduct_pack4_fp16s_neon_asimdfhm(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
void innerproduct_fp16s_neon_asimdfhm(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
void innerproduct_transform_kernel_fp16s_neon_asimdfhm(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void innerproduct_pack4_fp16s_neon_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
void innerproduct_fp16s_neon_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
void innerproduct_transform_kernel_fp16s_neon_asimdhp(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt);
#endif
#endif

static void innerproduct_pack4_fp16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
#if !(__ARM_FEATURE_FP16_FML || __ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#if NCNN_RUNTIME_CPU && NCNN_ARM82FP16FML && __aarch64__ && !__ARM_FEATURE_FP16_FML
    if (ncnn::cpu_support_arm_asimdfhm())
    {
        innerproduct_pack4_fp16s_neon_asimdfhm(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_asimdhp())
    {
        innerproduct_pack4_fp16s_neon_asimdhp(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
        return;
    }
#endif
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

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        const __fp16* sptr = bottom_blob;
        const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);
#else
        const float* sptr = bottom_blob;
        const unsigned short* kptr = weight_data_fp16.row<const unsigned short>(p);
#endif

        int i = 0;
#if NCNN_GNU_INLINE_ASM
        for (; i + 7 < num_input; i += 8)
        {
#if __aarch64__
#if __ARM_FEATURE_FP16_FML
            asm volatile(
                "prfm   pldl1keep, [%0, #128]       \n"
                "ld1    {v0.8h}, [%0], #16          \n"
                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v2.8h, v3.8h, v4.8h, v5.8h}, [%1], #64 \n"
                "fmlal  %2.4s, v2.4h, v0.h[0]       \n"
                "fmlal2 %3.4s, v2.4h, v0.h[1]       \n"
                "fmlal  %4.4s, v3.4h, v0.h[2]       \n"
                "fmlal2 %5.4s, v3.4h, v0.h[3]       \n"
                "fmlal  %2.4s, v4.4h, v0.h[4]       \n"
                "fmlal2 %3.4s, v4.4h, v0.h[5]       \n"
                "fmlal  %4.4s, v5.4h, v0.h[6]       \n"
                "fmlal2 %5.4s, v5.4h, v0.h[7]       \n"
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
                : "cc", "memory", "v0", "v2", "v3", "v4", "v5");
#elif __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
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
#endif // NCNN_GNU_INLINE_ASM
        for (; i + 3 < num_input; i += 4)
        {
#if __ARM_FEATURE_FP16_FML
            float16x4_t _val = vld1_f16(sptr);
            float16x8_t _w01 = vld1q_f16(kptr);
            float16x8_t _w23 = vld1q_f16(kptr + 8);

            _sum0 = vfmlalq_lane_low_f16(_sum0, _w01, _val, 0);
            _sum1 = vfmlalq_lane_high_f16(_sum1, _w01, _val, 1);
            _sum2 = vfmlalq_lane_low_f16(_sum2, _w23, _val, 2);
            _sum3 = vfmlalq_lane_high_f16(_sum3, _w23, _val, 3);
#else // __ARM_FEATURE_FP16_FML
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
            float32x4_t _w0 = vcvt_f32_f16((float16x4_t)(vget_low_u16(_w01)));
            float32x4_t _w1 = vcvt_f32_f16((float16x4_t)(vget_high_u16(_w01)));
            float32x4_t _w2 = vcvt_f32_f16((float16x4_t)(vget_low_u16(_w23)));
            float32x4_t _w3 = vcvt_f32_f16((float16x4_t)(vget_high_u16(_w23)));
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
#endif // __ARM_FEATURE_FP16_FML

            sptr += 4;
            kptr += 16;
        }
        for (; i < num_input; i++)
        {
            float32x4_t _val = vdupq_n_f32((float)sptr[0]);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr));
#else
            float32x4_t _w = vcvt_f32_f16((float16x4_t)(vld1_u16(kptr)));
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
}

static void innerproduct_fp16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
#if !(__ARM_FEATURE_FP16_FML || __ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#if NCNN_RUNTIME_CPU && NCNN_ARM82FP16FML && __aarch64__ && !__ARM_FEATURE_FP16_FML
    if (ncnn::cpu_support_arm_asimdfhm())
    {
        innerproduct_fp16s_neon_asimdfhm(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_asimdhp())
    {
        innerproduct_fp16s_neon_asimdhp(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
        return;
    }
#endif
#endif

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
#if __ARM_FEATURE_FP16_FML
            float16x4_t _val = vld1_f16(sptr);
            float16x4_t _w0 = vld1_f16(kptr0);
            float16x4_t _w1 = vld1_f16(kptr1);
            float16x4_t _w2 = vld1_f16(kptr2);
            float16x4_t _w3 = vld1_f16(kptr3);
            float16x8_t _w01 = vcombine_f16(_w0, _w1);
            float16x8_t _w23 = vcombine_f16(_w2, _w3);
            float16x8_t _valval = vcombine_f16(_val, _val);

            _sum0 = vfmlalq_low_f16(_sum0, _w01, _valval);
            _sum1 = vfmlalq_high_f16(_sum1, _w01, _valval);
            _sum2 = vfmlalq_low_f16(_sum2, _w23, _valval);
            _sum3 = vfmlalq_high_f16(_sum3, _w23, _valval);
#else // __ARM_FEATURE_FP16_FML
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr));
            float32x4_t _w0 = vcvt_f32_f16(vld1_f16(kptr0));
            float32x4_t _w1 = vcvt_f32_f16(vld1_f16(kptr1));
            float32x4_t _w2 = vcvt_f32_f16(vld1_f16(kptr2));
            float32x4_t _w3 = vcvt_f32_f16(vld1_f16(kptr3));
#else
            float32x4_t _val = vld1q_f32(sptr);
            float32x4_t _w0 = vcvt_f32_f16((float16x4_t)(vld1_u16(kptr0)));
            float32x4_t _w1 = vcvt_f32_f16((float16x4_t)(vld1_u16(kptr1)));
            float32x4_t _w2 = vcvt_f32_f16((float16x4_t)(vld1_u16(kptr2)));
            float32x4_t _w3 = vcvt_f32_f16((float16x4_t)(vld1_u16(kptr3)));
#endif

            _sum0 = vfmaq_f32(_sum0, _val, _w0);
            _sum1 = vfmaq_f32(_sum1, _val, _w1);
            _sum2 = vfmaq_f32(_sum2, _val, _w2);
            _sum3 = vfmaq_f32(_sum3, _val, _w3);
#endif // __ARM_FEATURE_FP16_FML

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
            float32x4_t _w = vcvt_f32_f16((float16x4_t)(vld1_u16(kptr)));
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
}

static void innerproduct_transform_kernel_fp16s_neon(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt)
{
#if !(__ARM_FEATURE_FP16_FML || __ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#if NCNN_RUNTIME_CPU && NCNN_ARM82FP16FML && __aarch64__ && !__ARM_FEATURE_FP16_FML
    if (ncnn::cpu_support_arm_asimdfhm())
    {
        innerproduct_transform_kernel_fp16s_neon_asimdfhm(weight_data, weight_data_tm, num_input, num_output, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_asimdhp())
    {
        innerproduct_transform_kernel_fp16s_neon_asimdhp(weight_data, weight_data_tm, num_input, num_output, opt);
        return;
    }
#endif
#endif

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        out_elempack = opt.use_fp16_arithmetic && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }

    // src = inch-outch
    // dst = pb-inch-outch/pb
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
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
#if NCNN_GNU_INLINE_ASM
            for (; p + 7 < num_input; p += 8)
            {
                // transpose 8x8
                asm volatile(
                    "ld1    {v0.4s, v1.4s}, [%0], #32   \n"
                    "ld1    {v2.4s, v3.4s}, [%1], #32   \n"
                    "ld1    {v4.4s, v5.4s}, [%2], #32   \n"
                    "ld1    {v6.4s, v7.4s}, [%3], #32   \n"
                    "ld1    {v8.4s, v9.4s}, [%4], #32   \n"
                    "ld1    {v10.4s, v11.4s}, [%5], #32 \n"
                    "ld1    {v12.4s, v13.4s}, [%6], #32 \n"
                    "ld1    {v14.4s, v15.4s}, [%7], #32 \n"

                    "fcvtn  v0.4h, v0.4s            \n"
                    "fcvtn2 v0.8h, v1.4s            \n"
                    "fcvtn  v1.4h, v2.4s            \n"
                    "fcvtn2 v1.8h, v3.4s            \n"
                    "fcvtn  v2.4h, v4.4s            \n"
                    "fcvtn2 v2.8h, v5.4s            \n"
                    "fcvtn  v3.4h, v6.4s            \n"
                    "fcvtn2 v3.8h, v7.4s            \n"
                    "fcvtn  v4.4h, v8.4s            \n"
                    "fcvtn2 v4.8h, v9.4s            \n"
                    "fcvtn  v5.4h, v10.4s           \n"
                    "fcvtn2 v5.8h, v11.4s           \n"
                    "fcvtn  v6.4h, v12.4s           \n"
                    "fcvtn2 v6.8h, v13.4s           \n"
                    "fcvtn  v7.4h, v14.4s           \n"
                    "fcvtn2 v7.8h, v15.4s           \n"

                    "zip1   v16.8h, v0.8h, v4.8h    \n"
                    "zip2   v20.8h, v0.8h, v4.8h    \n"
                    "zip1   v17.8h, v1.8h, v5.8h    \n"
                    "zip2   v21.8h, v1.8h, v5.8h    \n"
                    "zip1   v18.8h, v2.8h, v6.8h    \n"
                    "zip2   v22.8h, v2.8h, v6.8h    \n"
                    "zip1   v19.8h, v3.8h, v7.8h    \n"
                    "zip2   v23.8h, v3.8h, v7.8h    \n"

                    "st4    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"
                    "st4    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"
                    : "=r"(k0), // %0
                    "=r"(k1), // %1
                    "=r"(k2), // %2
                    "=r"(k3), // %3
                    "=r"(k4), // %4
                    "=r"(k5), // %5
                    "=r"(k6), // %6
                    "=r"(k7), // %7
                    "=r"(g0)  // %8
                    : "0"(k0),
                    "1"(k1),
                    "2"(k2),
                    "3"(k3),
                    "4"(k4),
                    "5"(k5),
                    "6"(k6),
                    "7"(k7),
                    "8"(g0)
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
            }
#endif // NCNN_GNU_INLINE_ASM
            for (; p < num_input; p++)
            {
                g0[0] = float32_to_float16(*k0++);
                g0[1] = float32_to_float16(*k1++);
                g0[2] = float32_to_float16(*k2++);
                g0[3] = float32_to_float16(*k3++);
                g0[4] = float32_to_float16(*k4++);
                g0[5] = float32_to_float16(*k5++);
                g0[6] = float32_to_float16(*k6++);
                g0[7] = float32_to_float16(*k7++);
                g0 += 8;
            }
        }
    }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

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
                // transpose 4x4
                uint16x4x4_t _p;
                _p.val[0] = (uint16x4_t)(vcvt_f16_f32(vld1q_f32(k0)));
                _p.val[1] = (uint16x4_t)(vcvt_f16_f32(vld1q_f32(k1)));
                _p.val[2] = (uint16x4_t)(vcvt_f16_f32(vld1q_f32(k2)));
                _p.val[3] = (uint16x4_t)(vcvt_f16_f32(vld1q_f32(k3)));
                vst4_u16(g0, _p);

                k0 += 4;
                k1 += 4;
                k2 += 4;
                k3 += 4;
                g0 += 16;
            }
            for (; p < num_input; p++)
            {
                g0[0] = float32_to_float16(*k0++);
                g0[1] = float32_to_float16(*k1++);
                g0[2] = float32_to_float16(*k2++);
                g0[3] = float32_to_float16(*k3++);
                g0 += 4;
            }
        }
    }

    if (out_elempack == 1)
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);
        ncnn::cast_float32_to_float16(weight_data_r2, weight_data_tm, opt);
    }
}
