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
void innerproduct_gemm_fp16s_neon_asimdfhm(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void innerproduct_gemm_fp16s_neon_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
#endif
#endif

static void innerproduct_gemm_fp16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
#if !(__ARM_FEATURE_FP16_FML || __ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#if NCNN_RUNTIME_CPU && NCNN_ARM82FP16FML && __aarch64__ && !__ARM_FEATURE_FP16_FML
    if (ncnn::cpu_support_arm_asimdfhm())
    {
        innerproduct_gemm_fp16s_neon_asimdfhm(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82 && __aarch64__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_asimdhp())
    {
        innerproduct_gemm_fp16s_neon_asimdhp(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
        return;
    }
#endif
#endif

    const int num_input = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int num_output = top_blob.w;
    const int h = bottom_blob.h;

    const float* bias_data_ptr = bias_data;

    int num_output_elempack = 1;
    if (opt.use_packing_layout)
    {
        num_output_elempack = num_output % 4 == 0 ? 4 : 1;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int j = 0; j < h; j++)
    {
        if (elempack == 4 && num_output_elempack == 4)
        {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            __fp16* outptr = top_blob.row<__fp16>(j);
#else
            float* outptr = top_blob.row(j);
#endif

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                const __fp16* m = bottom_blob.row<const __fp16>(j);
                const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);
#else
                const float* m = bottom_blob.row(j);
                const unsigned short* kptr = weight_data_fp16.row<const unsigned short>(p);
#endif

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
                for (; i < num_input; i++)
                {
#if __ARM_FEATURE_FP16_FML
                    float16x4_t _val = vld1_f16(m);
                    float16x4_t _w = vld1_f16(kptr);
                    float16x8_t _valval = vcombine_f16(_val, _val);

                    _sum0 = vfmlalq_lane_low_f16(_sum0, _valval, _w, 0);
                    _sum1 = vfmlalq_lane_low_f16(_sum1, _valval, _w, 1);
                    _sum2 = vfmlalq_lane_low_f16(_sum2, _valval, _w, 2);
                    _sum3 = vfmlalq_lane_low_f16(_sum3, _valval, _w, 3);
#else // __ARM_FEATURE_FP16_FML
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    float32x4_t _val = vcvt_f32_f16(vld1_f16(m));
                    float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr));
#else
                    float32x4_t _val = vld1q_f32(m);
                    float32x4_t _w = vcvt_f32_f16((float16x4_t)(vld1_u16(kptr)));
#endif

#if __aarch64__
                    _sum0 = vfmaq_laneq_f32(_sum0, _val, _w, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _val, _w, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _val, _w, 2);
                    _sum3 = vfmaq_laneq_f32(_sum3, _val, _w, 3);
#else
                    _sum0 = vmlaq_lane_f32(_sum0, _val, vget_low_f32(_w), 0);
                    _sum1 = vmlaq_lane_f32(_sum1, _val, vget_low_f32(_w), 1);
                    _sum2 = vmlaq_lane_f32(_sum2, _val, vget_high_f32(_w), 0);
                    _sum3 = vmlaq_lane_f32(_sum3, _val, vget_high_f32(_w), 1);
#endif
#endif // __ARM_FEATURE_FP16_FML

                    m += 4;
                    kptr += 4;
                }

                _sum0 = activation_ps(_sum0, activation_type, activation_params);
                _sum1 = activation_ps(_sum1, activation_type, activation_params);
                _sum2 = activation_ps(_sum2, activation_type, activation_params);
                _sum3 = activation_ps(_sum3, activation_type, activation_params);

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                vst1_f16(outptr, vcvt_f16_f32(_sum0));
                vst1_f16(outptr + 4, vcvt_f16_f32(_sum1));
                vst1_f16(outptr + 8, vcvt_f16_f32(_sum2));
                vst1_f16(outptr + 12, vcvt_f16_f32(_sum3));
#else
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 8, _sum2);
                vst1q_f32(outptr + 12, _sum3);
#endif
                outptr += 16;
            }
        }

        if (elempack == 1 && num_output_elempack == 4)
        {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            __fp16* outptr = top_blob.row<__fp16>(j);
#else
            float* outptr = top_blob.row(j);
#endif

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                const __fp16* m = bottom_blob.row<const __fp16>(j);
                const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);
#else
                const float* m = bottom_blob.row(j);
                const unsigned short* kptr = weight_data_fp16.row<const unsigned short>(p);
#endif

                float32x4_t _sum0 = vdupq_n_f32(0.f);

                if (bias_data_ptr)
                {
                    _sum0 = vld1q_f32(bias_data_ptr + p * 4);
                }

                float32x4_t _sum1 = vdupq_n_f32(0.f);
                float32x4_t _sum2 = vdupq_n_f32(0.f);
                float32x4_t _sum3 = vdupq_n_f32(0.f);

                int i = 0;
                for (; i + 3 < num_input; i += 4)
                {
#if __ARM_FEATURE_FP16_FML
                    float16x4_t _val = vld1_f16(m);
                    float16x8_t _w01 = vld1q_f16(kptr);
                    float16x8_t _w23 = vld1q_f16(kptr + 8);

                    _sum0 = vfmlalq_lane_low_f16(_sum0, _w01, _val, 0);
                    _sum1 = vfmlalq_lane_high_f16(_sum1, _w01, _val, 1);
                    _sum2 = vfmlalq_lane_low_f16(_sum2, _w23, _val, 2);
                    _sum3 = vfmlalq_lane_high_f16(_sum3, _w23, _val, 3);
#else // __ARM_FEATURE_FP16_FML
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    float32x4_t _val = vcvt_f32_f16(vld1_f16(m));
                    float16x8_t _w01 = vld1q_f16(kptr);
                    float16x8_t _w23 = vld1q_f16(kptr + 8);
                    float32x4_t _w0 = vcvt_f32_f16(vget_low_f16(_w01));
                    float32x4_t _w1 = vcvt_f32_f16(vget_high_f16(_w01));
                    float32x4_t _w2 = vcvt_f32_f16(vget_low_f16(_w23));
                    float32x4_t _w3 = vcvt_f32_f16(vget_high_f16(_w23));
#else
                    float32x4_t _val = vld1q_f32(m);
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

                    m += 4;
                    kptr += 16;
                }
                for (; i < num_input; i++)
                {
                    float32x4_t _val = vdupq_n_f32((float)m[0]);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr));
#else
                    float32x4_t _w = vcvt_f32_f16((float16x4_t)(vld1_u16(kptr)));
#endif
                    _sum0 = vfmaq_f32(_sum0, _val, _w);

                    m += 1;
                    kptr += 4;
                }

                _sum0 = vaddq_f32(_sum0, _sum1);
                _sum2 = vaddq_f32(_sum2, _sum3);
                _sum0 = vaddq_f32(_sum0, _sum2);

                _sum0 = activation_ps(_sum0, activation_type, activation_params);

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                vst1_f16(outptr, vcvt_f16_f32(_sum0));
#else
                vst1q_f32(outptr, _sum0);
#endif
                outptr += 4;
            }
        }

        if (elempack == 4 && num_output_elempack == 1)
        {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            __fp16* outptr = top_blob.row<__fp16>(j);
#else
            float* outptr = top_blob.row(j);
#endif

            for (int p = 0; p < num_output; p++)
            {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                const __fp16* m = bottom_blob.row<const __fp16>(j);
                const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);
#else
                const float* m = bottom_blob.row(j);
                const unsigned short* kptr = weight_data_fp16.row<const unsigned short>(p);
#endif

                float32x4_t _sum0 = vdupq_n_f32(0.f);
                float32x4_t _sum1 = vdupq_n_f32(0.f);
                float32x4_t _sum2 = vdupq_n_f32(0.f);
                float32x4_t _sum3 = vdupq_n_f32(0.f);

                if (bias_data_ptr)
                {
                    _sum0 = vdupq_n_f32(bias_data_ptr[p]);
                }

                int i = 0;
                for (; i + 3 < num_input; i += 4)
                {
#if __ARM_FEATURE_FP16_FML
                    float16x8_t _val01 = vld1q_f16(m);
                    float16x8_t _val23 = vld1q_f16(m + 8);
                    float16x4_t _w = vld1_f16(kptr);

                    _sum0 = vfmlalq_lane_low_f16(_sum0, _val01, _w, 0);
                    _sum1 = vfmlalq_lane_high_f16(_sum1, _val01, _w, 1);
                    _sum2 = vfmlalq_lane_low_f16(_sum2, _val23, _w, 2);
                    _sum3 = vfmlalq_lane_high_f16(_sum3, _val23, _w, 3);
#else // __ARM_FEATURE_FP16_FML
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    float32x4_t _val0 = vcvt_f32_f16(vld1_f16(m));
                    float32x4_t _val1 = vcvt_f32_f16(vld1_f16(m + 4));
                    float32x4_t _val2 = vcvt_f32_f16(vld1_f16(m + 8));
                    float32x4_t _val3 = vcvt_f32_f16(vld1_f16(m + 12));
                    float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr));
#else
                    float32x4_t _val0 = vld1q_f32(m);
                    float32x4_t _val1 = vld1q_f32(m + 4);
                    float32x4_t _val2 = vld1q_f32(m + 8);
                    float32x4_t _val3 = vld1q_f32(m + 12);
                    float32x4_t _w = vcvt_f32_f16((float16x4_t)(vld1_u16(kptr)));
#endif

#if __aarch64__
                    _sum0 = vfmaq_laneq_f32(_sum0, _val0, _w, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _val1, _w, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _val2, _w, 2);
                    _sum3 = vfmaq_laneq_f32(_sum3, _val3, _w, 3);
#else
                    _sum0 = vmlaq_lane_f32(_sum0, _val0, vget_low_f32(_w), 0);
                    _sum1 = vmlaq_lane_f32(_sum1, _val1, vget_low_f32(_w), 1);
                    _sum2 = vmlaq_lane_f32(_sum2, _val2, vget_high_f32(_w), 0);
                    _sum3 = vmlaq_lane_f32(_sum3, _val3, vget_high_f32(_w), 1);
#endif
#endif // __ARM_FEATURE_FP16_FML

                    m += 16;
                    kptr += 4;
                }
                for (; i < num_input; i++)
                {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    float32x4_t _val = vcvt_f32_f16(vld1_f16(m));
                    float32x4_t _k = vdupq_n_f32((float)(kptr[0]));
#else
                    float32x4_t _val = vld1q_f32(m);
                    float32x4_t _k = vdupq_n_f32(float16_to_float32(kptr[0]));
#endif
                    _sum0 = vfmaq_f32(_sum0, _val, _k);

                    m += 4;
                    kptr += 1;
                }

                _sum0 = vaddq_f32(_sum0, _sum1);
                _sum2 = vaddq_f32(_sum2, _sum3);
                _sum0 = vaddq_f32(_sum0, _sum2);

                _sum0 = activation_ps(_sum0, activation_type, activation_params);

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                vst1_f16(outptr, vcvt_f16_f32(_sum0));
#else
                vst1q_f32(outptr, _sum0);
#endif
                outptr += 4;
            }
        }

        if (elempack == 1 && num_output_elempack == 1)
        {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            __fp16* outptr = top_blob.row<__fp16>(j);
#else
            float* outptr = top_blob.row(j);
#endif

            for (int p = 0; p < num_output; p++)
            {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                const __fp16* m = bottom_blob.row<const __fp16>(j);
                const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);
#else
                const float* m = bottom_blob.row(j);
                const unsigned short* kptr = weight_data_fp16.row<const unsigned short>(p);
#endif

                float sum = 0.f;

                if (bias_data_ptr)
                {
                    sum = bias_data_ptr[p];
                }

                int i = 0;
                float32x4_t _sum0 = vdupq_n_f32(0.f);
                float32x4_t _sum1 = vdupq_n_f32(0.f);
                for (; i + 7 < num_input; i += 8)
                {
#if __ARM_FEATURE_FP16_FML
                    float16x8_t _val01 = vld1q_f16(m);
                    float16x8_t _w01 = vld1q_f16(kptr);

                    _sum0 = vfmlalq_low_f16(_sum0, _val01, _w01);
                    _sum1 = vfmlalq_high_f16(_sum1, _val01, _w01);
#else // __ARM_FEATURE_FP16_FML
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    float16x8_t _val01 = vld1q_f16(m);
                    float16x8_t _w01 = vld1q_f16(kptr);
                    float32x4_t _val0 = vcvt_f32_f16(vget_low_f16(_val01));
                    float32x4_t _val1 = vcvt_f32_f16(vget_high_f16(_val01));
                    float32x4_t _w0 = vcvt_f32_f16(vget_low_f16(_w01));
                    float32x4_t _w1 = vcvt_f32_f16(vget_high_f16(_w01));
#else
                    float32x4_t _val0 = vld1q_f32(m);
                    float32x4_t _val1 = vld1q_f32(m + 4);
                    uint16x8_t _w01 = vld1q_u16(kptr);
                    float32x4_t _w0 = vcvt_f32_f16((float16x4_t)(vget_low_u16(_w01)));
                    float32x4_t _w1 = vcvt_f32_f16((float16x4_t)(vget_high_u16(_w01)));
#endif

                    _sum0 = vfmaq_f32(_sum0, _val0, _w0);
                    _sum1 = vfmaq_f32(_sum1, _val1, _w1);
#endif // __ARM_FEATURE_FP16_FML

                    m += 8;
                    kptr += 8;
                }
                _sum0 = vaddq_f32(_sum0, _sum1);
                for (; i + 3 < num_input; i += 4)
                {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    float32x4_t _val = vcvt_f32_f16(vld1_f16(m));
                    float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr));
#else
                    float32x4_t _val = vld1q_f32(m);
                    float32x4_t _w = vcvt_f32_f16((float16x4_t)(vld1_u16(kptr)));
#endif

                    _sum0 = vfmaq_f32(_sum0, _val, _w);

                    m += 4;
                    kptr += 4;
                }
#if __aarch64__
                sum += vaddvq_f32(_sum0);
#else
                float32x2_t _ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                _ss = vpadd_f32(_ss, _ss);
                sum += vget_lane_f32(_ss, 0);
#endif
                for (; i < num_input; i++)
                {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    sum += (float)(*m++) * (float)(*kptr++);
#else
                    sum += *m++ * float16_to_float32(*kptr++);
#endif
                }

                sum = activation_ss(sum, activation_type, activation_params);

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                outptr[0] = (__fp16)sum;
#else
                outptr[0] = sum;
#endif
                outptr += 1;
            }
        }
    }
}
