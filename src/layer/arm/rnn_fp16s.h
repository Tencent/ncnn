// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static int rnn_fp16sa(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // num_output
    Mat gates(num_output, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    // unroll
    for (int t = 0; t < T; t++)
    {
        int ti = reverse ? T - 1 - t : t;

        const __fp16* x = bottom_blob.row<const __fp16>(ti);

        int nn_num_output = num_output >> 3;
        int remain_num_output_start = nn_num_output << 3;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 8;

            const __fp16* weight_xc_ptr = weight_xc.row<const __fp16>(q / 8);
            const __fp16* weight_hc_ptr = weight_hc.row<const __fp16>(q / 8);

            float16x8_t _rnn_H = vld1q_f16((const __fp16*)bias_c + q);
            float16x8_t _sum1 = vdupq_n_f16(0.f);
            float16x8_t _sum2 = vdupq_n_f16(0.f);
            float16x8_t _sum3 = vdupq_n_f16(0.f);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float16x4_t _x = vld1_f16(x + i);
                float16x8_t _weight_xc = vld1q_f16(weight_xc_ptr);
                float16x8_t _weight_xc_1 = vld1q_f16(weight_xc_ptr + 8);
                float16x8_t _weight_xc_2 = vld1q_f16(weight_xc_ptr + 16);
                float16x8_t _weight_xc_3 = vld1q_f16(weight_xc_ptr + 24);
                _rnn_H = vfmaq_lane_f16(_rnn_H, _weight_xc, _x, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _weight_xc_1, _x, 1);
                _sum2 = vfmaq_lane_f16(_sum2, _weight_xc_2, _x, 2);
                _sum3 = vfmaq_lane_f16(_sum3, _weight_xc_3, _x, 3);

                weight_xc_ptr += 32;
            }
            for (; i < size; i++)
            {
                float16x8_t _x = vdupq_n_f16(x[i]);
                float16x8_t _weight_xc = vld1q_f16(weight_xc_ptr);
                _rnn_H = vfmaq_f16(_rnn_H, _weight_xc, _x);

                weight_xc_ptr += 8;
            }

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                float16x4_t _hidden_state = vcvt_f16_f32(vld1q_f32((const float*)hidden_state + i));
                float16x8_t _weight_hc = vld1q_f16(weight_hc_ptr);
                float16x8_t _weight_hc_1 = vld1q_f16(weight_hc_ptr + 8);
                float16x8_t _weight_hc_2 = vld1q_f16(weight_hc_ptr + 16);
                float16x8_t _weight_hc_3 = vld1q_f16(weight_hc_ptr + 24);
                _rnn_H = vfmaq_lane_f16(_rnn_H, _weight_hc, _hidden_state, 0);
                _sum1 = vfmaq_lane_f16(_sum1, _weight_hc_1, _hidden_state, 1);
                _sum2 = vfmaq_lane_f16(_sum2, _weight_hc_2, _hidden_state, 2);
                _sum3 = vfmaq_lane_f16(_sum3, _weight_hc_3, _hidden_state, 3);

                weight_hc_ptr += 32;
            }
            for (; i < num_output; i++)
            {
                float16x8_t _hidden_state = vdupq_n_f16((__fp16)hidden_state[i]);
                float16x8_t _weight_hc = vld1q_f16(weight_hc_ptr);
                _rnn_H = vfmaq_f16(_rnn_H, _weight_hc, _hidden_state);

                weight_hc_ptr += 8;
            }

            _rnn_H = vaddq_f16(_rnn_H, _sum1);
            _sum2 = vaddq_f16(_sum2, _sum3);
            _rnn_H = vaddq_f16(_rnn_H, _sum2);

            float32x4_t _H32low = tanh_ps(vcvt_f32_f16(vget_low_f16(_rnn_H)));
            float32x4_t _H32high = tanh_ps(vcvt_f32_f16(vget_high_f16(_rnn_H)));

            vst1q_f32((float*)gates + q, _H32low);
            vst1q_f32((float*)gates + q + 4, _H32high);
        }
        nn_num_output = (num_output - remain_num_output_start) >> 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = remain_num_output_start + qq * 4;

            const __fp16* weight_xc_ptr = weight_xc.row<const __fp16>(q / 8 + (q % 8) / 4);
            const __fp16* weight_hc_ptr = weight_hc.row<const __fp16>(q / 8 + (q % 8) / 4);

            float16x4_t _rnn_H = vld1_f16((const __fp16*)bias_c + q);
            float16x4_t _sum1 = vdup_n_f16(0.f);
            float16x4_t _sum2 = vdup_n_f16(0.f);
            float16x4_t _sum3 = vdup_n_f16(0.f);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float16x4_t _x = vld1_f16(x + i);
                float16x4_t _weight_xc = vld1_f16(weight_xc_ptr);
                float16x4_t _weight_xc_1 = vld1_f16(weight_xc_ptr + 4);
                float16x4_t _weight_xc_2 = vld1_f16(weight_xc_ptr + 8);
                float16x4_t _weight_xc_3 = vld1_f16(weight_xc_ptr + 12);
                _rnn_H = vfma_lane_f16(_rnn_H, _weight_xc, _x, 0);
                _sum1 = vfma_lane_f16(_sum1, _weight_xc_1, _x, 1);
                _sum2 = vfma_lane_f16(_sum2, _weight_xc_2, _x, 2);
                _sum3 = vfma_lane_f16(_sum3, _weight_xc_3, _x, 3);

                weight_xc_ptr += 16;
            }
            for (; i < size; i++)
            {
                float16x4_t _x = vdup_n_f16(x[i]);
                float16x4_t _weight_xc = vld1_f16(weight_xc_ptr);
                _rnn_H = vfma_f16(_rnn_H, _weight_xc, _x);

                weight_xc_ptr += 4;
            }

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                float16x4_t _hidden_state = vcvt_f16_f32(vld1q_f32((const float*)hidden_state + i));
                float16x4_t _weight_hc = vld1_f16(weight_hc_ptr);
                float16x4_t _weight_hc_1 = vld1_f16(weight_hc_ptr + 4);
                float16x4_t _weight_hc_2 = vld1_f16(weight_hc_ptr + 8);
                float16x4_t _weight_hc_3 = vld1_f16(weight_hc_ptr + 12);
                _rnn_H = vfma_lane_f16(_rnn_H, _weight_hc, _hidden_state, 0);
                _sum1 = vfma_lane_f16(_sum1, _weight_hc_1, _hidden_state, 1);
                _sum2 = vfma_lane_f16(_sum2, _weight_hc_2, _hidden_state, 2);
                _sum3 = vfma_lane_f16(_sum3, _weight_hc_3, _hidden_state, 3);

                weight_hc_ptr += 16;
            }
            for (; i < num_output; i++)
            {
                float16x4_t _hidden_state = vdup_n_f16((__fp16)hidden_state[i]);
                float16x4_t _weight_hc = vld1_f16(weight_hc_ptr);
                _rnn_H = vfma_f16(_rnn_H, _weight_hc, _hidden_state);

                weight_hc_ptr += 4;
            }

            _rnn_H = vadd_f16(_rnn_H, _sum1);
            _sum2 = vadd_f16(_sum2, _sum3);
            _rnn_H = vadd_f16(_rnn_H, _sum2);

            float32x4_t _H32 = tanh_ps(vcvt_f32_f16(_rnn_H));

            vst1q_f32((float*)gates + q, _H32);
        }
        remain_num_output_start += nn_num_output << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            const __fp16* weight_xc_ptr = weight_xc.row<const __fp16>(q / 8 + (q % 8) / 4 + q % 4);
            const __fp16* weight_hc_ptr = weight_hc.row<const __fp16>(q / 8 + (q % 8) / 4 + q % 4);

            __fp16 H = ((const __fp16*)bias_c)[q];

            for (int i = 0; i < size; i++)
            {
                H += weight_xc_ptr[i] * x[i];
            }

            for (int i = 0; i < num_output; i++)
            {
                H += weight_hc_ptr[i] * (__fp16)hidden_state[i];
            }

            float H32 = tanhf((float)H);

            gates[q] = H32;
        }

        __fp16* output_data = top_blob.row<__fp16>(ti);

        float* hidden_ptr = hidden_state;

        nn_num_output = num_output >> 2;
        remain_num_output_start = nn_num_output << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            float32x4_t _rnn_H = vld1q_f32((float*)gates + q);

            vst1q_f32(hidden_ptr + q, _rnn_H);
            vst1_f16(output_data + q, vcvt_f16_f32(_rnn_H));
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            float H = gates[q];

            hidden_ptr[q] = H;
            output_data[q] = (__fp16)H;
        }
    }

    return 0;
}

static int rnn_fp16s(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, Mat& hidden_state, const Option& opt)
{
    if (opt.use_fp16_arithmetic)
        return rnn_fp16sa(bottom_blob, top_blob, reverse, weight_xc, bias_c, weight_hc, hidden_state, opt);

    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // num_output
    Mat gates(num_output, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    // unroll
    for (int t = 0; t < T; t++)
    {
        int ti = reverse ? T - 1 - t : t;

        const __fp16* x = bottom_blob.row<const __fp16>(ti);

        int nn_num_output = num_output >> 2;
        int remain_num_output_start = nn_num_output << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            const __fp16* weight_xc_ptr = weight_xc.row<const __fp16>(q / 4);
            const __fp16* weight_hc_ptr = weight_hc.row<const __fp16>(q / 4);

            float32x4_t _rnn_H = vcvt_f32_f16(vld1_f16((const __fp16*)bias_c + q));
            float32x4_t _sum1 = vdupq_n_f32(0.f);
            float32x4_t _sum2 = vdupq_n_f32(0.f);
            float32x4_t _sum3 = vdupq_n_f32(0.f);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _x = vcvt_f32_f16(vld1_f16(x + i));
                float32x4_t _weight_xc = vcvt_f32_f16(vld1_f16(weight_xc_ptr));
                float32x4_t _weight_xc_1 = vcvt_f32_f16(vld1_f16(weight_xc_ptr + 4));
                float32x4_t _weight_xc_2 = vcvt_f32_f16(vld1_f16(weight_xc_ptr + 8));
                float32x4_t _weight_xc_3 = vcvt_f32_f16(vld1_f16(weight_xc_ptr + 12));
                _rnn_H = vfmaq_laneq_f32(_rnn_H, _weight_xc, _x, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_xc_1, _x, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_xc_2, _x, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_xc_3, _x, 3);

                weight_xc_ptr += 16;
            }
            for (; i < size; i++)
            {
                float32x4_t _x = vcvt_f32_f16(vdup_n_f16(x[i]));
                float32x4_t _weight_xc = vcvt_f32_f16(vld1_f16(weight_xc_ptr));
                _rnn_H = vfmaq_f32(_rnn_H, _weight_xc, _x);

                weight_xc_ptr += 4;
            }

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                float32x4_t _hidden_state = vld1q_f32((const float*)hidden_state + i);
                float32x4_t _weight_hc = vcvt_f32_f16(vld1_f16(weight_hc_ptr));
                float32x4_t _weight_hc_1 = vcvt_f32_f16(vld1_f16(weight_hc_ptr + 4));
                float32x4_t _weight_hc_2 = vcvt_f32_f16(vld1_f16(weight_hc_ptr + 8));
                float32x4_t _weight_hc_3 = vcvt_f32_f16(vld1_f16(weight_hc_ptr + 12));
                _rnn_H = vfmaq_laneq_f32(_rnn_H, _weight_hc, _hidden_state, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _weight_hc_1, _hidden_state, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _weight_hc_2, _hidden_state, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _weight_hc_3, _hidden_state, 3);

                weight_hc_ptr += 16;
            }
            for (; i < num_output; i++)
            {
                float32x4_t _hidden_state = vdupq_n_f32(hidden_state[i]);
                float32x4_t _weight_hc = vcvt_f32_f16(vld1_f16(weight_hc_ptr));
                _rnn_H = vfmaq_f32(_rnn_H, _weight_hc, _hidden_state);

                weight_hc_ptr += 4;
            }

            _rnn_H = vaddq_f32(_rnn_H, _sum1);
            _sum2 = vaddq_f32(_sum2, _sum3);
            _rnn_H = vaddq_f32(_rnn_H, _sum2);

            _rnn_H = tanh_ps(_rnn_H);

            vst1q_f32((float*)gates + q, _rnn_H);
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            const __fp16* weight_xc_ptr = weight_xc.row<const __fp16>(q / 4 + q % 4);
            const __fp16* weight_hc_ptr = weight_hc.row<const __fp16>(q / 4 + q % 4);

            float H = (float)(((const __fp16*)bias_c)[q]);

            for (int i = 0; i < size; i++)
            {
                H += (float)weight_xc_ptr[i] * (float)x[i];
            }

            for (int i = 0; i < num_output; i++)
            {
                H += (float)weight_hc_ptr[i] * hidden_state[i];
            }

            H = tanhf(H);

            gates[q] = H;
        }

        __fp16* output_data = top_blob.row<__fp16>(ti);

        float* hidden_ptr = hidden_state;

        nn_num_output = num_output >> 2;
        remain_num_output_start = nn_num_output << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            float32x4_t _rnn_H = vld1q_f32((float*)gates + q);

            vst1q_f32(hidden_ptr + q, _rnn_H);
            vst1_f16(output_data + q, vcvt_f16_f32(_rnn_H));
        }
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            float H = gates[q];

            hidden_ptr[q] = H;
            output_data[q] = (__fp16)H;
        }
    }

    return 0;
}

static void rnn_transform_kernel_fp16s(const Mat& weight_xc, const Mat& weight_hc, Mat& weight_xc_data_packed_dr, Mat& weight_hc_data_packed_dr, int size, int num_output, bool use_fp16_arithmetic)
{
    int q = 0;
    if (use_fp16_arithmetic)
    {
        for (; q + 7 < num_output; q += 8)
        {
            const float* weight_xc_0 = weight_xc.row(q);
            const float* weight_xc_1 = weight_xc.row(q + 1);
            const float* weight_xc_2 = weight_xc.row(q + 2);
            const float* weight_xc_3 = weight_xc.row(q + 3);
            const float* weight_xc_4 = weight_xc.row(q + 4);
            const float* weight_xc_5 = weight_xc.row(q + 5);
            const float* weight_xc_6 = weight_xc.row(q + 6);
            const float* weight_xc_7 = weight_xc.row(q + 7);

            const float* weight_hc_0 = weight_hc.row(q);
            const float* weight_hc_1 = weight_hc.row(q + 1);
            const float* weight_hc_2 = weight_hc.row(q + 2);
            const float* weight_hc_3 = weight_hc.row(q + 3);
            const float* weight_hc_4 = weight_hc.row(q + 4);
            const float* weight_hc_5 = weight_hc.row(q + 5);
            const float* weight_hc_6 = weight_hc.row(q + 6);
            const float* weight_hc_7 = weight_hc.row(q + 7);

            __fp16* weight_xc = weight_xc_data_packed_dr.row<__fp16>(q / 8);
            __fp16* weight_hc = weight_hc_data_packed_dr.row<__fp16>(q / 8);

            for (int i = 0; i < size; i++)
            {
                weight_xc[0] = (__fp16)weight_xc_0[i];
                weight_xc[1] = (__fp16)weight_xc_1[i];
                weight_xc[2] = (__fp16)weight_xc_2[i];
                weight_xc[3] = (__fp16)weight_xc_3[i];
                weight_xc[4] = (__fp16)weight_xc_4[i];
                weight_xc[5] = (__fp16)weight_xc_5[i];
                weight_xc[6] = (__fp16)weight_xc_6[i];
                weight_xc[7] = (__fp16)weight_xc_7[i];

                weight_xc += 8;
            }

            for (int i = 0; i < num_output; i++)
            {
                weight_hc[0] = (__fp16)weight_hc_0[i];
                weight_hc[1] = (__fp16)weight_hc_1[i];
                weight_hc[2] = (__fp16)weight_hc_2[i];
                weight_hc[3] = (__fp16)weight_hc_3[i];
                weight_hc[4] = (__fp16)weight_hc_4[i];
                weight_hc[5] = (__fp16)weight_hc_5[i];
                weight_hc[6] = (__fp16)weight_hc_6[i];
                weight_hc[7] = (__fp16)weight_hc_7[i];

                weight_hc += 8;
            }
        }
    }
    for (; q + 3 < num_output; q += 4)
    {
        const float* weight_xc_0 = weight_xc.row(q);
        const float* weight_xc_1 = weight_xc.row(q + 1);
        const float* weight_xc_2 = weight_xc.row(q + 2);
        const float* weight_xc_3 = weight_xc.row(q + 3);

        const float* weight_hc_0 = weight_hc.row(q);
        const float* weight_hc_1 = weight_hc.row(q + 1);
        const float* weight_hc_2 = weight_hc.row(q + 2);
        const float* weight_hc_3 = weight_hc.row(q + 3);

        __fp16* weight_xc = use_fp16_arithmetic ? weight_xc_data_packed_dr.row<__fp16>(q / 8 + (q % 8) / 4) : weight_xc_data_packed_dr.row<__fp16>(q / 4);
        __fp16* weight_hc = use_fp16_arithmetic ? weight_hc_data_packed_dr.row<__fp16>(q / 8 + (q % 8) / 4) : weight_hc_data_packed_dr.row<__fp16>(q / 4);

        for (int i = 0; i < size; i++)
        {
            weight_xc[0] = (__fp16)weight_xc_0[i];
            weight_xc[1] = (__fp16)weight_xc_1[i];
            weight_xc[2] = (__fp16)weight_xc_2[i];
            weight_xc[3] = (__fp16)weight_xc_3[i];

            weight_xc += 4;
        }

        for (int i = 0; i < num_output; i++)
        {
            weight_hc[0] = (__fp16)weight_hc_0[i];
            weight_hc[1] = (__fp16)weight_hc_1[i];
            weight_hc[2] = (__fp16)weight_hc_2[i];
            weight_hc[3] = (__fp16)weight_hc_3[i];

            weight_hc += 4;
        }
    }
    for (; q < num_output; q++)
    {
        const float* weight_xc_0 = weight_xc.row(q);
        const float* weight_hc_0 = weight_hc.row(q);

        __fp16* weight_xc = use_fp16_arithmetic ? weight_xc_data_packed_dr.row<__fp16>(q / 8 + (q % 8) / 4 + q % 4) : weight_xc_data_packed_dr.row<__fp16>(q / 4 + q % 4);
        __fp16* weight_hc = use_fp16_arithmetic ? weight_hc_data_packed_dr.row<__fp16>(q / 8 + (q % 8) / 4 + q % 4) : weight_hc_data_packed_dr.row<__fp16>(q / 4 + q % 4);

        for (int i = 0; i < size; i++)
        {
            weight_xc[i] = (__fp16)weight_xc_0[i];
        }

        for (int i = 0; i < num_output; i++)
        {
            weight_hc[i] = (__fp16)weight_hc_0[i];
        }
    }
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
