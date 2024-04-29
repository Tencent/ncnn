// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
void rnn_transform_weight_int8_asimddp(const Mat& weight_xc, const Mat& weight_xc_int8_scales, const Mat& weight_hc, const Mat& weight_hc_int8_scales, const Mat& bias_c, Mat& weight_data_tm, Mat& weight_data_tm_int8_descales, Mat& bias_c_tm, int size, int num_output, int num_directions, const Option& opt);
void rnn_int8_asimddp(const Mat& bottom_blob_int8, const Mat& bottom_blob_int8_descales, Mat& top_blob, int elemtype, int reverse, const Mat& weight_data_tm, const Mat& weight_data_tm_int8_descales, const Mat& bias_c, Mat& hidden_state, const Option& opt);
#endif

static void rnn_transform_weight_int8(const Mat& weight_xc, const Mat& weight_xc_int8_scales, const Mat& weight_hc, const Mat& weight_hc_int8_scales, const Mat& bias_c, Mat& weight_data_tm, Mat& weight_data_tm_int8_descales, Mat& bias_c_tm, int size, int num_output, int num_directions, const Option& opt)
{
    // TODO dispatch for __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    // TODO dispatch for __ARM_FEATURE_DOTPROD

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
    if (ncnn::cpu_support_arm_asimddp())
    {
        rnn_transform_weight_int8_asimddp(weight_xc, weight_xc_int8_scales, weight_hc, weight_hc_int8_scales, bias_c, weight_data_tm, weight_data_tm_int8_descales, bias_c_tm, size, num_output, num_directions, opt);
        return;
    }
#endif

#if __ARM_NEON
    weight_data_tm.create(size * 4 + num_output * 4, num_output / 4 + num_output % 4, num_directions, 1u, 1);
    weight_data_tm_int8_descales.create(4 + 4, num_output / 4 + num_output % 4, num_directions);
#else
    weight_data_tm.create(size + num_output, num_output, num_directions, 1u, 1);
    weight_data_tm_int8_descales.create(1 + 1, num_output, num_directions);
#endif
    bias_c_tm = bias_c;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int dr = 0; dr < num_directions; dr++)
    {
        const Mat weight_xc_dr = weight_xc.channel(dr);
        const Mat weight_hc_dr = weight_hc.channel(dr);
        const float* weight_xc_int8_scales_ptr = weight_xc_int8_scales.row(dr);
        const float* weight_hc_int8_scales_ptr = weight_hc_int8_scales.row(dr);

        Mat weight_data_tm_dr = weight_data_tm.channel(dr);
        Mat weight_data_tm_int8_descales_dr = weight_data_tm_int8_descales.channel(dr);

        int q = 0;
#if __ARM_NEON
        for (; q + 3 < num_output; q += 4)
        {
            const signed char* weight_xc_0 = weight_xc_dr.row<const signed char>(q);
            const signed char* weight_xc_1 = weight_xc_dr.row<const signed char>(q + 1);
            const signed char* weight_xc_2 = weight_xc_dr.row<const signed char>(q + 2);
            const signed char* weight_xc_3 = weight_xc_dr.row<const signed char>(q + 3);

            const signed char* weight_hc_0 = weight_hc_dr.row<const signed char>(q);
            const signed char* weight_hc_1 = weight_hc_dr.row<const signed char>(q + 1);
            const signed char* weight_hc_2 = weight_hc_dr.row<const signed char>(q + 2);
            const signed char* weight_hc_3 = weight_hc_dr.row<const signed char>(q + 3);

            signed char* kptr = weight_data_tm_dr.row<signed char>(q / 4);
            float* descales_ptr = weight_data_tm_int8_descales_dr.row(q / 4);

            int i = 0;
#if __ARM_FEATURE_DOTPROD
            for (; i + 3 < size; i += 4)
            {
                kptr[0] = weight_xc_0[i];
                kptr[1] = weight_xc_0[i + 1];
                kptr[2] = weight_xc_0[i + 2];
                kptr[3] = weight_xc_0[i + 3];
                kptr[4] = weight_xc_1[i];
                kptr[5] = weight_xc_1[i + 1];
                kptr[6] = weight_xc_1[i + 2];
                kptr[7] = weight_xc_1[i + 3];
                kptr[8 + 0] = weight_xc_2[i];
                kptr[8 + 1] = weight_xc_2[i + 1];
                kptr[8 + 2] = weight_xc_2[i + 2];
                kptr[8 + 3] = weight_xc_2[i + 3];
                kptr[8 + 4] = weight_xc_3[i];
                kptr[8 + 5] = weight_xc_3[i + 1];
                kptr[8 + 6] = weight_xc_3[i + 2];
                kptr[8 + 7] = weight_xc_3[i + 3];

                kptr += 16;
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; i + 1 < size; i += 2)
            {
                kptr[0] = weight_xc_0[i];
                kptr[1] = weight_xc_0[i + 1];
                kptr[2] = weight_xc_1[i];
                kptr[3] = weight_xc_1[i + 1];
                kptr[4] = weight_xc_2[i];
                kptr[5] = weight_xc_2[i + 1];
                kptr[6] = weight_xc_3[i];
                kptr[7] = weight_xc_3[i + 1];

                kptr += 8;
            }
            for (; i < size; i++)
            {
                kptr[0] = weight_xc_0[i];
                kptr[1] = weight_xc_1[i];
                kptr[2] = weight_xc_2[i];
                kptr[3] = weight_xc_3[i];

                kptr += 4;
            }

            i = 0;
#if __ARM_FEATURE_DOTPROD
            for (; i + 3 < num_output; i += 4)
            {
                kptr[0] = weight_hc_0[i];
                kptr[1] = weight_hc_0[i + 1];
                kptr[2] = weight_hc_0[i + 2];
                kptr[3] = weight_hc_0[i + 3];
                kptr[4] = weight_hc_1[i];
                kptr[5] = weight_hc_1[i + 1];
                kptr[6] = weight_hc_1[i + 2];
                kptr[7] = weight_hc_1[i + 3];
                kptr[8 + 0] = weight_hc_2[i];
                kptr[8 + 1] = weight_hc_2[i + 1];
                kptr[8 + 2] = weight_hc_2[i + 2];
                kptr[8 + 3] = weight_hc_2[i + 3];
                kptr[8 + 4] = weight_hc_3[i];
                kptr[8 + 5] = weight_hc_3[i + 1];
                kptr[8 + 6] = weight_hc_3[i + 2];
                kptr[8 + 7] = weight_hc_3[i + 3];

                kptr += 16;
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; i + 1 < num_output; i += 2)
            {
                kptr[0] = weight_hc_0[i];
                kptr[1] = weight_hc_0[i + 1];
                kptr[2] = weight_hc_1[i];
                kptr[3] = weight_hc_1[i + 1];
                kptr[4] = weight_hc_2[i];
                kptr[5] = weight_hc_2[i + 1];
                kptr[6] = weight_hc_3[i];
                kptr[7] = weight_hc_3[i + 1];

                kptr += 8;
            }
            for (; i < num_output; i++)
            {
                kptr[0] = weight_hc_0[i];
                kptr[1] = weight_hc_1[i];
                kptr[2] = weight_hc_2[i];
                kptr[3] = weight_hc_3[i];

                kptr += 4;
            }

            float32x4_t _xc = vld1q_f32(weight_xc_int8_scales_ptr + q);
            float32x4_t _hc = vld1q_f32(weight_hc_int8_scales_ptr + q);

#if __aarch64__
            float32x4_t _one = vdupq_n_f32(1.f);
            float32x4_t _reciprocal_xc = vdivq_f32(_one, _xc);
            float32x4_t _reciprocal_hc = vdivq_f32(_one, _hc);
#else
            float32x4_t _reciprocal_xc = vrecpeq_f32(_xc);
            _reciprocal_xc = vmulq_f32(vrecpsq_f32(_xc, _reciprocal_xc), _reciprocal_xc);
            float32x4_t _reciprocal_hc = vrecpeq_f32(_hc);
            _reciprocal_hc = vmulq_f32(vrecpsq_f32(_hc, _reciprocal_hc), _reciprocal_hc);
#endif

            vst1q_f32(descales_ptr, _reciprocal_xc);
            vst1q_f32(descales_ptr + 4, _reciprocal_hc);
        }
#endif // __ARM_NEON
        for (; q < num_output; q++)
        {
            const signed char* weight_xc_0 = weight_xc_dr.row<const signed char>(q);
            const signed char* weight_hc_0 = weight_hc_dr.row<const signed char>(q);

#if __ARM_NEON
            signed char* kptr = weight_data_tm_dr.row<signed char>(q / 4 + q % 4);
            float* descales_ptr = weight_data_tm_int8_descales_dr.row(q / 4 + q % 4);
#else
            signed char* kptr = weight_data_tm_dr.row<signed char>(q);
            float* descales_ptr = weight_data_tm_int8_descales_dr.row(q);
#endif // __ARM_NEON

            for (int i = 0; i < size; i++)
            {
                kptr[0] = weight_xc_0[i];
                kptr += 1;
            }

            for (int i = 0; i < num_output; i++)
            {
                kptr[0] = weight_hc_0[i];
                kptr += 1;
            }

            descales_ptr[0] = 1.f / weight_xc_int8_scales_ptr[q];
            descales_ptr[1] = 1.f / weight_hc_int8_scales_ptr[q];
        }
    }
}

static void rnn_int8(const Mat& bottom_blob_int8, const Mat& bottom_blob_int8_descales, Mat& top_blob, int elemtype, int reverse, const Mat& weight_data_tm, const Mat& weight_data_tm_int8_descales, const Mat& bias_c, Mat& hidden_state, const Option& opt)
{
    // TODO dispatch for __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    // TODO dispatch for __ARM_FEATURE_DOTPROD

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
    if (ncnn::cpu_support_arm_asimddp())
    {
        rnn_int8_asimddp(bottom_blob_int8, bottom_blob_int8_descales, top_blob, elemtype, reverse, weight_data_tm, weight_data_tm_int8_descales, bias_c, hidden_state, opt);
        return;
    }
#endif

    int size = bottom_blob_int8.w;
    int T = bottom_blob_int8.h;

    int num_output = top_blob.w;

    // num_output
    Mat gates(num_output, 4u, opt.workspace_allocator);

    Mat hidden_state_int8(num_output, (size_t)1u, 1, opt.workspace_allocator);
    float hidden_state_int8_scale = 1.f;
    float hidden_state_int8_descale = 1.f;

    // unroll
    for (int t = 0; t < T; t++)
    {
        int ti = reverse ? T - 1 - t : t;

        // dynamic quantize hidden_state
        {
            float absmax = 0.f;
            for (int i = 0; i < num_output; i++)
            {
                absmax = std::max(absmax, (float)fabs(hidden_state[i]));
            }

            if (absmax == 0.f)
            {
                hidden_state_int8.fill<signed char>(0);
            }
            else
            {
                hidden_state_int8_scale = 127.f / absmax;
                hidden_state_int8_descale = absmax / 127.f;

                signed char* hs = hidden_state_int8;
                for (int i = 0; i < num_output; i++)
                {
                    hs[i] = float2int8(hidden_state[i] * hidden_state_int8_scale);
                }
            }
        }

        int remain_num_output_start = 0;
#if __ARM_NEON
        int nn_num_output = num_output >> 2;
        remain_num_output_start = nn_num_output << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            const signed char* x = bottom_blob_int8.row<const signed char>(ti);
            const signed char* hs = hidden_state_int8;
            const float descale_x = bottom_blob_int8_descales[ti];
            const float descale_h = hidden_state_int8_descale;

            const signed char* kptr = weight_data_tm.row<const signed char>(q / 4);

            const float* descales_ptr = weight_data_tm_int8_descales.row(q / 4);

            int32x4_t _rnn_Hx0 = vdupq_n_s32(0);
            int i = 0;
#if __ARM_FEATURE_DOTPROD
            int32x4_t _sum1 = vdupq_n_s32(0);
            int32x4_t _sum2 = vdupq_n_s32(0);
            int32x4_t _sum3 = vdupq_n_s32(0);
            for (; i + 15 < size; i += 16)
            {
                int8x16_t _xi = vld1q_s8(x + i);
                int8x16_t _w0 = vld1q_s8(kptr);
                int8x16_t _w1 = vld1q_s8(kptr + 16);
                int8x16_t _w2 = vld1q_s8(kptr + 32);
                int8x16_t _w3 = vld1q_s8(kptr + 48);
                _rnn_Hx0 = vdotq_laneq_s32(_rnn_Hx0, _w0, _xi, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _w1, _xi, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _w2, _xi, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _w3, _xi, 3);

                kptr += 64;
            }
            for (; i + 7 < size; i += 8)
            {
                int8x8_t _xi = vld1_s8(x + i);
                int8x16_t _w0 = vld1q_s8(kptr);
                int8x16_t _w1 = vld1q_s8(kptr + 16);
                _rnn_Hx0 = vdotq_lane_s32(_rnn_Hx0, _w0, _xi, 0);
                _sum1 = vdotq_lane_s32(_sum1, _w1, _xi, 1);

                kptr += 32;
            }
            _rnn_Hx0 = vaddq_s32(_rnn_Hx0, _sum1);
            _rnn_Hx0 = vaddq_s32(_rnn_Hx0, _sum2);
            _rnn_Hx0 = vaddq_s32(_rnn_Hx0, _sum3);
#endif // __ARM_FEATURE_DOTPROD
            for (; i + 3 < size; i += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _xi = vld1_s8(x + i);
                int8x16_t _w = vld1q_s8(kptr);
                _rnn_Hx0 = vdotq_lane_s32(_rnn_Hx0, _w, _xi, 0);
#else
                int16x4_t _xi01 = vreinterpret_s16_s8(vld1_s8(x + i));
                int8x8_t _xi0 = vreinterpret_s8_s16(vdup_lane_s16(_xi01, 0));
                int8x8_t _xi1 = vreinterpret_s8_s16(vdup_lane_s16(_xi01, 1));
                int8x16_t _w01 = vld1q_s8(kptr);

                int16x8_t _rnn_Hx = vmull_s8(vget_low_s8(_w01), _xi0);
                _rnn_Hx = vmlal_s8(_rnn_Hx, vget_high_s8(_w01), _xi1);
                _rnn_Hx0 = vpadalq_s16(_rnn_Hx0, _rnn_Hx);
#endif // __ARM_FEATURE_DOTPROD

                kptr += 16;
            }
            for (; i + 1 < size; i += 2)
            {
                int8x8_t _xi = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vld1_s8(x + i)), 0));
                int8x8_t _w = vld1_s8(kptr);

                int16x8_t _rnn_Hx = vmull_s8(_w, _xi);
                _rnn_Hx0 = vpadalq_s16(_rnn_Hx0, _rnn_Hx);

                kptr += 8;
            }
            for (; i < size; i++)
            {
                int8x8_t _xi = vdup_n_s8(x[i]);
                int8x8_t _w = vld1_s8(kptr);

                int16x8_t _rnn_Hx = vmull_s8(_w, _xi);
                _rnn_Hx0 = vaddw_s16(_rnn_Hx0, vget_low_s16(_rnn_Hx));

                kptr += 4;
            }

            int32x4_t _rnn_Hh0 = vdupq_n_s32(0);
            i = 0;
#if __ARM_FEATURE_DOTPROD
            _sum1 = vdupq_n_s32(0);
            _sum2 = vdupq_n_s32(0);
            _sum3 = vdupq_n_s32(0);
            for (; i + 15 < num_output; i += 16)
            {
                int8x16_t _h_cont = vld1q_s8(hs + i);
                int8x16_t _w0 = vld1q_s8(kptr);
                int8x16_t _w1 = vld1q_s8(kptr + 16);
                int8x16_t _w2 = vld1q_s8(kptr + 32);
                int8x16_t _w3 = vld1q_s8(kptr + 48);
                _rnn_Hh0 = vdotq_laneq_s32(_rnn_Hh0, _w0, _h_cont, 0);
                _sum1 = vdotq_laneq_s32(_sum1, _w1, _h_cont, 1);
                _sum2 = vdotq_laneq_s32(_sum2, _w2, _h_cont, 2);
                _sum3 = vdotq_laneq_s32(_sum3, _w3, _h_cont, 3);

                kptr += 64;
            }
            for (; i + 7 < num_output; i += 8)
            {
                int8x8_t _h_cont = vld1_s8(hs + i);
                int8x16_t _w0 = vld1q_s8(kptr);
                int8x16_t _w1 = vld1q_s8(kptr + 16);
                _rnn_Hh0 = vdotq_lane_s32(_rnn_Hh0, _w0, _h_cont, 0);
                _sum1 = vdotq_lane_s32(_sum1, _w1, _h_cont, 1);

                kptr += 32;
            }
            _rnn_Hh0 = vaddq_s32(_rnn_Hh0, _sum1);
            _rnn_Hh0 = vaddq_s32(_rnn_Hh0, _sum2);
            _rnn_Hh0 = vaddq_s32(_rnn_Hh0, _sum3);
#endif // __ARM_FEATURE_DOTPROD
            for (; i + 3 < num_output; i += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _h_cont = vld1_s8(hs + i);
                int8x16_t _w = vld1q_s8(kptr);
                _rnn_Hh0 = vdotq_lane_s32(_rnn_Hh0, _w, _h_cont, 0);
#else
                int16x4_t _h_cont01 = vreinterpret_s16_s8(vld1_s8(hs + i));
                int8x8_t _h_cont0 = vreinterpret_s8_s16(vdup_lane_s16(_h_cont01, 0));
                int8x8_t _h_cont1 = vreinterpret_s8_s16(vdup_lane_s16(_h_cont01, 1));
                int8x16_t _w01 = vld1q_s8(kptr);

                int16x8_t _rnn_Hh = vmull_s8(vget_low_s8(_w01), _h_cont0);
                _rnn_Hh = vmlal_s8(_rnn_Hh, vget_high_s8(_w01), _h_cont1);
                _rnn_Hh0 = vpadalq_s16(_rnn_Hh0, _rnn_Hh);
#endif // __ARM_FEATURE_DOTPROD

                kptr += 16;
            }
            for (; i + 1 < num_output; i += 2)
            {
                int8x8_t _h_cont = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vld1_s8(hs + i)), 0));
                int8x8_t _w = vld1_s8(kptr);

                int16x8_t _rnn_Hh = vmull_s8(_w, _h_cont);
                _rnn_Hh0 = vpadalq_s16(_rnn_Hh0, _rnn_Hh);

                kptr += 8;
            }
            for (; i < num_output; i++)
            {
                int8x8_t _h_cont = vdup_n_s8(hs[i]);
                int8x8_t _w = vld1_s8(kptr);

                int16x8_t _rnn_Hh = vmull_s8(_w, _h_cont);
                _rnn_Hh0 = vaddw_s16(_rnn_Hh0, vget_low_s16(_rnn_Hh));

                kptr += 4;
            }

            float32x4_t _descale_x = vdupq_n_f32(descale_x);
            float32x4_t _descale_h = vdupq_n_f32(descale_h);

            float32x4_t _rnn_H = vld1q_f32((const float*)bias_c + q);

            float32x4_t _descale_xc = vld1q_f32(descales_ptr);

            _rnn_H = vmlaq_f32(_rnn_H, vcvtq_f32_s32(_rnn_Hx0), vmulq_f32(_descale_x, _descale_xc));

            float32x4_t _descale_hc = vld1q_f32(descales_ptr + 4);

            _rnn_H = vmlaq_f32(_rnn_H, vcvtq_f32_s32(_rnn_Hh0), vmulq_f32(_descale_h, _descale_hc));

            _rnn_H = tanh_ps(_rnn_H);

            vst1q_f32((float*)gates + q, _rnn_H);
        }
#endif // __ARM_NEON
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            const signed char* x = bottom_blob_int8.row<const signed char>(ti);
            const signed char* hs = hidden_state_int8;
            const float descale_x = bottom_blob_int8_descales[ti];
            const float descale_h = hidden_state_int8_descale;

#if __ARM_NEON
            const signed char* kptr = weight_data_tm.row<const signed char>(q / 4 + q % 4);
            const float* descales_ptr = weight_data_tm_int8_descales.row(q / 4 + q % 4);
#else
            const signed char* kptr = weight_data_tm.row<const signed char>(q);
            const float* descales_ptr = weight_data_tm_int8_descales.row(q);
#endif // __ARM_NEON

            const float descale_xc = descales_ptr[0];
            const float descale_hc = descales_ptr[1];

            int Hx = 0;
            for (int i = 0; i < size; i++)
            {
                Hx += kptr[0] * x[i];
                kptr += 1;
            }

            int Hh = 0;
            for (int i = 0; i < num_output; i++)
            {
                Hh += kptr[0] * hs[i];
                kptr += 1;
            }

            float H = bias_c[q] + Hx * (descale_x * descale_xc) + Hh * (descale_h * descale_hc);

            H = tanhf(H);

            gates[q] = H;
        }

        float* output_data = top_blob.row(ti);

        float* hidden_ptr = hidden_state;

#if __ARM_NEON
        nn_num_output = num_output >> 2;
        remain_num_output_start = nn_num_output << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_num_output; qq++)
        {
            int q = qq * 4;

            float32x4_t _rnn_H = vld1q_f32((float*)gates + q);

            vst1q_f32(hidden_ptr + q, _rnn_H);

            if (elemtype == 1)
            {
                // fp32
                vst1q_f32(output_data + q, _rnn_H);
            }
            if (elemtype == 2)
            {
                // fp16
                vst1_u16((unsigned short*)output_data + q, (uint16x4_t)vcvt_f16_f32(_rnn_H));
            }
            if (elemtype == 4)
            {
                // bf16
                vst1_u16((unsigned short*)output_data + q, float2bfloat(_rnn_H));
            }
        }
#endif // __ARM_NEON
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            float H = gates[q];

            hidden_ptr[q] = H;

            if (elemtype == 1)
            {
                output_data[q] = H;
            }
            if (elemtype == 2)
            {
                ((unsigned short*)output_data)[q] = float32_to_float16(H);
            }
            if (elemtype == 4)
            {
                ((unsigned short*)output_data)[q] = float32_to_bfloat16(H);
            }
        }
    }
}