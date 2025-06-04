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
void gru_transform_weight_int8_asimddp(const Mat& weight_xc, const Mat& weight_xc_int8_scales, const Mat& weight_hc, const Mat& weight_hc_int8_scales, const Mat& bias_c, Mat& weight_data_tm, Mat& weight_data_tm_int8_descales, Mat& bias_c_tm, int size, int num_output, int num_directions, const Option& opt);
void gru_int8_asimddp(const Mat& bottom_blob_int8, const Mat& bottom_blob_int8_descales, Mat& top_blob, int elemtype, int reverse, const Mat& weight_data_tm, const Mat& weight_data_tm_int8_descales, const Mat& bias_c, Mat& hidden_state, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_VFPV4 && __ARM_NEON && !(__ARM_FP & 2)
void gru_int8_gate_output_vfpv4(const Mat& gates, Mat& hidden_state, Mat& top_blob, int ti, int elemtype, const Option& opt);
#endif

static void gru_transform_weight_int8(const Mat& weight_xc, const Mat& weight_xc_int8_scales, const Mat& weight_hc, const Mat& weight_hc_int8_scales, const Mat& bias_c, Mat& weight_data_tm, Mat& weight_data_tm_int8_descales, Mat& bias_c_tm, int size, int num_output, int num_directions, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
    if (ncnn::cpu_support_arm_asimddp())
    {
        gru_transform_weight_int8_asimddp(weight_xc, weight_xc_int8_scales, weight_hc, weight_hc_int8_scales, bias_c, weight_data_tm, weight_data_tm_int8_descales, bias_c_tm, size, num_output, num_directions, opt);
        return;
    }
#endif

#if __ARM_NEON
    weight_data_tm.create(size * 12 + num_output * 12, num_output / 4 + num_output % 4, num_directions, 1u, 1);
    weight_data_tm_int8_descales.create(12 + 12, num_output / 4 + num_output % 4, num_directions);
#else
    weight_data_tm.create(size * 3 + num_output * 3, num_output, num_directions, 1u, 1);
    weight_data_tm_int8_descales.create(3 + 3, num_output, num_directions);
#endif
    bias_c_tm.create(num_output, 1, num_directions, 16u, 4);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int dr = 0; dr < num_directions; dr++)
    {
        const Mat weight_xc_dr = weight_xc.channel(dr);
        const Mat weight_hc_dr = weight_hc.channel(dr);
        const Mat bias_c_dr = bias_c.channel(dr);
        const float* weight_xc_int8_scales_ptr = weight_xc_int8_scales.row(dr);
        const float* weight_hc_int8_scales_ptr = weight_hc_int8_scales.row(dr);

        Mat weight_data_tm_dr = weight_data_tm.channel(dr);
        Mat bias_c_tm_dr = bias_c_tm.channel(dr);
        Mat weight_data_tm_int8_descales_dr = weight_data_tm_int8_descales.channel(dr);

        const float* bias_c_R = bias_c_dr.row(0);
        const float* bias_c_U = bias_c_dr.row(1);
        const float* bias_c_WN = bias_c_dr.row(2);
        const float* bias_c_BN = bias_c_dr.row(3);

        float* bias_c_RUBNWN = bias_c_tm_dr.row(0);

        int q = 0;
#if __ARM_NEON
        for (; q + 3 < num_output; q += 4)
        {
            vst1q_f32(bias_c_RUBNWN, vld1q_f32(bias_c_R + q));
            vst1q_f32(bias_c_RUBNWN + 4, vld1q_f32(bias_c_U + q));
            vst1q_f32(bias_c_RUBNWN + 8, vld1q_f32(bias_c_BN + q));
            vst1q_f32(bias_c_RUBNWN + 12, vld1q_f32(bias_c_WN + q));

            bias_c_RUBNWN += 16;

            const signed char* weight_xc_R_0 = weight_xc_dr.row<const signed char>(num_output * 0 + q);
            const signed char* weight_xc_U_0 = weight_xc_dr.row<const signed char>(num_output * 1 + q);
            const signed char* weight_xc_N_0 = weight_xc_dr.row<const signed char>(num_output * 2 + q);

            const signed char* weight_xc_R_1 = weight_xc_dr.row<const signed char>(num_output * 0 + q + 1);
            const signed char* weight_xc_U_1 = weight_xc_dr.row<const signed char>(num_output * 1 + q + 1);
            const signed char* weight_xc_N_1 = weight_xc_dr.row<const signed char>(num_output * 2 + q + 1);

            const signed char* weight_xc_R_2 = weight_xc_dr.row<const signed char>(num_output * 0 + q + 2);
            const signed char* weight_xc_U_2 = weight_xc_dr.row<const signed char>(num_output * 1 + q + 2);
            const signed char* weight_xc_N_2 = weight_xc_dr.row<const signed char>(num_output * 2 + q + 2);

            const signed char* weight_xc_R_3 = weight_xc_dr.row<const signed char>(num_output * 0 + q + 3);
            const signed char* weight_xc_U_3 = weight_xc_dr.row<const signed char>(num_output * 1 + q + 3);
            const signed char* weight_xc_N_3 = weight_xc_dr.row<const signed char>(num_output * 2 + q + 3);

            const signed char* weight_hc_R_0 = weight_hc_dr.row<const signed char>(num_output * 0 + q);
            const signed char* weight_hc_U_0 = weight_hc_dr.row<const signed char>(num_output * 1 + q);
            const signed char* weight_hc_N_0 = weight_hc_dr.row<const signed char>(num_output * 2 + q);

            const signed char* weight_hc_R_1 = weight_hc_dr.row<const signed char>(num_output * 0 + q + 1);
            const signed char* weight_hc_U_1 = weight_hc_dr.row<const signed char>(num_output * 1 + q + 1);
            const signed char* weight_hc_N_1 = weight_hc_dr.row<const signed char>(num_output * 2 + q + 1);

            const signed char* weight_hc_R_2 = weight_hc_dr.row<const signed char>(num_output * 0 + q + 2);
            const signed char* weight_hc_U_2 = weight_hc_dr.row<const signed char>(num_output * 1 + q + 2);
            const signed char* weight_hc_N_2 = weight_hc_dr.row<const signed char>(num_output * 2 + q + 2);

            const signed char* weight_hc_R_3 = weight_hc_dr.row<const signed char>(num_output * 0 + q + 3);
            const signed char* weight_hc_U_3 = weight_hc_dr.row<const signed char>(num_output * 1 + q + 3);
            const signed char* weight_hc_N_3 = weight_hc_dr.row<const signed char>(num_output * 2 + q + 3);

            signed char* kptr = weight_data_tm_dr.row<signed char>(q / 4);
            float* descales_ptr = weight_data_tm_int8_descales_dr.row(q / 4);

            int i = 0;
#if __ARM_FEATURE_DOTPROD
            for (; i + 3 < size; i += 4)
            {
                kptr[0] = weight_xc_R_0[i];
                kptr[1] = weight_xc_R_0[i + 1];
                kptr[2] = weight_xc_R_0[i + 2];
                kptr[3] = weight_xc_R_0[i + 3];
                kptr[4] = weight_xc_R_1[i];
                kptr[5] = weight_xc_R_1[i + 1];
                kptr[6] = weight_xc_R_1[i + 2];
                kptr[7] = weight_xc_R_1[i + 3];
                kptr[8 + 0] = weight_xc_R_2[i];
                kptr[8 + 1] = weight_xc_R_2[i + 1];
                kptr[8 + 2] = weight_xc_R_2[i + 2];
                kptr[8 + 3] = weight_xc_R_2[i + 3];
                kptr[8 + 4] = weight_xc_R_3[i];
                kptr[8 + 5] = weight_xc_R_3[i + 1];
                kptr[8 + 6] = weight_xc_R_3[i + 2];
                kptr[8 + 7] = weight_xc_R_3[i + 3];
                kptr[16 + 0] = weight_xc_U_0[i];
                kptr[16 + 1] = weight_xc_U_0[i + 1];
                kptr[16 + 2] = weight_xc_U_0[i + 2];
                kptr[16 + 3] = weight_xc_U_0[i + 3];
                kptr[16 + 4] = weight_xc_U_1[i];
                kptr[16 + 5] = weight_xc_U_1[i + 1];
                kptr[16 + 6] = weight_xc_U_1[i + 2];
                kptr[16 + 7] = weight_xc_U_1[i + 3];
                kptr[24 + 0] = weight_xc_U_2[i];
                kptr[24 + 1] = weight_xc_U_2[i + 1];
                kptr[24 + 2] = weight_xc_U_2[i + 2];
                kptr[24 + 3] = weight_xc_U_2[i + 3];
                kptr[24 + 4] = weight_xc_U_3[i];
                kptr[24 + 5] = weight_xc_U_3[i + 1];
                kptr[24 + 6] = weight_xc_U_3[i + 2];
                kptr[24 + 7] = weight_xc_U_3[i + 3];

                kptr += 32;
            }
#else
            for (; i + 7 < size; i += 8)
            {
                int8x8_t _w0 = vld1_s8(weight_xc_R_0 + i);
                int8x8_t _w1 = vld1_s8(weight_xc_R_1 + i);
                int8x8_t _w2 = vld1_s8(weight_xc_R_2 + i);
                int8x8_t _w3 = vld1_s8(weight_xc_R_3 + i);
                int8x8_t _w4 = vld1_s8(weight_xc_U_0 + i);
                int8x8_t _w5 = vld1_s8(weight_xc_U_1 + i);
                int8x8_t _w6 = vld1_s8(weight_xc_U_2 + i);
                int8x8_t _w7 = vld1_s8(weight_xc_U_3 + i);

                int32x2x2_t _t0 = vtrn_s32(vreinterpret_s32_s8(_w0), vreinterpret_s32_s8(_w4));
                int32x2x2_t _t1 = vtrn_s32(vreinterpret_s32_s8(_w1), vreinterpret_s32_s8(_w5));
                int32x2x2_t _t2 = vtrn_s32(vreinterpret_s32_s8(_w2), vreinterpret_s32_s8(_w6));
                int32x2x2_t _t3 = vtrn_s32(vreinterpret_s32_s8(_w3), vreinterpret_s32_s8(_w7));

                int32x4x4_t _w;
                _w.val[0] = vcombine_s32(_t0.val[0], _t0.val[1]);
                _w.val[1] = vcombine_s32(_t1.val[0], _t1.val[1]);
                _w.val[2] = vcombine_s32(_t2.val[0], _t2.val[1]);
                _w.val[3] = vcombine_s32(_t3.val[0], _t3.val[1]);

                vst4q_s32((int*)kptr, _w);

                kptr += 64;
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; i + 1 < size; i += 2)
            {
                kptr[0] = weight_xc_R_0[i];
                kptr[1] = weight_xc_R_0[i + 1];
                kptr[2] = weight_xc_R_1[i];
                kptr[3] = weight_xc_R_1[i + 1];
                kptr[4] = weight_xc_R_2[i];
                kptr[5] = weight_xc_R_2[i + 1];
                kptr[6] = weight_xc_R_3[i];
                kptr[7] = weight_xc_R_3[i + 1];
                kptr[8 + 0] = weight_xc_U_0[i];
                kptr[8 + 1] = weight_xc_U_0[i + 1];
                kptr[8 + 2] = weight_xc_U_1[i];
                kptr[8 + 3] = weight_xc_U_1[i + 1];
                kptr[8 + 4] = weight_xc_U_2[i];
                kptr[8 + 5] = weight_xc_U_2[i + 1];
                kptr[8 + 6] = weight_xc_U_3[i];
                kptr[8 + 7] = weight_xc_U_3[i + 1];

                kptr += 16;
            }
            for (; i < size; i++)
            {
                kptr[0] = weight_xc_R_0[i];
                kptr[1] = weight_xc_R_1[i];
                kptr[2] = weight_xc_R_2[i];
                kptr[3] = weight_xc_R_3[i];
                kptr[4] = weight_xc_U_0[i];
                kptr[5] = weight_xc_U_1[i];
                kptr[6] = weight_xc_U_2[i];
                kptr[7] = weight_xc_U_3[i];

                kptr += 8;
            }

            i = 0;
#if __ARM_FEATURE_DOTPROD
            for (; i + 3 < num_output; i += 4)
            {
                kptr[0] = weight_hc_R_0[i];
                kptr[1] = weight_hc_R_0[i + 1];
                kptr[2] = weight_hc_R_0[i + 2];
                kptr[3] = weight_hc_R_0[i + 3];
                kptr[4] = weight_hc_R_1[i];
                kptr[5] = weight_hc_R_1[i + 1];
                kptr[6] = weight_hc_R_1[i + 2];
                kptr[7] = weight_hc_R_1[i + 3];
                kptr[8 + 0] = weight_hc_R_2[i];
                kptr[8 + 1] = weight_hc_R_2[i + 1];
                kptr[8 + 2] = weight_hc_R_2[i + 2];
                kptr[8 + 3] = weight_hc_R_2[i + 3];
                kptr[8 + 4] = weight_hc_R_3[i];
                kptr[8 + 5] = weight_hc_R_3[i + 1];
                kptr[8 + 6] = weight_hc_R_3[i + 2];
                kptr[8 + 7] = weight_hc_R_3[i + 3];
                kptr[16 + 0] = weight_hc_U_0[i];
                kptr[16 + 1] = weight_hc_U_0[i + 1];
                kptr[16 + 2] = weight_hc_U_0[i + 2];
                kptr[16 + 3] = weight_hc_U_0[i + 3];
                kptr[16 + 4] = weight_hc_U_1[i];
                kptr[16 + 5] = weight_hc_U_1[i + 1];
                kptr[16 + 6] = weight_hc_U_1[i + 2];
                kptr[16 + 7] = weight_hc_U_1[i + 3];
                kptr[24 + 0] = weight_hc_U_2[i];
                kptr[24 + 1] = weight_hc_U_2[i + 1];
                kptr[24 + 2] = weight_hc_U_2[i + 2];
                kptr[24 + 3] = weight_hc_U_2[i + 3];
                kptr[24 + 4] = weight_hc_U_3[i];
                kptr[24 + 5] = weight_hc_U_3[i + 1];
                kptr[24 + 6] = weight_hc_U_3[i + 2];
                kptr[24 + 7] = weight_hc_U_3[i + 3];

                kptr += 32;
            }
#else
            for (; i + 7 < num_output; i += 8)
            {
                int8x8_t _w0 = vld1_s8(weight_hc_R_0 + i);
                int8x8_t _w1 = vld1_s8(weight_hc_R_1 + i);
                int8x8_t _w2 = vld1_s8(weight_hc_R_2 + i);
                int8x8_t _w3 = vld1_s8(weight_hc_R_3 + i);
                int8x8_t _w4 = vld1_s8(weight_hc_U_0 + i);
                int8x8_t _w5 = vld1_s8(weight_hc_U_1 + i);
                int8x8_t _w6 = vld1_s8(weight_hc_U_2 + i);
                int8x8_t _w7 = vld1_s8(weight_hc_U_3 + i);

                int32x2x2_t _t0 = vtrn_s32(vreinterpret_s32_s8(_w0), vreinterpret_s32_s8(_w4));
                int32x2x2_t _t1 = vtrn_s32(vreinterpret_s32_s8(_w1), vreinterpret_s32_s8(_w5));
                int32x2x2_t _t2 = vtrn_s32(vreinterpret_s32_s8(_w2), vreinterpret_s32_s8(_w6));
                int32x2x2_t _t3 = vtrn_s32(vreinterpret_s32_s8(_w3), vreinterpret_s32_s8(_w7));

                int32x4x4_t _w;
                _w.val[0] = vcombine_s32(_t0.val[0], _t0.val[1]);
                _w.val[1] = vcombine_s32(_t1.val[0], _t1.val[1]);
                _w.val[2] = vcombine_s32(_t2.val[0], _t2.val[1]);
                _w.val[3] = vcombine_s32(_t3.val[0], _t3.val[1]);

                vst4q_s32((int*)kptr, _w);

                kptr += 64;
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; i + 1 < num_output; i += 2)
            {
                kptr[0] = weight_hc_R_0[i];
                kptr[1] = weight_hc_R_0[i + 1];
                kptr[2] = weight_hc_R_1[i];
                kptr[3] = weight_hc_R_1[i + 1];
                kptr[4] = weight_hc_R_2[i];
                kptr[5] = weight_hc_R_2[i + 1];
                kptr[6] = weight_hc_R_3[i];
                kptr[7] = weight_hc_R_3[i + 1];
                kptr[8 + 0] = weight_hc_U_0[i];
                kptr[8 + 1] = weight_hc_U_0[i + 1];
                kptr[8 + 2] = weight_hc_U_1[i];
                kptr[8 + 3] = weight_hc_U_1[i + 1];
                kptr[8 + 4] = weight_hc_U_2[i];
                kptr[8 + 5] = weight_hc_U_2[i + 1];
                kptr[8 + 6] = weight_hc_U_3[i];
                kptr[8 + 7] = weight_hc_U_3[i + 1];

                kptr += 16;
            }
            for (; i < num_output; i++)
            {
                kptr[0] = weight_hc_R_0[i];
                kptr[1] = weight_hc_R_1[i];
                kptr[2] = weight_hc_R_2[i];
                kptr[3] = weight_hc_R_3[i];
                kptr[4] = weight_hc_U_0[i];
                kptr[5] = weight_hc_U_1[i];
                kptr[6] = weight_hc_U_2[i];
                kptr[7] = weight_hc_U_3[i];

                kptr += 8;
            }

            i = 0;
#if __ARM_FEATURE_DOTPROD
            for (; i + 3 < num_output; i += 4)
            {
                kptr[0] = weight_hc_N_0[i];
                kptr[1] = weight_hc_N_0[i + 1];
                kptr[2] = weight_hc_N_0[i + 2];
                kptr[3] = weight_hc_N_0[i + 3];
                kptr[4] = weight_hc_N_1[i];
                kptr[5] = weight_hc_N_1[i + 1];
                kptr[6] = weight_hc_N_1[i + 2];
                kptr[7] = weight_hc_N_1[i + 3];
                kptr[8 + 0] = weight_hc_N_2[i];
                kptr[8 + 1] = weight_hc_N_2[i + 1];
                kptr[8 + 2] = weight_hc_N_2[i + 2];
                kptr[8 + 3] = weight_hc_N_2[i + 3];
                kptr[8 + 4] = weight_hc_N_3[i];
                kptr[8 + 5] = weight_hc_N_3[i + 1];
                kptr[8 + 6] = weight_hc_N_3[i + 2];
                kptr[8 + 7] = weight_hc_N_3[i + 3];

                kptr += 16;
            }
#else
            for (; i + 7 < num_output; i += 8)
            {
                vst1_s8(kptr, vld1_s8(weight_hc_N_0 + i));
                vst1_s8(kptr + 8, vld1_s8(weight_hc_N_1 + i));
                vst1_s8(kptr + 16, vld1_s8(weight_hc_N_2 + i));
                vst1_s8(kptr + 24, vld1_s8(weight_hc_N_3 + i));
                kptr += 32;
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; i + 1 < num_output; i += 2)
            {
                kptr[0] = weight_hc_N_0[i];
                kptr[1] = weight_hc_N_0[i + 1];
                kptr[2] = weight_hc_N_1[i];
                kptr[3] = weight_hc_N_1[i + 1];
                kptr[4] = weight_hc_N_2[i];
                kptr[5] = weight_hc_N_2[i + 1];
                kptr[6] = weight_hc_N_3[i];
                kptr[7] = weight_hc_N_3[i + 1];

                kptr += 8;
            }
            for (; i < num_output; i++)
            {
                kptr[0] = weight_hc_N_0[i];
                kptr[1] = weight_hc_N_1[i];
                kptr[2] = weight_hc_N_2[i];
                kptr[3] = weight_hc_N_3[i];

                kptr += 4;
            }

            i = 0;
#if __ARM_FEATURE_DOTPROD
            for (; i + 3 < size; i += 4)
            {
                kptr[0] = weight_xc_N_0[i];
                kptr[1] = weight_xc_N_0[i + 1];
                kptr[2] = weight_xc_N_0[i + 2];
                kptr[3] = weight_xc_N_0[i + 3];
                kptr[4] = weight_xc_N_1[i];
                kptr[5] = weight_xc_N_1[i + 1];
                kptr[6] = weight_xc_N_1[i + 2];
                kptr[7] = weight_xc_N_1[i + 3];
                kptr[8 + 0] = weight_xc_N_2[i];
                kptr[8 + 1] = weight_xc_N_2[i + 1];
                kptr[8 + 2] = weight_xc_N_2[i + 2];
                kptr[8 + 3] = weight_xc_N_2[i + 3];
                kptr[8 + 4] = weight_xc_N_3[i];
                kptr[8 + 5] = weight_xc_N_3[i + 1];
                kptr[8 + 6] = weight_xc_N_3[i + 2];
                kptr[8 + 7] = weight_xc_N_3[i + 3];

                kptr += 16;
            }
#else
            for (; i + 7 < size; i += 8)
            {
                vst1_s8(kptr, vld1_s8(weight_xc_N_0 + i));
                vst1_s8(kptr + 8, vld1_s8(weight_xc_N_1 + i));
                vst1_s8(kptr + 16, vld1_s8(weight_xc_N_2 + i));
                vst1_s8(kptr + 24, vld1_s8(weight_xc_N_3 + i));
                kptr += 32;
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; i + 1 < size; i += 2)
            {
                kptr[0] = weight_xc_N_0[i];
                kptr[1] = weight_xc_N_0[i + 1];
                kptr[2] = weight_xc_N_1[i];
                kptr[3] = weight_xc_N_1[i + 1];
                kptr[4] = weight_xc_N_2[i];
                kptr[5] = weight_xc_N_2[i + 1];
                kptr[6] = weight_xc_N_3[i];
                kptr[7] = weight_xc_N_3[i + 1];

                kptr += 8;
            }
            for (; i < size; i++)
            {
                kptr[0] = weight_xc_N_0[i];
                kptr[1] = weight_xc_N_1[i];
                kptr[2] = weight_xc_N_2[i];
                kptr[3] = weight_xc_N_3[i];

                kptr += 4;
            }

            float32x4_t _xc_R0 = vld1q_f32(weight_xc_int8_scales_ptr + q);
            float32x4_t _xc_U0 = vld1q_f32(weight_xc_int8_scales_ptr + num_output + q);
            float32x4_t _xc_N0 = vld1q_f32(weight_xc_int8_scales_ptr + num_output * 2 + q);
            float32x4_t _hc_R0 = vld1q_f32(weight_hc_int8_scales_ptr + q);
            float32x4_t _hc_U0 = vld1q_f32(weight_hc_int8_scales_ptr + num_output + q);
            float32x4_t _hc_N0 = vld1q_f32(weight_hc_int8_scales_ptr + num_output * 2 + q);

#if __aarch64__
            float32x4_t _one = vdupq_n_f32(1.f);
            float32x4_t _reciprocal_xc_R0 = vdivq_f32(_one, _xc_R0);
            float32x4_t _reciprocal_xc_U0 = vdivq_f32(_one, _xc_U0);
            float32x4_t _reciprocal_xc_N0 = vdivq_f32(_one, _xc_N0);
            float32x4_t _reciprocal_hc_R0 = vdivq_f32(_one, _hc_R0);
            float32x4_t _reciprocal_hc_U0 = vdivq_f32(_one, _hc_U0);
            float32x4_t _reciprocal_hc_N0 = vdivq_f32(_one, _hc_N0);
#else
            float32x4_t _reciprocal_xc_R0 = vrecpeq_f32(_xc_R0);
            float32x4_t _reciprocal_xc_U0 = vrecpeq_f32(_xc_U0);
            float32x4_t _reciprocal_xc_N0 = vrecpeq_f32(_xc_N0);
            _reciprocal_xc_R0 = vmulq_f32(vrecpsq_f32(_xc_R0, _reciprocal_xc_R0), _reciprocal_xc_R0);
            _reciprocal_xc_U0 = vmulq_f32(vrecpsq_f32(_xc_U0, _reciprocal_xc_U0), _reciprocal_xc_U0);
            _reciprocal_xc_N0 = vmulq_f32(vrecpsq_f32(_xc_N0, _reciprocal_xc_N0), _reciprocal_xc_N0);
            _reciprocal_xc_R0 = vmulq_f32(vrecpsq_f32(_xc_R0, _reciprocal_xc_R0), _reciprocal_xc_R0);
            _reciprocal_xc_U0 = vmulq_f32(vrecpsq_f32(_xc_U0, _reciprocal_xc_U0), _reciprocal_xc_U0);
            _reciprocal_xc_N0 = vmulq_f32(vrecpsq_f32(_xc_N0, _reciprocal_xc_N0), _reciprocal_xc_N0);
            float32x4_t _reciprocal_hc_R0 = vrecpeq_f32(_hc_R0);
            float32x4_t _reciprocal_hc_U0 = vrecpeq_f32(_hc_U0);
            float32x4_t _reciprocal_hc_N0 = vrecpeq_f32(_hc_N0);
            _reciprocal_hc_R0 = vmulq_f32(vrecpsq_f32(_hc_R0, _reciprocal_hc_R0), _reciprocal_hc_R0);
            _reciprocal_hc_U0 = vmulq_f32(vrecpsq_f32(_hc_U0, _reciprocal_hc_U0), _reciprocal_hc_U0);
            _reciprocal_hc_N0 = vmulq_f32(vrecpsq_f32(_hc_N0, _reciprocal_hc_N0), _reciprocal_hc_N0);
            _reciprocal_hc_R0 = vmulq_f32(vrecpsq_f32(_hc_R0, _reciprocal_hc_R0), _reciprocal_hc_R0);
            _reciprocal_hc_U0 = vmulq_f32(vrecpsq_f32(_hc_U0, _reciprocal_hc_U0), _reciprocal_hc_U0);
            _reciprocal_hc_N0 = vmulq_f32(vrecpsq_f32(_hc_N0, _reciprocal_hc_N0), _reciprocal_hc_N0);
#endif

            vst1q_f32(descales_ptr, _reciprocal_xc_R0);
            vst1q_f32(descales_ptr + 4, _reciprocal_xc_U0);
            vst1q_f32(descales_ptr + 8, _reciprocal_hc_R0);
            vst1q_f32(descales_ptr + 12, _reciprocal_hc_U0);
            vst1q_f32(descales_ptr + 16, _reciprocal_hc_N0);
            vst1q_f32(descales_ptr + 20, _reciprocal_xc_N0);
        }
#endif // __ARM_NEON
        for (; q < num_output; q++)
        {
            bias_c_RUBNWN[0] = bias_c_R[q];
            bias_c_RUBNWN[1] = bias_c_U[q];
            bias_c_RUBNWN[2] = bias_c_BN[q];
            bias_c_RUBNWN[3] = bias_c_WN[q];

            bias_c_RUBNWN += 4;

            const signed char* weight_xc_R = weight_xc_dr.row<const signed char>(num_output * 0 + q);
            const signed char* weight_xc_U = weight_xc_dr.row<const signed char>(num_output * 1 + q);
            const signed char* weight_xc_N = weight_xc_dr.row<const signed char>(num_output * 2 + q);

            const signed char* weight_hc_R = weight_hc_dr.row<const signed char>(num_output * 0 + q);
            const signed char* weight_hc_U = weight_hc_dr.row<const signed char>(num_output * 1 + q);
            const signed char* weight_hc_N = weight_hc_dr.row<const signed char>(num_output * 2 + q);

#if __ARM_NEON
            signed char* kptr = weight_data_tm_dr.row<signed char>(q / 4 + q % 4);
            float* descales_ptr = weight_data_tm_int8_descales_dr.row(q / 4 + q % 4);
#else
            signed char* kptr = weight_data_tm_dr.row<signed char>(q);
            float* descales_ptr = weight_data_tm_int8_descales_dr.row(q);
#endif // __ARM_NEON

            for (int i = 0; i < size; i++)
            {
                kptr[0] = weight_xc_R[i];
                kptr[1] = weight_xc_U[i];
                kptr += 2;
            }

            for (int i = 0; i < num_output; i++)
            {
                kptr[0] = weight_hc_R[i];
                kptr[1] = weight_hc_U[i];
                kptr += 2;
            }

            for (int i = 0; i < num_output; i++)
            {
                kptr[0] = weight_hc_N[i];
                kptr += 1;
            }

            for (int i = 0; i < size; i++)
            {
                kptr[0] = weight_xc_N[i];
                kptr += 1;
            }

            descales_ptr[0] = 1.f / weight_xc_int8_scales_ptr[num_output * 0 + q];
            descales_ptr[1] = 1.f / weight_xc_int8_scales_ptr[num_output * 1 + q];
            descales_ptr[2] = 1.f / weight_hc_int8_scales_ptr[num_output * 0 + q];
            descales_ptr[3] = 1.f / weight_hc_int8_scales_ptr[num_output * 1 + q];
            descales_ptr[4] = 1.f / weight_hc_int8_scales_ptr[num_output * 2 + q];
            descales_ptr[5] = 1.f / weight_xc_int8_scales_ptr[num_output * 2 + q];
        }
    }
}

static void gru_int8_gate_output(const Mat& gates, Mat& hidden_state, Mat& top_blob, int ti, int elemtype, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_VFPV4 && __ARM_NEON && !(__ARM_FP & 2)
    if (ncnn::cpu_support_arm_vfpv4())
    {
        gru_int8_gate_output_vfpv4(gates, hidden_state, top_blob, ti, elemtype, opt);
        return;
    }
#endif

    const int num_output = top_blob.w;

    // h_t := (1 - update) .* new + update .* h_{t-1}
    float* output_data = top_blob.row(ti);

    float* hidden_ptr = hidden_state;

    int remain_num_output_start = 0;
#if __ARM_NEON
    int nn_num_output = num_output >> 2;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int qq = 0; qq < nn_num_output; qq++)
    {
        int q = qq * 4;

        const float* gates_data = gates.row(q / 4);

        float32x4_t _gru_U0 = vld1q_f32(gates_data);
        float32x4_t _gru_N0 = vld1q_f32(gates_data + 4);

        float32x4_t _gru_H0 = vaddq_f32(vmulq_f32(vsubq_f32(vdupq_n_f32(1.f), _gru_U0), _gru_N0), vmulq_f32(_gru_U0, vld1q_f32(hidden_ptr + q)));

        vst1q_f32(hidden_ptr + q, _gru_H0);

        if (elemtype == 1)
        {
            // fp32
            vst1q_f32(output_data + q, _gru_H0);
        }
        if (elemtype == 2)
        {
            // fp16
            unsigned short* outptr = (unsigned short*)output_data + q;
#if (__ARM_FP & 2)
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            asm volatile(
                "fcvtn  v0.4h, %2.4s        \n"
                "st1    {v0.4h}, [%0]       \n"
                : "=r"(outptr) // %0
                : "0"(outptr),
                "w"(_gru_H0)
                : "memory", "v0");
#else  // __aarch64__
            asm volatile(
                "vcvt.f16.f32 d0, %q2       \n"
                "vst1.u16   {d0}, [%0]      \n"
                : "=r"(outptr) // %0
                : "0"(outptr),
                "w"(_gru_H0)
                : "memory", "q0");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
            vst1_u16(outptr, (uint16x4_t)vcvt_f16_f32(_gru_H0));
#endif // NCNN_GNU_INLINE_ASM
#else
            outptr[q] = float32_to_float16(hidden_ptr[q]);
            outptr[q + 1] = float32_to_float16(hidden_ptr[q + 1]);
            outptr[q + 2] = float32_to_float16(hidden_ptr[q + 2]);
            outptr[q + 3] = float32_to_float16(hidden_ptr[q + 3]);
#endif // (__ARM_FP & 2)
        }
        if (elemtype == 4)
        {
            // bf16
            vst1_u16((unsigned short*)output_data + q, float2bfloat(_gru_H0));
        }
    }
    remain_num_output_start += nn_num_output << 2;
#endif // __ARM_NEON
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = remain_num_output_start; q < num_output; q++)
    {
#if __ARM_NEON
        const float* gates_data = gates.row(q / 4 + q % 4);
#else
        const float* gates_data = gates.row(q);
#endif

        float U = gates_data[0];
        float N = gates_data[1];

        float H = (1 - U) * N + U * hidden_ptr[q];

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

static void gru_int8(const Mat& bottom_blob_int8, const Mat& bottom_blob_int8_descales, Mat& top_blob, int elemtype, int reverse, const Mat& weight_data_tm, const Mat& weight_data_tm_int8_descales, const Mat& bias_c, Mat& hidden_state, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
    if (ncnn::cpu_support_arm_asimddp())
    {
        gru_int8_asimddp(bottom_blob_int8, bottom_blob_int8_descales, top_blob, elemtype, reverse, weight_data_tm, weight_data_tm_int8_descales, bias_c, hidden_state, opt);
        return;
    }
#endif

    int size = bottom_blob_int8.w;
    int T = bottom_blob_int8.h;

    int num_output = top_blob.w;

    // 2 x num_output
#if __ARM_NEON
    Mat gates(4 * 2, num_output / 4 + num_output % 4, 4u, opt.workspace_allocator);
#else
    Mat gates(2, num_output, 4u, opt.workspace_allocator);
#endif

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

            // gate reset update
            const float* bias_c_RUBNWN = (const float*)bias_c + q * 4;

            const signed char* kptr = weight_data_tm.row<const signed char>(q / 4);

            const float* descales_ptr = weight_data_tm_int8_descales.row(q / 4);

            int32x4_t _gru_Rx0 = vdupq_n_s32(0);
            int32x4_t _gru_Ux0 = vdupq_n_s32(0);
            int i = 0;
#if __ARM_FEATURE_DOTPROD
            int32x4_t _sum1 = vdupq_n_s32(0);
            int32x4_t _sum2 = vdupq_n_s32(0);
            for (; i + 7 < size; i += 8)
            {
                int8x8_t _xi = vld1_s8(x + i);
                int8x16_t _w0 = vld1q_s8(kptr);
                int8x16_t _w1 = vld1q_s8(kptr + 16);
                int8x16_t _w2 = vld1q_s8(kptr + 32);
                int8x16_t _w3 = vld1q_s8(kptr + 48);
                _gru_Rx0 = vdotq_lane_s32(_gru_Rx0, _w0, _xi, 0);
                _gru_Ux0 = vdotq_lane_s32(_gru_Ux0, _w1, _xi, 0);
                _sum1 = vdotq_lane_s32(_sum1, _w2, _xi, 1);
                _sum2 = vdotq_lane_s32(_sum2, _w3, _xi, 1);

                kptr += 64;
            }
            _gru_Rx0 = vaddq_s32(_gru_Rx0, _sum1);
            _gru_Ux0 = vaddq_s32(_gru_Ux0, _sum2);
#else
            int32x4_t _sum0 = vdupq_n_s32(0);
            int32x4_t _sum1 = vdupq_n_s32(0);
            int32x4_t _sum2 = vdupq_n_s32(0);
            int32x4_t _sum3 = vdupq_n_s32(0);
            for (; i + 7 < size; i += 8)
            {
#if NCNN_GNU_INLINE_ASM && !__aarch64__
                const signed char* xptr = x + i;

                asm volatile(
                    "vldm       %1!, {d0-d7}        \n"
                    "vld1.s8    {d16}, [%0]         \n"
                    "vdup.32    d17, d16[0]         \n"
                    "vdup.32    d16, d16[1]         \n"
                    "vmull.s8   q4, d0, d17         \n"
                    "vmull.s8   q5, d1, d17         \n"
                    "vmull.s8   q6, d2, d17         \n"
                    "vmull.s8   q7, d3, d17         \n"
                    "vmlal.s8   q4, d4, d16         \n"
                    "vmlal.s8   q5, d5, d16         \n"
                    "vmlal.s8   q6, d6, d16         \n"
                    "vmlal.s8   q7, d7, d16         \n"
                    "vpadal.s16 %q2, q4             \n"
                    "vpadal.s16 %q3, q5             \n"
                    "vpadal.s16 %q4, q6             \n"
                    "vpadal.s16 %q5, q7             \n"
                    : "=r"(xptr), "=r"(kptr), "=w"(_sum0), "=w"(_sum1), "=w"(_sum2), "=w"(_sum3)
                    : "0"(xptr), "1"(kptr), "2"(_sum0), "3"(_sum1), "4"(_sum2), "5"(_sum3)
                    : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8");
#else
                int32x2_t _xi01 = vreinterpret_s32_s8(vld1_s8(x + i));
                int8x8_t _xi0 = vreinterpret_s8_s32(vdup_lane_s32(_xi01, 0));
                int8x8_t _xi1 = vreinterpret_s8_s32(vdup_lane_s32(_xi01, 1));
                int8x16_t _w0 = vld1q_s8(kptr);
                int8x16_t _w1 = vld1q_s8(kptr + 16);
                int8x16_t _w2 = vld1q_s8(kptr + 32);
                int8x16_t _w3 = vld1q_s8(kptr + 48);

                int16x8_t _s0 = vmull_s8(vget_low_s8(_w0), _xi0);
                int16x8_t _s1 = vmull_s8(vget_high_s8(_w0), _xi0);
                int16x8_t _s2 = vmull_s8(vget_low_s8(_w1), _xi0);
                int16x8_t _s3 = vmull_s8(vget_high_s8(_w1), _xi0);
                _s0 = vmlal_s8(_s0, vget_low_s8(_w2), _xi1);
                _s1 = vmlal_s8(_s1, vget_high_s8(_w2), _xi1);
                _s2 = vmlal_s8(_s2, vget_low_s8(_w3), _xi1);
                _s3 = vmlal_s8(_s3, vget_high_s8(_w3), _xi1);

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);

                kptr += 64;
#endif
            }
            {
                int32x2_t _s0 = vpadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                int32x2_t _s1 = vpadd_s32(vget_low_s32(_sum1), vget_high_s32(_sum1));
                int32x2_t _s2 = vpadd_s32(vget_low_s32(_sum2), vget_high_s32(_sum2));
                int32x2_t _s3 = vpadd_s32(vget_low_s32(_sum3), vget_high_s32(_sum3));
                _gru_Rx0 = vaddq_s32(_gru_Rx0, vcombine_s32(_s0, _s1));
                _gru_Ux0 = vaddq_s32(_gru_Ux0, vcombine_s32(_s2, _s3));
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; i + 3 < size; i += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _xi = vld1_s8(x + i);
                int8x16_t _w0 = vld1q_s8(kptr);
                int8x16_t _w1 = vld1q_s8(kptr + 16);
                _gru_Rx0 = vdotq_lane_s32(_gru_Rx0, _w0, _xi, 0);
                _gru_Ux0 = vdotq_lane_s32(_gru_Ux0, _w1, _xi, 0);
#else
                int16x4_t _xi01 = vreinterpret_s16_s8(vld1_s8(x + i));
                int8x8_t _xi0 = vreinterpret_s8_s16(vdup_lane_s16(_xi01, 0));
                int8x8_t _xi1 = vreinterpret_s8_s16(vdup_lane_s16(_xi01, 1));
                int8x16_t _weight_xc_RU0 = vld1q_s8(kptr);
                int8x16_t _weight_xc_RU1 = vld1q_s8(kptr + 16);

                int16x8_t _gru_Rx = vmull_s8(vget_low_s8(_weight_xc_RU0), _xi0);
                int16x8_t _gru_Ux = vmull_s8(vget_high_s8(_weight_xc_RU0), _xi0);
                _gru_Rx = vmlal_s8(_gru_Rx, vget_low_s8(_weight_xc_RU1), _xi1);
                _gru_Ux = vmlal_s8(_gru_Ux, vget_high_s8(_weight_xc_RU1), _xi1);

                _gru_Rx0 = vpadalq_s16(_gru_Rx0, _gru_Rx);
                _gru_Ux0 = vpadalq_s16(_gru_Ux0, _gru_Ux);
#endif // __ARM_FEATURE_DOTPROD

                kptr += 32;
            }
            for (; i + 1 < size; i += 2)
            {
                int8x8_t _xi = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vld1_s8(x + i)), 0));
                int8x16_t _weight_xc_RU = vld1q_s8(kptr);

                int16x8_t _gru_Rx = vmull_s8(vget_low_s8(_weight_xc_RU), _xi);
                int16x8_t _gru_Ux = vmull_s8(vget_high_s8(_weight_xc_RU), _xi);

                _gru_Rx0 = vpadalq_s16(_gru_Rx0, _gru_Rx);
                _gru_Ux0 = vpadalq_s16(_gru_Ux0, _gru_Ux);

                kptr += 16;
            }
            for (; i < size; i++)
            {
                int8x8_t _xi = vdup_n_s8(x[i]);
                int8x8_t _weight_xc_RU = vld1_s8(kptr);

                int16x8_t _gru_RxUx = vmull_s8(_weight_xc_RU, _xi);
                _gru_Rx0 = vaddw_s16(_gru_Rx0, vget_low_s16(_gru_RxUx));
                _gru_Ux0 = vaddw_s16(_gru_Ux0, vget_high_s16(_gru_RxUx));

                kptr += 8;
            }

            int32x4_t _gru_Rh0 = vdupq_n_s32(0);
            int32x4_t _gru_Uh0 = vdupq_n_s32(0);
            i = 0;
#if __ARM_FEATURE_DOTPROD
            _sum1 = vdupq_n_s32(0);
            _sum2 = vdupq_n_s32(0);
            for (; i + 7 < num_output; i += 8)
            {
                int8x8_t _h_cont = vld1_s8(hs + i);
                int8x16_t _w0 = vld1q_s8(kptr);
                int8x16_t _w1 = vld1q_s8(kptr + 16);
                int8x16_t _w2 = vld1q_s8(kptr + 32);
                int8x16_t _w3 = vld1q_s8(kptr + 48);
                _gru_Rh0 = vdotq_lane_s32(_gru_Rh0, _w0, _h_cont, 0);
                _gru_Uh0 = vdotq_lane_s32(_gru_Uh0, _w1, _h_cont, 0);
                _sum1 = vdotq_lane_s32(_sum1, _w2, _h_cont, 1);
                _sum2 = vdotq_lane_s32(_sum2, _w3, _h_cont, 1);

                kptr += 64;
            }
            _gru_Rh0 = vaddq_s32(_gru_Rh0, _sum1);
            _gru_Uh0 = vaddq_s32(_gru_Uh0, _sum2);
#else
            _sum0 = vdupq_n_s32(0);
            _sum1 = vdupq_n_s32(0);
            _sum2 = vdupq_n_s32(0);
            _sum3 = vdupq_n_s32(0);
            for (; i + 7 < num_output; i += 8)
            {
#if NCNN_GNU_INLINE_ASM && !__aarch64__
                const signed char* hsptr = hs + i;

                asm volatile(
                    "vldm       %1!, {d0-d7}        \n"
                    "vld1.s8    {d16}, [%0]         \n"
                    "vdup.32    d17, d16[0]         \n"
                    "vdup.32    d16, d16[1]         \n"
                    "vmull.s8   q4, d0, d17         \n"
                    "vmull.s8   q5, d1, d17         \n"
                    "vmull.s8   q6, d2, d17         \n"
                    "vmull.s8   q7, d3, d17         \n"
                    "vmlal.s8   q4, d4, d16         \n"
                    "vmlal.s8   q5, d5, d16         \n"
                    "vmlal.s8   q6, d6, d16         \n"
                    "vmlal.s8   q7, d7, d16         \n"
                    "vpadal.s16 %q2, q4             \n"
                    "vpadal.s16 %q3, q5             \n"
                    "vpadal.s16 %q4, q6             \n"
                    "vpadal.s16 %q5, q7             \n"
                    : "=r"(hsptr), "=r"(kptr), "=w"(_sum0), "=w"(_sum1), "=w"(_sum2), "=w"(_sum3)
                    : "0"(hsptr), "1"(kptr), "2"(_sum0), "3"(_sum1), "4"(_sum2), "5"(_sum3)
                    : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8");
#else
                int32x2_t _h_cont01 = vreinterpret_s32_s8(vld1_s8(hs + i));
                int8x8_t _h_cont0 = vreinterpret_s8_s32(vdup_lane_s32(_h_cont01, 0));
                int8x8_t _h_cont1 = vreinterpret_s8_s32(vdup_lane_s32(_h_cont01, 1));
                int8x16_t _w0 = vld1q_s8(kptr);
                int8x16_t _w1 = vld1q_s8(kptr + 16);
                int8x16_t _w2 = vld1q_s8(kptr + 32);
                int8x16_t _w3 = vld1q_s8(kptr + 48);

                int16x8_t _s0 = vmull_s8(vget_low_s8(_w0), _h_cont0);
                int16x8_t _s1 = vmull_s8(vget_high_s8(_w0), _h_cont0);
                int16x8_t _s2 = vmull_s8(vget_low_s8(_w1), _h_cont0);
                int16x8_t _s3 = vmull_s8(vget_high_s8(_w1), _h_cont0);
                _s0 = vmlal_s8(_s0, vget_low_s8(_w2), _h_cont1);
                _s1 = vmlal_s8(_s1, vget_high_s8(_w2), _h_cont1);
                _s2 = vmlal_s8(_s2, vget_low_s8(_w3), _h_cont1);
                _s3 = vmlal_s8(_s3, vget_high_s8(_w3), _h_cont1);

                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);

                kptr += 64;
#endif
            }
            {
                int32x2_t _s0 = vpadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                int32x2_t _s1 = vpadd_s32(vget_low_s32(_sum1), vget_high_s32(_sum1));
                int32x2_t _s2 = vpadd_s32(vget_low_s32(_sum2), vget_high_s32(_sum2));
                int32x2_t _s3 = vpadd_s32(vget_low_s32(_sum3), vget_high_s32(_sum3));
                _gru_Rh0 = vaddq_s32(_gru_Rh0, vcombine_s32(_s0, _s1));
                _gru_Uh0 = vaddq_s32(_gru_Uh0, vcombine_s32(_s2, _s3));
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; i + 3 < num_output; i += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _h_cont = vld1_s8(hs + i);
                int8x16_t _w0 = vld1q_s8(kptr);
                int8x16_t _w1 = vld1q_s8(kptr + 16);
                _gru_Rh0 = vdotq_lane_s32(_gru_Rh0, _w0, _h_cont, 0);
                _gru_Uh0 = vdotq_lane_s32(_gru_Uh0, _w1, _h_cont, 0);
#else
                int16x4_t _h_cont01 = vreinterpret_s16_s8(vld1_s8(hs + i));
                int8x8_t _h_cont0 = vreinterpret_s8_s16(vdup_lane_s16(_h_cont01, 0));
                int8x8_t _h_cont1 = vreinterpret_s8_s16(vdup_lane_s16(_h_cont01, 1));
                int8x16_t _weight_hc_RU0 = vld1q_s8(kptr);
                int8x16_t _weight_hc_RU1 = vld1q_s8(kptr + 16);

                int16x8_t _gru_Rh = vmull_s8(vget_low_s8(_weight_hc_RU0), _h_cont0);
                int16x8_t _gru_Uh = vmull_s8(vget_high_s8(_weight_hc_RU0), _h_cont0);
                _gru_Rh = vmlal_s8(_gru_Rh, vget_low_s8(_weight_hc_RU1), _h_cont1);
                _gru_Uh = vmlal_s8(_gru_Uh, vget_high_s8(_weight_hc_RU1), _h_cont1);

                _gru_Rh0 = vpadalq_s16(_gru_Rh0, _gru_Rh);
                _gru_Uh0 = vpadalq_s16(_gru_Uh0, _gru_Uh);
#endif // __ARM_FEATURE_DOTPROD

                kptr += 32;
            }
            for (; i + 1 < num_output; i += 2)
            {
                int8x8_t _h_cont = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vld1_s8(hs + i)), 0));
                int8x16_t _weight_hc_RU = vld1q_s8(kptr);

                int16x8_t _gru_Rh = vmull_s8(vget_low_s8(_weight_hc_RU), _h_cont);
                int16x8_t _gru_Uh = vmull_s8(vget_high_s8(_weight_hc_RU), _h_cont);

                _gru_Rh0 = vpadalq_s16(_gru_Rh0, _gru_Rh);
                _gru_Uh0 = vpadalq_s16(_gru_Uh0, _gru_Uh);

                kptr += 16;
            }
            for (; i < num_output; i++)
            {
                int8x8_t _h_cont = vdup_n_s8(hs[i]);
                int8x8_t _weight_hc_RU = vld1_s8(kptr);

                int16x8_t _gru_RhUh = vmull_s8(_weight_hc_RU, _h_cont);
                _gru_Rh0 = vaddw_s16(_gru_Rh0, vget_low_s16(_gru_RhUh));
                _gru_Uh0 = vaddw_s16(_gru_Uh0, vget_high_s16(_gru_RhUh));

                kptr += 8;
            }

            float32x4_t _descale_x = vdupq_n_f32(descale_x);
            float32x4_t _descale_h = vdupq_n_f32(descale_h);

            float32x4_t _gru_R0 = vld1q_f32(bias_c_RUBNWN);
            float32x4_t _gru_U0 = vld1q_f32(bias_c_RUBNWN + 4);

            float32x4_t _descale_xc_R0 = vld1q_f32(descales_ptr);
            float32x4_t _descale_xc_U0 = vld1q_f32(descales_ptr + 4);

            _gru_R0 = vmlaq_f32(_gru_R0, vcvtq_f32_s32(_gru_Rx0), vmulq_f32(_descale_x, _descale_xc_R0));
            _gru_U0 = vmlaq_f32(_gru_U0, vcvtq_f32_s32(_gru_Ux0), vmulq_f32(_descale_x, _descale_xc_U0));

            float32x4_t _descale_hc_R0 = vld1q_f32(descales_ptr + 8);
            float32x4_t _descale_hc_U0 = vld1q_f32(descales_ptr + 12);

            _gru_R0 = vmlaq_f32(_gru_R0, vcvtq_f32_s32(_gru_Rh0), vmulq_f32(_descale_h, _descale_hc_R0));
            _gru_U0 = vmlaq_f32(_gru_U0, vcvtq_f32_s32(_gru_Uh0), vmulq_f32(_descale_h, _descale_hc_U0));

            // sigmoid(R)
            // sigmoid(U)
            _gru_R0 = sigmoid_ps(_gru_R0);
            _gru_U0 = sigmoid_ps(_gru_U0);

            // gate new

            int32x4_t _gru_Nh0 = vdupq_n_s32(0);
            i = 0;
#if __ARM_FEATURE_DOTPROD
            _sum1 = vdupq_n_s32(0);
            for (; i + 7 < num_output; i += 8)
            {
                int8x8_t _h_cont = vld1_s8(hs + i);
                int8x16_t _w0 = vld1q_s8(kptr);
                int8x16_t _w1 = vld1q_s8(kptr + 16);
                _gru_Nh0 = vdotq_lane_s32(_gru_Nh0, _w0, _h_cont, 0);
                _sum1 = vdotq_lane_s32(_sum1, _w1, _h_cont, 1);

                kptr += 32;
            }
            _gru_Nh0 = vaddq_s32(_gru_Nh0, _sum1);
#else
            _sum0 = vdupq_n_s32(0);
            _sum1 = vdupq_n_s32(0);
            _sum2 = vdupq_n_s32(0);
            _sum3 = vdupq_n_s32(0);
            for (; i + 15 < num_output; i += 16)
            {
#if NCNN_GNU_INLINE_ASM && !__aarch64__
                const signed char* hsptr = hs + i;

                asm volatile(
                    "vldm       %1!, {d0-d7}        \n"
                    "vld1.s8    {d16-d17}, [%0]     \n"
                    "vmull.s8   q4, d0, d16         \n"
                    "vmull.s8   q5, d1, d16         \n"
                    "vmull.s8   q6, d2, d16         \n"
                    "vmull.s8   q7, d3, d16         \n"
                    "vmlal.s8   q4, d4, d17         \n"
                    "vmlal.s8   q5, d5, d17         \n"
                    "vmlal.s8   q6, d6, d17         \n"
                    "vmlal.s8   q7, d7, d17         \n"
                    "vpadal.s16 %q2, q4             \n"
                    "vpadal.s16 %q3, q5             \n"
                    "vpadal.s16 %q4, q6             \n"
                    "vpadal.s16 %q5, q7             \n"
                    : "=r"(hsptr), "=r"(kptr), "=w"(_sum0), "=w"(_sum1), "=w"(_sum2), "=w"(_sum3)
                    : "0"(hsptr), "1"(kptr), "2"(_sum0), "3"(_sum1), "4"(_sum2), "5"(_sum3)
                    : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8");
#else
                int8x16_t _h_cont = vld1q_s8(hs + i);
                int8x16_t _w0 = vld1q_s8(kptr);
                int8x16_t _w1 = vld1q_s8(kptr + 16);
                int8x16_t _w2 = vld1q_s8(kptr + 32);
                int8x16_t _w3 = vld1q_s8(kptr + 48);

                int16x8_t _s0 = vmull_s8(vget_low_s8(_w0), vget_low_s8(_h_cont));
                int16x8_t _s1 = vmull_s8(vget_high_s8(_w0), vget_low_s8(_h_cont));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_w1), vget_low_s8(_h_cont));
                int16x8_t _s3 = vmull_s8(vget_high_s8(_w1), vget_low_s8(_h_cont));
                _s0 = vmlal_s8(_s0, vget_low_s8(_w2), vget_high_s8(_h_cont));
                _s1 = vmlal_s8(_s1, vget_high_s8(_w2), vget_high_s8(_h_cont));
                _s2 = vmlal_s8(_s2, vget_low_s8(_w3), vget_high_s8(_h_cont));
                _s3 = vmlal_s8(_s3, vget_high_s8(_w3), vget_high_s8(_h_cont));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);

                kptr += 64;
#endif
            }
            for (; i + 7 < num_output; i += 8)
            {
                int8x8_t _h_cont = vld1_s8(hs + i);
                int8x16_t _w0 = vld1q_s8(kptr);
                int8x16_t _w1 = vld1q_s8(kptr + 16);

                int16x8_t _s0 = vmull_s8(vget_low_s8(_w0), _h_cont);
                int16x8_t _s1 = vmull_s8(vget_high_s8(_w0), _h_cont);
                int16x8_t _s2 = vmull_s8(vget_low_s8(_w1), _h_cont);
                int16x8_t _s3 = vmull_s8(vget_high_s8(_w1), _h_cont);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);

                kptr += 32;
            }
            {
                int32x4x2_t _tmp0 = vzipq_s32(_sum0, _sum1);
                int32x4x2_t _tmp1 = vzipq_s32(_sum2, _sum3);
                _sum0 = vcombine_s32(vget_low_s32(_tmp0.val[0]), vget_low_s32(_tmp1.val[0]));
                _sum1 = vcombine_s32(vget_high_s32(_tmp0.val[0]), vget_high_s32(_tmp1.val[0]));
                _sum2 = vcombine_s32(vget_low_s32(_tmp0.val[1]), vget_low_s32(_tmp1.val[1]));
                _sum3 = vcombine_s32(vget_high_s32(_tmp0.val[1]), vget_high_s32(_tmp1.val[1]));
            }
            _gru_Nh0 = vaddq_s32(_gru_Nh0, _sum0);
            _gru_Nh0 = vaddq_s32(_gru_Nh0, _sum1);
            _gru_Nh0 = vaddq_s32(_gru_Nh0, _sum2);
            _gru_Nh0 = vaddq_s32(_gru_Nh0, _sum3);
#endif // __ARM_FEATURE_DOTPROD
            for (; i + 3 < num_output; i += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _h_cont = vld1_s8(hs + i);
                int8x16_t _w = vld1q_s8(kptr);
                _gru_Nh0 = vdotq_lane_s32(_gru_Nh0, _w, _h_cont, 0);
#else
                int16x4_t _h_cont01 = vreinterpret_s16_s8(vld1_s8(hs + i));
                int8x8_t _h_cont0 = vreinterpret_s8_s16(vdup_lane_s16(_h_cont01, 0));
                int8x8_t _h_cont1 = vreinterpret_s8_s16(vdup_lane_s16(_h_cont01, 1));
                int8x16_t _w01 = vld1q_s8(kptr);

                int16x8_t _gru_Nh = vmull_s8(vget_low_s8(_w01), _h_cont0);
                _gru_Nh = vmlal_s8(_gru_Nh, vget_high_s8(_w01), _h_cont1);
                _gru_Nh0 = vpadalq_s16(_gru_Nh0, _gru_Nh);
#endif // __ARM_FEATURE_DOTPROD

                kptr += 16;
            }
            for (; i + 1 < num_output; i += 2)
            {
                int8x8_t _h_cont = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vld1_s8(hs + i)), 0));
                int8x8_t _w = vld1_s8(kptr);

                int16x8_t _gru_Nh = vmull_s8(_w, _h_cont);
                _gru_Nh0 = vpadalq_s16(_gru_Nh0, _gru_Nh);

                kptr += 8;
            }
            for (; i < num_output; i++)
            {
                int8x8_t _h_cont = vdup_n_s8(hs[i]);
                int8x8_t _w = vld1_s8(kptr);

                int16x8_t _gru_Nh = vmull_s8(_w, _h_cont);
                _gru_Nh0 = vaddw_s16(_gru_Nh0, vget_low_s16(_gru_Nh));

                kptr += 4;
            }

            int32x4_t _gru_Nx0 = vdupq_n_s32(0);
            i = 0;
#if __ARM_FEATURE_DOTPROD
            _sum1 = vdupq_n_s32(0);
            for (; i + 7 < size; i += 8)
            {
                int8x8_t _xi = vld1_s8(x + i);
                int8x16_t _w0 = vld1q_s8(kptr);
                int8x16_t _w1 = vld1q_s8(kptr + 16);
                _gru_Nx0 = vdotq_lane_s32(_gru_Nx0, _w0, _xi, 0);
                _sum1 = vdotq_lane_s32(_sum1, _w1, _xi, 1);

                kptr += 32;
            }
            _gru_Nx0 = vaddq_s32(_gru_Nx0, _sum1);
#else
            _sum0 = vdupq_n_s32(0);
            _sum1 = vdupq_n_s32(0);
            _sum2 = vdupq_n_s32(0);
            _sum3 = vdupq_n_s32(0);
            for (; i + 15 < size; i += 16)
            {
#if NCNN_GNU_INLINE_ASM && !__aarch64__
                const signed char* xptr = x + i;

                asm volatile(
                    "vldm       %1!, {d0-d7}        \n"
                    "vld1.s8    {d16-d17}, [%0]     \n"
                    "vmull.s8   q4, d0, d16         \n"
                    "vmull.s8   q5, d1, d16         \n"
                    "vmull.s8   q6, d2, d16         \n"
                    "vmull.s8   q7, d3, d16         \n"
                    "vmlal.s8   q4, d4, d17         \n"
                    "vmlal.s8   q5, d5, d17         \n"
                    "vmlal.s8   q6, d6, d17         \n"
                    "vmlal.s8   q7, d7, d17         \n"
                    "vpadal.s16 %q2, q4             \n"
                    "vpadal.s16 %q3, q5             \n"
                    "vpadal.s16 %q4, q6             \n"
                    "vpadal.s16 %q5, q7             \n"
                    : "=r"(xptr), "=r"(kptr), "=w"(_sum0), "=w"(_sum1), "=w"(_sum2), "=w"(_sum3)
                    : "0"(xptr), "1"(kptr), "2"(_sum0), "3"(_sum1), "4"(_sum2), "5"(_sum3)
                    : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8");
#else
                int8x16_t _xi = vld1q_s8(x + i);
                int8x16_t _w0 = vld1q_s8(kptr);
                int8x16_t _w1 = vld1q_s8(kptr + 16);
                int8x16_t _w2 = vld1q_s8(kptr + 32);
                int8x16_t _w3 = vld1q_s8(kptr + 48);

                int16x8_t _s0 = vmull_s8(vget_low_s8(_w0), vget_low_s8(_xi));
                int16x8_t _s1 = vmull_s8(vget_high_s8(_w0), vget_low_s8(_xi));
                int16x8_t _s2 = vmull_s8(vget_low_s8(_w1), vget_low_s8(_xi));
                int16x8_t _s3 = vmull_s8(vget_high_s8(_w1), vget_low_s8(_xi));
                _s0 = vmlal_s8(_s0, vget_low_s8(_w2), vget_high_s8(_xi));
                _s1 = vmlal_s8(_s1, vget_high_s8(_w2), vget_high_s8(_xi));
                _s2 = vmlal_s8(_s2, vget_low_s8(_w3), vget_high_s8(_xi));
                _s3 = vmlal_s8(_s3, vget_high_s8(_w3), vget_high_s8(_xi));
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);

                kptr += 64;
#endif
            }
            for (; i + 7 < size; i += 8)
            {
                int8x8_t _xi = vld1_s8(x + i);
                int8x16_t _w0 = vld1q_s8(kptr);
                int8x16_t _w1 = vld1q_s8(kptr + 16);

                int16x8_t _s0 = vmull_s8(vget_low_s8(_w0), _xi);
                int16x8_t _s1 = vmull_s8(vget_high_s8(_w0), _xi);
                int16x8_t _s2 = vmull_s8(vget_low_s8(_w1), _xi);
                int16x8_t _s3 = vmull_s8(vget_high_s8(_w1), _xi);
                _sum0 = vpadalq_s16(_sum0, _s0);
                _sum1 = vpadalq_s16(_sum1, _s1);
                _sum2 = vpadalq_s16(_sum2, _s2);
                _sum3 = vpadalq_s16(_sum3, _s3);

                kptr += 32;
            }
            {
                int32x4x2_t _tmp0 = vzipq_s32(_sum0, _sum1);
                int32x4x2_t _tmp1 = vzipq_s32(_sum2, _sum3);
                _sum0 = vcombine_s32(vget_low_s32(_tmp0.val[0]), vget_low_s32(_tmp1.val[0]));
                _sum1 = vcombine_s32(vget_high_s32(_tmp0.val[0]), vget_high_s32(_tmp1.val[0]));
                _sum2 = vcombine_s32(vget_low_s32(_tmp0.val[1]), vget_low_s32(_tmp1.val[1]));
                _sum3 = vcombine_s32(vget_high_s32(_tmp0.val[1]), vget_high_s32(_tmp1.val[1]));
            }
            _gru_Nx0 = vaddq_s32(_gru_Nx0, _sum0);
            _gru_Nx0 = vaddq_s32(_gru_Nx0, _sum1);
            _gru_Nx0 = vaddq_s32(_gru_Nx0, _sum2);
            _gru_Nx0 = vaddq_s32(_gru_Nx0, _sum3);
#endif // __ARM_FEATURE_DOTPROD
            for (; i + 3 < size; i += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _xi = vld1_s8(x + i);
                int8x16_t _w = vld1q_s8(kptr);
                _gru_Nx0 = vdotq_lane_s32(_gru_Nx0, _w, _xi, 0);
#else
                int16x4_t _xi01 = vreinterpret_s16_s8(vld1_s8(x + i));
                int8x8_t _xi0 = vreinterpret_s8_s16(vdup_lane_s16(_xi01, 0));
                int8x8_t _xi1 = vreinterpret_s8_s16(vdup_lane_s16(_xi01, 1));
                int8x16_t _w01 = vld1q_s8(kptr);

                int16x8_t _gru_Nx = vmull_s8(vget_low_s8(_w01), _xi0);
                _gru_Nx = vmlal_s8(_gru_Nx, vget_high_s8(_w01), _xi1);
                _gru_Nx0 = vpadalq_s16(_gru_Nx0, _gru_Nx);
#endif // __ARM_FEATURE_DOTPROD

                kptr += 16;
            }
            for (; i + 1 < size; i += 2)
            {
                int8x8_t _xi = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vld1_s8(x + i)), 0));
                int8x8_t _w = vld1_s8(kptr);

                int16x8_t _gru_Nx = vmull_s8(_w, _xi);
                _gru_Nx0 = vpadalq_s16(_gru_Nx0, _gru_Nx);

                kptr += 8;
            }
            for (; i < size; i++)
            {
                int8x8_t _xi = vdup_n_s8(x[i]);
                int8x8_t _w = vld1_s8(kptr);

                int16x8_t _gru_Nx = vmull_s8(_w, _xi);
                _gru_Nx0 = vaddw_s16(_gru_Nx0, vget_low_s16(_gru_Nx));

                kptr += 4;
            }

            float32x4_t _gru_N0 = vld1q_f32(bias_c_RUBNWN + 8);

            float32x4_t _descale_hc_N0 = vld1q_f32(descales_ptr + 16);

            _gru_N0 = vmlaq_f32(_gru_N0, vcvtq_f32_s32(_gru_Nh0), vmulq_f32(_descale_h, _descale_hc_N0));

            _gru_N0 = vmlaq_f32(vld1q_f32(bias_c_RUBNWN + 12), _gru_R0, _gru_N0);

            float32x4_t _descale_xc_N0 = vld1q_f32(descales_ptr + 20);

            _gru_N0 = vmlaq_f32(_gru_N0, vcvtq_f32_s32(_gru_Nx0), vmulq_f32(_descale_x, _descale_xc_N0));

            // tanh(N)
            _gru_N0 = tanh_ps(_gru_N0);

            float* gates_data = gates.row(q / 4);

            vst1q_f32(gates_data, _gru_U0);
            vst1q_f32(gates_data + 4, _gru_N0);
        }
#endif // __ARM_NEON
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            const signed char* x = bottom_blob_int8.row<const signed char>(ti);
            const signed char* hs = hidden_state_int8;
            const float descale_x = bottom_blob_int8_descales[ti];
            const float descale_h = hidden_state_int8_descale;

            // gate reset update
            const float* bias_c_RUBNWN = (const float*)bias_c + q * 4;

#if __ARM_NEON
            const signed char* kptr = weight_data_tm.row<const signed char>(q / 4 + q % 4);
            const float* descales_ptr = weight_data_tm_int8_descales.row(q / 4 + q % 4);
#else
            const signed char* kptr = weight_data_tm.row<const signed char>(q);
            const float* descales_ptr = weight_data_tm_int8_descales.row(q);
#endif

            const float descale_xc_R = descales_ptr[0];
            const float descale_xc_U = descales_ptr[1];
            const float descale_hc_R = descales_ptr[2];
            const float descale_hc_U = descales_ptr[3];
            const float descale_hc_N = descales_ptr[4];
            const float descale_xc_N = descales_ptr[5];

            int Rx = 0;
            int Ux = 0;
            for (int i = 0; i < size; i++)
            {
                signed char xi = x[i];

                Rx += kptr[0] * xi;
                Ux += kptr[1] * xi;

                kptr += 2;
            }

            int Rh = 0;
            int Uh = 0;
            for (int i = 0; i < num_output; i++)
            {
                signed char h_cont = hs[i];

                Rh += kptr[0] * h_cont;
                Uh += kptr[1] * h_cont;

                kptr += 2;
            }

            float R = bias_c_RUBNWN[0] + Rx * (descale_x * descale_xc_R) + Rh * (descale_h * descale_hc_R);
            float U = bias_c_RUBNWN[1] + Ux * (descale_x * descale_xc_U) + Uh * (descale_h * descale_hc_U);

            // sigmoid(R)
            // sigmoid(U)
            R = 1.f / (1.f + expf(-R));
            U = 1.f / (1.f + expf(-U));

            // gate new

            int Nh = 0;
            for (int i = 0; i < num_output; i++)
            {
                Nh += kptr[0] * hs[i];
                kptr += 1;
            }

            int Nx = 0;
            for (int i = 0; i < size; i++)
            {
                Nx += kptr[0] * x[i];
                kptr += 1;
            }

            float N = bias_c_RUBNWN[2] + Nh * (descale_h * descale_hc_N);
            N = bias_c_RUBNWN[3] + R * N + Nx * (descale_x * descale_xc_N);

            // tanh(N)
            N = tanhf(N);

#if __ARM_NEON
            float* gates_data = gates.row(q / 4 + q % 4);
#else
            float* gates_data = gates.row(q);
#endif

            gates_data[0] = U;
            gates_data[1] = N;
        }

        gru_int8_gate_output(gates, hidden_state, top_blob, ti, elemtype, opt);
    }
}
