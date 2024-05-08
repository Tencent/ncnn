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
void lstm_transform_weight_int8_asimddp(const Mat& weight_xc, const Mat& weight_xc_int8_scales, const Mat& weight_hc, const Mat& weight_hc_int8_scales, const Mat& bias_c, Mat& weight_data_tm, Mat& weight_data_tm_int8_descales, Mat& bias_c_tm, int size, int num_output, int num_directions, int hidden_size, const Option& opt);
void lstm_int8_asimddp(const Mat& bottom_blob_int8, const Mat& bottom_blob_int8_descales, Mat& top_blob, int elemtype, int reverse, const Mat& weight_data_tm, const Mat& weight_data_tm_int8_descales, const Mat& bias_c, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_VFPV4 && __ARM_NEON && !(__ARM_FP & 2)
void lstm_int8_gate_output_vfpv4(const Mat& gates, const Mat& weight_hr, Mat& hidden_state, Mat& tmp_hidden_state, Mat& cell_state, Mat& top_blob, int ti, int elemtype, const Option& opt);
#endif

static void lstm_transform_weight_int8(const Mat& weight_xc, const Mat& weight_xc_int8_scales, const Mat& weight_hc, const Mat& weight_hc_int8_scales, const Mat& bias_c, Mat& weight_data_tm, Mat& weight_data_tm_int8_descales, Mat& bias_c_tm, int size, int num_output, int num_directions, int hidden_size, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
    if (ncnn::cpu_support_arm_asimddp())
    {
        lstm_transform_weight_int8_asimddp(weight_xc, weight_xc_int8_scales, weight_hc, weight_hc_int8_scales, bias_c, weight_data_tm, weight_data_tm_int8_descales, bias_c_tm, size, num_output, num_directions, hidden_size, opt);
        return;
    }
#endif

    weight_data_tm.create(size + num_output, hidden_size, num_directions, 4u, 4);
    weight_data_tm_int8_descales.create(4 + 4, hidden_size, num_directions);
    bias_c_tm.create(hidden_size, 1, num_directions, 16u, 4);

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

        const float* bias_c_I = bias_c_dr.row(0);
        const float* bias_c_F = bias_c_dr.row(1);
        const float* bias_c_O = bias_c_dr.row(2);
        const float* bias_c_G = bias_c_dr.row(3);

        float* bias_c_IFOG = bias_c_tm_dr.row(0);

        int q = 0;
        for (; q < hidden_size; q++)
        {
            bias_c_IFOG[0] = bias_c_I[q];
            bias_c_IFOG[1] = bias_c_F[q];
            bias_c_IFOG[2] = bias_c_O[q];
            bias_c_IFOG[3] = bias_c_G[q];

            bias_c_IFOG += 4;

            const signed char* weight_xc_I = weight_xc_dr.row<const signed char>(hidden_size * 0 + q);
            const signed char* weight_xc_F = weight_xc_dr.row<const signed char>(hidden_size * 1 + q);
            const signed char* weight_xc_O = weight_xc_dr.row<const signed char>(hidden_size * 2 + q);
            const signed char* weight_xc_G = weight_xc_dr.row<const signed char>(hidden_size * 3 + q);

            const signed char* weight_hc_I = weight_hc_dr.row<const signed char>(hidden_size * 0 + q);
            const signed char* weight_hc_F = weight_hc_dr.row<const signed char>(hidden_size * 1 + q);
            const signed char* weight_hc_O = weight_hc_dr.row<const signed char>(hidden_size * 2 + q);
            const signed char* weight_hc_G = weight_hc_dr.row<const signed char>(hidden_size * 3 + q);

            signed char* kptr = weight_data_tm_dr.row<signed char>(q);
            float* descales_ptr = weight_data_tm_int8_descales_dr.row(q);

            int i = 0;
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
            for (; i + 3 < size; i += 4)
            {
                kptr[0] = weight_xc_I[i];
                kptr[1] = weight_xc_I[i + 1];
                kptr[2] = weight_xc_I[i + 2];
                kptr[3] = weight_xc_I[i + 3];
                kptr[4] = weight_xc_F[i];
                kptr[5] = weight_xc_F[i + 1];
                kptr[6] = weight_xc_F[i + 2];
                kptr[7] = weight_xc_F[i + 3];
                kptr[8 + 0] = weight_xc_O[i];
                kptr[8 + 1] = weight_xc_O[i + 1];
                kptr[8 + 2] = weight_xc_O[i + 2];
                kptr[8 + 3] = weight_xc_O[i + 3];
                kptr[8 + 4] = weight_xc_G[i];
                kptr[8 + 5] = weight_xc_G[i + 1];
                kptr[8 + 6] = weight_xc_G[i + 2];
                kptr[8 + 7] = weight_xc_G[i + 3];
                kptr += 16;
            }
#else
            for (; i + 7 < size; i += 8)
            {
                vst1_s8(kptr, vld1_s8(weight_xc_I + i));
                vst1_s8(kptr + 8, vld1_s8(weight_xc_F + i));
                vst1_s8(kptr + 16, vld1_s8(weight_xc_O + i));
                vst1_s8(kptr + 24, vld1_s8(weight_xc_G + i));
                kptr += 32;
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; i + 1 < size; i += 2)
            {
                kptr[0] = weight_xc_I[i];
                kptr[1] = weight_xc_I[i + 1];
                kptr[2] = weight_xc_F[i];
                kptr[3] = weight_xc_F[i + 1];
                kptr[4] = weight_xc_O[i];
                kptr[5] = weight_xc_O[i + 1];
                kptr[6] = weight_xc_G[i];
                kptr[7] = weight_xc_G[i + 1];
                kptr += 8;
            }
#endif // __ARM_NEON
            for (; i < size; i++)
            {
                kptr[0] = weight_xc_I[i];
                kptr[1] = weight_xc_F[i];
                kptr[2] = weight_xc_O[i];
                kptr[3] = weight_xc_G[i];
                kptr += 4;
            }

            i = 0;
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
            for (; i + 3 < num_output; i += 4)
            {
                kptr[0] = weight_hc_I[i];
                kptr[1] = weight_hc_I[i + 1];
                kptr[2] = weight_hc_I[i + 2];
                kptr[3] = weight_hc_I[i + 3];
                kptr[4] = weight_hc_F[i];
                kptr[5] = weight_hc_F[i + 1];
                kptr[6] = weight_hc_F[i + 2];
                kptr[7] = weight_hc_F[i + 3];
                kptr[8 + 0] = weight_hc_O[i];
                kptr[8 + 1] = weight_hc_O[i + 1];
                kptr[8 + 2] = weight_hc_O[i + 2];
                kptr[8 + 3] = weight_hc_O[i + 3];
                kptr[8 + 4] = weight_hc_G[i];
                kptr[8 + 5] = weight_hc_G[i + 1];
                kptr[8 + 6] = weight_hc_G[i + 2];
                kptr[8 + 7] = weight_hc_G[i + 3];
                kptr += 16;
            }
#else
            for (; i + 7 < num_output; i += 8)
            {
                vst1_s8(kptr, vld1_s8(weight_hc_I + i));
                vst1_s8(kptr + 8, vld1_s8(weight_hc_F + i));
                vst1_s8(kptr + 16, vld1_s8(weight_hc_O + i));
                vst1_s8(kptr + 24, vld1_s8(weight_hc_G + i));
                kptr += 32;
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; i + 1 < num_output; i += 2)
            {
                kptr[0] = weight_hc_I[i];
                kptr[1] = weight_hc_I[i + 1];
                kptr[2] = weight_hc_F[i];
                kptr[3] = weight_hc_F[i + 1];
                kptr[4] = weight_hc_O[i];
                kptr[5] = weight_hc_O[i + 1];
                kptr[6] = weight_hc_G[i];
                kptr[7] = weight_hc_G[i + 1];
                kptr += 8;
            }
#endif // __ARM_NEON
            for (; i < num_output; i++)
            {
                kptr[0] = weight_hc_I[i];
                kptr[1] = weight_hc_F[i];
                kptr[2] = weight_hc_O[i];
                kptr[3] = weight_hc_G[i];
                kptr += 4;
            }

            descales_ptr[0] = 1.f / weight_xc_int8_scales_ptr[hidden_size * 0 + q];
            descales_ptr[1] = 1.f / weight_xc_int8_scales_ptr[hidden_size * 1 + q];
            descales_ptr[2] = 1.f / weight_xc_int8_scales_ptr[hidden_size * 2 + q];
            descales_ptr[3] = 1.f / weight_xc_int8_scales_ptr[hidden_size * 3 + q];
            descales_ptr[4] = 1.f / weight_hc_int8_scales_ptr[hidden_size * 0 + q];
            descales_ptr[5] = 1.f / weight_hc_int8_scales_ptr[hidden_size * 1 + q];
            descales_ptr[6] = 1.f / weight_hc_int8_scales_ptr[hidden_size * 2 + q];
            descales_ptr[7] = 1.f / weight_hc_int8_scales_ptr[hidden_size * 3 + q];
        }
    }
}

static void lstm_int8_gate_output(const Mat& gates, const Mat& weight_hr, Mat& hidden_state, Mat& tmp_hidden_state, Mat& cell_state, Mat& top_blob, int ti, int elemtype, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_VFPV4 && __ARM_NEON && !(__ARM_FP & 2)
    if (ncnn::cpu_support_arm_vfpv4())
    {
        lstm_int8_gate_output_vfpv4(gates, weight_hr, hidden_state, tmp_hidden_state, cell_state, top_blob, ti, elemtype, opt);
        return;
    }
#endif

    const int num_output = top_blob.w;
    const int hidden_size = cell_state.w;

    // lstm unit
    // sigmoid(I)
    // sigmoid(F)
    // sigmoid(O)
    // tanh(G)
    // c_t := f_t .* c_{t-1} + i_t .* g_t
    // h_t := o_t .* tanh[c_t]
    float* output_data = top_blob.row(ti);

    float* cell_ptr = cell_state;
    float* hidden_ptr = hidden_state;
    float* tmp_hidden_ptr = tmp_hidden_state;

    int remain_hidden_size_start = 0;
#if __ARM_NEON
    int nn_hidden_size = hidden_size >> 2;
    remain_hidden_size_start = nn_hidden_size << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int qq = 0; qq < nn_hidden_size; qq++)
    {
        int q = qq * 4;

        const float* gates_data = gates.row(q);

        float32x4x4_t _IFOG_4x4 = vld4q_f32(gates_data);

        float32x4_t _lstm_I = sigmoid_ps(_IFOG_4x4.val[0]);
        float32x4_t _lstm_F = sigmoid_ps(_IFOG_4x4.val[1]);
        float32x4_t _lstm_O = sigmoid_ps(_IFOG_4x4.val[2]);
        float32x4_t _lstm_G = tanh_ps(_IFOG_4x4.val[3]);

        float32x4_t _cell2 = vaddq_f32(vmulq_f32(_lstm_F, vld1q_f32(cell_ptr + q)), vmulq_f32(_lstm_I, _lstm_G));
        float32x4_t _lstm_H = vmulq_f32(_lstm_O, tanh_ps(_cell2));

        vst1q_f32(cell_ptr + q, _cell2);

        if (num_output == hidden_size)
        {
            vst1q_f32(hidden_ptr + q, _lstm_H);

            if (elemtype == 1)
            {
                // fp32
                vst1q_f32(output_data + q, _lstm_H);
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
                    "w"(_lstm_H)
                    : "memory", "v0");
#else  // __aarch64__
                asm volatile(
                    "vcvt.f16.f32 d0, %q2       \n"
                    "vst1.u16   {d0}, [%0]      \n"
                    : "=r"(outptr) // %0
                    : "0"(outptr),
                    "w"(_lstm_H)
                    : "memory", "q0");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
                vst1_u16(outptr, (uint16x4_t)vcvt_f16_f32(_lstm_H));
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
                vst1_u16((unsigned short*)output_data + q, float2bfloat(_lstm_H));
            }
        }
        else
        {
            vst1q_f32(tmp_hidden_ptr + q, _lstm_H);
        }
    }
#endif // __ARM_NEON
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = remain_hidden_size_start; q < hidden_size; q++)
    {
        const float* gates_data = gates.row(q);

        float I = gates_data[0];
        float F = gates_data[1];
        float O = gates_data[2];
        float G = gates_data[3];

        I = 1.f / (1.f + expf(-I));
        F = 1.f / (1.f + expf(-F));
        O = 1.f / (1.f + expf(-O));
        G = tanhf(G);

        float cell2 = F * cell_ptr[q] + I * G;
        float H = O * tanhf(cell2);

        cell_ptr[q] = cell2;
        if (num_output == hidden_size)
        {
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
        else
        {
            tmp_hidden_ptr[q] = H;
        }
    }

    if (num_output != hidden_size)
    {
        // int nn_num_output = num_output >> 2;
        // int remain_num_output_start = nn_num_output << 2;
        // #pragma omp parallel for num_threads(opt.num_threads)
        // for (int qq = 0; qq < nn_num_output; qq++)
        // {
        //     int q = qq * 4;
        //
        // }
        int remain_num_output_start = 0;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_num_output_start; q < num_output; q++)
        {
            const float* hr = weight_hr.row(q);
            const float* tmp_hidden_ptr = tmp_hidden_state;

            float H = 0;
            for (int i = 0; i < hidden_size; i++)
            {
                H += tmp_hidden_ptr[i] * hr[i];
            }

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

static void lstm_int8(const Mat& bottom_blob_int8, const Mat& bottom_blob_int8_descales, Mat& top_blob, int elemtype, int reverse, const Mat& weight_data_tm, const Mat& weight_data_tm_int8_descales, const Mat& bias_c, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD
    if (ncnn::cpu_support_arm_asimddp())
    {
        lstm_int8_asimddp(bottom_blob_int8, bottom_blob_int8_descales, top_blob, elemtype, reverse, weight_data_tm, weight_data_tm_int8_descales, bias_c, weight_hr, hidden_state, cell_state, opt);
        return;
    }
#endif

    int size = bottom_blob_int8.w;
    int T = bottom_blob_int8.h;

    int num_output = top_blob.w;
    int hidden_size = cell_state.w;

    // 4 x hidden_size
    Mat gates(4, hidden_size, 4u, opt.workspace_allocator);

    Mat tmp_hidden_state;
    if (num_output != hidden_size)
    {
        tmp_hidden_state.create(hidden_size, 4u, opt.workspace_allocator);
    }

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

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < hidden_size; q++)
        {
            const signed char* x = bottom_blob_int8.row<const signed char>(ti);
            const signed char* hs = hidden_state_int8;
            const float descale_x = bottom_blob_int8_descales[ti];
            const float descale_h = hidden_state_int8_descale;

            // gate reset update
            const float* bias_c_IFOG = (const float*)bias_c + q * 4;

            const signed char* kptr = weight_data_tm.row<const signed char>(q);
            const float* descales_ptr = weight_data_tm_int8_descales.row(q);

            float* gates_data = gates.row(q);

#if __ARM_NEON
            int32x4_t _lstm_IFOGx0 = vdupq_n_s32(0);
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
                _lstm_IFOGx0 = vdotq_laneq_s32(_lstm_IFOGx0, _w0, _xi, 0);
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
                _lstm_IFOGx0 = vdotq_lane_s32(_lstm_IFOGx0, _w0, _xi, 0);
                _sum1 = vdotq_lane_s32(_sum1, _w1, _xi, 1);

                kptr += 32;
            }
            _lstm_IFOGx0 = vaddq_s32(_lstm_IFOGx0, _sum1);
            _lstm_IFOGx0 = vaddq_s32(_lstm_IFOGx0, _sum2);
            _lstm_IFOGx0 = vaddq_s32(_lstm_IFOGx0, _sum3);
#else
            int32x4_t _sum0 = vdupq_n_s32(0);
            int32x4_t _sum1 = vdupq_n_s32(0);
            int32x4_t _sum2 = vdupq_n_s32(0);
            int32x4_t _sum3 = vdupq_n_s32(0);
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
            _lstm_IFOGx0 = vaddq_s32(_lstm_IFOGx0, _sum0);
            _lstm_IFOGx0 = vaddq_s32(_lstm_IFOGx0, _sum1);
            _lstm_IFOGx0 = vaddq_s32(_lstm_IFOGx0, _sum2);
            _lstm_IFOGx0 = vaddq_s32(_lstm_IFOGx0, _sum3);
#endif // __ARM_FEATURE_DOTPROD
            for (; i + 3 < size; i += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _xi = vld1_s8(x + i);
                int8x16_t _w = vld1q_s8(kptr);
                _lstm_IFOGx0 = vdotq_lane_s32(_lstm_IFOGx0, _w, _xi, 0);
#else
                int16x4_t _xi01 = vreinterpret_s16_s8(vld1_s8(x + i));
                int8x8_t _xi0 = vreinterpret_s8_s16(vdup_lane_s16(_xi01, 0));
                int8x8_t _xi1 = vreinterpret_s8_s16(vdup_lane_s16(_xi01, 1));
                int8x16_t _w01 = vld1q_s8(kptr);

                int16x8_t _lstm_IFOGx = vmull_s8(vget_low_s8(_w01), _xi0);
                _lstm_IFOGx = vmlal_s8(_lstm_IFOGx, vget_high_s8(_w01), _xi1);
                _lstm_IFOGx0 = vpadalq_s16(_lstm_IFOGx0, _lstm_IFOGx);
#endif // __ARM_FEATURE_DOTPROD

                kptr += 16;
            }
            for (; i + 1 < size; i += 2)
            {
                int8x8_t _xi = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vld1_s8(x + i)), 0));
                int8x8_t _w = vld1_s8(kptr);

                int16x8_t _lstm_IFOGx = vmull_s8(_w, _xi);
                _lstm_IFOGx0 = vpadalq_s16(_lstm_IFOGx0, _lstm_IFOGx);

                kptr += 8;
            }
            for (; i < size; i++)
            {
                int8x8_t _xi = vdup_n_s8(x[i]);
                int8x8_t _w = vld1_s8(kptr);

                int16x8_t _lstm_IFOGx = vmull_s8(_w, _xi);
                _lstm_IFOGx0 = vaddw_s16(_lstm_IFOGx0, vget_low_s16(_lstm_IFOGx));

                kptr += 4;
            }

            int32x4_t _lstm_IFOGh0 = vdupq_n_s32(0);
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
                _lstm_IFOGh0 = vdotq_laneq_s32(_lstm_IFOGh0, _w0, _h_cont, 0);
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
                _lstm_IFOGh0 = vdotq_lane_s32(_lstm_IFOGh0, _w0, _h_cont, 0);
                _sum1 = vdotq_lane_s32(_sum1, _w1, _h_cont, 1);

                kptr += 32;
            }
            _lstm_IFOGh0 = vaddq_s32(_lstm_IFOGh0, _sum1);
            _lstm_IFOGh0 = vaddq_s32(_lstm_IFOGh0, _sum2);
            _lstm_IFOGh0 = vaddq_s32(_lstm_IFOGh0, _sum3);
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
            _lstm_IFOGh0 = vaddq_s32(_lstm_IFOGh0, _sum0);
            _lstm_IFOGh0 = vaddq_s32(_lstm_IFOGh0, _sum1);
            _lstm_IFOGh0 = vaddq_s32(_lstm_IFOGh0, _sum2);
            _lstm_IFOGh0 = vaddq_s32(_lstm_IFOGh0, _sum3);
#endif // __ARM_FEATURE_DOTPROD
            for (; i + 3 < num_output; i += 4)
            {
#if __ARM_FEATURE_DOTPROD
                int8x8_t _h_cont = vld1_s8(hs + i);
                int8x16_t _w = vld1q_s8(kptr);
                _lstm_IFOGh0 = vdotq_lane_s32(_lstm_IFOGh0, _w, _h_cont, 0);
#else
                int16x4_t _h_cont01 = vreinterpret_s16_s8(vld1_s8(hs + i));
                int8x8_t _h_cont0 = vreinterpret_s8_s16(vdup_lane_s16(_h_cont01, 0));
                int8x8_t _h_cont1 = vreinterpret_s8_s16(vdup_lane_s16(_h_cont01, 1));
                int8x16_t _w01 = vld1q_s8(kptr);

                int16x8_t _lstm_IFOGh = vmull_s8(vget_low_s8(_w01), _h_cont0);
                _lstm_IFOGh = vmlal_s8(_lstm_IFOGh, vget_high_s8(_w01), _h_cont1);
                _lstm_IFOGh0 = vpadalq_s16(_lstm_IFOGh0, _lstm_IFOGh);
#endif // __ARM_FEATURE_DOTPROD

                kptr += 16;
            }
            for (; i + 1 < num_output; i += 2)
            {
                int8x8_t _h_cont = vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(vld1_s8(hs + i)), 0));
                int8x8_t _w = vld1_s8(kptr);

                int16x8_t _lstm_IFOGh = vmull_s8(_w, _h_cont);
                _lstm_IFOGh0 = vpadalq_s16(_lstm_IFOGh0, _lstm_IFOGh);

                kptr += 8;
            }
            for (; i < num_output; i++)
            {
                int8x8_t _h_cont = vdup_n_s8(hs[i]);
                int8x8_t _w = vld1_s8(kptr);

                int16x8_t _lstm_IFOGh = vmull_s8(_w, _h_cont);
                _lstm_IFOGh0 = vaddw_s16(_lstm_IFOGh0, vget_low_s16(_lstm_IFOGh));

                kptr += 4;
            }

            float32x4_t _descale_x = vdupq_n_f32(descale_x);
            float32x4_t _descale_h = vdupq_n_f32(descale_h);

            float32x4_t _lstm_IFOG0 = vld1q_f32(bias_c_IFOG);

            float32x4_t _descale_xc_IFOG = vld1q_f32(descales_ptr);

            _lstm_IFOG0 = vmlaq_f32(_lstm_IFOG0, vcvtq_f32_s32(_lstm_IFOGx0), vmulq_f32(_descale_x, _descale_xc_IFOG));

            float32x4_t _descale_hc_IFOG = vld1q_f32(descales_ptr + 4);

            _lstm_IFOG0 = vmlaq_f32(_lstm_IFOG0, vcvtq_f32_s32(_lstm_IFOGh0), vmulq_f32(_descale_h, _descale_hc_IFOG));

            vst1q_f32(gates_data, _lstm_IFOG0);
#else
            int Ix = 0;
            int Fx = 0;
            int Ox = 0;
            int Gx = 0;
            for (int i = 0; i < size; i++)
            {
                signed char xi = x[i];

                Ix += kptr[0] * xi;
                Fx += kptr[1] * xi;
                Ox += kptr[2] * xi;
                Gx += kptr[3] * xi;

                kptr += 4;
            }

            int Ih = 0;
            int Fh = 0;
            int Oh = 0;
            int Gh = 0;
            for (int i = 0; i < num_output; i++)
            {
                signed char h_cont = hs[i];

                Ih += kptr[0] * h_cont;
                Fh += kptr[1] * h_cont;
                Oh += kptr[2] * h_cont;
                Gh += kptr[3] * h_cont;

                kptr += 4;
            }

            const float descale_xc_I = descales_ptr[0];
            const float descale_xc_F = descales_ptr[1];
            const float descale_xc_O = descales_ptr[2];
            const float descale_xc_G = descales_ptr[3];
            const float descale_hc_I = descales_ptr[4];
            const float descale_hc_F = descales_ptr[5];
            const float descale_hc_O = descales_ptr[6];
            const float descale_hc_G = descales_ptr[7];

            float I = bias_c_IFOG[0] + Ix * (descale_x * descale_xc_I) + Ih * (descale_h * descale_hc_I);
            float F = bias_c_IFOG[1] + Fx * (descale_x * descale_xc_F) + Fh * (descale_h * descale_hc_F);
            float O = bias_c_IFOG[2] + Ox * (descale_x * descale_xc_O) + Oh * (descale_h * descale_hc_O);
            float G = bias_c_IFOG[3] + Gx * (descale_x * descale_xc_G) + Gh * (descale_h * descale_hc_G);

            gates_data[0] = I;
            gates_data[1] = F;
            gates_data[2] = O;
            gates_data[3] = G;
#endif // __ARM_NEON
        }

        lstm_int8_gate_output(gates, weight_hr, hidden_state, tmp_hidden_state, cell_state, top_blob, ti, elemtype, opt);
    }
}
