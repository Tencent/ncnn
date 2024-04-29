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

static void lstm_transform_weight_int8(const Mat& weight_xc, const Mat& weight_xc_int8_scales, const Mat& weight_hc, const Mat& weight_hc_int8_scales, const Mat& bias_c, Mat& weight_data_tm, Mat& weight_data_tm_int8_descales, Mat& bias_c_tm, int size, int num_output, int num_directions, int hidden_size, const Option& opt)
{
    // TODO dispatch for __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    // TODO dispatch for __ARM_FEATURE_DOTPROD

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

            for (int i = 0; i < size; i++)
            {
                kptr[0] = weight_xc_I[i];
                kptr[1] = weight_xc_F[i];
                kptr[2] = weight_xc_O[i];
                kptr[3] = weight_xc_G[i];
                kptr += 4;
            }

            for (int i = 0; i < num_output; i++)
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

static void lstm_int8(const Mat& bottom_blob_int8, const Mat& bottom_blob_int8_descales, Mat& top_blob, int elemtype, int reverse, const Mat& weight_data_tm, const Mat& weight_data_tm_int8_descales, const Mat& bias_c, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    // TODO dispatch for __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    // TODO dispatch for __ARM_FEATURE_DOTPROD

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

            const float descale_xc_I = descales_ptr[0];
            const float descale_xc_F = descales_ptr[1];
            const float descale_xc_O = descales_ptr[2];
            const float descale_xc_G = descales_ptr[3];
            const float descale_hc_I = descales_ptr[4];
            const float descale_hc_F = descales_ptr[5];
            const float descale_hc_O = descales_ptr[6];
            const float descale_hc_G = descales_ptr[7];

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

            float I = bias_c_IFOG[0] + Ix * (descale_x * descale_xc_I) + Ih * (descale_h * descale_hc_I);
            float F = bias_c_IFOG[1] + Fx * (descale_x * descale_xc_F) + Fh * (descale_h * descale_hc_F);
            float O = bias_c_IFOG[2] + Ox * (descale_x * descale_xc_O) + Oh * (descale_h * descale_hc_O);
            float G = bias_c_IFOG[3] + Gx * (descale_x * descale_xc_G) + Gh * (descale_h * descale_hc_G);

            float* gates_data = gates.row(q);

            gates_data[0] = I;
            gates_data[1] = F;
            gates_data[2] = O;
            gates_data[3] = G;
        }

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
                    vst1_u16((unsigned short*)output_data + q, (uint16x4_t)vcvt_f16_f32(_lstm_H));
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
}
