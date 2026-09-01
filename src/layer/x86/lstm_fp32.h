// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
int lstm_fma(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
int lstm_fma4(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt);
#endif

static int lstm(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
        return lstm_fma(bottom_blob, top_blob, reverse, weight_xc, bias_c, weight_hc, weight_hr, hidden_state, cell_state, opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
        return lstm_fma4(bottom_blob, top_blob, reverse, weight_xc, bias_c, weight_hc, weight_hr, hidden_state, cell_state, opt);
#endif

    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;
    int hidden_size = cell_state.w;

    // 4 x hidden_size
    Mat gates(4, hidden_size, 4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    Mat tmp_hidden_state;
    if (num_output != hidden_size)
    {
        tmp_hidden_state.create(hidden_size, 4u, opt.workspace_allocator);
        if (tmp_hidden_state.empty())
            return -100;
    }

    // unroll
    for (int t = 0; t < T; t++)
    {
        // clip hidden by continuation indicator
        // h_cont_{t-1} = cont_t * h_{t-1}
        // h_cont_{t-1} = h_{t-1} if cont_t == 1
        //                0       otherwise
        // calculate hidden
        // gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c

        int ti = reverse ? T - 1 - t : t;

#if __AVX__
        int nn_hidden_size = hidden_size >> 1;
        int remain_hidden_size_start = nn_hidden_size << 1;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_hidden_size; qq++)
        {
            int q = qq * 2;

            const float* bias_c_IFOG = (const float*)bias_c + q * 4;

            // gate I F O G
            const float* weight_xc_IFOG = weight_xc.row(q / 2);
            const float* weight_hc_IFOG = weight_hc.row(q / 2);

            __m256 _IFOG = _mm256_loadu_ps(bias_c_IFOG);
            __m256 _sum1 = _mm256_setzero_ps();
            __m256 _sum2 = _mm256_setzero_ps();
            __m256 _sum3 = _mm256_setzero_ps();

            const float* x = bottom_blob.row(ti);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                __m256 _xi0 = _mm256_broadcast_ss(x);
                __m256 _xi1 = _mm256_broadcast_ss(x + 1);
                __m256 _xi2 = _mm256_broadcast_ss(x + 2);
                __m256 _xi3 = _mm256_broadcast_ss(x + 3);
                __m256 _weight_xc_IFOG0 = _mm256_loadu_ps(weight_xc_IFOG);
                __m256 _weight_xc_IFOG1 = _mm256_loadu_ps(weight_xc_IFOG + 8);
                __m256 _weight_xc_IFOG2 = _mm256_loadu_ps(weight_xc_IFOG + 16);
                __m256 _weight_xc_IFOG3 = _mm256_loadu_ps(weight_xc_IFOG + 24);
                _IFOG = _mm256_comp_fmadd_ps(_weight_xc_IFOG0, _xi0, _IFOG);
                _sum1 = _mm256_comp_fmadd_ps(_weight_xc_IFOG1, _xi1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_weight_xc_IFOG2, _xi2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_weight_xc_IFOG3, _xi3, _sum3);

                x += 4;
                weight_xc_IFOG += 32;
            }
            for (; i < size; i++)
            {
                __m256 _xi = _mm256_broadcast_ss(x);
                __m256 _weight_xc_IFOG = _mm256_loadu_ps(weight_xc_IFOG);
                _IFOG = _mm256_comp_fmadd_ps(_weight_xc_IFOG, _xi, _IFOG);

                x += 1;
                weight_xc_IFOG += 8;
            }

            const float* hidden_ptr = hidden_state;

            i = 0;
            for (; i + 3 < num_output; i += 4)
            {
                __m256 _h_cont0 = _mm256_broadcast_ss(hidden_ptr);
                __m256 _h_cont1 = _mm256_broadcast_ss(hidden_ptr + 1);
                __m256 _h_cont2 = _mm256_broadcast_ss(hidden_ptr + 2);
                __m256 _h_cont3 = _mm256_broadcast_ss(hidden_ptr + 3);
                __m256 _weight_hc_IFOG0 = _mm256_loadu_ps(weight_hc_IFOG);
                __m256 _weight_hc_IFOG1 = _mm256_loadu_ps(weight_hc_IFOG + 8);
                __m256 _weight_hc_IFOG2 = _mm256_loadu_ps(weight_hc_IFOG + 16);
                __m256 _weight_hc_IFOG3 = _mm256_loadu_ps(weight_hc_IFOG + 24);
                _IFOG = _mm256_comp_fmadd_ps(_weight_hc_IFOG0, _h_cont0, _IFOG);
                _sum1 = _mm256_comp_fmadd_ps(_weight_hc_IFOG1, _h_cont1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_weight_hc_IFOG2, _h_cont2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_weight_hc_IFOG3, _h_cont3, _sum3);

                hidden_ptr += 4;
                weight_hc_IFOG += 32;
            }
            for (; i < num_output; i++)
            {
                __m256 _h_cont = _mm256_broadcast_ss(hidden_ptr);
                __m256 _weight_hc_IFOG = _mm256_loadu_ps(weight_hc_IFOG);
                _IFOG = _mm256_comp_fmadd_ps(_weight_hc_IFOG, _h_cont, _IFOG);

                hidden_ptr += 1;
                weight_hc_IFOG += 8;
            }

            float* gates_data = gates.row(q);

            _IFOG = _mm256_add_ps(_IFOG, _sum1);
            _sum2 = _mm256_add_ps(_sum2, _sum3);
            _IFOG = _mm256_add_ps(_IFOG, _sum2);

            _mm256_storeu_ps(gates_data, _IFOG);
        }
#else
        int nn_hidden_size = 0;
        int remain_hidden_size_start = 0;
#endif // __AVX__

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_hidden_size_start; q < hidden_size; q++)
        {
            const float* bias_c_IFOG = (const float*)bias_c + q * 4;

            // gate I F O G
#if __AVX__
            const float* weight_xc_IFOG = weight_xc.row(q / 2 + q % 2);
            const float* weight_hc_IFOG = weight_hc.row(q / 2 + q % 2);
#else
            const float* weight_xc_IFOG = weight_xc.row(q);
            const float* weight_hc_IFOG = weight_hc.row(q);
#endif

#if __SSE2__
            __m128 _IFOG = _mm_loadu_ps(bias_c_IFOG);
            __m128 _sum1 = _mm_setzero_ps();
            __m128 _sum2 = _mm_setzero_ps();
            __m128 _sum3 = _mm_setzero_ps();
#else  // __SSE2__
            float I = bias_c_IFOG[0];
            float F = bias_c_IFOG[1];
            float O = bias_c_IFOG[2];
            float G = bias_c_IFOG[3];
#endif // __SSE2__

            const float* x = bottom_blob.row(ti);

            int i = 0;
#if __SSE2__
            for (; i + 3 < size; i += 4)
            {
                __m128 _xi0 = _mm_load1_ps(x);
                __m128 _xi1 = _mm_load1_ps(x + 1);
                __m128 _xi2 = _mm_load1_ps(x + 2);
                __m128 _xi3 = _mm_load1_ps(x + 3);
                __m128 _weight_xc_IFOG0 = _mm_loadu_ps(weight_xc_IFOG);
                __m128 _weight_xc_IFOG1 = _mm_loadu_ps(weight_xc_IFOG + 4);
                __m128 _weight_xc_IFOG2 = _mm_loadu_ps(weight_xc_IFOG + 8);
                __m128 _weight_xc_IFOG3 = _mm_loadu_ps(weight_xc_IFOG + 12);
                _IFOG = _mm_comp_fmadd_ps(_weight_xc_IFOG0, _xi0, _IFOG);
                _sum1 = _mm_comp_fmadd_ps(_weight_xc_IFOG1, _xi1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_weight_xc_IFOG2, _xi2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_weight_xc_IFOG3, _xi3, _sum3);

                x += 4;
                weight_xc_IFOG += 16;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
#if __SSE2__
                __m128 _xi = _mm_load1_ps(x);
                __m128 _weight_xc_IFOG = _mm_loadu_ps(weight_xc_IFOG);
                _IFOG = _mm_comp_fmadd_ps(_weight_xc_IFOG, _xi, _IFOG);
#else  // __SSE2__
                float xi = x[0];
                I += xi * weight_xc_IFOG[0];
                F += xi * weight_xc_IFOG[1];
                O += xi * weight_xc_IFOG[2];
                G += xi * weight_xc_IFOG[3];
#endif // __SSE2__

                x += 1;
                weight_xc_IFOG += 4;
            }

            const float* hidden_ptr = hidden_state;

            i = 0;
#if __SSE2__
            for (; i + 3 < num_output; i += 4)
            {
                __m128 _h_cont0 = _mm_load1_ps(hidden_ptr);
                __m128 _h_cont1 = _mm_load1_ps(hidden_ptr + 1);
                __m128 _h_cont2 = _mm_load1_ps(hidden_ptr + 2);
                __m128 _h_cont3 = _mm_load1_ps(hidden_ptr + 3);
                __m128 _weight_hc_IFOG0 = _mm_loadu_ps(weight_hc_IFOG);
                __m128 _weight_hc_IFOG1 = _mm_loadu_ps(weight_hc_IFOG + 4);
                __m128 _weight_hc_IFOG2 = _mm_loadu_ps(weight_hc_IFOG + 8);
                __m128 _weight_hc_IFOG3 = _mm_loadu_ps(weight_hc_IFOG + 12);
                _IFOG = _mm_comp_fmadd_ps(_weight_hc_IFOG0, _h_cont0, _IFOG);
                _sum1 = _mm_comp_fmadd_ps(_weight_hc_IFOG1, _h_cont1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_weight_hc_IFOG2, _h_cont2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_weight_hc_IFOG3, _h_cont3, _sum3);

                hidden_ptr += 4;
                weight_hc_IFOG += 16;
            }
#endif // __SSE2__
            for (; i < num_output; i++)
            {
#if __SSE2__
                __m128 _h_cont = _mm_load1_ps(hidden_ptr);
                __m128 _weight_hc_IFOG = _mm_loadu_ps(weight_hc_IFOG);
                _IFOG = _mm_comp_fmadd_ps(_weight_hc_IFOG, _h_cont, _IFOG);
#else  // __SSE2__
                float h_cont = hidden_ptr[0];
                I += h_cont * weight_hc_IFOG[0];
                F += h_cont * weight_hc_IFOG[1];
                O += h_cont * weight_hc_IFOG[2];
                G += h_cont * weight_hc_IFOG[3];
#endif // __SSE2__

                hidden_ptr += 1;
                weight_hc_IFOG += 4;
            }

            float* gates_data = gates.row(q);

#if __SSE2__
            _IFOG = _mm_add_ps(_IFOG, _sum1);
            _sum2 = _mm_add_ps(_sum2, _sum3);
            _IFOG = _mm_add_ps(_IFOG, _sum2);

            _mm_storeu_ps(gates_data, _IFOG);
#else  // __SSE2__
            gates_data[0] = I;
            gates_data[1] = F;
            gates_data[2] = O;
            gates_data[3] = G;
#endif // __SSE2__
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

#if __SSE2__
        nn_hidden_size = hidden_size >> 2;
        remain_hidden_size_start = nn_hidden_size << 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_hidden_size; qq++)
        {
            int q = qq * 4;

            const float* gates_data = gates.row(q);

            __m128 _IFOG_4x4_0 = _mm_loadu_ps(gates_data);
            __m128 _IFOG_4x4_1 = _mm_loadu_ps(gates_data + 4);
            __m128 _IFOG_4x4_2 = _mm_loadu_ps(gates_data + 8);
            __m128 _IFOG_4x4_3 = _mm_loadu_ps(gates_data + 12);

            _MM_TRANSPOSE4_PS(_IFOG_4x4_0, _IFOG_4x4_1, _IFOG_4x4_2, _IFOG_4x4_3);

            __m128 _lstm_I = sigmoid_sse(_IFOG_4x4_0);
            __m128 _lstm_F = sigmoid_sse(_IFOG_4x4_1);
            __m128 _lstm_O = sigmoid_sse(_IFOG_4x4_2);
            __m128 _lstm_G = tanh_sse(_IFOG_4x4_3);

            __m128 _cell2 = _mm_add_ps(_mm_mul_ps(_lstm_F, _mm_loadu_ps(cell_ptr + q)), _mm_mul_ps(_lstm_I, _lstm_G));
            __m128 _lstm_H = _mm_mul_ps(_lstm_O, tanh_sse(_cell2));

            _mm_storeu_ps(cell_ptr + q, _cell2);

            if (num_output == hidden_size)
            {
                _mm_storeu_ps(hidden_ptr + q, _lstm_H);
                _mm_storeu_ps(output_data + q, _lstm_H);
            }
            else
            {
                _mm_storeu_ps(tmp_hidden_ptr + q, _lstm_H);
            }
        }
#else  // __SSE2__
        remain_hidden_size_start = 0;
#endif // __SSE2__
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
                output_data[q] = H;
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

                output_data[q] = H;
                hidden_ptr[q] = H;
            }
        }
    }

    return 0;
}

