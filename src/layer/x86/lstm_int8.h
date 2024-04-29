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

static void lstm_transform_weight_int8(const Mat& weight_xc, const Mat& weight_xc_int8_scales, const Mat& weight_hc, const Mat& weight_hc_int8_scales, const Mat& bias_c, Mat& weight_data_tm, Mat& weight_data_tm_int8_descales, Mat& bias_c_tm, int size, int num_output, int num_directions, int hidden_size, const Option& opt)
{
    // TODO dispatch

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
#if __SSE2__
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
#endif // __SSE2__
            for (; i < size; i++)
            {
                kptr[0] = weight_xc_I[i];
                kptr[1] = weight_xc_F[i];
                kptr[2] = weight_xc_O[i];
                kptr[3] = weight_xc_G[i];
                kptr += 4;
            }

            i = 0;
#if __SSE2__
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
#endif // __SSE2__
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

static void lstm_int8(const Mat& bottom_blob_int8, const Mat& bottom_blob_int8_descales, Mat& top_blob, int reverse, const Mat& weight_data_tm, const Mat& weight_data_tm_int8_descales, const Mat& bias_c, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
    // TODO dispatch

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

#if __SSE2__
            __m128i _lstm_IFOGx0 = _mm_setzero_si128();
            int i = 0;
            for (; i + 1 < size; i += 2)
            {
                __m128i _xi = _mm_set1_epi16(((const short*)(x + i))[0]);
                __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);

#if __SSE4_1__
                _w = _mm_cvtepi8_epi16(_w);
                _xi = _mm_cvtepi8_epi16(_xi);
#else
                _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
                _xi = _mm_unpacklo_epi8(_xi, _mm_cmpgt_epi8(_mm_setzero_si128(), _xi));
#endif

#if __XOP__
                _lstm_IFOGx0 = _mm_maddd_epi16(_w, _xi, _lstm_IFOGx0);
#else
                _lstm_IFOGx0 = _mm_add_epi32(_lstm_IFOGx0, _mm_madd_epi16(_w, _xi));
#endif

                kptr += 8;
            }
            for (; i < size; i++)
            {
                __m128i _xi = _mm_set1_epi16(x[i]);
                __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);

#if __SSE4_1__
                _w = _mm_cvtepi8_epi16(_w);
#else
                _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
#endif

#if __XOP__
                _w = _mm_unpacklo_epi16(_w, _w);

                _lstm_IFOGx0 = _mm_maccd_epi16(_w, _xi, _lstm_IFOGx0);
#else
                __m128i _sl = _mm_mullo_epi16(_w, _xi);
                __m128i _sh = _mm_mulhi_epi16(_w, _xi);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);

                _lstm_IFOGx0 = _mm_add_epi32(_lstm_IFOGx0, _s0);
#endif

                kptr += 4;
            }

            __m128i _lstm_IFOGh0 = _mm_setzero_si128();
            i = 0;
            for (; i + 1 < num_output; i += 2)
            {
                __m128i _h_cont = _mm_set1_epi16(((const short*)(hs + i))[0]);
                __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);

#if __SSE4_1__
                _w = _mm_cvtepi8_epi16(_w);
                _h_cont = _mm_cvtepi8_epi16(_h_cont);
#else
                _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
                _h_cont = _mm_unpacklo_epi8(_h_cont, _mm_cmpgt_epi8(_mm_setzero_si128(), _h_cont));
#endif

#if __XOP__
                _lstm_IFOGh0 = _mm_maddd_epi16(_w, _h_cont, _lstm_IFOGh0);
#else
                _lstm_IFOGh0 = _mm_add_epi32(_lstm_IFOGh0, _mm_madd_epi16(_w, _h_cont));
#endif

                kptr += 8;
            }
            for (; i < num_output; i++)
            {
                __m128i _h_cont = _mm_set1_epi16(hs[i]);
                __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);

#if __SSE4_1__
                _w = _mm_cvtepi8_epi16(_w);
#else
                _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
#endif

#if __XOP__
                _w = _mm_unpacklo_epi16(_w, _w);

                _lstm_IFOGh0 = _mm_maccd_epi16(_w, _h_cont, _lstm_IFOGh0);
#else
                __m128i _sl = _mm_mullo_epi16(_w, _h_cont);
                __m128i _sh = _mm_mulhi_epi16(_w, _h_cont);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);

                _lstm_IFOGh0 = _mm_add_epi32(_lstm_IFOGh0, _s0);
#endif

                kptr += 4;
            }

            __m128 _descale_x = _mm_set1_ps(descale_x);
            __m128 _descale_h = _mm_set1_ps(descale_h);

            __m128 _lstm_IFOG0 = _mm_loadu_ps(bias_c_IFOG);

            __m128 _descale_xc_IFOG = _mm_loadu_ps(descales_ptr);

            _lstm_IFOG0 = _mm_add_ps(_lstm_IFOG0, _mm_mul_ps(_mm_cvtepi32_ps(_lstm_IFOGx0), _mm_mul_ps(_descale_x, _descale_xc_IFOG)));

            __m128 _descale_hc_IFOG = _mm_loadu_ps(descales_ptr + 4);

            _lstm_IFOG0 = _mm_add_ps(_lstm_IFOG0, _mm_mul_ps(_mm_cvtepi32_ps(_lstm_IFOGh0), _mm_mul_ps(_descale_h, _descale_hc_IFOG)));

            _mm_storeu_ps(gates_data, _lstm_IFOG0);
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

        int remain_hidden_size_start = 0;
#if __SSE2__
        int nn_hidden_size = hidden_size >> 2;
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

                hidden_ptr[q] = H;
                output_data[q] = H;
            }
        }
    }
}
