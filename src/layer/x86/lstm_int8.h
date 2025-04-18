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

#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
void lstm_transform_weight_int8_avx512vnni(const Mat& weight_xc, const Mat& weight_xc_int8_scales, const Mat& weight_hc, const Mat& weight_hc_int8_scales, const Mat& bias_c, Mat& weight_data_tm, Mat& weight_data_tm_int8_descales, Mat& bias_c_tm, int size, int num_output, int num_directions, int hidden_size, const Option& opt);
void lstm_dynamic_quantize_scale2int8_avx512vnni(const float* ptr, int size, float scale, signed char* outptr);
void lstm_int8_avx512vnni(const Mat& bottom_blob_int8, const Mat& bottom_blob_int8_descales, Mat& top_blob, int reverse, const Mat& weight_data_tm, const Mat& weight_data_tm_int8_descales, const Mat& bias_c, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVX512F__ && !__AVXVNNI__ && !__AVX512VNNI__
void lstm_transform_weight_int8_avxvnni(const Mat& weight_xc, const Mat& weight_xc_int8_scales, const Mat& weight_hc, const Mat& weight_hc_int8_scales, const Mat& bias_c, Mat& weight_data_tm, Mat& weight_data_tm_int8_descales, Mat& bias_c_tm, int size, int num_output, int num_directions, int hidden_size, const Option& opt);
void lstm_dynamic_quantize_scale2int8_avxvnni(const float* ptr, int size, float scale, signed char* outptr);
void lstm_int8_avxvnni(const Mat& bottom_blob_int8, const Mat& bottom_blob_int8_descales, Mat& top_blob, int reverse, const Mat& weight_data_tm, const Mat& weight_data_tm_int8_descales, const Mat& bias_c, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
void lstm_transform_weight_int8_avx2(const Mat& weight_xc, const Mat& weight_xc_int8_scales, const Mat& weight_hc, const Mat& weight_hc_int8_scales, const Mat& bias_c, Mat& weight_data_tm, Mat& weight_data_tm_int8_descales, Mat& bias_c_tm, int size, int num_output, int num_directions, int hidden_size, const Option& opt);
void lstm_int8_avx2(const Mat& bottom_blob_int8, const Mat& bottom_blob_int8_descales, Mat& top_blob, int reverse, const Mat& weight_data_tm, const Mat& weight_data_tm_int8_descales, const Mat& bias_c, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
void lstm_int8_xop(const Mat& bottom_blob_int8, const Mat& bottom_blob_int8_descales, Mat& top_blob, int reverse, const Mat& weight_data_tm, const Mat& weight_data_tm_int8_descales, const Mat& bias_c, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt);
#endif

static void lstm_transform_weight_int8(const Mat& weight_xc, const Mat& weight_xc_int8_scales, const Mat& weight_hc, const Mat& weight_hc_int8_scales, const Mat& bias_c, Mat& weight_data_tm, Mat& weight_data_tm_int8_descales, Mat& bias_c_tm, int size, int num_output, int num_directions, int hidden_size, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        lstm_transform_weight_int8_avx512vnni(weight_xc, weight_xc_int8_scales, weight_hc, weight_hc_int8_scales, bias_c, weight_data_tm, weight_data_tm_int8_descales, bias_c_tm, size, num_output, num_directions, hidden_size, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVX512F__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        lstm_transform_weight_int8_avxvnni(weight_xc, weight_xc_int8_scales, weight_hc, weight_hc_int8_scales, bias_c, weight_data_tm, weight_data_tm_int8_descales, bias_c_tm, size, num_output, num_directions, hidden_size, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        lstm_transform_weight_int8_avx2(weight_xc, weight_xc_int8_scales, weight_hc, weight_hc_int8_scales, bias_c, weight_data_tm, weight_data_tm_int8_descales, bias_c_tm, size, num_output, num_directions, hidden_size, opt);
        return;
    }
#endif

#if __AVX512F__
#if __AVX512VNNI__
    weight_data_tm.create(size + 4 + num_output + 4, hidden_size / 4 + hidden_size % 4, num_directions, 16u, 16);
#else
    weight_data_tm.create(size + num_output, hidden_size / 4 + hidden_size % 4, num_directions, 16u, 16);
#endif
    weight_data_tm_int8_descales.create(16 + 16, hidden_size / 4 + hidden_size % 4, num_directions);
#elif __AVX2__
#if __AVXVNNI__
    weight_data_tm.create(size + 4 + num_output + 4, hidden_size / 2 + hidden_size % 2, num_directions, 8u, 8);
#else
    weight_data_tm.create(size + num_output, hidden_size / 2 + hidden_size % 2, num_directions, 8u, 8);
#endif
    weight_data_tm_int8_descales.create(8 + 8, hidden_size / 2 + hidden_size % 2, num_directions);
#else
    weight_data_tm.create(size + num_output, hidden_size, num_directions, 4u, 4);
    weight_data_tm_int8_descales.create(4 + 4, hidden_size, num_directions);
#endif
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
#if __AVX2__
#if __AVX512F__
        for (; q + 3 < hidden_size; q += 4)
        {
            _mm_storeu_ps(bias_c_IFOG, _mm_loadu_ps(bias_c_I + q));
            _mm_storeu_ps(bias_c_IFOG + 4, _mm_loadu_ps(bias_c_F + q));
            _mm_storeu_ps(bias_c_IFOG + 8, _mm_loadu_ps(bias_c_O + q));
            _mm_storeu_ps(bias_c_IFOG + 12, _mm_loadu_ps(bias_c_G + q));
            bias_c_IFOG += 16;

            const signed char* weight_xc_I_0 = weight_xc_dr.row<const signed char>(hidden_size * 0 + q);
            const signed char* weight_xc_F_0 = weight_xc_dr.row<const signed char>(hidden_size * 1 + q);
            const signed char* weight_xc_O_0 = weight_xc_dr.row<const signed char>(hidden_size * 2 + q);
            const signed char* weight_xc_G_0 = weight_xc_dr.row<const signed char>(hidden_size * 3 + q);
            const signed char* weight_xc_I_1 = weight_xc_dr.row<const signed char>(hidden_size * 0 + q + 1);
            const signed char* weight_xc_F_1 = weight_xc_dr.row<const signed char>(hidden_size * 1 + q + 1);
            const signed char* weight_xc_O_1 = weight_xc_dr.row<const signed char>(hidden_size * 2 + q + 1);
            const signed char* weight_xc_G_1 = weight_xc_dr.row<const signed char>(hidden_size * 3 + q + 1);
            const signed char* weight_xc_I_2 = weight_xc_dr.row<const signed char>(hidden_size * 0 + q + 2);
            const signed char* weight_xc_F_2 = weight_xc_dr.row<const signed char>(hidden_size * 1 + q + 2);
            const signed char* weight_xc_O_2 = weight_xc_dr.row<const signed char>(hidden_size * 2 + q + 2);
            const signed char* weight_xc_G_2 = weight_xc_dr.row<const signed char>(hidden_size * 3 + q + 2);
            const signed char* weight_xc_I_3 = weight_xc_dr.row<const signed char>(hidden_size * 0 + q + 3);
            const signed char* weight_xc_F_3 = weight_xc_dr.row<const signed char>(hidden_size * 1 + q + 3);
            const signed char* weight_xc_O_3 = weight_xc_dr.row<const signed char>(hidden_size * 2 + q + 3);
            const signed char* weight_xc_G_3 = weight_xc_dr.row<const signed char>(hidden_size * 3 + q + 3);

            const signed char* weight_hc_I_0 = weight_hc_dr.row<const signed char>(hidden_size * 0 + q);
            const signed char* weight_hc_F_0 = weight_hc_dr.row<const signed char>(hidden_size * 1 + q);
            const signed char* weight_hc_O_0 = weight_hc_dr.row<const signed char>(hidden_size * 2 + q);
            const signed char* weight_hc_G_0 = weight_hc_dr.row<const signed char>(hidden_size * 3 + q);
            const signed char* weight_hc_I_1 = weight_hc_dr.row<const signed char>(hidden_size * 0 + q + 1);
            const signed char* weight_hc_F_1 = weight_hc_dr.row<const signed char>(hidden_size * 1 + q + 1);
            const signed char* weight_hc_O_1 = weight_hc_dr.row<const signed char>(hidden_size * 2 + q + 1);
            const signed char* weight_hc_G_1 = weight_hc_dr.row<const signed char>(hidden_size * 3 + q + 1);
            const signed char* weight_hc_I_2 = weight_hc_dr.row<const signed char>(hidden_size * 0 + q + 2);
            const signed char* weight_hc_F_2 = weight_hc_dr.row<const signed char>(hidden_size * 1 + q + 2);
            const signed char* weight_hc_O_2 = weight_hc_dr.row<const signed char>(hidden_size * 2 + q + 2);
            const signed char* weight_hc_G_2 = weight_hc_dr.row<const signed char>(hidden_size * 3 + q + 2);
            const signed char* weight_hc_I_3 = weight_hc_dr.row<const signed char>(hidden_size * 0 + q + 3);
            const signed char* weight_hc_F_3 = weight_hc_dr.row<const signed char>(hidden_size * 1 + q + 3);
            const signed char* weight_hc_O_3 = weight_hc_dr.row<const signed char>(hidden_size * 2 + q + 3);
            const signed char* weight_hc_G_3 = weight_hc_dr.row<const signed char>(hidden_size * 3 + q + 3);

            signed char* kptr = weight_data_tm_dr.row<signed char>(q / 4);
            float* descales_ptr = weight_data_tm_int8_descales_dr.row(q / 4);

            int i = 0;
#if __AVX512VNNI__
            __m512i _w_shift = _mm512_setzero_si512();
            __m512i _v127 = _mm512_set1_epi8(127);

            __m512i _w0_shift = _mm512_setzero_si512();
            __m512i _w1_shift = _mm512_setzero_si512();
#if defined(__x86_64__) || defined(_M_X64)
            __m512i _w2_shift = _mm512_setzero_si512();
            __m512i _w3_shift = _mm512_setzero_si512();
            for (; i + 15 < size; i += 16)
            {
                _mm_storeu_si128((__m128i*)kptr, _mm_loadu_si128((const __m128i*)(weight_xc_I_0 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16), _mm_loadu_si128((const __m128i*)(weight_xc_F_0 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 2), _mm_loadu_si128((const __m128i*)(weight_xc_O_0 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 3), _mm_loadu_si128((const __m128i*)(weight_xc_G_0 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 4), _mm_loadu_si128((const __m128i*)(weight_xc_I_1 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 5), _mm_loadu_si128((const __m128i*)(weight_xc_F_1 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 6), _mm_loadu_si128((const __m128i*)(weight_xc_O_1 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 7), _mm_loadu_si128((const __m128i*)(weight_xc_G_1 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 8), _mm_loadu_si128((const __m128i*)(weight_xc_I_2 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 9), _mm_loadu_si128((const __m128i*)(weight_xc_F_2 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 10), _mm_loadu_si128((const __m128i*)(weight_xc_O_2 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 11), _mm_loadu_si128((const __m128i*)(weight_xc_G_2 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 12), _mm_loadu_si128((const __m128i*)(weight_xc_I_3 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 13), _mm_loadu_si128((const __m128i*)(weight_xc_F_3 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 14), _mm_loadu_si128((const __m128i*)(weight_xc_O_3 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 15), _mm_loadu_si128((const __m128i*)(weight_xc_G_3 + i)));

                __m512i _w0 = _mm512_loadu_si512((const __m512i*)kptr);
                __m512i _w1 = _mm512_loadu_si512((const __m512i*)(kptr + 64));
                __m512i _w2 = _mm512_loadu_si512((const __m512i*)(kptr + 128));
                __m512i _w3 = _mm512_loadu_si512((const __m512i*)(kptr + 192));
                _w0_shift = _mm512_dpbusd_epi32(_w0_shift, _v127, _w0);
                _w1_shift = _mm512_dpbusd_epi32(_w1_shift, _v127, _w1);
                _w2_shift = _mm512_dpbusd_epi32(_w2_shift, _v127, _w2);
                _w3_shift = _mm512_dpbusd_epi32(_w3_shift, _v127, _w3);

                kptr += 256;
            }
            {
                __m512i _tmp0 = _mm512_unpacklo_epi32(_w0_shift, _w1_shift);
                __m512i _tmp1 = _mm512_unpackhi_epi32(_w0_shift, _w1_shift);
                __m512i _tmp2 = _mm512_unpacklo_epi32(_w2_shift, _w3_shift);
                __m512i _tmp3 = _mm512_unpackhi_epi32(_w2_shift, _w3_shift);
                _w0_shift = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                _w1_shift = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                _w2_shift = _mm512_unpacklo_epi64(_tmp1, _tmp3);
                _w3_shift = _mm512_unpackhi_epi64(_tmp1, _tmp3);

                _w_shift = _mm512_add_epi32(_w_shift, _w0_shift);
                _w_shift = _mm512_add_epi32(_w_shift, _w1_shift);
                _w_shift = _mm512_add_epi32(_w_shift, _w2_shift);
                _w_shift = _mm512_add_epi32(_w_shift, _w3_shift);
            }

            _w0_shift = _mm512_setzero_si512();
            _w1_shift = _mm512_setzero_si512();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < size; i += 8)
            {
                _mm_storel_epi64((__m128i*)kptr, _mm_loadl_epi64((const __m128i*)(weight_xc_I_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8), _mm_loadl_epi64((const __m128i*)(weight_xc_I_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 2), _mm_loadl_epi64((const __m128i*)(weight_xc_F_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 3), _mm_loadl_epi64((const __m128i*)(weight_xc_F_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 4), _mm_loadl_epi64((const __m128i*)(weight_xc_O_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 5), _mm_loadl_epi64((const __m128i*)(weight_xc_O_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 6), _mm_loadl_epi64((const __m128i*)(weight_xc_G_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 7), _mm_loadl_epi64((const __m128i*)(weight_xc_G_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 8), _mm_loadl_epi64((const __m128i*)(weight_xc_I_2 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 9), _mm_loadl_epi64((const __m128i*)(weight_xc_I_3 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 10), _mm_loadl_epi64((const __m128i*)(weight_xc_F_2 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 11), _mm_loadl_epi64((const __m128i*)(weight_xc_F_3 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 12), _mm_loadl_epi64((const __m128i*)(weight_xc_O_2 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 13), _mm_loadl_epi64((const __m128i*)(weight_xc_O_3 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 14), _mm_loadl_epi64((const __m128i*)(weight_xc_G_2 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 15), _mm_loadl_epi64((const __m128i*)(weight_xc_G_3 + i)));

                __m512i _w0 = _mm512_loadu_si512((const __m512i*)kptr);
                __m512i _w1 = _mm512_loadu_si512((const __m512i*)(kptr + 64));
                _w0_shift = _mm512_dpbusd_epi32(_w0_shift, _v127, _w0);
                _w1_shift = _mm512_dpbusd_epi32(_w1_shift, _v127, _w1);

                kptr += 128;
            }
            {
                __m512i _tmp0 = _mm512_castps_si512(_mm512_shuffle_ps(_mm512_castsi512_ps(_w0_shift), _mm512_castsi512_ps(_w1_shift), _MM_SHUFFLE(2, 0, 2, 0)));
                __m512i _tmp1 = _mm512_castps_si512(_mm512_shuffle_ps(_mm512_castsi512_ps(_w0_shift), _mm512_castsi512_ps(_w1_shift), _MM_SHUFFLE(3, 1, 3, 1)));

                _w_shift = _mm512_add_epi32(_w_shift, _tmp0);
                _w_shift = _mm512_add_epi32(_w_shift, _tmp1);
            }

            for (; i + 3 < size; i += 4)
            {
                kptr[0] = weight_xc_I_0[i];
                kptr[1] = weight_xc_I_0[i + 1];
                kptr[2] = weight_xc_I_0[i + 2];
                kptr[3] = weight_xc_I_0[i + 3];
                kptr[4] = weight_xc_I_1[i];
                kptr[5] = weight_xc_I_1[i + 1];
                kptr[6] = weight_xc_I_1[i + 2];
                kptr[7] = weight_xc_I_1[i + 3];
                kptr[8 + 0] = weight_xc_I_2[i];
                kptr[8 + 1] = weight_xc_I_2[i + 1];
                kptr[8 + 2] = weight_xc_I_2[i + 2];
                kptr[8 + 3] = weight_xc_I_2[i + 3];
                kptr[8 + 4] = weight_xc_I_3[i];
                kptr[8 + 5] = weight_xc_I_3[i + 1];
                kptr[8 + 6] = weight_xc_I_3[i + 2];
                kptr[8 + 7] = weight_xc_I_3[i + 3];
                kptr[16 + 0] = weight_xc_F_0[i];
                kptr[16 + 1] = weight_xc_F_0[i + 1];
                kptr[16 + 2] = weight_xc_F_0[i + 2];
                kptr[16 + 3] = weight_xc_F_0[i + 3];
                kptr[16 + 4] = weight_xc_F_1[i];
                kptr[16 + 5] = weight_xc_F_1[i + 1];
                kptr[16 + 6] = weight_xc_F_1[i + 2];
                kptr[16 + 7] = weight_xc_F_1[i + 3];
                kptr[24 + 0] = weight_xc_F_2[i];
                kptr[24 + 1] = weight_xc_F_2[i + 1];
                kptr[24 + 2] = weight_xc_F_2[i + 2];
                kptr[24 + 3] = weight_xc_F_2[i + 3];
                kptr[24 + 4] = weight_xc_F_3[i];
                kptr[24 + 5] = weight_xc_F_3[i + 1];
                kptr[24 + 6] = weight_xc_F_3[i + 2];
                kptr[24 + 7] = weight_xc_F_3[i + 3];
                kptr[32 + 0] = weight_xc_O_0[i];
                kptr[32 + 1] = weight_xc_O_0[i + 1];
                kptr[32 + 2] = weight_xc_O_0[i + 2];
                kptr[32 + 3] = weight_xc_O_0[i + 3];
                kptr[32 + 4] = weight_xc_O_1[i];
                kptr[32 + 5] = weight_xc_O_1[i + 1];
                kptr[32 + 6] = weight_xc_O_1[i + 2];
                kptr[32 + 7] = weight_xc_O_1[i + 3];
                kptr[40 + 0] = weight_xc_O_2[i];
                kptr[40 + 1] = weight_xc_O_2[i + 1];
                kptr[40 + 2] = weight_xc_O_2[i + 2];
                kptr[40 + 3] = weight_xc_O_2[i + 3];
                kptr[40 + 4] = weight_xc_O_3[i];
                kptr[40 + 5] = weight_xc_O_3[i + 1];
                kptr[40 + 6] = weight_xc_O_3[i + 2];
                kptr[40 + 7] = weight_xc_O_3[i + 3];
                kptr[48 + 0] = weight_xc_G_0[i];
                kptr[48 + 1] = weight_xc_G_0[i + 1];
                kptr[48 + 2] = weight_xc_G_0[i + 2];
                kptr[48 + 3] = weight_xc_G_0[i + 3];
                kptr[48 + 4] = weight_xc_G_1[i];
                kptr[48 + 5] = weight_xc_G_1[i + 1];
                kptr[48 + 6] = weight_xc_G_1[i + 2];
                kptr[48 + 7] = weight_xc_G_1[i + 3];
                kptr[56 + 0] = weight_xc_G_2[i];
                kptr[56 + 1] = weight_xc_G_2[i + 1];
                kptr[56 + 2] = weight_xc_G_2[i + 2];
                kptr[56 + 3] = weight_xc_G_2[i + 3];
                kptr[56 + 4] = weight_xc_G_3[i];
                kptr[56 + 5] = weight_xc_G_3[i + 1];
                kptr[56 + 6] = weight_xc_G_3[i + 2];
                kptr[56 + 7] = weight_xc_G_3[i + 3];

                __m512i _w = _mm512_loadu_si512((const __m512i*)kptr);
                _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _w);

                kptr += 64;
            }

            _mm512_storeu_si512((__m512i*)kptr, _w_shift);
            kptr += 64;
#else
#if defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < size; i += 8)
            {
                _mm_storel_epi64((__m128i*)kptr, _mm_loadl_epi64((const __m128i*)(weight_xc_I_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8), _mm_loadl_epi64((const __m128i*)(weight_xc_F_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 2), _mm_loadl_epi64((const __m128i*)(weight_xc_O_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 3), _mm_loadl_epi64((const __m128i*)(weight_xc_G_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 4), _mm_loadl_epi64((const __m128i*)(weight_xc_I_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 5), _mm_loadl_epi64((const __m128i*)(weight_xc_F_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 6), _mm_loadl_epi64((const __m128i*)(weight_xc_O_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 7), _mm_loadl_epi64((const __m128i*)(weight_xc_G_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 8), _mm_loadl_epi64((const __m128i*)(weight_xc_I_2 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 9), _mm_loadl_epi64((const __m128i*)(weight_xc_F_2 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 10), _mm_loadl_epi64((const __m128i*)(weight_xc_O_2 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 11), _mm_loadl_epi64((const __m128i*)(weight_xc_G_2 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 12), _mm_loadl_epi64((const __m128i*)(weight_xc_I_3 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 13), _mm_loadl_epi64((const __m128i*)(weight_xc_F_3 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 14), _mm_loadl_epi64((const __m128i*)(weight_xc_O_3 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 15), _mm_loadl_epi64((const __m128i*)(weight_xc_G_3 + i)));
                kptr += 128;
            }
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 3 < size; i += 4)
            {
                kptr[0] = weight_xc_I_0[i];
                kptr[1] = weight_xc_I_0[i + 1];
                kptr[2] = weight_xc_I_0[i + 2];
                kptr[3] = weight_xc_I_0[i + 3];
                kptr[4] = weight_xc_I_1[i];
                kptr[5] = weight_xc_I_1[i + 1];
                kptr[6] = weight_xc_I_1[i + 2];
                kptr[7] = weight_xc_I_1[i + 3];
                kptr[8 + 0] = weight_xc_F_0[i];
                kptr[8 + 1] = weight_xc_F_0[i + 1];
                kptr[8 + 2] = weight_xc_F_0[i + 2];
                kptr[8 + 3] = weight_xc_F_0[i + 3];
                kptr[8 + 4] = weight_xc_F_1[i];
                kptr[8 + 5] = weight_xc_F_1[i + 1];
                kptr[8 + 6] = weight_xc_F_1[i + 2];
                kptr[8 + 7] = weight_xc_F_1[i + 3];
                kptr[16 + 0] = weight_xc_O_0[i];
                kptr[16 + 1] = weight_xc_O_0[i + 1];
                kptr[16 + 2] = weight_xc_O_0[i + 2];
                kptr[16 + 3] = weight_xc_O_0[i + 3];
                kptr[16 + 4] = weight_xc_O_1[i];
                kptr[16 + 5] = weight_xc_O_1[i + 1];
                kptr[16 + 6] = weight_xc_O_1[i + 2];
                kptr[16 + 7] = weight_xc_O_1[i + 3];
                kptr[24 + 0] = weight_xc_G_0[i];
                kptr[24 + 1] = weight_xc_G_0[i + 1];
                kptr[24 + 2] = weight_xc_G_0[i + 2];
                kptr[24 + 3] = weight_xc_G_0[i + 3];
                kptr[24 + 4] = weight_xc_G_1[i];
                kptr[24 + 5] = weight_xc_G_1[i + 1];
                kptr[24 + 6] = weight_xc_G_1[i + 2];
                kptr[24 + 7] = weight_xc_G_1[i + 3];
                kptr[32 + 0] = weight_xc_I_2[i];
                kptr[32 + 1] = weight_xc_I_2[i + 1];
                kptr[32 + 2] = weight_xc_I_2[i + 2];
                kptr[32 + 3] = weight_xc_I_2[i + 3];
                kptr[32 + 4] = weight_xc_I_3[i];
                kptr[32 + 5] = weight_xc_I_3[i + 1];
                kptr[32 + 6] = weight_xc_I_3[i + 2];
                kptr[32 + 7] = weight_xc_I_3[i + 3];
                kptr[40 + 0] = weight_xc_F_2[i];
                kptr[40 + 1] = weight_xc_F_2[i + 1];
                kptr[40 + 2] = weight_xc_F_2[i + 2];
                kptr[40 + 3] = weight_xc_F_2[i + 3];
                kptr[40 + 4] = weight_xc_F_3[i];
                kptr[40 + 5] = weight_xc_F_3[i + 1];
                kptr[40 + 6] = weight_xc_F_3[i + 2];
                kptr[40 + 7] = weight_xc_F_3[i + 3];
                kptr[48 + 0] = weight_xc_O_2[i];
                kptr[48 + 1] = weight_xc_O_2[i + 1];
                kptr[48 + 2] = weight_xc_O_2[i + 2];
                kptr[48 + 3] = weight_xc_O_2[i + 3];
                kptr[48 + 4] = weight_xc_O_3[i];
                kptr[48 + 5] = weight_xc_O_3[i + 1];
                kptr[48 + 6] = weight_xc_O_3[i + 2];
                kptr[48 + 7] = weight_xc_O_3[i + 3];
                kptr[56 + 0] = weight_xc_G_2[i];
                kptr[56 + 1] = weight_xc_G_2[i + 1];
                kptr[56 + 2] = weight_xc_G_2[i + 2];
                kptr[56 + 3] = weight_xc_G_2[i + 3];
                kptr[56 + 4] = weight_xc_G_3[i];
                kptr[56 + 5] = weight_xc_G_3[i + 1];
                kptr[56 + 6] = weight_xc_G_3[i + 2];
                kptr[56 + 7] = weight_xc_G_3[i + 3];
                kptr += 64;
            }
#endif // __AVX512VNNI__
            for (; i + 1 < size; i += 2)
            {
                kptr[0] = weight_xc_I_0[i];
                kptr[1] = weight_xc_I_0[i + 1];
                kptr[2] = weight_xc_I_1[i];
                kptr[3] = weight_xc_I_1[i + 1];
                kptr[4] = weight_xc_I_2[i];
                kptr[5] = weight_xc_I_2[i + 1];
                kptr[6] = weight_xc_I_3[i];
                kptr[7] = weight_xc_I_3[i + 1];
                kptr[8 + 0] = weight_xc_F_0[i];
                kptr[8 + 1] = weight_xc_F_0[i + 1];
                kptr[8 + 2] = weight_xc_F_1[i];
                kptr[8 + 3] = weight_xc_F_1[i + 1];
                kptr[8 + 4] = weight_xc_F_2[i];
                kptr[8 + 5] = weight_xc_F_2[i + 1];
                kptr[8 + 6] = weight_xc_F_3[i];
                kptr[8 + 7] = weight_xc_F_3[i + 1];
                kptr[16 + 0] = weight_xc_O_0[i];
                kptr[16 + 1] = weight_xc_O_0[i + 1];
                kptr[16 + 2] = weight_xc_O_1[i];
                kptr[16 + 3] = weight_xc_O_1[i + 1];
                kptr[16 + 4] = weight_xc_O_2[i];
                kptr[16 + 5] = weight_xc_O_2[i + 1];
                kptr[16 + 6] = weight_xc_O_3[i];
                kptr[16 + 7] = weight_xc_O_3[i + 1];
                kptr[24 + 0] = weight_xc_G_0[i];
                kptr[24 + 1] = weight_xc_G_0[i + 1];
                kptr[24 + 2] = weight_xc_G_1[i];
                kptr[24 + 3] = weight_xc_G_1[i + 1];
                kptr[24 + 4] = weight_xc_G_2[i];
                kptr[24 + 5] = weight_xc_G_2[i + 1];
                kptr[24 + 6] = weight_xc_G_3[i];
                kptr[24 + 7] = weight_xc_G_3[i + 1];
                kptr += 32;
            }
            for (; i < size; i++)
            {
                kptr[0] = weight_xc_I_0[i];
                kptr[1] = weight_xc_I_1[i];
                kptr[2] = weight_xc_I_2[i];
                kptr[3] = weight_xc_I_3[i];
                kptr[4] = weight_xc_F_0[i];
                kptr[5] = weight_xc_F_1[i];
                kptr[6] = weight_xc_F_2[i];
                kptr[7] = weight_xc_F_3[i];
                kptr[8 + 0] = weight_xc_O_0[i];
                kptr[8 + 1] = weight_xc_O_1[i];
                kptr[8 + 2] = weight_xc_O_2[i];
                kptr[8 + 3] = weight_xc_O_3[i];
                kptr[8 + 4] = weight_xc_G_0[i];
                kptr[8 + 5] = weight_xc_G_1[i];
                kptr[8 + 6] = weight_xc_G_2[i];
                kptr[8 + 7] = weight_xc_G_3[i];
                kptr += 16;
            }

            i = 0;
#if __AVX512VNNI__
            _w_shift = _mm512_setzero_si512();
            _w0_shift = _mm512_setzero_si512();
            _w1_shift = _mm512_setzero_si512();
#if defined(__x86_64__) || defined(_M_X64)
            _w2_shift = _mm512_setzero_si512();
            _w3_shift = _mm512_setzero_si512();
            for (; i + 15 < num_output; i += 16)
            {
                _mm_storeu_si128((__m128i*)kptr, _mm_loadu_si128((const __m128i*)(weight_hc_I_0 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16), _mm_loadu_si128((const __m128i*)(weight_hc_F_0 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 2), _mm_loadu_si128((const __m128i*)(weight_hc_O_0 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 3), _mm_loadu_si128((const __m128i*)(weight_hc_G_0 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 4), _mm_loadu_si128((const __m128i*)(weight_hc_I_1 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 5), _mm_loadu_si128((const __m128i*)(weight_hc_F_1 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 6), _mm_loadu_si128((const __m128i*)(weight_hc_O_1 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 7), _mm_loadu_si128((const __m128i*)(weight_hc_G_1 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 8), _mm_loadu_si128((const __m128i*)(weight_hc_I_2 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 9), _mm_loadu_si128((const __m128i*)(weight_hc_F_2 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 10), _mm_loadu_si128((const __m128i*)(weight_hc_O_2 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 11), _mm_loadu_si128((const __m128i*)(weight_hc_G_2 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 12), _mm_loadu_si128((const __m128i*)(weight_hc_I_3 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 13), _mm_loadu_si128((const __m128i*)(weight_hc_F_3 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 14), _mm_loadu_si128((const __m128i*)(weight_hc_O_3 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 15), _mm_loadu_si128((const __m128i*)(weight_hc_G_3 + i)));

                __m512i _w0 = _mm512_loadu_si512((const __m512i*)kptr);
                __m512i _w1 = _mm512_loadu_si512((const __m512i*)(kptr + 64));
                __m512i _w2 = _mm512_loadu_si512((const __m512i*)(kptr + 128));
                __m512i _w3 = _mm512_loadu_si512((const __m512i*)(kptr + 192));
                _w0_shift = _mm512_dpbusd_epi32(_w0_shift, _v127, _w0);
                _w1_shift = _mm512_dpbusd_epi32(_w1_shift, _v127, _w1);
                _w2_shift = _mm512_dpbusd_epi32(_w2_shift, _v127, _w2);
                _w3_shift = _mm512_dpbusd_epi32(_w3_shift, _v127, _w3);

                kptr += 256;
            }
            {
                __m512i _tmp0 = _mm512_unpacklo_epi32(_w0_shift, _w1_shift);
                __m512i _tmp1 = _mm512_unpackhi_epi32(_w0_shift, _w1_shift);
                __m512i _tmp2 = _mm512_unpacklo_epi32(_w2_shift, _w3_shift);
                __m512i _tmp3 = _mm512_unpackhi_epi32(_w2_shift, _w3_shift);
                _w0_shift = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                _w1_shift = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                _w2_shift = _mm512_unpacklo_epi64(_tmp1, _tmp3);
                _w3_shift = _mm512_unpackhi_epi64(_tmp1, _tmp3);

                _w_shift = _mm512_add_epi32(_w_shift, _w0_shift);
                _w_shift = _mm512_add_epi32(_w_shift, _w1_shift);
                _w_shift = _mm512_add_epi32(_w_shift, _w2_shift);
                _w_shift = _mm512_add_epi32(_w_shift, _w3_shift);
            }

            _w0_shift = _mm512_setzero_si512();
            _w1_shift = _mm512_setzero_si512();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < num_output; i += 8)
            {
                _mm_storel_epi64((__m128i*)kptr, _mm_loadl_epi64((const __m128i*)(weight_hc_I_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8), _mm_loadl_epi64((const __m128i*)(weight_hc_I_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 2), _mm_loadl_epi64((const __m128i*)(weight_hc_F_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 3), _mm_loadl_epi64((const __m128i*)(weight_hc_F_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 4), _mm_loadl_epi64((const __m128i*)(weight_hc_O_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 5), _mm_loadl_epi64((const __m128i*)(weight_hc_O_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 6), _mm_loadl_epi64((const __m128i*)(weight_hc_G_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 7), _mm_loadl_epi64((const __m128i*)(weight_hc_G_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 8), _mm_loadl_epi64((const __m128i*)(weight_hc_I_2 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 9), _mm_loadl_epi64((const __m128i*)(weight_hc_I_3 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 10), _mm_loadl_epi64((const __m128i*)(weight_hc_F_2 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 11), _mm_loadl_epi64((const __m128i*)(weight_hc_F_3 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 12), _mm_loadl_epi64((const __m128i*)(weight_hc_O_2 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 13), _mm_loadl_epi64((const __m128i*)(weight_hc_O_3 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 14), _mm_loadl_epi64((const __m128i*)(weight_hc_G_2 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 15), _mm_loadl_epi64((const __m128i*)(weight_hc_G_3 + i)));

                __m512i _w0 = _mm512_loadu_si512((const __m512i*)kptr);
                __m512i _w1 = _mm512_loadu_si512((const __m512i*)(kptr + 64));
                _w0_shift = _mm512_dpbusd_epi32(_w0_shift, _v127, _w0);
                _w1_shift = _mm512_dpbusd_epi32(_w1_shift, _v127, _w1);

                kptr += 128;
            }
            {
                __m512i _tmp0 = _mm512_castps_si512(_mm512_shuffle_ps(_mm512_castsi512_ps(_w0_shift), _mm512_castsi512_ps(_w1_shift), _MM_SHUFFLE(2, 0, 2, 0)));
                __m512i _tmp1 = _mm512_castps_si512(_mm512_shuffle_ps(_mm512_castsi512_ps(_w0_shift), _mm512_castsi512_ps(_w1_shift), _MM_SHUFFLE(3, 1, 3, 1)));

                _w_shift = _mm512_add_epi32(_w_shift, _tmp0);
                _w_shift = _mm512_add_epi32(_w_shift, _tmp1);
            }

            for (; i + 3 < num_output; i += 4)
            {
                kptr[0] = weight_hc_I_0[i];
                kptr[1] = weight_hc_I_0[i + 1];
                kptr[2] = weight_hc_I_0[i + 2];
                kptr[3] = weight_hc_I_0[i + 3];
                kptr[4] = weight_hc_I_1[i];
                kptr[5] = weight_hc_I_1[i + 1];
                kptr[6] = weight_hc_I_1[i + 2];
                kptr[7] = weight_hc_I_1[i + 3];
                kptr[8 + 0] = weight_hc_I_2[i];
                kptr[8 + 1] = weight_hc_I_2[i + 1];
                kptr[8 + 2] = weight_hc_I_2[i + 2];
                kptr[8 + 3] = weight_hc_I_2[i + 3];
                kptr[8 + 4] = weight_hc_I_3[i];
                kptr[8 + 5] = weight_hc_I_3[i + 1];
                kptr[8 + 6] = weight_hc_I_3[i + 2];
                kptr[8 + 7] = weight_hc_I_3[i + 3];
                kptr[16 + 0] = weight_hc_F_0[i];
                kptr[16 + 1] = weight_hc_F_0[i + 1];
                kptr[16 + 2] = weight_hc_F_0[i + 2];
                kptr[16 + 3] = weight_hc_F_0[i + 3];
                kptr[16 + 4] = weight_hc_F_1[i];
                kptr[16 + 5] = weight_hc_F_1[i + 1];
                kptr[16 + 6] = weight_hc_F_1[i + 2];
                kptr[16 + 7] = weight_hc_F_1[i + 3];
                kptr[24 + 0] = weight_hc_F_2[i];
                kptr[24 + 1] = weight_hc_F_2[i + 1];
                kptr[24 + 2] = weight_hc_F_2[i + 2];
                kptr[24 + 3] = weight_hc_F_2[i + 3];
                kptr[24 + 4] = weight_hc_F_3[i];
                kptr[24 + 5] = weight_hc_F_3[i + 1];
                kptr[24 + 6] = weight_hc_F_3[i + 2];
                kptr[24 + 7] = weight_hc_F_3[i + 3];
                kptr[32 + 0] = weight_hc_O_0[i];
                kptr[32 + 1] = weight_hc_O_0[i + 1];
                kptr[32 + 2] = weight_hc_O_0[i + 2];
                kptr[32 + 3] = weight_hc_O_0[i + 3];
                kptr[32 + 4] = weight_hc_O_1[i];
                kptr[32 + 5] = weight_hc_O_1[i + 1];
                kptr[32 + 6] = weight_hc_O_1[i + 2];
                kptr[32 + 7] = weight_hc_O_1[i + 3];
                kptr[40 + 0] = weight_hc_O_2[i];
                kptr[40 + 1] = weight_hc_O_2[i + 1];
                kptr[40 + 2] = weight_hc_O_2[i + 2];
                kptr[40 + 3] = weight_hc_O_2[i + 3];
                kptr[40 + 4] = weight_hc_O_3[i];
                kptr[40 + 5] = weight_hc_O_3[i + 1];
                kptr[40 + 6] = weight_hc_O_3[i + 2];
                kptr[40 + 7] = weight_hc_O_3[i + 3];
                kptr[48 + 0] = weight_hc_G_0[i];
                kptr[48 + 1] = weight_hc_G_0[i + 1];
                kptr[48 + 2] = weight_hc_G_0[i + 2];
                kptr[48 + 3] = weight_hc_G_0[i + 3];
                kptr[48 + 4] = weight_hc_G_1[i];
                kptr[48 + 5] = weight_hc_G_1[i + 1];
                kptr[48 + 6] = weight_hc_G_1[i + 2];
                kptr[48 + 7] = weight_hc_G_1[i + 3];
                kptr[56 + 0] = weight_hc_G_2[i];
                kptr[56 + 1] = weight_hc_G_2[i + 1];
                kptr[56 + 2] = weight_hc_G_2[i + 2];
                kptr[56 + 3] = weight_hc_G_2[i + 3];
                kptr[56 + 4] = weight_hc_G_3[i];
                kptr[56 + 5] = weight_hc_G_3[i + 1];
                kptr[56 + 6] = weight_hc_G_3[i + 2];
                kptr[56 + 7] = weight_hc_G_3[i + 3];

                __m512i _w = _mm512_loadu_si512((const __m512i*)kptr);
                _w_shift = _mm512_dpbusd_epi32(_w_shift, _v127, _w);

                kptr += 64;
            }

            _mm512_storeu_si512((__m512i*)kptr, _w_shift);
            kptr += 64;
#else
#if defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < num_output; i += 8)
            {
                _mm_storel_epi64((__m128i*)kptr, _mm_loadl_epi64((const __m128i*)(weight_hc_I_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8), _mm_loadl_epi64((const __m128i*)(weight_hc_F_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 2), _mm_loadl_epi64((const __m128i*)(weight_hc_O_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 3), _mm_loadl_epi64((const __m128i*)(weight_hc_G_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 4), _mm_loadl_epi64((const __m128i*)(weight_hc_I_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 5), _mm_loadl_epi64((const __m128i*)(weight_hc_F_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 6), _mm_loadl_epi64((const __m128i*)(weight_hc_O_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 7), _mm_loadl_epi64((const __m128i*)(weight_hc_G_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 8), _mm_loadl_epi64((const __m128i*)(weight_hc_I_2 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 9), _mm_loadl_epi64((const __m128i*)(weight_hc_F_2 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 10), _mm_loadl_epi64((const __m128i*)(weight_hc_O_2 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 11), _mm_loadl_epi64((const __m128i*)(weight_hc_G_2 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 12), _mm_loadl_epi64((const __m128i*)(weight_hc_I_3 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 13), _mm_loadl_epi64((const __m128i*)(weight_hc_F_3 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 14), _mm_loadl_epi64((const __m128i*)(weight_hc_O_3 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8 * 15), _mm_loadl_epi64((const __m128i*)(weight_hc_G_3 + i)));
                kptr += 128;
            }
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 3 < num_output; i += 4)
            {
                kptr[0] = weight_hc_I_0[i];
                kptr[1] = weight_hc_I_0[i + 1];
                kptr[2] = weight_hc_I_0[i + 2];
                kptr[3] = weight_hc_I_0[i + 3];
                kptr[4] = weight_hc_I_1[i];
                kptr[5] = weight_hc_I_1[i + 1];
                kptr[6] = weight_hc_I_1[i + 2];
                kptr[7] = weight_hc_I_1[i + 3];
                kptr[8 + 0] = weight_hc_F_0[i];
                kptr[8 + 1] = weight_hc_F_0[i + 1];
                kptr[8 + 2] = weight_hc_F_0[i + 2];
                kptr[8 + 3] = weight_hc_F_0[i + 3];
                kptr[8 + 4] = weight_hc_F_1[i];
                kptr[8 + 5] = weight_hc_F_1[i + 1];
                kptr[8 + 6] = weight_hc_F_1[i + 2];
                kptr[8 + 7] = weight_hc_F_1[i + 3];
                kptr[16 + 0] = weight_hc_O_0[i];
                kptr[16 + 1] = weight_hc_O_0[i + 1];
                kptr[16 + 2] = weight_hc_O_0[i + 2];
                kptr[16 + 3] = weight_hc_O_0[i + 3];
                kptr[16 + 4] = weight_hc_O_1[i];
                kptr[16 + 5] = weight_hc_O_1[i + 1];
                kptr[16 + 6] = weight_hc_O_1[i + 2];
                kptr[16 + 7] = weight_hc_O_1[i + 3];
                kptr[24 + 0] = weight_hc_G_0[i];
                kptr[24 + 1] = weight_hc_G_0[i + 1];
                kptr[24 + 2] = weight_hc_G_0[i + 2];
                kptr[24 + 3] = weight_hc_G_0[i + 3];
                kptr[24 + 4] = weight_hc_G_1[i];
                kptr[24 + 5] = weight_hc_G_1[i + 1];
                kptr[24 + 6] = weight_hc_G_1[i + 2];
                kptr[24 + 7] = weight_hc_G_1[i + 3];
                kptr[32 + 0] = weight_hc_I_2[i];
                kptr[32 + 1] = weight_hc_I_2[i + 1];
                kptr[32 + 2] = weight_hc_I_2[i + 2];
                kptr[32 + 3] = weight_hc_I_2[i + 3];
                kptr[32 + 4] = weight_hc_I_3[i];
                kptr[32 + 5] = weight_hc_I_3[i + 1];
                kptr[32 + 6] = weight_hc_I_3[i + 2];
                kptr[32 + 7] = weight_hc_I_3[i + 3];
                kptr[40 + 0] = weight_hc_F_2[i];
                kptr[40 + 1] = weight_hc_F_2[i + 1];
                kptr[40 + 2] = weight_hc_F_2[i + 2];
                kptr[40 + 3] = weight_hc_F_2[i + 3];
                kptr[40 + 4] = weight_hc_F_3[i];
                kptr[40 + 5] = weight_hc_F_3[i + 1];
                kptr[40 + 6] = weight_hc_F_3[i + 2];
                kptr[40 + 7] = weight_hc_F_3[i + 3];
                kptr[48 + 0] = weight_hc_O_2[i];
                kptr[48 + 1] = weight_hc_O_2[i + 1];
                kptr[48 + 2] = weight_hc_O_2[i + 2];
                kptr[48 + 3] = weight_hc_O_2[i + 3];
                kptr[48 + 4] = weight_hc_O_3[i];
                kptr[48 + 5] = weight_hc_O_3[i + 1];
                kptr[48 + 6] = weight_hc_O_3[i + 2];
                kptr[48 + 7] = weight_hc_O_3[i + 3];
                kptr[56 + 0] = weight_hc_G_2[i];
                kptr[56 + 1] = weight_hc_G_2[i + 1];
                kptr[56 + 2] = weight_hc_G_2[i + 2];
                kptr[56 + 3] = weight_hc_G_2[i + 3];
                kptr[56 + 4] = weight_hc_G_3[i];
                kptr[56 + 5] = weight_hc_G_3[i + 1];
                kptr[56 + 6] = weight_hc_G_3[i + 2];
                kptr[56 + 7] = weight_hc_G_3[i + 3];
                kptr += 64;
            }
#endif // __AVX512VNNI__
            for (; i + 1 < num_output; i += 2)
            {
                kptr[0] = weight_hc_I_0[i];
                kptr[1] = weight_hc_I_0[i + 1];
                kptr[2] = weight_hc_I_1[i];
                kptr[3] = weight_hc_I_1[i + 1];
                kptr[4] = weight_hc_I_2[i];
                kptr[5] = weight_hc_I_2[i + 1];
                kptr[6] = weight_hc_I_3[i];
                kptr[7] = weight_hc_I_3[i + 1];
                kptr[8 + 0] = weight_hc_F_0[i];
                kptr[8 + 1] = weight_hc_F_0[i + 1];
                kptr[8 + 2] = weight_hc_F_1[i];
                kptr[8 + 3] = weight_hc_F_1[i + 1];
                kptr[8 + 4] = weight_hc_F_2[i];
                kptr[8 + 5] = weight_hc_F_2[i + 1];
                kptr[8 + 6] = weight_hc_F_3[i];
                kptr[8 + 7] = weight_hc_F_3[i + 1];
                kptr[16 + 0] = weight_hc_O_0[i];
                kptr[16 + 1] = weight_hc_O_0[i + 1];
                kptr[16 + 2] = weight_hc_O_1[i];
                kptr[16 + 3] = weight_hc_O_1[i + 1];
                kptr[16 + 4] = weight_hc_O_2[i];
                kptr[16 + 5] = weight_hc_O_2[i + 1];
                kptr[16 + 6] = weight_hc_O_3[i];
                kptr[16 + 7] = weight_hc_O_3[i + 1];
                kptr[24 + 0] = weight_hc_G_0[i];
                kptr[24 + 1] = weight_hc_G_0[i + 1];
                kptr[24 + 2] = weight_hc_G_1[i];
                kptr[24 + 3] = weight_hc_G_1[i + 1];
                kptr[24 + 4] = weight_hc_G_2[i];
                kptr[24 + 5] = weight_hc_G_2[i + 1];
                kptr[24 + 6] = weight_hc_G_3[i];
                kptr[24 + 7] = weight_hc_G_3[i + 1];
                kptr += 32;
            }
            for (; i < num_output; i++)
            {
                kptr[0] = weight_hc_I_0[i];
                kptr[1] = weight_hc_I_1[i];
                kptr[2] = weight_hc_I_2[i];
                kptr[3] = weight_hc_I_3[i];
                kptr[4] = weight_hc_F_0[i];
                kptr[5] = weight_hc_F_1[i];
                kptr[6] = weight_hc_F_2[i];
                kptr[7] = weight_hc_F_3[i];
                kptr[8 + 0] = weight_hc_O_0[i];
                kptr[8 + 1] = weight_hc_O_1[i];
                kptr[8 + 2] = weight_hc_O_2[i];
                kptr[8 + 3] = weight_hc_O_3[i];
                kptr[8 + 4] = weight_hc_G_0[i];
                kptr[8 + 5] = weight_hc_G_1[i];
                kptr[8 + 6] = weight_hc_G_2[i];
                kptr[8 + 7] = weight_hc_G_3[i];
                kptr += 16;
            }

            _mm_storeu_ps(bias_c_IFOG, _mm_loadu_ps(bias_c_I + q));

            __m128 _descale_xc_I = _mm_loadu_ps(weight_xc_int8_scales_ptr + hidden_size * 0 + q);
            __m128 _descale_xc_F = _mm_loadu_ps(weight_xc_int8_scales_ptr + hidden_size * 1 + q);
            __m128 _descale_xc_O = _mm_loadu_ps(weight_xc_int8_scales_ptr + hidden_size * 2 + q);
            __m128 _descale_xc_G = _mm_loadu_ps(weight_xc_int8_scales_ptr + hidden_size * 3 + q);
            __m128 _descale_hc_I = _mm_loadu_ps(weight_hc_int8_scales_ptr + hidden_size * 0 + q);
            __m128 _descale_hc_F = _mm_loadu_ps(weight_hc_int8_scales_ptr + hidden_size * 1 + q);
            __m128 _descale_hc_O = _mm_loadu_ps(weight_hc_int8_scales_ptr + hidden_size * 2 + q);
            __m128 _descale_hc_G = _mm_loadu_ps(weight_hc_int8_scales_ptr + hidden_size * 3 + q);

            __m512 _descale_xc_IFOG = _mm512_castps128_ps512(_descale_xc_I);
            _descale_xc_IFOG = _mm512_insertf32x4(_descale_xc_IFOG, _descale_xc_F, 1);
            _descale_xc_IFOG = _mm512_insertf32x4(_descale_xc_IFOG, _descale_xc_O, 2);
            _descale_xc_IFOG = _mm512_insertf32x4(_descale_xc_IFOG, _descale_xc_G, 3);
            __m512 _descale_hc_IFOG = _mm512_castps128_ps512(_descale_hc_I);
            _descale_hc_IFOG = _mm512_insertf32x4(_descale_hc_IFOG, _descale_hc_F, 1);
            _descale_hc_IFOG = _mm512_insertf32x4(_descale_hc_IFOG, _descale_hc_O, 2);
            _descale_hc_IFOG = _mm512_insertf32x4(_descale_hc_IFOG, _descale_hc_G, 3);

            _descale_xc_IFOG = _mm512_div_ps(_mm512_set1_ps(1.f), _descale_xc_IFOG);
            _descale_hc_IFOG = _mm512_div_ps(_mm512_set1_ps(1.f), _descale_hc_IFOG);

            _mm512_storeu_ps(descales_ptr, _descale_xc_IFOG);
            _mm512_storeu_ps(descales_ptr + 16, _descale_hc_IFOG);
        }
#endif // __AVX512F__
        for (; q + 1 < hidden_size; q += 2)
        {
            bias_c_IFOG[0] = bias_c_I[q];
            bias_c_IFOG[1] = bias_c_F[q];
            bias_c_IFOG[2] = bias_c_O[q];
            bias_c_IFOG[3] = bias_c_G[q];
            bias_c_IFOG[4] = bias_c_I[q + 1];
            bias_c_IFOG[5] = bias_c_F[q + 1];
            bias_c_IFOG[6] = bias_c_O[q + 1];
            bias_c_IFOG[7] = bias_c_G[q + 1];

            bias_c_IFOG += 8;

            const signed char* weight_xc_I_0 = weight_xc_dr.row<const signed char>(hidden_size * 0 + q);
            const signed char* weight_xc_F_0 = weight_xc_dr.row<const signed char>(hidden_size * 1 + q);
            const signed char* weight_xc_O_0 = weight_xc_dr.row<const signed char>(hidden_size * 2 + q);
            const signed char* weight_xc_G_0 = weight_xc_dr.row<const signed char>(hidden_size * 3 + q);
            const signed char* weight_xc_I_1 = weight_xc_dr.row<const signed char>(hidden_size * 0 + q + 1);
            const signed char* weight_xc_F_1 = weight_xc_dr.row<const signed char>(hidden_size * 1 + q + 1);
            const signed char* weight_xc_O_1 = weight_xc_dr.row<const signed char>(hidden_size * 2 + q + 1);
            const signed char* weight_xc_G_1 = weight_xc_dr.row<const signed char>(hidden_size * 3 + q + 1);

            const signed char* weight_hc_I_0 = weight_hc_dr.row<const signed char>(hidden_size * 0 + q);
            const signed char* weight_hc_F_0 = weight_hc_dr.row<const signed char>(hidden_size * 1 + q);
            const signed char* weight_hc_O_0 = weight_hc_dr.row<const signed char>(hidden_size * 2 + q);
            const signed char* weight_hc_G_0 = weight_hc_dr.row<const signed char>(hidden_size * 3 + q);
            const signed char* weight_hc_I_1 = weight_hc_dr.row<const signed char>(hidden_size * 0 + q + 1);
            const signed char* weight_hc_F_1 = weight_hc_dr.row<const signed char>(hidden_size * 1 + q + 1);
            const signed char* weight_hc_O_1 = weight_hc_dr.row<const signed char>(hidden_size * 2 + q + 1);
            const signed char* weight_hc_G_1 = weight_hc_dr.row<const signed char>(hidden_size * 3 + q + 1);

#if __AVX512F__
            signed char* kptr = weight_data_tm_dr.row<signed char>(q / 4 + (q % 4) / 2);
            float* descales_ptr = weight_data_tm_int8_descales_dr.row(q / 4 + (q % 4) / 2);
#else
            signed char* kptr = weight_data_tm_dr.row<signed char>(q / 2);
            float* descales_ptr = weight_data_tm_int8_descales_dr.row(q / 2);
#endif

            int i = 0;
#if __AVXVNNI__ || __AVX512VNNI__
            __m256i _w_shift = _mm256_setzero_si256();
            __m256i _v127 = _mm256_set1_epi8(127);

            __m256i _w0_shift = _mm256_setzero_si256();
            __m256i _w1_shift = _mm256_setzero_si256();
#if defined(__x86_64__) || defined(_M_X64)
            __m256i _w2_shift = _mm256_setzero_si256();
            __m256i _w3_shift = _mm256_setzero_si256();
            for (; i + 15 < size; i += 16)
            {
                _mm_storeu_si128((__m128i*)kptr, _mm_loadu_si128((const __m128i*)(weight_xc_I_0 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16), _mm_loadu_si128((const __m128i*)(weight_xc_I_1 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 2), _mm_loadu_si128((const __m128i*)(weight_xc_F_0 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 3), _mm_loadu_si128((const __m128i*)(weight_xc_F_1 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 4), _mm_loadu_si128((const __m128i*)(weight_xc_O_0 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 5), _mm_loadu_si128((const __m128i*)(weight_xc_O_1 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 6), _mm_loadu_si128((const __m128i*)(weight_xc_G_0 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 7), _mm_loadu_si128((const __m128i*)(weight_xc_G_1 + i)));

                __m256i _w0 = _mm256_loadu_si256((const __m256i*)kptr);
                __m256i _w1 = _mm256_loadu_si256((const __m256i*)(kptr + 32));
                __m256i _w2 = _mm256_loadu_si256((const __m256i*)(kptr + 64));
                __m256i _w3 = _mm256_loadu_si256((const __m256i*)(kptr + 96));
                _w0_shift = _mm256_comp_dpbusd_epi32(_w0_shift, _v127, _w0);
                _w1_shift = _mm256_comp_dpbusd_epi32(_w1_shift, _v127, _w1);
                _w2_shift = _mm256_comp_dpbusd_epi32(_w2_shift, _v127, _w2);
                _w3_shift = _mm256_comp_dpbusd_epi32(_w3_shift, _v127, _w3);

                kptr += 128;
            }
            {
                __m256i _tmp0 = _mm256_hadd_epi32(_w0_shift, _w1_shift);
                __m256i _tmp1 = _mm256_hadd_epi32(_w2_shift, _w3_shift);
                _tmp0 = _mm256_hadd_epi32(_tmp0, _tmp1);
                _w_shift = _mm256_add_epi32(_w_shift, _tmp0);
            }

            _w0_shift = _mm256_setzero_si256();
            _w1_shift = _mm256_setzero_si256();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < size; i += 8)
            {
                _mm_storel_epi64((__m128i*)kptr, _mm_loadl_epi64((const __m128i*)(weight_xc_I_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8), _mm_loadl_epi64((const __m128i*)(weight_xc_F_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 16), _mm_loadl_epi64((const __m128i*)(weight_xc_I_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 24), _mm_loadl_epi64((const __m128i*)(weight_xc_F_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 32), _mm_loadl_epi64((const __m128i*)(weight_xc_O_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 40), _mm_loadl_epi64((const __m128i*)(weight_xc_G_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 48), _mm_loadl_epi64((const __m128i*)(weight_xc_O_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 56), _mm_loadl_epi64((const __m128i*)(weight_xc_G_1 + i)));

                __m256i _w0 = _mm256_loadu_si256((const __m256i*)kptr);
                __m256i _w1 = _mm256_loadu_si256((const __m256i*)(kptr + 32));
                _w0_shift = _mm256_comp_dpbusd_epi32(_w0_shift, _v127, _w0);
                _w1_shift = _mm256_comp_dpbusd_epi32(_w1_shift, _v127, _w1);

                kptr += 64;
            }
            {
                __m256i _tmp0 = _mm256_hadd_epi32(_w0_shift, _w1_shift);
                _w_shift = _mm256_add_epi32(_w_shift, _tmp0);
            }

            for (; i + 3 < size; i += 4)
            {
                kptr[0] = weight_xc_I_0[i];
                kptr[1] = weight_xc_I_0[i + 1];
                kptr[2] = weight_xc_I_0[i + 2];
                kptr[3] = weight_xc_I_0[i + 3];
                kptr[4] = weight_xc_F_0[i];
                kptr[5] = weight_xc_F_0[i + 1];
                kptr[6] = weight_xc_F_0[i + 2];
                kptr[7] = weight_xc_F_0[i + 3];
                kptr[8 + 0] = weight_xc_O_0[i];
                kptr[8 + 1] = weight_xc_O_0[i + 1];
                kptr[8 + 2] = weight_xc_O_0[i + 2];
                kptr[8 + 3] = weight_xc_O_0[i + 3];
                kptr[8 + 4] = weight_xc_G_0[i];
                kptr[8 + 5] = weight_xc_G_0[i + 1];
                kptr[8 + 6] = weight_xc_G_0[i + 2];
                kptr[8 + 7] = weight_xc_G_0[i + 3];
                kptr[16 + 0] = weight_xc_I_1[i];
                kptr[16 + 1] = weight_xc_I_1[i + 1];
                kptr[16 + 2] = weight_xc_I_1[i + 2];
                kptr[16 + 3] = weight_xc_I_1[i + 3];
                kptr[16 + 4] = weight_xc_F_1[i];
                kptr[16 + 5] = weight_xc_F_1[i + 1];
                kptr[16 + 6] = weight_xc_F_1[i + 2];
                kptr[16 + 7] = weight_xc_F_1[i + 3];
                kptr[24 + 0] = weight_xc_O_1[i];
                kptr[24 + 1] = weight_xc_O_1[i + 1];
                kptr[24 + 2] = weight_xc_O_1[i + 2];
                kptr[24 + 3] = weight_xc_O_1[i + 3];
                kptr[24 + 4] = weight_xc_G_1[i];
                kptr[24 + 5] = weight_xc_G_1[i + 1];
                kptr[24 + 6] = weight_xc_G_1[i + 2];
                kptr[24 + 7] = weight_xc_G_1[i + 3];

                __m256i _w = _mm256_loadu_si256((const __m256i*)kptr);
                _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _v127, _w);

                kptr += 32;
            }

            _mm256_storeu_si256((__m256i*)kptr, _w_shift);
            kptr += 32;
#else
#if defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < size; i += 8)
            {
                _mm_storel_epi64((__m128i*)kptr, _mm_loadl_epi64((const __m128i*)(weight_xc_I_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8), _mm_loadl_epi64((const __m128i*)(weight_xc_I_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 16), _mm_loadl_epi64((const __m128i*)(weight_xc_F_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 24), _mm_loadl_epi64((const __m128i*)(weight_xc_F_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 32), _mm_loadl_epi64((const __m128i*)(weight_xc_O_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 40), _mm_loadl_epi64((const __m128i*)(weight_xc_O_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 48), _mm_loadl_epi64((const __m128i*)(weight_xc_G_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 56), _mm_loadl_epi64((const __m128i*)(weight_xc_G_1 + i)));
                kptr += 64;
            }
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 3 < size; i += 4)
            {
                kptr[0] = weight_xc_I_0[i];
                kptr[1] = weight_xc_I_0[i + 1];
                kptr[2] = weight_xc_I_0[i + 2];
                kptr[3] = weight_xc_I_0[i + 3];
                kptr[4] = weight_xc_F_0[i];
                kptr[5] = weight_xc_F_0[i + 1];
                kptr[6] = weight_xc_F_0[i + 2];
                kptr[7] = weight_xc_F_0[i + 3];
                kptr[8 + 0] = weight_xc_I_1[i];
                kptr[8 + 1] = weight_xc_I_1[i + 1];
                kptr[8 + 2] = weight_xc_I_1[i + 2];
                kptr[8 + 3] = weight_xc_I_1[i + 3];
                kptr[8 + 4] = weight_xc_F_1[i];
                kptr[8 + 5] = weight_xc_F_1[i + 1];
                kptr[8 + 6] = weight_xc_F_1[i + 2];
                kptr[8 + 7] = weight_xc_F_1[i + 3];
                kptr[16 + 0] = weight_xc_O_0[i];
                kptr[16 + 1] = weight_xc_O_0[i + 1];
                kptr[16 + 2] = weight_xc_O_0[i + 2];
                kptr[16 + 3] = weight_xc_O_0[i + 3];
                kptr[16 + 4] = weight_xc_G_0[i];
                kptr[16 + 5] = weight_xc_G_0[i + 1];
                kptr[16 + 6] = weight_xc_G_0[i + 2];
                kptr[16 + 7] = weight_xc_G_0[i + 3];
                kptr[24 + 0] = weight_xc_O_1[i];
                kptr[24 + 1] = weight_xc_O_1[i + 1];
                kptr[24 + 2] = weight_xc_O_1[i + 2];
                kptr[24 + 3] = weight_xc_O_1[i + 3];
                kptr[24 + 4] = weight_xc_G_1[i];
                kptr[24 + 5] = weight_xc_G_1[i + 1];
                kptr[24 + 6] = weight_xc_G_1[i + 2];
                kptr[24 + 7] = weight_xc_G_1[i + 3];
                kptr += 32;
            }
#endif // __AVXVNNI__ || __AVX512VNNI__
            for (; i + 1 < size; i += 2)
            {
                kptr[0] = weight_xc_I_0[i];
                kptr[1] = weight_xc_I_0[i + 1];
                kptr[2] = weight_xc_F_0[i];
                kptr[3] = weight_xc_F_0[i + 1];
                kptr[4] = weight_xc_O_0[i];
                kptr[5] = weight_xc_O_0[i + 1];
                kptr[6] = weight_xc_G_0[i];
                kptr[7] = weight_xc_G_0[i + 1];
                kptr[8 + 0] = weight_xc_I_1[i];
                kptr[8 + 1] = weight_xc_I_1[i + 1];
                kptr[8 + 2] = weight_xc_F_1[i];
                kptr[8 + 3] = weight_xc_F_1[i + 1];
                kptr[8 + 4] = weight_xc_O_1[i];
                kptr[8 + 5] = weight_xc_O_1[i + 1];
                kptr[8 + 6] = weight_xc_G_1[i];
                kptr[8 + 7] = weight_xc_G_1[i + 1];
                kptr += 16;
            }
            for (; i < size; i++)
            {
                kptr[0] = weight_xc_I_0[i];
                kptr[1] = weight_xc_F_0[i];
                kptr[2] = weight_xc_O_0[i];
                kptr[3] = weight_xc_G_0[i];
                kptr[4] = weight_xc_I_1[i];
                kptr[5] = weight_xc_F_1[i];
                kptr[6] = weight_xc_O_1[i];
                kptr[7] = weight_xc_G_1[i];
                kptr += 8;
            }

            i = 0;
#if __AVXVNNI__ || __AVX512VNNI__
            _w_shift = _mm256_setzero_si256();
            _v127 = _mm256_set1_epi8(127);
            _w0_shift = _mm256_setzero_si256();
            _w1_shift = _mm256_setzero_si256();
#if defined(__x86_64__) || defined(_M_X64)
            _w2_shift = _mm256_setzero_si256();
            _w3_shift = _mm256_setzero_si256();
            for (; i + 15 < num_output; i += 16)
            {
                _mm_storeu_si128((__m128i*)kptr, _mm_loadu_si128((const __m128i*)(weight_hc_I_0 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16), _mm_loadu_si128((const __m128i*)(weight_hc_I_1 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 2), _mm_loadu_si128((const __m128i*)(weight_hc_F_0 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 3), _mm_loadu_si128((const __m128i*)(weight_hc_F_1 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 4), _mm_loadu_si128((const __m128i*)(weight_hc_O_0 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 5), _mm_loadu_si128((const __m128i*)(weight_hc_O_1 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 6), _mm_loadu_si128((const __m128i*)(weight_hc_G_0 + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16 * 7), _mm_loadu_si128((const __m128i*)(weight_hc_G_1 + i)));

                __m256i _w0 = _mm256_loadu_si256((const __m256i*)kptr);
                __m256i _w1 = _mm256_loadu_si256((const __m256i*)(kptr + 32));
                __m256i _w2 = _mm256_loadu_si256((const __m256i*)(kptr + 64));
                __m256i _w3 = _mm256_loadu_si256((const __m256i*)(kptr + 96));
                _w0_shift = _mm256_comp_dpbusd_epi32(_w0_shift, _v127, _w0);
                _w1_shift = _mm256_comp_dpbusd_epi32(_w1_shift, _v127, _w1);
                _w2_shift = _mm256_comp_dpbusd_epi32(_w2_shift, _v127, _w2);
                _w3_shift = _mm256_comp_dpbusd_epi32(_w3_shift, _v127, _w3);

                kptr += 128;
            }
            {
                __m256i _tmp0 = _mm256_hadd_epi32(_w0_shift, _w1_shift);
                __m256i _tmp1 = _mm256_hadd_epi32(_w2_shift, _w3_shift);
                _tmp0 = _mm256_hadd_epi32(_tmp0, _tmp1);
                _w_shift = _mm256_add_epi32(_w_shift, _tmp0);
            }

            _w0_shift = _mm256_setzero_si256();
            _w1_shift = _mm256_setzero_si256();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < num_output; i += 8)
            {
                _mm_storel_epi64((__m128i*)kptr, _mm_loadl_epi64((const __m128i*)(weight_hc_I_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8), _mm_loadl_epi64((const __m128i*)(weight_hc_F_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 16), _mm_loadl_epi64((const __m128i*)(weight_hc_I_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 24), _mm_loadl_epi64((const __m128i*)(weight_hc_F_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 32), _mm_loadl_epi64((const __m128i*)(weight_hc_O_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 40), _mm_loadl_epi64((const __m128i*)(weight_hc_G_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 48), _mm_loadl_epi64((const __m128i*)(weight_hc_O_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 56), _mm_loadl_epi64((const __m128i*)(weight_hc_G_1 + i)));

                __m256i _w0 = _mm256_loadu_si256((const __m256i*)kptr);
                __m256i _w1 = _mm256_loadu_si256((const __m256i*)(kptr + 32));
                _w0_shift = _mm256_comp_dpbusd_epi32(_w0_shift, _v127, _w0);
                _w1_shift = _mm256_comp_dpbusd_epi32(_w1_shift, _v127, _w1);

                kptr += 64;
            }
            {
                __m256i _tmp0 = _mm256_hadd_epi32(_w0_shift, _w1_shift);
                _w_shift = _mm256_add_epi32(_w_shift, _tmp0);
            }

            for (; i + 3 < num_output; i += 4)
            {
                kptr[0] = weight_hc_I_0[i];
                kptr[1] = weight_hc_I_0[i + 1];
                kptr[2] = weight_hc_I_0[i + 2];
                kptr[3] = weight_hc_I_0[i + 3];
                kptr[4] = weight_hc_F_0[i];
                kptr[5] = weight_hc_F_0[i + 1];
                kptr[6] = weight_hc_F_0[i + 2];
                kptr[7] = weight_hc_F_0[i + 3];
                kptr[8 + 0] = weight_hc_O_0[i];
                kptr[8 + 1] = weight_hc_O_0[i + 1];
                kptr[8 + 2] = weight_hc_O_0[i + 2];
                kptr[8 + 3] = weight_hc_O_0[i + 3];
                kptr[8 + 4] = weight_hc_G_0[i];
                kptr[8 + 5] = weight_hc_G_0[i + 1];
                kptr[8 + 6] = weight_hc_G_0[i + 2];
                kptr[8 + 7] = weight_hc_G_0[i + 3];
                kptr[16 + 0] = weight_hc_I_1[i];
                kptr[16 + 1] = weight_hc_I_1[i + 1];
                kptr[16 + 2] = weight_hc_I_1[i + 2];
                kptr[16 + 3] = weight_hc_I_1[i + 3];
                kptr[16 + 4] = weight_hc_F_1[i];
                kptr[16 + 5] = weight_hc_F_1[i + 1];
                kptr[16 + 6] = weight_hc_F_1[i + 2];
                kptr[16 + 7] = weight_hc_F_1[i + 3];
                kptr[24 + 0] = weight_hc_O_1[i];
                kptr[24 + 1] = weight_hc_O_1[i + 1];
                kptr[24 + 2] = weight_hc_O_1[i + 2];
                kptr[24 + 3] = weight_hc_O_1[i + 3];
                kptr[24 + 4] = weight_hc_G_1[i];
                kptr[24 + 5] = weight_hc_G_1[i + 1];
                kptr[24 + 6] = weight_hc_G_1[i + 2];
                kptr[24 + 7] = weight_hc_G_1[i + 3];

                __m256i _w = _mm256_loadu_si256((const __m256i*)kptr);
                _w_shift = _mm256_comp_dpbusd_epi32(_w_shift, _v127, _w);

                kptr += 32;
            }

            _mm256_storeu_si256((__m256i*)kptr, _w_shift);
            kptr += 32;
#else
#if defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < num_output; i += 8)
            {
                _mm_storel_epi64((__m128i*)kptr, _mm_loadl_epi64((const __m128i*)(weight_hc_I_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8), _mm_loadl_epi64((const __m128i*)(weight_hc_I_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 16), _mm_loadl_epi64((const __m128i*)(weight_hc_F_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 24), _mm_loadl_epi64((const __m128i*)(weight_hc_F_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 32), _mm_loadl_epi64((const __m128i*)(weight_hc_O_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 40), _mm_loadl_epi64((const __m128i*)(weight_hc_O_1 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 48), _mm_loadl_epi64((const __m128i*)(weight_hc_G_0 + i)));
                _mm_storel_epi64((__m128i*)(kptr + 56), _mm_loadl_epi64((const __m128i*)(weight_hc_G_1 + i)));
                kptr += 64;
            }
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 3 < num_output; i += 4)
            {
                kptr[0] = weight_hc_I_0[i];
                kptr[1] = weight_hc_I_0[i + 1];
                kptr[2] = weight_hc_I_0[i + 2];
                kptr[3] = weight_hc_I_0[i + 3];
                kptr[4] = weight_hc_F_0[i];
                kptr[5] = weight_hc_F_0[i + 1];
                kptr[6] = weight_hc_F_0[i + 2];
                kptr[7] = weight_hc_F_0[i + 3];
                kptr[8 + 0] = weight_hc_I_1[i];
                kptr[8 + 1] = weight_hc_I_1[i + 1];
                kptr[8 + 2] = weight_hc_I_1[i + 2];
                kptr[8 + 3] = weight_hc_I_1[i + 3];
                kptr[8 + 4] = weight_hc_F_1[i];
                kptr[8 + 5] = weight_hc_F_1[i + 1];
                kptr[8 + 6] = weight_hc_F_1[i + 2];
                kptr[8 + 7] = weight_hc_F_1[i + 3];
                kptr[16 + 0] = weight_hc_O_0[i];
                kptr[16 + 1] = weight_hc_O_0[i + 1];
                kptr[16 + 2] = weight_hc_O_0[i + 2];
                kptr[16 + 3] = weight_hc_O_0[i + 3];
                kptr[16 + 4] = weight_hc_G_0[i];
                kptr[16 + 5] = weight_hc_G_0[i + 1];
                kptr[16 + 6] = weight_hc_G_0[i + 2];
                kptr[16 + 7] = weight_hc_G_0[i + 3];
                kptr[24 + 0] = weight_hc_O_1[i];
                kptr[24 + 1] = weight_hc_O_1[i + 1];
                kptr[24 + 2] = weight_hc_O_1[i + 2];
                kptr[24 + 3] = weight_hc_O_1[i + 3];
                kptr[24 + 4] = weight_hc_G_1[i];
                kptr[24 + 5] = weight_hc_G_1[i + 1];
                kptr[24 + 6] = weight_hc_G_1[i + 2];
                kptr[24 + 7] = weight_hc_G_1[i + 3];
                kptr += 32;
            }
#endif // __AVXVNNI__ || __AVX512VNNI__
            for (; i + 1 < num_output; i += 2)
            {
                kptr[0] = weight_hc_I_0[i];
                kptr[1] = weight_hc_I_0[i + 1];
                kptr[2] = weight_hc_F_0[i];
                kptr[3] = weight_hc_F_0[i + 1];
                kptr[4] = weight_hc_O_0[i];
                kptr[5] = weight_hc_O_0[i + 1];
                kptr[6] = weight_hc_G_0[i];
                kptr[7] = weight_hc_G_0[i + 1];
                kptr[8 + 0] = weight_hc_I_1[i];
                kptr[8 + 1] = weight_hc_I_1[i + 1];
                kptr[8 + 2] = weight_hc_F_1[i];
                kptr[8 + 3] = weight_hc_F_1[i + 1];
                kptr[8 + 4] = weight_hc_O_1[i];
                kptr[8 + 5] = weight_hc_O_1[i + 1];
                kptr[8 + 6] = weight_hc_G_1[i];
                kptr[8 + 7] = weight_hc_G_1[i + 1];
                kptr += 16;
            }
            for (; i < num_output; i++)
            {
                kptr[0] = weight_hc_I_0[i];
                kptr[1] = weight_hc_F_0[i];
                kptr[2] = weight_hc_O_0[i];
                kptr[3] = weight_hc_G_0[i];
                kptr[4] = weight_hc_I_1[i];
                kptr[5] = weight_hc_F_1[i];
                kptr[6] = weight_hc_O_1[i];
                kptr[7] = weight_hc_G_1[i];
                kptr += 8;
            }

            descales_ptr[0] = 1.f / weight_xc_int8_scales_ptr[hidden_size * 0 + q];
            descales_ptr[1] = 1.f / weight_xc_int8_scales_ptr[hidden_size * 1 + q];
            descales_ptr[2] = 1.f / weight_xc_int8_scales_ptr[hidden_size * 2 + q];
            descales_ptr[3] = 1.f / weight_xc_int8_scales_ptr[hidden_size * 3 + q];
            descales_ptr[4] = 1.f / weight_xc_int8_scales_ptr[hidden_size * 0 + q + 1];
            descales_ptr[5] = 1.f / weight_xc_int8_scales_ptr[hidden_size * 1 + q + 1];
            descales_ptr[6] = 1.f / weight_xc_int8_scales_ptr[hidden_size * 2 + q + 1];
            descales_ptr[7] = 1.f / weight_xc_int8_scales_ptr[hidden_size * 3 + q + 1];
            descales_ptr[8 + 0] = 1.f / weight_hc_int8_scales_ptr[hidden_size * 0 + q];
            descales_ptr[8 + 1] = 1.f / weight_hc_int8_scales_ptr[hidden_size * 1 + q];
            descales_ptr[8 + 2] = 1.f / weight_hc_int8_scales_ptr[hidden_size * 2 + q];
            descales_ptr[8 + 3] = 1.f / weight_hc_int8_scales_ptr[hidden_size * 3 + q];
            descales_ptr[8 + 4] = 1.f / weight_hc_int8_scales_ptr[hidden_size * 0 + q + 1];
            descales_ptr[8 + 5] = 1.f / weight_hc_int8_scales_ptr[hidden_size * 1 + q + 1];
            descales_ptr[8 + 6] = 1.f / weight_hc_int8_scales_ptr[hidden_size * 2 + q + 1];
            descales_ptr[8 + 7] = 1.f / weight_hc_int8_scales_ptr[hidden_size * 3 + q + 1];
        }
#endif // __AVX2__
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

#if __AVX512F__
            signed char* kptr = weight_data_tm_dr.row<signed char>(q / 4 + (q % 4) / 2 + q % 2);
            float* descales_ptr = weight_data_tm_int8_descales_dr.row(q / 4 + (q % 4) / 2 + q % 2);
#elif __AVX2__
            signed char* kptr = weight_data_tm_dr.row<signed char>(q / 2 + q % 2);
            float* descales_ptr = weight_data_tm_int8_descales_dr.row(q / 2 + q % 2);
#else
            signed char* kptr = weight_data_tm_dr.row<signed char>(q);
            float* descales_ptr = weight_data_tm_int8_descales_dr.row(q);
#endif

            int i = 0;
#if __SSE2__
#if __AVXVNNI__ || __AVX512VNNI__
            __m128i _w_shift = _mm_setzero_si128();
            __m128i _v127 = _mm_set1_epi8(127);
            __m128i _w0_shift = _mm_setzero_si128();
            __m128i _w1_shift = _mm_setzero_si128();
#if defined(__x86_64__) || defined(_M_X64)
            __m128i _w2_shift = _mm_setzero_si128();
            __m128i _w3_shift = _mm_setzero_si128();
            for (; i + 15 < size; i += 16)
            {
                _mm_storeu_si128((__m128i*)kptr, _mm_loadu_si128((const __m128i*)(weight_xc_I + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16), _mm_loadu_si128((const __m128i*)(weight_xc_F + i)));
                _mm_storeu_si128((__m128i*)(kptr + 32), _mm_loadu_si128((const __m128i*)(weight_xc_O + i)));
                _mm_storeu_si128((__m128i*)(kptr + 48), _mm_loadu_si128((const __m128i*)(weight_xc_G + i)));

                __m128i _w0 = _mm_loadu_si128((const __m128i*)kptr);
                __m128i _w1 = _mm_loadu_si128((const __m128i*)(kptr + 16));
                __m128i _w2 = _mm_loadu_si128((const __m128i*)(kptr + 32));
                __m128i _w3 = _mm_loadu_si128((const __m128i*)(kptr + 48));
                _w0_shift = _mm_comp_dpbusd_epi32(_w0_shift, _v127, _w0);
                _w1_shift = _mm_comp_dpbusd_epi32(_w1_shift, _v127, _w1);
                _w2_shift = _mm_comp_dpbusd_epi32(_w2_shift, _v127, _w2);
                _w3_shift = _mm_comp_dpbusd_epi32(_w3_shift, _v127, _w3);

                kptr += 64;
            }
            {
                transpose4x4_epi32(_w0_shift, _w1_shift, _w2_shift, _w3_shift);
                _w_shift = _mm_add_epi32(_w_shift, _w0_shift);
                _w_shift = _mm_add_epi32(_w_shift, _w1_shift);
                _w_shift = _mm_add_epi32(_w_shift, _w2_shift);
                _w_shift = _mm_add_epi32(_w_shift, _w3_shift);
            }

            _w0_shift = _mm_setzero_si128();
            _w1_shift = _mm_setzero_si128();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < size; i += 8)
            {
                _mm_storel_epi64((__m128i*)kptr, _mm_loadl_epi64((const __m128i*)(weight_xc_I + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8), _mm_loadl_epi64((const __m128i*)(weight_xc_F + i)));
                _mm_storel_epi64((__m128i*)(kptr + 16), _mm_loadl_epi64((const __m128i*)(weight_xc_O + i)));
                _mm_storel_epi64((__m128i*)(kptr + 24), _mm_loadl_epi64((const __m128i*)(weight_xc_G + i)));

                __m128i _w0 = _mm_loadu_si128((const __m128i*)kptr);
                __m128i _w1 = _mm_loadu_si128((const __m128i*)(kptr + 16));
                _w0_shift = _mm_comp_dpbusd_epi32(_w0_shift, _v127, _w0);
                _w1_shift = _mm_comp_dpbusd_epi32(_w1_shift, _v127, _w1);

                kptr += 32;
            }
            {
                __m128i _tmp0 = _mm_hadd_epi32(_w0_shift, _w1_shift);
                _w_shift = _mm_add_epi32(_w_shift, _tmp0);
            }

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

                __m128i _w = _mm_loadu_si128((const __m128i*)kptr);
                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _w);

                kptr += 16;
            }

            _mm_storeu_si128((__m128i*)kptr, _w_shift);
            kptr += 16;
#else
#if defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < size; i += 8)
            {
                _mm_storel_epi64((__m128i*)kptr, _mm_loadl_epi64((const __m128i*)(weight_xc_I + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8), _mm_loadl_epi64((const __m128i*)(weight_xc_F + i)));
                _mm_storel_epi64((__m128i*)(kptr + 16), _mm_loadl_epi64((const __m128i*)(weight_xc_O + i)));
                _mm_storel_epi64((__m128i*)(kptr + 24), _mm_loadl_epi64((const __m128i*)(weight_xc_G + i)));
                kptr += 32;
            }
#endif // defined(__x86_64__) || defined(_M_X64)
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
#endif // __AVXVNNI__ || __AVX512VNNI__
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
#if __AVXVNNI__ || __AVX512VNNI__
            _w_shift = _mm_setzero_si128();
            _w0_shift = _mm_setzero_si128();
            _w1_shift = _mm_setzero_si128();
#if defined(__x86_64__) || defined(_M_X64)
            _w2_shift = _mm_setzero_si128();
            _w3_shift = _mm_setzero_si128();
            for (; i + 15 < num_output; i += 16)
            {
                _mm_storeu_si128((__m128i*)kptr, _mm_loadu_si128((const __m128i*)(weight_hc_I + i)));
                _mm_storeu_si128((__m128i*)(kptr + 16), _mm_loadu_si128((const __m128i*)(weight_hc_F + i)));
                _mm_storeu_si128((__m128i*)(kptr + 32), _mm_loadu_si128((const __m128i*)(weight_hc_O + i)));
                _mm_storeu_si128((__m128i*)(kptr + 48), _mm_loadu_si128((const __m128i*)(weight_hc_G + i)));

                __m128i _w0 = _mm_loadu_si128((const __m128i*)kptr);
                __m128i _w1 = _mm_loadu_si128((const __m128i*)(kptr + 16));
                __m128i _w2 = _mm_loadu_si128((const __m128i*)(kptr + 32));
                __m128i _w3 = _mm_loadu_si128((const __m128i*)(kptr + 48));
                _w0_shift = _mm_comp_dpbusd_epi32(_w0_shift, _v127, _w0);
                _w1_shift = _mm_comp_dpbusd_epi32(_w1_shift, _v127, _w1);
                _w2_shift = _mm_comp_dpbusd_epi32(_w2_shift, _v127, _w2);
                _w3_shift = _mm_comp_dpbusd_epi32(_w3_shift, _v127, _w3);

                kptr += 64;
            }
            {
                transpose4x4_epi32(_w0_shift, _w1_shift, _w2_shift, _w3_shift);
                _w_shift = _mm_add_epi32(_w_shift, _w0_shift);
                _w_shift = _mm_add_epi32(_w_shift, _w1_shift);
                _w_shift = _mm_add_epi32(_w_shift, _w2_shift);
                _w_shift = _mm_add_epi32(_w_shift, _w3_shift);
            }

            _w0_shift = _mm_setzero_si128();
            _w1_shift = _mm_setzero_si128();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < num_output; i += 8)
            {
                _mm_storel_epi64((__m128i*)kptr, _mm_loadl_epi64((const __m128i*)(weight_hc_I + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8), _mm_loadl_epi64((const __m128i*)(weight_hc_F + i)));
                _mm_storel_epi64((__m128i*)(kptr + 16), _mm_loadl_epi64((const __m128i*)(weight_hc_O + i)));
                _mm_storel_epi64((__m128i*)(kptr + 24), _mm_loadl_epi64((const __m128i*)(weight_hc_G + i)));

                __m128i _w0 = _mm_loadu_si128((const __m128i*)kptr);
                __m128i _w1 = _mm_loadu_si128((const __m128i*)(kptr + 16));
                _w0_shift = _mm_comp_dpbusd_epi32(_w0_shift, _v127, _w0);
                _w1_shift = _mm_comp_dpbusd_epi32(_w1_shift, _v127, _w1);

                kptr += 32;
            }
            {
                __m128i _tmp0 = _mm_hadd_epi32(_w0_shift, _w1_shift);
                _w_shift = _mm_add_epi32(_w_shift, _tmp0);
            }

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

                __m128i _w = _mm_loadu_si128((const __m128i*)kptr);
                _w_shift = _mm_comp_dpbusd_epi32(_w_shift, _v127, _w);

                kptr += 16;
            }

            _mm_storeu_si128((__m128i*)kptr, _w_shift);
            kptr += 16;
#else
#if defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < num_output; i += 8)
            {
                _mm_storel_epi64((__m128i*)kptr, _mm_loadl_epi64((const __m128i*)(weight_hc_I + i)));
                _mm_storel_epi64((__m128i*)(kptr + 8), _mm_loadl_epi64((const __m128i*)(weight_hc_F + i)));
                _mm_storel_epi64((__m128i*)(kptr + 16), _mm_loadl_epi64((const __m128i*)(weight_hc_O + i)));
                _mm_storel_epi64((__m128i*)(kptr + 24), _mm_loadl_epi64((const __m128i*)(weight_hc_G + i)));
                kptr += 32;
            }
#endif // defined(__x86_64__) || defined(_M_X64)
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
#endif // __AVXVNNI__ || __AVX512VNNI__
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

static float lstm_dynamic_quantize_get_absmax(const float* ptr, int size)
{
    float absmax = 0.f;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _absmax_avx512 = _mm512_set1_ps(0.f);
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        _absmax_avx512 = _mm512_max_ps(_absmax_avx512, abs512_ps(_p));
        ptr += 16;
    }
    absmax = std::max(absmax, _mm512_comp_reduce_max_ps(_absmax_avx512));
#endif // __AVX512F__
    __m256 _absmax_avx = _mm256_set1_ps(0.f);
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        _absmax_avx = _mm256_max_ps(_absmax_avx, abs256_ps(_p));
        ptr += 8;
    }
    absmax = std::max(absmax, _mm256_reduce_max_ps(_absmax_avx));
#endif // __AVX__
    __m128 _absmax = _mm_set1_ps(0.f);
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        _absmax = _mm_max_ps(_absmax, abs_ps(_p));
        ptr += 4;
    }
    absmax = std::max(absmax, _mm_reduce_max_ps(_absmax));
#endif // __SSE2__
    for (; i < size; i++)
    {
        absmax = std::max(absmax, (float)fabs(*ptr));
        ptr++;
    }

    return absmax;
}

static void lstm_dynamic_quantize_scale2int8(const float* ptr, int size, float scale, signed char* outptr)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        lstm_dynamic_quantize_scale2int8_avx512vnni(ptr, size, scale, outptr);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVX512F__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        lstm_dynamic_quantize_scale2int8_avxvnni(ptr, size, scale, outptr);
        return;
    }
#endif

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _scale_avx512 = _mm512_set1_ps(scale);
#if __AVX512VNNI__
    __m128i _v127 = _mm_set1_epi8(127);
#endif
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        _p = _mm512_mul_ps(_p, _scale_avx512);
        __m128i _outp = float2int8_avx512(_p);
#if __AVX512VNNI__
        _outp = _mm_add_epi8(_outp, _v127);
#endif
        _mm_storeu_si128((__m128i*)outptr, _outp);
        ptr += 16;
        outptr += 16;
    }
#endif // __AVX512F__
    __m256 _scale_avx = _mm256_set1_ps(scale);
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        _p = _mm256_mul_ps(_p, _scale_avx);
        *(int64_t*)outptr = float2int8_avx(_p);
#if __AVXVNNI__ || __AVX512VNNI__
        outptr[0] += 127;
        outptr[1] += 127;
        outptr[2] += 127;
        outptr[3] += 127;
        outptr[4] += 127;
        outptr[5] += 127;
        outptr[6] += 127;
        outptr[7] += 127;
#endif
        ptr += 8;
        outptr += 8;
    }
#endif // __AVX__
    __m128 _scale = _mm_set1_ps(scale);
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        _p = _mm_mul_ps(_p, _scale);
        *(int32_t*)outptr = float2int8_sse(_p);
#ifndef _MSC_VER
        // vs2019 crash on 128bit vnni :L   --- nihui
#if __AVXVNNI__ || __AVX512VNNI__
        outptr[0] += 127;
        outptr[1] += 127;
        outptr[2] += 127;
        outptr[3] += 127;
#endif
#endif
        ptr += 4;
        outptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *outptr++ = float2int8(*ptr++ * scale);
    }
}

static void lstm_int8(const Mat& bottom_blob_int8, const Mat& bottom_blob_int8_descales, Mat& top_blob, int reverse, const Mat& weight_data_tm, const Mat& weight_data_tm_int8_descales, const Mat& bias_c, const Mat& weight_hr, Mat& hidden_state, Mat& cell_state, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512VNNI && __AVX512F__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx512_vnni())
    {
        lstm_int8_avx512vnni(bottom_blob_int8, bottom_blob_int8_descales, top_blob, reverse, weight_data_tm, weight_data_tm_int8_descales, bias_c, weight_hr, hidden_state, cell_state, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXVNNI && __AVX__ && !__AVX512F__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx_vnni())
    {
        lstm_int8_avxvnni(bottom_blob_int8, bottom_blob_int8_descales, top_blob, reverse, weight_data_tm, weight_data_tm_int8_descales, bias_c, weight_hr, hidden_state, cell_state, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_avx2())
    {
        lstm_int8_avx2(bottom_blob_int8, bottom_blob_int8_descales, top_blob, reverse, weight_data_tm, weight_data_tm_int8_descales, bias_c, weight_hr, hidden_state, cell_state, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_XOP && __SSE2__ && !__XOP__ && !__AVX2__ && !__AVXVNNI__ && !__AVX512VNNI__
    if (ncnn::cpu_support_x86_xop())
    {
        lstm_int8_xop(bottom_blob_int8, bottom_blob_int8_descales, top_blob, reverse, weight_data_tm, weight_data_tm_int8_descales, bias_c, weight_hr, hidden_state, cell_state, opt);
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
    float hidden_state_int8_descale = 1.f;

    // unroll
    for (int t = 0; t < T; t++)
    {
        int ti = reverse ? T - 1 - t : t;

        // dynamic quantize hidden_state
        {
            const float* ptr = hidden_state;

            const float absmax = lstm_dynamic_quantize_get_absmax(ptr, num_output);

            if (absmax == 0.f)
            {
#if __AVXVNNI__ || __AVX512VNNI__
                signed char* hs = hidden_state_int8;

                int i = 0;
                __m128i _v127 = _mm_set1_epi8(127);
                for (; i + 15 < num_output; i += 16)
                {
                    _mm_storeu_si128((__m128i*)hs, _v127);
                    hs += 16;
                }
                for (; i + 7 < num_output; i += 8)
                {
                    hs[0] = 127;
                    hs[1] = 127;
                    hs[2] = 127;
                    hs[3] = 127;
                    hs[4] = 127;
                    hs[5] = 127;
                    hs[6] = 127;
                    hs[7] = 127;
                    hs += 8;
                }
                for (; i + 3 < num_output; i += 4)
                {
#ifdef _MSC_VER
                    hs[0] = 0;
                    hs[1] = 0;
                    hs[2] = 0;
                    hs[3] = 0;
#else
                    hs[0] = 127;
                    hs[1] = 127;
                    hs[2] = 127;
                    hs[3] = 127;
#endif
                    hs += 4;
                }
                for (; i < num_output; i++)
                {
                    hs[0] = 0;
                    hs += 1;
                }
#else
                hidden_state_int8.fill<signed char>(0);
#endif
            }
            else
            {
                hidden_state_int8_descale = absmax / 127.f;

                signed char* hs = hidden_state_int8;

                const float scale = 127.f / absmax;
                lstm_dynamic_quantize_scale2int8(ptr, num_output, scale, hs);
            }
        }

        int remain_hidden_size_start = 0;
        int nn_hidden_size = 0;
#if __AVX2__
#if __AVX512F__
        nn_hidden_size = hidden_size >> 2;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_hidden_size; qq++)
        {
            int q = qq * 4;

            const signed char* x = bottom_blob_int8.row<const signed char>(ti);
            const signed char* hs = hidden_state_int8;
            const float descale_x = bottom_blob_int8_descales[ti];
            const float descale_h = hidden_state_int8_descale;

            const float* bias_c_IFOG = (const float*)bias_c + q * 4;

            const signed char* kptr = weight_data_tm.row<const signed char>(q / 4);
            const float* descales_ptr = weight_data_tm_int8_descales.row(q / 4);

            float* gates_data = gates.row(q);

            __m512i _lstm_IFOGx0 = _mm512_setzero_si512();
            __m512i _sum0 = _mm512_setzero_si512();
            __m512i _sum1 = _mm512_setzero_si512();
            int i = 0;
#if __AVX512VNNI__
#if defined(__x86_64__) || defined(_M_X64)
            __m512i _sum2 = _mm512_setzero_si512();
            __m512i _sum3 = _mm512_setzero_si512();
            for (; i + 15 < size; i += 16)
            {
                __m128i _xi = _mm_loadu_si128((const __m128i*)(x + i));
                __m512i _w0 = _mm512_loadu_si512((const __m512i*)kptr);
                __m512i _w1 = _mm512_loadu_si512((const __m512i*)(kptr + 64));
                __m512i _w2 = _mm512_loadu_si512((const __m512i*)(kptr + 128));
                __m512i _w3 = _mm512_loadu_si512((const __m512i*)(kptr + 192));

                __m512i _xii = _mm512_broadcast_i32x4(_xi);

                _sum0 = _mm512_dpbusd_epi32(_sum0, _xii, _w0);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _xii, _w1);
                _sum2 = _mm512_dpbusd_epi32(_sum2, _xii, _w2);
                _sum3 = _mm512_dpbusd_epi32(_sum3, _xii, _w3);

                kptr += 256;
            }
            {
                __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum1);
                __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum1);
                __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum3);
                __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum3);
                _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                _sum2 = _mm512_unpacklo_epi64(_tmp1, _tmp3);
                _sum3 = _mm512_unpackhi_epi64(_tmp1, _tmp3);

                _lstm_IFOGx0 = _mm512_add_epi32(_lstm_IFOGx0, _sum0);
                _lstm_IFOGx0 = _mm512_add_epi32(_lstm_IFOGx0, _sum1);
                _lstm_IFOGx0 = _mm512_add_epi32(_lstm_IFOGx0, _sum2);
                _lstm_IFOGx0 = _mm512_add_epi32(_lstm_IFOGx0, _sum3);
            }

            _sum0 = _mm512_setzero_si512();
            _sum1 = _mm512_setzero_si512();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < size; i += 8)
            {
                __m128i _xi = _mm_loadl_epi64((const __m128i*)(x + i));
                __m512i _w0 = _mm512_loadu_si512((const __m512i*)kptr);
                __m512i _w1 = _mm512_loadu_si512((const __m512i*)(kptr + 64));

                __m512i _xii = _mm512_broadcastq_epi64(_xi);

                _sum0 = _mm512_dpbusd_epi32(_sum0, _xii, _w0);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _xii, _w1);

                kptr += 128;
            }
            {
                __m512i _tmp0 = _mm512_castps_si512(_mm512_shuffle_ps(_mm512_castsi512_ps(_sum0), _mm512_castsi512_ps(_sum1), _MM_SHUFFLE(2, 0, 2, 0)));
                __m512i _tmp1 = _mm512_castps_si512(_mm512_shuffle_ps(_mm512_castsi512_ps(_sum0), _mm512_castsi512_ps(_sum1), _MM_SHUFFLE(3, 1, 3, 1)));

                _lstm_IFOGx0 = _mm512_add_epi32(_lstm_IFOGx0, _tmp0);
                _lstm_IFOGx0 = _mm512_add_epi32(_lstm_IFOGx0, _tmp1);
            }

            for (; i + 3 < size; i += 4)
            {
                __m512i _xi = _mm512_set1_epi32(((const int*)(x + i))[0]);
                __m512i _w = _mm512_loadu_si512((const __m512i*)kptr);

#ifdef _MSC_VER
                _xi = _mm512_add_epi32(_xi, _mm512_set1_epi8(127));
#endif
                _lstm_IFOGx0 = _mm512_dpbusd_epi32(_lstm_IFOGx0, _xi, _w);

                kptr += 64;
            }
            {
                __m512i _w_shift = _mm512_loadu_si512((const __m512i*)kptr);
                _lstm_IFOGx0 = _mm512_sub_epi32(_lstm_IFOGx0, _w_shift);
                kptr += 64;
            }
#else
#if defined(__x86_64__) || defined(_M_X64)
            __m512i _sum2 = _mm512_setzero_si512();
            __m512i _sum3 = _mm512_setzero_si512();
            for (; i + 7 < size; i += 8)
            {
                __m256i _xi = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)(x + i)));
                __m256i _w0 = _mm256_loadu_si256((const __m256i*)kptr);
                __m256i _w1 = _mm256_loadu_si256((const __m256i*)(kptr + 32));
                __m256i _w2 = _mm256_loadu_si256((const __m256i*)(kptr + 64));
                __m256i _w3 = _mm256_loadu_si256((const __m256i*)(kptr + 96));

                __m512i _xii = _mm512_cvtepi8_epi16(_xi);
                __m512i _ww0 = _mm512_cvtepi8_epi16(_w0);
                __m512i _ww1 = _mm512_cvtepi8_epi16(_w1);
                __m512i _ww2 = _mm512_cvtepi8_epi16(_w2);
                __m512i _ww3 = _mm512_cvtepi8_epi16(_w3);

                __m512i _s0 = _mm512_madd_epi16(_ww0, _xii);
                __m512i _s1 = _mm512_madd_epi16(_ww1, _xii);
                __m512i _s2 = _mm512_madd_epi16(_ww2, _xii);
                __m512i _s3 = _mm512_madd_epi16(_ww3, _xii);
                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);
                _sum2 = _mm512_add_epi32(_sum2, _s2);
                _sum3 = _mm512_add_epi32(_sum3, _s3);

                kptr += 128;
            }
            {
                __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum1);
                __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum1);
                __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum3);
                __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum3);
                _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                _sum2 = _mm512_unpacklo_epi64(_tmp1, _tmp3);
                _sum3 = _mm512_unpackhi_epi64(_tmp1, _tmp3);

                _lstm_IFOGx0 = _mm512_add_epi32(_lstm_IFOGx0, _sum0);
                _lstm_IFOGx0 = _mm512_add_epi32(_lstm_IFOGx0, _sum1);
                _lstm_IFOGx0 = _mm512_add_epi32(_lstm_IFOGx0, _sum2);
                _lstm_IFOGx0 = _mm512_add_epi32(_lstm_IFOGx0, _sum3);
            }

            _sum0 = _mm512_setzero_si512();
            _sum1 = _mm512_setzero_si512();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 3 < size; i += 4)
            {
                __m256i _xi = _mm256_castps_si256(_mm256_broadcast_ss((const float*)(x + i)));
                __m256i _w0 = _mm256_loadu_si256((const __m256i*)kptr);
                __m256i _w1 = _mm256_loadu_si256((const __m256i*)(kptr + 32));

                __m512i _xii = _mm512_cvtepi8_epi16(_xi);
                __m512i _ww0 = _mm512_cvtepi8_epi16(_w0);
                __m512i _ww1 = _mm512_cvtepi8_epi16(_w1);

                __m512i _s0 = _mm512_madd_epi16(_ww0, _xii);
                __m512i _s1 = _mm512_madd_epi16(_ww1, _xii);
                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);

                kptr += 64;
            }
            {
                __m512i _tmp0 = _mm512_castps_si512(_mm512_shuffle_ps(_mm512_castsi512_ps(_sum0), _mm512_castsi512_ps(_sum1), _MM_SHUFFLE(2, 0, 2, 0)));
                __m512i _tmp1 = _mm512_castps_si512(_mm512_shuffle_ps(_mm512_castsi512_ps(_sum0), _mm512_castsi512_ps(_sum1), _MM_SHUFFLE(3, 1, 3, 1)));

                _lstm_IFOGx0 = _mm512_add_epi32(_lstm_IFOGx0, _tmp0);
                _lstm_IFOGx0 = _mm512_add_epi32(_lstm_IFOGx0, _tmp1);
            }
#endif // __AVX512VNNI__
            for (; i + 1 < size; i += 2)
            {
                __m256i _w = _mm256_loadu_si256((const __m256i*)kptr);
                __m256i _xi = _mm256_castps_si256(_mm256_broadcast_ss((const float*)(x + i)));

                __m512i _ww = _mm512_cvtepi8_epi16(_w);
                __m512i _xixi = _mm512_cvtepi8_epi16(_xi);

                __m512i _xixi0 = _mm512_shuffle_epi32(_xixi, _MM_PERM_AAAA);

                _lstm_IFOGx0 = _mm512_comp_dpwssd_epi32(_lstm_IFOGx0, _ww, _xixi0);

                kptr += 32;
            }
            for (; i < size; i++)
            {
                __m128i _w = _mm_load_si128((const __m128i*)kptr);
                __m256i _xi = _mm256_set1_epi16(x[i]);

                __m256i _ww = _mm256_cvtepi8_epi16(_w);

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_ww, _xi));

                _lstm_IFOGx0 = _mm512_add_epi32(_lstm_IFOGx0, _s0);

                kptr += 16;
            }

            __m512i _lstm_IFOGh0 = _mm512_setzero_si512();
            _sum0 = _mm512_setzero_si512();
            _sum1 = _mm512_setzero_si512();
            i = 0;
#if __AVX512VNNI__
#if defined(__x86_64__) || defined(_M_X64)
            _sum2 = _mm512_setzero_si512();
            _sum3 = _mm512_setzero_si512();
            for (; i + 15 < num_output; i += 16)
            {
                __m128i _h_cont = _mm_loadu_si128((const __m128i*)(hs + i));
                __m512i _w0 = _mm512_loadu_si512((const __m512i*)kptr);
                __m512i _w1 = _mm512_loadu_si512((const __m512i*)(kptr + 64));
                __m512i _w2 = _mm512_loadu_si512((const __m512i*)(kptr + 128));
                __m512i _w3 = _mm512_loadu_si512((const __m512i*)(kptr + 192));

                __m512i _hh_cont = _mm512_broadcast_i32x4(_h_cont);

                _sum0 = _mm512_dpbusd_epi32(_sum0, _hh_cont, _w0);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _hh_cont, _w1);
                _sum2 = _mm512_dpbusd_epi32(_sum2, _hh_cont, _w2);
                _sum3 = _mm512_dpbusd_epi32(_sum3, _hh_cont, _w3);

                kptr += 256;
            }
            {
                __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum1);
                __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum1);
                __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum3);
                __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum3);
                _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                _sum2 = _mm512_unpacklo_epi64(_tmp1, _tmp3);
                _sum3 = _mm512_unpackhi_epi64(_tmp1, _tmp3);

                _lstm_IFOGh0 = _mm512_add_epi32(_lstm_IFOGh0, _sum0);
                _lstm_IFOGh0 = _mm512_add_epi32(_lstm_IFOGh0, _sum1);
                _lstm_IFOGh0 = _mm512_add_epi32(_lstm_IFOGh0, _sum2);
                _lstm_IFOGh0 = _mm512_add_epi32(_lstm_IFOGh0, _sum3);
            }

            _sum0 = _mm512_setzero_si512();
            _sum1 = _mm512_setzero_si512();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < num_output; i += 8)
            {
                __m128i _h_cont = _mm_loadl_epi64((const __m128i*)(hs + i));
                __m512i _w0 = _mm512_loadu_si512((const __m512i*)kptr);
                __m512i _w1 = _mm512_loadu_si512((const __m512i*)(kptr + 64));

                __m512i _hh_cont = _mm512_broadcastq_epi64(_h_cont);

                _sum0 = _mm512_dpbusd_epi32(_sum0, _hh_cont, _w0);
                _sum1 = _mm512_dpbusd_epi32(_sum1, _hh_cont, _w1);

                kptr += 128;
            }
            {
                __m512i _tmp0 = _mm512_castps_si512(_mm512_shuffle_ps(_mm512_castsi512_ps(_sum0), _mm512_castsi512_ps(_sum1), _MM_SHUFFLE(2, 0, 2, 0)));
                __m512i _tmp1 = _mm512_castps_si512(_mm512_shuffle_ps(_mm512_castsi512_ps(_sum0), _mm512_castsi512_ps(_sum1), _MM_SHUFFLE(3, 1, 3, 1)));

                _lstm_IFOGh0 = _mm512_add_epi32(_lstm_IFOGh0, _tmp0);
                _lstm_IFOGh0 = _mm512_add_epi32(_lstm_IFOGh0, _tmp1);
            }

            for (; i + 3 < num_output; i += 4)
            {
                __m512i _h_cont = _mm512_set1_epi32(((const int*)(hs + i))[0]);
                __m512i _w = _mm512_loadu_si512((const __m512i*)kptr);

#ifdef _MSC_VER
                _h_cont = _mm512_add_epi32(_h_cont, _mm512_set1_epi8(127));
#endif
                _lstm_IFOGh0 = _mm512_dpbusd_epi32(_lstm_IFOGh0, _h_cont, _w);

                kptr += 64;
            }
            {
                __m512i _w_shift = _mm512_loadu_si512((const __m512i*)kptr);
                _lstm_IFOGh0 = _mm512_sub_epi32(_lstm_IFOGh0, _w_shift);
                kptr += 64;
            }
#else
#if defined(__x86_64__) || defined(_M_X64)
            _sum2 = _mm512_setzero_si512();
            _sum3 = _mm512_setzero_si512();
            for (; i + 7 < num_output; i += 8)
            {
                __m256i _h_cont = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)(hs + i)));
                __m256i _w0 = _mm256_loadu_si256((const __m256i*)kptr);
                __m256i _w1 = _mm256_loadu_si256((const __m256i*)(kptr + 32));
                __m256i _w2 = _mm256_loadu_si256((const __m256i*)(kptr + 64));
                __m256i _w3 = _mm256_loadu_si256((const __m256i*)(kptr + 96));

                __m512i _hh_cont = _mm512_cvtepi8_epi16(_h_cont);
                __m512i _ww0 = _mm512_cvtepi8_epi16(_w0);
                __m512i _ww1 = _mm512_cvtepi8_epi16(_w1);
                __m512i _ww2 = _mm512_cvtepi8_epi16(_w2);
                __m512i _ww3 = _mm512_cvtepi8_epi16(_w3);

                __m512i _s0 = _mm512_madd_epi16(_ww0, _hh_cont);
                __m512i _s1 = _mm512_madd_epi16(_ww1, _hh_cont);
                __m512i _s2 = _mm512_madd_epi16(_ww2, _hh_cont);
                __m512i _s3 = _mm512_madd_epi16(_ww3, _hh_cont);
                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);
                _sum2 = _mm512_add_epi32(_sum2, _s2);
                _sum3 = _mm512_add_epi32(_sum3, _s3);

                kptr += 128;
            }
            {
                __m512i _tmp0 = _mm512_unpacklo_epi32(_sum0, _sum1);
                __m512i _tmp1 = _mm512_unpackhi_epi32(_sum0, _sum1);
                __m512i _tmp2 = _mm512_unpacklo_epi32(_sum2, _sum3);
                __m512i _tmp3 = _mm512_unpackhi_epi32(_sum2, _sum3);
                _sum0 = _mm512_unpacklo_epi64(_tmp0, _tmp2);
                _sum1 = _mm512_unpackhi_epi64(_tmp0, _tmp2);
                _sum2 = _mm512_unpacklo_epi64(_tmp1, _tmp3);
                _sum3 = _mm512_unpackhi_epi64(_tmp1, _tmp3);

                _lstm_IFOGh0 = _mm512_add_epi32(_lstm_IFOGh0, _sum0);
                _lstm_IFOGh0 = _mm512_add_epi32(_lstm_IFOGh0, _sum1);
                _lstm_IFOGh0 = _mm512_add_epi32(_lstm_IFOGh0, _sum2);
                _lstm_IFOGh0 = _mm512_add_epi32(_lstm_IFOGh0, _sum3);
            }

            _sum0 = _mm512_setzero_si512();
            _sum1 = _mm512_setzero_si512();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 3 < num_output; i += 4)
            {
                __m256i _h_cont = _mm256_castps_si256(_mm256_broadcast_ss((const float*)(hs + i)));
                __m256i _w0 = _mm256_loadu_si256((const __m256i*)kptr);
                __m256i _w1 = _mm256_loadu_si256((const __m256i*)(kptr + 32));

                __m512i _hh_cont = _mm512_cvtepi8_epi16(_h_cont);
                __m512i _ww0 = _mm512_cvtepi8_epi16(_w0);
                __m512i _ww1 = _mm512_cvtepi8_epi16(_w1);

                __m512i _s0 = _mm512_madd_epi16(_ww0, _hh_cont);
                __m512i _s1 = _mm512_madd_epi16(_ww1, _hh_cont);
                _sum0 = _mm512_add_epi32(_sum0, _s0);
                _sum1 = _mm512_add_epi32(_sum1, _s1);

                kptr += 64;
            }
            {
                __m512i _tmp0 = _mm512_castps_si512(_mm512_shuffle_ps(_mm512_castsi512_ps(_sum0), _mm512_castsi512_ps(_sum1), _MM_SHUFFLE(2, 0, 2, 0)));
                __m512i _tmp1 = _mm512_castps_si512(_mm512_shuffle_ps(_mm512_castsi512_ps(_sum0), _mm512_castsi512_ps(_sum1), _MM_SHUFFLE(3, 1, 3, 1)));

                _lstm_IFOGh0 = _mm512_add_epi32(_lstm_IFOGh0, _tmp0);
                _lstm_IFOGh0 = _mm512_add_epi32(_lstm_IFOGh0, _tmp1);
            }
#endif // __AVX512VNNI__
            for (; i + 1 < num_output; i += 2)
            {
                __m256i _w = _mm256_loadu_si256((const __m256i*)kptr);
                __m256i _h_cont = _mm256_castps_si256(_mm256_broadcast_ss((const float*)(hs + i)));

                __m512i _ww = _mm512_cvtepi8_epi16(_w);
                __m512i _hh_cont = _mm512_cvtepi8_epi16(_h_cont);

                __m512i _hh_cont0 = _mm512_shuffle_epi32(_hh_cont, _MM_PERM_AAAA);

                _lstm_IFOGh0 = _mm512_comp_dpwssd_epi32(_lstm_IFOGh0, _ww, _hh_cont0);

                kptr += 32;
            }
            for (; i < num_output; i++)
            {
                __m128i _w = _mm_load_si128((const __m128i*)kptr);
                __m256i _h_cont = _mm256_set1_epi16(hs[i]);

                __m256i _ww = _mm256_cvtepi8_epi16(_w);

                __m512i _s0 = _mm512_cvtepi16_epi32(_mm256_mullo_epi16(_ww, _h_cont));

                _lstm_IFOGh0 = _mm512_add_epi32(_lstm_IFOGh0, _s0);

                kptr += 16;
            }

            __m512 _descale_x = _mm512_set1_ps(descale_x);
            __m512 _descale_h = _mm512_set1_ps(descale_h);

            __m512 _lstm_IFOG0 = _mm512_loadu_ps(bias_c_IFOG);

            __m512 _descale_xc_IFOG = _mm512_loadu_ps(descales_ptr);

            _lstm_IFOG0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_lstm_IFOGx0), _mm512_mul_ps(_descale_x, _descale_xc_IFOG), _lstm_IFOG0);

            __m512 _descale_hc_IFOG = _mm512_loadu_ps(descales_ptr + 16);

            _lstm_IFOG0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_lstm_IFOGh0), _mm512_mul_ps(_descale_h, _descale_hc_IFOG), _lstm_IFOG0);

            _mm512_storeu_ps(gates_data, _lstm_IFOG0);
        }
        remain_hidden_size_start += nn_hidden_size << 2;
        nn_hidden_size = (hidden_size - remain_hidden_size_start) >> 1;
#else
        nn_hidden_size = hidden_size >> 1;
#endif // __AVX512F__
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_hidden_size; qq++)
        {
            int q = remain_hidden_size_start + qq * 2;

            const signed char* x = bottom_blob_int8.row<const signed char>(ti);
            const signed char* hs = hidden_state_int8;
            const float descale_x = bottom_blob_int8_descales[ti];
            const float descale_h = hidden_state_int8_descale;

            const float* bias_c_IFOG = (const float*)bias_c + q * 4;

#if __AVX512F__
            const signed char* kptr = weight_data_tm.row<const signed char>(q / 4 + (q % 4) / 2);
            const float* descales_ptr = weight_data_tm_int8_descales.row(q / 4 + (q % 4) / 2);
#else
            const signed char* kptr = weight_data_tm.row<const signed char>(q / 2);
            const float* descales_ptr = weight_data_tm_int8_descales.row(q / 2);
#endif

            float* gates_data = gates.row(q);

            __m256i _lstm_IFOGx0 = _mm256_setzero_si256();
            __m256i _sum0 = _mm256_setzero_si256();
            __m256i _sum1 = _mm256_setzero_si256();
            int i = 0;
#if __AVXVNNI__ || __AVX512VNNI__
#if defined(__x86_64__) || defined(_M_X64)
            __m256i _sum2 = _mm256_setzero_si256();
            __m256i _sum3 = _mm256_setzero_si256();
            for (; i + 15 < size; i += 16)
            {
                __m128i _xi = _mm_loadu_si128((const __m128i*)(x + i));
                __m256i _w0 = _mm256_loadu_si256((const __m256i*)kptr);
                __m256i _w1 = _mm256_loadu_si256((const __m256i*)(kptr + 32));
                __m256i _w2 = _mm256_loadu_si256((const __m256i*)(kptr + 64));
                __m256i _w3 = _mm256_loadu_si256((const __m256i*)(kptr + 96));

                __m256i _xii = _mm256_inserti128_si256(_mm256_castsi128_si256(_xi), _xi, 1);

                _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _xii, _w0);
                _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _xii, _w1);
                _sum2 = _mm256_comp_dpbusd_epi32(_sum2, _xii, _w2);
                _sum3 = _mm256_comp_dpbusd_epi32(_sum3, _xii, _w3);

                kptr += 128;
            }
            {
                __m256i _tmp0 = _mm256_hadd_epi32(_sum0, _sum1);
                __m256i _tmp1 = _mm256_hadd_epi32(_sum2, _sum3);
                _tmp0 = _mm256_hadd_epi32(_tmp0, _tmp1);
                _lstm_IFOGx0 = _mm256_add_epi32(_lstm_IFOGx0, _tmp0);
            }

            _sum0 = _mm256_setzero_si256();
            _sum1 = _mm256_setzero_si256();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < size; i += 8)
            {
                __m256i _xi = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)(x + i)));
                __m256i _w0 = _mm256_loadu_si256((const __m256i*)kptr);
                __m256i _w1 = _mm256_loadu_si256((const __m256i*)(kptr + 32));

                _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _xi, _w0);
                _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _xi, _w1);

                kptr += 64;
            }
            {
                __m256i _tmp0 = _mm256_hadd_epi32(_sum0, _sum1);
                _lstm_IFOGx0 = _mm256_add_epi32(_lstm_IFOGx0, _tmp0);
            }

            for (; i + 3 < size; i += 4)
            {
                __m256i _xi = _mm256_castps_si256(_mm256_broadcast_ss((const float*)(x + i)));
                __m256i _w = _mm256_loadu_si256((const __m256i*)kptr);

#ifdef _MSC_VER
                _xi = _mm256_add_epi32(_xi, _mm256_set1_epi8(127));
#endif
                _lstm_IFOGx0 = _mm256_comp_dpbusd_epi32(_lstm_IFOGx0, _xi, _w);

                kptr += 32;
            }
            {
                __m256i _w_shift = _mm256_loadu_si256((const __m256i*)kptr);
                _lstm_IFOGx0 = _mm256_sub_epi32(_lstm_IFOGx0, _w_shift);
                kptr += 32;
            }
#else
#if defined(__x86_64__) || defined(_M_X64)
            __m256i _sum2 = _mm256_setzero_si256();
            __m256i _sum3 = _mm256_setzero_si256();
            for (; i + 7 < size; i += 8)
            {
                __m128i _xi = _mm_castpd_si128(_mm_load1_pd((const double*)(x + i)));
                __m128i _w0 = _mm_loadu_si128((const __m128i*)kptr);
                __m128i _w1 = _mm_loadu_si128((const __m128i*)(kptr + 16));
                __m128i _w2 = _mm_loadu_si128((const __m128i*)(kptr + 32));
                __m128i _w3 = _mm_loadu_si128((const __m128i*)(kptr + 48));

                __m256i _xii = _mm256_cvtepi8_epi16(_xi);
                __m256i _ww0 = _mm256_cvtepi8_epi16(_w0);
                __m256i _ww1 = _mm256_cvtepi8_epi16(_w1);
                __m256i _ww2 = _mm256_cvtepi8_epi16(_w2);
                __m256i _ww3 = _mm256_cvtepi8_epi16(_w3);

                __m256i _s0 = _mm256_madd_epi16(_ww0, _xii);
                __m256i _s1 = _mm256_madd_epi16(_ww1, _xii);
                __m256i _s2 = _mm256_madd_epi16(_ww2, _xii);
                __m256i _s3 = _mm256_madd_epi16(_ww3, _xii);
                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);
                _sum2 = _mm256_add_epi32(_sum2, _s2);
                _sum3 = _mm256_add_epi32(_sum3, _s3);

                kptr += 64;
            }
            {
                __m256i _tmp0 = _mm256_hadd_epi32(_sum0, _sum1);
                __m256i _tmp1 = _mm256_hadd_epi32(_sum2, _sum3);
                _tmp0 = _mm256_hadd_epi32(_tmp0, _tmp1);
                _lstm_IFOGx0 = _mm256_add_epi32(_lstm_IFOGx0, _tmp0);
            }

            _sum0 = _mm256_setzero_si256();
            _sum1 = _mm256_setzero_si256();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 3 < size; i += 4)
            {
                __m128i _xi = _mm_castps_si128(_mm_load1_ps((const float*)(x + i)));
                __m128i _w0 = _mm_loadu_si128((const __m128i*)kptr);
                __m128i _w1 = _mm_loadu_si128((const __m128i*)(kptr + 16));

                __m256i _xii = _mm256_cvtepi8_epi16(_xi);
                __m256i _ww0 = _mm256_cvtepi8_epi16(_w0);
                __m256i _ww1 = _mm256_cvtepi8_epi16(_w1);

                __m256i _s0 = _mm256_madd_epi16(_ww0, _xii);
                __m256i _s1 = _mm256_madd_epi16(_ww1, _xii);
                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);

                kptr += 32;
            }
            {
                __m256i _tmp0 = _mm256_hadd_epi32(_sum0, _sum1);
                _lstm_IFOGx0 = _mm256_add_epi32(_lstm_IFOGx0, _tmp0);
            }
#endif // __AVXVNNI__ || __AVX512VNNI__
            for (; i + 1 < size; i += 2)
            {
                __m128i _w = _mm_loadu_si128((const __m128i*)kptr);
                __m128i _xi = _mm_castps_si128(_mm_load1_ps((const float*)(x + i)));

                __m256i _ww = _mm256_cvtepi8_epi16(_w);
                __m256i _xixi = _mm256_cvtepi8_epi16(_xi);

                __m256i _xixi0 = _mm256_shuffle_epi32(_xixi, _MM_SHUFFLE(0, 0, 0, 0));

                _lstm_IFOGx0 = _mm256_comp_dpwssd_epi32(_lstm_IFOGx0, _ww, _xixi0);

                kptr += 16;
            }
            for (; i < size; i++)
            {
                __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
                __m128i _xi = _mm_set1_epi16(x[i]);

                _w = _mm_cvtepi8_epi16(_w);

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_w, _xi));

                _lstm_IFOGx0 = _mm256_add_epi32(_lstm_IFOGx0, _s0);

                kptr += 8;
            }

            __m256i _lstm_IFOGh0 = _mm256_setzero_si256();
            _sum0 = _mm256_setzero_si256();
            _sum1 = _mm256_setzero_si256();
            i = 0;
#if __AVXVNNI__ || __AVX512VNNI__
#if defined(__x86_64__) || defined(_M_X64)
            _sum2 = _mm256_setzero_si256();
            _sum3 = _mm256_setzero_si256();
            for (; i + 15 < num_output; i += 16)
            {
                __m128i _h_cont = _mm_loadu_si128((const __m128i*)(hs + i));
                __m256i _w0 = _mm256_loadu_si256((const __m256i*)kptr);
                __m256i _w1 = _mm256_loadu_si256((const __m256i*)(kptr + 32));
                __m256i _w2 = _mm256_loadu_si256((const __m256i*)(kptr + 64));
                __m256i _w3 = _mm256_loadu_si256((const __m256i*)(kptr + 96));

                __m256i _hh_cont = _mm256_broadcastsi128_si256(_h_cont);

                _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _hh_cont, _w0);
                _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _hh_cont, _w1);
                _sum2 = _mm256_comp_dpbusd_epi32(_sum2, _hh_cont, _w2);
                _sum3 = _mm256_comp_dpbusd_epi32(_sum3, _hh_cont, _w3);

                kptr += 128;
            }
            {
                __m256i _tmp0 = _mm256_hadd_epi32(_sum0, _sum1);
                __m256i _tmp1 = _mm256_hadd_epi32(_sum2, _sum3);
                _tmp0 = _mm256_hadd_epi32(_tmp0, _tmp1);
                _lstm_IFOGh0 = _mm256_add_epi32(_lstm_IFOGh0, _tmp0);
            }

            _sum0 = _mm256_setzero_si256();
            _sum1 = _mm256_setzero_si256();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < num_output; i += 8)
            {
                __m256i _h_cont = _mm256_castpd_si256(_mm256_broadcast_sd((const double*)(hs + i)));
                __m256i _w0 = _mm256_loadu_si256((const __m256i*)kptr);
                __m256i _w1 = _mm256_loadu_si256((const __m256i*)(kptr + 32));

                _sum0 = _mm256_comp_dpbusd_epi32(_sum0, _h_cont, _w0);
                _sum1 = _mm256_comp_dpbusd_epi32(_sum1, _h_cont, _w1);

                kptr += 64;
            }
            {
                __m256i _tmp0 = _mm256_hadd_epi32(_sum0, _sum1);
                _lstm_IFOGh0 = _mm256_add_epi32(_lstm_IFOGh0, _tmp0);
            }

            for (; i + 3 < num_output; i += 4)
            {
                __m256i _h_cont = _mm256_castps_si256(_mm256_broadcast_ss((const float*)(hs + i)));
                __m256i _w = _mm256_loadu_si256((const __m256i*)kptr);

#ifdef _MSC_VER
                _h_cont = _mm256_add_epi32(_h_cont, _mm256_set1_epi8(127));
#endif
                _lstm_IFOGh0 = _mm256_comp_dpbusd_epi32(_lstm_IFOGh0, _h_cont, _w);

                kptr += 32;
            }
            {
                __m256i _w_shift = _mm256_loadu_si256((const __m256i*)kptr);
                _lstm_IFOGh0 = _mm256_sub_epi32(_lstm_IFOGh0, _w_shift);
                kptr += 32;
            }
#else
#if defined(__x86_64__) || defined(_M_X64)
            _sum2 = _mm256_setzero_si256();
            _sum3 = _mm256_setzero_si256();
            for (; i + 7 < num_output; i += 8)
            {
                __m128i _h_cont = _mm_castpd_si128(_mm_load1_pd((const double*)(hs + i)));
                __m128i _w0 = _mm_loadu_si128((const __m128i*)kptr);
                __m128i _w1 = _mm_loadu_si128((const __m128i*)(kptr + 16));
                __m128i _w2 = _mm_loadu_si128((const __m128i*)(kptr + 32));
                __m128i _w3 = _mm_loadu_si128((const __m128i*)(kptr + 48));

                __m256i _hh_cont = _mm256_cvtepi8_epi16(_h_cont);
                __m256i _ww0 = _mm256_cvtepi8_epi16(_w0);
                __m256i _ww1 = _mm256_cvtepi8_epi16(_w1);
                __m256i _ww2 = _mm256_cvtepi8_epi16(_w2);
                __m256i _ww3 = _mm256_cvtepi8_epi16(_w3);

                __m256i _s0 = _mm256_madd_epi16(_ww0, _hh_cont);
                __m256i _s1 = _mm256_madd_epi16(_ww1, _hh_cont);
                __m256i _s2 = _mm256_madd_epi16(_ww2, _hh_cont);
                __m256i _s3 = _mm256_madd_epi16(_ww3, _hh_cont);
                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);
                _sum2 = _mm256_add_epi32(_sum2, _s2);
                _sum3 = _mm256_add_epi32(_sum3, _s3);

                kptr += 64;
            }
            {
                __m256i _tmp0 = _mm256_hadd_epi32(_sum0, _sum1);
                __m256i _tmp1 = _mm256_hadd_epi32(_sum2, _sum3);
                _tmp0 = _mm256_hadd_epi32(_tmp0, _tmp1);
                _lstm_IFOGh0 = _mm256_add_epi32(_lstm_IFOGh0, _tmp0);
            }

            _sum0 = _mm256_setzero_si256();
            _sum1 = _mm256_setzero_si256();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 3 < num_output; i += 4)
            {
                __m128i _h_cont = _mm_castps_si128(_mm_load1_ps((const float*)(hs + i)));
                __m128i _w0 = _mm_loadu_si128((const __m128i*)kptr);
                __m128i _w1 = _mm_loadu_si128((const __m128i*)(kptr + 16));

                __m256i _hh_cont = _mm256_cvtepi8_epi16(_h_cont);
                __m256i _ww0 = _mm256_cvtepi8_epi16(_w0);
                __m256i _ww1 = _mm256_cvtepi8_epi16(_w1);

                __m256i _s0 = _mm256_madd_epi16(_ww0, _hh_cont);
                __m256i _s1 = _mm256_madd_epi16(_ww1, _hh_cont);
                _sum0 = _mm256_add_epi32(_sum0, _s0);
                _sum1 = _mm256_add_epi32(_sum1, _s1);

                kptr += 32;
            }
            {
                __m256i _tmp0 = _mm256_hadd_epi32(_sum0, _sum1);
                _lstm_IFOGh0 = _mm256_add_epi32(_lstm_IFOGh0, _tmp0);
            }
#endif // __AVXVNNI__ || __AVX512VNNI__
            for (; i + 1 < num_output; i += 2)
            {
                __m128i _w = _mm_loadu_si128((const __m128i*)kptr);
                __m128i _h_cont = _mm_castps_si128(_mm_load1_ps((const float*)(hs + i)));

                __m256i _ww = _mm256_cvtepi8_epi16(_w);
                __m256i _hh_cont = _mm256_cvtepi8_epi16(_h_cont);

                __m256i _hh_cont0 = _mm256_shuffle_epi32(_hh_cont, _MM_SHUFFLE(0, 0, 0, 0));

                _lstm_IFOGh0 = _mm256_comp_dpwssd_epi32(_lstm_IFOGh0, _ww, _hh_cont0);

                kptr += 16;
            }
            for (; i < num_output; i++)
            {
                __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
                __m128i _h_cont = _mm_set1_epi16(hs[i]);

                _w = _mm_cvtepi8_epi16(_w);

                __m256i _s0 = _mm256_cvtepi16_epi32(_mm_mullo_epi16(_w, _h_cont));

                _lstm_IFOGh0 = _mm256_add_epi32(_lstm_IFOGh0, _s0);

                kptr += 8;
            }

            __m256 _descale_x = _mm256_set1_ps(descale_x);
            __m256 _descale_h = _mm256_set1_ps(descale_h);

            __m256 _lstm_IFOG0 = _mm256_loadu_ps(bias_c_IFOG);

            __m256 _descale_xc_IFOG = _mm256_loadu_ps(descales_ptr);

            _lstm_IFOG0 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_lstm_IFOGx0), _mm256_mul_ps(_descale_x, _descale_xc_IFOG), _lstm_IFOG0);

            __m256 _descale_hc_IFOG = _mm256_loadu_ps(descales_ptr + 8);

            _lstm_IFOG0 = _mm256_comp_fmadd_ps(_mm256_cvtepi32_ps(_lstm_IFOGh0), _mm256_mul_ps(_descale_h, _descale_hc_IFOG), _lstm_IFOG0);

            _mm256_storeu_ps(gates_data, _lstm_IFOG0);
        }
        remain_hidden_size_start += nn_hidden_size << 1;
#endif // __AVX2__
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = remain_hidden_size_start; q < hidden_size; q++)
        {
            const signed char* x = bottom_blob_int8.row<const signed char>(ti);
            const signed char* hs = hidden_state_int8;
            const float descale_x = bottom_blob_int8_descales[ti];
            const float descale_h = hidden_state_int8_descale;

            // gate reset update
            const float* bias_c_IFOG = (const float*)bias_c + q * 4;

#if __AVX512F__
            const signed char* kptr = weight_data_tm.row<const signed char>(q / 4 + (q % 4) / 2 + q % 2);
            const float* descales_ptr = weight_data_tm_int8_descales.row(q / 4 + (q % 4) / 2 + q % 2);
#elif __AVX2__
            const signed char* kptr = weight_data_tm.row<const signed char>(q / 2 + q % 2);
            const float* descales_ptr = weight_data_tm_int8_descales.row(q / 2 + q % 2);
#else
            const signed char* kptr = weight_data_tm.row<const signed char>(q);
            const float* descales_ptr = weight_data_tm_int8_descales.row(q);
#endif

            float* gates_data = gates.row(q);

#if __SSE2__
            __m128i _lstm_IFOGx0 = _mm_setzero_si128();
            __m128i _sum0 = _mm_setzero_si128();
            __m128i _sum1 = _mm_setzero_si128();
            int i = 0;
#if __AVXVNNI__ || __AVX512VNNI__
#if defined(__x86_64__) || defined(_M_X64)
            __m128i _sum2 = _mm_setzero_si128();
            __m128i _sum3 = _mm_setzero_si128();
            for (; i + 15 < size; i += 16)
            {
                __m128i _xi = _mm_loadu_si128((const __m128i*)(x + i));
                __m128i _w0 = _mm_loadu_si128((const __m128i*)kptr);
                __m128i _w1 = _mm_loadu_si128((const __m128i*)(kptr + 16));
                __m128i _w2 = _mm_loadu_si128((const __m128i*)(kptr + 32));
                __m128i _w3 = _mm_loadu_si128((const __m128i*)(kptr + 48));

                _sum0 = _mm_comp_dpbusd_epi32(_sum0, _xi, _w0);
                _sum1 = _mm_comp_dpbusd_epi32(_sum1, _xi, _w1);
                _sum2 = _mm_comp_dpbusd_epi32(_sum2, _xi, _w2);
                _sum3 = _mm_comp_dpbusd_epi32(_sum3, _xi, _w3);

                kptr += 64;
            }
            {
                transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
                _lstm_IFOGx0 = _mm_add_epi32(_lstm_IFOGx0, _sum0);
                _lstm_IFOGx0 = _mm_add_epi32(_lstm_IFOGx0, _sum1);
                _lstm_IFOGx0 = _mm_add_epi32(_lstm_IFOGx0, _sum2);
                _lstm_IFOGx0 = _mm_add_epi32(_lstm_IFOGx0, _sum3);
            }

            _sum0 = _mm_setzero_si128();
            _sum1 = _mm_setzero_si128();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < size; i += 8)
            {
                __m128i _xi = _mm_castpd_si128(_mm_load1_pd((const double*)(x + i)));
                __m128i _w0 = _mm_loadu_si128((const __m128i*)kptr);
                __m128i _w1 = _mm_loadu_si128((const __m128i*)(kptr + 16));

                _sum0 = _mm_comp_dpbusd_epi32(_sum0, _xi, _w0);
                _sum1 = _mm_comp_dpbusd_epi32(_sum1, _xi, _w1);

                kptr += 32;
            }
            {
                __m128i _tmp0 = _mm_hadd_epi32(_sum0, _sum1);
                _lstm_IFOGx0 = _mm_add_epi32(_lstm_IFOGx0, _tmp0);
            }

            for (; i + 3 < size; i += 4)
            {
                __m128i _xi = _mm_castps_si128(_mm_load1_ps((const float*)(x + i)));
                __m128i _w = _mm_loadu_si128((const __m128i*)kptr);

#ifdef _MSC_VER
                _xi = _mm_add_epi32(_xi, _mm_set1_epi8(127));
#endif
                _lstm_IFOGx0 = _mm_comp_dpbusd_epi32(_lstm_IFOGx0, _xi, _w);

                kptr += 16;
            }
            {
                __m128i _w_shift = _mm_loadu_si128((const __m128i*)kptr);
                _lstm_IFOGx0 = _mm_sub_epi32(_lstm_IFOGx0, _w_shift);
                kptr += 16;
            }
#else
#if defined(__x86_64__) || defined(_M_X64)
            __m128i _sum2 = _mm_setzero_si128();
            __m128i _sum3 = _mm_setzero_si128();
            for (; i + 7 < size; i += 8)
            {
                __m128i _xi = _mm_castpd_si128(_mm_load1_pd((const double*)(x + i)));
                __m128i _w0 = _mm_loadl_epi64((const __m128i*)kptr);
                __m128i _w1 = _mm_loadl_epi64((const __m128i*)(kptr + 8));
                __m128i _w2 = _mm_loadl_epi64((const __m128i*)(kptr + 16));
                __m128i _w3 = _mm_loadl_epi64((const __m128i*)(kptr + 24));

#if __SSE4_1__
                _xi = _mm_cvtepi8_epi16(_xi);
                _w0 = _mm_cvtepi8_epi16(_w0);
                _w1 = _mm_cvtepi8_epi16(_w1);
                _w2 = _mm_cvtepi8_epi16(_w2);
                _w3 = _mm_cvtepi8_epi16(_w3);
#else
                _xi = _mm_unpacklo_epi8(_xi, _mm_cmpgt_epi8(_mm_setzero_si128(), _xi));
                _w0 = _mm_unpacklo_epi8(_w0, _mm_cmpgt_epi8(_mm_setzero_si128(), _w0));
                _w1 = _mm_unpacklo_epi8(_w1, _mm_cmpgt_epi8(_mm_setzero_si128(), _w1));
                _w2 = _mm_unpacklo_epi8(_w2, _mm_cmpgt_epi8(_mm_setzero_si128(), _w2));
                _w3 = _mm_unpacklo_epi8(_w3, _mm_cmpgt_epi8(_mm_setzero_si128(), _w3));
#endif

                _sum0 = _mm_comp_dpwssd_epi32(_sum0, _w0, _xi);
                _sum1 = _mm_comp_dpwssd_epi32(_sum1, _w1, _xi);
                _sum2 = _mm_comp_dpwssd_epi32(_sum2, _w2, _xi);
                _sum3 = _mm_comp_dpwssd_epi32(_sum3, _w3, _xi);

                kptr += 32;
            }
            {
                transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
                _lstm_IFOGx0 = _mm_add_epi32(_lstm_IFOGx0, _sum0);
                _lstm_IFOGx0 = _mm_add_epi32(_lstm_IFOGx0, _sum1);
                _lstm_IFOGx0 = _mm_add_epi32(_lstm_IFOGx0, _sum2);
                _lstm_IFOGx0 = _mm_add_epi32(_lstm_IFOGx0, _sum3);
            }

            _sum0 = _mm_setzero_si128();
            _sum1 = _mm_setzero_si128();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 3 < size; i += 4)
            {
                __m128i _xi = _mm_castps_si128(_mm_load1_ps((const float*)(x + i)));
                __m128i _w0 = _mm_loadl_epi64((const __m128i*)kptr);
                __m128i _w1 = _mm_loadl_epi64((const __m128i*)(kptr + 8));

#if __SSE4_1__
                _xi = _mm_cvtepi8_epi16(_xi);
                _w0 = _mm_cvtepi8_epi16(_w0);
                _w1 = _mm_cvtepi8_epi16(_w1);
#else
                _xi = _mm_unpacklo_epi8(_xi, _mm_cmpgt_epi8(_mm_setzero_si128(), _xi));
                _w0 = _mm_unpacklo_epi8(_w0, _mm_cmpgt_epi8(_mm_setzero_si128(), _w0));
                _w1 = _mm_unpacklo_epi8(_w1, _mm_cmpgt_epi8(_mm_setzero_si128(), _w1));
#endif

                _sum0 = _mm_comp_dpwssd_epi32(_sum0, _w0, _xi);
                _sum1 = _mm_comp_dpwssd_epi32(_sum1, _w1, _xi);

                kptr += 16;
            }
            {
#if __SSSE3__
                __m128i _tmp0 = _mm_hadd_epi32(_sum0, _sum1);
                _lstm_IFOGx0 = _mm_add_epi32(_lstm_IFOGx0, _tmp0);
#else
                __m128i _tmp0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_sum0), _mm_castsi128_ps(_sum1), _MM_SHUFFLE(2, 0, 2, 0)));
                __m128i _tmp1 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_sum0), _mm_castsi128_ps(_sum1), _MM_SHUFFLE(3, 1, 3, 1)));
                _lstm_IFOGx0 = _mm_add_epi32(_lstm_IFOGx0, _tmp0);
                _lstm_IFOGx0 = _mm_add_epi32(_lstm_IFOGx0, _tmp1);
#endif // __SSSE3__
            }
#endif // __AVXVNNI__ || __AVX512VNNI__
            for (; i + 1 < size; i += 2)
            {
                __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
                __m128i _xi = _mm_set1_epi16(((const short*)(x + i))[0]);

#if __SSE4_1__
                _w = _mm_cvtepi8_epi16(_w);
                _xi = _mm_cvtepi8_epi16(_xi);
#else
                _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
                _xi = _mm_unpacklo_epi8(_xi, _mm_cmpgt_epi8(_mm_setzero_si128(), _xi));
#endif

                _lstm_IFOGx0 = _mm_comp_dpwssd_epi32(_lstm_IFOGx0, _w, _xi);

                kptr += 8;
            }
            for (; i < size; i++)
            {
                __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
                __m128i _xi = _mm_set1_epi16(x[i]);

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
            _sum0 = _mm_setzero_si128();
            _sum1 = _mm_setzero_si128();
            i = 0;
#if __AVXVNNI__ || __AVX512VNNI__
#if defined(__x86_64__) || defined(_M_X64)
            _sum2 = _mm_setzero_si128();
            _sum3 = _mm_setzero_si128();
            for (; i + 15 < num_output; i += 16)
            {
                __m128i _h_cont = _mm_loadu_si128((const __m128i*)(hs + i));
                __m128i _w0 = _mm_loadu_si128((const __m128i*)kptr);
                __m128i _w1 = _mm_loadu_si128((const __m128i*)(kptr + 16));
                __m128i _w2 = _mm_loadu_si128((const __m128i*)(kptr + 32));
                __m128i _w3 = _mm_loadu_si128((const __m128i*)(kptr + 48));

                _sum0 = _mm_comp_dpbusd_epi32(_sum0, _h_cont, _w0);
                _sum1 = _mm_comp_dpbusd_epi32(_sum1, _h_cont, _w1);
                _sum2 = _mm_comp_dpbusd_epi32(_sum2, _h_cont, _w2);
                _sum3 = _mm_comp_dpbusd_epi32(_sum3, _h_cont, _w3);

                kptr += 64;
            }
            {
                transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
                _lstm_IFOGh0 = _mm_add_epi32(_lstm_IFOGh0, _sum0);
                _lstm_IFOGh0 = _mm_add_epi32(_lstm_IFOGh0, _sum1);
                _lstm_IFOGh0 = _mm_add_epi32(_lstm_IFOGh0, _sum2);
                _lstm_IFOGh0 = _mm_add_epi32(_lstm_IFOGh0, _sum3);
            }

            _sum0 = _mm_setzero_si128();
            _sum1 = _mm_setzero_si128();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 7 < num_output; i += 8)
            {
                __m128i _h_cont = _mm_castpd_si128(_mm_load1_pd((const double*)(hs + i)));
                __m128i _w0 = _mm_loadu_si128((const __m128i*)kptr);
                __m128i _w1 = _mm_loadu_si128((const __m128i*)(kptr + 16));

                _sum0 = _mm_comp_dpbusd_epi32(_sum0, _h_cont, _w0);
                _sum1 = _mm_comp_dpbusd_epi32(_sum1, _h_cont, _w1);

                kptr += 32;
            }
            {
                __m128i _tmp0 = _mm_hadd_epi32(_sum0, _sum1);
                _lstm_IFOGh0 = _mm_add_epi32(_lstm_IFOGh0, _tmp0);
            }

            for (; i + 3 < num_output; i += 4)
            {
                __m128i _h_cont = _mm_castps_si128(_mm_load1_ps((const float*)(hs + i)));
                __m128i _w = _mm_loadu_si128((const __m128i*)kptr);

#ifdef _MSC_VER
                _h_cont = _mm_add_epi32(_h_cont, _mm_set1_epi8(127));
#endif
                _lstm_IFOGh0 = _mm_comp_dpbusd_epi32(_lstm_IFOGh0, _h_cont, _w);

                kptr += 16;
            }
            {
                __m128i _w_shift = _mm_loadu_si128((const __m128i*)kptr);
                _lstm_IFOGh0 = _mm_sub_epi32(_lstm_IFOGh0, _w_shift);
                kptr += 16;
            }
#else
#if defined(__x86_64__) || defined(_M_X64)
            _sum2 = _mm_setzero_si128();
            _sum3 = _mm_setzero_si128();
            for (; i + 7 < num_output; i += 8)
            {
                __m128i _h_cont = _mm_castpd_si128(_mm_load1_pd((const double*)(hs + i)));
                __m128i _w0 = _mm_loadl_epi64((const __m128i*)kptr);
                __m128i _w1 = _mm_loadl_epi64((const __m128i*)(kptr + 8));
                __m128i _w2 = _mm_loadl_epi64((const __m128i*)(kptr + 16));
                __m128i _w3 = _mm_loadl_epi64((const __m128i*)(kptr + 24));

#if __SSE4_1__
                _h_cont = _mm_cvtepi8_epi16(_h_cont);
                _w0 = _mm_cvtepi8_epi16(_w0);
                _w1 = _mm_cvtepi8_epi16(_w1);
                _w2 = _mm_cvtepi8_epi16(_w2);
                _w3 = _mm_cvtepi8_epi16(_w3);
#else
                _h_cont = _mm_unpacklo_epi8(_h_cont, _mm_cmpgt_epi8(_mm_setzero_si128(), _h_cont));
                _w0 = _mm_unpacklo_epi8(_w0, _mm_cmpgt_epi8(_mm_setzero_si128(), _w0));
                _w1 = _mm_unpacklo_epi8(_w1, _mm_cmpgt_epi8(_mm_setzero_si128(), _w1));
                _w2 = _mm_unpacklo_epi8(_w2, _mm_cmpgt_epi8(_mm_setzero_si128(), _w2));
                _w3 = _mm_unpacklo_epi8(_w3, _mm_cmpgt_epi8(_mm_setzero_si128(), _w3));
#endif

                _sum0 = _mm_comp_dpwssd_epi32(_sum0, _w0, _h_cont);
                _sum1 = _mm_comp_dpwssd_epi32(_sum1, _w1, _h_cont);
                _sum2 = _mm_comp_dpwssd_epi32(_sum2, _w2, _h_cont);
                _sum3 = _mm_comp_dpwssd_epi32(_sum3, _w3, _h_cont);

                kptr += 32;
            }
            {
                transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
                _lstm_IFOGh0 = _mm_add_epi32(_lstm_IFOGh0, _sum0);
                _lstm_IFOGh0 = _mm_add_epi32(_lstm_IFOGh0, _sum1);
                _lstm_IFOGh0 = _mm_add_epi32(_lstm_IFOGh0, _sum2);
                _lstm_IFOGh0 = _mm_add_epi32(_lstm_IFOGh0, _sum3);
            }

            _sum0 = _mm_setzero_si128();
            _sum1 = _mm_setzero_si128();
#endif // defined(__x86_64__) || defined(_M_X64)
            for (; i + 3 < num_output; i += 4)
            {
                __m128i _h_cont = _mm_castps_si128(_mm_load1_ps((const float*)(hs + i)));
                __m128i _w0 = _mm_loadl_epi64((const __m128i*)kptr);
                __m128i _w1 = _mm_loadl_epi64((const __m128i*)(kptr + 8));

#if __SSE4_1__
                _h_cont = _mm_cvtepi8_epi16(_h_cont);
                _w0 = _mm_cvtepi8_epi16(_w0);
                _w1 = _mm_cvtepi8_epi16(_w1);
#else
                _h_cont = _mm_unpacklo_epi8(_h_cont, _mm_cmpgt_epi8(_mm_setzero_si128(), _h_cont));
                _w0 = _mm_unpacklo_epi8(_w0, _mm_cmpgt_epi8(_mm_setzero_si128(), _w0));
                _w1 = _mm_unpacklo_epi8(_w1, _mm_cmpgt_epi8(_mm_setzero_si128(), _w1));
#endif

                _sum0 = _mm_comp_dpwssd_epi32(_sum0, _w0, _h_cont);
                _sum1 = _mm_comp_dpwssd_epi32(_sum1, _w1, _h_cont);

                kptr += 16;
            }
            {
#if __SSSE3__
                __m128i _tmp0 = _mm_hadd_epi32(_sum0, _sum1);
                _lstm_IFOGh0 = _mm_add_epi32(_lstm_IFOGh0, _tmp0);
#else
                __m128i _tmp0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_sum0), _mm_castsi128_ps(_sum1), _MM_SHUFFLE(2, 0, 2, 0)));
                __m128i _tmp1 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_sum0), _mm_castsi128_ps(_sum1), _MM_SHUFFLE(3, 1, 3, 1)));
                _lstm_IFOGh0 = _mm_add_epi32(_lstm_IFOGh0, _tmp0);
                _lstm_IFOGh0 = _mm_add_epi32(_lstm_IFOGh0, _tmp1);
#endif // __SSSE3__
            }
#endif // __AVXVNNI__ || __AVX512VNNI__
            for (; i + 1 < num_output; i += 2)
            {
                __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
                __m128i _h_cont = _mm_set1_epi16(((const short*)(hs + i))[0]);

#if __SSE4_1__
                _w = _mm_cvtepi8_epi16(_w);
                _h_cont = _mm_cvtepi8_epi16(_h_cont);
#else
                _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));
                _h_cont = _mm_unpacklo_epi8(_h_cont, _mm_cmpgt_epi8(_mm_setzero_si128(), _h_cont));
#endif

                _lstm_IFOGh0 = _mm_comp_dpwssd_epi32(_lstm_IFOGh0, _w, _h_cont);

                kptr += 8;
            }
            for (; i < num_output; i++)
            {
                __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
                __m128i _h_cont = _mm_set1_epi16(hs[i]);

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

            _lstm_IFOG0 = _mm_comp_fmadd_ps(_mm_cvtepi32_ps(_lstm_IFOGx0), _mm_mul_ps(_descale_x, _descale_xc_IFOG), _lstm_IFOG0);

            __m128 _descale_hc_IFOG = _mm_loadu_ps(descales_ptr + 4);

            _lstm_IFOG0 = _mm_comp_fmadd_ps(_mm_cvtepi32_ps(_lstm_IFOGh0), _mm_mul_ps(_descale_h, _descale_hc_IFOG), _lstm_IFOG0);

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

        remain_hidden_size_start = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        nn_hidden_size = hidden_size >> 4;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_hidden_size; qq++)
        {
            int q = qq * 16;

            const float* gates_data = gates.row(q);

            __m512 _IFOG_0 = _mm512_loadu_ps(gates_data);
            __m512 _IFOG_1 = _mm512_loadu_ps(gates_data + 16);
            __m512 _IFOG_2 = _mm512_loadu_ps(gates_data + 32);
            __m512 _IFOG_3 = _mm512_loadu_ps(gates_data + 48);

            __m512 _tmp0 = _mm512_shuffle_f32x4(_IFOG_0, _IFOG_1, _MM_SHUFFLE(1, 0, 1, 0));
            __m512 _tmp1 = _mm512_shuffle_f32x4(_IFOG_2, _IFOG_3, _MM_SHUFFLE(1, 0, 1, 0));
            __m512 _tmp2 = _mm512_shuffle_f32x4(_IFOG_0, _IFOG_1, _MM_SHUFFLE(3, 2, 3, 2));
            __m512 _tmp3 = _mm512_shuffle_f32x4(_IFOG_2, _IFOG_3, _MM_SHUFFLE(3, 2, 3, 2));
            __m512 _lstm_I = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
            __m512 _lstm_F = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
            __m512 _lstm_O = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
            __m512 _lstm_G = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));

            _lstm_I = sigmoid_avx512(_lstm_I);
            _lstm_F = sigmoid_avx512(_lstm_F);
            _lstm_O = sigmoid_avx512(_lstm_O);
            _lstm_G = tanh_avx512(_lstm_G);

            __m512 _cell2 = _mm512_add_ps(_mm512_mul_ps(_lstm_F, _mm512_loadu_ps(cell_ptr + q)), _mm512_mul_ps(_lstm_I, _lstm_G));
            __m512 _lstm_H = _mm512_mul_ps(_lstm_O, tanh_avx512(_cell2));

            _mm512_storeu_ps(cell_ptr + q, _cell2);

            if (num_output == hidden_size)
            {
                _mm512_storeu_ps(hidden_ptr + q, _lstm_H);
                _mm512_storeu_ps(output_data + q, _lstm_H);
            }
            else
            {
                _mm512_storeu_ps(tmp_hidden_ptr + q, _lstm_H);
            }
        }
        remain_hidden_size_start += nn_hidden_size << 4;
        nn_hidden_size = (hidden_size - remain_hidden_size_start) >> 3;
#else
        nn_hidden_size = hidden_size >> 3;
#endif // __AVX512F__
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_hidden_size; qq++)
        {
            int q = remain_hidden_size_start + qq * 8;

            const float* gates_data = gates.row(q);

            __m256 _IFOG_0 = _mm256_loadu_ps(gates_data);
            __m256 _IFOG_1 = _mm256_loadu_ps(gates_data + 8);
            __m256 _IFOG_2 = _mm256_loadu_ps(gates_data + 16);
            __m256 _IFOG_3 = _mm256_loadu_ps(gates_data + 24);

#if __AVX512F__
            __m256 _lstm_I = _mm256_permute2f128_ps(_IFOG_0, _IFOG_2, _MM_SHUFFLE(0, 2, 0, 0));
            __m256 _lstm_F = _mm256_permute2f128_ps(_IFOG_0, _IFOG_2, _MM_SHUFFLE(0, 3, 0, 1));
            __m256 _lstm_O = _mm256_permute2f128_ps(_IFOG_1, _IFOG_3, _MM_SHUFFLE(0, 2, 0, 0));
            __m256 _lstm_G = _mm256_permute2f128_ps(_IFOG_1, _IFOG_3, _MM_SHUFFLE(0, 3, 0, 1));
#else
            // unzip4
            __m256 _tmp0 = _mm256_permute2f128_ps(_IFOG_0, _IFOG_2, _MM_SHUFFLE(0, 2, 0, 0));
            __m256 _tmp1 = _mm256_permute2f128_ps(_IFOG_1, _IFOG_3, _MM_SHUFFLE(0, 2, 0, 0));
            __m256 _tmp2 = _mm256_permute2f128_ps(_IFOG_0, _IFOG_2, _MM_SHUFFLE(0, 3, 0, 1));
            __m256 _tmp3 = _mm256_permute2f128_ps(_IFOG_1, _IFOG_3, _MM_SHUFFLE(0, 3, 0, 1));
            __m256 _tmp4 = _mm256_unpacklo_ps(_tmp0, _tmp1);
            __m256 _tmp5 = _mm256_unpacklo_ps(_tmp2, _tmp3);
            __m256 _tmp6 = _mm256_unpackhi_ps(_tmp0, _tmp1);
            __m256 _tmp7 = _mm256_unpackhi_ps(_tmp2, _tmp3);
            __m256 _lstm_I = _mm256_unpacklo_ps(_tmp4, _tmp5);
            __m256 _lstm_F = _mm256_unpackhi_ps(_tmp4, _tmp5);
            __m256 _lstm_O = _mm256_unpacklo_ps(_tmp6, _tmp7);
            __m256 _lstm_G = _mm256_unpackhi_ps(_tmp6, _tmp7);
#endif

            _lstm_I = sigmoid_avx(_lstm_I);
            _lstm_F = sigmoid_avx(_lstm_F);
            _lstm_O = sigmoid_avx(_lstm_O);
            _lstm_G = tanh_avx(_lstm_G);

            __m256 _cell2 = _mm256_add_ps(_mm256_mul_ps(_lstm_F, _mm256_loadu_ps(cell_ptr + q)), _mm256_mul_ps(_lstm_I, _lstm_G));
            __m256 _lstm_H = _mm256_mul_ps(_lstm_O, tanh_avx(_cell2));

            _mm256_storeu_ps(cell_ptr + q, _cell2);

            if (num_output == hidden_size)
            {
                _mm256_storeu_ps(hidden_ptr + q, _lstm_H);
                _mm256_storeu_ps(output_data + q, _lstm_H);
            }
            else
            {
                _mm256_storeu_ps(tmp_hidden_ptr + q, _lstm_H);
            }
        }
        remain_hidden_size_start += nn_hidden_size << 3;
        nn_hidden_size = (hidden_size - remain_hidden_size_start) >> 2;
#else
        nn_hidden_size = hidden_size >> 2;
#endif // __AVX__
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int qq = 0; qq < nn_hidden_size; qq++)
        {
            int q = remain_hidden_size_start + qq * 4;

            const float* gates_data = gates.row(q);

            __m128 _lstm_I = _mm_loadu_ps(gates_data);
            __m128 _lstm_F = _mm_loadu_ps(gates_data + 4);
            __m128 _lstm_O = _mm_loadu_ps(gates_data + 8);
            __m128 _lstm_G = _mm_loadu_ps(gates_data + 12);

#if !__AVX512F__
            _MM_TRANSPOSE4_PS(_lstm_I, _lstm_F, _lstm_O, _lstm_G);
#endif

            _lstm_I = sigmoid_sse(_lstm_I);
            _lstm_F = sigmoid_sse(_lstm_F);
            _lstm_O = sigmoid_sse(_lstm_O);
            _lstm_G = tanh_sse(_lstm_G);

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
        remain_hidden_size_start += nn_hidden_size << 2;
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
