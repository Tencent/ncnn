// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void innerproduct_gemm_bf16s_sse_avx512bf16(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
#endif

static void innerproduct_gemm_bf16s_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        innerproduct_gemm_bf16s_sse_avx512bf16(bottom_blob, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);
        return;
    }
#else // NCNN_RUNTIME_CPU

    const int num_input = bottom_blob.w;
    const int elempack = bottom_blob.elempack;
    const int num_output = top_blob.w;
    const int h = bottom_blob.h;

    const float* bias_data_ptr = bias_data;

    int num_output_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        num_output_elempack = num_output % 16 == 0 ? 16 : num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#elif __AVX__
        num_output_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        num_output_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int j = 0; j < h; j++)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16 && num_output_elempack == 16)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m512 _sum0 = _mm512_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm512_loadu_ps(bias_data_ptr + p * 16);
                }

                __m512 _sum1 = _sum0;
                __m512 _sum2 = _sum0;
                __m512 _sum3 = _sum0;
                __m512 _sum4 = _sum0;
                __m512 _sum5 = _sum0;
                __m512 _sum6 = _sum0;
                __m512 _sum7 = _sum0;
                __m512 _sum8 = _sum0;
                __m512 _sum9 = _sum0;
                __m512 _suma = _sum0;
                __m512 _sumb = _sum0;
                __m512 _sumc = _sum0;
                __m512 _sumd = _sum0;
                __m512 _sume = _sum0;
                __m512 _sumf = _sum0;

                for (int i = 0; i < num_input; i++)
                {
                    __m512 _val0 = _mm512_set1_ps(bfloat16_to_float32(m[0]));
                    __m512 _val1 = _mm512_set1_ps(bfloat16_to_float32(m[1]));
                    __m512 _val2 = _mm512_set1_ps(bfloat16_to_float32(m[2]));
                    __m512 _val3 = _mm512_set1_ps(bfloat16_to_float32(m[3]));
                    __m512 _val4 = _mm512_set1_ps(bfloat16_to_float32(m[4]));
                    __m512 _val5 = _mm512_set1_ps(bfloat16_to_float32(m[5]));
                    __m512 _val6 = _mm512_set1_ps(bfloat16_to_float32(m[6]));
                    __m512 _val7 = _mm512_set1_ps(bfloat16_to_float32(m[7]));
                    __m512 _val8 = _mm512_set1_ps(bfloat16_to_float32(m[8]));
                    __m512 _val9 = _mm512_set1_ps(bfloat16_to_float32(m[9]));
                    __m512 _vala = _mm512_set1_ps(bfloat16_to_float32(m[10]));
                    __m512 _valb = _mm512_set1_ps(bfloat16_to_float32(m[11]));
                    __m512 _valc = _mm512_set1_ps(bfloat16_to_float32(m[12]));
                    __m512 _vald = _mm512_set1_ps(bfloat16_to_float32(m[13]));
                    __m512 _vale = _mm512_set1_ps(bfloat16_to_float32(m[14]));
                    __m512 _valf = _mm512_set1_ps(bfloat16_to_float32(m[15]));

                    __m512 _w = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)kptr));

                    _sum0 = _mm512_fmadd_ps(_val0, _w, _sum0);
                    _sum1 = _mm512_fmadd_ps(_val1, _w, _sum1);
                    _sum2 = _mm512_fmadd_ps(_val2, _w, _sum2);
                    _sum3 = _mm512_fmadd_ps(_val3, _w, _sum3);
                    _sum4 = _mm512_fmadd_ps(_val4, _w, _sum4);
                    _sum5 = _mm512_fmadd_ps(_val5, _w, _sum5);
                    _sum6 = _mm512_fmadd_ps(_val6, _w, _sum6);
                    _sum7 = _mm512_fmadd_ps(_val7, _w, _sum7);
                    _sum8 = _mm512_fmadd_ps(_val8, _w, _sum8);
                    _sum9 = _mm512_fmadd_ps(_val9, _w, _sum9);
                    _suma = _mm512_fmadd_ps(_vala, _w, _suma);
                    _sumb = _mm512_fmadd_ps(_valb, _w, _sumb);
                    _sumc = _mm512_fmadd_ps(_valc, _w, _sumc);
                    _sumd = _mm512_fmadd_ps(_vald, _w, _sumd);
                    _sume = _mm512_fmadd_ps(_vale, _w, _sume);
                    _sumf = _mm512_fmadd_ps(_valf, _w, _sumf);

                    m += 16;
                    kptr += 16;
                }

                _sum0 = activation_avx512(_sum0, activation_type, activation_params);
                _sum1 = activation_avx512(_sum1, activation_type, activation_params);
                _sum2 = activation_avx512(_sum2, activation_type, activation_params);
                _sum3 = activation_avx512(_sum3, activation_type, activation_params);
                _sum4 = activation_avx512(_sum4, activation_type, activation_params);
                _sum5 = activation_avx512(_sum5, activation_type, activation_params);
                _sum6 = activation_avx512(_sum6, activation_type, activation_params);
                _sum7 = activation_avx512(_sum7, activation_type, activation_params);
                _sum8 = activation_avx512(_sum8, activation_type, activation_params);
                _sum9 = activation_avx512(_sum9, activation_type, activation_params);
                _suma = activation_avx512(_suma, activation_type, activation_params);
                _sumb = activation_avx512(_sumb, activation_type, activation_params);
                _sumc = activation_avx512(_sumc, activation_type, activation_params);
                _sumd = activation_avx512(_sumd, activation_type, activation_params);
                _sume = activation_avx512(_sume, activation_type, activation_params);
                _sumf = activation_avx512(_sumf, activation_type, activation_params);

                transpose16x16_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, _sum8, _sum9, _suma, _sumb, _sumc, _sumd, _sume, _sumf);

                _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_sum0));
                _mm256_storeu_si256((__m256i*)(outptr + 16), float2bfloat_avx512(_sum1));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 2), float2bfloat_avx512(_sum2));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 3), float2bfloat_avx512(_sum3));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 4), float2bfloat_avx512(_sum4));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 5), float2bfloat_avx512(_sum5));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 6), float2bfloat_avx512(_sum6));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 7), float2bfloat_avx512(_sum7));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 8), float2bfloat_avx512(_sum8));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 9), float2bfloat_avx512(_sum9));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 10), float2bfloat_avx512(_suma));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 11), float2bfloat_avx512(_sumb));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 12), float2bfloat_avx512(_sumc));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 13), float2bfloat_avx512(_sumd));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 14), float2bfloat_avx512(_sume));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 15), float2bfloat_avx512(_sumf));
                outptr += 256;
            }
        }

        if (elempack == 1 && num_output_elempack == 16)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m512 _sum = _mm512_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum = _mm512_loadu_ps(bias_data_ptr + p * 16);
                }

                int i = 0;
                for (; i < num_input; i++)
                {
                    __m512 _val = _mm512_set1_ps(bfloat16_to_float32(m[0]));
                    __m512 _w = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)kptr));

                    _sum = _mm512_fmadd_ps(_val, _w, _sum);

                    m += 1;
                    kptr += 16;
                }

                _sum = activation_avx512(_sum, activation_type, activation_params);

                _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_sum));
                outptr += 16;
            }
        }

        if (elempack == 4 && num_output_elempack == 16)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m512 _sum0 = _mm512_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm512_loadu_ps(bias_data_ptr + p * 16);
                }

                __m512 _sum1 = _sum0;
                __m512 _sum2 = _sum0;
                __m512 _sum3 = _sum0;

                int i = 0;
                for (; i < num_input; i++)
                {
                    __m512 _val0 = _mm512_set1_ps(bfloat16_to_float32(m[0]));
                    __m512 _val1 = _mm512_set1_ps(bfloat16_to_float32(m[1]));
                    __m512 _val2 = _mm512_set1_ps(bfloat16_to_float32(m[2]));
                    __m512 _val3 = _mm512_set1_ps(bfloat16_to_float32(m[3]));
                    __m512 _w = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)kptr));

                    _sum0 = _mm512_fmadd_ps(_val0, _w, _sum0);
                    _sum1 = _mm512_fmadd_ps(_val1, _w, _sum1);
                    _sum2 = _mm512_fmadd_ps(_val2, _w, _sum2);
                    _sum3 = _mm512_fmadd_ps(_val3, _w, _sum3);

                    m += 4;
                    kptr += 16;
                }

                _sum0 = activation_avx512(_sum0, activation_type, activation_params);
                _sum1 = activation_avx512(_sum1, activation_type, activation_params);
                _sum2 = activation_avx512(_sum2, activation_type, activation_params);
                _sum3 = activation_avx512(_sum3, activation_type, activation_params);

                transpose16x4_ps(_sum0, _sum1, _sum2, _sum3);

                _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_sum0));
                _mm256_storeu_si256((__m256i*)(outptr + 16), float2bfloat_avx512(_sum1));
                _mm256_storeu_si256((__m256i*)(outptr + 32), float2bfloat_avx512(_sum2));
                _mm256_storeu_si256((__m256i*)(outptr + 48), float2bfloat_avx512(_sum3));
                outptr += 64;
            }
        }

        if (elempack == 8 && num_output_elempack == 16)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m512 _sum0 = _mm512_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm512_loadu_ps(bias_data_ptr + p * 16);
                }

                __m512 _sum1 = _sum0;
                __m512 _sum2 = _sum0;
                __m512 _sum3 = _sum0;
                __m512 _sum4 = _sum0;
                __m512 _sum5 = _sum0;
                __m512 _sum6 = _sum0;
                __m512 _sum7 = _sum0;

                int i = 0;
                for (; i < num_input; i++)
                {
                    __m512 _val0 = _mm512_set1_ps(bfloat16_to_float32(m[0]));
                    __m512 _val1 = _mm512_set1_ps(bfloat16_to_float32(m[1]));
                    __m512 _val2 = _mm512_set1_ps(bfloat16_to_float32(m[2]));
                    __m512 _val3 = _mm512_set1_ps(bfloat16_to_float32(m[3]));
                    __m512 _val4 = _mm512_set1_ps(bfloat16_to_float32(m[4]));
                    __m512 _val5 = _mm512_set1_ps(bfloat16_to_float32(m[5]));
                    __m512 _val6 = _mm512_set1_ps(bfloat16_to_float32(m[6]));
                    __m512 _val7 = _mm512_set1_ps(bfloat16_to_float32(m[7]));
                    __m512 _w = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)kptr));

                    _sum0 = _mm512_fmadd_ps(_val0, _w, _sum0);
                    _sum1 = _mm512_fmadd_ps(_val1, _w, _sum1);
                    _sum2 = _mm512_fmadd_ps(_val2, _w, _sum2);
                    _sum3 = _mm512_fmadd_ps(_val3, _w, _sum3);
                    _sum4 = _mm512_fmadd_ps(_val4, _w, _sum4);
                    _sum5 = _mm512_fmadd_ps(_val5, _w, _sum5);
                    _sum6 = _mm512_fmadd_ps(_val6, _w, _sum6);
                    _sum7 = _mm512_fmadd_ps(_val7, _w, _sum7);

                    m += 8;
                    kptr += 16;
                }

                _sum0 = activation_avx512(_sum0, activation_type, activation_params);
                _sum1 = activation_avx512(_sum1, activation_type, activation_params);
                _sum2 = activation_avx512(_sum2, activation_type, activation_params);
                _sum3 = activation_avx512(_sum3, activation_type, activation_params);
                _sum4 = activation_avx512(_sum4, activation_type, activation_params);
                _sum5 = activation_avx512(_sum5, activation_type, activation_params);
                _sum6 = activation_avx512(_sum6, activation_type, activation_params);
                _sum7 = activation_avx512(_sum7, activation_type, activation_params);

                transpose16x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_sum0));
                _mm256_storeu_si256((__m256i*)(outptr + 16), float2bfloat_avx512(_sum1));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 2), float2bfloat_avx512(_sum2));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 3), float2bfloat_avx512(_sum3));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 4), float2bfloat_avx512(_sum4));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 5), float2bfloat_avx512(_sum5));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 6), float2bfloat_avx512(_sum6));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 7), float2bfloat_avx512(_sum7));
                outptr += 128;
            }
        }

        if (elempack == 16 && num_output_elempack == 1)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();
                __m512 _sum2 = _mm512_setzero_ps();
                __m512 _sum3 = _mm512_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm512_set1_ps(bias_data_ptr[p]);
                }

                int i = 0;
                for (; i + 3 < num_input; i += 4)
                {
                    __m512 _val0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)m));
                    __m512 _val1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(m + 16)));
                    __m512 _val2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(m + 32)));
                    __m512 _val3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(m + 48)));

                    __m128 _w = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)kptr));
                    __m256 _ww = combine4x2_ps(_w, _w);
                    __m512 _www = combine8x2_ps(_ww, _ww);

                    __m512 _w0 = _mm512_permute_ps(_www, _MM_SHUFFLE(0, 0, 0, 0));
                    __m512 _w1 = _mm512_permute_ps(_www, _MM_SHUFFLE(1, 1, 1, 1));
                    __m512 _w2 = _mm512_permute_ps(_www, _MM_SHUFFLE(2, 2, 2, 2));
                    __m512 _w3 = _mm512_permute_ps(_www, _MM_SHUFFLE(3, 3, 3, 3));

                    _sum0 = _mm512_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm512_fmadd_ps(_val1, _w1, _sum1);
                    _sum2 = _mm512_fmadd_ps(_val2, _w2, _sum2);
                    _sum3 = _mm512_fmadd_ps(_val3, _w3, _sum3);

                    m += 64;
                    kptr += 4;
                }
                for (; i < num_input; i++)
                {
                    __m512 _val = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)m));
                    __m512 _w = _mm512_set1_ps(bfloat16_to_float32(kptr[0]));
                    _sum0 = _mm512_fmadd_ps(_val, _w, _sum0);

                    m += 16;
                    kptr += 1;
                }

                _sum0 = _mm512_add_ps(_sum0, _sum1);
                _sum2 = _mm512_add_ps(_sum2, _sum3);
                _sum0 = _mm512_add_ps(_sum0, _sum2);

                _sum0 = activation_avx512(_sum0, activation_type, activation_params);

                _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_sum0));
                outptr += 16;
            }
        }

        if (elempack == 16 && num_output_elempack == 4)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();
                __m512 _sum2 = _mm512_setzero_ps();
                __m512 _sum3 = _mm512_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm512_set1_ps(bias_data_ptr[p * 4 + 0]);
                    _sum1 = _mm512_set1_ps(bias_data_ptr[p * 4 + 1]);
                    _sum2 = _mm512_set1_ps(bias_data_ptr[p * 4 + 2]);
                    _sum3 = _mm512_set1_ps(bias_data_ptr[p * 4 + 3]);
                }

                int i = 0;
                for (; i < num_input; i++)
                {
                    __m512 _val = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)m));

                    __m128 _w = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)kptr));
                    __m256 _ww = combine4x2_ps(_w, _w);
                    __m512 _www = combine8x2_ps(_ww, _ww);

                    __m512 _w0 = _mm512_permute_ps(_www, _MM_SHUFFLE(0, 0, 0, 0));
                    __m512 _w1 = _mm512_permute_ps(_www, _MM_SHUFFLE(1, 1, 1, 1));
                    __m512 _w2 = _mm512_permute_ps(_www, _MM_SHUFFLE(2, 2, 2, 2));
                    __m512 _w3 = _mm512_permute_ps(_www, _MM_SHUFFLE(3, 3, 3, 3));

                    _sum0 = _mm512_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm512_fmadd_ps(_val, _w1, _sum1);
                    _sum2 = _mm512_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm512_fmadd_ps(_val, _w3, _sum3);

                    m += 16;
                    kptr += 4;
                }

                _sum0 = activation_avx512(_sum0, activation_type, activation_params);
                _sum1 = activation_avx512(_sum1, activation_type, activation_params);
                _sum2 = activation_avx512(_sum2, activation_type, activation_params);
                _sum3 = activation_avx512(_sum3, activation_type, activation_params);

                _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_sum0));
                _mm256_storeu_si256((__m256i*)(outptr + 16), float2bfloat_avx512(_sum1));
                _mm256_storeu_si256((__m256i*)(outptr + 32), float2bfloat_avx512(_sum2));
                _mm256_storeu_si256((__m256i*)(outptr + 48), float2bfloat_avx512(_sum3));
                outptr += 64;
            }
        }

        if (elempack == 16 && num_output_elempack == 8)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m512 _sum0 = _mm512_setzero_ps();
                __m512 _sum1 = _mm512_setzero_ps();
                __m512 _sum2 = _mm512_setzero_ps();
                __m512 _sum3 = _mm512_setzero_ps();
                __m512 _sum4 = _mm512_setzero_ps();
                __m512 _sum5 = _mm512_setzero_ps();
                __m512 _sum6 = _mm512_setzero_ps();
                __m512 _sum7 = _mm512_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm512_set1_ps(bias_data_ptr[p * 8 + 0]);
                    _sum1 = _mm512_set1_ps(bias_data_ptr[p * 8 + 1]);
                    _sum2 = _mm512_set1_ps(bias_data_ptr[p * 8 + 2]);
                    _sum3 = _mm512_set1_ps(bias_data_ptr[p * 8 + 3]);
                    _sum4 = _mm512_set1_ps(bias_data_ptr[p * 8 + 4]);
                    _sum5 = _mm512_set1_ps(bias_data_ptr[p * 8 + 5]);
                    _sum6 = _mm512_set1_ps(bias_data_ptr[p * 8 + 6]);
                    _sum7 = _mm512_set1_ps(bias_data_ptr[p * 8 + 7]);
                }

                int i = 0;
                for (; i < num_input; i++)
                {
                    __m512 _val = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)m));

                    __m256 _w = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)kptr));
                    __m512 _ww = _mm512_castps256_ps512(_w);
                    __m512 _www0 = _mm512_shuffle_f32x4(_ww, _ww, _MM_SHUFFLE(0, 0, 0, 0));
                    __m512 _www1 = _mm512_shuffle_f32x4(_ww, _ww, _MM_SHUFFLE(1, 1, 1, 1));

                    __m512 _w0 = _mm512_permute_ps(_www0, _MM_SHUFFLE(0, 0, 0, 0));
                    __m512 _w1 = _mm512_permute_ps(_www0, _MM_SHUFFLE(1, 1, 1, 1));
                    __m512 _w2 = _mm512_permute_ps(_www0, _MM_SHUFFLE(2, 2, 2, 2));
                    __m512 _w3 = _mm512_permute_ps(_www0, _MM_SHUFFLE(3, 3, 3, 3));
                    __m512 _w4 = _mm512_permute_ps(_www1, _MM_SHUFFLE(0, 0, 0, 0));
                    __m512 _w5 = _mm512_permute_ps(_www1, _MM_SHUFFLE(1, 1, 1, 1));
                    __m512 _w6 = _mm512_permute_ps(_www1, _MM_SHUFFLE(2, 2, 2, 2));
                    __m512 _w7 = _mm512_permute_ps(_www1, _MM_SHUFFLE(3, 3, 3, 3));

                    _sum0 = _mm512_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm512_fmadd_ps(_val, _w1, _sum1);
                    _sum2 = _mm512_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm512_fmadd_ps(_val, _w3, _sum3);
                    _sum4 = _mm512_fmadd_ps(_val, _w4, _sum4);
                    _sum5 = _mm512_fmadd_ps(_val, _w5, _sum5);
                    _sum6 = _mm512_fmadd_ps(_val, _w6, _sum6);
                    _sum7 = _mm512_fmadd_ps(_val, _w7, _sum7);

                    m += 16;
                    kptr += 8;
                }

                _sum0 = activation_avx512(_sum0, activation_type, activation_params);
                _sum1 = activation_avx512(_sum1, activation_type, activation_params);
                _sum2 = activation_avx512(_sum2, activation_type, activation_params);
                _sum3 = activation_avx512(_sum3, activation_type, activation_params);
                _sum4 = activation_avx512(_sum4, activation_type, activation_params);
                _sum5 = activation_avx512(_sum5, activation_type, activation_params);
                _sum6 = activation_avx512(_sum6, activation_type, activation_params);
                _sum7 = activation_avx512(_sum7, activation_type, activation_params);

                _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_sum0));
                _mm256_storeu_si256((__m256i*)(outptr + 16), float2bfloat_avx512(_sum1));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 2), float2bfloat_avx512(_sum2));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 3), float2bfloat_avx512(_sum3));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 4), float2bfloat_avx512(_sum4));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 5), float2bfloat_avx512(_sum5));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 6), float2bfloat_avx512(_sum6));
                _mm256_storeu_si256((__m256i*)(outptr + 16 * 7), float2bfloat_avx512(_sum7));
                outptr += 128;
            }
        }

#endif // __AVX512F__

        if (elempack == 8 && num_output_elempack == 8)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m256 _sum0 = _mm256_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm256_loadu_ps(bias_data_ptr + p * 8);
                }

                __m256 _sum1 = _sum0;
                __m256 _sum2 = _sum0;
                __m256 _sum3 = _sum0;
                __m256 _sum4 = _sum0;
                __m256 _sum5 = _sum0;
                __m256 _sum6 = _sum0;
                __m256 _sum7 = _sum0;

                for (int i = 0; i < num_input; i++)
                {
                    __m256 _val0 = _mm256_set1_ps(bfloat16_to_float32(m[0]));
                    __m256 _val1 = _mm256_set1_ps(bfloat16_to_float32(m[1]));
                    __m256 _val2 = _mm256_set1_ps(bfloat16_to_float32(m[2]));
                    __m256 _val3 = _mm256_set1_ps(bfloat16_to_float32(m[3]));
                    __m256 _val4 = _mm256_set1_ps(bfloat16_to_float32(m[4]));
                    __m256 _val5 = _mm256_set1_ps(bfloat16_to_float32(m[5]));
                    __m256 _val6 = _mm256_set1_ps(bfloat16_to_float32(m[6]));
                    __m256 _val7 = _mm256_set1_ps(bfloat16_to_float32(m[7]));
                    __m256 _w = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)kptr));

                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val1, _w, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_val2, _w, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val3, _w, _sum3);
                    _sum4 = _mm256_comp_fmadd_ps(_val4, _w, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_val5, _w, _sum5);
                    _sum6 = _mm256_comp_fmadd_ps(_val6, _w, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_val7, _w, _sum7);

                    m += 8;
                    kptr += 8;
                }

                _sum0 = activation_avx(_sum0, activation_type, activation_params);
                _sum1 = activation_avx(_sum1, activation_type, activation_params);
                _sum2 = activation_avx(_sum2, activation_type, activation_params);
                _sum3 = activation_avx(_sum3, activation_type, activation_params);
                _sum4 = activation_avx(_sum4, activation_type, activation_params);
                _sum5 = activation_avx(_sum5, activation_type, activation_params);
                _sum6 = activation_avx(_sum6, activation_type, activation_params);
                _sum7 = activation_avx(_sum7, activation_type, activation_params);

                transpose8x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                _mm_storeu_si128((__m128i*)outptr, float2bfloat_avx(_sum0));
                _mm_storeu_si128((__m128i*)(outptr + 8), float2bfloat_avx(_sum1));
                _mm_storeu_si128((__m128i*)(outptr + 16), float2bfloat_avx(_sum2));
                _mm_storeu_si128((__m128i*)(outptr + 24), float2bfloat_avx(_sum3));
                _mm_storeu_si128((__m128i*)(outptr + 32), float2bfloat_avx(_sum4));
                _mm_storeu_si128((__m128i*)(outptr + 40), float2bfloat_avx(_sum5));
                _mm_storeu_si128((__m128i*)(outptr + 48), float2bfloat_avx(_sum6));
                _mm_storeu_si128((__m128i*)(outptr + 56), float2bfloat_avx(_sum7));
                outptr += 64;
            }
        }

        if (elempack == 1 && num_output_elempack == 8)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();
                __m256 _sum2 = _mm256_setzero_ps();
                __m256 _sum3 = _mm256_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm256_loadu_ps(bias_data_ptr + p * 8);
                }

                int i = 0;
                for (; i + 7 < num_input; i += 8)
                {
                    __m256 _val0 = _mm256_set1_ps(bfloat16_to_float32(m[0]));
                    __m256 _val1 = _mm256_set1_ps(bfloat16_to_float32(m[1]));
                    __m256 _val2 = _mm256_set1_ps(bfloat16_to_float32(m[2]));
                    __m256 _val3 = _mm256_set1_ps(bfloat16_to_float32(m[3]));
                    __m256i _w01 = _mm256_lddqu_si256((const __m256i*)kptr);
                    __m256i _w23 = _mm256_lddqu_si256((const __m256i*)(kptr + 16));
                    __m256 _w0 = bfloat2float_avx(_mm256_extractf128_si256(_w01, 0));
                    __m256 _w1 = bfloat2float_avx(_mm256_extractf128_si256(_w01, 1));
                    __m256 _w2 = bfloat2float_avx(_mm256_extractf128_si256(_w23, 0));
                    __m256 _w3 = bfloat2float_avx(_mm256_extractf128_si256(_w23, 1));

                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val1, _w1, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_val2, _w2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val3, _w3, _sum3);

                    __m256 _val4 = _mm256_set1_ps(bfloat16_to_float32(m[4]));
                    __m256 _val5 = _mm256_set1_ps(bfloat16_to_float32(m[5]));
                    __m256 _val6 = _mm256_set1_ps(bfloat16_to_float32(m[6]));
                    __m256 _val7 = _mm256_set1_ps(bfloat16_to_float32(m[7]));
                    __m256i _w45 = _mm256_lddqu_si256((const __m256i*)(kptr + 32));
                    __m256i _w67 = _mm256_lddqu_si256((const __m256i*)(kptr + 48));
                    __m256 _w4 = bfloat2float_avx(_mm256_extractf128_si256(_w45, 0));
                    __m256 _w5 = bfloat2float_avx(_mm256_extractf128_si256(_w45, 1));
                    __m256 _w6 = bfloat2float_avx(_mm256_extractf128_si256(_w67, 0));
                    __m256 _w7 = bfloat2float_avx(_mm256_extractf128_si256(_w67, 1));

                    _sum0 = _mm256_comp_fmadd_ps(_val4, _w4, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val5, _w5, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_val6, _w6, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val7, _w7, _sum3);

                    m += 8;
                    kptr += 64;
                }
                for (; i + 3 < num_input; i += 4)
                {
                    __m256 _val0 = _mm256_set1_ps(bfloat16_to_float32(m[0]));
                    __m256 _val1 = _mm256_set1_ps(bfloat16_to_float32(m[1]));
                    __m256 _val2 = _mm256_set1_ps(bfloat16_to_float32(m[2]));
                    __m256 _val3 = _mm256_set1_ps(bfloat16_to_float32(m[3]));
                    __m256i _w01 = _mm256_lddqu_si256((const __m256i*)kptr);
                    __m256i _w23 = _mm256_lddqu_si256((const __m256i*)(kptr + 16));
                    __m256 _w0 = bfloat2float_avx(_mm256_extractf128_si256(_w01, 0));
                    __m256 _w1 = bfloat2float_avx(_mm256_extractf128_si256(_w01, 1));
                    __m256 _w2 = bfloat2float_avx(_mm256_extractf128_si256(_w23, 0));
                    __m256 _w3 = bfloat2float_avx(_mm256_extractf128_si256(_w23, 1));

                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val1, _w1, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_val2, _w2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val3, _w3, _sum3);

                    m += 4;
                    kptr += 32;
                }
                for (; i < num_input; i++)
                {
                    __m256 _val = _mm256_set1_ps(bfloat16_to_float32(m[0]));
                    __m256 _w = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)kptr));
                    _sum0 = _mm256_comp_fmadd_ps(_val, _w, _sum0);

                    m += 1;
                    kptr += 8;
                }

                _sum0 = _mm256_add_ps(_sum0, _sum1);
                _sum2 = _mm256_add_ps(_sum2, _sum3);
                _sum0 = _mm256_add_ps(_sum0, _sum2);

                _sum0 = activation_avx(_sum0, activation_type, activation_params);

                _mm_storeu_si128((__m128i*)outptr, float2bfloat_avx(_sum0));
                outptr += 8;
            }
        }

        if (elempack == 4 && num_output_elempack == 8)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m256 _sum0 = _mm256_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm256_loadu_ps(bias_data_ptr + p * 8);
                }

                __m256 _sum1 = _sum0;
                __m256 _sum2 = _sum0;
                __m256 _sum3 = _sum0;

                int i = 0;
                for (; i < num_input; i++)
                {
                    __m256 _val0 = _mm256_set1_ps(bfloat16_to_float32(m[0]));
                    __m256 _val1 = _mm256_set1_ps(bfloat16_to_float32(m[1]));
                    __m256 _val2 = _mm256_set1_ps(bfloat16_to_float32(m[2]));
                    __m256 _val3 = _mm256_set1_ps(bfloat16_to_float32(m[3]));
                    __m256 _w = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)kptr));

                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val1, _w, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_val2, _w, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val3, _w, _sum3);

                    m += 4;
                    kptr += 8;
                }

                _sum0 = activation_avx(_sum0, activation_type, activation_params);
                _sum1 = activation_avx(_sum1, activation_type, activation_params);
                _sum2 = activation_avx(_sum2, activation_type, activation_params);
                _sum3 = activation_avx(_sum3, activation_type, activation_params);

                transpose8x4_ps(_sum0, _sum1, _sum2, _sum3);

                _mm_storeu_si128((__m128i*)outptr, float2bfloat_avx(_sum0));
                _mm_storeu_si128((__m128i*)(outptr + 8), float2bfloat_avx(_sum1));
                _mm_storeu_si128((__m128i*)(outptr + 16), float2bfloat_avx(_sum2));
                _mm_storeu_si128((__m128i*)(outptr + 24), float2bfloat_avx(_sum3));
                outptr += 32;
            }
        }

        if (elempack == 8 && num_output_elempack == 1)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();
                __m256 _sum2 = _mm256_setzero_ps();
                __m256 _sum3 = _mm256_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm256_set1_ps(bias_data_ptr[p]);
                }

                int i = 0;
                for (; i + 3 < num_input; i += 4)
                {
                    __m256 _val0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)m));
                    __m256 _val1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(m + 8)));
                    __m256 _val2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(m + 16)));
                    __m256 _val3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(m + 24)));

                    __m128 _w = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)kptr));
                    __m256 _ww = combine4x2_ps(_w, _w);

                    __m256 _w0 = _mm256_permute_ps(_ww, _MM_SHUFFLE(0, 0, 0, 0));
                    __m256 _w1 = _mm256_permute_ps(_ww, _MM_SHUFFLE(1, 1, 1, 1));
                    __m256 _w2 = _mm256_permute_ps(_ww, _MM_SHUFFLE(2, 2, 2, 2));
                    __m256 _w3 = _mm256_permute_ps(_ww, _MM_SHUFFLE(3, 3, 3, 3));

                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val1, _w1, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_val2, _w2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val3, _w3, _sum3);

                    m += 32;
                    kptr += 4;
                }
                for (; i < num_input; i++)
                {
                    __m256 _val = bfloat2float_avx(_mm_loadu_si128((const __m128i*)m));
                    __m256 _w = _mm256_set1_ps(bfloat16_to_float32(kptr[0]));
                    _sum0 = _mm256_comp_fmadd_ps(_val, _w, _sum0);

                    m += 8;
                    kptr += 1;
                }

                _sum0 = _mm256_add_ps(_sum0, _sum1);
                _sum2 = _mm256_add_ps(_sum2, _sum3);
                _sum0 = _mm256_add_ps(_sum0, _sum2);

                _sum0 = activation_avx(_sum0, activation_type, activation_params);

                _mm_storeu_si128((__m128i*)outptr, float2bfloat_avx(_sum0));
                outptr += 8;
            }
        }

        if (elempack == 8 && num_output_elempack == 4)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m256 _sum0 = _mm256_setzero_ps();
                __m256 _sum1 = _mm256_setzero_ps();
                __m256 _sum2 = _mm256_setzero_ps();
                __m256 _sum3 = _mm256_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm256_set1_ps(bias_data_ptr[p * 4 + 0]);
                    _sum1 = _mm256_set1_ps(bias_data_ptr[p * 4 + 1]);
                    _sum2 = _mm256_set1_ps(bias_data_ptr[p * 4 + 2]);
                    _sum3 = _mm256_set1_ps(bias_data_ptr[p * 4 + 3]);
                }

                int i = 0;
                for (; i < num_input; i++)
                {
                    __m256 _val = bfloat2float_avx(_mm_loadu_si128((const __m128i*)m));

                    __m128 _w = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)kptr));
                    __m256 _ww = combine4x2_ps(_w, _w);

                    __m256 _w0 = _mm256_permute_ps(_ww, _MM_SHUFFLE(0, 0, 0, 0));
                    __m256 _w1 = _mm256_permute_ps(_ww, _MM_SHUFFLE(1, 1, 1, 1));
                    __m256 _w2 = _mm256_permute_ps(_ww, _MM_SHUFFLE(2, 2, 2, 2));
                    __m256 _w3 = _mm256_permute_ps(_ww, _MM_SHUFFLE(3, 3, 3, 3));

                    _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                    m += 8;
                    kptr += 4;
                }

                _sum0 = activation_avx(_sum0, activation_type, activation_params);
                _sum1 = activation_avx(_sum1, activation_type, activation_params);
                _sum2 = activation_avx(_sum2, activation_type, activation_params);
                _sum3 = activation_avx(_sum3, activation_type, activation_params);

                _mm_storeu_si128((__m128i*)outptr, float2bfloat_avx(_sum0));
                _mm_storeu_si128((__m128i*)(outptr + 8), float2bfloat_avx(_sum1));
                _mm_storeu_si128((__m128i*)(outptr + 16), float2bfloat_avx(_sum2));
                _mm_storeu_si128((__m128i*)(outptr + 24), float2bfloat_avx(_sum3));
                outptr += 32;
            }
        }
#endif // __AVX__

        if (elempack == 4 && num_output_elempack == 4)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m128 _sum0 = _mm_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm_loadu_ps(bias_data_ptr + p * 4);
                }

                __m128 _sum1 = _sum0;
                __m128 _sum2 = _sum0;
                __m128 _sum3 = _sum0;

                int i = 0;
                for (; i < num_input; i++)
                {
                    __m128 _val0 = _mm_set1_ps(bfloat16_to_float32(m[0]));
                    __m128 _val1 = _mm_set1_ps(bfloat16_to_float32(m[1]));
                    __m128 _val2 = _mm_set1_ps(bfloat16_to_float32(m[2]));
                    __m128 _val3 = _mm_set1_ps(bfloat16_to_float32(m[3]));
                    __m128 _w = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)kptr));
                    _sum0 = _mm_comp_fmadd_ps(_val0, _w, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val1, _w, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_val2, _w, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val3, _w, _sum3);

                    m += 4;
                    kptr += 4;
                }

                _sum0 = activation_sse(_sum0, activation_type, activation_params);
                _sum1 = activation_sse(_sum1, activation_type, activation_params);
                _sum2 = activation_sse(_sum2, activation_type, activation_params);
                _sum3 = activation_sse(_sum3, activation_type, activation_params);

                _MM_TRANSPOSE4_PS(_sum0, _sum1, _sum2, _sum3);

                __m128i _bf16_01 = float2bfloat_sse(_sum0, _sum1);
                __m128i _bf16_23 = float2bfloat_sse(_sum2, _sum3);
                _mm_storeu_si128((__m128i*)outptr, _bf16_01);
                _mm_storeu_si128((__m128i*)(outptr + 8), _bf16_23);
                outptr += 16;
            }
        }

        if (elempack == 1 && num_output_elempack == 4)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output / num_output_elempack; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m128 _sum = _mm_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum = _mm_loadu_ps(bias_data_ptr + p * 4);
                }

                int i = 0;
                for (; i < num_input; i++)
                {
                    __m128 _val = _mm_set1_ps(bfloat16_to_float32(m[0]));
                    __m128 _w = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)kptr));
                    _sum = _mm_comp_fmadd_ps(_val, _w, _sum);

                    m += 1;
                    kptr += 4;
                }

                _sum = activation_sse(_sum, activation_type, activation_params);

                _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_sum));
                outptr += 4;
            }
        }

        if (elempack == 4 && num_output_elempack == 1)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                __m128 _sum0 = _mm_setzero_ps();
                __m128 _sum1 = _mm_setzero_ps();
                __m128 _sum2 = _mm_setzero_ps();
                __m128 _sum3 = _mm_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum0 = _mm_set1_ps(bias_data_ptr[p]);
                }

                int i = 0;
                for (; i + 3 < num_input; i += 4)
                {
                    __m128 _val0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)m));
                    __m128 _val1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(m + 4)));
                    __m128 _val2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(m + 8)));
                    __m128 _val3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(m + 12)));

                    __m128 _w = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)kptr));

                    __m128 _w0 = _mm_shuffle_ps(_w, _w, _MM_SHUFFLE(0, 0, 0, 0));
                    __m128 _w1 = _mm_shuffle_ps(_w, _w, _MM_SHUFFLE(1, 1, 1, 1));
                    __m128 _w2 = _mm_shuffle_ps(_w, _w, _MM_SHUFFLE(2, 2, 2, 2));
                    __m128 _w3 = _mm_shuffle_ps(_w, _w, _MM_SHUFFLE(3, 3, 3, 3));

                    _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val1, _w1, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_val2, _w2, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val3, _w3, _sum3);

                    m += 16;
                    kptr += 4;
                }
                for (; i < num_input; i++)
                {
                    __m128 _val = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)m));
                    __m128 _w = _mm_set1_ps(bfloat16_to_float32(kptr[0]));
                    _sum0 = _mm_comp_fmadd_ps(_val, _w, _sum0);

                    m += 4;
                    kptr += 1;
                }

                _sum0 = _mm_add_ps(_sum0, _sum1);
                _sum2 = _mm_add_ps(_sum2, _sum3);
                _sum0 = _mm_add_ps(_sum0, _sum2);

                _sum0 = activation_sse(_sum0, activation_type, activation_params);

                _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_sum0));
                outptr += 4;
            }
        }
#endif // __SSE2__

        if (elempack == 1 && num_output_elempack == 1)
        {
            unsigned short* outptr = top_blob.row<unsigned short>(j);

            for (int p = 0; p < num_output; p++)
            {
                const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                const unsigned short* m = bottom_blob.row<const unsigned short>(j);

                float sum = 0.f;

                if (bias_data_ptr)
                {
                    sum = bias_data_ptr[p];
                }

                int i = 0;
#if __SSE2__
#if __AVX__
                __m256 _sum = _mm256_setzero_ps();
                for (; i + 7 < num_input; i += 8)
                {
                    __m256 _m = bfloat2float_avx(_mm_loadu_si128((const __m128i*)m));
                    __m256 _w = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)kptr));
                    _sum = _mm256_comp_fmadd_ps(_m, _w, _sum);

                    m += 8;
                    kptr += 8;
                }
#endif // __AVX__
                __m128 _suml = _mm_setzero_ps();
                for (; i + 3 < num_input; i += 4)
                {
                    __m128 _val = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)m));
                    __m128 _w = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)kptr));
                    _suml = _mm_comp_fmadd_ps(_val, _w, _suml);

                    m += 4;
                    kptr += 4;
                }
#endif // __SSE2__
                for (; i < num_input; i++)
                {
                    sum += bfloat16_to_float32(*m++) * bfloat16_to_float32(*kptr++);
                }

#if __SSE2__
#if __AVX__
                _suml = _mm_add_ps(_suml, _mm256_extractf128_ps(_sum, 1));
                _suml = _mm_add_ps(_suml, _mm256_castps256_ps128(_sum));
#endif // __AVX__
                sum += _mm_reduce_add_ps(_suml);
#endif // __SSE2__

                sum = activation_ss(sum, activation_type, activation_params);

                outptr[0] = float32_to_bfloat16(sum);
                outptr += 1;
            }
        }
    }
#endif // NCNN_RUNTIME_CPU
}
