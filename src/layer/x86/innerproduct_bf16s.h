// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void innerproduct_bf16s_sse_avx512bf16(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
void innerproduct_transform_kernel_bf16s_sse_avx512bf16(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt);
#endif

static void innerproduct_bf16s_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        innerproduct_bf16s_sse_avx512bf16(bottom_blob, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);
        return;
    }
#else // NCNN_RUNTIME_CPU

    const int num_input = bottom_blob.w * bottom_blob.elempack;
    const int outw = top_blob.w;
    const int out_elempack = top_blob.elempack;

    const float* bias_data_ptr = bias_data;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (out_elempack == 16)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outw; p++)
        {
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
                _sum0 = _mm512_loadu_ps(bias_data_ptr + p * 16);
            }

            const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* sptr = bottom_blob;

            int i = 0;
            for (; i + 7 < num_input; i += 8)
            {
                __m512 _val0 = _mm512_set1_ps(bfloat16_to_float32(sptr[0]));
                __m512 _val1 = _mm512_set1_ps(bfloat16_to_float32(sptr[1]));
                __m512 _val2 = _mm512_set1_ps(bfloat16_to_float32(sptr[2]));
                __m512 _val3 = _mm512_set1_ps(bfloat16_to_float32(sptr[3]));

                __m512 _w0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)kptr));
                __m512 _w1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(kptr + 16)));
                __m512 _w2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(kptr + 32)));
                __m512 _w3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(kptr + 48)));

                _sum0 = _mm512_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm512_fmadd_ps(_val1, _w1, _sum1);
                _sum2 = _mm512_fmadd_ps(_val2, _w2, _sum2);
                _sum3 = _mm512_fmadd_ps(_val3, _w3, _sum3);

                __m512 _val4 = _mm512_set1_ps(bfloat16_to_float32(sptr[4]));
                __m512 _val5 = _mm512_set1_ps(bfloat16_to_float32(sptr[5]));
                __m512 _val6 = _mm512_set1_ps(bfloat16_to_float32(sptr[6]));
                __m512 _val7 = _mm512_set1_ps(bfloat16_to_float32(sptr[7]));

                __m512 _w4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(kptr + 64)));
                __m512 _w5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(kptr + 80)));
                __m512 _w6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(kptr + 96)));
                __m512 _w7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(kptr + 112)));

                _sum4 = _mm512_fmadd_ps(_val4, _w4, _sum4);
                _sum5 = _mm512_fmadd_ps(_val5, _w5, _sum5);
                _sum6 = _mm512_fmadd_ps(_val6, _w6, _sum6);
                _sum7 = _mm512_fmadd_ps(_val7, _w7, _sum7);

                sptr += 8;
                kptr += 128;
            }
            for (; i + 3 < num_input; i += 4)
            {
                __m512 _val0 = _mm512_set1_ps(bfloat16_to_float32(sptr[0]));
                __m512 _val1 = _mm512_set1_ps(bfloat16_to_float32(sptr[1]));
                __m512 _val2 = _mm512_set1_ps(bfloat16_to_float32(sptr[2]));
                __m512 _val3 = _mm512_set1_ps(bfloat16_to_float32(sptr[3]));

                __m512 _w0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)kptr));
                __m512 _w1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(kptr + 16)));
                __m512 _w2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(kptr + 32)));
                __m512 _w3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(kptr + 48)));

                _sum0 = _mm512_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm512_fmadd_ps(_val1, _w1, _sum1);
                _sum2 = _mm512_fmadd_ps(_val2, _w2, _sum2);
                _sum3 = _mm512_fmadd_ps(_val3, _w3, _sum3);

                sptr += 4;
                kptr += 64;
            }
            for (; i < num_input; i++)
            {
                __m512 _val = _mm512_set1_ps(bfloat16_to_float32(sptr[0]));
                __m512 _w = bfloat2float_avx512(_mm256_lddqu_si256((const __m256i*)kptr));
                _sum0 = _mm512_fmadd_ps(_val, _w, _sum0);

                sptr += 1;
                kptr += 16;
            }

            _sum0 = _mm512_add_ps(_sum0, _sum1);
            _sum2 = _mm512_add_ps(_sum2, _sum3);
            _sum4 = _mm512_add_ps(_sum4, _sum5);
            _sum6 = _mm512_add_ps(_sum6, _sum7);
            _sum0 = _mm512_add_ps(_sum0, _sum2);
            _sum4 = _mm512_add_ps(_sum4, _sum6);
            _sum0 = _mm512_add_ps(_sum0, _sum4);

            _sum0 = activation_avx512(_sum0, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            _mm256_storeu_si256((__m256i*)(outptr + p * 16), float2bfloat_avx512(_sum0));
        }
    }
#endif // __AVX512F__

    if (out_elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outw; p++)
        {
            __m256 _sum0 = _mm256_setzero_ps();
            __m256 _sum1 = _mm256_setzero_ps();
            __m256 _sum2 = _mm256_setzero_ps();
            __m256 _sum3 = _mm256_setzero_ps();
            __m256 _sum4 = _mm256_setzero_ps();
            __m256 _sum5 = _mm256_setzero_ps();
            __m256 _sum6 = _mm256_setzero_ps();
            __m256 _sum7 = _mm256_setzero_ps();

            if (bias_data_ptr)
            {
                _sum0 = _mm256_loadu_ps(bias_data_ptr + p * 8);
            }

            const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* sptr = bottom_blob;

            int i = 0;
            for (; i + 7 < num_input; i += 8)
            {
                __m256 _val0 = _mm256_set1_ps(bfloat16_to_float32(sptr[0]));
                __m256 _val1 = _mm256_set1_ps(bfloat16_to_float32(sptr[1]));
                __m256 _val2 = _mm256_set1_ps(bfloat16_to_float32(sptr[2]));
                __m256 _val3 = _mm256_set1_ps(bfloat16_to_float32(sptr[3]));

                __m256 _w0 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)kptr));
                __m256 _w1 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)(kptr + 8)));
                __m256 _w2 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)(kptr + 16)));
                __m256 _w3 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)(kptr + 24)));

                _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_val1, _w1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_val2, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_val3, _w3, _sum3);

                __m256 _val4 = _mm256_set1_ps(bfloat16_to_float32(sptr[4]));
                __m256 _val5 = _mm256_set1_ps(bfloat16_to_float32(sptr[5]));
                __m256 _val6 = _mm256_set1_ps(bfloat16_to_float32(sptr[6]));
                __m256 _val7 = _mm256_set1_ps(bfloat16_to_float32(sptr[7]));

                __m256 _w4 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)(kptr + 32)));
                __m256 _w5 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)(kptr + 40)));
                __m256 _w6 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)(kptr + 48)));
                __m256 _w7 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)(kptr + 56)));

                _sum4 = _mm256_comp_fmadd_ps(_val4, _w4, _sum4);
                _sum5 = _mm256_comp_fmadd_ps(_val5, _w5, _sum5);
                _sum6 = _mm256_comp_fmadd_ps(_val6, _w6, _sum6);
                _sum7 = _mm256_comp_fmadd_ps(_val7, _w7, _sum7);

                sptr += 8;
                kptr += 64;
            }
            for (; i + 3 < num_input; i += 4)
            {
                __m256 _val0 = _mm256_set1_ps(bfloat16_to_float32(sptr[0]));
                __m256 _val1 = _mm256_set1_ps(bfloat16_to_float32(sptr[1]));
                __m256 _val2 = _mm256_set1_ps(bfloat16_to_float32(sptr[2]));
                __m256 _val3 = _mm256_set1_ps(bfloat16_to_float32(sptr[3]));

                __m256 _w0 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)kptr));
                __m256 _w1 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)(kptr + 8)));
                __m256 _w2 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)(kptr + 16)));
                __m256 _w3 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)(kptr + 24)));

                _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_val1, _w1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_val2, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_val3, _w3, _sum3);

                sptr += 4;
                kptr += 32;
            }
            for (; i < num_input; i++)
            {
                __m256 _val = _mm256_set1_ps(bfloat16_to_float32(sptr[0]));
                __m256 _w = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)kptr));
                _sum0 = _mm256_comp_fmadd_ps(_val, _w, _sum0);

                sptr += 1;
                kptr += 8;
            }

            _sum0 = _mm256_add_ps(_sum0, _sum1);
            _sum2 = _mm256_add_ps(_sum2, _sum3);
            _sum4 = _mm256_add_ps(_sum4, _sum5);
            _sum6 = _mm256_add_ps(_sum6, _sum7);
            _sum0 = _mm256_add_ps(_sum0, _sum2);
            _sum4 = _mm256_add_ps(_sum4, _sum6);
            _sum0 = _mm256_add_ps(_sum0, _sum4);

            _sum0 = activation_avx(_sum0, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            _mm_storeu_si128((__m128i*)(outptr + p * 8), float2bfloat_avx(_sum0));
        }
    }
#endif // __AVX__

    if (out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outw; p++)
        {
            __m128 _sum0 = _mm_setzero_ps();
#if __AVX__
            __m256 _sum01 = _mm256_setzero_ps();
            __m256 _sum23 = _mm256_setzero_ps();
            __m256 _sum45 = _mm256_setzero_ps();
            __m256 _sum67 = _mm256_setzero_ps();
#else
            __m128 _sum1 = _mm_setzero_ps();
            __m128 _sum2 = _mm_setzero_ps();
            __m128 _sum3 = _mm_setzero_ps();
#endif

            if (bias_data_ptr)
            {
                _sum0 = _mm_loadu_ps(bias_data_ptr + p * 4);
            }

            const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* sptr = bottom_blob;

            int i = 0;
#if __AVX__
            for (; i + 7 < num_input; i += 8)
            {
                __m128 _val0 = _mm_set1_ps(bfloat16_to_float32(sptr[0]));
                __m128 _val1 = _mm_set1_ps(bfloat16_to_float32(sptr[1]));
                __m128 _val2 = _mm_set1_ps(bfloat16_to_float32(sptr[2]));
                __m128 _val3 = _mm_set1_ps(bfloat16_to_float32(sptr[3]));
                __m128 _val4 = _mm_set1_ps(bfloat16_to_float32(sptr[4]));
                __m128 _val5 = _mm_set1_ps(bfloat16_to_float32(sptr[5]));
                __m128 _val6 = _mm_set1_ps(bfloat16_to_float32(sptr[6]));
                __m128 _val7 = _mm_set1_ps(bfloat16_to_float32(sptr[7]));

                __m256 _val01 = combine4x2_ps(_val0, _val1);
                __m256 _val23 = combine4x2_ps(_val2, _val3);
                __m256 _val45 = combine4x2_ps(_val4, _val5);
                __m256 _val67 = combine4x2_ps(_val6, _val7);

                __m256 _w01 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)kptr));
                __m256 _w23 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)(kptr + 8)));
                __m256 _w45 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)(kptr + 16)));
                __m256 _w67 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)(kptr + 24)));

                _sum01 = _mm256_comp_fmadd_ps(_val01, _w01, _sum01);
                _sum23 = _mm256_comp_fmadd_ps(_val23, _w23, _sum23);
                _sum45 = _mm256_comp_fmadd_ps(_val45, _w45, _sum45);
                _sum67 = _mm256_comp_fmadd_ps(_val67, _w67, _sum67);

                sptr += 8;
                kptr += 32;
            }
#endif
            for (; i + 3 < num_input; i += 4)
            {
#if __AVX__
                __m128 _val0 = _mm_set1_ps(bfloat16_to_float32(sptr[0]));
                __m128 _val1 = _mm_set1_ps(bfloat16_to_float32(sptr[1]));
                __m128 _val2 = _mm_set1_ps(bfloat16_to_float32(sptr[2]));
                __m128 _val3 = _mm_set1_ps(bfloat16_to_float32(sptr[3]));

                __m256 _val01 = combine4x2_ps(_val0, _val1);
                __m256 _val23 = combine4x2_ps(_val2, _val3);

                __m256 _w01 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)kptr));
                __m256 _w23 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)(kptr + 8)));

                _sum01 = _mm256_comp_fmadd_ps(_val01, _w01, _sum01);
                _sum23 = _mm256_comp_fmadd_ps(_val23, _w23, _sum23);
#else
                __m128 _val0 = _mm_set1_ps(bfloat16_to_float32(sptr[0]));
                __m128 _val1 = _mm_set1_ps(bfloat16_to_float32(sptr[1]));
                __m128 _val2 = _mm_set1_ps(bfloat16_to_float32(sptr[2]));
                __m128 _val3 = _mm_set1_ps(bfloat16_to_float32(sptr[3]));

                __m128 _w0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)kptr));
                __m128 _w1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(kptr + 4)));
                __m128 _w2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(kptr + 8)));
                __m128 _w3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(kptr + 12)));

                _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm_comp_fmadd_ps(_val1, _w1, _sum1);
                _sum2 = _mm_comp_fmadd_ps(_val2, _w2, _sum2);
                _sum3 = _mm_comp_fmadd_ps(_val3, _w3, _sum3);
#endif

                sptr += 4;
                kptr += 16;
            }
            for (; i < num_input; i++)
            {
                __m128 _val = _mm_set1_ps(bfloat16_to_float32(sptr[0]));
                __m128 _w = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)kptr));
                _sum0 = _mm_comp_fmadd_ps(_val, _w, _sum0);

                sptr += 1;
                kptr += 4;
            }

#if __AVX__
            _sum01 = _mm256_add_ps(_sum01, _sum23);
            _sum45 = _mm256_add_ps(_sum45, _sum67);
            _sum01 = _mm256_add_ps(_sum01, _sum45);

            _sum0 = _mm_add_ps(_sum0, _mm256_extractf128_ps(_sum01, 0));
            _sum0 = _mm_add_ps(_sum0, _mm256_extractf128_ps(_sum01, 1));
#else
            _sum0 = _mm_add_ps(_sum0, _sum1);
            _sum2 = _mm_add_ps(_sum2, _sum3);
            _sum0 = _mm_add_ps(_sum0, _sum2);
#endif

            _sum0 = activation_sse(_sum0, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            _mm_storel_epi64((__m128i*)(outptr + p * 4), float2bfloat_sse(_sum0, _mm_setzero_ps()));
        }
    }
#endif // __SSE2__

    if (out_elempack == 1)
    {
#if __SSE2__
#if __AVX__
        int remain_outw_start = 0;
        int nn_outw = outw >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outw; pp++)
        {
            int p = pp * 8;

            float sums[8] = {0.0f};
            if (bias_data_ptr)
            {
                sums[0] = bias_data_ptr[p];
                sums[1] = bias_data_ptr[p + 1];
                sums[2] = bias_data_ptr[p + 2];
                sums[3] = bias_data_ptr[p + 3];
                sums[4] = bias_data_ptr[p + 4];
                sums[5] = bias_data_ptr[p + 5];
                sums[6] = bias_data_ptr[p + 6];
                sums[7] = bias_data_ptr[p + 7];
            }

            const unsigned short* w0 = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* w1 = weight_data_tm.row<const unsigned short>(p + 1);
            const unsigned short* w2 = weight_data_tm.row<const unsigned short>(p + 2);
            const unsigned short* w3 = weight_data_tm.row<const unsigned short>(p + 3);
            const unsigned short* w4 = weight_data_tm.row<const unsigned short>(p + 4);
            const unsigned short* w5 = weight_data_tm.row<const unsigned short>(p + 5);
            const unsigned short* w6 = weight_data_tm.row<const unsigned short>(p + 6);
            const unsigned short* w7 = weight_data_tm.row<const unsigned short>(p + 7);
            const unsigned short* m = bottom_blob;

            __m256 _sum0 = _mm256_setzero_ps();
            __m256 _sum1 = _mm256_setzero_ps();
            __m256 _sum2 = _mm256_setzero_ps();
            __m256 _sum3 = _mm256_setzero_ps();
            __m256 _sum4 = _mm256_setzero_ps();
            __m256 _sum5 = _mm256_setzero_ps();
            __m256 _sum6 = _mm256_setzero_ps();
            __m256 _sum7 = _mm256_setzero_ps();

            int i = 0;
            for (; i + 7 < num_input; i += 8)
            {
                __m256 _m = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)m));

                __m256 _w0 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)w0));
                __m256 _w1 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)w1));
                __m256 _w2 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)w2));
                __m256 _w3 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)w3));

                _sum0 = _mm256_comp_fmadd_ps(_m, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_m, _w1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_m, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_m, _w3, _sum3);

                __m256 _w4 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)w4));
                __m256 _w5 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)w5));
                __m256 _w6 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)w6));
                __m256 _w7 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)w7));

                _sum4 = _mm256_comp_fmadd_ps(_m, _w4, _sum4);
                _sum5 = _mm256_comp_fmadd_ps(_m, _w5, _sum5);
                _sum6 = _mm256_comp_fmadd_ps(_m, _w6, _sum6);
                _sum7 = _mm256_comp_fmadd_ps(_m, _w7, _sum7);

                m += 8;
                w0 += 8;
                w1 += 8;
                w2 += 8;
                w3 += 8;
                w4 += 8;
                w5 += 8;
                w6 += 8;
                w7 += 8;
            }
            for (; i < num_input; i++)
            {
                sums[0] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w0);
                sums[1] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w1);
                sums[2] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w2);
                sums[3] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w3);
                sums[4] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w4);
                sums[5] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w5);
                sums[6] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w6);
                sums[7] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w7);

                m++;
                w0++;
                w1++;
                w2++;
                w3++;
                w4++;
                w5++;
                w6++;
                w7++;
            }

            __m256 _sums = HorizontalSums(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
            __m256 _sums_f = _mm256_loadu_ps(sums);
            _sums = _mm256_add_ps(_sums_f, _sums);
            _sums = activation_avx(_sums, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            __m128i _sums_bf16 = float2bfloat_avx(_sums);
            _mm_storeu_si128((__m128i*)(outptr + p), _sums_bf16);
        }

        remain_outw_start += (nn_outw << 3);
        nn_outw = (outw - remain_outw_start) >> 2;
#else
        int remain_outw_start = 0;
        int nn_outw = outw >> 2;
#endif // __AVX__

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outw; pp++)
        {
            int p = remain_outw_start + (pp * 4);

            float sums[4] = {0.0f};
            if (bias_data_ptr)
            {
                sums[0] = bias_data_ptr[p];
                sums[1] = bias_data_ptr[p + 1];
                sums[2] = bias_data_ptr[p + 2];
                sums[3] = bias_data_ptr[p + 3];
            }

            const unsigned short* w0 = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* w1 = weight_data_tm.row<const unsigned short>(p + 1);
            const unsigned short* w2 = weight_data_tm.row<const unsigned short>(p + 2);
            const unsigned short* w3 = weight_data_tm.row<const unsigned short>(p + 3);
            const unsigned short* m = bottom_blob;

            int i = 0;
#if __AVX__
            __m256 _sum0 = _mm256_setzero_ps();
            __m256 _sum1 = _mm256_setzero_ps();
            __m256 _sum2 = _mm256_setzero_ps();
            __m256 _sum3 = _mm256_setzero_ps();
            for (; i + 7 < num_input; i += 8)
            {
                __m256 _m = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)m));

                __m256 _w0 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)w0));
                __m256 _w1 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)w1));
                __m256 _w2 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)w2));
                __m256 _w3 = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)w3));

                _sum0 = _mm256_comp_fmadd_ps(_m, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_m, _w1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_m, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_m, _w3, _sum3);

                m += 8;
                w0 += 8;
                w1 += 8;
                w2 += 8;
                w3 += 8;
            }
#endif // __AVX__

            __m128 _sum0l = _mm_setzero_ps();
            __m128 _sum1l = _mm_setzero_ps();
            __m128 _sum2l = _mm_setzero_ps();
            __m128 _sum3l = _mm_setzero_ps();
            for (; i + 3 < num_input; i += 4)
            {
                __m128 _m = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)m));

                __m128 _w0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)w0));
                __m128 _w1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)w1));
                __m128 _w2 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)w2));
                __m128 _w3 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)w3));

                _sum0l = _mm_comp_fmadd_ps(_m, _w0, _sum0l);
                _sum1l = _mm_comp_fmadd_ps(_m, _w1, _sum1l);
                _sum2l = _mm_comp_fmadd_ps(_m, _w2, _sum2l);
                _sum3l = _mm_comp_fmadd_ps(_m, _w3, _sum3l);

                m += 4;
                w0 += 4;
                w1 += 4;
                w2 += 4;
                w3 += 4;
            }
            for (; i < num_input; i++)
            {
                sums[0] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w0);
                sums[1] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w1);
                sums[2] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w2);
                sums[3] += bfloat16_to_float32(*m) * bfloat16_to_float32(*w3);

                m++;
                w0++;
                w1++;
                w2++;
                w3++;
            }

            __m128 _sums = _mm_loadu_ps(sums);
#if __AVX__
            _sums = _mm_add_ps(HorizontalSums(_sum0, _sum1, _sum2, _sum3), _sums);
#endif
            _MM_TRANSPOSE4_PS(_sum0l, _sum1l, _sum2l, _sum3l);
            _sums = _mm_add_ps(_sum0l, _sums);
            _sums = _mm_add_ps(_sum1l, _sums);
            _sums = _mm_add_ps(_sum2l, _sums);
            _sums = _mm_add_ps(_sum3l, _sums);
            _sums = activation_sse(_sums, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            _mm_storel_epi64((__m128i*)(outptr + p), float2bfloat_sse(_sums, _mm_setzero_ps()));
        }

        remain_outw_start += (nn_outw << 2);
#else
        int remain_outw_start = 0;
#endif // __SSE2__

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outw_start; p < outw; p++)
        {
            float sum = 0.f;

            if (bias_data_ptr)
                sum = bias_data_ptr[p];

            const unsigned short* w = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* m = bottom_blob;

            int i = 0;
#if __SSE2__
#if __AVX__
            __m256 _sum = _mm256_setzero_ps();
            for (; i + 7 < num_input; i += 8)
            {
                __m256 _m = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)m));
                __m256 _w = bfloat2float_avx(_mm_lddqu_si128((const __m128i*)w));
                _sum = _mm256_comp_fmadd_ps(_m, _w, _sum);

                m += 8;
                w += 8;
            }
#endif // __AVX__
            __m128 _suml = _mm_setzero_ps();
            for (; i + 3 < num_input; i += 4)
            {
                __m128 _m = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)m));
                __m128 _w = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)w));
                _suml = _mm_comp_fmadd_ps(_m, _w, _suml);

                m += 4;
                w += 4;
            }
#endif // __SSE2__
            for (; i < num_input; i++)
            {
                sum += bfloat16_to_float32(*m) * bfloat16_to_float32(*w);
                m++;
                w++;
            }

#if __SSE2__
#if __AVX__
            _suml = _mm_add_ps(_suml, _mm256_extractf128_ps(_sum, 1));
            _suml = _mm_add_ps(_suml, _mm256_castps256_ps128(_sum));
#endif // __AVX__
            sum += _mm_reduce_add_ps(_suml);
#endif // __SSE2__

            sum = activation_ss(sum, activation_type, activation_params);

            unsigned short* outptr = (unsigned short*)top_blob;
            outptr[p] = float32_to_bfloat16(sum);
        }
    }
#endif // NCNN_RUNTIME_CPU
}

static void innerproduct_transform_kernel_bf16s_sse(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        innerproduct_transform_kernel_bf16s_sse_avx512bf16(weight_data, weight_data_tm, num_input, num_output, opt);
        return;
    }
#else // NCNN_RUNTIME_CPU

    int out_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        out_elempack = num_output % 16 == 0 ? 16 : num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#elif __AVX__
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

    // src = inch-outch
    // dst = pb-inch-outch/pb
#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (out_elempack == 16)
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_tm.create(num_input, num_output / 16, (size_t)32u, 16);

        for (int q = 0; q + 15 < num_output; q += 16)
        {
            unsigned short* g0 = weight_data_tm.row<unsigned short>(q / 16);

            const float* k0 = weight_data_r2.row(q);
            const float* k1 = weight_data_r2.row(q + 1);
            const float* k2 = weight_data_r2.row(q + 2);
            const float* k3 = weight_data_r2.row(q + 3);
            const float* k4 = weight_data_r2.row(q + 4);
            const float* k5 = weight_data_r2.row(q + 5);
            const float* k6 = weight_data_r2.row(q + 6);
            const float* k7 = weight_data_r2.row(q + 7);
            const float* k8 = weight_data_r2.row(q + 8);
            const float* k9 = weight_data_r2.row(q + 9);
            const float* ka = weight_data_r2.row(q + 10);
            const float* kb = weight_data_r2.row(q + 11);
            const float* kc = weight_data_r2.row(q + 12);
            const float* kd = weight_data_r2.row(q + 13);
            const float* ke = weight_data_r2.row(q + 14);
            const float* kf = weight_data_r2.row(q + 15);

            int p = 0;
            for (; p + 15 < num_input; p += 16)
            {
                // transpose 16x16
                __m256i _r0 = float2bfloat_avx512(_mm512_loadu_ps(k0));
                __m256i _r1 = float2bfloat_avx512(_mm512_loadu_ps(k1));
                __m256i _r2 = float2bfloat_avx512(_mm512_loadu_ps(k2));
                __m256i _r3 = float2bfloat_avx512(_mm512_loadu_ps(k3));
                __m256i _r4 = float2bfloat_avx512(_mm512_loadu_ps(k4));
                __m256i _r5 = float2bfloat_avx512(_mm512_loadu_ps(k5));
                __m256i _r6 = float2bfloat_avx512(_mm512_loadu_ps(k6));
                __m256i _r7 = float2bfloat_avx512(_mm512_loadu_ps(k7));
                __m256i _r8 = float2bfloat_avx512(_mm512_loadu_ps(k8));
                __m256i _r9 = float2bfloat_avx512(_mm512_loadu_ps(k9));
                __m256i _ra = float2bfloat_avx512(_mm512_loadu_ps(ka));
                __m256i _rb = float2bfloat_avx512(_mm512_loadu_ps(kb));
                __m256i _rc = float2bfloat_avx512(_mm512_loadu_ps(kc));
                __m256i _rd = float2bfloat_avx512(_mm512_loadu_ps(kd));
                __m256i _re = float2bfloat_avx512(_mm512_loadu_ps(ke));
                __m256i _rf = float2bfloat_avx512(_mm512_loadu_ps(kf));

                transpose16x16_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);

                _mm256_storeu_si256((__m256i*)g0, _r0);
                _mm256_storeu_si256((__m256i*)(g0 + 16), _r1);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 2), _r2);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 3), _r3);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 4), _r4);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 5), _r5);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 6), _r6);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 7), _r7);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 8), _r8);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 9), _r9);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 10), _ra);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 11), _rb);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 12), _rc);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 13), _rd);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 14), _re);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 15), _rf);

                k0 += 16;
                k1 += 16;
                k2 += 16;
                k3 += 16;
                k4 += 16;
                k5 += 16;
                k6 += 16;
                k7 += 16;
                k8 += 16;
                k9 += 16;
                ka += 16;
                kb += 16;
                kc += 16;
                kd += 16;
                ke += 16;
                kf += 16;
                g0 += 256;
            }
            for (; p + 7 < num_input; p += 8)
            {
                // transpose 8x16
                __m128i _r0 = float2bfloat_avx(_mm256_loadu_ps(k0));
                __m128i _r1 = float2bfloat_avx(_mm256_loadu_ps(k1));
                __m128i _r2 = float2bfloat_avx(_mm256_loadu_ps(k2));
                __m128i _r3 = float2bfloat_avx(_mm256_loadu_ps(k3));
                __m128i _r4 = float2bfloat_avx(_mm256_loadu_ps(k4));
                __m128i _r5 = float2bfloat_avx(_mm256_loadu_ps(k5));
                __m128i _r6 = float2bfloat_avx(_mm256_loadu_ps(k6));
                __m128i _r7 = float2bfloat_avx(_mm256_loadu_ps(k7));
                __m128i _r8 = float2bfloat_avx(_mm256_loadu_ps(k8));
                __m128i _r9 = float2bfloat_avx(_mm256_loadu_ps(k9));
                __m128i _ra = float2bfloat_avx(_mm256_loadu_ps(ka));
                __m128i _rb = float2bfloat_avx(_mm256_loadu_ps(kb));
                __m128i _rc = float2bfloat_avx(_mm256_loadu_ps(kc));
                __m128i _rd = float2bfloat_avx(_mm256_loadu_ps(kd));
                __m128i _re = float2bfloat_avx(_mm256_loadu_ps(ke));
                __m128i _rf = float2bfloat_avx(_mm256_loadu_ps(kf));

                __m256i _r08 = combine4x2_epi32(_r0, _r8);
                __m256i _r19 = combine4x2_epi32(_r1, _r9);
                __m256i _r2a = combine4x2_epi32(_r2, _ra);
                __m256i _r3b = combine4x2_epi32(_r3, _rb);
                __m256i _r4c = combine4x2_epi32(_r4, _rc);
                __m256i _r5d = combine4x2_epi32(_r5, _rd);
                __m256i _r6e = combine4x2_epi32(_r6, _re);
                __m256i _r7f = combine4x2_epi32(_r7, _rf);

                __m256i _tmp0 = _mm256_unpacklo_epi16(_r08, _r19);
                __m256i _tmp1 = _mm256_unpackhi_epi16(_r08, _r19);
                __m256i _tmp2 = _mm256_unpacklo_epi16(_r2a, _r3b);
                __m256i _tmp3 = _mm256_unpackhi_epi16(_r2a, _r3b);
                __m256i _tmp4 = _mm256_unpacklo_epi16(_r4c, _r5d);
                __m256i _tmp5 = _mm256_unpackhi_epi16(_r4c, _r5d);
                __m256i _tmp6 = _mm256_unpacklo_epi16(_r6e, _r7f);
                __m256i _tmp7 = _mm256_unpackhi_epi16(_r6e, _r7f);

                __m256i _tmpg = _mm256_unpacklo_epi32(_tmp0, _tmp2);
                __m256i _tmph = _mm256_unpackhi_epi32(_tmp0, _tmp2);
                __m256i _tmpi = _mm256_unpacklo_epi32(_tmp1, _tmp3);
                __m256i _tmpj = _mm256_unpackhi_epi32(_tmp1, _tmp3);
                __m256i _tmpk = _mm256_unpacklo_epi32(_tmp4, _tmp6);
                __m256i _tmpl = _mm256_unpackhi_epi32(_tmp4, _tmp6);
                __m256i _tmpm = _mm256_unpacklo_epi32(_tmp5, _tmp7);
                __m256i _tmpn = _mm256_unpackhi_epi32(_tmp5, _tmp7);

                _r08 = _mm256_unpacklo_epi64(_tmpg, _tmpk);
                _r19 = _mm256_unpackhi_epi64(_tmpg, _tmpk);
                _r2a = _mm256_unpacklo_epi64(_tmph, _tmpl);
                _r3b = _mm256_unpackhi_epi64(_tmph, _tmpl);
                _r4c = _mm256_unpacklo_epi64(_tmpi, _tmpm);
                _r5d = _mm256_unpackhi_epi64(_tmpi, _tmpm);
                _r6e = _mm256_unpacklo_epi64(_tmpj, _tmpn);
                _r7f = _mm256_unpackhi_epi64(_tmpj, _tmpn);

                _mm256_storeu_si256((__m256i*)g0, _r08);
                _mm256_storeu_si256((__m256i*)(g0 + 16), _r19);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 2), _r2a);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 3), _r3b);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 4), _r4c);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 5), _r5d);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 6), _r6e);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 7), _r7f);

                k0 += 8;
                k1 += 8;
                k2 += 8;
                k3 += 8;
                k4 += 8;
                k5 += 8;
                k6 += 8;
                k7 += 8;
                k8 += 8;
                k9 += 8;
                ka += 8;
                kb += 8;
                kc += 8;
                kd += 8;
                ke += 8;
                kf += 8;
                g0 += 128;
            }
            for (; p < num_input; p++)
            {
                g0[0] = float32_to_bfloat16(*k0++);
                g0[1] = float32_to_bfloat16(*k1++);
                g0[2] = float32_to_bfloat16(*k2++);
                g0[3] = float32_to_bfloat16(*k3++);
                g0[4] = float32_to_bfloat16(*k4++);
                g0[5] = float32_to_bfloat16(*k5++);
                g0[6] = float32_to_bfloat16(*k6++);
                g0[7] = float32_to_bfloat16(*k7++);
                g0[8] = float32_to_bfloat16(*k8++);
                g0[9] = float32_to_bfloat16(*k9++);
                g0[10] = float32_to_bfloat16(*ka++);
                g0[11] = float32_to_bfloat16(*kb++);
                g0[12] = float32_to_bfloat16(*kc++);
                g0[13] = float32_to_bfloat16(*kd++);
                g0[14] = float32_to_bfloat16(*ke++);
                g0[15] = float32_to_bfloat16(*kf++);
                g0 += 16;
            }
        }
    }
#endif // __AVX512F__

    if (out_elempack == 8)
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_tm.create(num_input, num_output / 8, (size_t)16u, 8);

        for (int q = 0; q + 7 < num_output; q += 8)
        {
            unsigned short* g0 = weight_data_tm.row<unsigned short>(q / 8);

            const float* k0 = weight_data_r2.row(q);
            const float* k1 = weight_data_r2.row(q + 1);
            const float* k2 = weight_data_r2.row(q + 2);
            const float* k3 = weight_data_r2.row(q + 3);
            const float* k4 = weight_data_r2.row(q + 4);
            const float* k5 = weight_data_r2.row(q + 5);
            const float* k6 = weight_data_r2.row(q + 6);
            const float* k7 = weight_data_r2.row(q + 7);

            int p = 0;
#if __AVX512F__
            for (; p + 15 < num_input; p += 16)
            {
                // transpose 16x8
                __m256i _r0 = float2bfloat_avx512(_mm512_loadu_ps(k0));
                __m256i _r1 = float2bfloat_avx512(_mm512_loadu_ps(k1));
                __m256i _r2 = float2bfloat_avx512(_mm512_loadu_ps(k2));
                __m256i _r3 = float2bfloat_avx512(_mm512_loadu_ps(k3));
                __m256i _r4 = float2bfloat_avx512(_mm512_loadu_ps(k4));
                __m256i _r5 = float2bfloat_avx512(_mm512_loadu_ps(k5));
                __m256i _r6 = float2bfloat_avx512(_mm512_loadu_ps(k6));
                __m256i _r7 = float2bfloat_avx512(_mm512_loadu_ps(k7));

                transpose16x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                _mm256_storeu_si256((__m256i*)g0, _r0);
                _mm256_storeu_si256((__m256i*)(g0 + 16), _r1);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 2), _r2);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 3), _r3);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 4), _r4);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 5), _r5);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 6), _r6);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 7), _r7);

                k0 += 16;
                k1 += 16;
                k2 += 16;
                k3 += 16;
                k4 += 16;
                k5 += 16;
                k6 += 16;
                k7 += 16;
                g0 += 128;
            }
#endif // __AVX512F__
            for (; p + 7 < num_input; p += 8)
            {
                // transpose 8x8
                __m128i _r0 = float2bfloat_avx(_mm256_loadu_ps(k0));
                __m128i _r1 = float2bfloat_avx(_mm256_loadu_ps(k1));
                __m128i _r2 = float2bfloat_avx(_mm256_loadu_ps(k2));
                __m128i _r3 = float2bfloat_avx(_mm256_loadu_ps(k3));
                __m128i _r4 = float2bfloat_avx(_mm256_loadu_ps(k4));
                __m128i _r5 = float2bfloat_avx(_mm256_loadu_ps(k5));
                __m128i _r6 = float2bfloat_avx(_mm256_loadu_ps(k6));
                __m128i _r7 = float2bfloat_avx(_mm256_loadu_ps(k7));

                transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                _mm_storeu_si128((__m128i*)g0, _r0);
                _mm_storeu_si128((__m128i*)(g0 + 8), _r1);
                _mm_storeu_si128((__m128i*)(g0 + 16), _r2);
                _mm_storeu_si128((__m128i*)(g0 + 24), _r3);
                _mm_storeu_si128((__m128i*)(g0 + 32), _r4);
                _mm_storeu_si128((__m128i*)(g0 + 40), _r5);
                _mm_storeu_si128((__m128i*)(g0 + 48), _r6);
                _mm_storeu_si128((__m128i*)(g0 + 56), _r7);

                k0 += 8;
                k1 += 8;
                k2 += 8;
                k3 += 8;
                k4 += 8;
                k5 += 8;
                k6 += 8;
                k7 += 8;
                g0 += 64;
            }
            for (; p < num_input; p++)
            {
                g0[0] = float32_to_bfloat16(*k0++);
                g0[1] = float32_to_bfloat16(*k1++);
                g0[2] = float32_to_bfloat16(*k2++);
                g0[3] = float32_to_bfloat16(*k3++);
                g0[4] = float32_to_bfloat16(*k4++);
                g0[5] = float32_to_bfloat16(*k5++);
                g0[6] = float32_to_bfloat16(*k6++);
                g0[7] = float32_to_bfloat16(*k7++);
                g0 += 8;
            }
        }
    }
#endif // __AVX__

    if (out_elempack == 4)
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_tm.create(num_input, num_output / 4, (size_t)8u, 4);

        for (int q = 0; q + 3 < num_output; q += 4)
        {
            unsigned short* g0 = weight_data_tm.row<unsigned short>(q / 4);

            const float* k0 = weight_data_r2.row(q);
            const float* k1 = weight_data_r2.row(q + 1);
            const float* k2 = weight_data_r2.row(q + 2);
            const float* k3 = weight_data_r2.row(q + 3);

            int p = 0;
            for (; p + 3 < num_input; p += 4)
            {
                // transpose 4x4
                __m128 _r0 = _mm_loadu_ps(k0);
                __m128 _r1 = _mm_loadu_ps(k1);
                __m128 _r2 = _mm_loadu_ps(k2);
                __m128 _r3 = _mm_loadu_ps(k3);
                _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
#if __AVX__
                __m128i _r01_bf16 = float2bfloat_avx(combine4x2_ps(_r0, _r1));
                __m128i _r23_bf16 = float2bfloat_avx(combine4x2_ps(_r2, _r3));
                _mm_storeu_si128((__m128i*)g0, _r01_bf16);
                _mm_storeu_si128((__m128i*)(g0 + 8), _r23_bf16);
#else
                __m128i _r01_bf16 = float2bfloat_sse(_r0, _r1);
                __m128i _r23_bf16 = float2bfloat_sse(_r2, _r3);
                _mm_storeu_si128((__m128i*)g0, _r01_bf16);
                _mm_storeu_si128((__m128i*)(g0 + 8), _r23_bf16);
#endif

                k0 += 4;
                k1 += 4;
                k2 += 4;
                k3 += 4;
                g0 += 16;
            }
            for (; p < num_input; p++)
            {
                g0[0] = float32_to_bfloat16(*k0++);
                g0[1] = float32_to_bfloat16(*k1++);
                g0[2] = float32_to_bfloat16(*k2++);
                g0[3] = float32_to_bfloat16(*k3++);
                g0 += 4;
            }
        }
    }
#endif // __SSE2__

    if (out_elempack == 1)
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);
        ncnn::cast_float32_to_bfloat16(weight_data_r2, weight_data_tm, opt);
    }
#endif // NCNN_RUNTIME_CPU
}
