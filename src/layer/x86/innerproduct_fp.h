// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#if NCNN_RUNTIME_CPU && NCNN_F16C && __AVX__ && !__F16C__
void innerproduct_fp16s_sse_f16c(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
void innerproduct_transform_kernel_fp16s_sse_f16c(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt);
#endif

#if NCNN_IMPL_FP16S
static void innerproduct_fp16s_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
#else
static void innerproduct_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
#endif
{
#if NCNN_RUNTIME_CPU && NCNN_IMPL_FP16S && NCNN_F16C && __AVX__ && !__F16C__
    if (ncnn::cpu_support_x86_f16c())
    {
        innerproduct_fp16s_sse_f16c(bottom_blob, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);
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

#if NCNN_IMPL_FP16S
            const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
#else
            const float* kptr = weight_data_tm.row(p);
#endif
            const float* sptr = bottom_blob;

            int i = 0;
            for (; i + 7 < num_input; i += 8)
            {
                __m512 _val0 = _mm512_set1_ps(sptr[0]);
                __m512 _val1 = _mm512_set1_ps(sptr[1]);
                __m512 _val2 = _mm512_set1_ps(sptr[2]);
                __m512 _val3 = _mm512_set1_ps(sptr[3]);
#if NCNN_IMPL_FP16S
                __m512i _w01 = _mm512_loadu_si512(kptr);
                __m512i _w23 = _mm512_loadu_si512(kptr + 32);
                __m512 _w0 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w01, 0));
                __m512 _w1 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w01, 1));
                __m512 _w2 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w23, 0));
                __m512 _w3 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w23, 1));
#else
                __m512 _w0 = _mm512_loadu_ps(kptr + 16 * 0);
                __m512 _w1 = _mm512_loadu_ps(kptr + 16 * 1);
                __m512 _w2 = _mm512_loadu_ps(kptr + 16 * 2);
                __m512 _w3 = _mm512_loadu_ps(kptr + 16 * 3);
#endif

                _sum0 = _mm512_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm512_fmadd_ps(_val1, _w1, _sum1);
                _sum2 = _mm512_fmadd_ps(_val2, _w2, _sum2);
                _sum3 = _mm512_fmadd_ps(_val3, _w3, _sum3);

                __m512 _val4 = _mm512_set1_ps(sptr[4]);
                __m512 _val5 = _mm512_set1_ps(sptr[5]);
                __m512 _val6 = _mm512_set1_ps(sptr[6]);
                __m512 _val7 = _mm512_set1_ps(sptr[7]);
#if NCNN_IMPL_FP16S
                __m512i _w45 = _mm512_loadu_si512(kptr + 64);
                __m512i _w67 = _mm512_loadu_si512(kptr + 96);
                __m512 _w4 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w45, 0));
                __m512 _w5 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w45, 1));
                __m512 _w6 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w67, 0));
                __m512 _w7 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w67, 1));
#else
                __m512 _w4 = _mm512_loadu_ps(kptr + 16 * 4);
                __m512 _w5 = _mm512_loadu_ps(kptr + 16 * 5);
                __m512 _w6 = _mm512_loadu_ps(kptr + 16 * 6);
                __m512 _w7 = _mm512_loadu_ps(kptr + 16 * 7);
#endif

                _sum4 = _mm512_fmadd_ps(_val4, _w4, _sum4);
                _sum5 = _mm512_fmadd_ps(_val5, _w5, _sum5);
                _sum6 = _mm512_fmadd_ps(_val6, _w6, _sum6);
                _sum7 = _mm512_fmadd_ps(_val7, _w7, _sum7);

                sptr += 8;
                kptr += 128;
            }
            for (; i + 3 < num_input; i += 4)
            {
                __m512 _val0 = _mm512_set1_ps(sptr[0]);
                __m512 _val1 = _mm512_set1_ps(sptr[1]);
                __m512 _val2 = _mm512_set1_ps(sptr[2]);
                __m512 _val3 = _mm512_set1_ps(sptr[3]);
#if NCNN_IMPL_FP16S
                __m512i _w01 = _mm512_loadu_si512(kptr);
                __m512i _w23 = _mm512_loadu_si512(kptr + 32);
                __m512 _w0 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w01, 0));
                __m512 _w1 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w01, 1));
                __m512 _w2 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w23, 0));
                __m512 _w3 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w23, 1));
#else
                __m512 _w0 = _mm512_loadu_ps(kptr);
                __m512 _w1 = _mm512_loadu_ps(kptr + 16);
                __m512 _w2 = _mm512_loadu_ps(kptr + 32);
                __m512 _w3 = _mm512_loadu_ps(kptr + 48);
#endif

                _sum0 = _mm512_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm512_fmadd_ps(_val1, _w1, _sum1);
                _sum2 = _mm512_fmadd_ps(_val2, _w2, _sum2);
                _sum3 = _mm512_fmadd_ps(_val3, _w3, _sum3);

                sptr += 4;
                kptr += 64;
            }
            for (; i < num_input; i++)
            {
                __m512 _val = _mm512_set1_ps(sptr[0]);
#if NCNN_IMPL_FP16S
                __m512 _w = _mm512_cvtph_ps(_mm256_lddqu_si256((const __m256i*)kptr));
#else
                __m512 _w = _mm512_loadu_ps(kptr);
#endif
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

            float* outptr = top_blob;
            _mm512_storeu_ps(outptr + p * 16, _sum0);
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

#if NCNN_IMPL_FP16S
            const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
#else
            const float* kptr = weight_data_tm.row(p);
#endif
            const float* sptr = bottom_blob;

            int i = 0;
            for (; i + 7 < num_input; i += 8)
            {
                __m256 _val0 = _mm256_broadcast_ss(sptr);
                __m256 _val1 = _mm256_broadcast_ss(sptr + 1);
                __m256 _val2 = _mm256_broadcast_ss(sptr + 2);
                __m256 _val3 = _mm256_broadcast_ss(sptr + 3);
#if NCNN_IMPL_FP16S
                __m256i _w01 = _mm256_lddqu_si256((const __m256i*)kptr);
                __m256i _w23 = _mm256_lddqu_si256((const __m256i*)(kptr + 16));
                __m256 _w0 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w01, 0));
                __m256 _w1 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w01, 1));
                __m256 _w2 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w23, 0));
                __m256 _w3 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w23, 1));
#else
                __m256 _w0 = _mm256_loadu_ps(kptr);
                __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                __m256 _w3 = _mm256_loadu_ps(kptr + 24);
#endif

                _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_val1, _w1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_val2, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_val3, _w3, _sum3);

                __m256 _val4 = _mm256_broadcast_ss(sptr + 4);
                __m256 _val5 = _mm256_broadcast_ss(sptr + 5);
                __m256 _val6 = _mm256_broadcast_ss(sptr + 6);
                __m256 _val7 = _mm256_broadcast_ss(sptr + 7);
#if NCNN_IMPL_FP16S
                __m256i _w45 = _mm256_lddqu_si256((const __m256i*)(kptr + 32));
                __m256i _w67 = _mm256_lddqu_si256((const __m256i*)(kptr + 48));
                __m256 _w4 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w45, 0));
                __m256 _w5 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w45, 1));
                __m256 _w6 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w67, 0));
                __m256 _w7 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w67, 1));
#else
                __m256 _w4 = _mm256_loadu_ps(kptr + 32);
                __m256 _w5 = _mm256_loadu_ps(kptr + 40);
                __m256 _w6 = _mm256_loadu_ps(kptr + 48);
                __m256 _w7 = _mm256_loadu_ps(kptr + 56);
#endif

                _sum4 = _mm256_comp_fmadd_ps(_val4, _w4, _sum4);
                _sum5 = _mm256_comp_fmadd_ps(_val5, _w5, _sum5);
                _sum6 = _mm256_comp_fmadd_ps(_val6, _w6, _sum6);
                _sum7 = _mm256_comp_fmadd_ps(_val7, _w7, _sum7);

                sptr += 8;
                kptr += 64;
            }
            for (; i + 3 < num_input; i += 4)
            {
                __m256 _val0 = _mm256_broadcast_ss(sptr);
                __m256 _val1 = _mm256_broadcast_ss(sptr + 1);
                __m256 _val2 = _mm256_broadcast_ss(sptr + 2);
                __m256 _val3 = _mm256_broadcast_ss(sptr + 3);
#if NCNN_IMPL_FP16S
                __m256i _w01 = _mm256_lddqu_si256((const __m256i*)kptr);
                __m256i _w23 = _mm256_lddqu_si256((const __m256i*)(kptr + 16));
                __m256 _w0 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w01, 0));
                __m256 _w1 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w01, 1));
                __m256 _w2 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w23, 0));
                __m256 _w3 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w23, 1));
#else
                __m256 _w0 = _mm256_loadu_ps(kptr);
                __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                __m256 _w3 = _mm256_loadu_ps(kptr + 24);
#endif

                _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_val1, _w1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_val2, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_val3, _w3, _sum3);

                sptr += 4;
                kptr += 32;
            }
            for (; i < num_input; i++)
            {
                __m256 _val = _mm256_set1_ps(sptr[0]);
#if NCNN_IMPL_FP16S
                __m256 _w = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)kptr));
#else
                __m256 _w = _mm256_loadu_ps(kptr);
#endif
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

            float* outptr = top_blob;
            _mm256_storeu_ps(outptr + p * 8, _sum0);
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

#if NCNN_IMPL_FP16S
            const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
#else
            const float* kptr = weight_data_tm.row(p);
#endif
            const float* sptr = bottom_blob;

            int i = 0;
#if __AVX__
            for (; i + 7 < num_input; i += 8)
            {
                __m128 _val0 = _mm_broadcast_ss(sptr);
                __m128 _val1 = _mm_broadcast_ss(sptr + 1);
                __m128 _val2 = _mm_broadcast_ss(sptr + 2);
                __m128 _val3 = _mm_broadcast_ss(sptr + 3);
                __m128 _val4 = _mm_broadcast_ss(sptr + 4);
                __m128 _val5 = _mm_broadcast_ss(sptr + 5);
                __m128 _val6 = _mm_broadcast_ss(sptr + 6);
                __m128 _val7 = _mm_broadcast_ss(sptr + 7);

                __m256 _val01 = _mm256_insertf128_ps(_mm256_castps128_ps256(_val0), _val1, 1);
                __m256 _val23 = _mm256_insertf128_ps(_mm256_castps128_ps256(_val2), _val3, 1);
                __m256 _val45 = _mm256_insertf128_ps(_mm256_castps128_ps256(_val4), _val5, 1);
                __m256 _val67 = _mm256_insertf128_ps(_mm256_castps128_ps256(_val6), _val7, 1);

#if NCNN_IMPL_FP16S
                __m256i _w0123 = _mm256_lddqu_si256((const __m256i*)kptr);
                __m256i _w4567 = _mm256_lddqu_si256((const __m256i*)(kptr + 16));
                __m256 _w01 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w0123, 0));
                __m256 _w23 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w0123, 1));
                __m256 _w45 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w4567, 0));
                __m256 _w67 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w4567, 1));
#else
                __m256 _w01 = _mm256_loadu_ps(kptr);
                __m256 _w23 = _mm256_loadu_ps(kptr + 8);
                __m256 _w45 = _mm256_loadu_ps(kptr + 16);
                __m256 _w67 = _mm256_loadu_ps(kptr + 24);
#endif

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
                __m128 _val0 = _mm_broadcast_ss(sptr);
                __m128 _val1 = _mm_broadcast_ss(sptr + 1);
                __m128 _val2 = _mm_broadcast_ss(sptr + 2);
                __m128 _val3 = _mm_broadcast_ss(sptr + 3);

                __m256 _val01 = _mm256_insertf128_ps(_mm256_castps128_ps256(_val0), _val1, 1);
                __m256 _val23 = _mm256_insertf128_ps(_mm256_castps128_ps256(_val2), _val3, 1);

#if NCNN_IMPL_FP16S
                __m256i _w0123 = _mm256_lddqu_si256((const __m256i*)kptr);
                __m256 _w01 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w0123, 0));
                __m256 _w23 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w0123, 1));
#else
                __m256 _w01 = _mm256_loadu_ps(kptr);
                __m256 _w23 = _mm256_loadu_ps(kptr + 8);
#endif

                _sum01 = _mm256_comp_fmadd_ps(_val01, _w01, _sum01);
                _sum23 = _mm256_comp_fmadd_ps(_val23, _w23, _sum23);
#else
                __m128 _val0 = _mm_set1_ps(sptr[0]);
                __m128 _val1 = _mm_set1_ps(sptr[1]);
                __m128 _val2 = _mm_set1_ps(sptr[2]);
                __m128 _val3 = _mm_set1_ps(sptr[3]);

                __m128 _w0 = _mm_loadu_ps(kptr);
                __m128 _w1 = _mm_loadu_ps(kptr + 4);
                __m128 _w2 = _mm_loadu_ps(kptr + 8);
                __m128 _w3 = _mm_loadu_ps(kptr + 12);

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
                __m128 _val = _mm_set1_ps(sptr[0]);
#if NCNN_IMPL_FP16S
                __m128 _w = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)kptr));
#else
                __m128 _w = _mm_loadu_ps(kptr);
#endif
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

            float* outptr = top_blob;
            _mm_storeu_ps(outptr + p * 4, _sum0);
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

#if NCNN_IMPL_FP16S
            const unsigned short* w0 = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* w1 = weight_data_tm.row<const unsigned short>(p + 1);
            const unsigned short* w2 = weight_data_tm.row<const unsigned short>(p + 2);
            const unsigned short* w3 = weight_data_tm.row<const unsigned short>(p + 3);
            const unsigned short* w4 = weight_data_tm.row<const unsigned short>(p + 4);
            const unsigned short* w5 = weight_data_tm.row<const unsigned short>(p + 5);
            const unsigned short* w6 = weight_data_tm.row<const unsigned short>(p + 6);
            const unsigned short* w7 = weight_data_tm.row<const unsigned short>(p + 7);
#else
            const float* w0 = (const float*)weight_data_tm + num_input * p;
            const float* w1 = (const float*)weight_data_tm + num_input * (p + 1);
            const float* w2 = (const float*)weight_data_tm + num_input * (p + 2);
            const float* w3 = (const float*)weight_data_tm + num_input * (p + 3);
            const float* w4 = (const float*)weight_data_tm + num_input * (p + 4);
            const float* w5 = (const float*)weight_data_tm + num_input * (p + 5);
            const float* w6 = (const float*)weight_data_tm + num_input * (p + 6);
            const float* w7 = (const float*)weight_data_tm + num_input * (p + 7);
#endif
            const float* m = bottom_blob;

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
                __m256 _m = _mm256_loadu_ps(m);

#if NCNN_IMPL_FP16S
                __m256 _w0 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w0));
                __m256 _w1 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w1));
                __m256 _w2 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w2));
                __m256 _w3 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w3));
#else
                __m256 _w0 = _mm256_loadu_ps(w0);
                __m256 _w1 = _mm256_loadu_ps(w1);
                __m256 _w2 = _mm256_loadu_ps(w2);
                __m256 _w3 = _mm256_loadu_ps(w3);
#endif

                _sum0 = _mm256_comp_fmadd_ps(_m, _w0, _sum0);
                _sum1 = _mm256_comp_fmadd_ps(_m, _w1, _sum1);
                _sum2 = _mm256_comp_fmadd_ps(_m, _w2, _sum2);
                _sum3 = _mm256_comp_fmadd_ps(_m, _w3, _sum3);

#if NCNN_IMPL_FP16S
                __m256 _w4 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w4));
                __m256 _w5 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w5));
                __m256 _w6 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w6));
                __m256 _w7 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w7));
#else
                __m256 _w4 = _mm256_loadu_ps(w4);
                __m256 _w5 = _mm256_loadu_ps(w5);
                __m256 _w6 = _mm256_loadu_ps(w6);
                __m256 _w7 = _mm256_loadu_ps(w7);
#endif

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
#if NCNN_IMPL_FP16S
                sums[0] += *m * float16_to_float32(*w0);
                sums[1] += *m * float16_to_float32(*w1);
                sums[2] += *m * float16_to_float32(*w2);
                sums[3] += *m * float16_to_float32(*w3);
                sums[4] += *m * float16_to_float32(*w4);
                sums[5] += *m * float16_to_float32(*w5);
                sums[6] += *m * float16_to_float32(*w6);
                sums[7] += *m * float16_to_float32(*w7);
#else
                sums[0] += *m * *w0;
                sums[1] += *m * *w1;
                sums[2] += *m * *w2;
                sums[3] += *m * *w3;
                sums[4] += *m * *w4;
                sums[5] += *m * *w5;
                sums[6] += *m * *w6;
                sums[7] += *m * *w7;
#endif

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

            float* outptr = top_blob;
            _mm256_storeu_ps(outptr + p, _sums);
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

#if NCNN_IMPL_FP16S
            const unsigned short* w0 = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* w1 = weight_data_tm.row<const unsigned short>(p + 1);
            const unsigned short* w2 = weight_data_tm.row<const unsigned short>(p + 2);
            const unsigned short* w3 = weight_data_tm.row<const unsigned short>(p + 3);
#else
            const float* w0 = (const float*)weight_data_tm + num_input * p;
            const float* w1 = (const float*)weight_data_tm + num_input * (p + 1);
            const float* w2 = (const float*)weight_data_tm + num_input * (p + 2);
            const float* w3 = (const float*)weight_data_tm + num_input * (p + 3);
#endif
            const float* m = bottom_blob;

            int i = 0;
#if __AVX__
            __m256 _sum0 = _mm256_setzero_ps();
            __m256 _sum1 = _mm256_setzero_ps();
            __m256 _sum2 = _mm256_setzero_ps();
            __m256 _sum3 = _mm256_setzero_ps();
            for (; i + 7 < num_input; i += 8)
            {
                __m256 _m = _mm256_loadu_ps(m);

#if NCNN_IMPL_FP16S
                __m256 _w0 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w0));
                __m256 _w1 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w1));
                __m256 _w2 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w2));
                __m256 _w3 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w3));
#else
                __m256 _w0 = _mm256_loadu_ps(w0);
                __m256 _w1 = _mm256_loadu_ps(w1);
                __m256 _w2 = _mm256_loadu_ps(w2);
                __m256 _w3 = _mm256_loadu_ps(w3);
#endif

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
                __m128 _m = _mm_loadu_ps(m);

#if NCNN_IMPL_FP16S
                __m128 _w0 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)w0));
                __m128 _w1 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)w1));
                __m128 _w2 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)w2));
                __m128 _w3 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)w3));
#else
                __m128 _w0 = _mm_loadu_ps(w0);
                __m128 _w1 = _mm_loadu_ps(w1);
                __m128 _w2 = _mm_loadu_ps(w2);
                __m128 _w3 = _mm_loadu_ps(w3);
#endif

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
#if NCNN_IMPL_FP16S
                sums[0] += *m * float16_to_float32(*w0);
                sums[1] += *m * float16_to_float32(*w1);
                sums[2] += *m * float16_to_float32(*w2);
                sums[3] += *m * float16_to_float32(*w3);
#else
                sums[0] += *m * *w0;
                sums[1] += *m * *w1;
                sums[2] += *m * *w2;
                sums[3] += *m * *w3;
#endif

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

            float* outptr = top_blob;
            _mm_storeu_ps(outptr + p, _sums);
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

#if NCNN_IMPL_FP16S
            const unsigned short* w = weight_data_tm.row<const unsigned short>(p);
#else
            const float* w = (const float*)weight_data_tm + num_input * p;
#endif
            const float* m = bottom_blob;

            int i = 0;
#if __SSE2__
#if __AVX__
            __m256 _sum = _mm256_setzero_ps();
            for (; i + 7 < num_input; i += 8)
            {
                __m256 _m = _mm256_loadu_ps(m);
#if NCNN_IMPL_FP16S
                __m256 _w = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w));
#else
                __m256 _w = _mm256_loadu_ps(w);
#endif
                _sum = _mm256_comp_fmadd_ps(_m, _w, _sum);

                m += 8;
                w += 8;
            }
#endif // __AVX__
            __m128 _suml = _mm_setzero_ps();
            for (; i + 3 < num_input; i += 4)
            {
                __m128 _m = _mm_loadu_ps(m);
#if NCNN_IMPL_FP16S
                __m128 _w = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)w));
#else
                __m128 _w = _mm_loadu_ps(w);
#endif
                _suml = _mm_comp_fmadd_ps(_m, _w, _suml);

                m += 4;
                w += 4;
            }
#endif // __SSE2__
            for (; i < num_input; i++)
            {
#if NCNN_IMPL_FP16S
                sum += *m * float16_to_float32(*w);
#else
                sum += *m * *w;
#endif
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

            float* outptr = top_blob;
            outptr[p] = sum;
        }
    }
#endif // NCNN_RUNTIME_CPU
}

#if NCNN_IMPL_FP16S
static void innerproduct_transform_kernel_fp16s_sse(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt)
#else
static void innerproduct_transform_kernel_sse(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt)
#endif
{
#if NCNN_RUNTIME_CPU && NCNN_IMPL_FP16S && NCNN_F16C && __AVX__ && !__F16C__
    if (ncnn::cpu_support_x86_f16c())
    {
        innerproduct_transform_kernel_fp16s_sse_f16c(weight_data, weight_data_tm, num_input, num_output, opt);
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

#if NCNN_IMPL_FP16S
        weight_data_tm.create(num_input, num_output / 16, (size_t)32u, 16);
#else
        weight_data_tm.create(num_input, num_output / 16, (size_t)64u, 16);
#endif

        for (int q = 0; q + 15 < num_output; q += 16)
        {
#if NCNN_IMPL_FP16S
            unsigned short* g0 = weight_data_tm.row<unsigned short>(q / 16);
#else
            float* g0 = weight_data_tm.row(q / 16);
#endif

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
#if NCNN_IMPL_FP16S
                __m256i _r0 = _mm512_cvtps_ph(_mm512_loadu_ps(k0), _MM_FROUND_TRUNC);
                __m256i _r1 = _mm512_cvtps_ph(_mm512_loadu_ps(k1), _MM_FROUND_TRUNC);
                __m256i _r2 = _mm512_cvtps_ph(_mm512_loadu_ps(k2), _MM_FROUND_TRUNC);
                __m256i _r3 = _mm512_cvtps_ph(_mm512_loadu_ps(k3), _MM_FROUND_TRUNC);
                __m256i _r4 = _mm512_cvtps_ph(_mm512_loadu_ps(k4), _MM_FROUND_TRUNC);
                __m256i _r5 = _mm512_cvtps_ph(_mm512_loadu_ps(k5), _MM_FROUND_TRUNC);
                __m256i _r6 = _mm512_cvtps_ph(_mm512_loadu_ps(k6), _MM_FROUND_TRUNC);
                __m256i _r7 = _mm512_cvtps_ph(_mm512_loadu_ps(k7), _MM_FROUND_TRUNC);
                __m256i _r8 = _mm512_cvtps_ph(_mm512_loadu_ps(k8), _MM_FROUND_TRUNC);
                __m256i _r9 = _mm512_cvtps_ph(_mm512_loadu_ps(k9), _MM_FROUND_TRUNC);
                __m256i _ra = _mm512_cvtps_ph(_mm512_loadu_ps(ka), _MM_FROUND_TRUNC);
                __m256i _rb = _mm512_cvtps_ph(_mm512_loadu_ps(kb), _MM_FROUND_TRUNC);
                __m256i _rc = _mm512_cvtps_ph(_mm512_loadu_ps(kc), _MM_FROUND_TRUNC);
                __m256i _rd = _mm512_cvtps_ph(_mm512_loadu_ps(kd), _MM_FROUND_TRUNC);
                __m256i _re = _mm512_cvtps_ph(_mm512_loadu_ps(ke), _MM_FROUND_TRUNC);
                __m256i _rf = _mm512_cvtps_ph(_mm512_loadu_ps(kf), _MM_FROUND_TRUNC);

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
#else
                __m512 _r0 = _mm512_loadu_ps(k0);
                __m512 _r1 = _mm512_loadu_ps(k1);
                __m512 _r2 = _mm512_loadu_ps(k2);
                __m512 _r3 = _mm512_loadu_ps(k3);
                __m512 _r4 = _mm512_loadu_ps(k4);
                __m512 _r5 = _mm512_loadu_ps(k5);
                __m512 _r6 = _mm512_loadu_ps(k6);
                __m512 _r7 = _mm512_loadu_ps(k7);
                __m512 _r8 = _mm512_loadu_ps(k8);
                __m512 _r9 = _mm512_loadu_ps(k9);
                __m512 _ra = _mm512_loadu_ps(ka);
                __m512 _rb = _mm512_loadu_ps(kb);
                __m512 _rc = _mm512_loadu_ps(kc);
                __m512 _rd = _mm512_loadu_ps(kd);
                __m512 _re = _mm512_loadu_ps(ke);
                __m512 _rf = _mm512_loadu_ps(kf);

                transpose16x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);

                _mm512_storeu_ps(g0, _r0);
                _mm512_storeu_ps(g0 + 16, _r1);
                _mm512_storeu_ps(g0 + 16 * 2, _r2);
                _mm512_storeu_ps(g0 + 16 * 3, _r3);
                _mm512_storeu_ps(g0 + 16 * 4, _r4);
                _mm512_storeu_ps(g0 + 16 * 5, _r5);
                _mm512_storeu_ps(g0 + 16 * 6, _r6);
                _mm512_storeu_ps(g0 + 16 * 7, _r7);
                _mm512_storeu_ps(g0 + 16 * 8, _r8);
                _mm512_storeu_ps(g0 + 16 * 9, _r9);
                _mm512_storeu_ps(g0 + 16 * 10, _ra);
                _mm512_storeu_ps(g0 + 16 * 11, _rb);
                _mm512_storeu_ps(g0 + 16 * 12, _rc);
                _mm512_storeu_ps(g0 + 16 * 13, _rd);
                _mm512_storeu_ps(g0 + 16 * 14, _re);
                _mm512_storeu_ps(g0 + 16 * 15, _rf);
#endif

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
#if NCNN_IMPL_FP16S
                __m128i _r0 = _mm256_cvtps_ph(_mm256_loadu_ps(k0), _MM_FROUND_TRUNC);
                __m128i _r1 = _mm256_cvtps_ph(_mm256_loadu_ps(k1), _MM_FROUND_TRUNC);
                __m128i _r2 = _mm256_cvtps_ph(_mm256_loadu_ps(k2), _MM_FROUND_TRUNC);
                __m128i _r3 = _mm256_cvtps_ph(_mm256_loadu_ps(k3), _MM_FROUND_TRUNC);
                __m128i _r4 = _mm256_cvtps_ph(_mm256_loadu_ps(k4), _MM_FROUND_TRUNC);
                __m128i _r5 = _mm256_cvtps_ph(_mm256_loadu_ps(k5), _MM_FROUND_TRUNC);
                __m128i _r6 = _mm256_cvtps_ph(_mm256_loadu_ps(k6), _MM_FROUND_TRUNC);
                __m128i _r7 = _mm256_cvtps_ph(_mm256_loadu_ps(k7), _MM_FROUND_TRUNC);
                __m128i _r8 = _mm256_cvtps_ph(_mm256_loadu_ps(k8), _MM_FROUND_TRUNC);
                __m128i _r9 = _mm256_cvtps_ph(_mm256_loadu_ps(k9), _MM_FROUND_TRUNC);
                __m128i _ra = _mm256_cvtps_ph(_mm256_loadu_ps(ka), _MM_FROUND_TRUNC);
                __m128i _rb = _mm256_cvtps_ph(_mm256_loadu_ps(kb), _MM_FROUND_TRUNC);
                __m128i _rc = _mm256_cvtps_ph(_mm256_loadu_ps(kc), _MM_FROUND_TRUNC);
                __m128i _rd = _mm256_cvtps_ph(_mm256_loadu_ps(kd), _MM_FROUND_TRUNC);
                __m128i _re = _mm256_cvtps_ph(_mm256_loadu_ps(ke), _MM_FROUND_TRUNC);
                __m128i _rf = _mm256_cvtps_ph(_mm256_loadu_ps(kf), _MM_FROUND_TRUNC);

                __m256i _r08 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r0), _r8, 1);
                __m256i _r19 = _mm256_inserti128_si256(_mm256_castsi128_si256(_r1), _r9, 1);
                __m256i _r2a = _mm256_inserti128_si256(_mm256_castsi128_si256(_r2), _ra, 1);
                __m256i _r3b = _mm256_inserti128_si256(_mm256_castsi128_si256(_r3), _rb, 1);
                __m256i _r4c = _mm256_inserti128_si256(_mm256_castsi128_si256(_r4), _rc, 1);
                __m256i _r5d = _mm256_inserti128_si256(_mm256_castsi128_si256(_r5), _rd, 1);
                __m256i _r6e = _mm256_inserti128_si256(_mm256_castsi128_si256(_r6), _re, 1);
                __m256i _r7f = _mm256_inserti128_si256(_mm256_castsi128_si256(_r7), _rf, 1);

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
#else
                __m256 _r0 = _mm256_loadu_ps(k0);
                __m256 _r1 = _mm256_loadu_ps(k1);
                __m256 _r2 = _mm256_loadu_ps(k2);
                __m256 _r3 = _mm256_loadu_ps(k3);
                __m256 _r4 = _mm256_loadu_ps(k4);
                __m256 _r5 = _mm256_loadu_ps(k5);
                __m256 _r6 = _mm256_loadu_ps(k6);
                __m256 _r7 = _mm256_loadu_ps(k7);
                __m256 _r8 = _mm256_loadu_ps(k8);
                __m256 _r9 = _mm256_loadu_ps(k9);
                __m256 _ra = _mm256_loadu_ps(ka);
                __m256 _rb = _mm256_loadu_ps(kb);
                __m256 _rc = _mm256_loadu_ps(kc);
                __m256 _rd = _mm256_loadu_ps(kd);
                __m256 _re = _mm256_loadu_ps(ke);
                __m256 _rf = _mm256_loadu_ps(kf);

                transpose8x16_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, _r8, _r9, _ra, _rb, _rc, _rd, _re, _rf);

                _mm256_storeu_ps(g0, _r0);
                _mm256_storeu_ps(g0 + 8, _r1);
                _mm256_storeu_ps(g0 + 8 * 2, _r2);
                _mm256_storeu_ps(g0 + 8 * 3, _r3);
                _mm256_storeu_ps(g0 + 8 * 4, _r4);
                _mm256_storeu_ps(g0 + 8 * 5, _r5);
                _mm256_storeu_ps(g0 + 8 * 6, _r6);
                _mm256_storeu_ps(g0 + 8 * 7, _r7);
                _mm256_storeu_ps(g0 + 8 * 8, _r8);
                _mm256_storeu_ps(g0 + 8 * 9, _r9);
                _mm256_storeu_ps(g0 + 8 * 10, _ra);
                _mm256_storeu_ps(g0 + 8 * 11, _rb);
                _mm256_storeu_ps(g0 + 8 * 12, _rc);
                _mm256_storeu_ps(g0 + 8 * 13, _rd);
                _mm256_storeu_ps(g0 + 8 * 14, _re);
                _mm256_storeu_ps(g0 + 8 * 15, _rf);
#endif

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
#if NCNN_IMPL_FP16S
                g0[0] = float32_to_float16(*k0++);
                g0[1] = float32_to_float16(*k1++);
                g0[2] = float32_to_float16(*k2++);
                g0[3] = float32_to_float16(*k3++);
                g0[4] = float32_to_float16(*k4++);
                g0[5] = float32_to_float16(*k5++);
                g0[6] = float32_to_float16(*k6++);
                g0[7] = float32_to_float16(*k7++);
                g0[8] = float32_to_float16(*k8++);
                g0[9] = float32_to_float16(*k9++);
                g0[10] = float32_to_float16(*ka++);
                g0[11] = float32_to_float16(*kb++);
                g0[12] = float32_to_float16(*kc++);
                g0[13] = float32_to_float16(*kd++);
                g0[14] = float32_to_float16(*ke++);
                g0[15] = float32_to_float16(*kf++);
#else
                g0[0] = *k0++;
                g0[1] = *k1++;
                g0[2] = *k2++;
                g0[3] = *k3++;
                g0[4] = *k4++;
                g0[5] = *k5++;
                g0[6] = *k6++;
                g0[7] = *k7++;
                g0[8] = *k8++;
                g0[9] = *k9++;
                g0[10] = *ka++;
                g0[11] = *kb++;
                g0[12] = *kc++;
                g0[13] = *kd++;
                g0[14] = *ke++;
                g0[15] = *kf++;
#endif
                g0 += 16;
            }
        }
    }
#endif // __AVX512F__

    if (out_elempack == 8)
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

#if NCNN_IMPL_FP16S
        weight_data_tm.create(num_input, num_output / 8, (size_t)16u, 8);
#else
        weight_data_tm.create(num_input, num_output / 8, (size_t)32u, 8);
#endif

        for (int q = 0; q + 7 < num_output; q += 8)
        {
#if NCNN_IMPL_FP16S
            unsigned short* g0 = weight_data_tm.row<unsigned short>(q / 8);
#else
            float* g0 = weight_data_tm.row(q / 8);
#endif

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
#if NCNN_IMPL_FP16S
                __m256i _r0 = _mm512_cvtps_ph(_mm512_loadu_ps(k0), _MM_FROUND_TRUNC);
                __m256i _r1 = _mm512_cvtps_ph(_mm512_loadu_ps(k1), _MM_FROUND_TRUNC);
                __m256i _r2 = _mm512_cvtps_ph(_mm512_loadu_ps(k2), _MM_FROUND_TRUNC);
                __m256i _r3 = _mm512_cvtps_ph(_mm512_loadu_ps(k3), _MM_FROUND_TRUNC);
                __m256i _r4 = _mm512_cvtps_ph(_mm512_loadu_ps(k4), _MM_FROUND_TRUNC);
                __m256i _r5 = _mm512_cvtps_ph(_mm512_loadu_ps(k5), _MM_FROUND_TRUNC);
                __m256i _r6 = _mm512_cvtps_ph(_mm512_loadu_ps(k6), _MM_FROUND_TRUNC);
                __m256i _r7 = _mm512_cvtps_ph(_mm512_loadu_ps(k7), _MM_FROUND_TRUNC);

                transpose16x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                _mm256_storeu_si256((__m256i*)g0, _r0);
                _mm256_storeu_si256((__m256i*)(g0 + 16), _r1);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 2), _r2);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 3), _r3);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 4), _r4);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 5), _r5);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 6), _r6);
                _mm256_storeu_si256((__m256i*)(g0 + 16 * 7), _r7);
#else
                __m512 _r0 = _mm512_loadu_ps(k0);
                __m512 _r1 = _mm512_loadu_ps(k1);
                __m512 _r2 = _mm512_loadu_ps(k2);
                __m512 _r3 = _mm512_loadu_ps(k3);
                __m512 _r4 = _mm512_loadu_ps(k4);
                __m512 _r5 = _mm512_loadu_ps(k5);
                __m512 _r6 = _mm512_loadu_ps(k6);
                __m512 _r7 = _mm512_loadu_ps(k7);

                transpose16x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                _mm512_storeu_ps(g0, _r0);
                _mm512_storeu_ps(g0 + 16, _r1);
                _mm512_storeu_ps(g0 + 16 * 2, _r2);
                _mm512_storeu_ps(g0 + 16 * 3, _r3);
                _mm512_storeu_ps(g0 + 16 * 4, _r4);
                _mm512_storeu_ps(g0 + 16 * 5, _r5);
                _mm512_storeu_ps(g0 + 16 * 6, _r6);
                _mm512_storeu_ps(g0 + 16 * 7, _r7);
#endif

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
#if NCNN_IMPL_FP16S
                __m128i _r0 = _mm256_cvtps_ph(_mm256_loadu_ps(k0), _MM_FROUND_TRUNC);
                __m128i _r1 = _mm256_cvtps_ph(_mm256_loadu_ps(k1), _MM_FROUND_TRUNC);
                __m128i _r2 = _mm256_cvtps_ph(_mm256_loadu_ps(k2), _MM_FROUND_TRUNC);
                __m128i _r3 = _mm256_cvtps_ph(_mm256_loadu_ps(k3), _MM_FROUND_TRUNC);
                __m128i _r4 = _mm256_cvtps_ph(_mm256_loadu_ps(k4), _MM_FROUND_TRUNC);
                __m128i _r5 = _mm256_cvtps_ph(_mm256_loadu_ps(k5), _MM_FROUND_TRUNC);
                __m128i _r6 = _mm256_cvtps_ph(_mm256_loadu_ps(k6), _MM_FROUND_TRUNC);
                __m128i _r7 = _mm256_cvtps_ph(_mm256_loadu_ps(k7), _MM_FROUND_TRUNC);

                transpose8x8_epi16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                _mm_storeu_si128((__m128i*)g0, _r0);
                _mm_storeu_si128((__m128i*)(g0 + 8), _r1);
                _mm_storeu_si128((__m128i*)(g0 + 16), _r2);
                _mm_storeu_si128((__m128i*)(g0 + 24), _r3);
                _mm_storeu_si128((__m128i*)(g0 + 32), _r4);
                _mm_storeu_si128((__m128i*)(g0 + 40), _r5);
                _mm_storeu_si128((__m128i*)(g0 + 48), _r6);
                _mm_storeu_si128((__m128i*)(g0 + 56), _r7);
#else
                __m256 _r0 = _mm256_loadu_ps(k0);
                __m256 _r1 = _mm256_loadu_ps(k1);
                __m256 _r2 = _mm256_loadu_ps(k2);
                __m256 _r3 = _mm256_loadu_ps(k3);
                __m256 _r4 = _mm256_loadu_ps(k4);
                __m256 _r5 = _mm256_loadu_ps(k5);
                __m256 _r6 = _mm256_loadu_ps(k6);
                __m256 _r7 = _mm256_loadu_ps(k7);

                transpose8x8_ps(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);

                _mm256_storeu_ps(g0, _r0);
                _mm256_storeu_ps(g0 + 8, _r1);
                _mm256_storeu_ps(g0 + 16, _r2);
                _mm256_storeu_ps(g0 + 24, _r3);
                _mm256_storeu_ps(g0 + 32, _r4);
                _mm256_storeu_ps(g0 + 40, _r5);
                _mm256_storeu_ps(g0 + 48, _r6);
                _mm256_storeu_ps(g0 + 56, _r7);
#endif

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
#if NCNN_IMPL_FP16S
                g0[0] = float32_to_float16(*k0++);
                g0[1] = float32_to_float16(*k1++);
                g0[2] = float32_to_float16(*k2++);
                g0[3] = float32_to_float16(*k3++);
                g0[4] = float32_to_float16(*k4++);
                g0[5] = float32_to_float16(*k5++);
                g0[6] = float32_to_float16(*k6++);
                g0[7] = float32_to_float16(*k7++);
#else
                g0[0] = *k0++;
                g0[1] = *k1++;
                g0[2] = *k2++;
                g0[3] = *k3++;
                g0[4] = *k4++;
                g0[5] = *k5++;
                g0[6] = *k6++;
                g0[7] = *k7++;
#endif
                g0 += 8;
            }
        }
    }
#endif // __AVX__

    if (out_elempack == 4)
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

#if NCNN_IMPL_FP16S
        weight_data_tm.create(num_input, num_output / 4, (size_t)8u, 4);
#else
        weight_data_tm.create(num_input, num_output / 4, (size_t)16u, 4);
#endif

        for (int q = 0; q + 3 < num_output; q += 4)
        {
#if NCNN_IMPL_FP16S
            unsigned short* g0 = weight_data_tm.row<unsigned short>(q / 4);
#else
            float* g0 = weight_data_tm.row(q / 4);
#endif

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
#if NCNN_IMPL_FP16S
                __m256 _r01 = _mm256_insertf128_ps(_mm256_castps128_ps256(_r0), _r1, 1);
                __m256 _r23 = _mm256_insertf128_ps(_mm256_castps128_ps256(_r2), _r3, 1);
                __m128i _r01_fp16 = _mm256_cvtps_ph(_r01, _MM_FROUND_TRUNC);
                __m128i _r23_fp16 = _mm256_cvtps_ph(_r23, _MM_FROUND_TRUNC);
                _mm_storeu_si128((__m128i*)g0, _r01_fp16);
                _mm_storeu_si128((__m128i*)(g0 + 8), _r23_fp16);
#else
                _mm_storeu_ps(g0, _r0);
                _mm_storeu_ps(g0 + 4, _r1);
                _mm_storeu_ps(g0 + 8, _r2);
                _mm_storeu_ps(g0 + 12, _r3);
#endif

                k0 += 4;
                k1 += 4;
                k2 += 4;
                k3 += 4;
                g0 += 16;
            }
            for (; p < num_input; p++)
            {
#if NCNN_IMPL_FP16S
                g0[0] = float32_to_float16(*k0++);
                g0[1] = float32_to_float16(*k1++);
                g0[2] = float32_to_float16(*k2++);
                g0[3] = float32_to_float16(*k3++);
#else
                g0[0] = *k0++;
                g0[1] = *k1++;
                g0[2] = *k2++;
                g0[3] = *k3++;
#endif
                g0 += 4;
            }
        }
    }
#endif // __SSE2__

    if (out_elempack == 1)
    {
#if NCNN_IMPL_FP16S
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);
        ncnn::cast_float32_to_float16(weight_data_r2, weight_data_tm, opt);
#else
        weight_data_tm = weight_data;
#endif
    }
#endif // NCNN_RUNTIME_CPU
}
