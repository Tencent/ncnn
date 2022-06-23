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

#if __AVX512F__
static void innerproduct_fp16s_pack16_avx512(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int num_input = bottom_blob.w * bottom_blob.elempack;
    const int num_output = top_blob.w;

    const float* bias_data_ptr = bias_data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < num_output; p++)
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

        const unsigned short* kptr = weight_data_fp16.row<const unsigned short>(p);

        const float* sptr = bottom_blob;

        int i = 0;
        for (; i + 7 < num_input; i += 8)
        {
            __m512i _w01 = _mm512_loadu_si512(kptr);
            __m512 _val0 = _mm512_set1_ps(sptr[0]);
            __m512 _w0 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w01, 0));
            _sum0 = _mm512_fmadd_ps(_val0, _w0, _sum0);

            __m512 _val1 = _mm512_set1_ps(sptr[1]);
            __m512 _w1 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w01, 1));
            _sum1 = _mm512_fmadd_ps(_val1, _w1, _sum1);

            __m512i _w23 = _mm512_loadu_si512(kptr + 32);
            __m512 _val2 = _mm512_set1_ps(sptr[2]);
            __m512 _w2 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w23, 0));
            _sum2 = _mm512_fmadd_ps(_val2, _w2, _sum2);

            __m512 _val3 = _mm512_set1_ps(sptr[3]);
            __m512 _w3 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w23, 1));
            _sum3 = _mm512_fmadd_ps(_val3, _w3, _sum3);

            __m512i _w45 = _mm512_loadu_si512(kptr + 64);
            __m512 _val4 = _mm512_set1_ps(sptr[4]);
            __m512 _w4 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w45, 0));
            _sum4 = _mm512_fmadd_ps(_val4, _w4, _sum4);

            __m512 _val5 = _mm512_set1_ps(sptr[5]);
            __m512 _w5 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w45, 1));
            _sum5 = _mm512_fmadd_ps(_val5, _w5, _sum5);

            __m512i _w67 = _mm512_loadu_si512(kptr + 96);
            __m512 _val6 = _mm512_set1_ps(sptr[6]);
            __m512 _w6 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w67, 0));
            _sum6 = _mm512_fmadd_ps(_val6, _w6, _sum6);

            __m512 _val7 = _mm512_set1_ps(sptr[7]);
            __m512 _w7 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w67, 1));
            _sum7 = _mm512_fmadd_ps(_val7, _w7, _sum7);

            sptr += 8;
            kptr += 128;
        }
        for (; i + 3 < num_input; i += 4)
        {
            __m512i _w01 = _mm512_loadu_si512(kptr);
            __m512 _val0 = _mm512_set1_ps(sptr[0]);
            __m512 _w0 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w01, 0));
            _sum0 = _mm512_fmadd_ps(_val0, _w0, _sum0);

            __m512 _val1 = _mm512_set1_ps(sptr[1]);
            __m512 _w1 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w01, 1));
            _sum1 = _mm512_fmadd_ps(_val1, _w1, _sum1);

            __m512i _w23 = _mm512_loadu_si512(kptr + 32);
            __m512 _val2 = _mm512_set1_ps(sptr[2]);
            __m512 _w2 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w23, 0));
            _sum2 = _mm512_fmadd_ps(_val2, _w2, _sum2);

            __m512 _val3 = _mm512_set1_ps(sptr[3]);
            __m512 _w3 = _mm512_cvtph_ps(_mm512_extracti32x8_epi32(_w23, 1));
            _sum3 = _mm512_fmadd_ps(_val3, _w3, _sum3);

            sptr += 4;
            kptr += 64;
        }
        for (; i < num_input; i++)
        {
            __m512 _val = _mm512_set1_ps(sptr[0]);
            __m512 _w = _mm512_cvtph_ps(_mm256_lddqu_si256((const __m256i*)kptr));
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

#if NCNN_RUNTIME_CPU && NCNN_F16C && __AVX__ && !__F16C__
void innerproduct_fp16s_pack8_avx_f16c(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
void innerproduct_fp16s_pack4_sse_f16c(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
void innerproduct_fp16s_sse_f16c(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt);
void innerproduct_transform_kernel_fp16s_sse_f16c(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt);
#endif

static void innerproduct_fp16s_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_F16C && __AVX__ && !__F16C__
    if (ncnn::cpu_support_x86_f16c())
    {
        innerproduct_fp16s_pack8_avx_f16c(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
        return;
    }
#endif

#if __F16C__
    const int num_input = bottom_blob.w * bottom_blob.elempack;
    const int num_output = top_blob.w;

    const float* bias_data_ptr = bias_data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < num_output; p++)
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

        const unsigned short* kptr = weight_data_fp16.row<const unsigned short>(p);

        const float* sptr = bottom_blob;

        int i = 0;
        for (; i + 7 < num_input; i += 8)
        {
            __m256i _w01 = _mm256_lddqu_si256((const __m256i*)kptr);
            __m256 _val0 = _mm256_broadcast_ss(sptr);
            __m128i _w0_fp16 = _mm256_extractf128_si256(_w01, 0);
            __m256 _w0 = _mm256_cvtph_ps(_w0_fp16);
            _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);

            __m256 _val1 = _mm256_broadcast_ss(sptr + 1);
            __m128i _w1_fp16 = _mm256_extractf128_si256(_w01, 1);
            __m256 _w1 = _mm256_cvtph_ps(_w1_fp16);
            _sum1 = _mm256_comp_fmadd_ps(_val1, _w1, _sum1);

            __m256i _w23 = _mm256_lddqu_si256((const __m256i*)(kptr + 16));
            __m256 _val2 = _mm256_broadcast_ss(sptr + 2);
            __m128i _w2_fp16 = _mm256_extractf128_si256(_w23, 0);
            __m256 _w2 = _mm256_cvtph_ps(_w2_fp16);
            _sum2 = _mm256_comp_fmadd_ps(_val2, _w2, _sum2);

            __m256 _val3 = _mm256_broadcast_ss(sptr + 3);
            __m128i _w3_fp16 = _mm256_extractf128_si256(_w23, 1);
            __m256 _w3 = _mm256_cvtph_ps(_w3_fp16);
            _sum3 = _mm256_comp_fmadd_ps(_val3, _w3, _sum3);

            __m256i _w45 = _mm256_lddqu_si256((const __m256i*)(kptr + 32));
            __m256 _val4 = _mm256_broadcast_ss(sptr + 4);
            __m128i _w4_fp16 = _mm256_extractf128_si256(_w45, 0);
            __m256 _w4 = _mm256_cvtph_ps(_w4_fp16);
            _sum4 = _mm256_comp_fmadd_ps(_val4, _w4, _sum4);

            __m256 _val5 = _mm256_broadcast_ss(sptr + 5);
            __m128i _w5_fp16 = _mm256_extractf128_si256(_w45, 1);
            __m256 _w5 = _mm256_cvtph_ps(_w5_fp16);
            _sum5 = _mm256_comp_fmadd_ps(_val5, _w5, _sum5);

            __m256i _w67 = _mm256_lddqu_si256((const __m256i*)(kptr + 48));
            __m256 _val6 = _mm256_broadcast_ss(sptr + 6);
            __m128i _w6_fp16 = _mm256_extractf128_si256(_w67, 0);
            __m256 _w6 = _mm256_cvtph_ps(_w6_fp16);
            _sum6 = _mm256_comp_fmadd_ps(_val6, _w6, _sum6);

            __m256 _val7 = _mm256_broadcast_ss(sptr + 7);
            __m128i _w7_fp16 = _mm256_extractf128_si256(_w67, 1);
            __m256 _w7 = _mm256_cvtph_ps(_w7_fp16);
            _sum7 = _mm256_comp_fmadd_ps(_val7, _w7, _sum7);

            sptr += 8;
            kptr += 64;
        }
        for (; i + 3 < num_input; i += 4)
        {
            __m256i _w01 = _mm256_lddqu_si256((const __m256i*)kptr);
            __m256 _val0 = _mm256_broadcast_ss(sptr);
            __m256 _w0 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w01, 0));
            _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);

            __m256 _val1 = _mm256_broadcast_ss(sptr + 1);
            __m256 _w1 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w01, 1));
            _sum1 = _mm256_comp_fmadd_ps(_val1, _w1, _sum1);

            __m256i _w23 = _mm256_lddqu_si256((const __m256i*)(kptr + 16));
            __m256 _val2 = _mm256_broadcast_ss(sptr + 2);
            __m256 _w2 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w23, 0));
            _sum2 = _mm256_comp_fmadd_ps(_val2, _w2, _sum2);

            __m256 _val3 = _mm256_broadcast_ss(sptr + 3);
            __m256 _w3 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w23, 1));
            _sum3 = _mm256_comp_fmadd_ps(_val3, _w3, _sum3);

            sptr += 4;
            kptr += 32;
        }
        for (; i < num_input; i++)
        {
            __m256 _val = _mm256_set1_ps(sptr[0]);
            __m256 _w = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)kptr));
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
#else  // __F16C__
    (void)bottom_blob;
    (void)top_blob;
    (void)weight_data_fp16;
    (void)bias_data;
    (void)activation_type;
    (void)activation_params;
    (void)opt;
#endif // __F16C__
}

static void innerproduct_fp16s_pack4_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_F16C && __AVX__ && !__F16C__
    if (ncnn::cpu_support_x86_f16c())
    {
        innerproduct_fp16s_pack4_sse_f16c(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
        return;
    }
#endif

#if __F16C__
    const int num_input = bottom_blob.w * bottom_blob.elempack;
    const int num_output = top_blob.w;

    const float* bias_data_ptr = bias_data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < num_output; p++)
    {
        __m128 _sum0 = _mm_setzero_ps();

        __m256 _sum01 = _mm256_setzero_ps();
        __m256 _sum23 = _mm256_setzero_ps();
        __m256 _sum45 = _mm256_setzero_ps();
        __m256 _sum67 = _mm256_setzero_ps();

        if (bias_data_ptr)
        {
            _sum0 = _mm_loadu_ps(bias_data_ptr + p * 4);
        }

        const unsigned short* kptr = weight_data_fp16.row<const unsigned short>(p);

        const float* sptr = bottom_blob;

        int i = 0;
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

            __m256i _w0123 = _mm256_lddqu_si256((const __m256i*)kptr);
            __m256i _w4567 = _mm256_lddqu_si256((const __m256i*)(kptr + 16));

            __m256 _w01 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w0123, 0));
            __m256 _w23 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w0123, 1));
            __m256 _w45 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w4567, 0));
            __m256 _w67 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w4567, 1));

            _sum01 = _mm256_comp_fmadd_ps(_val01, _w01, _sum01);
            _sum23 = _mm256_comp_fmadd_ps(_val23, _w23, _sum23);
            _sum45 = _mm256_comp_fmadd_ps(_val45, _w45, _sum45);
            _sum67 = _mm256_comp_fmadd_ps(_val67, _w67, _sum67);

            sptr += 8;
            kptr += 32;
        }
        for (; i + 3 < num_input; i += 4)
        {
            __m128 _val0 = _mm_set1_ps(sptr[0]);
            __m128 _val1 = _mm_set1_ps(sptr[1]);
            __m128 _val2 = _mm_set1_ps(sptr[2]);
            __m128 _val3 = _mm_set1_ps(sptr[3]);

            __m256 _val01 = _mm256_insertf128_ps(_mm256_castps128_ps256(_val0), _val1, 1);
            __m256 _val23 = _mm256_insertf128_ps(_mm256_castps128_ps256(_val2), _val3, 1);

            __m256i _w0123 = _mm256_lddqu_si256((const __m256i*)kptr);
            __m256 _w01 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w0123, 0));
            __m256 _w23 = _mm256_cvtph_ps(_mm256_extractf128_si256(_w0123, 1));

            _sum01 = _mm256_comp_fmadd_ps(_val01, _w01, _sum01);
            _sum23 = _mm256_comp_fmadd_ps(_val23, _w23, _sum23);

            sptr += 4;
            kptr += 16;
        }
        for (; i < num_input; i++)
        {
            __m128 _val = _mm_set1_ps(sptr[0]);
            __m128 _w = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)kptr));
            _sum0 = _mm_comp_fmadd_ps(_val, _w, _sum0);

            sptr += 1;
            kptr += 4;
        }

        _sum01 = _mm256_add_ps(_sum01, _sum23);
        _sum45 = _mm256_add_ps(_sum45, _sum67);
        _sum01 = _mm256_add_ps(_sum01, _sum45);

        _sum0 = _mm_add_ps(_sum0, _mm256_extractf128_ps(_sum01, 0));
        _sum0 = _mm_add_ps(_sum0, _mm256_extractf128_ps(_sum01, 1));

        _sum0 = activation_sse(_sum0, activation_type, activation_params);

        float* outptr = top_blob;
        _mm_storeu_ps(outptr + p * 4, _sum0);
    }
#else  // __F16C__
    (void)bottom_blob;
    (void)top_blob;
    (void)weight_data_fp16;
    (void)bias_data;
    (void)activation_type;
    (void)activation_params;
    (void)opt;
#endif // __F16C__
}

static void innerproduct_fp16s_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_F16C && __AVX__ && !__F16C__
    if (ncnn::cpu_support_x86_f16c())
    {
        innerproduct_fp16s_sse_f16c(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
        return;
    }
#endif

#if __F16C__
    const int num_input = bottom_blob.w * bottom_blob.elempack;
    const int num_output = top_blob.w;

    const float* bias_data_ptr = bias_data;

    int remain_num_output_start = 0;
    int nn_num_output = num_output >> 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_num_output; pp++)
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

        const unsigned short* w0 = weight_data_fp16.row<const unsigned short>(p);
        const unsigned short* w1 = weight_data_fp16.row<const unsigned short>(p + 1);
        const unsigned short* w2 = weight_data_fp16.row<const unsigned short>(p + 2);
        const unsigned short* w3 = weight_data_fp16.row<const unsigned short>(p + 3);
        const unsigned short* w4 = weight_data_fp16.row<const unsigned short>(p + 4);
        const unsigned short* w5 = weight_data_fp16.row<const unsigned short>(p + 5);
        const unsigned short* w6 = weight_data_fp16.row<const unsigned short>(p + 6);
        const unsigned short* w7 = weight_data_fp16.row<const unsigned short>(p + 7);

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

            __m256 _w0 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w0));
            __m256 _w1 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w1));
            __m256 _w2 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w2));
            __m256 _w3 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w3));
            __m256 _w4 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w4));
            __m256 _w5 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w5));
            __m256 _w6 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w6));
            __m256 _w7 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w7));

            _sum0 = _mm256_comp_fmadd_ps(_m, _w0, _sum0);
            _sum1 = _mm256_comp_fmadd_ps(_m, _w1, _sum1);
            _sum2 = _mm256_comp_fmadd_ps(_m, _w2, _sum2);
            _sum3 = _mm256_comp_fmadd_ps(_m, _w3, _sum3);
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
            sums[0] += *m * float16_to_float32(*w0);
            sums[1] += *m * float16_to_float32(*w1);
            sums[2] += *m * float16_to_float32(*w2);
            sums[3] += *m * float16_to_float32(*w3);
            sums[4] += *m * float16_to_float32(*w4);
            sums[5] += *m * float16_to_float32(*w5);
            sums[6] += *m * float16_to_float32(*w6);
            sums[7] += *m * float16_to_float32(*w7);

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

    remain_num_output_start += (nn_num_output << 3);
    nn_num_output = (num_output - remain_num_output_start) >> 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_num_output; pp++)
    {
        int p = remain_num_output_start + (pp * 4);

        float sums[4] = {0.0f};
        if (bias_data_ptr)
        {
            sums[0] = bias_data_ptr[p];
            sums[1] = bias_data_ptr[p + 1];
            sums[2] = bias_data_ptr[p + 2];
            sums[3] = bias_data_ptr[p + 3];
        }

        const unsigned short* w0 = weight_data_fp16.row<const unsigned short>(p);
        const unsigned short* w1 = weight_data_fp16.row<const unsigned short>(p + 1);
        const unsigned short* w2 = weight_data_fp16.row<const unsigned short>(p + 2);
        const unsigned short* w3 = weight_data_fp16.row<const unsigned short>(p + 3);

        const float* m = bottom_blob;

        int i = 0;

        __m256 _sum0 = _mm256_setzero_ps();
        __m256 _sum1 = _mm256_setzero_ps();
        __m256 _sum2 = _mm256_setzero_ps();
        __m256 _sum3 = _mm256_setzero_ps();
        for (; i + 7 < num_input; i += 8)
        {
            __m256 _m = _mm256_loadu_ps(m);

            __m256 _w0 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w0));
            __m256 _w1 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w1));
            __m256 _w2 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w2));
            __m256 _w3 = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w3));

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

        __m128 _sum0l = _mm_setzero_ps();
        __m128 _sum1l = _mm_setzero_ps();
        __m128 _sum2l = _mm_setzero_ps();
        __m128 _sum3l = _mm_setzero_ps();
        for (; i + 3 < num_input; i += 4)
        {
            __m128 _m = _mm_loadu_ps(m);

            __m128 _w0 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)w0));
            __m128 _w1 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)w1));
            __m128 _w2 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)w2));
            __m128 _w3 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)w3));

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
            sums[0] += *m * float16_to_float32(*w0);
            sums[1] += *m * float16_to_float32(*w1);
            sums[2] += *m * float16_to_float32(*w2);
            sums[3] += *m * float16_to_float32(*w3);

            m++;
            w0++;
            w1++;
            w2++;
            w3++;
        }

        __m128 _sums = _mm_loadu_ps(sums);

        _sums = _mm_add_ps(HorizontalSums(_sum0, _sum1, _sum2, _sum3), _sums);

        _MM_TRANSPOSE4_PS(_sum0l, _sum1l, _sum2l, _sum3l);
        _sums = _mm_add_ps(_sum0l, _sums);
        _sums = _mm_add_ps(_sum1l, _sums);
        _sums = _mm_add_ps(_sum2l, _sums);
        _sums = _mm_add_ps(_sum3l, _sums);
        _sums = activation_sse(_sums, activation_type, activation_params);

        float* outptr = top_blob;
        _mm_storeu_ps(outptr + p, _sums);
    }

    remain_num_output_start += (nn_num_output << 2);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_num_output_start; p < num_output; p++)
    {
        float sum = 0.f;

        if (bias_data_ptr)
            sum = bias_data_ptr[p];

        const unsigned short* w = weight_data_fp16.row<const unsigned short>(p);

        const float* m = bottom_blob;

        int i = 0;

        __m256 _sum = _mm256_setzero_ps();
        for (; i + 7 < num_input; i += 8)
        {
            __m256 _m = _mm256_loadu_ps(m);
            __m256 _w = _mm256_cvtph_ps(_mm_lddqu_si128((const __m128i*)w));
            _sum = _mm256_comp_fmadd_ps(_m, _w, _sum);

            m += 8;
            w += 8;
        }

        __m128 _suml = _mm_setzero_ps();
        for (; i + 3 < num_input; i += 4)
        {
            __m128 _m = _mm_loadu_ps(m);
            __m128 _w = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)w));
            _suml = _mm_comp_fmadd_ps(_m, _w, _suml);

            m += 4;
            w += 4;
        }
        for (; i < num_input; i++)
        {
            sum += *m * float16_to_float32(*w);
            m++;
            w++;
        }

        sum += _mm256_reduce_add_ps(_sum);

        sum += _mm_reduce_add_ps(_suml);

        sum = activation_ss(sum, activation_type, activation_params);

        float* outptr = top_blob;
        outptr[p] = sum;
    }
#else  // __F16C__
    (void)bottom_blob;
    (void)top_blob;
    (void)weight_data_fp16;
    (void)bias_data;
    (void)activation_type;
    (void)activation_params;
    (void)opt;
#endif // __F16C__
}

static void innerproduct_transform_kernel_fp16s_sse(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_F16C && __AVX__ && !__F16C__
    if (ncnn::cpu_support_x86_f16c())
    {
        innerproduct_transform_kernel_fp16s_sse_f16c(weight_data, weight_data_tm, num_input, num_output, opt);
        return;
    }
#endif

#if __F16C__
    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        out_elempack = num_output % 16 == 0 ? 16 : num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#endif
    }

    Mat weight_data_fp16;
    ncnn::cast_float32_to_float16(weight_data, weight_data_fp16, opt);

    // src = inch-outch
    // dst = pb-inch-outch/pb
    {
        Mat weight_data_r2 = weight_data_fp16.reshape(num_input, num_output);

        weight_data_tm.create(num_input, num_output / out_elempack, (size_t)2u * out_elempack, out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            unsigned short* g0 = weight_data_tm.row<unsigned short>(q / out_elempack);

            for (int p = 0; p < num_input; p++)
            {
                for (int j = 0; j < out_elempack; j++)
                {
                    *g0++ = weight_data_r2.row<const unsigned short>(q + j)[p];
                }
            }
        }
    }
#else  // __F16C__
    (void)weight_data;
    (void)weight_data_tm;
    (void)num_input;
    (void)num_output;
    (void)opt;
#endif // __F16C__
}
