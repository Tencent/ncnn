// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "innerproduct_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif // __SSE2__

#include "x86_activation.h"
#include "x86_usability.h"

#include "layer_type.h"

#include "cpu.h"

namespace ncnn {

#if NCNN_F16C
#include "innerproduct_fp16s.h"
#include "innerproduct_gemm_fp16s.h"
#endif

InnerProduct_x86::InnerProduct_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__

    flatten = 0;
}

int InnerProduct_x86::create_pipeline(const Option& opt)
{
    //     if (opt.use_packing_layout)
    {
        flatten = ncnn::create_layer(ncnn::LayerType::Flatten);

        ncnn::ParamDict pd;

        flatten->load_param(pd);

        flatten->create_pipeline(opt);
    }

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return create_pipeline_int8_x86(opt);
    }
#endif

#if NCNN_F16C
    if (cpu_support_x86_f16c() && opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

    const int num_input = weight_data_size / num_output;

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

    if (out_elempack != 1)
    {
        // src = inch-outch
        // dst = pb-inch-outch/pb
        {
            Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

            weight_data_tm.create(num_input, num_output / out_elempack, (size_t)4u * out_elempack, out_elempack);

            for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
            {
                float* g0 = weight_data_tm.row(q / out_elempack);

                for (int p = 0; p < num_input; p++)
                {
                    for (int j = 0; j < out_elempack; j++)
                    {
                        *g0++ = weight_data_r2.row(q + j)[p];
                    }
                }
            }
        }
    }
    else
    {
        weight_data_tm = weight_data;
    }

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}

int InnerProduct_x86::destroy_pipeline(const Option& opt)
{
    if (flatten)
    {
        flatten->destroy_pipeline(opt);
        delete flatten;
        flatten = 0;
    }

    return 0;
}

int InnerProduct_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && int8_scale_term)
    {
        return forward_int8_x86(bottom_blob, top_blob, opt);
    }
#endif

#if NCNN_F16C
    if (cpu_support_x86_f16c() && opt.use_fp16_storage)
    {
        return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input && bottom_blob.h * bottom_blob.elempack > 1)
    {
        // gemm
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

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
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    __m512 _sum0 = _mm512_set1_ps(0.f);
                    __m512 _sum1 = _mm512_set1_ps(0.f);
                    __m512 _sum2 = _mm512_set1_ps(0.f);
                    __m512 _sum3 = _mm512_set1_ps(0.f);
                    __m512 _sum4 = _mm512_set1_ps(0.f);
                    __m512 _sum5 = _mm512_set1_ps(0.f);
                    __m512 _sum6 = _mm512_set1_ps(0.f);
                    __m512 _sum7 = _mm512_set1_ps(0.f);
                    __m512 _sum8 = _mm512_set1_ps(0.f);
                    __m512 _sum9 = _mm512_set1_ps(0.f);
                    __m512 _suma = _mm512_set1_ps(0.f);
                    __m512 _sumb = _mm512_set1_ps(0.f);
                    __m512 _sumc = _mm512_set1_ps(0.f);
                    __m512 _sumd = _mm512_set1_ps(0.f);
                    __m512 _sume = _mm512_set1_ps(0.f);
                    __m512 _sumf = _mm512_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum0 = _mm512_set1_ps(bias_data[p * 16 + 0]);
                        _sum1 = _mm512_set1_ps(bias_data[p * 16 + 1]);
                        _sum2 = _mm512_set1_ps(bias_data[p * 16 + 2]);
                        _sum3 = _mm512_set1_ps(bias_data[p * 16 + 3]);
                        _sum4 = _mm512_set1_ps(bias_data[p * 16 + 4]);
                        _sum5 = _mm512_set1_ps(bias_data[p * 16 + 5]);
                        _sum6 = _mm512_set1_ps(bias_data[p * 16 + 6]);
                        _sum7 = _mm512_set1_ps(bias_data[p * 16 + 7]);
                        _sum8 = _mm512_set1_ps(bias_data[p * 16 + 8]);
                        _sum9 = _mm512_set1_ps(bias_data[p * 16 + 9]);
                        _suma = _mm512_set1_ps(bias_data[p * 16 + 10]);
                        _sumb = _mm512_set1_ps(bias_data[p * 16 + 11]);
                        _sumc = _mm512_set1_ps(bias_data[p * 16 + 12]);
                        _sumd = _mm512_set1_ps(bias_data[p * 16 + 13]);
                        _sume = _mm512_set1_ps(bias_data[p * 16 + 14]);
                        _sumf = _mm512_set1_ps(bias_data[p * 16 + 15]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        __m512 _val = _mm512_loadu_ps(m);
                        _sum0 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[3]), _sum3);
                        _sum4 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[4]), _sum4);
                        _sum5 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[5]), _sum5);
                        _sum6 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[6]), _sum6);
                        _sum7 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[7]), _sum7);
                        _sum8 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[8]), _sum8);
                        _sum9 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[9]), _sum9);
                        _suma = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[10]), _suma);
                        _sumb = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[11]), _sumb);
                        _sumc = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[12]), _sumc);
                        _sumd = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[13]), _sumd);
                        _sume = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[14]), _sume);
                        _sumf = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[15]), _sumf);

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

                    _mm512_storeu_ps(outptr, _sum0);
                    _mm512_storeu_ps(outptr + 16, _sum1);
                    _mm512_storeu_ps(outptr + 16 * 2, _sum2);
                    _mm512_storeu_ps(outptr + 16 * 3, _sum3);
                    _mm512_storeu_ps(outptr + 16 * 4, _sum4);
                    _mm512_storeu_ps(outptr + 16 * 5, _sum5);
                    _mm512_storeu_ps(outptr + 16 * 6, _sum6);
                    _mm512_storeu_ps(outptr + 16 * 7, _sum7);
                    _mm512_storeu_ps(outptr + 16 * 8, _sum8);
                    _mm512_storeu_ps(outptr + 16 * 9, _sum9);
                    _mm512_storeu_ps(outptr + 16 * 10, _suma);
                    _mm512_storeu_ps(outptr + 16 * 11, _sumb);
                    _mm512_storeu_ps(outptr + 16 * 12, _sumc);
                    _mm512_storeu_ps(outptr + 16 * 13, _sumd);
                    _mm512_storeu_ps(outptr + 16 * 14, _sume);
                    _mm512_storeu_ps(outptr + 16 * 15, _sumf);
                    outptr += 256;
                }
            }

            if (elempack == 1 && num_output_elempack == 16)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    __m512 _sum = _mm512_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum = _mm512_loadu_ps((const float*)bias_data + p * 16);
                    }

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        __m512 _val = _mm512_set1_ps(m[0]);
                        __m512 _w = _mm512_loadu_ps(kptr);
                        _sum = _mm512_fmadd_ps(_val, _w, _sum);

                        m += 1;
                        kptr += 16;
                    }

                    _sum = activation_avx512(_sum, activation_type, activation_params);

                    _mm512_storeu_ps(outptr, _sum);
                    outptr += 16;
                }
            }

            if (elempack == 4 && num_output_elempack == 16)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    __m128 _sum0 = _mm_set1_ps(0.f);
                    __m128 _sum1 = _mm_set1_ps(0.f);
                    __m128 _sum2 = _mm_set1_ps(0.f);
                    __m128 _sum3 = _mm_set1_ps(0.f);
                    __m128 _sum4 = _mm_set1_ps(0.f);
                    __m128 _sum5 = _mm_set1_ps(0.f);
                    __m128 _sum6 = _mm_set1_ps(0.f);
                    __m128 _sum7 = _mm_set1_ps(0.f);
                    __m128 _sum8 = _mm_set1_ps(0.f);
                    __m128 _sum9 = _mm_set1_ps(0.f);
                    __m128 _suma = _mm_set1_ps(0.f);
                    __m128 _sumb = _mm_set1_ps(0.f);
                    __m128 _sumc = _mm_set1_ps(0.f);
                    __m128 _sumd = _mm_set1_ps(0.f);
                    __m128 _sume = _mm_set1_ps(0.f);
                    __m128 _sumf = _mm_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum0 = _mm_set1_ps(bias_data[p * 16 + 0]);
                        _sum1 = _mm_set1_ps(bias_data[p * 16 + 1]);
                        _sum2 = _mm_set1_ps(bias_data[p * 16 + 2]);
                        _sum3 = _mm_set1_ps(bias_data[p * 16 + 3]);
                        _sum4 = _mm_set1_ps(bias_data[p * 16 + 4]);
                        _sum5 = _mm_set1_ps(bias_data[p * 16 + 5]);
                        _sum6 = _mm_set1_ps(bias_data[p * 16 + 6]);
                        _sum7 = _mm_set1_ps(bias_data[p * 16 + 7]);
                        _sum8 = _mm_set1_ps(bias_data[p * 16 + 8]);
                        _sum9 = _mm_set1_ps(bias_data[p * 16 + 9]);
                        _suma = _mm_set1_ps(bias_data[p * 16 + 10]);
                        _sumb = _mm_set1_ps(bias_data[p * 16 + 11]);
                        _sumc = _mm_set1_ps(bias_data[p * 16 + 12]);
                        _sumd = _mm_set1_ps(bias_data[p * 16 + 13]);
                        _sume = _mm_set1_ps(bias_data[p * 16 + 14]);
                        _sumf = _mm_set1_ps(bias_data[p * 16 + 15]);
                    }

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        __m128 _val = _mm_loadu_ps(m);
                        _sum0 = _mm_fmadd_ps(_val, _mm_set1_ps(kptr[0]), _sum0);
                        _sum1 = _mm_fmadd_ps(_val, _mm_set1_ps(kptr[1]), _sum1);
                        _sum2 = _mm_fmadd_ps(_val, _mm_set1_ps(kptr[2]), _sum2);
                        _sum3 = _mm_fmadd_ps(_val, _mm_set1_ps(kptr[3]), _sum3);
                        _sum4 = _mm_fmadd_ps(_val, _mm_set1_ps(kptr[4]), _sum4);
                        _sum5 = _mm_fmadd_ps(_val, _mm_set1_ps(kptr[5]), _sum5);
                        _sum6 = _mm_fmadd_ps(_val, _mm_set1_ps(kptr[6]), _sum6);
                        _sum7 = _mm_fmadd_ps(_val, _mm_set1_ps(kptr[7]), _sum7);
                        _sum8 = _mm_fmadd_ps(_val, _mm_set1_ps(kptr[8]), _sum8);
                        _sum9 = _mm_fmadd_ps(_val, _mm_set1_ps(kptr[9]), _sum9);
                        _suma = _mm_fmadd_ps(_val, _mm_set1_ps(kptr[10]), _suma);
                        _sumb = _mm_fmadd_ps(_val, _mm_set1_ps(kptr[11]), _sumb);
                        _sumc = _mm_fmadd_ps(_val, _mm_set1_ps(kptr[12]), _sumc);
                        _sumd = _mm_fmadd_ps(_val, _mm_set1_ps(kptr[13]), _sumd);
                        _sume = _mm_fmadd_ps(_val, _mm_set1_ps(kptr[14]), _sume);
                        _sumf = _mm_fmadd_ps(_val, _mm_set1_ps(kptr[15]), _sumf);

                        m += 4;
                        kptr += 16;
                    }

                    _sum0 = activation_sse(_sum0, activation_type, activation_params);
                    _sum1 = activation_sse(_sum1, activation_type, activation_params);
                    _sum2 = activation_sse(_sum2, activation_type, activation_params);
                    _sum3 = activation_sse(_sum3, activation_type, activation_params);
                    _sum4 = activation_sse(_sum4, activation_type, activation_params);
                    _sum5 = activation_sse(_sum5, activation_type, activation_params);
                    _sum6 = activation_sse(_sum6, activation_type, activation_params);
                    _sum7 = activation_sse(_sum7, activation_type, activation_params);
                    _sum8 = activation_sse(_sum8, activation_type, activation_params);
                    _sum9 = activation_sse(_sum9, activation_type, activation_params);
                    _suma = activation_sse(_suma, activation_type, activation_params);
                    _sumb = activation_sse(_sumb, activation_type, activation_params);
                    _sumc = activation_sse(_sumc, activation_type, activation_params);
                    _sumd = activation_sse(_sumd, activation_type, activation_params);
                    _sume = activation_sse(_sume, activation_type, activation_params);
                    _sumf = activation_sse(_sumf, activation_type, activation_params);

                    _mm_storeu_ps(outptr, _sum0);
                    _mm_storeu_ps(outptr + 4, _sum1);
                    _mm_storeu_ps(outptr + 4 * 2, _sum2);
                    _mm_storeu_ps(outptr + 4 * 3, _sum3);
                    _mm_storeu_ps(outptr + 4 * 4, _sum4);
                    _mm_storeu_ps(outptr + 4 * 5, _sum5);
                    _mm_storeu_ps(outptr + 4 * 6, _sum6);
                    _mm_storeu_ps(outptr + 4 * 7, _sum7);
                    _mm_storeu_ps(outptr + 4 * 8, _sum8);
                    _mm_storeu_ps(outptr + 4 * 9, _sum9);
                    _mm_storeu_ps(outptr + 4 * 10, _suma);
                    _mm_storeu_ps(outptr + 4 * 11, _sumb);
                    _mm_storeu_ps(outptr + 4 * 12, _sumc);
                    _mm_storeu_ps(outptr + 4 * 13, _sumd);
                    _mm_storeu_ps(outptr + 4 * 14, _sume);
                    _mm_storeu_ps(outptr + 4 * 15, _sumf);
                    outptr += 64;
                }
            }

            if (elempack == 8 && num_output_elempack == 16)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    __m256 _sum0 = _mm256_set1_ps(0.f);
                    __m256 _sum1 = _mm256_set1_ps(0.f);
                    __m256 _sum2 = _mm256_set1_ps(0.f);
                    __m256 _sum3 = _mm256_set1_ps(0.f);
                    __m256 _sum4 = _mm256_set1_ps(0.f);
                    __m256 _sum5 = _mm256_set1_ps(0.f);
                    __m256 _sum6 = _mm256_set1_ps(0.f);
                    __m256 _sum7 = _mm256_set1_ps(0.f);
                    __m256 _sum8 = _mm256_set1_ps(0.f);
                    __m256 _sum9 = _mm256_set1_ps(0.f);
                    __m256 _suma = _mm256_set1_ps(0.f);
                    __m256 _sumb = _mm256_set1_ps(0.f);
                    __m256 _sumc = _mm256_set1_ps(0.f);
                    __m256 _sumd = _mm256_set1_ps(0.f);
                    __m256 _sume = _mm256_set1_ps(0.f);
                    __m256 _sumf = _mm256_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum0 = _mm256_set1_ps(bias_data[p * 16 + 0]);
                        _sum1 = _mm256_set1_ps(bias_data[p * 16 + 1]);
                        _sum2 = _mm256_set1_ps(bias_data[p * 16 + 2]);
                        _sum3 = _mm256_set1_ps(bias_data[p * 16 + 3]);
                        _sum4 = _mm256_set1_ps(bias_data[p * 16 + 4]);
                        _sum5 = _mm256_set1_ps(bias_data[p * 16 + 5]);
                        _sum6 = _mm256_set1_ps(bias_data[p * 16 + 6]);
                        _sum7 = _mm256_set1_ps(bias_data[p * 16 + 7]);
                        _sum8 = _mm256_set1_ps(bias_data[p * 16 + 8]);
                        _sum9 = _mm256_set1_ps(bias_data[p * 16 + 9]);
                        _suma = _mm256_set1_ps(bias_data[p * 16 + 10]);
                        _sumb = _mm256_set1_ps(bias_data[p * 16 + 11]);
                        _sumc = _mm256_set1_ps(bias_data[p * 16 + 12]);
                        _sumd = _mm256_set1_ps(bias_data[p * 16 + 13]);
                        _sume = _mm256_set1_ps(bias_data[p * 16 + 14]);
                        _sumf = _mm256_set1_ps(bias_data[p * 16 + 15]);
                    }

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        __m256 _val = _mm256_loadu_ps(m);
                        _sum0 = _mm256_fmadd_ps(_val, _mm256_set1_ps(kptr[0]), _sum0);
                        _sum1 = _mm256_fmadd_ps(_val, _mm256_set1_ps(kptr[1]), _sum1);
                        _sum2 = _mm256_fmadd_ps(_val, _mm256_set1_ps(kptr[2]), _sum2);
                        _sum3 = _mm256_fmadd_ps(_val, _mm256_set1_ps(kptr[3]), _sum3);
                        _sum4 = _mm256_fmadd_ps(_val, _mm256_set1_ps(kptr[4]), _sum4);
                        _sum5 = _mm256_fmadd_ps(_val, _mm256_set1_ps(kptr[5]), _sum5);
                        _sum6 = _mm256_fmadd_ps(_val, _mm256_set1_ps(kptr[6]), _sum6);
                        _sum7 = _mm256_fmadd_ps(_val, _mm256_set1_ps(kptr[7]), _sum7);
                        _sum8 = _mm256_fmadd_ps(_val, _mm256_set1_ps(kptr[8]), _sum8);
                        _sum9 = _mm256_fmadd_ps(_val, _mm256_set1_ps(kptr[9]), _sum9);
                        _suma = _mm256_fmadd_ps(_val, _mm256_set1_ps(kptr[10]), _suma);
                        _sumb = _mm256_fmadd_ps(_val, _mm256_set1_ps(kptr[11]), _sumb);
                        _sumc = _mm256_fmadd_ps(_val, _mm256_set1_ps(kptr[12]), _sumc);
                        _sumd = _mm256_fmadd_ps(_val, _mm256_set1_ps(kptr[13]), _sumd);
                        _sume = _mm256_fmadd_ps(_val, _mm256_set1_ps(kptr[14]), _sume);
                        _sumf = _mm256_fmadd_ps(_val, _mm256_set1_ps(kptr[15]), _sumf);

                        m += 8;
                        kptr += 16;
                    }

                    _sum0 = activation_avx(_sum0, activation_type, activation_params);
                    _sum1 = activation_avx(_sum1, activation_type, activation_params);
                    _sum2 = activation_avx(_sum2, activation_type, activation_params);
                    _sum3 = activation_avx(_sum3, activation_type, activation_params);
                    _sum4 = activation_avx(_sum4, activation_type, activation_params);
                    _sum5 = activation_avx(_sum5, activation_type, activation_params);
                    _sum6 = activation_avx(_sum6, activation_type, activation_params);
                    _sum7 = activation_avx(_sum7, activation_type, activation_params);
                    _sum8 = activation_avx(_sum8, activation_type, activation_params);
                    _sum9 = activation_avx(_sum9, activation_type, activation_params);
                    _suma = activation_avx(_suma, activation_type, activation_params);
                    _sumb = activation_avx(_sumb, activation_type, activation_params);
                    _sumc = activation_avx(_sumc, activation_type, activation_params);
                    _sumd = activation_avx(_sumd, activation_type, activation_params);
                    _sume = activation_avx(_sume, activation_type, activation_params);
                    _sumf = activation_avx(_sumf, activation_type, activation_params);

                    _mm256_storeu_ps(outptr, _sum0);
                    _mm256_storeu_ps(outptr + 8, _sum1);
                    _mm256_storeu_ps(outptr + 8 * 2, _sum2);
                    _mm256_storeu_ps(outptr + 8 * 3, _sum3);
                    _mm256_storeu_ps(outptr + 8 * 4, _sum4);
                    _mm256_storeu_ps(outptr + 8 * 5, _sum5);
                    _mm256_storeu_ps(outptr + 8 * 6, _sum6);
                    _mm256_storeu_ps(outptr + 8 * 7, _sum7);
                    _mm256_storeu_ps(outptr + 8 * 8, _sum8);
                    _mm256_storeu_ps(outptr + 8 * 9, _sum9);
                    _mm256_storeu_ps(outptr + 8 * 10, _suma);
                    _mm256_storeu_ps(outptr + 8 * 11, _sumb);
                    _mm256_storeu_ps(outptr + 8 * 12, _sumc);
                    _mm256_storeu_ps(outptr + 8 * 13, _sumd);
                    _mm256_storeu_ps(outptr + 8 * 14, _sume);
                    _mm256_storeu_ps(outptr + 8 * 15, _sumf);
                    outptr += 128;
                }
            }

            if (elempack == 16 && num_output_elempack == 1)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const float* kptr = (const float*)weight_data_tm + num_input * p;
                    const float* m = bottom_blob.row(j);

                    __m512 _sum0 = _mm512_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum0 = _mm512_set1_ps(bias_data[p]);
                    }

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        __m512 _val = _mm512_loadu_ps(m);
                        __m512 _k = _mm512_set1_ps(kptr[0]);
                        _sum0 = _mm512_fmadd_ps(_val, _k, _sum0);

                        m += 16;
                        kptr += 1;
                    }

                    _sum0 = activation_avx512(_sum0, activation_type, activation_params);

                    _mm512_storeu_ps(outptr, _sum0);
                    outptr += 16;
                }
            }

            if (elempack == 16 && num_output_elempack == 4)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    __m512 _sum0 = _mm512_set1_ps(0.f);
                    __m512 _sum1 = _mm512_set1_ps(0.f);
                    __m512 _sum2 = _mm512_set1_ps(0.f);
                    __m512 _sum3 = _mm512_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum0 = _mm512_set1_ps(bias_data[p * 4 + 0]);
                        _sum1 = _mm512_set1_ps(bias_data[p * 4 + 1]);
                        _sum2 = _mm512_set1_ps(bias_data[p * 4 + 2]);
                        _sum3 = _mm512_set1_ps(bias_data[p * 4 + 3]);
                    }

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        __m512 _val = _mm512_loadu_ps(m);
                        _sum0 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[3]), _sum3);

                        m += 16;
                        kptr += 4;
                    }

                    _sum0 = activation_avx512(_sum0, activation_type, activation_params);
                    _sum1 = activation_avx512(_sum1, activation_type, activation_params);
                    _sum2 = activation_avx512(_sum2, activation_type, activation_params);
                    _sum3 = activation_avx512(_sum3, activation_type, activation_params);

                    _mm512_storeu_ps(outptr, _sum0);
                    _mm512_storeu_ps(outptr + 16, _sum1);
                    _mm512_storeu_ps(outptr + 32, _sum2);
                    _mm512_storeu_ps(outptr + 48, _sum3);
                    outptr += 64;
                }
            }

            if (elempack == 16 && num_output_elempack == 8)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    __m512 _sum0 = _mm512_set1_ps(0.f);
                    __m512 _sum1 = _mm512_set1_ps(0.f);
                    __m512 _sum2 = _mm512_set1_ps(0.f);
                    __m512 _sum3 = _mm512_set1_ps(0.f);
                    __m512 _sum4 = _mm512_set1_ps(0.f);
                    __m512 _sum5 = _mm512_set1_ps(0.f);
                    __m512 _sum6 = _mm512_set1_ps(0.f);
                    __m512 _sum7 = _mm512_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum0 = _mm512_set1_ps(bias_data[p * 8 + 0]);
                        _sum1 = _mm512_set1_ps(bias_data[p * 8 + 1]);
                        _sum2 = _mm512_set1_ps(bias_data[p * 8 + 2]);
                        _sum3 = _mm512_set1_ps(bias_data[p * 8 + 3]);
                        _sum4 = _mm512_set1_ps(bias_data[p * 8 + 4]);
                        _sum5 = _mm512_set1_ps(bias_data[p * 8 + 5]);
                        _sum6 = _mm512_set1_ps(bias_data[p * 8 + 6]);
                        _sum7 = _mm512_set1_ps(bias_data[p * 8 + 7]);
                    }

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        __m512 _val = _mm512_loadu_ps(m);
                        _sum0 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[0]), _sum0);
                        _sum1 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[1]), _sum1);
                        _sum2 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[2]), _sum2);
                        _sum3 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[3]), _sum3);
                        _sum4 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[4]), _sum4);
                        _sum5 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[5]), _sum5);
                        _sum6 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[6]), _sum6);
                        _sum7 = _mm512_fmadd_ps(_val, _mm512_set1_ps(kptr[7]), _sum7);

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

                    _mm512_storeu_ps(outptr, _sum0);
                    _mm512_storeu_ps(outptr + 16, _sum1);
                    _mm512_storeu_ps(outptr + 16 * 2, _sum2);
                    _mm512_storeu_ps(outptr + 16 * 3, _sum3);
                    _mm512_storeu_ps(outptr + 16 * 4, _sum4);
                    _mm512_storeu_ps(outptr + 16 * 5, _sum5);
                    _mm512_storeu_ps(outptr + 16 * 6, _sum6);
                    _mm512_storeu_ps(outptr + 16 * 7, _sum7);
                    outptr += 128;
                }
            }

#endif // __AVX512F__

            if (elempack == 8 && num_output_elempack == 8)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    __m256 _sum0 = _mm256_set1_ps(0.f);
                    __m256 _sum1 = _mm256_set1_ps(0.f);
                    __m256 _sum2 = _mm256_set1_ps(0.f);
                    __m256 _sum3 = _mm256_set1_ps(0.f);
                    __m256 _sum4 = _mm256_set1_ps(0.f);
                    __m256 _sum5 = _mm256_set1_ps(0.f);
                    __m256 _sum6 = _mm256_set1_ps(0.f);
                    __m256 _sum7 = _mm256_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum0 = _mm256_set1_ps(bias_data[p * 8 + 0]);
                        _sum1 = _mm256_set1_ps(bias_data[p * 8 + 1]);
                        _sum2 = _mm256_set1_ps(bias_data[p * 8 + 2]);
                        _sum3 = _mm256_set1_ps(bias_data[p * 8 + 3]);
                        _sum4 = _mm256_set1_ps(bias_data[p * 8 + 4]);
                        _sum5 = _mm256_set1_ps(bias_data[p * 8 + 5]);
                        _sum6 = _mm256_set1_ps(bias_data[p * 8 + 6]);
                        _sum7 = _mm256_set1_ps(bias_data[p * 8 + 7]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        __m256 _val = _mm256_loadu_ps(m);
                        __m256 _k0 = _mm256_set1_ps(kptr[0]);
                        __m256 _k1 = _mm256_set1_ps(kptr[1]);
                        __m256 _k2 = _mm256_set1_ps(kptr[2]);
                        __m256 _k3 = _mm256_set1_ps(kptr[3]);
                        __m256 _k4 = _mm256_set1_ps(kptr[4]);
                        __m256 _k5 = _mm256_set1_ps(kptr[5]);
                        __m256 _k6 = _mm256_set1_ps(kptr[6]);
                        __m256 _k7 = _mm256_set1_ps(kptr[7]);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _k0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _k1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _k2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _k3, _sum3);
                        _sum4 = _mm256_comp_fmadd_ps(_val, _k4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val, _k5, _sum5);
                        _sum6 = _mm256_comp_fmadd_ps(_val, _k6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val, _k7, _sum7);

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

                    _mm256_storeu_ps(outptr, _sum0);
                    _mm256_storeu_ps(outptr + 8, _sum1);
                    _mm256_storeu_ps(outptr + 16, _sum2);
                    _mm256_storeu_ps(outptr + 24, _sum3);
                    _mm256_storeu_ps(outptr + 32, _sum4);
                    _mm256_storeu_ps(outptr + 40, _sum5);
                    _mm256_storeu_ps(outptr + 48, _sum6);
                    _mm256_storeu_ps(outptr + 56, _sum7);
                    outptr += 64;
                }
            }

            if (elempack == 1 && num_output_elempack == 8)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    __m256 _sum = _mm256_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum = _mm256_loadu_ps((const float*)bias_data + p * 8);
                    }

                    int i = 0;
                    for (; i + 7 < num_input; i += 8)
                    {
                        __m256 _val0 = _mm256_broadcast_ss(m);
                        __m256 _val1 = _mm256_broadcast_ss(m + 1);
                        __m256 _val2 = _mm256_broadcast_ss(m + 2);
                        __m256 _val3 = _mm256_broadcast_ss(m + 3);
                        __m256 _val4 = _mm256_broadcast_ss(m + 4);
                        __m256 _val5 = _mm256_broadcast_ss(m + 5);
                        __m256 _val6 = _mm256_broadcast_ss(m + 6);
                        __m256 _val7 = _mm256_broadcast_ss(m + 7);

                        __m256 _w0 = _mm256_loadu_ps(kptr);
                        _sum = _mm256_comp_fmadd_ps(_val0, _w0, _sum);
                        __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                        _sum = _mm256_comp_fmadd_ps(_val1, _w1, _sum);
                        __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                        _sum = _mm256_comp_fmadd_ps(_val2, _w2, _sum);
                        __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                        _sum = _mm256_comp_fmadd_ps(_val3, _w3, _sum);
                        __m256 _w4 = _mm256_loadu_ps(kptr + 32);
                        _sum = _mm256_comp_fmadd_ps(_val4, _w4, _sum);
                        __m256 _w5 = _mm256_loadu_ps(kptr + 40);
                        _sum = _mm256_comp_fmadd_ps(_val5, _w5, _sum);
                        __m256 _w6 = _mm256_loadu_ps(kptr + 48);
                        _sum = _mm256_comp_fmadd_ps(_val6, _w6, _sum);
                        __m256 _w7 = _mm256_loadu_ps(kptr + 56);
                        _sum = _mm256_comp_fmadd_ps(_val7, _w7, _sum);

                        m += 8;
                        kptr += 64;
                    }
                    for (; i + 3 < num_input; i += 4)
                    {
                        __m256 _val0 = _mm256_broadcast_ss(m);
                        __m256 _val1 = _mm256_broadcast_ss(m + 1);
                        __m256 _val2 = _mm256_broadcast_ss(m + 2);
                        __m256 _val3 = _mm256_broadcast_ss(m + 3);

                        __m256 _w0 = _mm256_loadu_ps(kptr);
                        _sum = _mm256_comp_fmadd_ps(_val0, _w0, _sum);
                        __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                        _sum = _mm256_comp_fmadd_ps(_val1, _w1, _sum);
                        __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                        _sum = _mm256_comp_fmadd_ps(_val2, _w2, _sum);
                        __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                        _sum = _mm256_comp_fmadd_ps(_val3, _w3, _sum);

                        m += 4;
                        kptr += 32;
                    }
                    for (; i < num_input; i++)
                    {
                        __m256 _val = _mm256_set1_ps(m[0]);
                        __m256 _w = _mm256_loadu_ps(kptr);
                        _sum = _mm256_comp_fmadd_ps(_val, _w, _sum);

                        m += 1;
                        kptr += 8;
                    }

                    _sum = activation_avx(_sum, activation_type, activation_params);

                    _mm256_storeu_ps(outptr, _sum);
                    outptr += 8;
                }
            }

            if (elempack == 4 && num_output_elempack == 8)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    __m128 _sum0 = _mm_set1_ps(0.f);
                    __m128 _sum1 = _mm_set1_ps(0.f);
                    __m128 _sum2 = _mm_set1_ps(0.f);
                    __m128 _sum3 = _mm_set1_ps(0.f);
                    __m128 _sum4 = _mm_set1_ps(0.f);
                    __m128 _sum5 = _mm_set1_ps(0.f);
                    __m128 _sum6 = _mm_set1_ps(0.f);
                    __m128 _sum7 = _mm_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum0 = _mm_set1_ps(bias_data[p * 8 + 0]);
                        _sum1 = _mm_set1_ps(bias_data[p * 8 + 1]);
                        _sum2 = _mm_set1_ps(bias_data[p * 8 + 2]);
                        _sum3 = _mm_set1_ps(bias_data[p * 8 + 3]);
                        _sum4 = _mm_set1_ps(bias_data[p * 8 + 4]);
                        _sum5 = _mm_set1_ps(bias_data[p * 8 + 5]);
                        _sum6 = _mm_set1_ps(bias_data[p * 8 + 6]);
                        _sum7 = _mm_set1_ps(bias_data[p * 8 + 7]);
                    }

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        __m128 _val = _mm_loadu_ps(m);
                        _sum0 = _mm_comp_fmadd_ps(_val, _mm_set1_ps(kptr[0]), _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _mm_set1_ps(kptr[1]), _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val, _mm_set1_ps(kptr[2]), _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _mm_set1_ps(kptr[3]), _sum3);
                        _sum4 = _mm_comp_fmadd_ps(_val, _mm_set1_ps(kptr[4]), _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val, _mm_set1_ps(kptr[5]), _sum5);
                        _sum6 = _mm_comp_fmadd_ps(_val, _mm_set1_ps(kptr[6]), _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val, _mm_set1_ps(kptr[7]), _sum7);

                        m += 4;
                        kptr += 8;
                    }

                    _sum0 = activation_sse(_sum0, activation_type, activation_params);
                    _sum1 = activation_sse(_sum1, activation_type, activation_params);
                    _sum2 = activation_sse(_sum2, activation_type, activation_params);
                    _sum3 = activation_sse(_sum3, activation_type, activation_params);
                    _sum4 = activation_sse(_sum4, activation_type, activation_params);
                    _sum5 = activation_sse(_sum5, activation_type, activation_params);
                    _sum6 = activation_sse(_sum6, activation_type, activation_params);
                    _sum7 = activation_sse(_sum7, activation_type, activation_params);

                    _mm_storeu_ps(outptr, _sum0);
                    _mm_storeu_ps(outptr + 4, _sum1);
                    _mm_storeu_ps(outptr + 8, _sum2);
                    _mm_storeu_ps(outptr + 12, _sum3);
                    _mm_storeu_ps(outptr + 16, _sum4);
                    _mm_storeu_ps(outptr + 20, _sum5);
                    _mm_storeu_ps(outptr + 24, _sum6);
                    _mm_storeu_ps(outptr + 28, _sum7);
                    outptr += 32;
                }
            }

            if (elempack == 8 && num_output_elempack == 1)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const float* kptr = (const float*)weight_data_tm + num_input * p;
                    const float* m = bottom_blob.row(j);

                    __m256 _sum0 = _mm256_set1_ps(0.f);
                    __m256 _sum1 = _mm256_set1_ps(0.f);
                    __m256 _sum2 = _mm256_set1_ps(0.f);
                    __m256 _sum3 = _mm256_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum0 = _mm256_set1_ps(bias_data[p]);
                    }

                    int i = 0;
                    for (; i + 7 < num_input; i += 8)
                    {
                        __m256 _val0 = _mm256_loadu_ps(m);
                        __m256 _val1 = _mm256_loadu_ps(m + 8);
                        __m256 _val2 = _mm256_loadu_ps(m + 16);
                        __m256 _val3 = _mm256_loadu_ps(m + 24);
                        __m256 _val4 = _mm256_loadu_ps(m + 32);
                        __m256 _val5 = _mm256_loadu_ps(m + 40);
                        __m256 _val6 = _mm256_loadu_ps(m + 48);
                        __m256 _val7 = _mm256_loadu_ps(m + 56);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _mm256_set1_ps(kptr[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val1, _mm256_set1_ps(kptr[1]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val2, _mm256_set1_ps(kptr[2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val3, _mm256_set1_ps(kptr[3]), _sum3);
                        _sum0 = _mm256_comp_fmadd_ps(_val4, _mm256_set1_ps(kptr[4]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val5, _mm256_set1_ps(kptr[5]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val6, _mm256_set1_ps(kptr[6]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val7, _mm256_set1_ps(kptr[7]), _sum3);

                        m += 64;
                        kptr += 8;
                    }
                    for (; i + 3 < num_input; i += 4)
                    {
                        __m256 _val0 = _mm256_loadu_ps(m);
                        __m256 _val1 = _mm256_loadu_ps(m + 8);
                        __m256 _val2 = _mm256_loadu_ps(m + 16);
                        __m256 _val3 = _mm256_loadu_ps(m + 24);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _mm256_set1_ps(kptr[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val1, _mm256_set1_ps(kptr[1]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val2, _mm256_set1_ps(kptr[2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val3, _mm256_set1_ps(kptr[3]), _sum3);

                        m += 32;
                        kptr += 4;
                    }
                    for (; i < num_input; i++)
                    {
                        __m256 _val = _mm256_loadu_ps(m);
                        __m256 _k = _mm256_set1_ps(kptr[0]);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _k, _sum0);

                        m += 8;
                        kptr += 1;
                    }

                    _sum0 = _mm256_add_ps(_sum0, _sum1);
                    _sum2 = _mm256_add_ps(_sum2, _sum3);
                    _sum0 = _mm256_add_ps(_sum0, _sum2);

                    _sum0 = activation_avx(_sum0, activation_type, activation_params);

                    _mm256_storeu_ps(outptr, _sum0);
                    outptr += 8;
                }
            }

            if (elempack == 8 && num_output_elempack == 4)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    __m256 _sum0 = _mm256_set1_ps(0.f);
                    __m256 _sum1 = _mm256_set1_ps(0.f);
                    __m256 _sum2 = _mm256_set1_ps(0.f);
                    __m256 _sum3 = _mm256_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum0 = _mm256_set1_ps(bias_data[p * 4 + 0]);
                        _sum1 = _mm256_set1_ps(bias_data[p * 4 + 1]);
                        _sum2 = _mm256_set1_ps(bias_data[p * 4 + 2]);
                        _sum3 = _mm256_set1_ps(bias_data[p * 4 + 3]);
                    }

                    int i = 0;
                    for (; i + 3 < num_input; i += 4)
                    {
                        __m256 _val0 = _mm256_loadu_ps(m);
                        __m256 _val1 = _mm256_loadu_ps(m + 8);
                        __m256 _val2 = _mm256_loadu_ps(m + 16);
                        __m256 _val3 = _mm256_loadu_ps(m + 24);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _mm256_set1_ps(kptr[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val0, _mm256_set1_ps(kptr[1]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val0, _mm256_set1_ps(kptr[2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val0, _mm256_set1_ps(kptr[3]), _sum3);
                        _sum0 = _mm256_comp_fmadd_ps(_val1, _mm256_set1_ps(kptr[4]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val1, _mm256_set1_ps(kptr[5]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val1, _mm256_set1_ps(kptr[6]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val1, _mm256_set1_ps(kptr[7]), _sum3);
                        kptr += 8;

                        _sum0 = _mm256_comp_fmadd_ps(_val2, _mm256_set1_ps(kptr[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val2, _mm256_set1_ps(kptr[1]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val2, _mm256_set1_ps(kptr[2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val2, _mm256_set1_ps(kptr[3]), _sum3);
                        _sum0 = _mm256_comp_fmadd_ps(_val3, _mm256_set1_ps(kptr[4]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val3, _mm256_set1_ps(kptr[5]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val3, _mm256_set1_ps(kptr[6]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val3, _mm256_set1_ps(kptr[7]), _sum3);

                        m += 32;
                        kptr += 8;
                    }
                    for (; i < num_input; i++)
                    {
                        __m256 _val = _mm256_loadu_ps(m);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _mm256_set1_ps(kptr[0]), _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _mm256_set1_ps(kptr[1]), _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _mm256_set1_ps(kptr[2]), _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _mm256_set1_ps(kptr[3]), _sum3);

                        m += 8;
                        kptr += 4;
                    }

                    _sum0 = activation_avx(_sum0, activation_type, activation_params);
                    _sum1 = activation_avx(_sum1, activation_type, activation_params);
                    _sum2 = activation_avx(_sum2, activation_type, activation_params);
                    _sum3 = activation_avx(_sum3, activation_type, activation_params);

                    _mm256_storeu_ps(outptr, _sum0);
                    _mm256_storeu_ps(outptr + 8, _sum1);
                    _mm256_storeu_ps(outptr + 16, _sum2);
                    _mm256_storeu_ps(outptr + 24, _sum3);
                    outptr += 32;
                }
            }
#endif // __AVX__

            if (elempack == 4 && num_output_elempack == 4)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    __m128 _sum0 = _mm_set1_ps(0.f);
                    __m128 _sum1 = _mm_set1_ps(0.f);
                    __m128 _sum2 = _mm_set1_ps(0.f);
                    __m128 _sum3 = _mm_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum0 = _mm_set1_ps(bias_data[p * 4 + 0]);
                        _sum1 = _mm_set1_ps(bias_data[p * 4 + 1]);
                        _sum2 = _mm_set1_ps(bias_data[p * 4 + 2]);
                        _sum3 = _mm_set1_ps(bias_data[p * 4 + 3]);
                    }

                    int i = 0;
                    for (; i + 3 < num_input; i += 4)
                    {
                        __m128 _val0 = _mm_loadu_ps(m);
                        __m128 _val1 = _mm_loadu_ps(m + 4);
                        __m128 _val2 = _mm_loadu_ps(m + 8);
                        __m128 _val3 = _mm_loadu_ps(m + 12);
                        _sum0 = _mm_add_ps(_mm_mul_ps(_val0, _mm_set1_ps(kptr[0])), _sum0);
                        _sum1 = _mm_add_ps(_mm_mul_ps(_val0, _mm_set1_ps(kptr[1])), _sum1);
                        _sum2 = _mm_add_ps(_mm_mul_ps(_val0, _mm_set1_ps(kptr[2])), _sum2);
                        _sum3 = _mm_add_ps(_mm_mul_ps(_val0, _mm_set1_ps(kptr[3])), _sum3);
                        _sum0 = _mm_add_ps(_mm_mul_ps(_val1, _mm_set1_ps(kptr[4])), _sum0);
                        _sum1 = _mm_add_ps(_mm_mul_ps(_val1, _mm_set1_ps(kptr[5])), _sum1);
                        _sum2 = _mm_add_ps(_mm_mul_ps(_val1, _mm_set1_ps(kptr[6])), _sum2);
                        _sum3 = _mm_add_ps(_mm_mul_ps(_val1, _mm_set1_ps(kptr[7])), _sum3);
                        _sum0 = _mm_add_ps(_mm_mul_ps(_val2, _mm_set1_ps(kptr[8])), _sum0);
                        _sum1 = _mm_add_ps(_mm_mul_ps(_val2, _mm_set1_ps(kptr[9])), _sum1);
                        _sum2 = _mm_add_ps(_mm_mul_ps(_val2, _mm_set1_ps(kptr[10])), _sum2);
                        _sum3 = _mm_add_ps(_mm_mul_ps(_val2, _mm_set1_ps(kptr[11])), _sum3);
                        _sum0 = _mm_add_ps(_mm_mul_ps(_val3, _mm_set1_ps(kptr[12])), _sum0);
                        _sum1 = _mm_add_ps(_mm_mul_ps(_val3, _mm_set1_ps(kptr[13])), _sum1);
                        _sum2 = _mm_add_ps(_mm_mul_ps(_val3, _mm_set1_ps(kptr[14])), _sum2);
                        _sum3 = _mm_add_ps(_mm_mul_ps(_val3, _mm_set1_ps(kptr[15])), _sum3);

                        m += 16;
                        kptr += 16;
                    }
                    for (; i < num_input; i++)
                    {
                        __m128 _val = _mm_loadu_ps(m);
                        _sum0 = _mm_add_ps(_mm_mul_ps(_val, _mm_set1_ps(kptr[0])), _sum0);
                        _sum1 = _mm_add_ps(_mm_mul_ps(_val, _mm_set1_ps(kptr[1])), _sum1);
                        _sum2 = _mm_add_ps(_mm_mul_ps(_val, _mm_set1_ps(kptr[2])), _sum2);
                        _sum3 = _mm_add_ps(_mm_mul_ps(_val, _mm_set1_ps(kptr[3])), _sum3);

                        m += 4;
                        kptr += 4;
                    }

                    _sum0 = activation_sse(_sum0, activation_type, activation_params);
                    _sum1 = activation_sse(_sum1, activation_type, activation_params);
                    _sum2 = activation_sse(_sum2, activation_type, activation_params);
                    _sum3 = activation_sse(_sum3, activation_type, activation_params);

                    _mm_storeu_ps(outptr, _sum0);
                    _mm_storeu_ps(outptr + 4, _sum1);
                    _mm_storeu_ps(outptr + 8, _sum2);
                    _mm_storeu_ps(outptr + 12, _sum3);
                    outptr += 16;
                }
            }

            if (elempack == 1 && num_output_elempack == 4)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    __m128 _sum = _mm_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum = _mm_loadu_ps((const float*)bias_data + p * 4);
                    }

                    int i = 0;
#if __AVX__
                    for (; i + 7 < num_input; i += 8)
                    {
                        __m128 _val0 = _mm_broadcast_ss(m);
                        __m128 _val1 = _mm_broadcast_ss(m + 1);
                        __m128 _val2 = _mm_broadcast_ss(m + 2);
                        __m128 _val3 = _mm_broadcast_ss(m + 3);
                        __m128 _val4 = _mm_broadcast_ss(m + 4);
                        __m128 _val5 = _mm_broadcast_ss(m + 5);
                        __m128 _val6 = _mm_broadcast_ss(m + 6);
                        __m128 _val7 = _mm_broadcast_ss(m + 7);

                        __m128 _w0 = _mm_loadu_ps(kptr);
                        _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);
                        __m128 _w1 = _mm_loadu_ps(kptr + 4);
                        _sum = _mm_comp_fmadd_ps(_val1, _w1, _sum);
                        __m128 _w2 = _mm_loadu_ps(kptr + 8);
                        _sum = _mm_comp_fmadd_ps(_val2, _w2, _sum);
                        __m128 _w3 = _mm_loadu_ps(kptr + 12);
                        _sum = _mm_comp_fmadd_ps(_val3, _w3, _sum);
                        __m128 _w4 = _mm_loadu_ps(kptr + 16);
                        _sum = _mm_comp_fmadd_ps(_val4, _w4, _sum);
                        __m128 _w5 = _mm_loadu_ps(kptr + 20);
                        _sum = _mm_comp_fmadd_ps(_val5, _w5, _sum);
                        __m128 _w6 = _mm_loadu_ps(kptr + 24);
                        _sum = _mm_comp_fmadd_ps(_val6, _w6, _sum);
                        __m128 _w7 = _mm_loadu_ps(kptr + 28);
                        _sum = _mm_comp_fmadd_ps(_val7, _w7, _sum);

                        m += 8;
                        kptr += 32;
                    }
#endif // __AVX__
                    for (; i + 3 < num_input; i += 4)
                    {
                        __m128 _val0 = _mm_set1_ps(m[0]);
                        __m128 _val1 = _mm_set1_ps(m[1]);
                        __m128 _val2 = _mm_set1_ps(m[2]);
                        __m128 _val3 = _mm_set1_ps(m[3]);

                        __m128 _w0 = _mm_loadu_ps(kptr);
                        _sum = _mm_add_ps(_mm_mul_ps(_val0, _w0), _sum);
                        __m128 _w1 = _mm_loadu_ps(kptr + 4);
                        _sum = _mm_add_ps(_mm_mul_ps(_val1, _w1), _sum);
                        __m128 _w2 = _mm_loadu_ps(kptr + 8);
                        _sum = _mm_add_ps(_mm_mul_ps(_val2, _w2), _sum);
                        __m128 _w3 = _mm_loadu_ps(kptr + 12);
                        _sum = _mm_add_ps(_mm_mul_ps(_val3, _w3), _sum);

                        m += 4;
                        kptr += 16;
                    }
                    for (; i < num_input; i++)
                    {
                        __m128 _val = _mm_set1_ps(m[0]);
                        __m128 _k = _mm_loadu_ps(kptr);
                        _sum = _mm_add_ps(_mm_mul_ps(_val, _k), _sum);

                        m += 1;
                        kptr += 4;
                    }

                    _sum = activation_sse(_sum, activation_type, activation_params);

                    _mm_storeu_ps(outptr, _sum);
                    outptr += 4;
                }
            }

            if (elempack == 4 && num_output_elempack == 1)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const float* kptr = (const float*)weight_data_tm + num_input * p;
                    const float* m = bottom_blob.row(j);

                    __m128 _sum0 = _mm_set1_ps(0.f);
                    __m128 _sum1 = _mm_set1_ps(0.f);
                    __m128 _sum2 = _mm_set1_ps(0.f);
                    __m128 _sum3 = _mm_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum0 = _mm_set1_ps(bias_data[p]);
                    }

                    int i = 0;
                    for (; i + 7 < num_input; i += 8)
                    {
                        __m128 _val0 = _mm_loadu_ps(m);
                        __m128 _val1 = _mm_loadu_ps(m + 4);
                        __m128 _val2 = _mm_loadu_ps(m + 8);
                        __m128 _val3 = _mm_loadu_ps(m + 12);
                        __m128 _val4 = _mm_loadu_ps(m + 16);
                        __m128 _val5 = _mm_loadu_ps(m + 20);
                        __m128 _val6 = _mm_loadu_ps(m + 24);
                        __m128 _val7 = _mm_loadu_ps(m + 28);
                        _sum0 = _mm_add_ps(_mm_mul_ps(_val0, _mm_set1_ps(kptr[0])), _sum0);
                        _sum1 = _mm_add_ps(_mm_mul_ps(_val1, _mm_set1_ps(kptr[1])), _sum1);
                        _sum2 = _mm_add_ps(_mm_mul_ps(_val2, _mm_set1_ps(kptr[2])), _sum2);
                        _sum3 = _mm_add_ps(_mm_mul_ps(_val3, _mm_set1_ps(kptr[3])), _sum3);
                        _sum0 = _mm_add_ps(_mm_mul_ps(_val4, _mm_set1_ps(kptr[4])), _sum0);
                        _sum1 = _mm_add_ps(_mm_mul_ps(_val5, _mm_set1_ps(kptr[5])), _sum1);
                        _sum2 = _mm_add_ps(_mm_mul_ps(_val6, _mm_set1_ps(kptr[6])), _sum2);
                        _sum3 = _mm_add_ps(_mm_mul_ps(_val7, _mm_set1_ps(kptr[7])), _sum3);

                        m += 32;
                        kptr += 8;
                    }
                    for (; i + 3 < num_input; i += 4)
                    {
                        __m128 _val0 = _mm_loadu_ps(m);
                        __m128 _val1 = _mm_loadu_ps(m + 4);
                        __m128 _val2 = _mm_loadu_ps(m + 8);
                        __m128 _val3 = _mm_loadu_ps(m + 12);
                        _sum0 = _mm_add_ps(_mm_mul_ps(_val0, _mm_set1_ps(kptr[0])), _sum0);
                        _sum1 = _mm_add_ps(_mm_mul_ps(_val1, _mm_set1_ps(kptr[1])), _sum1);
                        _sum2 = _mm_add_ps(_mm_mul_ps(_val2, _mm_set1_ps(kptr[2])), _sum2);
                        _sum3 = _mm_add_ps(_mm_mul_ps(_val3, _mm_set1_ps(kptr[3])), _sum3);

                        m += 16;
                        kptr += 4;
                    }
                    for (; i < num_input; i++)
                    {
                        __m128 _val = _mm_loadu_ps(m);
                        __m128 _k = _mm_set1_ps(kptr[0]);
                        _sum0 = _mm_add_ps(_mm_mul_ps(_val, _k), _sum0);

                        m += 4;
                        kptr += 1;
                    }

                    _sum0 = _mm_add_ps(_sum0, _sum1);
                    _sum2 = _mm_add_ps(_sum2, _sum3);
                    _sum0 = _mm_add_ps(_sum0, _sum2);

                    _sum0 = activation_sse(_sum0, activation_type, activation_params);

                    _mm_storeu_ps(outptr, _sum0);
                    outptr += 4;
                }
            }
#endif // __SSE2__

            if (elempack == 1 && num_output_elempack == 1)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const float* kptr = (const float*)weight_data_tm + num_input * p;
                    const float* m = bottom_blob.row(j);

                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    int i = 0;
#if __SSE2__
#if __AVX__
                    __m256 _sum = _mm256_set1_ps(0.f);
                    for (; i + 7 < num_input; i += 8)
                    {
                        __m256 _m = _mm256_loadu_ps(m);
                        __m256 _w = _mm256_loadu_ps(kptr);
                        _sum = _mm256_comp_fmadd_ps(_m, _w, _sum);

                        m += 8;
                        kptr += 8;
                    }
#endif // __AVX__
                    __m128 _suml = _mm_set1_ps(0.f);
                    for (; i + 3 < num_input; i += 4)
                    {
                        __m128 _val = _mm_loadu_ps(m);
                        __m128 _k = _mm_loadu_ps(kptr);
                        _suml = _mm_add_ps(_mm_mul_ps(_val, _k), _suml);

                        m += 4;
                        kptr += 4;
                    }
#endif // __SSE2__
                    for (; i < num_input; i++)
                    {
                        sum += *m++ * *kptr++;
                    }

#if __SSE2__
#if __AVX__
                    sum += _mm256_reduce_add_ps(_sum);
#endif // __AVX__
                    sum += _mm_reduce_add_ps(_suml);
#endif // __SSE2__

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[0] = sum;
                    outptr += 1;
                }
            }
        }

        return 0;
    }

    // flatten
    Mat bottom_blob_flattened = bottom_blob;
    if (bottom_blob.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
    }

    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

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
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (out_elempack == 16)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            __m512 _sum0 = _mm512_set1_ps(0.f);
            __m512 _sum1 = _mm512_set1_ps(0.f);
            __m512 _sum2 = _mm512_set1_ps(0.f);
            __m512 _sum3 = _mm512_set1_ps(0.f);
            __m512 _sum4 = _mm512_set1_ps(0.f);
            __m512 _sum5 = _mm512_set1_ps(0.f);
            __m512 _sum6 = _mm512_set1_ps(0.f);
            __m512 _sum7 = _mm512_set1_ps(0.f);

            if (bias_term)
            {
                _sum0 = _mm512_loadu_ps((const float*)bias_data + p * 16);
            }

            const float* kptr = weight_data_tm.row(p);

            const float* sptr = bottom_blob_flattened;

            int i = 0;
            for (; i + 7 < num_input; i += 8)
            {
                __m512 _val0 = _mm512_set1_ps(sptr[0]);
                __m512 _val1 = _mm512_set1_ps(sptr[1]);
                __m512 _val2 = _mm512_set1_ps(sptr[2]);
                __m512 _val3 = _mm512_set1_ps(sptr[3]);
                __m512 _val4 = _mm512_set1_ps(sptr[4]);
                __m512 _val5 = _mm512_set1_ps(sptr[5]);
                __m512 _val6 = _mm512_set1_ps(sptr[6]);
                __m512 _val7 = _mm512_set1_ps(sptr[7]);

                __m512 _w0 = _mm512_loadu_ps(kptr + 16 * 0);
                __m512 _w1 = _mm512_loadu_ps(kptr + 16 * 1);
                __m512 _w2 = _mm512_loadu_ps(kptr + 16 * 2);
                __m512 _w3 = _mm512_loadu_ps(kptr + 16 * 3);
                __m512 _w4 = _mm512_loadu_ps(kptr + 16 * 4);
                __m512 _w5 = _mm512_loadu_ps(kptr + 16 * 5);
                __m512 _w6 = _mm512_loadu_ps(kptr + 16 * 6);
                __m512 _w7 = _mm512_loadu_ps(kptr + 16 * 7);

                _sum0 = _mm512_fmadd_ps(_val0, _w0, _sum0);
                _sum1 = _mm512_fmadd_ps(_val1, _w1, _sum1);
                _sum2 = _mm512_fmadd_ps(_val2, _w2, _sum2);
                _sum3 = _mm512_fmadd_ps(_val3, _w3, _sum3);
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

                __m512 _w0 = _mm512_loadu_ps(kptr);
                __m512 _w1 = _mm512_loadu_ps(kptr + 16);
                __m512 _w2 = _mm512_loadu_ps(kptr + 32);
                __m512 _w3 = _mm512_loadu_ps(kptr + 48);
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
                __m512 _w = _mm512_loadu_ps(kptr);
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
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            __m256 _sum0 = _mm256_set1_ps(0.f);
            __m256 _sum1 = _mm256_set1_ps(0.f);
            __m256 _sum2 = _mm256_set1_ps(0.f);
            __m256 _sum3 = _mm256_set1_ps(0.f);
            __m256 _sum4 = _mm256_set1_ps(0.f);
            __m256 _sum5 = _mm256_set1_ps(0.f);
            __m256 _sum6 = _mm256_set1_ps(0.f);
            __m256 _sum7 = _mm256_set1_ps(0.f);

            if (bias_term)
            {
                _sum0 = _mm256_loadu_ps((const float*)bias_data + p * 8);
            }

            const float* kptr = weight_data_tm.row(p);

            const float* sptr = bottom_blob_flattened;

            int i = 0;
            for (; i + 7 < num_input; i += 8)
            {
                __m256 _val0 = _mm256_broadcast_ss(sptr);
                __m256 _val1 = _mm256_broadcast_ss(sptr + 1);
                __m256 _val2 = _mm256_broadcast_ss(sptr + 2);
                __m256 _val3 = _mm256_broadcast_ss(sptr + 3);
                __m256 _val4 = _mm256_broadcast_ss(sptr + 4);
                __m256 _val5 = _mm256_broadcast_ss(sptr + 5);
                __m256 _val6 = _mm256_broadcast_ss(sptr + 6);
                __m256 _val7 = _mm256_broadcast_ss(sptr + 7);

                __m256 _w0 = _mm256_loadu_ps(kptr);
                _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                _sum1 = _mm256_comp_fmadd_ps(_val1, _w1, _sum1);
                __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                _sum2 = _mm256_comp_fmadd_ps(_val2, _w2, _sum2);
                __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                _sum3 = _mm256_comp_fmadd_ps(_val3, _w3, _sum3);
                __m256 _w4 = _mm256_loadu_ps(kptr + 32);
                _sum4 = _mm256_comp_fmadd_ps(_val4, _w4, _sum4);
                __m256 _w5 = _mm256_loadu_ps(kptr + 40);
                _sum5 = _mm256_comp_fmadd_ps(_val5, _w5, _sum5);
                __m256 _w6 = _mm256_loadu_ps(kptr + 48);
                _sum6 = _mm256_comp_fmadd_ps(_val6, _w6, _sum6);
                __m256 _w7 = _mm256_loadu_ps(kptr + 56);
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

                __m256 _w0 = _mm256_loadu_ps(kptr);
                _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                _sum1 = _mm256_comp_fmadd_ps(_val1, _w1, _sum1);
                __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                _sum2 = _mm256_comp_fmadd_ps(_val2, _w2, _sum2);
                __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                _sum3 = _mm256_comp_fmadd_ps(_val3, _w3, _sum3);

                sptr += 4;
                kptr += 32;
            }
            for (; i < num_input; i++)
            {
                __m256 _val = _mm256_set1_ps(sptr[0]);
                __m256 _w = _mm256_loadu_ps(kptr);
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
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            __m128 _sum0 = _mm_set1_ps(0.f);
            __m128 _sum1 = _mm_set1_ps(0.f);
            __m128 _sum2 = _mm_set1_ps(0.f);
            __m128 _sum3 = _mm_set1_ps(0.f);
#if __AVX__
            __m128 _sum4 = _mm_set1_ps(0.f);
            __m128 _sum5 = _mm_set1_ps(0.f);
            __m128 _sum6 = _mm_set1_ps(0.f);
            __m128 _sum7 = _mm_set1_ps(0.f);
#endif

            if (bias_term)
            {
                _sum0 = _mm_loadu_ps((const float*)bias_data + p * 4);
            }

            const float* kptr = weight_data_tm.row(p);

            const float* sptr = bottom_blob_flattened;

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

                __m128 _w0 = _mm_loadu_ps(kptr);
                _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                __m128 _w1 = _mm_loadu_ps(kptr + 4);
                _sum1 = _mm_comp_fmadd_ps(_val1, _w1, _sum1);
                __m128 _w2 = _mm_loadu_ps(kptr + 8);
                _sum2 = _mm_comp_fmadd_ps(_val2, _w2, _sum2);
                __m128 _w3 = _mm_loadu_ps(kptr + 12);
                _sum3 = _mm_comp_fmadd_ps(_val3, _w3, _sum3);
                __m128 _w4 = _mm_loadu_ps(kptr + 16);
                _sum4 = _mm_comp_fmadd_ps(_val4, _w4, _sum4);
                __m128 _w5 = _mm_loadu_ps(kptr + 20);
                _sum5 = _mm_comp_fmadd_ps(_val5, _w5, _sum5);
                __m128 _w6 = _mm_loadu_ps(kptr + 24);
                _sum6 = _mm_comp_fmadd_ps(_val6, _w6, _sum6);
                __m128 _w7 = _mm_loadu_ps(kptr + 28);
                _sum7 = _mm_comp_fmadd_ps(_val7, _w7, _sum7);

                sptr += 8;
                kptr += 32;
            }
#endif
            for (; i + 3 < num_input; i += 4)
            {
                __m128 _val0 = _mm_set1_ps(sptr[0]);
                __m128 _val1 = _mm_set1_ps(sptr[1]);
                __m128 _val2 = _mm_set1_ps(sptr[2]);
                __m128 _val3 = _mm_set1_ps(sptr[3]);

                __m128 _w0 = _mm_loadu_ps(kptr);
                _sum0 = _mm_add_ps(_mm_mul_ps(_val0, _w0), _sum0);
                __m128 _w1 = _mm_loadu_ps(kptr + 4);
                _sum1 = _mm_add_ps(_mm_mul_ps(_val1, _w1), _sum1);
                __m128 _w2 = _mm_loadu_ps(kptr + 8);
                _sum2 = _mm_add_ps(_mm_mul_ps(_val2, _w2), _sum2);
                __m128 _w3 = _mm_loadu_ps(kptr + 12);
                _sum3 = _mm_add_ps(_mm_mul_ps(_val3, _w3), _sum3);

                sptr += 4;
                kptr += 16;
            }
            for (; i < num_input; i++)
            {
                __m128 _val = _mm_set1_ps(sptr[0]);
                __m128 _w = _mm_loadu_ps(kptr);
                _sum0 = _mm_add_ps(_mm_mul_ps(_val, _w), _sum0);

                sptr += 1;
                kptr += 4;
            }

            _sum0 = _mm_add_ps(_sum0, _sum1);
            _sum2 = _mm_add_ps(_sum2, _sum3);
#if __AVX__
            _sum4 = _mm_add_ps(_sum4, _sum5);
            _sum6 = _mm_add_ps(_sum6, _sum7);
#endif
            _sum0 = _mm_add_ps(_sum0, _sum2);
#if __AVX__
            _sum4 = _mm_add_ps(_sum4, _sum6);
            _sum0 = _mm_add_ps(_sum0, _sum4);
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
        int remain_num_output_start = 0;
        int nn_num_output = num_output >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_num_output; pp++)
        {
            int p = pp * 8;

            float sums[8] = {0.0f};
            if (bias_term)
            {
                sums[0] = bias_data[p];
                sums[1] = bias_data[p + 1];
                sums[2] = bias_data[p + 2];
                sums[3] = bias_data[p + 3];
                sums[4] = bias_data[p + 4];
                sums[5] = bias_data[p + 5];
                sums[6] = bias_data[p + 6];
                sums[7] = bias_data[p + 7];
            }

            const float* w0 = (const float*)weight_data_tm + num_input * p;
            const float* w1 = (const float*)weight_data_tm + num_input * (p + 1);
            const float* w2 = (const float*)weight_data_tm + num_input * (p + 2);
            const float* w3 = (const float*)weight_data_tm + num_input * (p + 3);
            const float* w4 = (const float*)weight_data_tm + num_input * (p + 4);
            const float* w5 = (const float*)weight_data_tm + num_input * (p + 5);
            const float* w6 = (const float*)weight_data_tm + num_input * (p + 6);
            const float* w7 = (const float*)weight_data_tm + num_input * (p + 7);

            const float* m = bottom_blob_flattened;

            __m256 _sum0 = _mm256_set1_ps(0.f);
            __m256 _sum1 = _mm256_set1_ps(0.f);
            __m256 _sum2 = _mm256_set1_ps(0.f);
            __m256 _sum3 = _mm256_set1_ps(0.f);
            __m256 _sum4 = _mm256_set1_ps(0.f);
            __m256 _sum5 = _mm256_set1_ps(0.f);
            __m256 _sum6 = _mm256_set1_ps(0.f);
            __m256 _sum7 = _mm256_set1_ps(0.f);

            int i = 0;
            for (; i + 7 < num_input; i += 8)
            {
                __m256 _m = _mm256_loadu_ps(m);

                __m256 _w0 = _mm256_loadu_ps(w0);
                _sum0 = _mm256_comp_fmadd_ps(_m, _w0, _sum0);
                __m256 _w1 = _mm256_loadu_ps(w1);
                _sum1 = _mm256_comp_fmadd_ps(_m, _w1, _sum1);
                __m256 _w2 = _mm256_loadu_ps(w2);
                _sum2 = _mm256_comp_fmadd_ps(_m, _w2, _sum2);
                __m256 _w3 = _mm256_loadu_ps(w3);
                _sum3 = _mm256_comp_fmadd_ps(_m, _w3, _sum3);
                __m256 _w4 = _mm256_loadu_ps(w4);
                _sum4 = _mm256_comp_fmadd_ps(_m, _w4, _sum4);
                __m256 _w5 = _mm256_loadu_ps(w5);
                _sum5 = _mm256_comp_fmadd_ps(_m, _w5, _sum5);
                __m256 _w6 = _mm256_loadu_ps(w6);
                _sum6 = _mm256_comp_fmadd_ps(_m, _w6, _sum6);
                __m256 _w7 = _mm256_loadu_ps(w7);
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
                sums[0] += *m * *w0;
                sums[1] += *m * *w1;
                sums[2] += *m * *w2;
                sums[3] += *m * *w3;
                sums[4] += *m * *w4;
                sums[5] += *m * *w5;
                sums[6] += *m * *w6;
                sums[7] += *m * *w7;

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
#else
        int remain_num_output_start = 0;
        int nn_num_output = num_output >> 2;
#endif // __AVX__

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_num_output; pp++)
        {
            int p = remain_num_output_start + (pp * 4);

            float sums[4] = {0.0f};
            if (bias_term)
            {
                sums[0] = bias_data[p];
                sums[1] = bias_data[p + 1];
                sums[2] = bias_data[p + 2];
                sums[3] = bias_data[p + 3];
            }

            const float* w0 = (const float*)weight_data_tm + num_input * p;
            const float* w1 = (const float*)weight_data_tm + num_input * (p + 1);
            const float* w2 = (const float*)weight_data_tm + num_input * (p + 2);
            const float* w3 = (const float*)weight_data_tm + num_input * (p + 3);

            const float* m = bottom_blob_flattened;

            int i = 0;
#if __AVX__
            __m256 _sum0 = _mm256_set1_ps(0.f);
            __m256 _sum1 = _mm256_set1_ps(0.f);
            __m256 _sum2 = _mm256_set1_ps(0.f);
            __m256 _sum3 = _mm256_set1_ps(0.f);
            for (; i + 7 < num_input; i += 8)
            {
                __m256 _m = _mm256_loadu_ps(m);

                __m256 _w0 = _mm256_loadu_ps(w0);
                _sum0 = _mm256_comp_fmadd_ps(_m, _w0, _sum0);
                __m256 _w1 = _mm256_loadu_ps(w1);
                _sum1 = _mm256_comp_fmadd_ps(_m, _w1, _sum1);
                __m256 _w2 = _mm256_loadu_ps(w2);
                _sum2 = _mm256_comp_fmadd_ps(_m, _w2, _sum2);
                __m256 _w3 = _mm256_loadu_ps(w3);
                _sum3 = _mm256_comp_fmadd_ps(_m, _w3, _sum3);

                m += 8;
                w0 += 8;
                w1 += 8;
                w2 += 8;
                w3 += 8;
            }
#endif // __AVX__
            __m128 _sum0l = _mm_set1_ps(0.f);
            __m128 _sum1l = _mm_set1_ps(0.f);
            __m128 _sum2l = _mm_set1_ps(0.f);
            __m128 _sum3l = _mm_set1_ps(0.f);
            for (; i + 3 < num_input; i += 4)
            {
                __m128 _m = _mm_loadu_ps(m);

                __m128 _w0 = _mm_loadu_ps(w0);
                _sum0l = _mm_add_ps(_mm_mul_ps(_m, _w0), _sum0l);
                __m128 _w1 = _mm_loadu_ps(w1);
                _sum1l = _mm_add_ps(_mm_mul_ps(_m, _w1), _sum1l);
                __m128 _w2 = _mm_loadu_ps(w2);
                _sum2l = _mm_add_ps(_mm_mul_ps(_m, _w2), _sum2l);
                __m128 _w3 = _mm_loadu_ps(w3);
                _sum3l = _mm_add_ps(_mm_mul_ps(_m, _w3), _sum3l);

                m += 4;
                w0 += 4;
                w1 += 4;
                w2 += 4;
                w3 += 4;
            }
            for (; i < num_input; i++)
            {
                sums[0] += *m * *w0;
                sums[1] += *m * *w1;
                sums[2] += *m * *w2;
                sums[3] += *m * *w3;

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

        remain_num_output_start += (nn_num_output << 2);
#else
        int remain_num_output_start = 0;
#endif // __SSE2__

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_num_output_start; p < num_output; p++)
        {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            const float* w = (const float*)weight_data_tm + num_input * p;

            const float* m = bottom_blob_flattened;

            int i = 0;
#if __SSE2__
#if __AVX__
            __m256 _sum = _mm256_set1_ps(0.f);
            for (; i + 7 < num_input; i += 8)
            {
                __m256 _m = _mm256_loadu_ps(m);

                __m256 _w = _mm256_loadu_ps(w);
                _sum = _mm256_comp_fmadd_ps(_m, _w, _sum);

                m += 8;
                w += 8;
            }
#endif // __AVX__
            __m128 _suml = _mm_set1_ps(0.f);
            for (; i + 3 < num_input; i += 4)
            {
                __m128 _m = _mm_loadu_ps(m);

                __m128 _w = _mm_loadu_ps(w);
                _suml = _mm_add_ps(_mm_mul_ps(_m, _w), _suml);

                m += 4;
                w += 4;
            }
#endif // __SSE2__
            for (; i < num_input; i++)
            {
                sum += *m * *w;
                m++;
                w++;
            }

#if __SSE2__
#if __AVX__
            sum += _mm256_reduce_add_ps(_sum);
#endif
            sum += _mm_reduce_add_ps(_suml);
#endif // __SSE2__

            sum = activation_ss(sum, activation_type, activation_params);

            float* outptr = top_blob;
            outptr[p] = sum;
        }
    }

    return 0;
}

#if NCNN_F16C
int InnerProduct_x86::create_pipeline_fp16s(const Option& opt)
{
    const int num_input = weight_data_size / num_output;

    innerproduct_transform_kernel_fp16s_sse(weight_data, weight_data_tm, num_input, num_output, opt);

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}

int InnerProduct_x86::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input && bottom_blob.h * bottom_blob.elempack > 1)
    {
        // gemm
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        innerproduct_gemm_fp16s_sse(bottom_blob, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);

        return 0;
    }

    // flatten
    Mat bottom_blob_flattened = bottom_blob;
    if (bottom_blob.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
    }

    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        out_elempack = num_output % 16 == 0 ? 16 : num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#endif
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __AVX512F__
    if (out_elempack == 16)
    {
        innerproduct_fp16s_pack16_avx512(bottom_blob_flattened, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);
    }
#endif // __AVX512F__

    if (out_elempack == 8)
    {
        innerproduct_fp16s_pack8_avx(bottom_blob_flattened, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);
    }

    if (out_elempack == 4)
    {
        innerproduct_fp16s_pack4_sse(bottom_blob_flattened, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);
    }

    if (out_elempack == 1)
    {
        innerproduct_fp16s_sse(bottom_blob_flattened, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);
    }

    return 0;
}
#endif // NCNN_F16C

#if NCNN_INT8
int InnerProduct_x86::create_pipeline_int8_x86(const Option& opt)
{
    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 8 == 0 ? 8 : 1;
    }
#endif // __SSE2__

    // src = inch-outch
    // dst = pb-inch-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_tm.create(num_input, num_output / out_elempack, (size_t)out_elempack, out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            signed char* g0 = weight_data_tm.row<signed char>(q / out_elempack);

            for (int p = 0; p < num_input; p++)
            {
                for (int j = 0; j < out_elempack; j++)
                {
                    *g0++ = weight_data_r2.row<signed char>(q + j)[p];
                }
            }
        }
    }

    scale_in_data.create(num_output);
    for (int p = 0; p < num_output; p++)
    {
        // dequantize
        float scale_in;
        if (weight_data_int8_scales[p] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);

        scale_in_data[p] = scale_in;
    }

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}

int InnerProduct_x86::forward_int8_x86(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;

    int elembits = bottom_blob.elembits();

    Mat bottom_blob_int8 = bottom_blob;
    if (elembits != 8)
    {
        Option opt_q = opt;
        opt_q.blob_allocator = opt.workspace_allocator;
        quantize_to_int8(bottom_blob, bottom_blob_int8, bottom_blob_int8_scales, opt_q);
    }

    if (bottom_blob_int8.dims == 2 && bottom_blob_int8.w == num_input && bottom_blob_int8.h * bottom_blob_int8.elempack > 1)
    {
        // gemm
        Mat bottom_blob_int8_unpacked;
        Option opt_unpack = opt;
        opt_unpack.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob_int8, bottom_blob_int8_unpacked, 1, opt_unpack);

        int h = bottom_blob_int8_unpacked.h;

        int out_elempack = 1;
#if __SSE2__
        if (opt.use_packing_layout)
        {
            out_elempack = h % 4 == 0 ? 4 : 1;
        }
#endif

        int outh = h / out_elempack;

        top_blob.create(num_output, outh, (size_t)(4u * out_elempack), out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int num_output_elempack = 1;
#if __SSE2__
        if (opt.use_packing_layout)
        {
            num_output_elempack = num_output % 8 == 0 ? 8 : 1;
        }
#endif

#if __SSE2__
        if (num_output_elempack == 8 && out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < outh; j++)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const signed char* kptr = weight_data_tm.row<const signed char>(p);
                    const signed char* m0 = bottom_blob_int8_unpacked.row<const signed char>(j * 4);
                    const signed char* m1 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 1);
                    const signed char* m2 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 2);
                    const signed char* m3 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 3);

                    __m128i _sum00 = _mm_setzero_si128();
                    __m128i _sum01 = _mm_setzero_si128();
                    __m128i _sum10 = _mm_setzero_si128();
                    __m128i _sum11 = _mm_setzero_si128();
                    __m128i _sum20 = _mm_setzero_si128();
                    __m128i _sum21 = _mm_setzero_si128();
                    __m128i _sum30 = _mm_setzero_si128();
                    __m128i _sum31 = _mm_setzero_si128();

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        // TODO use _mm_cvtepi8_epi16 on sse4.1
                        __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
                        _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));

                        __m128i _val0 = _mm_set1_epi16((short)m0[0]);
                        __m128i _val1 = _mm_set1_epi16((short)m1[0]);
                        __m128i _val2 = _mm_set1_epi16((short)m2[0]);
                        __m128i _val3 = _mm_set1_epi16((short)m3[0]);

                        __m128i _s0l = _mm_mullo_epi16(_val0, _w);
                        __m128i _s0h = _mm_mulhi_epi16(_val0, _w);
                        __m128i _s1l = _mm_mullo_epi16(_val1, _w);
                        __m128i _s1h = _mm_mulhi_epi16(_val1, _w);
                        __m128i _s2l = _mm_mullo_epi16(_val2, _w);
                        __m128i _s2h = _mm_mulhi_epi16(_val2, _w);
                        __m128i _s3l = _mm_mullo_epi16(_val3, _w);
                        __m128i _s3h = _mm_mulhi_epi16(_val3, _w);
                        __m128i _s00 = _mm_unpacklo_epi16(_s0l, _s0h);
                        __m128i _s01 = _mm_unpackhi_epi16(_s0l, _s0h);
                        __m128i _s10 = _mm_unpacklo_epi16(_s1l, _s1h);
                        __m128i _s11 = _mm_unpackhi_epi16(_s1l, _s1h);
                        __m128i _s20 = _mm_unpacklo_epi16(_s2l, _s2h);
                        __m128i _s21 = _mm_unpackhi_epi16(_s2l, _s2h);
                        __m128i _s30 = _mm_unpacklo_epi16(_s3l, _s3h);
                        __m128i _s31 = _mm_unpackhi_epi16(_s3l, _s3h);

                        _sum00 = _mm_add_epi32(_sum00, _s00);
                        _sum01 = _mm_add_epi32(_sum01, _s01);
                        _sum10 = _mm_add_epi32(_sum10, _s10);
                        _sum11 = _mm_add_epi32(_sum11, _s11);
                        _sum20 = _mm_add_epi32(_sum20, _s20);
                        _sum21 = _mm_add_epi32(_sum21, _s21);
                        _sum30 = _mm_add_epi32(_sum30, _s30);
                        _sum31 = _mm_add_epi32(_sum31, _s31);

                        m0++;
                        m1++;
                        m2++;
                        m3++;
                        kptr += 8;
                    }

                    // dequantize and relu
                    __m128 _scale_in0 = _mm_loadu_ps((const float*)scale_in_data + p * 8);
                    __m128 _scale_in1 = _mm_loadu_ps((const float*)scale_in_data + p * 8 + 4);

                    __m128 _sumfp32_00 = _mm_cvtepi32_ps(_sum00);
                    __m128 _sumfp32_01 = _mm_cvtepi32_ps(_sum01);
                    __m128 _sumfp32_10 = _mm_cvtepi32_ps(_sum10);
                    __m128 _sumfp32_11 = _mm_cvtepi32_ps(_sum11);
                    __m128 _sumfp32_20 = _mm_cvtepi32_ps(_sum20);
                    __m128 _sumfp32_21 = _mm_cvtepi32_ps(_sum21);
                    __m128 _sumfp32_30 = _mm_cvtepi32_ps(_sum30);
                    __m128 _sumfp32_31 = _mm_cvtepi32_ps(_sum31);
                    if (bias_term)
                    {
                        __m128 _bias0 = _mm_loadu_ps((const float*)bias_data + p * 8);
                        __m128 _bias1 = _mm_loadu_ps((const float*)bias_data + p * 8 + 4);
                        _sumfp32_00 = _mm_add_ps(_bias0, _mm_mul_ps(_sumfp32_00, _scale_in0));
                        _sumfp32_01 = _mm_add_ps(_bias1, _mm_mul_ps(_sumfp32_01, _scale_in1));
                        _sumfp32_10 = _mm_add_ps(_bias0, _mm_mul_ps(_sumfp32_10, _scale_in0));
                        _sumfp32_11 = _mm_add_ps(_bias1, _mm_mul_ps(_sumfp32_11, _scale_in1));
                        _sumfp32_20 = _mm_add_ps(_bias0, _mm_mul_ps(_sumfp32_20, _scale_in0));
                        _sumfp32_21 = _mm_add_ps(_bias1, _mm_mul_ps(_sumfp32_21, _scale_in1));
                        _sumfp32_30 = _mm_add_ps(_bias0, _mm_mul_ps(_sumfp32_30, _scale_in0));
                        _sumfp32_31 = _mm_add_ps(_bias1, _mm_mul_ps(_sumfp32_31, _scale_in1));
                    }
                    else
                    {
                        _sumfp32_00 = _mm_mul_ps(_sumfp32_00, _scale_in0);
                        _sumfp32_01 = _mm_mul_ps(_sumfp32_01, _scale_in1);
                        _sumfp32_10 = _mm_mul_ps(_sumfp32_10, _scale_in0);
                        _sumfp32_11 = _mm_mul_ps(_sumfp32_11, _scale_in1);
                        _sumfp32_20 = _mm_mul_ps(_sumfp32_20, _scale_in0);
                        _sumfp32_21 = _mm_mul_ps(_sumfp32_21, _scale_in1);
                        _sumfp32_30 = _mm_mul_ps(_sumfp32_30, _scale_in0);
                        _sumfp32_31 = _mm_mul_ps(_sumfp32_31, _scale_in1);
                    }

                    _sumfp32_00 = activation_sse(_sumfp32_00, activation_type, activation_params);
                    _sumfp32_01 = activation_sse(_sumfp32_01, activation_type, activation_params);
                    _sumfp32_10 = activation_sse(_sumfp32_10, activation_type, activation_params);
                    _sumfp32_11 = activation_sse(_sumfp32_11, activation_type, activation_params);
                    _sumfp32_20 = activation_sse(_sumfp32_20, activation_type, activation_params);
                    _sumfp32_21 = activation_sse(_sumfp32_21, activation_type, activation_params);
                    _sumfp32_30 = activation_sse(_sumfp32_30, activation_type, activation_params);
                    _sumfp32_31 = activation_sse(_sumfp32_31, activation_type, activation_params);

                    // transpose 4x8
                    _MM_TRANSPOSE4_PS(_sumfp32_00, _sumfp32_10, _sumfp32_20, _sumfp32_30);
                    _MM_TRANSPOSE4_PS(_sumfp32_01, _sumfp32_11, _sumfp32_21, _sumfp32_31);

                    _mm_storeu_ps(outptr, _sumfp32_00);
                    _mm_storeu_ps(outptr + 4, _sumfp32_10);
                    _mm_storeu_ps(outptr + 8, _sumfp32_20);
                    _mm_storeu_ps(outptr + 12, _sumfp32_30);
                    _mm_storeu_ps(outptr + 16, _sumfp32_01);
                    _mm_storeu_ps(outptr + 20, _sumfp32_11);
                    _mm_storeu_ps(outptr + 24, _sumfp32_21);
                    _mm_storeu_ps(outptr + 28, _sumfp32_31);

                    outptr += 32;
                }
            }
        }

        if (num_output_elempack == 1 && out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < outh; j++)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const signed char* kptr = weight_data_tm.row<const signed char>(p);
                    const signed char* m0 = bottom_blob_int8_unpacked.row<const signed char>(j * 4);
                    const signed char* m1 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 1);
                    const signed char* m2 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 2);
                    const signed char* m3 = bottom_blob_int8_unpacked.row<const signed char>(j * 4 + 3);

                    int sum0 = 0;
                    int sum1 = 0;
                    int sum2 = 0;
                    int sum3 = 0;

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        sum0 += *m0++ * kptr[0];
                        sum1 += *m1++ * kptr[0];
                        sum2 += *m2++ * kptr[0];
                        sum3 += *m3++ * kptr[0];
                        kptr += 1;
                    }

                    // dequantize and relu
                    float sumfp32_0 = sum0 * scale_in_data[p];
                    float sumfp32_1 = sum1 * scale_in_data[p];
                    float sumfp32_2 = sum2 * scale_in_data[p];
                    float sumfp32_3 = sum3 * scale_in_data[p];

                    if (bias_term)
                    {
                        sumfp32_0 += bias_data[p];
                        sumfp32_1 += bias_data[p];
                        sumfp32_2 += bias_data[p];
                        sumfp32_3 += bias_data[p];
                    }

                    outptr[0] = activation_ss(sumfp32_0, activation_type, activation_params);
                    outptr[1] = activation_ss(sumfp32_1, activation_type, activation_params);
                    outptr[2] = activation_ss(sumfp32_2, activation_type, activation_params);
                    outptr[3] = activation_ss(sumfp32_3, activation_type, activation_params);
                    outptr += 4;
                }
            }
        }

        if (num_output_elempack == 8 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < outh; j++)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const signed char* kptr = weight_data_tm.row<const signed char>(p);
                    const signed char* m = bottom_blob_int8_unpacked.row<const signed char>(j);

                    __m128i _sum0 = _mm_setzero_si128();
                    __m128i _sum1 = _mm_setzero_si128();

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        __m128i _val = _mm_set1_epi16((short)m[0]);

                        // TODO use _mm_cvtepi8_epi16 on sse4.1
                        __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
                        _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));

                        __m128i _sl = _mm_mullo_epi16(_val, _w);
                        __m128i _sh = _mm_mulhi_epi16(_val, _w);
                        __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                        __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);

                        _sum0 = _mm_add_epi32(_sum0, _s0);
                        _sum1 = _mm_add_epi32(_sum1, _s1);

                        m++;
                        kptr += 8;
                    }

                    // dequantize and relu
                    __m128 _scale_in0 = _mm_loadu_ps((const float*)scale_in_data + p * 8);
                    __m128 _scale_in1 = _mm_loadu_ps((const float*)scale_in_data + p * 8 + 4);

                    __m128 _sumfp32_0 = _mm_cvtepi32_ps(_sum0);
                    __m128 _sumfp32_1 = _mm_cvtepi32_ps(_sum1);

                    if (bias_term)
                    {
                        __m128 _bias0 = _mm_loadu_ps((const float*)bias_data + p * 8);
                        __m128 _bias1 = _mm_loadu_ps((const float*)bias_data + p * 8 + 4);
                        _sumfp32_0 = _mm_add_ps(_bias0, _mm_mul_ps(_sumfp32_0, _scale_in0));
                        _sumfp32_1 = _mm_add_ps(_bias1, _mm_mul_ps(_sumfp32_1, _scale_in1));
                    }
                    else
                    {
                        _sumfp32_0 = _mm_mul_ps(_sumfp32_0, _scale_in0);
                        _sumfp32_1 = _mm_mul_ps(_sumfp32_1, _scale_in1);
                    }

                    _sumfp32_0 = activation_sse(_sumfp32_0, activation_type, activation_params);
                    _sumfp32_1 = activation_sse(_sumfp32_1, activation_type, activation_params);

                    _mm_storeu_ps(outptr, _sumfp32_0);
                    _mm_storeu_ps(outptr + 4, _sumfp32_1);
                    outptr += 8;
                }
            }
        }
#endif // __SSE2__

        if (num_output_elempack == 1 && out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < outh; j++)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const signed char* kptr = weight_data_tm.row<const signed char>(p);
                    const signed char* m = bottom_blob_int8_unpacked.row<const signed char>(j);

                    int sum = 0;

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        sum += *m++ * *kptr++;
                    }

                    // dequantize and relu
                    float sumfp32 = sum * scale_in_data[p];

                    if (bias_term)
                        sumfp32 += bias_data[p];

                    outptr[0] = activation_ss(sumfp32, activation_type, activation_params);
                    outptr += 1;
                }
            }
        }

        return 0;
    }

    Mat bottom_blob_int8_flattened = bottom_blob_int8;
    if (bottom_blob_int8.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;
        flatten->forward(bottom_blob_int8, bottom_blob_int8_flattened, opt_flatten);
    }

    //     int elempack = bottom_blob_int8_flattened.elempack;

    int out_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 8 == 0 ? 8 : 1;
    }
#endif // __SSE2__
    //     size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, (size_t)(4u * out_elempack), out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __SSE2__
    if (out_elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            __m128i _sum0 = _mm_setzero_si128();
            __m128i _sum1 = _mm_setzero_si128();

            const signed char* kptr = weight_data_tm.row<const signed char>(p);
            const signed char* sptr = bottom_blob_int8_flattened;

            int i = 0;
            for (; i < num_input; i++)
            {
                __m128i _val = _mm_set1_epi16((short)sptr[0]);

                // TODO use _mm_cvtepi8_epi16 on sse4.1
                __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
                _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));

                __m128i _sl = _mm_mullo_epi16(_val, _w);
                __m128i _sh = _mm_mulhi_epi16(_val, _w);
                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);

                _sum0 = _mm_add_epi32(_sum0, _s0);
                _sum1 = _mm_add_epi32(_sum1, _s1);

                sptr += 1;
                kptr += 8;
            }

            // dequantize and relu
            __m128 _scale_in0 = _mm_loadu_ps((const float*)scale_in_data + p * 8);
            __m128 _scale_in1 = _mm_loadu_ps((const float*)scale_in_data + p * 8 + 4);

            __m128 _sumfp32_0 = _mm_cvtepi32_ps(_sum0);
            __m128 _sumfp32_1 = _mm_cvtepi32_ps(_sum1);

            if (bias_term)
            {
                __m128 _bias0 = _mm_loadu_ps((const float*)bias_data + p * 8);
                __m128 _bias1 = _mm_loadu_ps((const float*)bias_data + p * 8 + 4);
                _sumfp32_0 = _mm_add_ps(_bias0, _mm_mul_ps(_sumfp32_0, _scale_in0));
                _sumfp32_1 = _mm_add_ps(_bias1, _mm_mul_ps(_sumfp32_1, _scale_in1));
            }
            else
            {
                _sumfp32_0 = _mm_mul_ps(_sumfp32_0, _scale_in0);
                _sumfp32_1 = _mm_mul_ps(_sumfp32_1, _scale_in1);
            }

            _sumfp32_0 = activation_sse(_sumfp32_0, activation_type, activation_params);
            _sumfp32_1 = activation_sse(_sumfp32_1, activation_type, activation_params);

            float* outptr = (float*)top_blob + p * 8;
            _mm_storeu_ps(outptr, _sumfp32_0);
            _mm_storeu_ps(outptr + 4, _sumfp32_1);
        }
    }
#endif // __SSE2__

    if (out_elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            int sum = 0;

            const signed char* kptr = weight_data_tm.row<const signed char>(p);
            const signed char* sptr = bottom_blob_int8_flattened;

            int i = 0;
            for (; i < num_input; i++)
            {
                signed char val = sptr[0];

                signed char w = kptr[0];

                sum += val * w;

                sptr += 1;
                kptr += 1;
            }

            // dequantize and relu
            float sumfp32 = sum * scale_in_data[p];

            if (bias_term)
                sumfp32 += bias_data[p];

            sumfp32 = activation_ss(sumfp32, activation_type, activation_params);

            top_blob[p] = sumfp32;
        }
    }

    return 0;
}
#endif // NCNN_INT8

} // namespace ncnn
