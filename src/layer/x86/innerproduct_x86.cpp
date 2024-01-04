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

#include "innerproduct_fp.h"
#include "innerproduct_gemm_fp.h"

#if NCNN_F16C && __AVX__
#define NCNN_IMPL_FP16S 1
#include "innerproduct_fp.h"
#include "innerproduct_gemm_fp.h"
#undef NCNN_IMPL_FP16S
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
        flatten = ncnn::create_layer_cpu(ncnn::LayerType::Flatten);

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

#if NCNN_F16C && __AVX__
    if (cpu_support_x86_f16c() && opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

    const int num_input = weight_data_size / num_output;

    innerproduct_transform_kernel_sse(weight_data, weight_data_tm, num_input, num_output, opt);

    weight_data.release();

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

#if NCNN_F16C && __AVX__
    if (cpu_support_x86_f16c() && opt.use_fp16_storage)
    {
        return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input)
    {
        // gemm
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;
        int elempack = bottom_blob.elempack;

        top_blob.create(num_output, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        innerproduct_gemm_sse(bottom_blob, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);

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

    innerproduct_sse(bottom_blob_flattened, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);

    return 0;
}

#if NCNN_F16C && __AVX__
int InnerProduct_x86::create_pipeline_fp16s(const Option& opt)
{
    const int num_input = weight_data_size / num_output;

    innerproduct_transform_kernel_fp16s_sse(weight_data, weight_data_tm, num_input, num_output, opt);

    weight_data.release();

    return 0;
}

int InnerProduct_x86::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input)
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

    innerproduct_fp16s_sse(bottom_blob_flattened, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);

    return 0;
}
#endif // NCNN_F16C && __AVX__

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

    weight_data.release();

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

    if (bottom_blob_int8.dims == 2 && bottom_blob_int8.w == num_input)
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
