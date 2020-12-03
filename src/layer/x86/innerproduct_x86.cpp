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
#include "sse_activation.h"
#include "sse_usability.h"

#if __AVX__
#include <immintrin.h>
#include "avx_activation.h"
#include "avx_usability.h"
#endif
#endif // __SSE2__

#include "layer_type.h"

namespace ncnn {

InnerProduct_x86::InnerProduct_x86()
{
#if __SSE2__
    support_packing = true;
#if __AVX__
    support_weight_fp16_storage = true;
#endif
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
#if __AVX__
    if (opt.use_weight_fp16_storage && weight_data.elemsize == 4u)
    {
        ncnn::cast_float32_to_float16(weight_data, weight_data_fp16, opt);

        return 0;
    }
#endif

    const int num_input = weight_data_size / num_output;

    int elempack = 1;
    int out_elempack = 1;

#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX__
        elempack = num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        elempack = num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

    if (elempack == 1 && out_elempack == 1)
    {
        return 0;
    }

    // src = inch-outch
    // dst = pb-pa-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_packed.create(num_input / elempack, num_output / out_elempack, (size_t)4u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g0 = weight_data_packed.row(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int i = 0; i < elempack; i++)
                {
                    for (int j = 0; j < out_elempack; j++)
                    {
                        *g0++ = weight_data_r2.row(q + j)[p + i];
                    }
                }
            }
        }
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
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        // TODO
        return InnerProduct::forward(bottom_blob, top_blob, opt);
    }

#if __AVX__
    if (opt.use_weight_fp16_storage)
    {
        return forward_fp16(bottom_blob, top_blob, opt);
    }
#endif // __AVX__

    // flatten
    Mat bottom_blob_flattened = bottom_blob;
    if (bottom_blob.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
    }

    int size = bottom_blob_flattened.w;
    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

    int out_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX__
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
    if (elempack == 8 && out_elempack == 8)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            __m256 _sum = _mm256_set1_ps(0.f);

            if (bias_term)
            {
                _sum = _mm256_loadu_ps((const float*)bias_data + p * 8);
            }

            const float* kptr = weight_data_packed.row(p);

            const float* sptr = bottom_blob_flattened;

            for (int i = 0; i < size; i++)
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
                _sum = _mm256_fmadd_ps(_val0, _w0, _sum);
                __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                _sum = _mm256_fmadd_ps(_val1, _w1, _sum);
                __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                _sum = _mm256_fmadd_ps(_val2, _w2, _sum);
                __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                _sum = _mm256_fmadd_ps(_val3, _w3, _sum);
                __m256 _w4 = _mm256_loadu_ps(kptr + 32);
                _sum = _mm256_fmadd_ps(_val4, _w4, _sum);
                __m256 _w5 = _mm256_loadu_ps(kptr + 40);
                _sum = _mm256_fmadd_ps(_val5, _w5, _sum);
                __m256 _w6 = _mm256_loadu_ps(kptr + 48);
                _sum = _mm256_fmadd_ps(_val6, _w6, _sum);
                __m256 _w7 = _mm256_loadu_ps(kptr + 56);
                _sum = _mm256_fmadd_ps(_val7, _w7, _sum);

                sptr += 8;
                kptr += 64;
            }

            _sum = activation_ps(_sum, activation_type, activation_params);

            float* outptr = top_blob;
            _mm256_storeu_ps(outptr + p * 8, _sum);
        }
    }

    if (elempack == 1 && out_elempack == 8)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            __m256 _sum = _mm256_set1_ps(0.f);

            if (bias_term)
            {
                _sum = _mm256_loadu_ps((const float*)bias_data + p * 8);
            }

            const float* kptr = weight_data_packed.row(p);

            const float* sptr = bottom_blob_flattened;

            for (int i = 0; i < size; i++)
            {
                __m256 _val = _mm256_set1_ps(sptr[0]);
                __m256 _w = _mm256_loadu_ps(kptr);
                _sum = _mm256_fmadd_ps(_val, _w, _sum);

                sptr += 1;
                kptr += 8;
            }

            _sum = activation_ps(_sum, activation_type, activation_params);

            float* outptr = top_blob;
            _mm256_storeu_ps(outptr + p * 8, _sum);
        }
    }

    if (elempack == 4 && out_elempack == 8)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            __m256 _sum = _mm256_set1_ps(0.f);

            if (bias_term)
            {
                _sum = _mm256_loadu_ps((const float*)bias_data + p * 8);
            }

            const float* kptr = weight_data_packed.row(p);

            const float* sptr = bottom_blob_flattened;

            for (int i = 0; i < size; i++)
            {
                __m256 _val0 = _mm256_broadcast_ss(sptr);
                __m256 _val1 = _mm256_broadcast_ss(sptr + 1);
                __m256 _val2 = _mm256_broadcast_ss(sptr + 2);
                __m256 _val3 = _mm256_broadcast_ss(sptr + 3);

                __m256 _w0 = _mm256_loadu_ps(kptr);
                _sum = _mm256_fmadd_ps(_val0, _w0, _sum);
                __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                _sum = _mm256_fmadd_ps(_val1, _w1, _sum);
                __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                _sum = _mm256_fmadd_ps(_val2, _w2, _sum);
                __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                _sum = _mm256_fmadd_ps(_val3, _w3, _sum);

                sptr += 4;
                kptr += 32;
            }

            _sum = activation_ps(_sum, activation_type, activation_params);

            float* outptr = top_blob;
            _mm256_storeu_ps(outptr + p * 8, _sum);
        }
    }

    if (elempack == 8 && out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float sum = 0.f;

            if (bias_term)
            {
                sum = bias_data[p];
            }

            const float* kptr = weight_data_packed.row(p);

            const float* sptr = bottom_blob_flattened;

            __m256 _sum = _mm256_set1_ps(0.f);

            for (int i = 0; i < size; i++)
            {
                __m256 _val = _mm256_loadu_ps(sptr);
                __m256 _w = _mm256_loadu_ps(kptr);
                _sum = _mm256_fmadd_ps(_val, _w, _sum);

                sptr += 8;
                kptr += 8;
            }

            sum += _mm256_reduce_add_ps(_sum); // dot

            sum = activation_ss(sum, activation_type, activation_params);

            float* outptr = top_blob;
            outptr[p] = sum;
        }
    }

    if (elempack == 8 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            __m128 _sum = _mm_set1_ps(0.f);

            if (bias_term)
            {
                _sum = _mm_loadu_ps((const float*)bias_data + p * 4);
            }

            const float* kptr = weight_data_packed.row(p);

            const float* sptr = bottom_blob_flattened;

            for (int i = 0; i < size; i++)
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
                _sum = _mm_fmadd_ps(_val0, _w0, _sum);
                __m128 _w1 = _mm_loadu_ps(kptr + 4);
                _sum = _mm_fmadd_ps(_val1, _w1, _sum);
                __m128 _w2 = _mm_loadu_ps(kptr + 8);
                _sum = _mm_fmadd_ps(_val2, _w2, _sum);
                __m128 _w3 = _mm_loadu_ps(kptr + 12);
                _sum = _mm_fmadd_ps(_val3, _w3, _sum);
                __m128 _w4 = _mm_loadu_ps(kptr + 16);
                _sum = _mm_fmadd_ps(_val4, _w4, _sum);
                __m128 _w5 = _mm_loadu_ps(kptr + 20);
                _sum = _mm_fmadd_ps(_val5, _w5, _sum);
                __m128 _w6 = _mm_loadu_ps(kptr + 24);
                _sum = _mm_fmadd_ps(_val6, _w6, _sum);
                __m128 _w7 = _mm_loadu_ps(kptr + 28);
                _sum = _mm_fmadd_ps(_val7, _w7, _sum);

                sptr += 8;
                kptr += 32;
            }

            _sum = activation_ps(_sum, activation_type, activation_params);

            float* outptr = top_blob;
            _mm_storeu_ps(outptr + p * 4, _sum);
        }
    }
#endif // __AVX__

    if (elempack == 4 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            __m128 _sum = _mm_set1_ps(0.f);

            if (bias_term)
            {
                _sum = _mm_loadu_ps((const float*)bias_data + p * 4);
            }

            const float* kptr = weight_data_packed.row(p);

            const float* sptr = bottom_blob_flattened;

            for (int i = 0; i < size; i++)
            {
                __m128 _val0 = _mm_set1_ps(sptr[0]);
                __m128 _val1 = _mm_set1_ps(sptr[1]);
                __m128 _val2 = _mm_set1_ps(sptr[2]);
                __m128 _val3 = _mm_set1_ps(sptr[3]);

                __m128 _w0 = _mm_loadu_ps(kptr);
                _sum = _mm_add_ps(_mm_mul_ps(_val0, _w0), _sum);
                __m128 _w1 = _mm_loadu_ps(kptr + 4);
                _sum = _mm_add_ps(_mm_mul_ps(_val1, _w1), _sum);
                __m128 _w2 = _mm_loadu_ps(kptr + 8);
                _sum = _mm_add_ps(_mm_mul_ps(_val2, _w2), _sum);
                __m128 _w3 = _mm_loadu_ps(kptr + 12);
                _sum = _mm_add_ps(_mm_mul_ps(_val3, _w3), _sum);

                sptr += 4;
                kptr += 16;
            }

            _sum = activation_ps(_sum, activation_type, activation_params);

            float* outptr = top_blob;
            _mm_storeu_ps(outptr + p * 4, _sum);
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            __m128 _sum = _mm_set1_ps(0.f);

            if (bias_term)
            {
                _sum = _mm_loadu_ps((const float*)bias_data + p * 4);
            }

            const float* kptr = weight_data_packed.row(p);

            const float* sptr = bottom_blob_flattened;

            for (int i = 0; i < size; i++)
            {
                __m128 _val = _mm_set1_ps(sptr[0]);
                __m128 _w = _mm_loadu_ps(kptr);
                _sum = _mm_add_ps(_mm_mul_ps(_val, _w), _sum);

                sptr += 1;
                kptr += 4;
            }

            _sum = activation_ps(_sum, activation_type, activation_params);

            float* outptr = top_blob;
            _mm_storeu_ps(outptr + p * 4, _sum);
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            float sum = 0.f;

            if (bias_term)
            {
                sum = bias_data[p];
            }

            const float* kptr = weight_data_packed.row(p);

            const float* sptr = bottom_blob_flattened;

            __m128 _sum = _mm_set1_ps(0.f);

            for (int i = 0; i < size; i++)
            {
                __m128 _val = _mm_loadu_ps(sptr);
                __m128 _w = _mm_loadu_ps(kptr);
                _sum = _mm_add_ps(_mm_mul_ps(_val, _w), _sum);

                sptr += 4;
                kptr += 4;
            }

            sum += _mm_reduce_add_ps(_sum); // dot

            sum = activation_ss(sum, activation_type, activation_params);

            float* outptr = top_blob;
            outptr[p] = sum;
        }
    }
#endif // __SSE2__

    if (elempack == 1 && out_elempack == 1)
    {
        const float* weight_data_ptr = weight_data;

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

            const float* w0 = weight_data_ptr + size * p;
            const float* w1 = weight_data_ptr + size * (p + 1);
            const float* w2 = weight_data_ptr + size * (p + 2);
            const float* w3 = weight_data_ptr + size * (p + 3);
            const float* w4 = weight_data_ptr + size * (p + 4);
            const float* w5 = weight_data_ptr + size * (p + 5);
            const float* w6 = weight_data_ptr + size * (p + 6);
            const float* w7 = weight_data_ptr + size * (p + 7);

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
            for (; i + 7 < size; i += 8)
            {
                __m256 _m = _mm256_loadu_ps(m);

                __m256 _w0 = _mm256_loadu_ps(w0);
                _sum0 = _mm256_fmadd_ps(_m, _w0, _sum0);

                __m256 _w1 = _mm256_loadu_ps(w1);
                _sum1 = _mm256_fmadd_ps(_m, _w1, _sum1);

                __m256 _w2 = _mm256_loadu_ps(w2);
                _sum2 = _mm256_fmadd_ps(_m, _w2, _sum2);

                __m256 _w3 = _mm256_loadu_ps(w3);
                _sum3 = _mm256_fmadd_ps(_m, _w3, _sum3);

                __m256 _w4 = _mm256_loadu_ps(w4);
                _sum4 = _mm256_fmadd_ps(_m, _w4, _sum4);

                __m256 _w5 = _mm256_loadu_ps(w5);
                _sum5 = _mm256_fmadd_ps(_m, _w5, _sum5);

                __m256 _w6 = _mm256_loadu_ps(w6);
                _sum6 = _mm256_fmadd_ps(_m, _w6, _sum6);

                __m256 _w7 = _mm256_loadu_ps(w7);
                _sum7 = _mm256_fmadd_ps(_m, _w7, _sum7);

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
            for (; i < size; i++)
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
            _sums = activation_ps(_sums, activation_type, activation_params);

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

            const float* w0 = weight_data_ptr + size * p;
            const float* w1 = weight_data_ptr + size * (p + 1);
            const float* w2 = weight_data_ptr + size * (p + 2);
            const float* w3 = weight_data_ptr + size * (p + 3);

            const float* m = bottom_blob_flattened;

            int i = 0;
#if __AVX__
            __m256 _sum0 = _mm256_set1_ps(0.f);
            __m256 _sum1 = _mm256_set1_ps(0.f);
            __m256 _sum2 = _mm256_set1_ps(0.f);
            __m256 _sum3 = _mm256_set1_ps(0.f);
            for (; i + 7 < size; i += 8)
            {
                __m256 _m = _mm256_loadu_ps(m);

                __m256 _w0 = _mm256_loadu_ps(w0);
                _sum0 = _mm256_fmadd_ps(_m, _w0, _sum0);

                __m256 _w1 = _mm256_loadu_ps(w1);
                _sum1 = _mm256_fmadd_ps(_m, _w1, _sum1);

                __m256 _w2 = _mm256_loadu_ps(w2);
                _sum2 = _mm256_fmadd_ps(_m, _w2, _sum2);

                __m256 _w3 = _mm256_loadu_ps(w3);
                _sum3 = _mm256_fmadd_ps(_m, _w3, _sum3);

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
            for (; i + 3 < size; i += 4)
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
            for (; i < size; i++)
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
            _sums = activation_ps(_sums, activation_type, activation_params);

            float* outptr = top_blob;
            _mm_storeu_ps(outptr + p, _sums);
        }

        remain_num_output_start += (nn_num_output << 2);
#else
        int remain_num_output_start = 0;
#endif // __SSE2__

        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_num_output_start; p < num_output; p++)
        {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            const float* w = weight_data_ptr + size * p;

            const float* m = bottom_blob_flattened;

            int i = 0;
#if __SSE2__
#if __AVX__
            __m256 _sum = _mm256_set1_ps(0.f);
            for (; i + 7 < size; i += 8)
            {
                __m256 _m = _mm256_loadu_ps(m);

                __m256 _w = _mm256_loadu_ps(w);
                _sum = _mm256_fmadd_ps(_m, _w, _sum);

                m += 8;
                w += 8;
            }
#endif // __AVX__
            __m128 _suml = _mm_set1_ps(0.f);
            for (; i + 3 < size; i += 4)
            {
                __m128 _m = _mm_loadu_ps(m);

                __m128 _w = _mm_loadu_ps(w);
                _suml = _mm_add_ps(_mm_mul_ps(_m, _w), _suml);

                m += 4;
                w += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
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

            if (activation_type == 1)
            {
                sum = std::max(sum, 0.f);
            }
            else if (activation_type == 2)
            {
                float slope = activation_params[0];
                sum = sum > 0.f ? sum : sum * slope;
            }
            else if (activation_type == 3)
            {
                float min = activation_params[0];
                float max = activation_params[1];
                if (sum < min)
                    sum = min;
                if (sum > max)
                    sum = max;
            }
            else if (activation_type == 4)
            {
                sum = static_cast<float>(1.f / (1.f + exp(-sum)));
            }
            else if (activation_type == 5)
            {
                sum = static_cast<float>(sum * tanh(log(exp(sum) + 1.f)));
            }

            float* outptr = top_blob;
            outptr[p] = sum;
        }
    }

    return 0;
}
#if __AVX__

int InnerProduct_x86::forward_fp16(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // flatten
    Mat bottom_blob_flattened = bottom_blob;
    if (bottom_blob.dims != 1)
    {
        Option opt_flatten = opt;
        opt_flatten.blob_allocator = opt.workspace_allocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, opt_flatten);
    }

    // pack1
    {
        bottom_blob_flattened.w *= bottom_blob_flattened.elempack;
        bottom_blob_flattened.cstep = bottom_blob_flattened.w;
        bottom_blob_flattened.elemsize = 4u;
        bottom_blob_flattened.elempack = 1;
    }

    int w = bottom_blob_flattened.w;
    int h = bottom_blob_flattened.h;
    size_t elemsize = bottom_blob_flattened.elemsize;
    int size = w * h;
    top_blob.create(num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const unsigned short* weight_data_ptr = (const unsigned short*)weight_data_fp16;
    float* output_ptr = top_blob;
    int nn_num_output = num_output >> 3;
    int remain_num_output_start = nn_num_output << 3;

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
        __m256 _sum0 = _mm256_set1_ps(0.f);
        __m256 _sum1 = _mm256_set1_ps(0.f);
        __m256 _sum2 = _mm256_set1_ps(0.f);
        __m256 _sum3 = _mm256_set1_ps(0.f);
        __m256 _sum4 = _mm256_set1_ps(0.f);
        __m256 _sum5 = _mm256_set1_ps(0.f);
        __m256 _sum6 = _mm256_set1_ps(0.f);
        __m256 _sum7 = _mm256_set1_ps(0.f);

        const unsigned short* w0 = (const unsigned short*)weight_data_ptr + size * p;
        const unsigned short* w1 = (const unsigned short*)weight_data_ptr + size * (p + 1);
        const unsigned short* w2 = (const unsigned short*)weight_data_ptr + size * (p + 2);
        const unsigned short* w3 = (const unsigned short*)weight_data_ptr + size * (p + 3);
        const unsigned short* w4 = (const unsigned short*)weight_data_ptr + size * (p + 4);
        const unsigned short* w5 = (const unsigned short*)weight_data_ptr + size * (p + 5);
        const unsigned short* w6 = (const unsigned short*)weight_data_ptr + size * (p + 6);
        const unsigned short* w7 = (const unsigned short*)weight_data_ptr + size * (p + 7);

        const float* m = bottom_blob_flattened;
        int nn = size >> 3;
        int remain = size & 7;

        for (; nn > 0; nn--)
        {
            __m256 _m = _mm256_loadu_ps(m);

            __m256 _w0 = loadfp16(w0);
            _sum0 = _mm256_fmadd_ps(_m, _w0, _sum0);

            __m256 _w1 = loadfp16(w1);
            _sum1 = _mm256_fmadd_ps(_m, _w1, _sum1);

            __m256 _w2 = loadfp16(w2);
            _sum2 = _mm256_fmadd_ps(_m, _w2, _sum2);

            __m256 _w3 = loadfp16(w3);
            _sum3 = _mm256_fmadd_ps(_m, _w3, _sum3);

            __m256 _w4 = loadfp16(w4);
            _sum4 = _mm256_fmadd_ps(_m, _w4, _sum4);

            __m256 _w5 = loadfp16(w5);
            _sum5 = _mm256_fmadd_ps(_m, _w5, _sum5);

            __m256 _w6 = loadfp16(w6);
            _sum6 = _mm256_fmadd_ps(_m, _w6, _sum6);

            __m256 _w7 = loadfp16(w7);
            _sum7 = _mm256_fmadd_ps(_m, _w7, _sum7);

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
        if (remain != 0)
        {
            unsigned short fp16_weights[8][8] = {{0}};
            float _m_f[8] = {0};
            int i = 0;
            // No fast way to convert to fp32 one element at the time
            // so batch an 8 lane vector.
            for (; remain > 0; remain--)
            {
                _m_f[i] = *m;
                fp16_weights[0][i] = *w0;
                fp16_weights[1][i] = *w1;
                fp16_weights[2][i] = *w2;
                fp16_weights[3][i] = *w3;
                fp16_weights[4][i] = *w4;
                fp16_weights[5][i] = *w5;
                fp16_weights[6][i] = *w6;
                fp16_weights[7][i] = *w7;
                i++;
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
            __m256 _m = _mm256_loadu_ps(_m_f);

            __m256 _w0 = loadfp16(fp16_weights[0]);
            _sum0 = _mm256_fmadd_ps(_m, _w0, _sum0);

            __m256 _w1 = loadfp16(fp16_weights[1]);
            _sum1 = _mm256_fmadd_ps(_m, _w1, _sum1);

            __m256 _w2 = loadfp16(fp16_weights[2]);
            _sum2 = _mm256_fmadd_ps(_m, _w2, _sum2);

            __m256 _w3 = loadfp16(fp16_weights[3]);
            _sum3 = _mm256_fmadd_ps(_m, _w3, _sum3);

            __m256 _w4 = loadfp16(fp16_weights[4]);
            _sum4 = _mm256_fmadd_ps(_m, _w4, _sum4);

            __m256 _w5 = loadfp16(fp16_weights[5]);
            _sum5 = _mm256_fmadd_ps(_m, _w5, _sum5);

            __m256 _w6 = loadfp16(fp16_weights[6]);
            _sum6 = _mm256_fmadd_ps(_m, _w6, _sum6);

            __m256 _w7 = loadfp16(fp16_weights[7]);
            _sum7 = _mm256_fmadd_ps(_m, _w7, _sum7);
        }

        __m256 _sums = HorizontalSums(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
        __m256 _sums_f = _mm256_loadu_ps(sums);
        _sums = activation_ps(_mm256_add_ps(_sums_f, _sums), activation_type, activation_params);
        _mm256_storeu_ps(output_ptr + p, _sums);
    }

    nn_num_output = (num_output - remain_num_output_start) >> 2;
    int nn_offset = remain_num_output_start;
    remain_num_output_start += (nn_num_output << 2);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_num_output; pp++)
    {
        int p = nn_offset + (pp * 4);

        float sums[4] = {0.0f};
        if (bias_term)
        {
            sums[0] = bias_data[p];
            sums[1] = bias_data[p + 1];
            sums[2] = bias_data[p + 2];
            sums[3] = bias_data[p + 3];
        }
        __m256 _sum0 = _mm256_set1_ps(0.f);
        __m256 _sum1 = _mm256_set1_ps(0.f);
        __m256 _sum2 = _mm256_set1_ps(0.f);
        __m256 _sum3 = _mm256_set1_ps(0.f);

        const unsigned short* w0 = (const unsigned short*)weight_data_ptr + size * p;
        const unsigned short* w1 = (const unsigned short*)weight_data_ptr + size * (p + 1);
        const unsigned short* w2 = (const unsigned short*)weight_data_ptr + size * (p + 2);
        const unsigned short* w3 = (const unsigned short*)weight_data_ptr + size * (p + 3);

        const float* m = bottom_blob_flattened;
        int nn = size >> 3;
        int remain = size & 7;

        for (; nn > 0; nn--)
        {
            __m256 _m = _mm256_loadu_ps(m);

            __m256 _w0 = loadfp16(w0);
            _sum0 = _mm256_fmadd_ps(_m, _w0, _sum0);

            __m256 _w1 = loadfp16(w1);
            _sum1 = _mm256_fmadd_ps(_m, _w1, _sum1);

            __m256 _w2 = loadfp16(w2);
            _sum2 = _mm256_fmadd_ps(_m, _w2, _sum2);

            __m256 _w3 = loadfp16(w3);
            _sum3 = _mm256_fmadd_ps(_m, _w3, _sum3);

            m += 8;
            w0 += 8;
            w1 += 8;
            w2 += 8;
            w3 += 8;
        }
        if (remain != 0)
        {
            unsigned short fp16_weights[4][8] = {{0}};
            float _m_f[8] = {0};
            int i = 0;
            for (; remain > 0; remain--)
            {
                _m_f[i] = *m;
                fp16_weights[0][i] = *w0;
                fp16_weights[1][i] = *w1;
                fp16_weights[2][i] = *w2;
                fp16_weights[3][i] = *w3;
                i++;
                m++;
                w0++;
                w1++;
                w2++;
                w3++;
            }
            __m256 _m = _mm256_loadu_ps(_m_f);

            __m256 _w0 = loadfp16(fp16_weights[0]);
            _sum0 = _mm256_fmadd_ps(_m, _w0, _sum0);

            __m256 _w1 = loadfp16(fp16_weights[1]);
            _sum1 = _mm256_fmadd_ps(_m, _w1, _sum1);

            __m256 _w2 = loadfp16(fp16_weights[2]);
            _sum2 = _mm256_fmadd_ps(_m, _w2, _sum2);

            __m256 _w3 = loadfp16(fp16_weights[3]);
            _sum3 = _mm256_fmadd_ps(_m, _w3, _sum3);
        }

        __m128 _sums = HorizontalSums(_sum0, _sum1, _sum2, _sum3);
        __m256 _sums_a = activation_ps(_mm256_castps128_ps256(_mm_add_ps(_mm_loadu_ps(sums), _sums)), activation_type, activation_params);
        _mm_storeu_ps(output_ptr + p, _mm256_castps256_ps128(_sums_a));
    }

// num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_num_output_start; p < num_output; p++)
    {
        float sum = 0.f;

        if (bias_term)
            sum = bias_data[p];

        const unsigned short* w = (const unsigned short*)weight_data_ptr + size * p;

        __m256 _sum = _mm256_set1_ps(0.f);

        const float* m = bottom_blob_flattened;

        int nn = size >> 3;
        int remain = size & 7;
        for (; nn > 0; nn--)
        {
            __m256 _m = _mm256_loadu_ps(m);

            __m256 _w = loadfp16(w);
            _sum = _mm256_fmadd_ps(_m, _w, _sum);

            m += 8;
            w += 8;
        }
        if (remain != 0)
        {
            unsigned short fp16_weights[8] = {0};
            float _m_f[8] = {0};
            int i = 0;
            for (; remain > 0; remain--)
            {
                _m_f[i] = *m;
                fp16_weights[i] = *w;
                i++;
                m++;
                w++;
            }
            __m256 _m = _mm256_loadu_ps(_m_f);

            __m256 _w = loadfp16(fp16_weights);
            _sum = _mm256_fmadd_ps(_m, _w, _sum);
        }

        sum += _mm256_reduce_add_ps(_sum);
        sum = activation_ss(sum, activation_type, activation_params);

        output_ptr[p] = sum;
    }
    return 0;
}
#endif // __AVX__

} // namespace ncnn
