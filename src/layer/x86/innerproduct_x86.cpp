// Tencent is pleased to support the open source community by making ncnn
// available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#ifdef __AVX__
#include "avx_activation.h"
#include "avx_usability.h"
#endif // __AVX__

#include "innerproduct_x86.h"

#include "layer_type.h"

namespace ncnn {

InnerProduct_x86::InnerProduct_x86()
{
#if __AVX__
    support_packing = true;
    support_weight_fp16_storage = true;
#endif // __AVX__

    flatten = 0;
}

int InnerProduct_x86::create_pipeline(const Option& opt)
{
#if __AVX__
    if (opt.use_packing_layout)
    {
        flatten = ncnn::create_layer(ncnn::LayerType::Flatten);

        ncnn::ParamDict pd;

        flatten->load_param(pd);

        flatten->create_pipeline(opt);
    }
    if (opt.use_weight_fp16_storage && weight_data.elemsize == 4u)
    {
        ncnn::cast_float32_to_float16(weight_data, weight_data_fp16, opt);
    }
#else
    (void)(opt);
#endif // __AVX__

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
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

    if (elempack == 8)
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

        return forward(bottom_blob_flattened, top_blob, opt);
    }

    if (opt.use_weight_fp16_storage)
    {
        return forward_fp16(bottom_blob, top_blob, opt);
    }

    top_blob.create(num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* weight_data_ptr = weight_data;
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

        const float* w0 = weight_data_ptr + size * channels * p;
        const float* w1 = weight_data_ptr + size * channels * (p + 1);
        const float* w2 = weight_data_ptr + size * channels * (p + 2);
        const float* w3 = weight_data_ptr + size * channels * (p + 3);
        const float* w4 = weight_data_ptr + size * channels * (p + 4);
        const float* w5 = weight_data_ptr + size * channels * (p + 5);
        const float* w6 = weight_data_ptr + size * channels * (p + 6);
        const float* w7 = weight_data_ptr + size * channels * (p + 7);

        // channels
        for (int q = 0; q < channels; q++)
        {
            const float* m = bottom_blob.channel(q);
            int nn = size >> 3;
            int remain = size & 7;

            for (; nn > 0; nn--)
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

            for (; remain > 0; remain--)
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

        const float* w0 = weight_data_ptr + size * channels * p;
        const float* w1 = weight_data_ptr + size * channels * (p + 1);
        const float* w2 = weight_data_ptr + size * channels * (p + 2);
        const float* w3 = weight_data_ptr + size * channels * (p + 3);

        // channels
        for (int q = 0; q < channels; q++)
        {
            const float* m = bottom_blob.channel(q);
            int nn = size >> 3;
            int remain = size & 7;

            for (; nn > 0; nn--)
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

            for (; remain > 0; remain--)
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

        const float* w = weight_data_ptr + size * channels * p;

        __m256 _sum = _mm256_set1_ps(0.f);
        // channels
        for (int q = 0; q < channels; q++)
        {
            const float* m = bottom_blob.channel(q);

            int nn = size >> 3;
            int remain = size & 7;
            for (; nn > 0; nn--)
            {
                __m256 _m = _mm256_loadu_ps(m);

                __m256 _w = _mm256_loadu_ps(w);
                _sum = _mm256_fmadd_ps(_m, _w, _sum);

                m += 8;
                w += 8;
            }
            for (; remain > 0; remain--)
            {
                sum += *m * *w;
                m++;
                w++;
            }
        }

        sum += _mm256_reduce_add_ps(_sum);
        sum = activation_ss(sum, activation_type, activation_params);

        output_ptr[p] = sum;
    }
    return 0;
#else
    return InnerProduct::forward(bottom_blob, top_blob, opt);
#endif // __AVX__
}
#if __AVX__

int InnerProduct_x86::forward_fp16(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
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

        const unsigned short* w0 = (const unsigned short*)weight_data_ptr + size * channels * p;
        const unsigned short* w1 = (const unsigned short*)weight_data_ptr + size * channels * (p + 1);
        const unsigned short* w2 = (const unsigned short*)weight_data_ptr + size * channels * (p + 2);
        const unsigned short* w3 = (const unsigned short*)weight_data_ptr + size * channels * (p + 3);
        const unsigned short* w4 = (const unsigned short*)weight_data_ptr + size * channels * (p + 4);
        const unsigned short* w5 = (const unsigned short*)weight_data_ptr + size * channels * (p + 5);
        const unsigned short* w6 = (const unsigned short*)weight_data_ptr + size * channels * (p + 6);
        const unsigned short* w7 = (const unsigned short*)weight_data_ptr + size * channels * (p + 7);

        // channels
        for (int q = 0; q < channels; q++)
        {
            const float* m = bottom_blob.channel(q);
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

        const unsigned short* w0 = (const unsigned short*)weight_data_ptr + size * channels * p;
        const unsigned short* w1 = (const unsigned short*)weight_data_ptr + size * channels * (p + 1);
        const unsigned short* w2 = (const unsigned short*)weight_data_ptr + size * channels * (p + 2);
        const unsigned short* w3 = (const unsigned short*)weight_data_ptr + size * channels * (p + 3);

        // channels
        for (int q = 0; q < channels; q++)
        {
            const float* m = bottom_blob.channel(q);
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

        const unsigned short* w = (const unsigned short*)weight_data_ptr + size * channels * p;

        __m256 _sum = _mm256_set1_ps(0.f);
        // channels
        for (int q = 0; q < channels; q++)
        {
            const float* m = bottom_blob.channel(q);

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
        }

        sum += _mm256_reduce_add_ps(_sum);
        sum = activation_ss(sum, activation_type, activation_params);

        output_ptr[p] = sum;
    }
    return 0;
}
#endif // __AVX__

} // namespace ncnn
