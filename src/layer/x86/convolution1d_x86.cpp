// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "convolution1d_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif // __SSE2__
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

Convolution1D_x86::Convolution1D_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int Convolution1D_x86::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    int num_input = weight_data_size / kernel_w / num_output;

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

    // src = kw-inch-outch
    // dst = pb-pa-kw-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(kernel_w, num_input, num_output);

        weight_data_packed.create(kernel_w, num_input / elempack, num_output / out_elempack, (size_t)4u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_packed.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < kernel_w; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        for (int j = 0; j < out_elempack; j++)
                        {
                            const float* k00 = weight_data_r2.channel(q + j).row(p + i);

                            g00[0] = k00[k];

                            g00++;
                        }
                    }
                }
            }
        }
    }

    return 0;
}

int Convolution1D_x86::destroy_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Convolution1D_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __AVX512F__
    if (elempack == 16)
    {
        Mat tmp;
        convert_packing(bottom_blob, tmp, 8, opt);

        Mat tmpout;
        forward(tmp, tmpout, opt);

        convert_packing(tmpout, top_blob, 16, opt);

        return 0;
    }
#endif // __AVX512F__

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

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

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = num_output / out_elempack;

    top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __SSE2__
#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                float* outptr = top_blob.row(p);

                for (int j = 0; j < outw; j++)
                {
                    __m256 _sum = _mm256_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum = _mm256_loadu_ps(((const float*)bias_data) + p * 8);
                    }

                    const float* kptr = weight_data_packed.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const float* sptr = bottom_blob_bordered.row(q) + j * stride_w * 8;

                        for (int k = 0; k < kernel_w; k++)
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
                            __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                            __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                            __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                            __m256 _w4 = _mm256_loadu_ps(kptr + 32);
                            __m256 _w5 = _mm256_loadu_ps(kptr + 40);
                            __m256 _w6 = _mm256_loadu_ps(kptr + 48);
                            __m256 _w7 = _mm256_loadu_ps(kptr + 56);

                            _mm256_comp_fmadd_ps8(_sum,
                                                  _val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7,
                                                  _w0, _w1, _w2, _w3, _w4, _w5, _w6, _w7);

                            sptr += dilation_w * 8;
                            kptr += 64;
                        }
                    }

                    _sum = activation_avx(_sum, activation_type, activation_params);

                    _mm256_storeu_ps(outptr, _sum);
                    outptr += 8;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 8)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                float* outptr = top_blob.row(p);

                for (int j = 0; j < outw; j++)
                {
                    __m256 _sum = _mm256_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum = _mm256_loadu_ps(((const float*)bias_data) + p * 8);
                    }

                    const float* kptr = weight_data_packed.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const float* sptr = bottom_blob_bordered.row(q) + j * stride_w;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            __m256 _val = _mm256_set1_ps(sptr[0]);
                            __m256 _w = _mm256_loadu_ps(kptr);
                            _sum = _mm256_comp_fmadd_ps(_val, _w, _sum);

                            sptr += dilation_w;
                            kptr += 8;
                        }
                    }

                    _sum = activation_avx(_sum, activation_type, activation_params);

                    _mm256_storeu_ps(outptr, _sum);
                    outptr += 8;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 8)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                float* outptr = top_blob.row(p);

                for (int j = 0; j < outw; j++)
                {
                    __m256 _sum = _mm256_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum = _mm256_loadu_ps((const float*)bias_data + p * 8);
                    }

                    const float* kptr = weight_data_packed.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const float* sptr = bottom_blob_bordered.row(q) + j * stride_w * 4;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            __m256 _val0 = _mm256_broadcast_ss(sptr);
                            __m256 _val1 = _mm256_broadcast_ss(sptr + 1);
                            __m256 _val2 = _mm256_broadcast_ss(sptr + 2);
                            __m256 _val3 = _mm256_broadcast_ss(sptr + 3);

                            __m256 _w0 = _mm256_loadu_ps(kptr);
                            _sum = _mm256_comp_fmadd_ps(_val0, _w0, _sum);
                            __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                            _sum = _mm256_comp_fmadd_ps(_val1, _w1, _sum);
                            __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                            _sum = _mm256_comp_fmadd_ps(_val2, _w2, _sum);
                            __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                            _sum = _mm256_comp_fmadd_ps(_val3, _w3, _sum);

                            sptr += dilation_w * 4;
                            kptr += 32;
                        }
                    }

                    _sum = activation_avx(_sum, activation_type, activation_params);

                    _mm256_storeu_ps(outptr, _sum);
                    outptr += 8;
                }
            }
        }
    }

    if (elempack == 8 && out_elempack == 1)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                float* outptr = top_blob.row(p);

                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    const float* kptr = weight_data_packed.channel(p);

                    __m256 _sum8 = _mm256_set1_ps(0);

                    for (int q = 0; q < h; q++)
                    {
                        const float* sptr = bottom_blob_bordered.row(q) + j * stride_w * 8;

                        for (int k = 0; k < kernel_w; k++) // 29.23
                        {
                            __m256 _val = _mm256_loadu_ps(sptr);
                            __m256 _w = _mm256_loadu_ps(kptr);
                            __m256 _s8 = _mm256_mul_ps(_val, _w);
                            _sum8 = _mm256_add_ps(_sum8, _s8);

                            sptr += dilation_w * 8;
                            kptr += 8;
                        }
                    }
                    sum += _mm256_reduce_add_ps(_sum8); // dot
                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[j] = sum;
                }
            }
        }
    }

    if (elempack == 8 && out_elempack == 4)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                float* outptr = top_blob.row(p);

                for (int j = 0; j < outw; j++)
                {
                    __m128 _sum = _mm_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum = _mm_loadu_ps((const float*)bias_data + p * 4);
                    }

                    const float* kptr = weight_data_packed.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const float* sptr = bottom_blob_bordered.row(q) + j * stride_w * 8;

                        for (int k = 0; k < kernel_w; k++)
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

                            sptr += dilation_w * 8;
                            kptr += 32;
                        }
                    }

                    _sum = activation_sse(_sum, activation_type, activation_params);

                    _mm_storeu_ps(outptr, _sum);
                    outptr += 4;
                }
            }
        }
    }
#endif

    if (elempack == 4 && out_elempack == 4)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                float* outptr = top_blob.row(p);

                for (int j = 0; j < outw; j++)
                {
                    __m128 _sum = _mm_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum = _mm_loadu_ps((const float*)bias_data + p * 4);
                    }

                    const float* kptr = weight_data_packed.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const float* sptr = bottom_blob_bordered.row(q) + j * stride_w * 4;

                        for (int k = 0; k < kernel_w; k++)
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

                            sptr += dilation_w * 4;
                            kptr += 16;
                        }
                    }

                    _sum = activation_sse(_sum, activation_type, activation_params);

                    _mm_storeu_ps(outptr, _sum);
                    outptr += 4;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                float* outptr = top_blob.row(p);

                for (int j = 0; j < outw; j++)
                {
                    __m128 _sum = _mm_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum = _mm_loadu_ps((const float*)bias_data + p * 4);
                    }

                    const float* kptr = weight_data_packed.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const float* sptr = bottom_blob_bordered.row(q) + j * stride_w;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            __m128 _val = _mm_set1_ps(sptr[0]);
                            __m128 _w = _mm_loadu_ps(kptr);
                            _sum = _mm_add_ps(_mm_mul_ps(_val, _w), _sum);

                            sptr += dilation_w;
                            kptr += 4;
                        }
                    }

                    _sum = activation_sse(_sum, activation_type, activation_params);

                    _mm_storeu_ps(outptr, _sum);
                    outptr += 4;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                float* outptr = top_blob.row(p);

                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    const float* kptr = weight_data_packed.channel(p);

                    for (int q = 0; q < h; q++)
                    {
                        const float* sptr = bottom_blob_bordered.row(q) + j * stride_w * 4;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            __m128 _val = _mm_loadu_ps(sptr);
                            __m128 _w = _mm_loadu_ps(kptr);
                            __m128 _s4 = _mm_mul_ps(_val, _w);
                            sum += _mm_reduce_add_ps(_s4); // dot

                            sptr += dilation_w * 4;
                            kptr += 4;
                        }
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[j] = sum;
                }
            }
        }
    }
#endif // __SSE2__

    if (elempack == 1 && out_elempack == 1)
    {
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < outh; p++)
            {
                float* outptr = top_blob.row(p);

                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    const float* kptr = (const float*)weight_data + kernel_w * h * p;

                    for (int q = 0; q < h; q++)
                    {
                        const float* sptr = bottom_blob_bordered.row(q) + j * stride_w;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            float val = sptr[0];
                            float wt = kptr[0];
                            sum += val * wt;

                            sptr += dilation_w;
                            kptr += 1;
                        }
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[j] = sum;
                }
            }
        }
    }

    return 0;
}

int Convolution1D_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _kernel_w = _weight_data.w;
    const int _num_output = _weight_data.c * _weight_data.elempack;

    Mat weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

    // weight_data_flattened as pack1
    weight_data_flattened.w *= weight_data_flattened.elempack;
    weight_data_flattened.elemsize /= weight_data_flattened.elempack;
    weight_data_flattened.elempack = 1;

    Mat bias_data_flattened;
    if (bias_term)
    {
        const Mat& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;

        // bias_data_flattened as pack1
        bias_data_flattened.w *= bias_data_flattened.elempack;
        bias_data_flattened.elemsize /= bias_data_flattened.elempack;
        bias_data_flattened.elempack = 1;
    }

    ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Convolution1D);

    ncnn::ParamDict pd;
    pd.set(0, _num_output);
    pd.set(1, _kernel_w);
    pd.set(2, dilation_w);
    pd.set(3, stride_w);
    pd.set(4, pad_left);
    pd.set(15, pad_right);
    pd.set(18, pad_value);
    pd.set(5, bias_term);
    pd.set(6, weight_data_flattened.w);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    op->load_param(pd);

    ncnn::Mat weights[2];
    weights[0] = weight_data_flattened;
    weights[1] = bias_data_flattened;

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    op->forward(bottom_blob, top_blob, opt);

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

} // namespace ncnn
