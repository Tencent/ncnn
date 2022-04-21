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

#include "innerproduct_mips.h"

#include "layer_type.h"

#if __mips_msa
#include <msa.h>
#include "msa_mathfun.h"
#endif // __mips_msa

#include "mips_activation.h"

namespace ncnn {

InnerProduct_mips::InnerProduct_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa

    flatten = 0;
    activation = 0;
}

int InnerProduct_mips::create_pipeline(const Option& opt)
{
    {
        flatten = ncnn::create_layer(ncnn::LayerType::Flatten);

        ncnn::ParamDict pd;

        flatten->load_param(pd);

        flatten->create_pipeline(opt);
    }

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return create_pipeline_int8_mips(opt);
    }
#endif

    return 0;
}

int InnerProduct_mips::destroy_pipeline(const Option& opt)
{
    if (flatten)
    {
        flatten->destroy_pipeline(opt);
        delete flatten;
        flatten = 0;
    }

    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    return 0;
}

int InnerProduct_mips::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8_mips(bottom_blob, top_blob, opt);
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

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
#if __mips_msa
            if (elempack == 4)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const float* kptr = (const float*)weight_data + num_input * p;
                    const float* m = bottom_blob.row(j);

                    v4f32 _sum = (v4f32)__msa_fill_w(0);

                    if (bias_term)
                    {
                        _sum = __msa_fill_w_f32(bias_data[p]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 4);
                        v4f32 _val = (v4f32)__msa_ld_w(m, 0);
                        v4f32 _k = __msa_fill_w_f32(kptr[0]);
                        _sum = __msa_fmadd_w(_sum, _val, _k);

                        m += 4;
                        kptr += 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    __msa_st_w((v4i32)_sum, outptr, 0);
                    outptr += 4;
                }
            }
#endif // __mips_msa

            if (elempack == 1)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const float* kptr = (const float*)weight_data + num_input * p;
                    const float* m = bottom_blob.row(j);

                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        sum += m[i] * kptr[i];
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[0] = sum;
                    outptr += 1;
                }
            }
        }

        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

#if __mips_msa
    if (elempack == 4)
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
#endif

    top_blob.create(num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* weight_data_ptr = weight_data;

    int nn_num_output = num_output / 4;
    int remain_num_output_start = nn_num_output * 4;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_num_output; pp++)
    {
        int p = pp * 4;

        float sum0 = 0.f;
        float sum1 = 0.f;
        float sum2 = 0.f;
        float sum3 = 0.f;

        if (bias_term)
        {
            sum0 = bias_data[p];
            sum1 = bias_data[p + 1];
            sum2 = bias_data[p + 2];
            sum3 = bias_data[p + 3];
        }

        const float* w0 = weight_data_ptr + size * channels * p;
        const float* w1 = weight_data_ptr + size * channels * (p + 1);
        const float* w2 = weight_data_ptr + size * channels * (p + 2);
        const float* w3 = weight_data_ptr + size * channels * (p + 3);

#if __mips_msa
        v4f32 _sum0 = (v4f32)__msa_fill_w(0);
        v4f32 _sum1 = (v4f32)__msa_fill_w(0);
        v4f32 _sum2 = (v4f32)__msa_fill_w(0);
        v4f32 _sum3 = (v4f32)__msa_fill_w(0);
#endif // __mips_msa

        // channels
        for (int q = 0; q < channels; q++)
        {
            const float* m = bottom_blob.channel(q);

            int i = 0;
#if __mips_msa
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(m + 16);
                __builtin_prefetch(w0 + 16);
                __builtin_prefetch(w1 + 16);
                __builtin_prefetch(w2 + 16);
                __builtin_prefetch(w3 + 16);
                v4f32 _m = (v4f32)__msa_ld_w(m, 0);

                v4f32 _w0 = (v4f32)__msa_ld_w(w0, 0);
                _sum0 = __msa_fmadd_w(_sum0, _m, _w0);

                v4f32 _w1 = (v4f32)__msa_ld_w(w1, 0);
                _sum1 = __msa_fmadd_w(_sum1, _m, _w1);

                v4f32 _w2 = (v4f32)__msa_ld_w(w2, 0);
                _sum2 = __msa_fmadd_w(_sum2, _m, _w2);

                v4f32 _w3 = (v4f32)__msa_ld_w(w3, 0);
                _sum3 = __msa_fmadd_w(_sum3, _m, _w3);

                m += 4;
                w0 += 4;
                w1 += 4;
                w2 += 4;
                w3 += 4;
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                sum0 += *m * *w0;
                sum1 += *m * *w1;
                sum2 += *m * *w2;
                sum3 += *m * *w3;

                m++;
                w0++;
                w1++;
                w2++;
                w3++;
            }
        }

#if __mips_msa
        sum0 += __msa_reduce_fadd_w(_sum0);
        sum1 += __msa_reduce_fadd_w(_sum1);
        sum2 += __msa_reduce_fadd_w(_sum2);
        sum3 += __msa_reduce_fadd_w(_sum3);
#endif // __mips_msa

        sum0 = activation_ss(sum0, activation_type, activation_params);
        sum1 = activation_ss(sum1, activation_type, activation_params);
        sum2 = activation_ss(sum2, activation_type, activation_params);
        sum3 = activation_ss(sum3, activation_type, activation_params);

        top_blob[p] = sum0;
        top_blob[p + 1] = sum1;
        top_blob[p + 2] = sum2;
        top_blob[p + 3] = sum3;
    }

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_num_output_start; p < num_output; p++)
    {
        float sum = 0.f;

        if (bias_term)
            sum = bias_data[p];

        const float* w = weight_data_ptr + num_input * p;

        // channels
        for (int q = 0; q < channels; q++)
        {
            const float* m = bottom_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                sum += *m * *w;

                m++;
                w++;
            }
        }

        sum = activation_ss(sum, activation_type, activation_params);

        top_blob[p] = sum;
    }

    return 0;
}

#if NCNN_INT8
int InnerProduct_mips::create_pipeline_int8_mips(const Option& opt)
{
    activation = create_activation_layer(activation_type, activation_params, opt);

    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;
#if __mips_msa
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 8 == 0 ? 8 : 1;
    }
#endif // __mips_msa

    // src = inch-outch
    // dst = pb-inch-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_int8.create(num_input, num_output / out_elempack, (size_t)out_elempack, out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            signed char* g0 = weight_data_int8.row<signed char>(q / out_elempack);

            for (int p = 0; p < num_input; p++)
            {
                for (int j = 0; j < out_elempack; j++)
                {
                    *g0++ = weight_data_r2.row<signed char>(q + j)[p];
                }
            }
        }
    }

    return 0;
}

int InnerProduct_mips::forward_int8_mips(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;

    if (bottom_blob.dims == 2 && bottom_blob.w == num_input && bottom_blob.h * bottom_blob.elempack > 1)
    {
        // gemm
        Mat bottom_blob_unpacked;
        Option opt_unpack = opt;
        opt_unpack.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_unpack);

        return forward_int8(bottom_blob_unpacked, top_blob, opt);
    }

    int elembits = bottom_blob.elembits();

    Mat bottom_blob_int8 = bottom_blob;
    if (elembits != 8)
    {
        Option opt_q = opt;
        opt_q.blob_allocator = opt.workspace_allocator;
        quantize_to_int8(bottom_blob, bottom_blob_int8, bottom_blob_int8_scales, opt_q);
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
#if __mips_msa
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 8 == 0 ? 8 : 1;
    }
#endif // __mips_msa
    //     size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, (size_t)(4u * out_elempack), out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    Mat top_blob_int32;
    top_blob_int32.create(num_output / out_elempack, (size_t)(4u * out_elempack), out_elempack, opt.workspace_allocator);
    if (top_blob_int32.empty())
        return -100;

#if __mips_msa
    if (out_elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            v4i32 _sum0 = __msa_fill_w(0);
            v4i32 _sum1 = __msa_fill_w(0);

            const signed char* kptr = weight_data_int8.row<const signed char>(p);
            const signed char* sptr = bottom_blob_int8_flattened;

            int i = 0;
            for (; i < num_input; i++)
            {
                __builtin_prefetch(sptr + 4);
                __builtin_prefetch(kptr + 32);
                v8i16 _val = __msa_fill_h((short)sptr[0]);

                v16i8 _w = __msa_ld_b(kptr, 0);
                v8i16 _w16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_w, 0), _w);

                v8i16 _s0 = __msa_mulv_h(_val, _w16);
                v8i16 _exts0 = __msa_clti_s_h(_s0, 0);
                v4i32 _s0l = (v4i32)__msa_ilvr_h(_exts0, _s0);
                v4i32 _s0h = (v4i32)__msa_ilvl_h(_exts0, _s0);

                _sum0 = __msa_addv_w(_sum0, _s0l);
                _sum1 = __msa_addv_w(_sum1, _s0h);

                sptr += 1;
                kptr += 8;
            }

            int* outptr = (int*)top_blob_int32;
            __msa_st_w((v4i32)_sum0, outptr + p * 8, 0);
            __msa_st_w((v4i32)_sum1, outptr + p * 8 + 4, 0);
        }
    }
#endif // __mips_msa

    if (out_elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            int sum = 0;

            const signed char* kptr = weight_data_int8.row<const signed char>(p);
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

            int* outptr = (int*)top_blob_int32;
            outptr[p] = sum;
        }
    }

    Mat scale_data(num_output);
    for (int p = 0; p < num_output; p++)
    {
        // dequantize
        float scale_in;
        if (weight_data_int8_scales[p] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);

        scale_data[p] = scale_in;
    }

    dequantize_from_int32(top_blob_int32, top_blob, scale_data, bias_data, opt);

    if (activation)
    {
        activation->forward_inplace(top_blob, opt);
    }

    return 0;
}
#endif // NCNN_INT8

} // namespace ncnn
