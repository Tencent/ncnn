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
}

int InnerProduct_mips::create_pipeline(const Option& opt)
{
    {
        flatten = ncnn::create_layer_cpu(ncnn::LayerType::Flatten);

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

#if __mips_msa
    if (opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;

#if __mips_msa
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __mips_msa

    if (out_elempack == 4)
    {
        // src = inch-outch
        // dst = 4-inch-outch/4
        {
            Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

            weight_data_tm.create(num_input, num_output / 4, (size_t)4u * 4, 4);

            for (int q = 0; q + 3 < num_output; q += 4)
            {
                float* g0 = weight_data_tm.row(q / 4);

                for (int p = 0; p < num_input; p++)
                {
                    for (int j = 0; j < 4; j++)
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

    weight_data.release();

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

    return 0;
}

int InnerProduct_mips::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && int8_scale_term)
    {
        return forward_int8_mips(bottom_blob, top_blob, opt);
    }
#endif

#if __mips_msa
    if (opt.use_fp16_storage)
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

        int num_output_elempack = 1;
#if __mips_msa
        if (opt.use_packing_layout)
        {
            num_output_elempack = num_output % 4 == 0 ? 4 : 1;
        }
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
#if __mips_msa
            if (elempack == 4 && num_output_elempack == 4)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                    v4f32 _sum1 = (v4f32)__msa_fill_w(0);
                    v4f32 _sum2 = (v4f32)__msa_fill_w(0);
                    v4f32 _sum3 = (v4f32)__msa_fill_w(0);

                    if (bias_term)
                    {
                        _sum0 = __msa_fill_w_f32(bias_data[p * 4 + 0]);
                        _sum1 = __msa_fill_w_f32(bias_data[p * 4 + 1]);
                        _sum2 = __msa_fill_w_f32(bias_data[p * 4 + 2]);
                        _sum3 = __msa_fill_w_f32(bias_data[p * 4 + 3]);
                    }

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 16);
                        v4f32 _val = (v4f32)__msa_ld_w(m, 0);
                        v4i32 _w = __msa_ld_w(kptr, 0);
                        _sum0 = __msa_fmadd_w(_sum0, _val, (v4f32)__msa_splati_w(_w, 0));
                        _sum1 = __msa_fmadd_w(_sum1, _val, (v4f32)__msa_splati_w(_w, 1));
                        _sum2 = __msa_fmadd_w(_sum2, _val, (v4f32)__msa_splati_w(_w, 2));
                        _sum3 = __msa_fmadd_w(_sum3, _val, (v4f32)__msa_splati_w(_w, 3));

                        m += 4;
                        kptr += 4;
                    }

                    _sum0 = activation_ps(_sum0, activation_type, activation_params);
                    _sum1 = activation_ps(_sum1, activation_type, activation_params);
                    _sum2 = activation_ps(_sum2, activation_type, activation_params);
                    _sum3 = activation_ps(_sum3, activation_type, activation_params);

                    __msa_st_w((v4i32)_sum0, outptr, 0);
                    __msa_st_w((v4i32)_sum1, outptr + 4, 0);
                    __msa_st_w((v4i32)_sum2, outptr + 8, 0);
                    __msa_st_w((v4i32)_sum3, outptr + 12, 0);
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

                    v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                    v4f32 _sum1 = (v4f32)__msa_fill_w(0);
                    v4f32 _sum2 = (v4f32)__msa_fill_w(0);
                    v4f32 _sum3 = (v4f32)__msa_fill_w(0);

                    if (bias_term)
                    {
                        _sum0 = (v4f32)__msa_ld_w((const float*)bias_data + p * 4, 0);
                    }

                    int i = 0;
                    for (; i + 3 < num_input; i += 4)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 64);
                        v4i32 _val = __msa_ld_w(m, 0);
                        v4f32 _w0 = (v4f32)__msa_ld_w(kptr, 0);
                        v4f32 _w1 = (v4f32)__msa_ld_w(kptr + 4, 0);
                        v4f32 _w2 = (v4f32)__msa_ld_w(kptr + 8, 0);
                        v4f32 _w3 = (v4f32)__msa_ld_w(kptr + 12, 0);
                        _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_val, 0), _w0);
                        _sum1 = __msa_fmadd_w(_sum1, (v4f32)__msa_splati_w(_val, 1), _w1);
                        _sum2 = __msa_fmadd_w(_sum2, (v4f32)__msa_splati_w(_val, 2), _w2);
                        _sum3 = __msa_fmadd_w(_sum3, (v4f32)__msa_splati_w(_val, 3), _w3);

                        m += 4;
                        kptr += 16;
                    }
                    for (; i < num_input; i++)
                    {
                        v4f32 _val = __msa_fill_w_f32(m[0]);
                        v4f32 _w = (v4f32)__msa_ld_w(kptr, 0);
                        _sum0 = __msa_fmadd_w(_sum0, _val, _w);

                        m += 1;
                        kptr += 4;
                    }

                    _sum0 = __msa_fadd_w(_sum0, _sum1);
                    _sum2 = __msa_fadd_w(_sum2, _sum3);
                    _sum0 = __msa_fadd_w(_sum0, _sum2);

                    _sum0 = activation_ps(_sum0, activation_type, activation_params);

                    __msa_st_w((v4i32)_sum0, outptr, 0);
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
#if __mips_msa
                    v4f32 _sum = (v4f32)__msa_fill_w(0);
                    for (; i + 3 < num_input; i += 4)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 16);
                        v4f32 _m = (v4f32)__msa_ld_w(m, 0);
                        v4f32 _w = (v4f32)__msa_ld_w(kptr, 0);
                        _sum = __msa_fmadd_w(_sum, _m, _w);

                        m += 4;
                        kptr += 4;
                    }
                    sum += __msa_reduce_fadd_w(_sum);
#endif // __mips_msa
                    for (; i < num_input; i++)
                    {
                        sum += *m * *kptr;

                        m += 1;
                        kptr += 1;
                    }

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
#if __mips_msa
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __mips_msa
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __mips_msa
    if (out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            v4f32 _sum0 = (v4f32)__msa_fill_w(0);
            v4f32 _sum1 = (v4f32)__msa_fill_w(0);
            v4f32 _sum2 = (v4f32)__msa_fill_w(0);
            v4f32 _sum3 = (v4f32)__msa_fill_w(0);

            if (bias_term)
            {
                _sum0 = (v4f32)__msa_ld_w((const float*)bias_data + p * 4, 0);
            }

            const float* kptr = weight_data_tm.row(p);

            const float* sptr = bottom_blob_flattened;

            int i = 0;
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(sptr + 16);
                __builtin_prefetch(kptr + 64);
                v4i32 _val = __msa_ld_w(sptr, 0);
                v4f32 _w0 = (v4f32)__msa_ld_w(kptr, 0);
                v4f32 _w1 = (v4f32)__msa_ld_w(kptr + 4, 0);
                v4f32 _w2 = (v4f32)__msa_ld_w(kptr + 8, 0);
                v4f32 _w3 = (v4f32)__msa_ld_w(kptr + 12, 0);
                _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_val, 0), _w0);
                _sum1 = __msa_fmadd_w(_sum1, (v4f32)__msa_splati_w(_val, 1), _w1);
                _sum2 = __msa_fmadd_w(_sum2, (v4f32)__msa_splati_w(_val, 2), _w2);
                _sum3 = __msa_fmadd_w(_sum3, (v4f32)__msa_splati_w(_val, 3), _w3);

                sptr += 4;
                kptr += 16;
            }
            for (; i < num_input; i++)
            {
                v4f32 _val = __msa_fill_w_f32(sptr[0]);
                v4f32 _w = (v4f32)__msa_ld_w(kptr, 0);
                _sum0 = __msa_fmadd_w(_sum0, _val, _w);

                sptr += 1;
                kptr += 4;
            }

            _sum0 = __msa_fadd_w(_sum0, _sum1);
            _sum2 = __msa_fadd_w(_sum2, _sum3);
            _sum0 = __msa_fadd_w(_sum0, _sum2);

            _sum0 = activation_ps(_sum0, activation_type, activation_params);

            float* outptr = top_blob;
            __msa_st_w((v4i32)_sum0, outptr + p * 4, 0);
        }
    }
#endif // __mips_msa

    if (out_elempack == 1)
    {
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

            const float* w0 = (const float*)weight_data_tm + num_input * p;
            const float* w1 = (const float*)weight_data_tm + num_input * (p + 1);
            const float* w2 = (const float*)weight_data_tm + num_input * (p + 2);
            const float* w3 = (const float*)weight_data_tm + num_input * (p + 3);

            const float* m = bottom_blob_flattened;

            int i = 0;
#if __mips_msa
            v4f32 _sum0 = (v4f32)__msa_fill_w(0);
            v4f32 _sum1 = (v4f32)__msa_fill_w(0);
            v4f32 _sum2 = (v4f32)__msa_fill_w(0);
            v4f32 _sum3 = (v4f32)__msa_fill_w(0);
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(m + 16);
                __builtin_prefetch(w0 + 16);
                __builtin_prefetch(w1 + 16);
                __builtin_prefetch(w2 + 16);
                __builtin_prefetch(w3 + 16);
                v4f32 _m = (v4f32)__msa_ld_w(m, 0);
                v4f32 _w0 = (v4f32)__msa_ld_w(w0, 0);
                v4f32 _w1 = (v4f32)__msa_ld_w(w1, 0);
                v4f32 _w2 = (v4f32)__msa_ld_w(w2, 0);
                v4f32 _w3 = (v4f32)__msa_ld_w(w3, 0);
                _sum0 = __msa_fmadd_w(_sum0, _m, _w0);
                _sum1 = __msa_fmadd_w(_sum1, _m, _w1);
                _sum2 = __msa_fmadd_w(_sum2, _m, _w2);
                _sum3 = __msa_fmadd_w(_sum3, _m, _w3);

                m += 4;
                w0 += 4;
                w1 += 4;
                w2 += 4;
                w3 += 4;
            }
#endif // __mips_msa
            for (; i < num_input; i++)
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

            const float* w = (const float*)weight_data_tm + num_input * p;

            const float* m = bottom_blob_flattened;

            int i = 0;
#if __mips_msa
            v4f32 _sum0 = (v4f32)__msa_fill_w(0);
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(m + 16);
                __builtin_prefetch(w + 16);
                v4f32 _m = (v4f32)__msa_ld_w(m, 0);
                v4f32 _w = (v4f32)__msa_ld_w(w, 0);
                _sum0 = __msa_fmadd_w(_sum0, _m, _w);

                m += 4;
                w += 4;
            }
            sum += __msa_reduce_fadd_w(_sum0);
#endif // __mips_msa
            for (; i < num_input; i++)
            {
                sum += *m * *w;

                m++;
                w++;
            }

            sum = activation_ss(sum, activation_type, activation_params);

            top_blob[p] = sum;
        }
    }

    return 0;
}

#if __mips_msa
int InnerProduct_mips::create_pipeline_fp16s(const Option& opt)
{
    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }

    // src = inch-outch
    // dst = pb-inch-outch/pb
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
                v4f32 _r0 = (v4f32)__msa_ld_w(k0, 0);
                v4f32 _r1 = (v4f32)__msa_ld_w(k1, 0);
                v4f32 _r2 = (v4f32)__msa_ld_w(k2, 0);
                v4f32 _r3 = (v4f32)__msa_ld_w(k3, 0);

                v4i32 _r01r = __msa_ilvr_w((v4i32)_r1, (v4i32)_r0);
                v4i32 _r01l = __msa_ilvl_w((v4i32)_r1, (v4i32)_r0);
                v4i32 _r23r = __msa_ilvr_w((v4i32)_r3, (v4i32)_r2);
                v4i32 _r23l = __msa_ilvl_w((v4i32)_r3, (v4i32)_r2);
                v2i64 _r0123_0 = __msa_ilvr_d((v2i64)_r23r, (v2i64)_r01r);
                v2i64 _r0123_1 = __msa_ilvl_d((v2i64)_r23r, (v2i64)_r01r);
                v2i64 _r0123_2 = __msa_ilvr_d((v2i64)_r23l, (v2i64)_r01l);
                v2i64 _r0123_3 = __msa_ilvl_d((v2i64)_r23l, (v2i64)_r01l);

                v8i16 _p0 = __msa_fexdo_h((v4f32)_r0123_1, (v4f32)_r0123_0);
                v8i16 _p1 = __msa_fexdo_h((v4f32)_r0123_3, (v4f32)_r0123_2);

                __msa_st_h(_p0, g0, 0);
                __msa_st_h(_p1, g0 + 8, 0);

                k0 += 4;
                k1 += 4;
                k2 += 4;
                k3 += 4;
                g0 += 16;
            }
            for (; p < num_input; p++)
            {
                g0[0] = float32_to_float16(*k0++);
                g0[1] = float32_to_float16(*k1++);
                g0[2] = float32_to_float16(*k2++);
                g0[3] = float32_to_float16(*k3++);
                g0 += 4;
            }
        }
    }

    if (out_elempack == 1)
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);
        ncnn::cast_float32_to_float16(weight_data_r2, weight_data_tm, opt);
    }

    weight_data.release();

    return 0;
}

int InnerProduct_mips::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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

        int num_output_elempack = 1;
        if (opt.use_packing_layout)
        {
            num_output_elempack = num_output % 4 == 0 ? 4 : 1;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
            if (elempack == 4 && num_output_elempack == 4)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                    const float* m = bottom_blob.row(j);

                    v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                    v4f32 _sum1 = (v4f32)__msa_fill_w(0);
                    v4f32 _sum2 = (v4f32)__msa_fill_w(0);
                    v4f32 _sum3 = (v4f32)__msa_fill_w(0);

                    if (bias_term)
                    {
                        _sum0 = __msa_fill_w_f32(bias_data[p * 4 + 0]);
                        _sum1 = __msa_fill_w_f32(bias_data[p * 4 + 1]);
                        _sum2 = __msa_fill_w_f32(bias_data[p * 4 + 2]);
                        _sum3 = __msa_fill_w_f32(bias_data[p * 4 + 3]);
                    }

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 16);
                        v4f32 _val = (v4f32)__msa_ld_w(m, 0);
                        v4i32 _w = (v4i32)__msa_fexupr_w(__msa_ld_h(kptr, 0));
                        _sum0 = __msa_fmadd_w(_sum0, _val, (v4f32)__msa_splati_w(_w, 0));
                        _sum1 = __msa_fmadd_w(_sum1, _val, (v4f32)__msa_splati_w(_w, 1));
                        _sum2 = __msa_fmadd_w(_sum2, _val, (v4f32)__msa_splati_w(_w, 2));
                        _sum3 = __msa_fmadd_w(_sum3, _val, (v4f32)__msa_splati_w(_w, 3));

                        m += 4;
                        kptr += 4;
                    }

                    _sum0 = activation_ps(_sum0, activation_type, activation_params);
                    _sum1 = activation_ps(_sum1, activation_type, activation_params);
                    _sum2 = activation_ps(_sum2, activation_type, activation_params);
                    _sum3 = activation_ps(_sum3, activation_type, activation_params);

                    __msa_st_w((v4i32)_sum0, outptr, 0);
                    __msa_st_w((v4i32)_sum1, outptr + 4, 0);
                    __msa_st_w((v4i32)_sum2, outptr + 8, 0);
                    __msa_st_w((v4i32)_sum3, outptr + 12, 0);
                    outptr += 16;
                }
            }

            if (elempack == 1 && num_output_elempack == 4)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                    const float* m = bottom_blob.row(j);

                    v4f32 _sum0 = (v4f32)__msa_fill_w(0);
                    v4f32 _sum1 = (v4f32)__msa_fill_w(0);
                    v4f32 _sum2 = (v4f32)__msa_fill_w(0);
                    v4f32 _sum3 = (v4f32)__msa_fill_w(0);

                    if (bias_term)
                    {
                        _sum0 = (v4f32)__msa_ld_w((const float*)bias_data + p * 4, 0);
                    }

                    int i = 0;
                    for (; i + 3 < num_input; i += 4)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 64);
                        v4i32 _val = __msa_ld_w(m, 0);
                        v8i16 _w01 = __msa_ld_h(kptr, 0);
                        v8i16 _w23 = __msa_ld_h(kptr + 8, 0);
                        v4f32 _w0 = __msa_fexupr_w(_w01);
                        v4f32 _w1 = __msa_fexupl_w(_w01);
                        v4f32 _w2 = __msa_fexupr_w(_w23);
                        v4f32 _w3 = __msa_fexupl_w(_w23);
                        _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_val, 0), _w0);
                        _sum1 = __msa_fmadd_w(_sum1, (v4f32)__msa_splati_w(_val, 1), _w1);
                        _sum2 = __msa_fmadd_w(_sum2, (v4f32)__msa_splati_w(_val, 2), _w2);
                        _sum3 = __msa_fmadd_w(_sum3, (v4f32)__msa_splati_w(_val, 3), _w3);

                        m += 4;
                        kptr += 16;
                    }
                    for (; i < num_input; i++)
                    {
                        v4f32 _val = __msa_fill_w_f32(m[0]);
                        v4f32 _w = __msa_fexupr_w(__msa_ld_h(kptr, 0));
                        _sum0 = __msa_fmadd_w(_sum0, _val, _w);

                        m += 1;
                        kptr += 4;
                    }

                    _sum0 = __msa_fadd_w(_sum0, _sum1);
                    _sum2 = __msa_fadd_w(_sum2, _sum3);
                    _sum0 = __msa_fadd_w(_sum0, _sum2);

                    _sum0 = activation_ps(_sum0, activation_type, activation_params);

                    __msa_st_w((v4i32)_sum0, outptr, 0);
                    outptr += 4;
                }
            }

            if (elempack == 4 && num_output_elempack == 1)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
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
                        v4f32 _k = __msa_fill_w_f32(float16_to_float32(kptr[0]));
                        _sum = __msa_fmadd_w(_sum, _val, _k);

                        m += 4;
                        kptr += 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    __msa_st_w((v4i32)_sum, outptr, 0);
                    outptr += 4;
                }
            }

            if (elempack == 1 && num_output_elempack == 1)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);
                    const float* m = bottom_blob.row(j);

                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    int i = 0;
                    v4f32 _sum = (v4f32)__msa_fill_w(0);
                    for (; i + 3 < num_input; i += 4)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 16);
                        v4f32 _m = (v4f32)__msa_ld_w(m, 0);
                        v4f32 _w = __msa_fexupr_w(__msa_ld_h(kptr, 0));
                        _sum = __msa_fmadd_w(_sum, _m, _w);

                        m += 4;
                        kptr += 4;
                    }
                    sum += __msa_reduce_fadd_w(_sum);
                    for (; i < num_input; i++)
                    {
                        sum += *m * float16_to_float32(*kptr);

                        m += 1;
                        kptr += 1;
                    }

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
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            v4f32 _sum0 = (v4f32)__msa_fill_w(0);
            v4f32 _sum1 = (v4f32)__msa_fill_w(0);
            v4f32 _sum2 = (v4f32)__msa_fill_w(0);
            v4f32 _sum3 = (v4f32)__msa_fill_w(0);

            if (bias_term)
            {
                _sum0 = (v4f32)__msa_ld_w((const float*)bias_data + p * 4, 0);
            }

            const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);

            const float* sptr = bottom_blob_flattened;

            int i = 0;
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(sptr + 16);
                __builtin_prefetch(kptr + 64);
                v4i32 _val = __msa_ld_w(sptr, 0);
                v8i16 _w01 = __msa_ld_h(kptr, 0);
                v8i16 _w23 = __msa_ld_h(kptr + 8, 0);
                v4f32 _w0 = __msa_fexupr_w(_w01);
                v4f32 _w1 = __msa_fexupl_w(_w01);
                v4f32 _w2 = __msa_fexupr_w(_w23);
                v4f32 _w3 = __msa_fexupl_w(_w23);
                _sum0 = __msa_fmadd_w(_sum0, (v4f32)__msa_splati_w(_val, 0), _w0);
                _sum1 = __msa_fmadd_w(_sum1, (v4f32)__msa_splati_w(_val, 1), _w1);
                _sum2 = __msa_fmadd_w(_sum2, (v4f32)__msa_splati_w(_val, 2), _w2);
                _sum3 = __msa_fmadd_w(_sum3, (v4f32)__msa_splati_w(_val, 3), _w3);

                sptr += 4;
                kptr += 16;
            }
            for (; i < num_input; i++)
            {
                v4f32 _val = __msa_fill_w_f32(sptr[0]);
                v4f32 _w = __msa_fexupr_w(__msa_ld_h(kptr, 0));
                _sum0 = __msa_fmadd_w(_sum0, _val, _w);

                sptr += 1;
                kptr += 4;
            }

            _sum0 = __msa_fadd_w(_sum0, _sum1);
            _sum2 = __msa_fadd_w(_sum2, _sum3);
            _sum0 = __msa_fadd_w(_sum0, _sum2);

            _sum0 = activation_ps(_sum0, activation_type, activation_params);

            float* outptr = top_blob;
            __msa_st_w((v4i32)_sum0, outptr + p * 4, 0);
        }
    }

    if (out_elempack == 1)
    {
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

            const unsigned short* w0 = weight_data_tm.row<const unsigned short>(p);
            const unsigned short* w1 = weight_data_tm.row<const unsigned short>(p + 1);
            const unsigned short* w2 = weight_data_tm.row<const unsigned short>(p + 2);
            const unsigned short* w3 = weight_data_tm.row<const unsigned short>(p + 3);

            const float* m = bottom_blob_flattened;

            int i = 0;
            v4f32 _sum0 = (v4f32)__msa_fill_w(0);
            v4f32 _sum1 = (v4f32)__msa_fill_w(0);
            v4f32 _sum2 = (v4f32)__msa_fill_w(0);
            v4f32 _sum3 = (v4f32)__msa_fill_w(0);
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(m + 16);
                __builtin_prefetch(w0 + 16);
                __builtin_prefetch(w1 + 16);
                __builtin_prefetch(w2 + 16);
                __builtin_prefetch(w3 + 16);
                v4f32 _m = (v4f32)__msa_ld_w(m, 0);
                v4f32 _w0 = __msa_fexupr_w(__msa_ld_h(w0, 0));
                v4f32 _w1 = __msa_fexupr_w(__msa_ld_h(w1, 0));
                v4f32 _w2 = __msa_fexupr_w(__msa_ld_h(w2, 0));
                v4f32 _w3 = __msa_fexupr_w(__msa_ld_h(w3, 0));
                _sum0 = __msa_fmadd_w(_sum0, _m, _w0);
                _sum1 = __msa_fmadd_w(_sum1, _m, _w1);
                _sum2 = __msa_fmadd_w(_sum2, _m, _w2);
                _sum3 = __msa_fmadd_w(_sum3, _m, _w3);

                m += 4;
                w0 += 4;
                w1 += 4;
                w2 += 4;
                w3 += 4;
            }
            for (; i < num_input; i++)
            {
                sum0 += *m * float16_to_float32(*w0);
                sum1 += *m * float16_to_float32(*w1);
                sum2 += *m * float16_to_float32(*w2);
                sum3 += *m * float16_to_float32(*w3);

                m++;
                w0++;
                w1++;
                w2++;
                w3++;
            }

            sum0 += __msa_reduce_fadd_w(_sum0);
            sum1 += __msa_reduce_fadd_w(_sum1);
            sum2 += __msa_reduce_fadd_w(_sum2);
            sum3 += __msa_reduce_fadd_w(_sum3);

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

            const unsigned short* w = weight_data_tm.row<const unsigned short>(p);

            const float* m = bottom_blob_flattened;

            int i = 0;
            v4f32 _sum0 = (v4f32)__msa_fill_w(0);
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(m + 16);
                __builtin_prefetch(w + 16);
                v4f32 _m = (v4f32)__msa_ld_w(m, 0);
                v4f32 _w = __msa_fexupr_w(__msa_ld_h(w, 0));
                _sum0 = __msa_fmadd_w(_sum0, _m, _w);

                m += 4;
                w += 4;
            }
            sum += __msa_reduce_fadd_w(_sum0);
            for (; i < num_input; i++)
            {
                sum += *m * float16_to_float32(*w);

                m++;
                w++;
            }

            sum = activation_ss(sum, activation_type, activation_params);

            top_blob[p] = sum;
        }
    }

    return 0;
}
#endif // __mips_msa

#if NCNN_INT8
int InnerProduct_mips::create_pipeline_int8_mips(const Option& opt)
{
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

int InnerProduct_mips::forward_int8_mips(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
#if __mips_msa
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
#if __mips_msa
        if (opt.use_packing_layout)
        {
            num_output_elempack = num_output % 8 == 0 ? 8 : 1;
        }
#endif

#if __mips_msa
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

                    v4i32 _sum00 = __msa_fill_w(0);
                    v4i32 _sum01 = __msa_fill_w(0);
                    v4i32 _sum10 = __msa_fill_w(0);
                    v4i32 _sum11 = __msa_fill_w(0);
                    v4i32 _sum20 = __msa_fill_w(0);
                    v4i32 _sum21 = __msa_fill_w(0);
                    v4i32 _sum30 = __msa_fill_w(0);
                    v4i32 _sum31 = __msa_fill_w(0);

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        __builtin_prefetch(m0 + 4);
                        __builtin_prefetch(m1 + 4);
                        __builtin_prefetch(m2 + 4);
                        __builtin_prefetch(m3 + 4);
                        __builtin_prefetch(kptr + 32);
                        v8i16 _val0 = __msa_fill_h((short)m0[0]);
                        v8i16 _val1 = __msa_fill_h((short)m1[0]);
                        v8i16 _val2 = __msa_fill_h((short)m2[0]);
                        v8i16 _val3 = __msa_fill_h((short)m3[0]);

                        v16i8 _w = __msa_ld_b(kptr, 0);
                        v8i16 _w16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_w, 0), _w);

                        v8i16 _s0 = __msa_mulv_h(_val0, _w16);
                        v8i16 _s1 = __msa_mulv_h(_val1, _w16);
                        v8i16 _s2 = __msa_mulv_h(_val2, _w16);
                        v8i16 _s3 = __msa_mulv_h(_val3, _w16);
                        v8i16 _exts0 = __msa_clti_s_h(_s0, 0);
                        v8i16 _exts1 = __msa_clti_s_h(_s1, 0);
                        v8i16 _exts2 = __msa_clti_s_h(_s2, 0);
                        v8i16 _exts3 = __msa_clti_s_h(_s3, 0);
                        v4i32 _s0l = (v4i32)__msa_ilvr_h(_exts0, _s0);
                        v4i32 _s0h = (v4i32)__msa_ilvl_h(_exts0, _s0);
                        v4i32 _s1l = (v4i32)__msa_ilvr_h(_exts1, _s1);
                        v4i32 _s1h = (v4i32)__msa_ilvl_h(_exts1, _s1);
                        v4i32 _s2l = (v4i32)__msa_ilvr_h(_exts2, _s2);
                        v4i32 _s2h = (v4i32)__msa_ilvl_h(_exts2, _s2);
                        v4i32 _s3l = (v4i32)__msa_ilvr_h(_exts3, _s3);
                        v4i32 _s3h = (v4i32)__msa_ilvl_h(_exts3, _s3);

                        _sum00 = __msa_addv_w(_sum00, _s0l);
                        _sum01 = __msa_addv_w(_sum01, _s0h);
                        _sum10 = __msa_addv_w(_sum10, _s1l);
                        _sum11 = __msa_addv_w(_sum11, _s1h);
                        _sum20 = __msa_addv_w(_sum20, _s2l);
                        _sum21 = __msa_addv_w(_sum21, _s2h);
                        _sum30 = __msa_addv_w(_sum30, _s3l);
                        _sum31 = __msa_addv_w(_sum31, _s3h);

                        m0++;
                        m1++;
                        m2++;
                        m3++;
                        kptr += 8;
                    }

                    // dequantize and relu
                    v4f32 _scale_in0 = (v4f32)__msa_ld_w((const float*)scale_in_data + p * 8, 0);
                    v4f32 _scale_in1 = (v4f32)__msa_ld_w((const float*)scale_in_data + p * 8 + 4, 0);

                    v4f32 _sumfp32_00 = (v4f32)__msa_ffint_s_w(_sum00);
                    v4f32 _sumfp32_01 = (v4f32)__msa_ffint_s_w(_sum01);
                    v4f32 _sumfp32_10 = (v4f32)__msa_ffint_s_w(_sum10);
                    v4f32 _sumfp32_11 = (v4f32)__msa_ffint_s_w(_sum11);
                    v4f32 _sumfp32_20 = (v4f32)__msa_ffint_s_w(_sum20);
                    v4f32 _sumfp32_21 = (v4f32)__msa_ffint_s_w(_sum21);
                    v4f32 _sumfp32_30 = (v4f32)__msa_ffint_s_w(_sum30);
                    v4f32 _sumfp32_31 = (v4f32)__msa_ffint_s_w(_sum31);
                    if (bias_term)
                    {
                        v4f32 _bias0 = (v4f32)__msa_ld_w((const float*)bias_data + p * 8, 0);
                        v4f32 _bias1 = (v4f32)__msa_ld_w((const float*)bias_data + p * 8 + 4, 0);
                        _sumfp32_00 = __msa_fmadd_w(_bias0, _sumfp32_00, _scale_in0);
                        _sumfp32_01 = __msa_fmadd_w(_bias1, _sumfp32_01, _scale_in1);
                        _sumfp32_10 = __msa_fmadd_w(_bias0, _sumfp32_10, _scale_in0);
                        _sumfp32_11 = __msa_fmadd_w(_bias1, _sumfp32_11, _scale_in1);
                        _sumfp32_20 = __msa_fmadd_w(_bias0, _sumfp32_20, _scale_in0);
                        _sumfp32_21 = __msa_fmadd_w(_bias1, _sumfp32_21, _scale_in1);
                        _sumfp32_30 = __msa_fmadd_w(_bias0, _sumfp32_30, _scale_in0);
                        _sumfp32_31 = __msa_fmadd_w(_bias1, _sumfp32_31, _scale_in1);
                    }
                    else
                    {
                        _sumfp32_00 = __msa_fmul_w(_sumfp32_00, _scale_in0);
                        _sumfp32_01 = __msa_fmul_w(_sumfp32_01, _scale_in1);
                        _sumfp32_10 = __msa_fmul_w(_sumfp32_10, _scale_in0);
                        _sumfp32_11 = __msa_fmul_w(_sumfp32_11, _scale_in1);
                        _sumfp32_20 = __msa_fmul_w(_sumfp32_20, _scale_in0);
                        _sumfp32_21 = __msa_fmul_w(_sumfp32_21, _scale_in1);
                        _sumfp32_30 = __msa_fmul_w(_sumfp32_30, _scale_in0);
                        _sumfp32_31 = __msa_fmul_w(_sumfp32_31, _scale_in1);
                    }

                    _sumfp32_00 = activation_ps(_sumfp32_00, activation_type, activation_params);
                    _sumfp32_01 = activation_ps(_sumfp32_01, activation_type, activation_params);
                    _sumfp32_10 = activation_ps(_sumfp32_10, activation_type, activation_params);
                    _sumfp32_11 = activation_ps(_sumfp32_11, activation_type, activation_params);
                    _sumfp32_20 = activation_ps(_sumfp32_20, activation_type, activation_params);
                    _sumfp32_21 = activation_ps(_sumfp32_21, activation_type, activation_params);
                    _sumfp32_30 = activation_ps(_sumfp32_30, activation_type, activation_params);
                    _sumfp32_31 = activation_ps(_sumfp32_31, activation_type, activation_params);

                    // transpose 4x8
                    v4i32 _r01r = __msa_ilvr_w((v4i32)_sumfp32_10, (v4i32)_sumfp32_00);
                    v4i32 _r01l = __msa_ilvl_w((v4i32)_sumfp32_10, (v4i32)_sumfp32_00);
                    v4i32 _r23r = __msa_ilvr_w((v4i32)_sumfp32_30, (v4i32)_sumfp32_20);
                    v4i32 _r23l = __msa_ilvl_w((v4i32)_sumfp32_30, (v4i32)_sumfp32_20);
                    v4i32 _r45r = __msa_ilvr_w((v4i32)_sumfp32_11, (v4i32)_sumfp32_01);
                    v4i32 _r45l = __msa_ilvl_w((v4i32)_sumfp32_11, (v4i32)_sumfp32_01);
                    v4i32 _r67r = __msa_ilvr_w((v4i32)_sumfp32_31, (v4i32)_sumfp32_21);
                    v4i32 _r67l = __msa_ilvl_w((v4i32)_sumfp32_31, (v4i32)_sumfp32_21);
                    _sumfp32_00 = (v4f32)__msa_ilvr_d((v2i64)_r23r, (v2i64)_r01r);
                    _sumfp32_10 = (v4f32)__msa_ilvl_d((v2i64)_r23r, (v2i64)_r01r);
                    _sumfp32_20 = (v4f32)__msa_ilvr_d((v2i64)_r23l, (v2i64)_r01l);
                    _sumfp32_30 = (v4f32)__msa_ilvl_d((v2i64)_r23l, (v2i64)_r01l);
                    _sumfp32_01 = (v4f32)__msa_ilvr_d((v2i64)_r67r, (v2i64)_r45r);
                    _sumfp32_11 = (v4f32)__msa_ilvl_d((v2i64)_r67r, (v2i64)_r45r);
                    _sumfp32_21 = (v4f32)__msa_ilvr_d((v2i64)_r67l, (v2i64)_r45l);
                    _sumfp32_31 = (v4f32)__msa_ilvl_d((v2i64)_r67l, (v2i64)_r45l);

                    __msa_st_w((v4i32)_sumfp32_00, outptr, 0);
                    __msa_st_w((v4i32)_sumfp32_10, outptr + 4, 0);
                    __msa_st_w((v4i32)_sumfp32_20, outptr + 8, 0);
                    __msa_st_w((v4i32)_sumfp32_30, outptr + 12, 0);
                    __msa_st_w((v4i32)_sumfp32_01, outptr + 16, 0);
                    __msa_st_w((v4i32)_sumfp32_11, outptr + 20, 0);
                    __msa_st_w((v4i32)_sumfp32_21, outptr + 24, 0);
                    __msa_st_w((v4i32)_sumfp32_31, outptr + 28, 0);

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

                    v4i32 _sum0 = __msa_fill_w(0);
                    v4i32 _sum1 = __msa_fill_w(0);

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        __builtin_prefetch(m + 4);
                        __builtin_prefetch(kptr + 32);
                        v8i16 _val = __msa_fill_h((short)m[0]);

                        v16i8 _w = __msa_ld_b(kptr, 0);
                        v8i16 _w16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_w, 0), _w);

                        v8i16 _s0 = __msa_mulv_h(_val, _w16);
                        v8i16 _exts0 = __msa_clti_s_h(_s0, 0);
                        v4i32 _s0l = (v4i32)__msa_ilvr_h(_exts0, _s0);
                        v4i32 _s0h = (v4i32)__msa_ilvl_h(_exts0, _s0);

                        _sum0 = __msa_addv_w(_sum0, _s0l);
                        _sum1 = __msa_addv_w(_sum1, _s0h);

                        m++;
                        kptr += 8;
                    }

                    // dequantize and relu
                    v4f32 _scale_in0 = (v4f32)__msa_ld_w((const float*)scale_in_data + p * 8, 0);
                    v4f32 _scale_in1 = (v4f32)__msa_ld_w((const float*)scale_in_data + p * 8 + 4, 0);

                    v4f32 _sumfp32_0 = (v4f32)__msa_ffint_s_w(_sum0);
                    v4f32 _sumfp32_1 = (v4f32)__msa_ffint_s_w(_sum1);

                    if (bias_term)
                    {
                        v4f32 _bias0 = (v4f32)__msa_ld_w((const float*)bias_data + p * 8, 0);
                        v4f32 _bias1 = (v4f32)__msa_ld_w((const float*)bias_data + p * 8 + 4, 0);
                        _sumfp32_0 = __msa_fmadd_w(_bias0, _sumfp32_0, _scale_in0);
                        _sumfp32_1 = __msa_fmadd_w(_bias1, _sumfp32_1, _scale_in1);
                    }
                    else
                    {
                        _sumfp32_0 = __msa_fmul_w(_sumfp32_0, _scale_in0);
                        _sumfp32_1 = __msa_fmul_w(_sumfp32_1, _scale_in1);
                    }

                    _sumfp32_0 = activation_ps(_sumfp32_0, activation_type, activation_params);
                    _sumfp32_1 = activation_ps(_sumfp32_1, activation_type, activation_params);

                    __msa_st_w((v4i32)_sumfp32_0, outptr, 0);
                    __msa_st_w((v4i32)_sumfp32_1, outptr + 4, 0);
                    outptr += 8;
                }
            }
        }
#endif // __mips_msa

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

#if __mips_msa
    if (out_elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            v4i32 _sum0 = __msa_fill_w(0);
            v4i32 _sum1 = __msa_fill_w(0);

            const signed char* kptr = weight_data_tm.row<const signed char>(p);
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

            // dequantize and relu
            v4f32 _scale_in0 = (v4f32)__msa_ld_w((const float*)scale_in_data + p * 8, 0);
            v4f32 _scale_in1 = (v4f32)__msa_ld_w((const float*)scale_in_data + p * 8 + 4, 0);

            v4f32 _sumfp32_0 = (v4f32)__msa_ffint_s_w(_sum0);
            v4f32 _sumfp32_1 = (v4f32)__msa_ffint_s_w(_sum1);

            if (bias_term)
            {
                v4f32 _bias0 = (v4f32)__msa_ld_w((const float*)bias_data + p * 8, 0);
                v4f32 _bias1 = (v4f32)__msa_ld_w((const float*)bias_data + p * 8 + 4, 0);
                _sumfp32_0 = __msa_fmadd_w(_bias0, _sumfp32_0, _scale_in0);
                _sumfp32_1 = __msa_fmadd_w(_bias1, _sumfp32_1, _scale_in1);
            }
            else
            {
                _sumfp32_0 = __msa_fmul_w(_sumfp32_0, _scale_in0);
                _sumfp32_1 = __msa_fmul_w(_sumfp32_1, _scale_in1);
            }

            _sumfp32_0 = activation_ps(_sumfp32_0, activation_type, activation_params);
            _sumfp32_1 = activation_ps(_sumfp32_1, activation_type, activation_params);

            float* outptr = (float*)top_blob + p * 8;
            __msa_st_w((v4i32)_sumfp32_0, outptr, 0);
            __msa_st_w((v4i32)_sumfp32_1, outptr + 4, 0);
        }
    }
#endif // __mips_msa

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
