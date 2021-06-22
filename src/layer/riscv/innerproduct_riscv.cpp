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

#include "innerproduct_riscv.h"

#include "layer_type.h"

#if __riscv_vector
#ifdef RVV_SPEC_0_7
#include "riscv_v_071_fix.h"
#else
#include <riscv_vector.h>
#endif
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

namespace ncnn {

InnerProduct_riscv::InnerProduct_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector

    flatten = 0;
}

int InnerProduct_riscv::create_pipeline(const Option& opt)
{
#if __riscv_vector
    if (opt.use_packing_layout || opt.use_int8_inference)
    {
        flatten = ncnn::create_layer(ncnn::LayerType::Flatten);

        ncnn::ParamDict pd;

        flatten->load_param(pd);

        flatten->create_pipeline(opt);
    }
#endif // __riscv_vector

#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

    return 0;
}

int InnerProduct_riscv::destroy_pipeline(const Option& opt)
{
    if (flatten)
    {
        flatten->destroy_pipeline(opt);
        delete flatten;
        flatten = 0;
    }

    return 0;
}

int InnerProduct_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        Mat bottom_blob_unpacked = bottom_blob;
        if (bottom_blob.elempack != 1)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_allocator = opt.workspace_allocator;

            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
        }

        Mat bottom_blob_unpacked_fp32 = bottom_blob_unpacked;
        if (bottom_blob_unpacked.elembits() == 16)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_allocator = opt.workspace_allocator;

            cast_float16_to_float32(bottom_blob_unpacked, bottom_blob_unpacked_fp32, opt_pack1);
        }

        Option opt_unpacked = opt;
        opt_unpacked.use_packing_layout = false;
        return InnerProduct::forward_int8(bottom_blob_unpacked_fp32, top_blob, opt_unpacked);
    }
#endif

    int elembits = bottom_blob.elembits();

#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
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
#if __riscv_vector
            if (elempack == packn)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output; p++)
                {
                    const float* kptr = (const float*)weight_data + num_input * p;
                    const float* m = bottom_blob.row(j);

                    const word_type vl = vsetvl_e32m1(packn);
                    vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);

                    if (bias_term)
                    {
                        _sum = vfmv_v_f_f32m1(bias_data[p], vl);
                    }

                    int n = num_input;
                    while (n > 0)
                    {
                        vfloat32m1_t _val = vle32_v_f32m1(m, vl);
                        _sum = vfmacc_vf_f32m1(_sum, *kptr, _val, vl);

                        m += packn;
                        kptr += 1;
                        n -= 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params, vl);

                    vse32_v_f32m1(outptr, _sum, vl);
                    outptr += packn;
                }
            }
#endif // __riscv_vector

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

#if __riscv_vector
    if (elempack == packn)
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

#if __riscv_vector
    int nn_num_output = num_output / packn;
    int remain_num_output_start = nn_num_output * packn;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_num_output; pp++)
    {
        int p = pp * packn;

        const word_type vl = vsetvl_e32m1(packn);
        vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);

        if (bias_term)
        {
            _sum = vle32_v_f32m1((const float*)bias_data + p, vl);
        }

        const float* w = weight_data_ptr + num_input * p;

        // channels
        for (int q = 0; q < channels; q++)
        {
            const float* m = bottom_blob.channel(q);

            int n = size;
            while (n > 0)
            {
                vfloat32m1_t _w = vlse32_v_f32m1(w, num_input * sizeof(float), vl);

                _sum = vfmacc_vf_f32m1(_sum, *m, _w, vl);

                m += 1;
                w += 1;
                n -= 1;
            }
        }

        _sum = activation_ps(_sum, activation_type, activation_params, vl);

        vse32_v_f32m1((float*)top_blob + p, _sum, vl);
    }
#else  // __riscv_vector
    int remain_num_output_start = 0;
#endif // __riscv_vector

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

#if __riscv_vector && __riscv_zfh
int InnerProduct_riscv::create_pipeline_fp16s(const Option& opt)
{
    const int packn = csrr_vlenb() / 2;

    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;

    if (opt.use_packing_layout)
    {
        out_elempack = num_output % packn == 0 ? packn : 1;
    }

    // src = inch-outch
    // dst = pb-inch-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_fp16.create(num_input, num_output / out_elempack, (size_t)2u * out_elempack, out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            __fp16* g0 = weight_data_fp16.row<__fp16>(q / out_elempack);

            for (int p = 0; p < num_input; p++)
            {
                for (int j = 0; j < out_elempack; j++)
                {
                    *g0++ = (__fp16)(weight_data_r2.row(q + j)[p]);
                }
            }
        }
    }

    ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);

    return 0;
}

int InnerProduct_riscv::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int packn = csrr_vlenb() / 2;

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

        int num_output_elempack = opt.use_packing_layout && num_output % packn == 0 ? packn : 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
            if (elempack == packn && num_output_elempack == packn)
            {
                const word_type vl = vsetvl_e16m1(packn);

                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    for (int l = 0; l < packn; l++)
                    {
                        const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p * packn + l;
                        const __fp16* m = bottom_blob.row<const __fp16>(j);

                        vfloat32m2_t _sum = vfmv_v_f_f32m2(0.f, vl);

                        if (bias_term)
                        {
                            _sum = vfmv_v_f_f32m2(bias_data[p * packn + l], vl);
                        }

                        int n = num_input;
                        while (n > 0)
                        {
                            vfloat32m2_t _val = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(m, vl), vl);

                            _sum = vfmacc_vf_f32m2(_sum, *kptr, _val, vl);

                            m += packn;
                            kptr += packn;
                            n -= 1;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params, vl);

                        vse16_v_f16m1(outptr, vfncvt_f_f_w_f16m1(_sum, vl), vl);
                        outptr += packn;
                    }
                }
            }

            if (elempack == 1 && num_output_elempack == packn)
            {
                const word_type vl = vsetvl_e16m1(packn);

                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p * packn;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    vfloat32m2_t _sum = vfmv_v_f_f32m2(0.f, vl);

                    if (bias_term)
                    {
                        _sum = vle32_v_f32m2((const float*)bias_data + p * packn, vl);
                    }

                    int n = num_input;
                    while (n > 0)
                    {
                        vfloat32m2_t _w = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(kptr, vl), vl);

                        _sum = vfmacc_vf_f32m2(_sum, *m, _w, vl);

                        m += 1;
                        kptr += packn;
                        n -= 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params, vl);

                    vse16_v_f16m1(outptr, vfncvt_f_f_w_f16m1(_sum, vl), vl);
                    outptr += packn;
                }
            }

            if (elempack == packn && num_output_elempack == 1)
            {
                const word_type vl = vsetvl_e16m1(packn);

                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    vfloat32m2_t _sum = vfmv_v_f_f32m2(0.f, vl);

                    if (bias_term)
                    {
                        _sum = vfmv_v_f_f32m2(bias_data[p], vl);
                    }

                    int n = num_input;
                    while (n > 0)
                    {
                        vfloat32m2_t _val = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(m, vl), vl);

                        _sum = vfmacc_vf_f32m2(_sum, *kptr, _val, vl);

                        m += packn;
                        kptr += 1;
                        n -= 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params, vl);

                    vse16_v_f16m1(outptr, vfncvt_f_f_w_f16m1(_sum, vl), vl);
                    outptr += packn;
                }
            }

            if (elempack == 1 && num_output_elempack == 1)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        sum += (float)m[i] * (float)kptr[i];
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[0] = (__fp16)sum;
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

    int out_elempack = opt.use_packing_layout && num_output % packn == 0 ? packn : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (out_elempack == packn)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            const word_type vl = vsetvl_e16m1(packn);
            vfloat32m2_t _sum = vfmv_v_f_f32m2(0.f, vl);

            if (bias_term)
            {
                _sum = vle32_v_f32m2((const float*)bias_data + p * packn, vl);
            }

            const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            int n = num_input;
            while (n > 0)
            {
                vfloat32m2_t _w = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(kptr, vl), vl);

                _sum = vfmacc_vf_f32m2(_sum, (float)(*sptr), _w, vl);

                sptr += 1;
                kptr += packn;
                n -= 1;
            }

            _sum = activation_ps(_sum, activation_type, activation_params, vl);

            __fp16* outptr = (__fp16*)top_blob;
            vse16_v_f16m1(outptr + p * packn, vfncvt_f_f_w_f16m1(_sum, vl), vl);
        }
    }

    if (out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output; p++)
        {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            const __fp16* kptr = weight_data_fp16.row<__fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            int i = 0;
            for (; i < num_input; i++)
            {
                float v = (float)(*sptr);
                float k = (float)(*kptr);

                sum += v * k;

                sptr++;
                kptr++;
            }

            sum = activation_ss(sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            outptr[p] = (__fp16)sum;
        }
    }

    return 0;
}

int InnerProduct_riscv::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int packn = csrr_vlenb() / 2;

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

        int num_output_elempack = opt.use_packing_layout && num_output % packn == 0 ? packn : 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
            if (elempack == packn && num_output_elempack == packn)
            {
                const word_type vl = vsetvl_e16m1(packn);

                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    for (int l = 0; l < packn; l++)
                    {
                        const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p * packn + l;
                        const __fp16* m = bottom_blob.row<const __fp16>(j);

                        vfloat16m1_t _sum = vfmv_v_f_f16m1((__fp16)0.f, vl);

                        if (bias_term)
                        {
                            _sum = vfmv_v_f_f16m1(((const __fp16*)bias_data_fp16)[p * packn + l], vl);
                        }

                        int n = num_input;
                        while (n > 0)
                        {
                            vfloat16m1_t _val = vle16_v_f16m1(m, vl);

                            _sum = vfmacc_vf_f16m1(_sum, *kptr, _val, vl);

                            m += packn;
                            kptr += packn;
                            n -= 1;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params, vl);

                        vse16_v_f16m1(outptr, _sum, vl);
                        outptr += packn;
                    }
                }
            }

            if (elempack == 1 && num_output_elempack == packn)
            {
                const word_type vl = vsetvl_e16m1(packn);

                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p * packn;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    vfloat16m1_t _sum = vfmv_v_f_f16m1(0.f, vl);

                    if (bias_term)
                    {
                        _sum = vle16_v_f16m1((const __fp16*)bias_data_fp16 + p * packn, vl);
                    }

                    int n = num_input;
                    while (n > 0)
                    {
                        vfloat16m1_t _w = vle16_v_f16m1(kptr, vl);

                        _sum = vfmacc_vf_f16m1(_sum, *m, _w, vl);

                        m += 1;
                        kptr += packn;
                        n -= 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params, vl);

                    vse16_v_f16m1(outptr, _sum, vl);
                    outptr += packn;
                }
            }

            if (elempack == packn && num_output_elempack == 1)
            {
                const word_type vl = vsetvl_e16m1(packn);

                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    vfloat16m1_t _sum = vfmv_v_f_f16m1(0.f, vl);

                    if (bias_term)
                    {
                        _sum = vfmv_v_f_f16m1(((const __fp16*)bias_data_fp16)[p], vl);
                    }

                    int n = num_input;
                    while (n > 0)
                    {
                        vfloat16m1_t _val = vle16_v_f16m1(m, vl);

                        _sum = vfmacc_vf_f16m1(_sum, *kptr, _val, vl);

                        m += packn;
                        kptr += 1;
                        n -= 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params, vl);

                    vse16_v_f16m1(outptr, _sum, vl);
                    outptr += packn;
                }
            }

            if (elempack == 1 && num_output_elempack == 1)
            {
                __fp16* outptr = top_blob.row<__fp16>(j);

                for (int p = 0; p < num_output; p++)
                {
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + num_input * p;
                    const __fp16* m = bottom_blob.row<const __fp16>(j);

                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        sum += (float)(m[i] * kptr[i]);
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[0] = (__fp16)sum;
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

    int out_elempack = opt.use_packing_layout && num_output % packn == 0 ? packn : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (out_elempack == packn)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            const word_type vl = vsetvl_e16m1(packn);
            vfloat16m1_t _sum = vfmv_v_f_f16m1(0.f, vl);

            if (bias_term)
            {
                _sum = vle16_v_f16m1((const __fp16*)bias_data_fp16 + p * packn, vl);
            }

            const __fp16* kptr = weight_data_fp16.row<const __fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            int n = num_input;
            while (n > 0)
            {
                vfloat16m1_t _w = vle16_v_f16m1(kptr, vl);

                _sum = vfmacc_vf_f16m1(_sum, *sptr, _w, vl);

                sptr += 1;
                kptr += packn;
                n -= 1;
            }

            _sum = activation_ps(_sum, activation_type, activation_params, vl);

            __fp16* outptr = (__fp16*)top_blob;
            vse16_v_f16m1(outptr + p * packn, _sum, vl);
        }
    }

    if (out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output; p++)
        {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            const __fp16* kptr = weight_data_fp16.row<__fp16>(p);

            const __fp16* sptr = bottom_blob_flattened;

            int i = 0;
            for (; i < num_input; i++)
            {
                __fp16 v = *sptr;
                __fp16 k = *kptr;

                sum += (float)(v * k);

                sptr++;
                kptr++;
            }

            sum = activation_ss(sum, activation_type, activation_params);

            __fp16* outptr = (__fp16*)top_blob;
            outptr[p] = (__fp16)sum;
        }
    }

    return 0;
}
#endif // __riscv_vector && __riscv_zfh

} // namespace ncnn
