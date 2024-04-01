// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "innerproduct_loongarch.h"

#include "layer_type.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "lsx_mathfun.h"
#endif // __loongarch_sx

#include "loongarch_activation.h"

namespace ncnn {

InnerProduct_loongarch::InnerProduct_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx

    flatten = 0;
}

int InnerProduct_loongarch::create_pipeline(const Option& opt)
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
        return create_pipeline_int8_loongarch(opt);
    }
#endif

#if __loongarch_sx
    if (opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;

#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __loongarch_sx

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

int InnerProduct_loongarch::destroy_pipeline(const Option& opt)
{
    if (flatten)
    {
        flatten->destroy_pipeline(opt);
        delete flatten;
        flatten = 0;
    }

    return 0;
}

int InnerProduct_loongarch::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && int8_scale_term)
    {
        return forward_int8_loongarch(bottom_blob, top_blob, opt);
    }
#endif

#if __loongarch_sx
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
#if __loongarch_sx
        if (opt.use_packing_layout)
        {
            num_output_elempack = num_output % 4 == 0 ? 4 : 1;
        }
#endif

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
#if __loongarch_sx
            if (elempack == 4 && num_output_elempack == 4)
            {
                float* outptr = top_blob.row(j);

                for (int p = 0; p < num_output / num_output_elempack; p++)
                {
                    const float* kptr = weight_data_tm.row(p);
                    const float* m = bottom_blob.row(j);

                    __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _sum3 = (__m128)__lsx_vreplgr2vr_w(0);

                    if (bias_term)
                    {
                        _sum0 = __lsx_vreplfr2vr_s(bias_data[p * 4 + 0]);
                        _sum1 = __lsx_vreplfr2vr_s(bias_data[p * 4 + 1]);
                        _sum2 = __lsx_vreplfr2vr_s(bias_data[p * 4 + 2]);
                        _sum3 = __lsx_vreplfr2vr_s(bias_data[p * 4 + 3]);
                    }

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 16);
                        __m128 _val = (__m128)__lsx_vld(m, 0);
                        __m128i _w = __lsx_vld(kptr, 0);
                        _sum0 = __lsx_vfmadd_s((__m128)__lsx_vreplvei_w(_w, 0), _val, _sum0);
                        _sum1 = __lsx_vfmadd_s((__m128)__lsx_vreplvei_w(_w, 1), _val, _sum1);
                        _sum2 = __lsx_vfmadd_s((__m128)__lsx_vreplvei_w(_w, 2), _val, _sum2);
                        _sum3 = __lsx_vfmadd_s((__m128)__lsx_vreplvei_w(_w, 3), _val, _sum3);

                        m += 4;
                        kptr += 4;
                    }

                    _sum0 = activation_ps(_sum0, activation_type, activation_params);
                    _sum1 = activation_ps(_sum1, activation_type, activation_params);
                    _sum2 = activation_ps(_sum2, activation_type, activation_params);
                    _sum3 = activation_ps(_sum3, activation_type, activation_params);

                    __lsx_vst(_sum0, outptr, 0);
                    __lsx_vst(_sum1, outptr + 4, 0);
                    __lsx_vst(_sum2, outptr + 8, 0);
                    __lsx_vst(_sum3, outptr + 12, 0);
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

                    __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _sum3 = (__m128)__lsx_vreplgr2vr_w(0);

                    if (bias_term)
                    {
                        _sum0 = (__m128)__lsx_vld((const float*)bias_data + p * 4, 0);
                    }

                    int i = 0;
                    for (; i + 3 < num_input; i += 4)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 64);
                        __m128i _val = __lsx_vld(m, 0);
                        __m128 _w0 = (__m128)__lsx_vld(kptr, 0);
                        __m128 _w1 = (__m128)__lsx_vld(kptr + 4, 0);
                        __m128 _w2 = (__m128)__lsx_vld(kptr + 8, 0);
                        __m128 _w3 = (__m128)__lsx_vld(kptr + 12, 0);
                        _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val, 0), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w1, (__m128)__lsx_vreplvei_w(_val, 1), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w2, (__m128)__lsx_vreplvei_w(_val, 2), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w3, (__m128)__lsx_vreplvei_w(_val, 3), _sum3);

                        m += 4;
                        kptr += 16;
                    }
                    for (; i < num_input; i++)
                    {
                        __m128 _val = __lsx_vreplfr2vr_s(m[0]);
                        __m128 _w = (__m128)__lsx_vld(kptr, 0);
                        _sum0 = __lsx_vfmadd_s(_w, _val, _sum0);

                        m += 1;
                        kptr += 4;
                    }

                    _sum0 = __lsx_vfadd_s(_sum0, _sum1);
                    _sum2 = __lsx_vfadd_s(_sum2, _sum3);
                    _sum0 = __lsx_vfadd_s(_sum0, _sum2);

                    _sum0 = activation_ps(_sum0, activation_type, activation_params);

                    __lsx_vst(_sum0, outptr, 0);
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

                    __m128 _sum = (__m128)__lsx_vreplgr2vr_w(0);

                    if (bias_term)
                    {
                        _sum = __lsx_vreplfr2vr_s(bias_data[p]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 4);
                        __m128 _val = (__m128)__lsx_vld(m, 0);
                        __m128 _k = __lsx_vreplfr2vr_s(kptr[0]);
                        _sum = __lsx_vfmadd_s(_k, _val, _sum);

                        m += 4;
                        kptr += 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    __lsx_vst(_sum, outptr, 0);
                    outptr += 4;
                }
            }
#endif // __loongarch_sx

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
#if __loongarch_sx
                    __m128 _sum = (__m128)__lsx_vreplgr2vr_w(0);
                    for (; i + 3 < num_input; i += 4)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 16);
                        __m128 _m = (__m128)__lsx_vld(m, 0);
                        __m128 _w = (__m128)__lsx_vld(kptr, 0);
                        _sum = __lsx_vfmadd_s(_w, _m, _sum);

                        m += 4;
                        kptr += 4;
                    }
                    sum += __lsx_reduce_fadd_s(_sum);
#endif // __loongarch_sx
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
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __loongarch_sx
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __loongarch_sx
    if (out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum3 = (__m128)__lsx_vreplgr2vr_w(0);

            if (bias_term)
            {
                _sum0 = (__m128)__lsx_vld((const float*)bias_data + p * 4, 0);
            }

            const float* kptr = weight_data_tm.row(p);

            const float* sptr = bottom_blob_flattened;

            int i = 0;
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(sptr + 16);
                __builtin_prefetch(kptr + 64);
                __m128i _val = __lsx_vld(sptr, 0);
                __m128 _w0 = (__m128)__lsx_vld(kptr, 0);
                __m128 _w1 = (__m128)__lsx_vld(kptr + 4, 0);
                __m128 _w2 = (__m128)__lsx_vld(kptr + 8, 0);
                __m128 _w3 = (__m128)__lsx_vld(kptr + 12, 0);
                _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val, 0), _sum0);
                _sum1 = __lsx_vfmadd_s(_w1, (__m128)__lsx_vreplvei_w(_val, 1), _sum1);
                _sum2 = __lsx_vfmadd_s(_w2, (__m128)__lsx_vreplvei_w(_val, 2), _sum2);
                _sum3 = __lsx_vfmadd_s(_w3, (__m128)__lsx_vreplvei_w(_val, 3), _sum3);

                sptr += 4;
                kptr += 16;
            }
            for (; i < num_input; i++)
            {
                __m128 _val = __lsx_vreplfr2vr_s(sptr[0]);
                __m128 _w = (__m128)__lsx_vld(kptr, 0);
                _sum0 = __lsx_vfmadd_s(_w, _val, _sum0);

                sptr += 1;
                kptr += 4;
            }

            _sum0 = __lsx_vfadd_s(_sum0, _sum1);
            _sum2 = __lsx_vfadd_s(_sum2, _sum3);
            _sum0 = __lsx_vfadd_s(_sum0, _sum2);

            _sum0 = activation_ps(_sum0, activation_type, activation_params);

            float* outptr = top_blob;
            __lsx_vst(_sum0, outptr + p * 4, 0);
        }
    }
#endif // __loongarch_sx

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
#if __loongarch_sx
            __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum3 = (__m128)__lsx_vreplgr2vr_w(0);
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(m + 16);
                __builtin_prefetch(w0 + 16);
                __builtin_prefetch(w1 + 16);
                __builtin_prefetch(w2 + 16);
                __builtin_prefetch(w3 + 16);
                __m128 _m = (__m128)__lsx_vld(m, 0);
                __m128 _w0 = (__m128)__lsx_vld(w0, 0);
                __m128 _w1 = (__m128)__lsx_vld(w1, 0);
                __m128 _w2 = (__m128)__lsx_vld(w2, 0);
                __m128 _w3 = (__m128)__lsx_vld(w3, 0);
                _sum0 = __lsx_vfmadd_s(_w0, _m, _sum0);
                _sum1 = __lsx_vfmadd_s(_w1, _m, _sum1);
                _sum2 = __lsx_vfmadd_s(_w2, _m, _sum2);
                _sum3 = __lsx_vfmadd_s(_w3, _m, _sum3);

                m += 4;
                w0 += 4;
                w1 += 4;
                w2 += 4;
                w3 += 4;
            }
#endif // __loongarch_sx
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

#if __loongarch_sx
            sum0 += __lsx_reduce_fadd_s(_sum0);
            sum1 += __lsx_reduce_fadd_s(_sum1);
            sum2 += __lsx_reduce_fadd_s(_sum2);
            sum3 += __lsx_reduce_fadd_s(_sum3);
#endif // __loongarch_sx

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
#if __loongarch_sx
            __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(m + 16);
                __builtin_prefetch(w + 16);
                __m128 _m = (__m128)__lsx_vld(m, 0);
                __m128 _w = (__m128)__lsx_vld(w, 0);
                _sum0 = __lsx_vfmadd_s(_w, _m, _sum0);

                m += 4;
                w += 4;
            }
            sum += __lsx_reduce_fadd_s(_sum0);
#endif // __loongarch_sx
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

#if __loongarch_sx
int InnerProduct_loongarch::create_pipeline_fp16s(const Option& opt)
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
                __m128i _r0 = __lsx_vld(k0, 0);
                __m128i _r1 = __lsx_vld(k1, 0);
                __m128i _r2 = __lsx_vld(k2, 0);
                __m128i _r3 = __lsx_vld(k3, 0);

                __m128i _r01r = __lsx_vilvl_w(_r1, _r0);
                __m128i _r01l = __lsx_vilvh_w(_r1, _r0);
                __m128i _r23r = __lsx_vilvl_w(_r3, _r2);
                __m128i _r23l = __lsx_vilvh_w(_r3, _r2);
                __m128i _r0123_0 = __lsx_vilvl_d(_r23r, _r01r);
                __m128i _r0123_1 = __lsx_vilvh_d(_r23r, _r01r);
                __m128i _r0123_2 = __lsx_vilvl_d(_r23l, _r01l);
                __m128i _r0123_3 = __lsx_vilvh_d(_r23l, _r01l);

                __m128i _p0 = __lsx_vfcvt_h_s((__m128)_r0123_1, (__m128)_r0123_0);
                __m128i _p1 = __lsx_vfcvt_h_s((__m128)_r0123_3, (__m128)_r0123_2);

                __lsx_vst(_p0, g0, 0);
                __lsx_vst(_p1, g0 + 8, 0);

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

int InnerProduct_loongarch::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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

                    __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _sum3 = (__m128)__lsx_vreplgr2vr_w(0);

                    if (bias_term)
                    {
                        _sum0 = (__m128)__lsx_vreplfr2vr_s(bias_data[p * 4 + 0]);
                        _sum1 = (__m128)__lsx_vreplfr2vr_s(bias_data[p * 4 + 1]);
                        _sum2 = (__m128)__lsx_vreplfr2vr_s(bias_data[p * 4 + 2]);
                        _sum3 = (__m128)__lsx_vreplfr2vr_s(bias_data[p * 4 + 3]);
                    }

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 16);
                        __m128 _val = (__m128)__lsx_vld(m, 0);
                        __m128i _w = (__m128i)__lsx_vfcvtl_s_h(__lsx_vld(kptr, 0));
                        _sum0 = __lsx_vfmadd_s((__m128)__lsx_vreplvei_w(_w, 0), _val, _sum0);
                        _sum1 = __lsx_vfmadd_s((__m128)__lsx_vreplvei_w(_w, 1), _val, _sum1);
                        _sum2 = __lsx_vfmadd_s((__m128)__lsx_vreplvei_w(_w, 2), _val, _sum2);
                        _sum3 = __lsx_vfmadd_s((__m128)__lsx_vreplvei_w(_w, 3), _val, _sum3);

                        m += 4;
                        kptr += 4;
                    }

                    _sum0 = activation_ps(_sum0, activation_type, activation_params);
                    _sum1 = activation_ps(_sum1, activation_type, activation_params);
                    _sum2 = activation_ps(_sum2, activation_type, activation_params);
                    _sum3 = activation_ps(_sum3, activation_type, activation_params);

                    __lsx_vst(_sum0, outptr, 0);
                    __lsx_vst(_sum1, outptr + 4, 0);
                    __lsx_vst(_sum2, outptr + 8, 0);
                    __lsx_vst(_sum3, outptr + 12, 0);
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

                    __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
                    __m128 _sum3 = (__m128)__lsx_vreplgr2vr_w(0);

                    if (bias_term)
                    {
                        _sum0 = (__m128)__lsx_vld((const float*)bias_data + p * 4, 0);
                    }

                    int i = 0;
                    for (; i + 3 < num_input; i += 4)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 64);
                        __m128i _val = __lsx_vld(m, 0);
                        __m128i _w01 = __lsx_vld(kptr, 0);
                        __m128i _w23 = __lsx_vld(kptr + 8, 0);
                        __m128 _w0 = __lsx_vfcvtl_s_h(_w01);
                        __m128 _w1 = __lsx_vfcvth_s_h(_w01);
                        __m128 _w2 = __lsx_vfcvtl_s_h(_w23);
                        __m128 _w3 = __lsx_vfcvth_s_h(_w23);
                        _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val, 0), _sum0);
                        _sum1 = __lsx_vfmadd_s(_w1, (__m128)__lsx_vreplvei_w(_val, 1), _sum1);
                        _sum2 = __lsx_vfmadd_s(_w2, (__m128)__lsx_vreplvei_w(_val, 2), _sum2);
                        _sum3 = __lsx_vfmadd_s(_w3, (__m128)__lsx_vreplvei_w(_val, 3), _sum3);

                        m += 4;
                        kptr += 16;
                    }
                    for (; i < num_input; i++)
                    {
                        __m128 _val = __lsx_vreplfr2vr_s(m[0]);
                        __m128 _w = __lsx_vfcvtl_s_h(__lsx_vld(kptr, 0));
                        _sum0 = __lsx_vfmadd_s(_w, _val, _sum0);

                        m += 1;
                        kptr += 4;
                    }

                    _sum0 = __lsx_vfadd_s(_sum0, _sum1);
                    _sum2 = __lsx_vfadd_s(_sum2, _sum3);
                    _sum0 = __lsx_vfadd_s(_sum0, _sum2);

                    _sum0 = activation_ps(_sum0, activation_type, activation_params);

                    __lsx_vst(_sum0, outptr, 0);
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

                    __m128 _sum = (__m128)__lsx_vreplgr2vr_w(0);

                    if (bias_term)
                    {
                        _sum = __lsx_vreplfr2vr_s(bias_data[p]);
                    }

                    for (int i = 0; i < num_input; i++)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 4);
                        __m128 _val = (__m128)__lsx_vld(m, 0);
                        __m128 _k = __lsx_vreplfr2vr_s(float16_to_float32(kptr[0]));
                        _sum = __lsx_vfmadd_s(_k, _val, _sum);

                        m += 4;
                        kptr += 1;
                    }

                    _sum = activation_ps(_sum, activation_type, activation_params);

                    __lsx_vst(_sum, outptr, 0);
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
                    __m128 _sum = (__m128)__lsx_vreplgr2vr_w(0);
                    for (; i + 3 < num_input; i += 4)
                    {
                        __builtin_prefetch(m + 16);
                        __builtin_prefetch(kptr + 16);
                        __m128 _m = (__m128)__lsx_vld(m, 0);
                        __m128 _w = __lsx_vfcvtl_s_h(__lsx_vld(kptr, 0));
                        _sum = __lsx_vfmadd_s(_w, _m, _sum);

                        m += 4;
                        kptr += 4;
                    }
                    sum += __lsx_reduce_fadd_s(_sum);
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
            __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum3 = (__m128)__lsx_vreplgr2vr_w(0);

            if (bias_term)
            {
                _sum0 = (__m128)__lsx_vld((const float*)bias_data + p * 4, 0);
            }

            const unsigned short* kptr = weight_data_tm.row<const unsigned short>(p);

            const float* sptr = bottom_blob_flattened;

            int i = 0;
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(sptr + 16);
                __builtin_prefetch(kptr + 64);
                __m128i _val = __lsx_vld(sptr, 0);
                __m128i _w01 = __lsx_vld(kptr, 0);
                __m128i _w23 = __lsx_vld(kptr + 8, 0);
                __m128 _w0 = __lsx_vfcvtl_s_h(_w01);
                __m128 _w1 = __lsx_vfcvth_s_h(_w01);
                __m128 _w2 = __lsx_vfcvtl_s_h(_w23);
                __m128 _w3 = __lsx_vfcvth_s_h(_w23);
                _sum0 = __lsx_vfmadd_s(_w0, (__m128)__lsx_vreplvei_w(_val, 0), _sum0);
                _sum1 = __lsx_vfmadd_s(_w1, (__m128)__lsx_vreplvei_w(_val, 1), _sum1);
                _sum2 = __lsx_vfmadd_s(_w2, (__m128)__lsx_vreplvei_w(_val, 2), _sum2);
                _sum3 = __lsx_vfmadd_s(_w3, (__m128)__lsx_vreplvei_w(_val, 3), _sum3);

                sptr += 4;
                kptr += 16;
            }
            for (; i < num_input; i++)
            {
                __m128 _val = __lsx_vreplfr2vr_s(sptr[0]);
                __m128 _w = __lsx_vfcvtl_s_h(__lsx_vld(kptr, 0));
                _sum0 = __lsx_vfmadd_s(_w, _val, _sum0);

                sptr += 1;
                kptr += 4;
            }

            _sum0 = __lsx_vfadd_s(_sum0, _sum1);
            _sum2 = __lsx_vfadd_s(_sum2, _sum3);
            _sum0 = __lsx_vfadd_s(_sum0, _sum2);

            _sum0 = activation_ps(_sum0, activation_type, activation_params);

            float* outptr = top_blob;
            __lsx_vst(_sum0, outptr + p * 4, 0);
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
            __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum1 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum2 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _sum3 = (__m128)__lsx_vreplgr2vr_w(0);
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(m + 16);
                __builtin_prefetch(w0 + 16);
                __builtin_prefetch(w1 + 16);
                __builtin_prefetch(w2 + 16);
                __builtin_prefetch(w3 + 16);
                __m128 _m = (__m128)__lsx_vld(m, 0);
                __m128 _w0 = __lsx_vfcvtl_s_h(__lsx_vld(w0, 0));
                __m128 _w1 = __lsx_vfcvtl_s_h(__lsx_vld(w1, 0));
                __m128 _w2 = __lsx_vfcvtl_s_h(__lsx_vld(w2, 0));
                __m128 _w3 = __lsx_vfcvtl_s_h(__lsx_vld(w3, 0));
                _sum0 = __lsx_vfmadd_s(_w0, _m, _sum0);
                _sum1 = __lsx_vfmadd_s(_w1, _m, _sum1);
                _sum2 = __lsx_vfmadd_s(_w2, _m, _sum2);
                _sum3 = __lsx_vfmadd_s(_w3, _m, _sum3);

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

            sum0 += __lsx_reduce_fadd_s(_sum0);
            sum1 += __lsx_reduce_fadd_s(_sum1);
            sum2 += __lsx_reduce_fadd_s(_sum2);
            sum3 += __lsx_reduce_fadd_s(_sum3);

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
            __m128 _sum0 = (__m128)__lsx_vreplgr2vr_w(0);
            for (; i + 3 < num_input; i += 4)
            {
                __builtin_prefetch(m + 16);
                __builtin_prefetch(w + 16);
                __m128 _m = (__m128)__lsx_vld(m, 0);
                __m128 _w = __lsx_vfcvtl_s_h(__lsx_vld(w, 0));
                _sum0 = __lsx_vfmadd_s(_w, _m, _sum0);

                m += 4;
                w += 4;
            }
            sum += __lsx_reduce_fadd_s(_sum0);
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
#endif // __loongarch_sx

#if NCNN_INT8
int InnerProduct_loongarch::create_pipeline_int8_loongarch(const Option& opt)
{
    const int num_input = weight_data_size / num_output;

    int out_elempack = 1;
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 8 == 0 ? 8 : 1;
    }
#endif // __loongarch_sx

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

int InnerProduct_loongarch::forward_int8_loongarch(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
#if __loongarch_sx
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
#if __loongarch_sx
        if (opt.use_packing_layout)
        {
            num_output_elempack = num_output % 8 == 0 ? 8 : 1;
        }
#endif

#if __loongarch_sx
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

                    __m128i _sum00 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum01 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum10 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum11 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum20 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum21 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum30 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum31 = __lsx_vreplgr2vr_w(0);

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        __builtin_prefetch(m0 + 4);
                        __builtin_prefetch(m1 + 4);
                        __builtin_prefetch(m2 + 4);
                        __builtin_prefetch(m3 + 4);
                        __builtin_prefetch(kptr + 32);
                        __m128i _val0 = __lsx_vreplgr2vr_h((short)m0[0]);
                        __m128i _val1 = __lsx_vreplgr2vr_h((short)m1[0]);
                        __m128i _val2 = __lsx_vreplgr2vr_h((short)m2[0]);
                        __m128i _val3 = __lsx_vreplgr2vr_h((short)m3[0]);

                        __m128i _w = __lsx_vld(kptr, 0);
                        __m128i _w16 = __lsx_vilvl_b(__lsx_vslti_b(_w, 0), _w);

                        __m128i _s0 = __lsx_vmul_h(_val0, _w16);
                        __m128i _s1 = __lsx_vmul_h(_val1, _w16);
                        __m128i _s2 = __lsx_vmul_h(_val2, _w16);
                        __m128i _s3 = __lsx_vmul_h(_val3, _w16);
                        __m128i _exts0 = __lsx_vslti_h(_s0, 0);
                        __m128i _exts1 = __lsx_vslti_h(_s1, 0);
                        __m128i _exts2 = __lsx_vslti_h(_s2, 0);
                        __m128i _exts3 = __lsx_vslti_h(_s3, 0);
                        __m128i _s0l = __lsx_vilvl_h(_exts0, _s0);
                        __m128i _s0h = __lsx_vilvh_h(_exts0, _s0);
                        __m128i _s1l = __lsx_vilvl_h(_exts1, _s1);
                        __m128i _s1h = __lsx_vilvh_h(_exts1, _s1);
                        __m128i _s2l = __lsx_vilvl_h(_exts2, _s2);
                        __m128i _s2h = __lsx_vilvh_h(_exts2, _s2);
                        __m128i _s3l = __lsx_vilvl_h(_exts3, _s3);
                        __m128i _s3h = __lsx_vilvh_h(_exts3, _s3);

                        _sum00 = __lsx_vadd_w(_sum00, _s0l);
                        _sum01 = __lsx_vadd_w(_sum01, _s0h);
                        _sum10 = __lsx_vadd_w(_sum10, _s1l);
                        _sum11 = __lsx_vadd_w(_sum11, _s1h);
                        _sum20 = __lsx_vadd_w(_sum20, _s2l);
                        _sum21 = __lsx_vadd_w(_sum21, _s2h);
                        _sum30 = __lsx_vadd_w(_sum30, _s3l);
                        _sum31 = __lsx_vadd_w(_sum31, _s3h);

                        m0++;
                        m1++;
                        m2++;
                        m3++;
                        kptr += 8;
                    }

                    // dequantize and relu
                    __m128 _scale_in0 = (__m128)__lsx_vld((const float*)scale_in_data + p * 8, 0);
                    __m128 _scale_in1 = (__m128)__lsx_vld((const float*)scale_in_data + p * 8 + 4, 0);

                    __m128 _sumfp32_00 = __lsx_vffint_s_w(_sum00);
                    __m128 _sumfp32_01 = __lsx_vffint_s_w(_sum01);
                    __m128 _sumfp32_10 = __lsx_vffint_s_w(_sum10);
                    __m128 _sumfp32_11 = __lsx_vffint_s_w(_sum11);
                    __m128 _sumfp32_20 = __lsx_vffint_s_w(_sum20);
                    __m128 _sumfp32_21 = __lsx_vffint_s_w(_sum21);
                    __m128 _sumfp32_30 = __lsx_vffint_s_w(_sum30);
                    __m128 _sumfp32_31 = __lsx_vffint_s_w(_sum31);
                    if (bias_term)
                    {
                        __m128 _bias0 = (__m128)__lsx_vld((const float*)bias_data + p * 8, 0);
                        __m128 _bias1 = (__m128)__lsx_vld((const float*)bias_data + p * 8 + 4, 0);
                        _sumfp32_00 = __lsx_vfmadd_s(_scale_in0, _sumfp32_00, _bias0);
                        _sumfp32_01 = __lsx_vfmadd_s(_scale_in1, _sumfp32_01, _bias1);
                        _sumfp32_10 = __lsx_vfmadd_s(_scale_in0, _sumfp32_10, _bias0);
                        _sumfp32_11 = __lsx_vfmadd_s(_scale_in1, _sumfp32_11, _bias1);
                        _sumfp32_20 = __lsx_vfmadd_s(_scale_in0, _sumfp32_20, _bias0);
                        _sumfp32_21 = __lsx_vfmadd_s(_scale_in1, _sumfp32_21, _bias1);
                        _sumfp32_30 = __lsx_vfmadd_s(_scale_in0, _sumfp32_30, _bias0);
                        _sumfp32_31 = __lsx_vfmadd_s(_scale_in1, _sumfp32_31, _bias1);
                    }
                    else
                    {
                        _sumfp32_00 = __lsx_vfmul_s(_sumfp32_00, _scale_in0);
                        _sumfp32_01 = __lsx_vfmul_s(_sumfp32_01, _scale_in1);
                        _sumfp32_10 = __lsx_vfmul_s(_sumfp32_10, _scale_in0);
                        _sumfp32_11 = __lsx_vfmul_s(_sumfp32_11, _scale_in1);
                        _sumfp32_20 = __lsx_vfmul_s(_sumfp32_20, _scale_in0);
                        _sumfp32_21 = __lsx_vfmul_s(_sumfp32_21, _scale_in1);
                        _sumfp32_30 = __lsx_vfmul_s(_sumfp32_30, _scale_in0);
                        _sumfp32_31 = __lsx_vfmul_s(_sumfp32_31, _scale_in1);
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
                    __m128i _r01r = __lsx_vilvl_w((__m128i)_sumfp32_10, (__m128i)_sumfp32_00);
                    __m128i _r01l = __lsx_vilvh_w((__m128i)_sumfp32_10, (__m128i)_sumfp32_00);
                    __m128i _r23r = __lsx_vilvl_w((__m128i)_sumfp32_30, (__m128i)_sumfp32_20);
                    __m128i _r23l = __lsx_vilvh_w((__m128i)_sumfp32_30, (__m128i)_sumfp32_20);
                    __m128i _r45r = __lsx_vilvl_w((__m128i)_sumfp32_11, (__m128i)_sumfp32_01);
                    __m128i _r45l = __lsx_vilvh_w((__m128i)_sumfp32_11, (__m128i)_sumfp32_01);
                    __m128i _r67r = __lsx_vilvl_w((__m128i)_sumfp32_31, (__m128i)_sumfp32_21);
                    __m128i _r67l = __lsx_vilvh_w((__m128i)_sumfp32_31, (__m128i)_sumfp32_21);
                    _sumfp32_00 = (__m128)__lsx_vilvl_d(_r23r, _r01r);
                    _sumfp32_10 = (__m128)__lsx_vilvh_d(_r23r, _r01r);
                    _sumfp32_20 = (__m128)__lsx_vilvl_d(_r23l, _r01l);
                    _sumfp32_30 = (__m128)__lsx_vilvh_d(_r23l, _r01l);
                    _sumfp32_01 = (__m128)__lsx_vilvl_d(_r67r, _r45r);
                    _sumfp32_11 = (__m128)__lsx_vilvh_d(_r67r, _r45r);
                    _sumfp32_21 = (__m128)__lsx_vilvl_d(_r67l, _r45l);
                    _sumfp32_31 = (__m128)__lsx_vilvh_d(_r67l, _r45l);

                    __lsx_vst(_sumfp32_00, outptr, 0);
                    __lsx_vst(_sumfp32_10, outptr + 4, 0);
                    __lsx_vst(_sumfp32_20, outptr + 8, 0);
                    __lsx_vst(_sumfp32_30, outptr + 12, 0);
                    __lsx_vst(_sumfp32_01, outptr + 16, 0);
                    __lsx_vst(_sumfp32_11, outptr + 20, 0);
                    __lsx_vst(_sumfp32_21, outptr + 24, 0);
                    __lsx_vst(_sumfp32_31, outptr + 28, 0);

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

                    __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                    __m128i _sum1 = __lsx_vreplgr2vr_w(0);

                    int i = 0;
                    for (; i < num_input; i++)
                    {
                        __builtin_prefetch(m + 4);
                        __builtin_prefetch(kptr + 32);
                        __m128i _val = __lsx_vreplgr2vr_h((short)m[0]);

                        __m128i _w = __lsx_vld(kptr, 0);
                        __m128i _w16 = __lsx_vilvl_b(__lsx_vslti_b(_w, 0), _w);

                        __m128i _s0 = __lsx_vmul_h(_val, _w16);
                        __m128i _exts0 = __lsx_vslti_h(_s0, 0);
                        __m128i _s0l = __lsx_vilvl_h(_exts0, _s0);
                        __m128i _s0h = __lsx_vilvh_h(_exts0, _s0);

                        _sum0 = __lsx_vadd_w(_sum0, _s0l);
                        _sum1 = __lsx_vadd_w(_sum1, _s0h);

                        m++;
                        kptr += 8;
                    }

                    // dequantize and relu
                    __m128 _scale_in0 = (__m128)__lsx_vld((const float*)scale_in_data + p * 8, 0);
                    __m128 _scale_in1 = (__m128)__lsx_vld((const float*)scale_in_data + p * 8 + 4, 0);

                    __m128 _sumfp32_0 = __lsx_vffint_s_w(_sum0);
                    __m128 _sumfp32_1 = __lsx_vffint_s_w(_sum1);

                    if (bias_term)
                    {
                        __m128 _bias0 = (__m128)__lsx_vld((const float*)bias_data + p * 8, 0);
                        __m128 _bias1 = (__m128)__lsx_vld((const float*)bias_data + p * 8 + 4, 0);
                        _sumfp32_0 = __lsx_vfmadd_s(_scale_in0, _sumfp32_0, _bias0);
                        _sumfp32_1 = __lsx_vfmadd_s(_scale_in1, _sumfp32_1, _bias1);
                    }
                    else
                    {
                        _sumfp32_0 = __lsx_vfmul_s(_sumfp32_0, _scale_in0);
                        _sumfp32_1 = __lsx_vfmul_s(_sumfp32_1, _scale_in1);
                    }

                    _sumfp32_0 = activation_ps(_sumfp32_0, activation_type, activation_params);
                    _sumfp32_1 = activation_ps(_sumfp32_1, activation_type, activation_params);

                    __lsx_vst(_sumfp32_0, outptr, 0);
                    __lsx_vst(_sumfp32_1, outptr + 4, 0);
                    outptr += 8;
                }
            }
        }
#endif // __loongarch_sx

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
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 8 == 0 ? 8 : 1;
    }
#endif // __loongarch_sx
    //     size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(num_output / out_elempack, (size_t)(4u * out_elempack), out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __loongarch_sx
    if (out_elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < num_output / out_elempack; p++)
        {
            __m128i _sum0 = __lsx_vreplgr2vr_w(0);
            __m128i _sum1 = __lsx_vreplgr2vr_w(0);

            const signed char* kptr = weight_data_tm.row<const signed char>(p);
            const signed char* sptr = bottom_blob_int8_flattened;

            int i = 0;
            for (; i < num_input; i++)
            {
                __builtin_prefetch(sptr + 4);
                __builtin_prefetch(kptr + 32);
                __m128i _val = __lsx_vreplgr2vr_h((short)sptr[0]);

                __m128i _w = __lsx_vld(kptr, 0);
                __m128i _w16 = __lsx_vilvl_b(__lsx_vslti_b(_w, 0), _w);

                __m128i _s0 = __lsx_vmul_h(_val, _w16);
                __m128i _exts0 = __lsx_vslti_h(_s0, 0);
                __m128i _s0l = __lsx_vilvl_h(_exts0, _s0);
                __m128i _s0h = __lsx_vilvh_h(_exts0, _s0);

                _sum0 = __lsx_vadd_w(_sum0, _s0l);
                _sum1 = __lsx_vadd_w(_sum1, _s0h);

                sptr += 1;
                kptr += 8;
            }

            // dequantize and relu
            __m128 _scale_in0 = (__m128)__lsx_vld((const float*)scale_in_data + p * 8, 0);
            __m128 _scale_in1 = (__m128)__lsx_vld((const float*)scale_in_data + p * 8 + 4, 0);

            __m128 _sumfp32_0 = __lsx_vffint_s_w(_sum0);
            __m128 _sumfp32_1 = __lsx_vffint_s_w(_sum1);

            if (bias_term)
            {
                __m128 _bias0 = (__m128)__lsx_vld((const float*)bias_data + p * 8, 0);
                __m128 _bias1 = (__m128)__lsx_vld((const float*)bias_data + p * 8 + 4, 0);
                _sumfp32_0 = __lsx_vfmadd_s(_scale_in0, _sumfp32_0, _bias0);
                _sumfp32_1 = __lsx_vfmadd_s(_scale_in1, _sumfp32_1, _bias1);
            }
            else
            {
                _sumfp32_0 = __lsx_vfmul_s(_sumfp32_0, _scale_in0);
                _sumfp32_1 = __lsx_vfmul_s(_sumfp32_1, _scale_in1);
            }

            _sumfp32_0 = activation_ps(_sumfp32_0, activation_type, activation_params);
            _sumfp32_1 = activation_ps(_sumfp32_1, activation_type, activation_params);

            float* outptr = (float*)top_blob + p * 8;
            __lsx_vst(_sumfp32_0, outptr, 0);
            __lsx_vst(_sumfp32_1, outptr + 4, 0);
        }
    }
#endif // __loongarch_sx

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
