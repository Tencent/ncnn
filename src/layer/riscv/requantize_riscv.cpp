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

#include "requantize_riscv.h"

#include "riscv_activation.h"
#include "riscv_usability.h"

namespace ncnn {

Requantize_riscv::Requantize_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int Requantize_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __riscv_vector
    int packn = csrr_vlenb();
    size_t vl = vsetvl_e32m4(packn);
#endif
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

#if __riscv_vector
    if (elempack != packn && elempack != 1)
    {
        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
        return forward(bottom_blob_unpacked, top_blob, opt);
    }

    if (elempack == packn)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;

            top_blob.create(w, (size_t)packn, packn, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_in_data_size == 1 && scale_out_data_size == 1)
            {
                const float scale_in = scale_in_data[0];
                const float scale_out = scale_out_data[0];

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        signed char* ptr = (signed char*)top_blob + i * packn;

                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = activation_ps(vfmul_vf_f32m4(_v, scale_in, vl), activation_type, activation_params, vl);
                        vint8m1_t _out = float2int8(vfmul_vf_f32m4(_v, scale_out, vl), vl);
                        vse8_v_i8m1(ptr, _out, vl);
                    }
                }
                else if (bias_data_size == 1)
                {
                    vfloat32m4_t _bias = vfmv_v_f_f32m4(bias_data[0], vl);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        signed char* ptr = (signed char*)top_blob + i * packn;

                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = activation_ps(vfmacc_vf_f32m4(_v, scale_in, _bias, vl), activation_type, activation_params, vl);
                        vint8m1_t _out = float2int8(vfmul_vf_f32m4(_v, scale_out, vl), vl);
                        vse8_v_i8m1(ptr, _out, vl);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        signed char* ptr = (signed char*)top_blob + i * packn;

                        vfloat32m4_t _bias = vle32_v_f32m4((const float*)bias_data + i * packn, vl);
                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = activation_ps(vfmacc_vf_f32m4(_v, scale_in, _bias, vl), activation_type, activation_params, vl);
                        vint8m1_t _out = float2int8(vfmul_vf_f32m4(_v, scale_out, vl), vl);
                        vse8_v_i8m1(ptr, _out, vl);
                    }
                }
            }
            else if (scale_in_data_size == 1 && scale_out_data_size > 1)
            {
                const float scale_in = scale_in_data[0];

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        signed char* ptr = (signed char*)top_blob + i * packn;

                        vfloat32m4_t _scale_out = vle32_v_f32m4((const float*)scale_out_data + i * packn, vl);

                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = activation_ps(vfmul_vf_f32m4(_v, scale_in, vl), activation_type, activation_params, vl);
                        vint8m1_t _out = float2int8(vfmul_vv_f32m4(_v, _scale_out, vl), vl);
                        vse8_v_i8m1(ptr, _out, vl);
                    }
                }
                else if (bias_data_size == 1)
                {
                    vfloat32m4_t _bias = vfmv_v_f_f32m4(bias_data[0], vl);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        signed char* ptr = (signed char*)top_blob + i * packn;

                        vfloat32m4_t _scale_out = vle32_v_f32m4((const float*)scale_out_data + i * packn, vl);

                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = activation_ps(vfmacc_vf_f32m4(_v, scale_in, _bias, vl), activation_type, activation_params, vl);
                        vint8m1_t _out = float2int8(vfmul_vv_f32m4(_v, _scale_out, vl), vl);
                        vse8_v_i8m1(ptr, _out, vl);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        signed char* ptr = (signed char*)top_blob + i * packn;

                        vfloat32m4_t _scale_out = vle32_v_f32m4((const float*)scale_out_data + i * packn, vl);
                        vfloat32m4_t _bias = vle32_v_f32m4((const float*)bias_data + i * packn, vl);

                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = activation_ps(vfmacc_vf_f32m4(_v, scale_in, _bias, vl), activation_type, activation_params, vl);
                        vint8m1_t _out = float2int8(vfmul_vv_f32m4(_v, _scale_out, vl), vl);
                        vse8_v_i8m1(ptr, _out, vl);
                    }
                }
            }
            else if (scale_in_data_size > 1 && scale_out_data_size == 1)
            {
                const float scale_out = scale_out_data[0];

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        signed char* ptr = (signed char*)top_blob + i * packn;

                        vfloat32m4_t _scale_in = vle32_v_f32m4((const float*)scale_in_data + i * packn, vl);

                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = activation_ps(vfmul_vv_f32m4(_v, _scale_in, vl), activation_type, activation_params, vl);
                        vint8m1_t _out = float2int8(vfmul_vf_f32m4(_v, scale_out, vl), vl);
                        vse8_v_i8m1(ptr, _out, vl);
                    }
                }
                else if (bias_data_size == 1)
                {
                    vfloat32m4_t _bias = vfmv_v_f_f32m4(bias_data[0], vl);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        signed char* ptr = (signed char*)top_blob + i * packn;

                        vfloat32m4_t _scale_in = vle32_v_f32m4((const float*)scale_in_data + i * packn, vl);

                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = activation_ps(vfmacc_vv_f32m4(_v, _scale_in, _bias, vl), activation_type, activation_params, vl);
                        vint8m1_t _out = float2int8(vfmul_vf_f32m4(_v, scale_out, vl), vl);
                        vse8_v_i8m1(ptr, _out, vl);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        signed char* ptr = (signed char*)top_blob + i * packn;

                        vfloat32m4_t _scale_in = vle32_v_f32m4((const float*)scale_in_data + i * packn, vl);
                        vfloat32m4_t _bias = vle32_v_f32m4((const float*)bias_data + i * packn, vl);

                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = activation_ps(vfmacc_vv_f32m4(_v, _scale_in, _bias, vl), activation_type, activation_params, vl);
                        vint8m1_t _out = float2int8(vfmul_vf_f32m4(_v, scale_out, vl), vl);
                        vse8_v_i8m1(ptr, _out, vl);
                    }
                }
            }
            else // if (scale_in_data_size > 1 && scale_out_data_size > 1)
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        signed char* ptr = (signed char*)top_blob + i * packn;

                        vfloat32m4_t _scale_in = vle32_v_f32m4((const float*)scale_in_data + i * packn, vl);
                        vfloat32m4_t _scale_out = vle32_v_f32m4((const float*)scale_out_data + i * packn, vl);

                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = activation_ps(vfmul_vv_f32m4(_v, _scale_in, vl), activation_type, activation_params, vl);
                        vint8m1_t _out = float2int8(vfmul_vv_f32m4(_v, _scale_out, vl), vl);
                        vse8_v_i8m1(ptr, _out, vl);
                    }
                }
                else if (bias_data_size == 1)
                {
                    vfloat32m4_t _bias = vfmv_v_f_f32m4(bias_data[0], vl);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        signed char* ptr = (signed char*)top_blob + i * packn;

                        vfloat32m4_t _scale_in = vle32_v_f32m4((const float*)scale_in_data + i * packn, vl);
                        vfloat32m4_t _scale_out = vle32_v_f32m4((const float*)scale_out_data + i * packn, vl);

                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = activation_ps(vfmacc_vv_f32m4(_v, _scale_in, _bias, vl), activation_type, activation_params, vl);
                        vint8m1_t _out = float2int8(vfmul_vv_f32m4(_v, _scale_out, vl), vl);
                        vse8_v_i8m1(ptr, _out, vl);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        signed char* ptr = (signed char*)top_blob + i * packn;

                        vfloat32m4_t _scale_in = vle32_v_f32m4((const float*)scale_in_data + i * packn, vl);
                        vfloat32m4_t _scale_out = vle32_v_f32m4((const float*)scale_out_data + i * packn, vl);
                        vfloat32m4_t _bias = vle32_v_f32m4((const float*)bias_data + i * packn, vl);

                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = activation_ps(vfmacc_vv_f32m4(_v, _scale_in, _bias, vl), activation_type, activation_params, vl);
                        vint8m1_t _out = float2int8(vfmul_vv_f32m4(_v, _scale_out, vl), vl);
                        vse8_v_i8m1(ptr, _out, vl);
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)packn, packn, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    signed char* ptr = top_blob.row<signed char>(i);

                    vfloat32m4_t scale_in = scale_in_data_size == 1 ? vfmv_v_f_f32m4(scale_in_data[0], vl) : vle32_v_f32m4((const float*)scale_in_data + i * packn, vl);
                    vfloat32m4_t scale_out = scale_out_data_size == 1 ? vfmv_v_f_f32m4(scale_out_data[0], vl) : vle32_v_f32m4((const float*)scale_out_data + i * packn, vl);

                    for (int j = 0; j < w; j++)
                    {
                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = activation_ps(vfmul_vv_f32m4(_v, scale_in, vl), activation_type, activation_params, vl);
                        vint8m1_t _out = float2int8(vfmul_vv_f32m4(_v, scale_out, vl), vl);
                        vse8_v_i8m1(ptr, _out, vl);

                        intptr += packn;
                        ptr += packn;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    signed char* ptr = top_blob.row<signed char>(i);

                    vfloat32m4_t scale_in = scale_in_data_size == 1 ? vfmv_v_f_f32m4(scale_in_data[0], vl) : vle32_v_f32m4((const float*)scale_in_data + i * packn, vl);
                    vfloat32m4_t scale_out = scale_out_data_size == 1 ? vfmv_v_f_f32m4(scale_out_data[0], vl) : vle32_v_f32m4((const float*)scale_out_data + i * packn, vl);
                    vfloat32m4_t bias = bias_data_size == 1 ? vfmv_v_f_f32m4(bias_data[0], vl) : vle32_v_f32m4((const float*)bias_data + i * packn, vl);

                    for (int j = 0; j < w; j++)
                    {
                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = activation_ps(vfmacc_vv_f32m4(_v, scale_in, bias, vl), activation_type, activation_params, vl);
                        vint8m1_t _out = float2int8(vfmul_vv_f32m4(_v, scale_out, vl), vl);
                        vse8_v_i8m1(ptr, _out, vl);

                        intptr += packn;
                        ptr += packn;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;

            top_blob.create(w, h, channels, (size_t)packn, packn, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            // if (activation_type == 1)
            // {
            //     requantize_relu_pack8_rvv(bottom_blob, top_blob, scale_in_data, scale_out_data, bias_data, opt);
            //     return 0;
            // }

            // if (activation_type == 2 && activation_params[0] > 0.f)
            // {
            //     requantize_leakyrelu_pack8_rvv(bottom_blob, top_blob, scale_in_data, scale_out_data, bias_data, activation_params[0], opt);
            //     return 0;
            // }

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    signed char* ptr = top_blob.channel(q);

                    vfloat32m4_t scale_in = scale_in_data_size == 1 ? vfmv_v_f_f32m4(scale_in_data[0], vl) : vle32_v_f32m4((const float*)scale_in_data + q * packn, vl);
                    vfloat32m4_t scale_out = scale_out_data_size == 1 ? vfmv_v_f_f32m4(scale_out_data[0], vl) : vle32_v_f32m4((const float*)scale_out_data + q * packn, vl);

                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = activation_ps(vfmul_vv_f32m4(_v, scale_in, vl), activation_type, activation_params, vl);
                        vint8m1_t _out = float2int8(vfmul_vv_f32m4(_v, scale_out, vl), vl);
                        vse8_v_i8m1(ptr, _out, vl);

                        intptr += packn;
                        ptr += packn;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    signed char* ptr = top_blob.channel(q);

                    vfloat32m4_t scale_in = scale_in_data_size == 1 ? vfmv_v_f_f32m4(scale_in_data[0], vl) : vle32_v_f32m4((const float*)scale_in_data + q * packn, vl);
                    vfloat32m4_t scale_out = scale_out_data_size == 1 ? vfmv_v_f_f32m4(scale_out_data[0], vl) : vle32_v_f32m4((const float*)scale_out_data + q * packn, vl);
                    vfloat32m4_t bias = bias_data_size == 1 ? vfmv_v_f_f32m4(bias_data[0], vl) : vle32_v_f32m4((const float*)bias_data + q * packn, vl);

                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = activation_ps(vfmacc_vv_f32m4(_v, scale_in, bias, vl), activation_type, activation_params, vl);
                        vint8m1_t _out = float2int8(vfmul_vv_f32m4(_v, scale_out, vl), vl);
                        vse8_v_i8m1(ptr, _out, vl);

                        intptr += packn;
                        ptr += packn;
                    }
                }
            }
        }

        return 0;
    }
#endif // __riscv_vector

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int* intptr = bottom_blob;
        signed char* ptr = top_blob;

        if (scale_in_data_size == 1 && scale_out_data_size == 1)
        {
            const float scale_in = scale_in_data[0];
            const float scale_out = scale_out_data[0];

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in;
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in + bias;
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in + bias_data[i];
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
        }
        else if (scale_in_data_size == 1 && scale_out_data_size > 1)
        {
            const float scale_in = scale_in_data[0];

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in;
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data[i]);
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in + bias;
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data[i]);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in + bias_data[i];
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data[i]);
                }
            }
        }
        else if (scale_in_data_size > 1 && scale_out_data_size == 1)
        {
            const float scale_out = scale_out_data[0];

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in_data[i];
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in_data[i] + bias;
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in_data[i] + bias_data[i];
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
        }
        else // if (scale_in_data_size > 1 && scale_out_data_size > 1)
        {
            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in_data[i];
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data[i]);
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in_data[i] + bias;
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data[i]);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float v = intptr[i] * scale_in_data[i] + bias_data[i];
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data[i]);
                }
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                signed char* ptr = top_blob.row<signed char>(i);

                const float scale_in = scale_in_data_size == 1 ? scale_in_data[0] : scale_in_data[i];
                const float scale_out = scale_out_data_size == 1 ? scale_out_data[0] : scale_out_data[i];

                for (int j = 0; j < w; j++)
                {
                    float v = intptr[j] * scale_in;
                    ptr[j] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                signed char* ptr = top_blob.row<signed char>(i);

                const float scale_in = scale_in_data_size == 1 ? scale_in_data[0] : scale_in_data[i];
                const float scale_out = scale_out_data_size == 1 ? scale_out_data[0] : scale_out_data[i];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[i];

                for (int j = 0; j < w; j++)
                {
                    float v = intptr[j] * scale_in + bias;
                    ptr[j] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels, (size_t)1u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                signed char* ptr = top_blob.channel(q);

                const float scale_in = scale_in_data_size == 1 ? scale_in_data[0] : scale_in_data[q];
                const float scale_out = scale_out_data_size == 1 ? scale_out_data[0] : scale_out_data[q];

#if __riscv_vector
                int num_nn = size / (packn * 2);
                int remain_i_start = num_nn * packn * 2;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < num_nn; i++)
                {
                    vfloat32m4_t _p0 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                    vfloat32m4_t _p1 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr + packn, vl), vl);
                    _p0 = activation_ps(vfmul_vf_f32m4(_p0, scale_in, vl), activation_type, activation_params, vl);
                    _p1 = activation_ps(vfmul_vf_f32m4(_p1, scale_in, vl), activation_type, activation_params, vl);
                    vint8m1_t _outp0 = float2int8(vfmul_vf_f32m4(_p0, scale_out, vl), vl);
                    vint8m1_t _outp1 = float2int8(vfmul_vf_f32m4(_p1, scale_out, vl), vl);
                    vse8_v_i8m1(ptr, _outp0, vl);
                    vse8_v_i8m1(ptr + packn, _outp1, vl);
                    ptr += packn * 2;
                    intptr += packn * 2;
                }
#else 
                int remain_i_start = 0;
#endif
                for (int i = remain_i_start; i < size; i++)
                {
                    float v = *intptr * scale_in;
                    *ptr = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    intptr++;
                    ptr++;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                signed char* ptr = top_blob.channel(q);

                const float scale_in = scale_in_data_size == 1 ? scale_in_data[0] : scale_in_data[q];
                const float scale_out = scale_out_data_size == 1 ? scale_out_data[0] : scale_out_data[q];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[q];

                for (int i = 0; i < size; i++)
                {
                    float v = intptr[i] * scale_in + bias;
                    ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
