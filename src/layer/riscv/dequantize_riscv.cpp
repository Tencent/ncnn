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

#include "dequantize_riscv.h"

#include "riscv_activation.h"
#include "riscv_usability.h"

namespace ncnn {

Dequantize_riscv::Dequantize_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif
}

int Dequantize_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __riscv_vector
    int packn = csrr_vlenb() / 4;
    int in_packn = packn * 4;
    size_t vl = vsetvl_e32m4(packn);
#endif
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

#if __riscv_vector
    if (elempack != in_packn && elempack != 1)
    {
        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
        return forward(bottom_blob_unpacked, top_blob, opt);
    }

    if (elempack == in_packn)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int outw = w * 4;

            top_blob.create(outw, (size_t)4u * packn, packn, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                const float scale = scale_data[0];

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        float* ptr = (float*)top_blob + i * packn;

                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = vfmul_vf_f32m4(_v, scale, vl);
                        vse32_v_f32m4(ptr, _v, vl);
                    }
                }
                else if (bias_data_size == 1)
                {
                    vfloat32m4_t _bias = vfmv_v_f_f32m4(bias_data[0], vl);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        float* ptr = (float*)top_blob + i * packn;

                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = vfmacc_vf_f32m4(_v, scale, _bias, vl);
                        vse32_v_f32m4(ptr, _v, vl);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        float* ptr = (float*)top_blob + i * packn;

                        vfloat32m4_t _bias = vle32_v_f32m4((const float*)bias_data + i * packn, vl);
                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = vfmacc_vf_f32m4(_v, scale, _bias, vl);
                        vse32_v_f32m4(ptr, _v, vl);
                    }
                }
            }
            else
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        float* ptr = (float*)top_blob + i * packn;

                        vfloat32m4_t _scale = vle32_v_f32m4((const float*)scale_data + i * packn, vl);
                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = vfmul_vv_f32m4(_v, _scale, vl);
                        vse32_v_f32m4(ptr, _v, vl);
                    }
                }
                else if (bias_data_size == 1)
                {
                    vfloat32m4_t _bias = vfmv_v_f_f32m4(bias_data[0], vl);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        float* ptr = (float*)top_blob + i * packn;

                        vfloat32m4_t _scale = vle32_v_f32m4((const float*)scale_data + i * packn, vl);
                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = vfmacc_vv_f32m4(_v, _scale, _bias, vl);       
                        vse32_v_f32m4(ptr, _v, vl);                 
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * packn;
                        float* ptr = (float*)top_blob + i * packn;

                        vfloat32m4_t _scale = vle32_v_f32m4((const float*)scale_data + i * packn, vl);
                        vfloat32m4_t _bias = vle32_v_f32m4((const float*)bias_data + i * packn, vl);
                        vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        _v = vfmacc_vv_f32m4(_v, _scale, _bias, vl);
                        vse32_v_f32m4(ptr, _v, vl);
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int outh = h * 4;

            top_blob.create(w, outh, (size_t)4u * packn, packn, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr0 = top_blob.row(i * 4);
                    float* ptr1 = top_blob.row(i * 4 + 1);
                    float* ptr2 = top_blob.row(i * 4 + 2);
                    float* ptr3 = top_blob.row(i * 4 + 3);

                    vfloat32m4_t _scale0 = scale_data_size == 1 ? vfmv_v_f_f32m4(scale_data[0], vl) : vle32_v_f32m4((const float*)scale_data + (i * 4) * packn, vl);
                    vfloat32m4_t _scale1 = scale_data_size == 1 ? vfmv_v_f_f32m4(scale_data[0], vl) : vle32_v_f32m4((const float*)scale_data + (i * 4 + 1) * packn, vl);
                    vfloat32m4_t _scale2 = scale_data_size == 1 ? vfmv_v_f_f32m4(scale_data[0], vl) : vle32_v_f32m4((const float*)scale_data + (i * 4 + 2) * packn, vl);
                    vfloat32m4_t _scale3 = scale_data_size == 1 ? vfmv_v_f_f32m4(scale_data[0], vl) : vle32_v_f32m4((const float*)scale_data + (i * 4 + 3) * packn, vl);

                    for (int j = 0; j < w; j++)
                    {
                        vfloat32m4_t _v0 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        vfloat32m4_t _v1 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr + packn, vl), vl);
                        vfloat32m4_t _v2 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr + packn * 2, vl), vl);
                        vfloat32m4_t _v3 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr + packn * 3, vl), vl);
                        _v0 = vfmul_vv_f32m4(_v0, _scale0, vl);
                        _v1 = vfmul_vv_f32m4(_v1, _scale1, vl);
                        _v2 = vfmul_vv_f32m4(_v2, _scale2, vl);
                        _v3 = vfmul_vv_f32m4(_v3, _scale3, vl);
                        vse32_v_f32m4(ptr0, _v0, vl);
                        vse32_v_f32m4(ptr1, _v1, vl);
                        vse32_v_f32m4(ptr2, _v2, vl);
                        vse32_v_f32m4(ptr3, _v3, vl);

                        intptr += in_packn;
                        ptr0 += packn;
                        ptr1 += packn;
                        ptr2 += packn;
                        ptr3 += packn;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr0 = top_blob.row(i * 4);
                    float* ptr1 = top_blob.row(i * 4 + 1);
                    float* ptr2 = top_blob.row(i * 4 + 2);
                    float* ptr3 = top_blob.row(i * 4 + 3);

                    vfloat32m4_t _scale0 = scale_data_size == 1 ? vfmv_v_f_f32m4(scale_data[0], vl) : vle32_v_f32m4((const float*)scale_data + (i * 4) * packn, vl);
                    vfloat32m4_t _scale1 = scale_data_size == 1 ? vfmv_v_f_f32m4(scale_data[0], vl) : vle32_v_f32m4((const float*)scale_data + (i * 4 + 1) * packn, vl);
                    vfloat32m4_t _scale2 = scale_data_size == 1 ? vfmv_v_f_f32m4(scale_data[0], vl) : vle32_v_f32m4((const float*)scale_data + (i * 4 + 2) * packn, vl);
                    vfloat32m4_t _scale3 = scale_data_size == 1 ? vfmv_v_f_f32m4(scale_data[0], vl) : vle32_v_f32m4((const float*)scale_data + (i * 4 + 3) * packn, vl);
                    vfloat32m4_t _bias0 = bias_data_size == 1 ? vfmv_v_f_f32m4(bias_data[0], vl) : vle32_v_f32m4((const float*)bias_data + (i * 4) * packn, vl);
                    vfloat32m4_t _bias1 = bias_data_size == 1 ? vfmv_v_f_f32m4(bias_data[0], vl) : vle32_v_f32m4((const float*)bias_data + (i * 4 + 1) * packn, vl);
                    vfloat32m4_t _bias2 = bias_data_size == 1 ? vfmv_v_f_f32m4(bias_data[0], vl) : vle32_v_f32m4((const float*)bias_data + (i * 4 + 2) * packn, vl);
                    vfloat32m4_t _bias3 = bias_data_size == 1 ? vfmv_v_f_f32m4(bias_data[0], vl) : vle32_v_f32m4((const float*)bias_data + (i * 4 + 3) * packn, vl);

                    for (int j = 0; j < w; j++)
                    {
                        vfloat32m4_t _v0 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        vfloat32m4_t _v1 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr + packn, vl), vl);
                        vfloat32m4_t _v2 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr + packn * 2, vl), vl);
                        vfloat32m4_t _v3 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr + packn * 3, vl), vl);
                        _v0 = vfmacc_vv_f32m4(_v0, _scale0, _bias0, vl);
                        _v1 = vfmacc_vv_f32m4(_v1, _scale1, _bias1, vl);
                        _v2 = vfmacc_vv_f32m4(_v2, _scale2, _bias2, vl);
                        _v3 = vfmacc_vv_f32m4(_v3, _scale3, _bias3, vl);
                        vse32_v_f32m4(ptr0, _v0, vl);
                        vse32_v_f32m4(ptr1, _v1, vl);
                        vse32_v_f32m4(ptr2, _v2, vl);
                        vse32_v_f32m4(ptr3, _v3, vl);

                        intptr += in_packn;
                        ptr0 += packn;
                        ptr1 += packn;
                        ptr2 += packn;
                        ptr3 += packn;
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
            int outc = channels * 4;

            top_blob.create(w, h, outc, (size_t)4u * packn, packn, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr0 = top_blob.channel(q * 4);
                    float* ptr1 = top_blob.channel(q * 4 + 1);
                    float* ptr2 = top_blob.channel(q * 4 + 2);
                    float* ptr3 = top_blob.channel(q * 4 + 3);

                    vfloat32m4_t _scale0 = scale_data_size == 1 ? vfmv_v_f_f32m4(scale_data[0], vl) : vle32_v_f32m4((const float*)scale_data + (q * 4) * packn, vl);
                    vfloat32m4_t _scale1 = scale_data_size == 1 ? vfmv_v_f_f32m4(scale_data[0], vl) : vle32_v_f32m4((const float*)scale_data + (q * 4 + 1) * packn, vl);
                    vfloat32m4_t _scale2 = scale_data_size == 1 ? vfmv_v_f_f32m4(scale_data[0], vl) : vle32_v_f32m4((const float*)scale_data + (q * 4 + 2) * packn, vl);
                    vfloat32m4_t _scale3 = scale_data_size == 1 ? vfmv_v_f_f32m4(scale_data[0], vl) : vle32_v_f32m4((const float*)scale_data + (q * 4 + 3) * packn, vl);

                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m4_t _v0 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        vfloat32m4_t _v1 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr + packn, vl), vl);
                        vfloat32m4_t _v2 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr + packn * 2, vl), vl);
                        vfloat32m4_t _v3 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr + packn * 3, vl), vl);
                        _v0 = vfmul_vv_f32m4(_v0, _scale0, vl);
                        _v1 = vfmul_vv_f32m4(_v1, _scale1, vl);
                        _v2 = vfmul_vv_f32m4(_v2, _scale2, vl);
                        _v3 = vfmul_vv_f32m4(_v3, _scale3, vl);
                        vse32_v_f32m4(ptr0, _v0, vl);
                        vse32_v_f32m4(ptr1, _v1, vl);
                        vse32_v_f32m4(ptr2, _v2, vl);
                        vse32_v_f32m4(ptr3, _v3, vl);

                        intptr += in_packn;
                        ptr0 += packn;
                        ptr1 += packn;
                        ptr2 += packn;
                        ptr3 += packn;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr0 = top_blob.channel(q * 4);
                    float* ptr1 = top_blob.channel(q * 4 + 1);
                    float* ptr2 = top_blob.channel(q * 4 + 2);
                    float* ptr3 = top_blob.channel(q * 4 + 3);

                    vfloat32m4_t _scale0 = scale_data_size == 1 ? vfmv_v_f_f32m4(scale_data[0], vl) : vle32_v_f32m4((const float*)scale_data + (q * 4) * packn, vl);
                    vfloat32m4_t _scale1 = scale_data_size == 1 ? vfmv_v_f_f32m4(scale_data[0], vl) : vle32_v_f32m4((const float*)scale_data + (q * 4 + 1) * packn, vl);
                    vfloat32m4_t _scale2 = scale_data_size == 1 ? vfmv_v_f_f32m4(scale_data[0], vl) : vle32_v_f32m4((const float*)scale_data + (q * 4 + 2) * packn, vl);
                    vfloat32m4_t _scale3 = scale_data_size == 1 ? vfmv_v_f_f32m4(scale_data[0], vl) : vle32_v_f32m4((const float*)scale_data + (q * 4 + 3) * packn, vl);
                    vfloat32m4_t _bias0 = bias_data_size == 1 ? vfmv_v_f_f32m4(bias_data[0], vl) : vle32_v_f32m4((const float*)bias_data + (q * 4) * packn, vl);
                    vfloat32m4_t _bias1 = bias_data_size == 1 ? vfmv_v_f_f32m4(bias_data[0], vl) : vle32_v_f32m4((const float*)bias_data + (q * 4 + 1) * packn, vl);
                    vfloat32m4_t _bias2 = bias_data_size == 1 ? vfmv_v_f_f32m4(bias_data[0], vl) : vle32_v_f32m4((const float*)bias_data + (q * 4 + 2) * packn, vl);
                    vfloat32m4_t _bias3 = bias_data_size == 1 ? vfmv_v_f_f32m4(bias_data[0], vl) : vle32_v_f32m4((const float*)bias_data + (q * 4 + 3) * packn, vl);

                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m4_t _v0 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                        vfloat32m4_t _v1 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr + packn, vl), vl);
                        vfloat32m4_t _v2 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr + packn * 2, vl), vl);
                        vfloat32m4_t _v3 = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr + packn * 3, vl), vl);
                        _v0 = vfmacc_vv_f32m4(_v0, _scale0, _bias0, vl);
                        _v1 = vfmacc_vv_f32m4(_v1, _scale1, _bias1, vl);
                        _v2 = vfmacc_vv_f32m4(_v2, _scale2, _bias2, vl);
                        _v3 = vfmacc_vv_f32m4(_v3, _scale3, _bias3, vl);
                        vse32_v_f32m4(ptr0, _v0, vl);
                        vse32_v_f32m4(ptr1, _v1, vl);
                        vse32_v_f32m4(ptr2, _v2, vl);
                        vse32_v_f32m4(ptr3, _v3, vl);

                        intptr += in_packn;
                        ptr0 += packn;
                        ptr1 += packn;
                        ptr2 += packn;
                        ptr3 += packn;
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

        top_blob.create(w, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int* intptr = bottom_blob;
        float* ptr = top_blob;

        if (scale_data_size == 1)
        {
            const float scale = scale_data[0];

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale;
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale + bias;
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale + bias_data[i];
                }
            }
        }
        else
        {
            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i];
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i] + bias;
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i] + bias_data[i];
                }
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                float* ptr = top_blob.row(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

                int j = 0;
#if __riscv_vector
                for (; j + packn < w; j += packn)
                {
                    vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                    _v = vfmul_vf_f32m4(_v, scale, vl);
                    vse32_v_f32m4(ptr, _v, vl);

                    intptr += packn;
                    ptr += packn;
                }
#endif // __riscv_vector
                for (; j < w; j++)
                {
                    *ptr++ = *intptr++ * scale;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                float* ptr = top_blob.row(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[i];

                int j = 0;
#if __riscv_vector
                vfloat32m4_t _bias = vfmv_v_f_f32m4(bias, vl);
                for (; j + packn < w; j += packn)
                {
                    vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                    _v = vfmacc_vf_f32m4(_bias, scale, _v, vl);
                    vse32_v_f32m4(ptr, _v, vl);

                    intptr += packn;
                    ptr += packn;
                }
#endif // __riscv_vector
                for (; j < w; j++)
                {
                    *ptr++ = *intptr++ * scale + bias;
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

        top_blob.create(w, h, channels, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                float* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

                int i = 0;
#if __riscv_vector
                for (; i + packn < size; i += packn)
                {
                    vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                    _v = vfmul_vf_f32m4(_v, scale, vl);
                    vse32_v_f32m4(ptr, _v, vl);

                    intptr += packn;
                    ptr += packn;
                }
#endif // __riscv_vector
                for (; i < size; i++)
                {
                    *ptr++ = *intptr++ * scale;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                float* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[q];

                int i = 0;
#if __riscv_vector
                vfloat32m4_t _bias = vfmv_v_f_f32m4(bias, vl);
                for (; i + packn < size; i += packn)
                {
                    vfloat32m4_t _v = vfcvt_f_x_v_f32m4(vle32_v_i32m4(intptr, vl), vl);
                    _v = vfmacc_vf_f32m4(_bias, scale, _v, vl);
                    vse32_v_f32m4(ptr, _v, vl);

                    intptr += packn;
                    ptr += packn;
                }
#endif // __riscv_vector
                for (; i < size; i++)
                {
                    *ptr++ = *intptr++ * scale + bias;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
