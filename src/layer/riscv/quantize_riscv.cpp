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

#include "quantize_riscv.h"

#include "riscv_usability.h"

namespace ncnn {

Quantize_riscv::Quantize_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
}

int Quantize_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __riscv_vector
    int packn = csrr_vlenb() / 4;
    int out_packn = packn * 4;
    size_t vl = vsetvl_e32m4(packn);
#endif
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

#if __riscv_vector
    if (elempack == packn)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int out_elempack = opt.use_packing_layout && w * elempack % out_packn == 0 ? out_packn : 1;
            int outw = w * elempack / out_elempack;

            top_blob.create(outw, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                const float scale = scale_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const float* ptr0 = (const float*)bottom_blob + i * 4;
                    signed char* outptr = (signed char*)top_blob + i * 4;

                    outptr[0] = float2int8(ptr0[0] * scale);
                    outptr[1] = float2int8(ptr0[1] * scale);
                    outptr[2] = float2int8(ptr0[2] * scale);
                    outptr[3] = float2int8(ptr0[3] * scale);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    const float* ptr0 = (const float*)bottom_blob + i * 4;
                    signed char* outptr = (signed char*)top_blob + i * 4;

                    outptr[0] = float2int8(ptr0[0] * scale_data[i * 4]);
                    outptr[1] = float2int8(ptr0[1] * scale_data[i * 4 + 1]);
                    outptr[2] = float2int8(ptr0[2] * scale_data[i * 4 + 2]);
                    outptr[3] = float2int8(ptr0[3] * scale_data[i * 4 + 3]);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int out_elempack = opt.use_packing_layout && h * elempack % out_packn == 0 ? out_packn : 1;
            int outh = h * elempack / out_elempack;

            top_blob.create(w, outh, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (out_elempack == out_packn)
            {
                if (scale_data_size == 1)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outh; i++)
                    {
                        const float* ptr0 = bottom_blob.row(i * 4);
                        const float* ptr1 = bottom_blob.row(i * 4 + 1);
                        const float* ptr2 = bottom_blob.row(i * 4 + 2);
                        const float* ptr3 = bottom_blob.row(i * 4 + 3);
                        signed char* outptr = top_blob.row<signed char>(i);

                        for (int j = 0; j < w; j++)
                        {
                            vfloat32m4_t _ptr0 = vle32_v_f32m4(ptr0, vl);
                            vfloat32m4_t _ptr1 = vle32_v_f32m4(ptr1, vl);
                            vfloat32m4_t _ptr2 = vle32_v_f32m4(ptr2, vl);
                            vfloat32m4_t _ptr3 = vle32_v_f32m4(ptr3, vl);
                            _ptr0 = vfmul_vf_f32m4(_ptr0, scale_data[0], vl);
                            _ptr1 = vfmul_vf_f32m4(_ptr1, scale_data[0], vl);
                            _ptr2 = vfmul_vf_f32m4(_ptr2, scale_data[0], vl);
                            _ptr3 = vfmul_vf_f32m4(_ptr3, scale_data[0], vl);
                            vint8m1_t out0 = float2int8(_ptr0, vl);
                            vint8m1_t out1 = float2int8(_ptr1, vl);
                            vint8m1_t out2 = float2int8(_ptr2, vl);
                            vint8m1_t out3 = float2int8(_ptr3, vl);
                            vse8_v_i8m1(outptr, out0, vl);
                            vse8_v_i8m1(outptr + packn, out1, vl);
                            vse8_v_i8m1(outptr + 2 * packn, out2, vl);
                            vse8_v_i8m1(outptr + 3 * packn, out3, vl);

                            ptr0 += packn;
                            ptr1 += packn;
                            ptr2 += packn;
                            ptr3 += packn;
                            outptr += out_packn;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outh; i++)
                    {
                        const float* ptr0 = bottom_blob.row(i * 4);
                        const float* ptr1 = bottom_blob.row(i * 4 + 1);
                        const float* ptr2 = bottom_blob.row(i * 4 + 2);
                        const float* ptr3 = bottom_blob.row(i * 4 + 3);
                        signed char* outptr = top_blob.row<signed char>(i);
                        vfloat32m4_t _scale0 = vle32_v_f32m4((const float*)scale_data + 4 * i * packn, vl);
                        vfloat32m4_t _scale1 = vle32_v_f32m4((const float*)scale_data + (4 * i + 1) * packn, vl);
                        vfloat32m4_t _scale2 = vle32_v_f32m4((const float*)scale_data + (4 * i + 2) * packn, vl);
                        vfloat32m4_t _scale3 = vle32_v_f32m4((const float*)scale_data + (4 * i + 3) * packn, vl);

                        for (int j = 0; j < w; j++)
                        {
                            vfloat32m4_t _ptr0 = vle32_v_f32m4(ptr0, vl);
                            vfloat32m4_t _ptr1 = vle32_v_f32m4(ptr1, vl);
                            vfloat32m4_t _ptr2 = vle32_v_f32m4(ptr2, vl);
                            vfloat32m4_t _ptr3 = vle32_v_f32m4(ptr3, vl);
                            _ptr0 = vfmul_vv_f32m4(_ptr0, _scale0, vl);
                            _ptr1 = vfmul_vv_f32m4(_ptr1, _scale1, vl);
                            _ptr2 = vfmul_vv_f32m4(_ptr2, _scale2, vl);
                            _ptr3 = vfmul_vv_f32m4(_ptr3, _scale3, vl);
                            vint8m1_t out0 = float2int8(_ptr0, vl);
                            vint8m1_t out1 = float2int8(_ptr1, vl);
                            vint8m1_t out2 = float2int8(_ptr2, vl);
                            vint8m1_t out3 = float2int8(_ptr3, vl);
                            vse8_v_i8m1(outptr, out0, vl);
                            vse8_v_i8m1(outptr + packn, out1, vl);
                            vse8_v_i8m1(outptr + 2 * packn, out2, vl);
                            vse8_v_i8m1(outptr + 3 * packn, out3, vl);

                            ptr0 += packn;
                            ptr1 += packn;
                            ptr2 += packn;
                            ptr3 += packn;
                            outptr += out_packn;
                        }
                    }
                }
            }
            if (out_elempack == 1)
            {
                if (scale_data_size == 1)
                {
                    const float scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < h; i++)
                    {
                        const float* ptr0 = bottom_blob.row(i);
                        signed char* outptr0 = top_blob.row<signed char>(i * packn);

                        for (int j = 0; j < w; j++)
                        {
                            vfloat32m4_t _ptr0 = vle32_v_f32m4(ptr0, vl);
                            _ptr0 = vfmul_vf_f32m4(_ptr0, scale, vl);
                            vint8m1_t out0 = float2int8(_ptr0, vl);
                            vsse8_v_i8m1(outptr0, top_blob.w * sizeof(int8_t), out0, vl);

                            ptr0 += packn;
                            outptr0 += 1;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < h; i++)
                    {
                        const float* ptr0 = bottom_blob.row(i);
                        signed char* outptr0 = top_blob.row<signed char>(i * packn);

                        vfloat32m4_t _scale = vle32_v_f32m4((const float*)scale_data + i * packn, vl);

                        for (int j = 0; j < w; j++)
                        {
                            vfloat32m4_t _ptr0 = vle32_v_f32m4(ptr0, vl);
                            _ptr0 = vfmul_vv_f32m4(_ptr0, _scale, vl);
                            vint8m1_t out0 = float2int8(_ptr0, vl);
                            vsse8_v_i8m1(outptr0, top_blob.w * sizeof(int8_t), out0, vl);

                            ptr0 += packn;
                            outptr0 += 1;
                        }
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
            int out_elempack = opt.use_packing_layout && channels * elempack % out_packn == 0 ? out_packn : 1;
            int outc = channels * elempack / out_elempack;
            NCNN_LOGE("out_elempack:%d", out_elempack);
            NCNN_LOGE("outc:%d", outc);

            top_blob.create(w, h, outc, (size_t)out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (out_elempack == out_packn)
            {
                if (scale_data_size == 1)
                {
                    const float scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < outc; q++)
                    {
                        const float* ptr0 = bottom_blob.channel(q * 4);
                        const float* ptr1 = bottom_blob.channel(q * 4 + 1);
                        const float* ptr2 = bottom_blob.channel(q * 4 + 2);
                        const float* ptr3 = bottom_blob.channel(q * 4 + 3);
                        signed char* outptr = top_blob.channel(q);

                        for (int i = 0; i < size; i++)
                        {
                            vfloat32m4_t _ptr0 = vle32_v_f32m4(ptr0, vl);
                            vfloat32m4_t _ptr1 = vle32_v_f32m4(ptr1, vl);
                            vfloat32m4_t _ptr2 = vle32_v_f32m4(ptr2, vl);
                            vfloat32m4_t _ptr3 = vle32_v_f32m4(ptr3, vl);
                            _ptr0 = vfmul_vf_f32m4(_ptr0, scale, vl);
                            _ptr1 = vfmul_vf_f32m4(_ptr1, scale, vl);
                            _ptr2 = vfmul_vf_f32m4(_ptr2, scale, vl);
                            _ptr3 = vfmul_vf_f32m4(_ptr3, scale, vl);
                            vint8m1_t out0 = float2int8(_ptr0, vl);
                            vint8m1_t out1 = float2int8(_ptr1, vl);
                            vint8m1_t out2 = float2int8(_ptr2, vl);
                            vint8m1_t out3 = float2int8(_ptr3, vl);
                            vse8_v_i8m1(outptr, out0, vl);
                            vse8_v_i8m1(outptr + packn, out1, vl);
                            vse8_v_i8m1(outptr + 2 * packn, out2, vl);
                            vse8_v_i8m1(outptr + 3 * packn, out3, vl);

                            ptr0 += packn;
                            ptr1 += packn;
                            ptr2 += packn;
                            ptr3 += packn;
                            outptr += out_packn;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < outc; q++)
                    {
                        const float* ptr0 = bottom_blob.channel(q * 4);
                        const float* ptr1 = bottom_blob.channel(q * 4 + 1);
                        const float* ptr2 = bottom_blob.channel(q * 4 + 2);
                        const float* ptr3 = bottom_blob.channel(q * 4 + 3);
                        signed char* outptr = top_blob.channel(q);

                        vfloat32m4_t _scale0 = vle32_v_f32m4((const float*)scale_data + q * 4 * packn, vl);
                        vfloat32m4_t _scale1 = vle32_v_f32m4((const float*)scale_data + (q * 4 + 1) * packn, vl);
                        vfloat32m4_t _scale2 = vle32_v_f32m4((const float*)scale_data + (q * 4 + 2) * packn, vl);
                        vfloat32m4_t _scale3 = vle32_v_f32m4((const float*)scale_data + (q * 4 + 3) * packn, vl);

                        int i = 0;
                        for (; i < size; i++)
                        {
                            vfloat32m4_t _ptr0 = vle32_v_f32m4(ptr0, vl);
                            vfloat32m4_t _ptr1 = vle32_v_f32m4(ptr1, vl);
                            vfloat32m4_t _ptr2 = vle32_v_f32m4(ptr2, vl);
                            vfloat32m4_t _ptr3 = vle32_v_f32m4(ptr3, vl);
                            _ptr0 = vfmul_vv_f32m4(_ptr0, _scale0, vl);
                            _ptr1 = vfmul_vv_f32m4(_ptr1, _scale1, vl);
                            _ptr2 = vfmul_vv_f32m4(_ptr2, _scale2, vl);
                            _ptr3 = vfmul_vv_f32m4(_ptr3, _scale3, vl);
                            vint8m1_t out0 = float2int8(_ptr0, vl);
                            vint8m1_t out1 = float2int8(_ptr1, vl);
                            vint8m1_t out2 = float2int8(_ptr2, vl);
                            vint8m1_t out3 = float2int8(_ptr3, vl);
                            vse8_v_i8m1(outptr, out0, vl);
                            vse8_v_i8m1(outptr + packn, out1, vl);
                            vse8_v_i8m1(outptr + 2 * packn, out2, vl);
                            vse8_v_i8m1(outptr + 3 * packn, out3, vl);

                            ptr0 += packn;
                            ptr1 += packn;
                            ptr2 += packn;
                            ptr3 += packn;
                            outptr += out_packn;
                        }
                    }
                }
            }
            if (out_elempack == 1)
            {
                if (scale_data_size == 1)
                {
                    const float scale = scale_data[0];

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr0 = bottom_blob.channel(q);
                        signed char* outptr0 = top_blob.channel(q * packn);

                        for (int i = 0; i < size; i++)
                        {
                            vfloat32m4_t _ptr0 = vle32_v_f32m4(ptr0, vl);
                            _ptr0 = vfmul_vf_f32m4(_ptr0, scale, vl);
                            vint8m1_t out0 = float2int8(_ptr0, vl);
                            vsse8_v_i8m1(outptr0, top_blob.cstep * sizeof(int8_t), out0, vl);

                            ptr0 += packn;
                            outptr0 += 1;
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr0 = bottom_blob.channel(q);
                        signed char* outptr0 = top_blob.channel(q * packn);

                        vfloat32m4_t _scale = vle32_v_f32m4((const float*)scale_data + q * packn, vl);

                        for (int i = 0; i < size; i++)
                        {
                            vfloat32m4_t _ptr0 = vle32_v_f32m4(ptr0, vl);
                            _ptr0 = vfmul_vv_f32m4(_ptr0, _scale, vl);
                            vint8m1_t out0 = float2int8(_ptr0, vl);
                            vsse8_v_i8m1(outptr0, top_blob.cstep * sizeof(int8_t), out0, vl);

                            ptr0 += packn;
                            outptr0 += 1;
                        }
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

        const float* ptr = bottom_blob;
        signed char* outptr = top_blob;

        if (scale_data_size == 1)
        {
            const float scale = scale_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                
                *outptr++ = float2int8(*ptr++ * scale);
            }
        }
        else
        {
            const float* scaleptr = scale_data;
#if __riscv_vector
            int num_nn = w / (packn * 8);
            int remain_w_start = num_nn * packn * 8;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < num_nn; i++)
            {
                vfloat32m4_t _p0 = vle32_v_f32m4(ptr, vl);
                vfloat32m4_t _p1 = vle32_v_f32m4(ptr + packn, vl);
                vfloat32m4_t _p2 = vle32_v_f32m4(ptr + 2 * packn, vl);
                vfloat32m4_t _p3 = vle32_v_f32m4(ptr + 3 * packn, vl);
                vfloat32m4_t _p4 = vle32_v_f32m4(ptr + 4 * packn, vl);
                vfloat32m4_t _p5 = vle32_v_f32m4(ptr + 5 * packn, vl);
                vfloat32m4_t _p6 = vle32_v_f32m4(ptr + 6 * packn, vl);
                vfloat32m4_t _p7 = vle32_v_f32m4(ptr + 7 * packn, vl);
                vfloat32m4_t _scale0 = vle32_v_f32m4(scaleptr, vl);
                vfloat32m4_t _scale1 = vle32_v_f32m4(scaleptr + packn, vl);
                vfloat32m4_t _scale2 = vle32_v_f32m4(scaleptr + 2 * packn, vl);
                vfloat32m4_t _scale3 = vle32_v_f32m4(scaleptr + 3 * packn, vl);
                vfloat32m4_t _scale4 = vle32_v_f32m4(scaleptr + 4 * packn, vl);
                vfloat32m4_t _scale5 = vle32_v_f32m4(scaleptr + 5 * packn, vl);
                vfloat32m4_t _scale6 = vle32_v_f32m4(scaleptr + 6 * packn, vl);
                vfloat32m4_t _scale7 = vle32_v_f32m4(scaleptr + 7 * packn, vl);
                _p0 = vfmul_vv_f32m4(_p0, _scale0, vl);
                _p1 = vfmul_vv_f32m4(_p1, _scale1, vl);
                _p2 = vfmul_vv_f32m4(_p2, _scale2, vl);
                _p3 = vfmul_vv_f32m4(_p3, _scale3, vl);
                _p4 = vfmul_vv_f32m4(_p4, _scale4, vl);
                _p5 = vfmul_vv_f32m4(_p5, _scale5, vl);
                _p6 = vfmul_vv_f32m4(_p6, _scale6, vl);
                _p7 = vfmul_vv_f32m4(_p7, _scale7, vl);
                vint8m1_t _outp0 = float2int8(_p0, vl);
                vint8m1_t _outp1 = float2int8(_p1, vl);
                vint8m1_t _outp2 = float2int8(_p2, vl);
                vint8m1_t _outp3 = float2int8(_p3, vl);
                vint8m1_t _outp4 = float2int8(_p4, vl);
                vint8m1_t _outp5 = float2int8(_p5, vl);
                vint8m1_t _outp6 = float2int8(_p6, vl);
                vint8m1_t _outp7 = float2int8(_p7, vl);
                vse8_v_i8m1(outptr, _outp0, vl);
                vse8_v_i8m1(outptr + packn, _outp1, vl);
                vse8_v_i8m1(outptr + 2 * packn, _outp2, vl);
                vse8_v_i8m1(outptr + 3 * packn, _outp3, vl);
                vse8_v_i8m1(outptr + 4 * packn, _outp4, vl);
                vse8_v_i8m1(outptr + 5 * packn, _outp5, vl);
                vse8_v_i8m1(outptr + 6 * packn, _outp6, vl);
                vse8_v_i8m1(outptr + 7 * packn, _outp7, vl);
                ptr += 8 * packn;
                outptr += 8 * packn;
                scaleptr += 8 * packn;
            }

            num_nn = (w - remain_w_start) / (packn * 4);
            remain_w_start += num_nn * packn * 4;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < num_nn; i++)
            {
                vfloat32m4_t _p0 = vle32_v_f32m4(ptr, vl);
                vfloat32m4_t _p1 = vle32_v_f32m4(ptr + packn, vl);
                vfloat32m4_t _p2 = vle32_v_f32m4(ptr + 2 * packn, vl);
                vfloat32m4_t _p3 = vle32_v_f32m4(ptr + 3 * packn, vl);
                vfloat32m4_t _scale0 = vle32_v_f32m4(scaleptr, vl);
                vfloat32m4_t _scale1 = vle32_v_f32m4(scaleptr + packn, vl);
                vfloat32m4_t _scale2 = vle32_v_f32m4(scaleptr + 2 * packn, vl);
                vfloat32m4_t _scale3 = vle32_v_f32m4(scaleptr + 3 * packn, vl);
                _p0 = vfmul_vv_f32m4(_p0, _scale0, vl);
                _p1 = vfmul_vv_f32m4(_p1, _scale1, vl);
                _p2 = vfmul_vv_f32m4(_p2, _scale2, vl);
                _p3 = vfmul_vv_f32m4(_p3, _scale3, vl);
                vint8m1_t _outp0 = float2int8(_p0, vl);
                vint8m1_t _outp1 = float2int8(_p1, vl);
                vint8m1_t _outp2 = float2int8(_p2, vl);
                vint8m1_t _outp3 = float2int8(_p3, vl);
                vse8_v_i8m1(outptr, _outp0, vl);
                vse8_v_i8m1(outptr + packn, _outp1, vl);
                vse8_v_i8m1(outptr + 2 * packn, _outp2, vl);
                vse8_v_i8m1(outptr + 3 * packn, _outp3, vl);
                ptr += 4 * packn;
                outptr += 4 * packn;
                scaleptr += 4 * packn;
            }
#else
            int remain_w_start = 0;
#endif
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = remain_w_start; i < w; i++)
            {
                *outptr++ = float2int8(*ptr++ * *scaleptr++);
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

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const float* ptr = bottom_blob.row(i);
            signed char* outptr = top_blob.row<signed char>(i);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

#if __riscv_vector
            int num_nn = w / (packn * 8);
            int remain_w_start = num_nn * packn * 8;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < num_nn; i++)
            {
                vfloat32m4_t _p0 = vle32_v_f32m4(ptr, vl);
                vfloat32m4_t _p1 = vle32_v_f32m4(ptr + packn, vl);
                vfloat32m4_t _p2 = vle32_v_f32m4(ptr + 2 * packn, vl);
                vfloat32m4_t _p3 = vle32_v_f32m4(ptr + 3 * packn, vl);
                vfloat32m4_t _p4 = vle32_v_f32m4(ptr + 4 * packn, vl);
                vfloat32m4_t _p5 = vle32_v_f32m4(ptr + 5 * packn, vl);
                vfloat32m4_t _p6 = vle32_v_f32m4(ptr + 6 * packn, vl);
                vfloat32m4_t _p7 = vle32_v_f32m4(ptr + 7 * packn, vl);
                _p0 = vfmul_vf_f32m4(_p0, scale, vl);
                _p1 = vfmul_vf_f32m4(_p1, scale, vl);
                _p2 = vfmul_vf_f32m4(_p2, scale, vl);
                _p3 = vfmul_vf_f32m4(_p3, scale, vl);
                _p4 = vfmul_vf_f32m4(_p4, scale, vl);
                _p5 = vfmul_vf_f32m4(_p5, scale, vl);
                _p6 = vfmul_vf_f32m4(_p6, scale, vl);
                _p7 = vfmul_vf_f32m4(_p7, scale, vl);
                vint8m1_t _outp0 = float2int8(_p0, vl);
                vint8m1_t _outp1 = float2int8(_p1, vl);
                vint8m1_t _outp2 = float2int8(_p2, vl);
                vint8m1_t _outp3 = float2int8(_p3, vl);
                vint8m1_t _outp4 = float2int8(_p4, vl);
                vint8m1_t _outp5 = float2int8(_p5, vl);
                vint8m1_t _outp6 = float2int8(_p6, vl);
                vint8m1_t _outp7 = float2int8(_p7, vl);
                vse8_v_i8m1(outptr, _outp0, vl);
                vse8_v_i8m1(outptr + packn, _outp1, vl);
                vse8_v_i8m1(outptr + 2 * packn, _outp2, vl);
                vse8_v_i8m1(outptr + 3 * packn, _outp3, vl);
                vse8_v_i8m1(outptr + 4 * packn, _outp4, vl);
                vse8_v_i8m1(outptr + 5 * packn, _outp5, vl);
                vse8_v_i8m1(outptr + 6 * packn, _outp6, vl);
                vse8_v_i8m1(outptr + 7 * packn, _outp7, vl);
                ptr += 8 * packn;
                outptr += 8 * packn;
            }

            num_nn = (w - remain_w_start) / (packn * 4);
            remain_w_start += num_nn * packn * 4;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < num_nn; i++)
            {
                vfloat32m4_t _p0 = vle32_v_f32m4(ptr, vl);
                vfloat32m4_t _p1 = vle32_v_f32m4(ptr + packn, vl);
                vfloat32m4_t _p2 = vle32_v_f32m4(ptr + 2 * packn, vl);
                vfloat32m4_t _p3 = vle32_v_f32m4(ptr + 3 * packn, vl);
                _p0 = vfmul_vf_f32m4(_p0, scale, vl);
                _p1 = vfmul_vf_f32m4(_p1, scale, vl);
                _p2 = vfmul_vf_f32m4(_p2, scale, vl);
                _p3 = vfmul_vf_f32m4(_p3, scale, vl);
                vint8m1_t _outp0 = float2int8(_p0, vl);
                vint8m1_t _outp1 = float2int8(_p1, vl);
                vint8m1_t _outp2 = float2int8(_p2, vl);
                vint8m1_t _outp3 = float2int8(_p3, vl);
                vse8_v_i8m1(outptr, _outp0, vl);
                vse8_v_i8m1(outptr + packn, _outp1, vl);
                vse8_v_i8m1(outptr + 2 * packn, _outp2, vl);
                vse8_v_i8m1(outptr + 3 * packn, _outp3, vl);
                ptr += 4 * packn;
                outptr += 4 * packn;
            }
#else
            int remain_w_start = 0;
#endif
            for (int j = remain_w_start; j < w; j++)
            {
                *outptr++ = float2int8(*ptr++ * scale);
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

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            signed char* outptr = top_blob.channel(q);

            const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

#if __riscv_vector
            int num_nn = w / (packn * 8);
            int remain_w_start = num_nn * packn * 8;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < num_nn; i++)
            {
                vfloat32m4_t _p0 = vle32_v_f32m4(ptr, vl);
                vfloat32m4_t _p1 = vle32_v_f32m4(ptr + packn, vl);
                vfloat32m4_t _p2 = vle32_v_f32m4(ptr + 2 * packn, vl);
                vfloat32m4_t _p3 = vle32_v_f32m4(ptr + 3 * packn, vl);
                vfloat32m4_t _p4 = vle32_v_f32m4(ptr + 4 * packn, vl);
                vfloat32m4_t _p5 = vle32_v_f32m4(ptr + 5 * packn, vl);
                vfloat32m4_t _p6 = vle32_v_f32m4(ptr + 6 * packn, vl);
                vfloat32m4_t _p7 = vle32_v_f32m4(ptr + 7 * packn, vl);
                _p0 = vfmul_vf_f32m4(_p0, scale, vl);
                _p1 = vfmul_vf_f32m4(_p1, scale, vl);
                _p2 = vfmul_vf_f32m4(_p2, scale, vl);
                _p3 = vfmul_vf_f32m4(_p3, scale, vl);
                _p4 = vfmul_vf_f32m4(_p4, scale, vl);
                _p5 = vfmul_vf_f32m4(_p5, scale, vl);
                _p6 = vfmul_vf_f32m4(_p6, scale, vl);
                _p7 = vfmul_vf_f32m4(_p7, scale, vl);
                vint8m1_t _outp0 = float2int8(_p0, vl);
                vint8m1_t _outp1 = float2int8(_p1, vl);
                vint8m1_t _outp2 = float2int8(_p2, vl);
                vint8m1_t _outp3 = float2int8(_p3, vl);
                vint8m1_t _outp4 = float2int8(_p4, vl);
                vint8m1_t _outp5 = float2int8(_p5, vl);
                vint8m1_t _outp6 = float2int8(_p6, vl);
                vint8m1_t _outp7 = float2int8(_p7, vl);
                vse8_v_i8m1(outptr, _outp0, vl);
                vse8_v_i8m1(outptr + packn, _outp1, vl);
                vse8_v_i8m1(outptr + 2 * packn, _outp2, vl);
                vse8_v_i8m1(outptr + 3 * packn, _outp3, vl);
                vse8_v_i8m1(outptr + 4 * packn, _outp4, vl);
                vse8_v_i8m1(outptr + 5 * packn, _outp5, vl);
                vse8_v_i8m1(outptr + 6 * packn, _outp6, vl);
                vse8_v_i8m1(outptr + 7 * packn, _outp7, vl);
                ptr += 8 * packn;
                outptr += 8 * packn;
            }

            num_nn = (w - remain_w_start) / (packn * 4);
            remain_w_start += num_nn * packn * 4;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < num_nn; i++)
            {
                vfloat32m4_t _p0 = vle32_v_f32m4(ptr, vl);
                vfloat32m4_t _p1 = vle32_v_f32m4(ptr + packn, vl);
                vfloat32m4_t _p2 = vle32_v_f32m4(ptr + 2 * packn, vl);
                vfloat32m4_t _p3 = vle32_v_f32m4(ptr + 3 * packn, vl);
                _p0 = vfmul_vf_f32m4(_p0, scale, vl);
                _p1 = vfmul_vf_f32m4(_p1, scale, vl);
                _p2 = vfmul_vf_f32m4(_p2, scale, vl);
                _p3 = vfmul_vf_f32m4(_p3, scale, vl);
                vint8m1_t _outp0 = float2int8(_p0, vl);
                vint8m1_t _outp1 = float2int8(_p1, vl);
                vint8m1_t _outp2 = float2int8(_p2, vl);
                vint8m1_t _outp3 = float2int8(_p3, vl);
                vse8_v_i8m1(outptr, _outp0, vl);
                vse8_v_i8m1(outptr + packn, _outp1, vl);
                vse8_v_i8m1(outptr + 2 * packn, _outp2, vl);
                vse8_v_i8m1(outptr + 3 * packn, _outp3, vl);
                ptr += 4 * packn;
                outptr += 4 * packn;
            }
#else
            int remain_w_start = 0;
#endif
            for (int i = remain_w_start; i < size; i++)
            {
                *outptr++ = float2int8(*ptr++ * scale);
            }
        }
    }

    return 0;
}

} // namespace ncnn
