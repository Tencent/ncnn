// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "interp_arm.h"
#include <math.h>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

#include "interp_bilinear.h"
#include "interp_bicubic.h"
#include "interp_bilinear_bf16s.h"
#include "interp_bicubic_bf16s.h"

#if __ARM_NEON
#include "interp_bilinear_pack4.h"
#include "interp_bicubic_pack4.h"
#include "interp_bilinear_pack4_bf16s.h"
#include "interp_bicubic_pack4_bf16s.h"
#endif

DEFINE_LAYER_CREATOR(Interp_arm)

Interp_arm::Interp_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON

    support_bf16_storage = true;
}

int Interp_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (opt.use_bf16_storage)
        return forward_bf16s(bottom_blob, top_blob, opt);

    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (dims == 1)
    {
        return Interp::forward(bottom_blob, top_blob, opt);
    }

    int outh = output_height;
    int outw = output_width;

    if (outh == 0 || outw == 0)
    {
        outh = h * height_scale;
        outw = w * width_scale;
    }

    if (outh == h && outw == w)
    {
        top_blob = bottom_blob;
        return 0;
    }

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __ARM_NEON
    if (elempack == 4)
    {
        if (resize_type == 1)// nearest
        {
            const float hs = output_height ? h / (float)output_height : 1.f / height_scale;
            const float ws = output_width ? w / (float)output_width : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    int in_y = std::min((int) (y * hs), (h - 1));

                    const float* ptr = src.row(in_y);
                    float* outptr = dst.row(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int) (x * ws), (w - 1));

                        float32x4_t _p = vld1q_f32(ptr + in_x * 4);
                        vst1q_f32(outptr, _p);

                        outptr += 4;
                    }
                }
            }
        }

        if (resize_type == 2)// bilinear
        {
            int* buf = new int[outw + outh + outw*2 + outh*2];

            int* xofs = buf;//new int[outw];
            int* yofs = buf + outw;//new int[outh];

            float* alpha = (float*)(buf + outw + outh);//new float[outw * 2];
            float* beta = (float*)(buf + outw + outh + outw*2);//new float[outh * 2];

            linear_coeffs(w, outw, xofs, alpha);
            linear_coeffs(h, outh, yofs, beta);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bilinear_image_pack4(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        if (resize_type == 3)// bicubic
        {
            int* buf = new int[outw + outh + outw*4 + outh*4];

            int* xofs = buf;//new int[outw];
            int* yofs = buf + outw;//new int[outh];

            float* alpha = (float*)(buf + outw + outh);//new float[outw * 4];
            float* beta = (float*)(buf + outw + outh + outw*4);//new float[outh * 4];

            cubic_coeffs(w, outw, xofs, alpha);
            cubic_coeffs(h, outh, yofs, beta);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bicubic_image_pack4(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        return 0;
    }
#endif // __ARM_NEON

    if (resize_type == 1)// nearest
    {
        const float hs = output_height ? h / (float)output_height : 1.f / height_scale;
        const float ws = output_width ? w / (float)output_width : 1.f / width_scale;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            for (int y = 0; y < outh; y++)
            {
                int in_y = std::min((int) (y * hs), (h - 1));

                const float* ptr = src.row(in_y);
                float* outptr = dst.row(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int) (x * ws), (w - 1));
                    *outptr++ = ptr[in_x];
                }
            }
        }
    }

    if (resize_type == 2)// bilinear
    {
        int* buf = new int[outw + outh + outw*2 + outh*2];

        int* xofs = buf;//new int[outw];
        int* yofs = buf + outw;//new int[outh];

        float* alpha = (float*)(buf + outw + outh);//new float[outw * 2];
        float* beta = (float*)(buf + outw + outh + outw*2);//new float[outh * 2];

        linear_coeffs(w, outw, xofs, alpha);
        linear_coeffs(h, outh, yofs, beta);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bilinear_image(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    if (resize_type == 3)// bicubic
    {
        int* buf = new int[outw + outh + outw*4 + outh*4];

        int* xofs = buf;//new int[outw];
        int* yofs = buf + outw;//new int[outh];

        float* alpha = (float*)(buf + outw + outh);//new float[outw * 4];
        float* beta = (float*)(buf + outw + outh + outw*4);//new float[outh * 4];

        cubic_coeffs(w, outw, xofs, alpha);
        cubic_coeffs(h, outh, yofs, beta);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bicubic_image(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    return 0;
}

int Interp_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (dims == 1)
    {
        return Interp::forward(bottom_blob, top_blob, opt);
    }

    int outh = output_height;
    int outw = output_width;

    if (outh == 0 || outw == 0)
    {
        outh = h * height_scale;
        outw = w * width_scale;
    }

    if (outh == h && outw == w)
    {
        top_blob = bottom_blob;
        return 0;
    }

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __ARM_NEON
    if (elempack == 4)
    {
        if (resize_type == 1)// nearest
        {
            const float hs = output_height ? h / (float)output_height : 1.f / height_scale;
            const float ws = output_width ? w / (float)output_width : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    int in_y = std::min((int) (y * hs), (h - 1));

                    const unsigned short* ptr = src.row<const unsigned short>(in_y);
                    unsigned short* outptr = dst.row<unsigned short>(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int) (x * ws), (w - 1));

                        uint16x4_t _p = vld1_u16(ptr + in_x * 4);
                        vst1_u16(outptr, _p);

                        outptr += 4;
                    }
                }
            }
        }

        if (resize_type == 2)// bilinear
        {
            int* buf = new int[outw + outh + outw*2 + outh*2];

            int* xofs = buf;//new int[outw];
            int* yofs = buf + outw;//new int[outh];

            float* alpha = (float*)(buf + outw + outh);//new float[outw * 2];
            float* beta = (float*)(buf + outw + outh + outw*2);//new float[outh * 2];

            linear_coeffs(w, outw, xofs, alpha);
            linear_coeffs(h, outh, yofs, beta);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bilinear_image_pack4_bf16s(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        if (resize_type == 3)// bicubic
        {
            int* buf = new int[outw + outh + outw*4 + outh*4];

            int* xofs = buf;//new int[outw];
            int* yofs = buf + outw;//new int[outh];

            float* alpha = (float*)(buf + outw + outh);//new float[outw * 4];
            float* beta = (float*)(buf + outw + outh + outw*4);//new float[outh * 4];

            cubic_coeffs(w, outw, xofs, alpha);
            cubic_coeffs(h, outh, yofs, beta);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bicubic_image_pack4_bf16s(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        return 0;
    }
#endif // __ARM_NEON

    if (resize_type == 1)// nearest
    {
        const float hs = output_height ? h / (float)output_height : 1.f / height_scale;
        const float ws = output_width ? w / (float)output_width : 1.f / width_scale;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            for (int y = 0; y < outh; y++)
            {
                int in_y = std::min((int) (y * hs), (h - 1));

                const unsigned short* ptr = src.row<const unsigned short>(in_y);
                unsigned short* outptr = dst.row<unsigned short>(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int) (x * ws), (w - 1));
                    *outptr++ = ptr[in_x];
                }
            }
        }
    }

    if (resize_type == 2)// bilinear
    {
        int* buf = new int[outw + outh + outw*2 + outh*2];

        int* xofs = buf;//new int[outw];
        int* yofs = buf + outw;//new int[outh];

        float* alpha = (float*)(buf + outw + outh);//new float[outw * 2];
        float* beta = (float*)(buf + outw + outh + outw*2);//new float[outh * 2];

        linear_coeffs(w, outw, xofs, alpha);
        linear_coeffs(h, outh, yofs, beta);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bilinear_image_bf16s(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    if (resize_type == 3)// bicubic
    {
        int* buf = new int[outw + outh + outw*4 + outh*4];

        int* xofs = buf;//new int[outw];
        int* yofs = buf + outw;//new int[outh];

        float* alpha = (float*)(buf + outw + outh);//new float[outw * 4];
        float* beta = (float*)(buf + outw + outh + outw*4);//new float[outh * 4];

        cubic_coeffs(w, outw, xofs, alpha);
        cubic_coeffs(h, outh, yofs, beta);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bicubic_image_bf16s(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    return 0;
}

} // namespace ncnn
