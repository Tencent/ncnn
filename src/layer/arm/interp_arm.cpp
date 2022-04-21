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

#include "interp_bicubic.h"
#include "interp_bilinear.h"

#if NCNN_BF16
#include "interp_bicubic_bf16s.h"
#include "interp_bilinear_bf16s.h"
#endif

#if __ARM_NEON
#include "interp_bicubic_pack4.h"
#include "interp_bilinear_pack4.h"
#if NCNN_BF16
#include "interp_bicubic_pack4_bf16s.h"
#include "interp_bilinear_pack4_bf16s.h"
#endif
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "interp_bicubic_fp16s.h"
#include "interp_bicubic_pack4_fp16s.h"
#include "interp_bicubic_pack8_fp16s.h"
#include "interp_bilinear_fp16s.h"
#include "interp_bilinear_pack4_fp16s.h"
#include "interp_bilinear_pack8_fp16s.h"
#endif
#endif

Interp_arm::Interp_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Interp_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int elembits = bottom_blob.elembits();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blobs, top_blobs, opt);
        else
            return forward_fp16s(bottom_blobs, top_blobs, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);
#endif

    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = reference_blob.w;
    int outh = reference_blob.h;

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __ARM_NEON
        if (elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                float32x4_t _v = vld1q_f32((const float*)bottom_blob + q * 4);
                top_blob_c.fill(_v);
            }

            return 0;
        }
#endif // __ARM_NEON

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < w; q++)
        {
            Mat top_blob_c = top_blob.channel(q);
            const float v = bottom_blob[q];
            top_blob_c.fill(v);
        }

        return 0;
    }

    if (dims == 2)
    {
        if (outw == w)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __ARM_NEON
        if (elempack == 4)
        {
            if (resize_type == 1) // nearest
            {
                const float ws = output_width ? w / (float)outw : 1.f / width_scale;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        float32x4_t _p = vld1q_f32(ptr + in_x * 4);
                        vst1q_f32(outptr, _p);

                        outptr += 4;
                    }
                }
            }

            if (resize_type == 2) // bilinear
            {
                int* buf = new int[outw + outw * 2];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                linear_coeffs(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 4;
                        const float* Sp = ptr + sx;

                        float32x2_t _a01 = vld1_f32(alphap);

                        float32x4_t _S0 = vld1q_f32(Sp);
                        float32x4_t _S1 = vld1q_f32(Sp + 4);
                        float32x4_t _p = vmulq_lane_f32(_S0, _a01, 0);
                        _p = vmlaq_lane_f32(_p, _S1, _a01, 1);
                        vst1q_f32(outptr, _p);

                        alphap += 2;
                        outptr += 4;
                    }
                }

                delete[] buf;
            }

            if (resize_type == 3) // bicubic
            {
                int* buf = new int[outw + outw * 4];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                cubic_coeffs(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 4;
                        const float* Sp = ptr + sx;

                        float32x4_t _a0123 = vld1q_f32(alphap);

                        float32x4_t _S0 = vld1q_f32(Sp - 4);
                        float32x4_t _S1 = vld1q_f32(Sp + 0);
                        float32x4_t _S2 = vld1q_f32(Sp + 4);
                        float32x4_t _S3 = vld1q_f32(Sp + 8);
                        float32x4_t _p = vmulq_lane_f32(_S0, vget_low_f32(_a0123), 0);
                        _p = vmlaq_lane_f32(_p, _S1, vget_low_f32(_a0123), 1);
                        _p = vmlaq_lane_f32(_p, _S2, vget_high_f32(_a0123), 0);
                        _p = vmlaq_lane_f32(_p, _S3, vget_high_f32(_a0123), 1);
                        vst1q_f32(outptr, _p);

                        alphap += 4;
                        outptr += 4;
                    }
                }

                delete[] buf;
            }

            return 0;
        }
#endif // __ARM_NEON

        if (resize_type == 1) // nearest
        {
            const float ws = output_width ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.row(y);
                float* outptr = top_blob.row(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    *outptr++ = ptr[in_x];
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outw * 2];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            linear_coeffs(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.row(y);
                float* outptr = top_blob.row(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const float* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    *outptr++ = Sp[0] * a0 + Sp[1] * a1;
                    alphap += 2;
                }
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outw * 4];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            cubic_coeffs(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.row(y);
                float* outptr = top_blob.row(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const float* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    float a2 = alphap[2];
                    float a3 = alphap[3];
                    *outptr++ = Sp[-1] * a0 + Sp[0] * a1 + Sp[1] * a2 + Sp[2] * a3;
                    alphap += 4;
                }
            }

            delete[] buf;
        }

        return 0;
    }

    if (outw == w && outh == h)
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
        if (resize_type == 1) // nearest
        {
            const float hs = output_height ? h / (float)outh : 1.f / height_scale;
            const float ws = output_width ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    int in_y = std::min((int)(y * hs), (h - 1));

                    const float* ptr = src.row(in_y);
                    float* outptr = dst.row(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        float32x4_t _p = vld1q_f32(ptr + in_x * 4);
                        vst1q_f32(outptr, _p);

                        outptr += 4;
                    }
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outh + outw * 2 + outh * 2];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
            float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

            linear_coeffs(w, outw, xofs, alpha, align_corner);
            linear_coeffs(h, outh, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bilinear_image_pack4(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outh + outw * 4 + outh * 4];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
            float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

            cubic_coeffs(w, outw, xofs, alpha, align_corner);
            cubic_coeffs(h, outh, yofs, beta, align_corner);

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

    if (resize_type == 1) // nearest
    {
        const float hs = output_height ? h / (float)outh : 1.f / height_scale;
        const float ws = output_width ? w / (float)outw : 1.f / width_scale;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            for (int y = 0; y < outh; y++)
            {
                int in_y = std::min((int)(y * hs), (h - 1));

                const float* ptr = src.row(in_y);
                float* outptr = dst.row(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    *outptr++ = ptr[in_x];
                }
            }
        }
    }

    if (resize_type == 2) // bilinear
    {
        int* buf = new int[outw + outh + outw * 2 + outh * 2];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
        float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

        linear_coeffs(w, outw, xofs, alpha, align_corner);
        linear_coeffs(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bilinear_image(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    if (resize_type == 3) // bicubic
    {
        int* buf = new int[outw + outh + outw * 4 + outh * 4];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
        float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

        cubic_coeffs(w, outw, xofs, alpha, align_corner);
        cubic_coeffs(h, outh, yofs, beta, align_corner);

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

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int Interp_arm::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = reference_blob.w;
    int outh = reference_blob.h;

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                float16x4_t _v = vld1_f16((const __fp16*)bottom_blob + q * 4);
                top_blob_c.fill(_v);
            }

            return 0;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < w; q++)
        {
            Mat top_blob_c = top_blob.channel(q);
            const __fp16* ptr = bottom_blob;
            top_blob_c.fill(ptr[q]);
        }

        return 0;
    }

    if (dims == 2)
    {
        if (outw == w)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elempack == 4)
        {
            if (resize_type == 1) // nearest
            {
                const float ws = output_width ? w / (float)outw : 1.f / width_scale;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                    __fp16* outptr = top_blob.row<__fp16>(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        float16x4_t _p = vld1_f16(ptr + in_x * 4);
                        vst1_f16(outptr, _p);

                        outptr += 4;
                    }
                }
            }

            if (resize_type == 2) // bilinear
            {
                int* buf = new int[outw + outw * 2];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                linear_coeffs(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                    __fp16* outptr = top_blob.row<__fp16>(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 4;
                        const __fp16* Sp = ptr + sx;

                        float32x2_t _a01 = vld1_f32(alphap);

                        float32x4_t _S0 = vcvt_f32_f16(vld1_f16(Sp));
                        float32x4_t _S1 = vcvt_f32_f16(vld1_f16(Sp + 4));
                        float32x4_t _p = vmulq_lane_f32(_S0, _a01, 0);
                        _p = vmlaq_lane_f32(_p, _S1, _a01, 1);
                        vst1_f16(outptr, vcvt_f16_f32(_p));

                        alphap += 2;
                        outptr += 4;
                    }
                }

                delete[] buf;
            }

            if (resize_type == 3) // bicubic
            {
                int* buf = new int[outw + outw * 4];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                cubic_coeffs(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                    __fp16* outptr = top_blob.row<__fp16>(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 4;
                        const __fp16* Sp = ptr + sx;

                        float32x4_t _a0123 = vld1q_f32(alphap);

                        float32x4_t _S0 = vcvt_f32_f16(vld1_f16(Sp - 4));
                        float32x4_t _S1 = vcvt_f32_f16(vld1_f16(Sp + 0));
                        float32x4_t _S2 = vcvt_f32_f16(vld1_f16(Sp + 4));
                        float32x4_t _S3 = vcvt_f32_f16(vld1_f16(Sp + 8));
                        float32x4_t _p = vmulq_laneq_f32(_S0, _a0123, 0);
                        _p = vfmaq_laneq_f32(_p, _S1, _a0123, 1);
                        _p = vfmaq_laneq_f32(_p, _S2, _a0123, 2);
                        _p = vfmaq_laneq_f32(_p, _S3, _a0123, 3);
                        vst1_f16(outptr, vcvt_f16_f32(_p));

                        alphap += 4;
                        outptr += 4;
                    }
                }

                delete[] buf;
            }

            return 0;
        }

        if (resize_type == 1) // nearest
        {
            const float ws = output_width ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                __fp16* outptr = top_blob.row<__fp16>(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    *outptr++ = ptr[in_x];
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outw * 2];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            linear_coeffs(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                __fp16* outptr = top_blob.row<__fp16>(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const __fp16* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    *outptr++ = (__fp16)((float)Sp[0] * a0 + (float)Sp[1] * a1);
                    alphap += 2;
                }
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outw * 4];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            cubic_coeffs(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                __fp16* outptr = top_blob.row<__fp16>(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const __fp16* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    float a2 = alphap[2];
                    float a3 = alphap[3];
                    *outptr++ = (__fp16)((float)Sp[-1] * a0 + (float)Sp[0] * a1 + (float)Sp[1] * a2 + (float)Sp[2] * a3);
                    alphap += 4;
                }
            }

            delete[] buf;
        }

        return 0;
    }

    if (outw == w && outh == h)
    {
        top_blob = bottom_blob;
        return 0;
    }

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (elempack == 4)
    {
        if (resize_type == 1) // nearest
        {
            const float hs = output_height ? h / (float)outh : 1.f / height_scale;
            const float ws = output_width ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    int in_y = std::min((int)(y * hs), (h - 1));

                    const __fp16* ptr = src.row<const __fp16>(in_y);
                    __fp16* outptr = dst.row<__fp16>(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        float16x4_t _p = vld1_f16(ptr + in_x * 4);
                        vst1_f16(outptr, _p);

                        outptr += 4;
                    }
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outh + outw * 2 + outh * 2];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
            float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

            linear_coeffs(w, outw, xofs, alpha, align_corner);
            linear_coeffs(h, outh, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bilinear_image_pack4_fp16s(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outh + outw * 4 + outh * 4];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
            float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

            cubic_coeffs(w, outw, xofs, alpha, align_corner);
            cubic_coeffs(h, outh, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bicubic_image_pack4_fp16s(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        return 0;
    }

    if (resize_type == 1) // nearest
    {
        const float hs = output_height ? h / (float)outh : 1.f / height_scale;
        const float ws = output_width ? w / (float)outw : 1.f / width_scale;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            for (int y = 0; y < outh; y++)
            {
                int in_y = std::min((int)(y * hs), (h - 1));

                const __fp16* ptr = src.row<const __fp16>(in_y);
                __fp16* outptr = dst.row<__fp16>(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    *outptr++ = ptr[in_x];
                }
            }
        }
    }

    if (resize_type == 2) // bilinear
    {
        int* buf = new int[outw + outh + outw * 2 + outh * 2];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
        float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

        linear_coeffs(w, outw, xofs, alpha, align_corner);
        linear_coeffs(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bilinear_image_fp16s(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    if (resize_type == 3) // bicubic
    {
        int* buf = new int[outw + outh + outw * 4 + outh * 4];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
        float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

        cubic_coeffs(w, outw, xofs, alpha, align_corner);
        cubic_coeffs(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bicubic_image_fp16s(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    return 0;
}

int Interp_arm::forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = reference_blob.w;
    int outh = reference_blob.h;

    if ((elempack == 1 || elempack == 4) && (dims == 1 || resize_type == 1)) // nearest
    {
        return forward_fp16s(bottom_blobs, top_blobs, opt);
    }

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                float16x8_t _v = vld1q_f16((const __fp16*)bottom_blob + q * 8);
                top_blob_c.fill(_v);
            }

            return 0;
        }

        return 0;
    }

    if (dims == 2)
    {
        if (outw == w)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elempack == 8)
        {
            if (resize_type == 1) // nearest
            {
                const float ws = output_width ? w / (float)outw : 1.f / width_scale;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                    __fp16* outptr = top_blob.row<__fp16>(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        float16x8_t _p = vld1q_f16(ptr + in_x * 8);
                        vst1q_f16(outptr, _p);

                        outptr += 8;
                    }
                }
            }

            if (resize_type == 2) // bilinear
            {
                int* buf = new int[outw + outw * 2];

                int* xofs = buf;
                __fp16* alpha = (__fp16*)(buf + outw);

                linear_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                    __fp16* outptr = top_blob.row<__fp16>(y);
                    const __fp16* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 8;
                        const __fp16* Sp = ptr + sx;

                        float16x4_t _a01 = vld1_f16(alphap);

                        float16x8_t _S0 = vld1q_f16(Sp);
                        float16x8_t _S1 = vld1q_f16(Sp + 8);
                        float16x8_t _p = vmulq_lane_f16(_S0, _a01, 0);
                        _p = vfmaq_lane_f16(_p, _S1, _a01, 1);
                        vst1q_f16(outptr, _p);

                        alphap += 2;
                        outptr += 8;
                    }
                }

                delete[] buf;
            }

            if (resize_type == 3) // bicubic
            {
                int* buf = new int[outw + outw * 4];

                int* xofs = buf;
                __fp16* alpha = (__fp16*)(buf + outw);

                cubic_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                    __fp16* outptr = top_blob.row<__fp16>(y);
                    const __fp16* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 8;
                        const __fp16* Sp = ptr + sx;

                        float16x4_t _a0123 = vld1_f16(alphap);

                        float16x8_t _S0 = vld1q_f16(Sp - 8);
                        float16x8_t _S1 = vld1q_f16(Sp + 0);
                        float16x8_t _S2 = vld1q_f16(Sp + 8);
                        float16x8_t _S3 = vld1q_f16(Sp + 16);
                        float16x8_t _p = vmulq_lane_f16(_S0, _a0123, 0);
                        _p = vfmaq_lane_f16(_p, _S1, _a0123, 1);
                        _p = vfmaq_lane_f16(_p, _S2, _a0123, 2);
                        _p = vfmaq_lane_f16(_p, _S3, _a0123, 3);
                        vst1q_f16(outptr, _p);

                        alphap += 4;
                        outptr += 8;
                    }
                }

                delete[] buf;
            }

            return 0;
        }

        if (elempack == 4)
        {
            if (resize_type == 2) // bilinear
            {
                int* buf = new int[outw + outw * 2];

                int* xofs = buf;
                __fp16* alpha = (__fp16*)(buf + outw);

                linear_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                    __fp16* outptr = top_blob.row<__fp16>(y);
                    const __fp16* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 4;
                        const __fp16* Sp = ptr + sx;

                        float16x4_t _a01 = vld1_f16(alphap);

                        float16x4_t _S0 = vld1_f16(Sp);
                        float16x4_t _S1 = vld1_f16(Sp + 4);
                        float16x4_t _p = vmul_lane_f16(_S0, _a01, 0);
                        _p = vfma_lane_f16(_p, _S1, _a01, 1);
                        vst1_f16(outptr, _p);

                        alphap += 2;
                        outptr += 4;
                    }
                }

                delete[] buf;
            }

            if (resize_type == 3) // bicubic
            {
                int* buf = new int[outw + outw * 4];

                int* xofs = buf;
                __fp16* alpha = (__fp16*)(buf + outw);

                cubic_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                    __fp16* outptr = top_blob.row<__fp16>(y);
                    const __fp16* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 4;
                        const __fp16* Sp = ptr + sx;

                        float16x4_t _a0123 = vld1_f16(alphap);

                        float16x4_t _S0 = vld1_f16(Sp - 4);
                        float16x4_t _S1 = vld1_f16(Sp + 0);
                        float16x4_t _S2 = vld1_f16(Sp + 4);
                        float16x4_t _S3 = vld1_f16(Sp + 8);
                        float16x4_t _p = vmul_lane_f16(_S0, _a0123, 0);
                        _p = vfma_lane_f16(_p, _S1, _a0123, 1);
                        _p = vfma_lane_f16(_p, _S2, _a0123, 2);
                        _p = vfma_lane_f16(_p, _S3, _a0123, 3);
                        vst1_f16(outptr, _p);

                        alphap += 4;
                        outptr += 4;
                    }
                }

                delete[] buf;
            }

            return 0;
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outw * 2];

            int* xofs = buf;
            __fp16* alpha = (__fp16*)(buf + outw);

            linear_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                __fp16* outptr = top_blob.row<__fp16>(y);
                const __fp16* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const __fp16* Sp = ptr + sx;
                    __fp16 a0 = alphap[0];
                    __fp16 a1 = alphap[1];
                    *outptr++ = Sp[0] * a0 + Sp[1] * a1;
                    alphap += 2;
                }
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outw * 4];

            int* xofs = buf;
            __fp16* alpha = (__fp16*)(buf + outw);

            cubic_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                __fp16* outptr = top_blob.row<__fp16>(y);
                const __fp16* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const __fp16* Sp = ptr + sx;
                    __fp16 a0 = alphap[0];
                    __fp16 a1 = alphap[1];
                    __fp16 a2 = alphap[2];
                    __fp16 a3 = alphap[3];
                    *outptr++ = Sp[-1] * a0 + Sp[0] * a1 + Sp[1] * a2 + Sp[2] * a3;
                    alphap += 4;
                }
            }

            delete[] buf;
        }

        return 0;
    }

    if (outw == w && outh == h)
    {
        top_blob = bottom_blob;
        return 0;
    }

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (elempack == 8)
    {
        if (resize_type == 1) // nearest
        {
            const float hs = output_height ? h / (float)outh : 1.f / height_scale;
            const float ws = output_width ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    int in_y = std::min((int)(y * hs), (h - 1));

                    const __fp16* ptr = src.row<const __fp16>(in_y);
                    __fp16* outptr = dst.row<__fp16>(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        float16x8_t _p = vld1q_f16(ptr + in_x * 8);
                        vst1q_f16(outptr, _p);

                        outptr += 8;
                    }
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outh + outw * 2 + outh * 2];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            __fp16* alpha = (__fp16*)(buf + outw + outh);           //new __fp16[outw * 2];
            __fp16* beta = (__fp16*)(buf + outw + outh + outw * 2); //new __fp16[outh * 2];

            linear_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);
            linear_coeffs_fp16sa(h, outh, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bilinear_image_pack8_fp16sa(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outh + outw * 4 + outh * 4];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            __fp16* alpha = (__fp16*)(buf + outw + outh);           //new __fp16[outw * 4];
            __fp16* beta = (__fp16*)(buf + outw + outh + outw * 4); //new __fp16[outh * 4];

            cubic_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);
            cubic_coeffs_fp16sa(h, outh, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bicubic_image_pack8_fp16sa(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        return 0;
    }

    if (elempack == 4)
    {
        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outh + outw * 2 + outh * 2];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            __fp16* alpha = (__fp16*)(buf + outw + outh);           //new __fp16[outw * 2];
            __fp16* beta = (__fp16*)(buf + outw + outh + outw * 2); //new __fp16[outh * 2];

            linear_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);
            linear_coeffs_fp16sa(h, outh, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bilinear_image_pack4_fp16sa(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outh + outw * 4 + outh * 4];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            __fp16* alpha = (__fp16*)(buf + outw + outh);           //new __fp16[outw * 4];
            __fp16* beta = (__fp16*)(buf + outw + outh + outw * 4); //new __fp16[outh * 4];

            cubic_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);
            cubic_coeffs_fp16sa(h, outh, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bicubic_image_pack4_fp16sa(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        return 0;
    }

    if (resize_type == 2) // bilinear
    {
        int* buf = new int[outw + outh + outw * 2 + outh * 2];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        __fp16* alpha = (__fp16*)(buf + outw + outh);           //new __fp16[outw * 2];
        __fp16* beta = (__fp16*)(buf + outw + outh + outw * 2); //new __fp16[outh * 2];

        linear_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);
        linear_coeffs_fp16sa(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bilinear_image_fp16sa(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    if (resize_type == 3) // bicubic
    {
        int* buf = new int[outw + outh + outw * 4 + outh * 4];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        __fp16* alpha = (__fp16*)(buf + outw + outh);           //new __fp16[outw * 4];
        __fp16* beta = (__fp16*)(buf + outw + outh + outw * 4); //new __fp16[outh * 4];

        cubic_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);
        cubic_coeffs_fp16sa(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bicubic_image_fp16sa(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if NCNN_BF16
int Interp_arm::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = reference_blob.w;
    int outh = reference_blob.h;

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __ARM_NEON
        if (elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                uint16x4_t _v = vld1_u16((const unsigned short*)bottom_blob + q * 4);
                top_blob_c.fill(_v);
            }

            return 0;
        }
#endif // __ARM_NEON

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < w; q++)
        {
            Mat top_blob_c = top_blob.channel(q);
            const unsigned short* ptr = bottom_blob;
            top_blob_c.fill(ptr[q]);
        }

        return 0;
    }

    if (dims == 2)
    {
        if (outw == w)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __ARM_NEON
        if (elempack == 4)
        {
            if (resize_type == 1) // nearest
            {
                const float ws = output_width ? w / (float)outw : 1.f / width_scale;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                    unsigned short* outptr = top_blob.row<unsigned short>(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        uint16x4_t _p = vld1_u16(ptr + in_x * 4);
                        vst1_u16(outptr, _p);

                        outptr += 4;
                    }
                }
            }

            if (resize_type == 2) // bilinear
            {
                int* buf = new int[outw + outw * 2];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                linear_coeffs(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                    unsigned short* outptr = top_blob.row<unsigned short>(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 4;
                        const unsigned short* Sp = ptr + sx;

                        float32x2_t _a01 = vld1_f32(alphap);

                        float32x4_t _S0 = vcvt_f32_bf16(vld1_u16(Sp));
                        float32x4_t _S1 = vcvt_f32_bf16(vld1_u16(Sp + 4));
                        float32x4_t _p = vmulq_lane_f32(_S0, _a01, 0);
                        _p = vmlaq_lane_f32(_p, _S1, _a01, 1);
                        vst1_u16(outptr, vcvt_bf16_f32(_p));

                        alphap += 2;
                        outptr += 4;
                    }
                }

                delete[] buf;
            }

            if (resize_type == 3) // bicubic
            {
                int* buf = new int[outw + outw * 4];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                cubic_coeffs(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                    unsigned short* outptr = top_blob.row<unsigned short>(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 4;
                        const unsigned short* Sp = ptr + sx;

                        float32x4_t _a0123 = vld1q_f32(alphap);

                        float32x4_t _S0 = vcvt_f32_bf16(vld1_u16(Sp - 4));
                        float32x4_t _S1 = vcvt_f32_bf16(vld1_u16(Sp + 0));
                        float32x4_t _S2 = vcvt_f32_bf16(vld1_u16(Sp + 4));
                        float32x4_t _S3 = vcvt_f32_bf16(vld1_u16(Sp + 8));
                        float32x4_t _p = vmulq_lane_f32(_S0, vget_low_f32(_a0123), 0);
                        _p = vmlaq_lane_f32(_p, _S1, vget_low_f32(_a0123), 1);
                        _p = vmlaq_lane_f32(_p, _S2, vget_high_f32(_a0123), 0);
                        _p = vmlaq_lane_f32(_p, _S3, vget_high_f32(_a0123), 1);
                        vst1_u16(outptr, vcvt_bf16_f32(_p));

                        alphap += 4;
                        outptr += 4;
                    }
                }

                delete[] buf;
            }

            return 0;
        }
#endif // __ARM_NEON

        if (resize_type == 1) // nearest
        {
            const float ws = output_width ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                unsigned short* outptr = top_blob.row<unsigned short>(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    *outptr++ = ptr[in_x];
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outw * 2];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            linear_coeffs(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                unsigned short* outptr = top_blob.row<unsigned short>(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const unsigned short* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    *outptr++ = float32_to_bfloat16(bfloat16_to_float32(Sp[0]) * a0 + bfloat16_to_float32(Sp[1]) * a1);
                    alphap += 2;
                }
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outw * 4];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            cubic_coeffs(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                unsigned short* outptr = top_blob.row<unsigned short>(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const unsigned short* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    float a2 = alphap[2];
                    float a3 = alphap[3];
                    *outptr++ = float32_to_bfloat16(bfloat16_to_float32(Sp[-1]) * a0 + bfloat16_to_float32(Sp[0]) * a1 + bfloat16_to_float32(Sp[1]) * a2 + bfloat16_to_float32(Sp[2]) * a3);
                    alphap += 4;
                }
            }

            delete[] buf;
        }

        return 0;
    }

    if (outw == w && outh == h)
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
        if (resize_type == 1) // nearest
        {
            const float hs = output_height ? h / (float)outh : 1.f / height_scale;
            const float ws = output_width ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    int in_y = std::min((int)(y * hs), (h - 1));

                    const unsigned short* ptr = src.row<const unsigned short>(in_y);
                    unsigned short* outptr = dst.row<unsigned short>(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        uint16x4_t _p = vld1_u16(ptr + in_x * 4);
                        vst1_u16(outptr, _p);

                        outptr += 4;
                    }
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outh + outw * 2 + outh * 2];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
            float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

            linear_coeffs(w, outw, xofs, alpha, align_corner);
            linear_coeffs(h, outh, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bilinear_image_pack4_bf16s(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outh + outw * 4 + outh * 4];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
            float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

            cubic_coeffs(w, outw, xofs, alpha, align_corner);
            cubic_coeffs(h, outh, yofs, beta, align_corner);

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

    if (resize_type == 1) // nearest
    {
        const float hs = output_height ? h / (float)outh : 1.f / height_scale;
        const float ws = output_width ? w / (float)outw : 1.f / width_scale;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            for (int y = 0; y < outh; y++)
            {
                int in_y = std::min((int)(y * hs), (h - 1));

                const unsigned short* ptr = src.row<const unsigned short>(in_y);
                unsigned short* outptr = dst.row<unsigned short>(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    *outptr++ = ptr[in_x];
                }
            }
        }
    }

    if (resize_type == 2) // bilinear
    {
        int* buf = new int[outw + outh + outw * 2 + outh * 2];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
        float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

        linear_coeffs(w, outw, xofs, alpha, align_corner);
        linear_coeffs(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bilinear_image_bf16s(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    if (resize_type == 3) // bicubic
    {
        int* buf = new int[outw + outh + outw * 4 + outh * 4];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
        float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

        cubic_coeffs(w, outw, xofs, alpha, align_corner);
        cubic_coeffs(h, outh, yofs, beta, align_corner);

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
#endif // NCNN_BF16

} // namespace ncnn
