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

#include "pixelshuffle_arm.h"

#include "layer_type.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

PixelShuffle_arm::PixelShuffle_arm()
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

int PixelShuffle_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = w * upscale_factor;
    int outh = h * upscale_factor;
    int outc = channels * elempack / (upscale_factor * upscale_factor);

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        out_elempack = outc % 4 == 0 ? 4 : 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (upscale_factor != 2 || mode != 0)
    {
        Option opt_pack = opt;
        opt_pack.blob_allocator = opt.workspace_allocator;

        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);

        return PixelShuffle::forward(bottom_blob_unpacked, top_blob, opt);
    }

    top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __ARM_NEON
    if (elempack == 4 && out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc / out_elempack; p++)
        {
            Mat m = top_blob.channel(p);

            const float* sptr0 = bottom_blob.channel(p * 4);
            const float* sptr1 = bottom_blob.channel(p * 4 + 1);
            const float* sptr2 = bottom_blob.channel(p * 4 + 2);
            const float* sptr3 = bottom_blob.channel(p * 4 + 3);

            for (int i = 0; i < h; i++)
            {
                float* outptr0 = m.row(i * 2);
                float* outptr1 = m.row(i * 2 + 1);

                int j = 0;
                for (; j + 1 < w; j += 2)
                {
                    float32x4_t _p00 = vld1q_f32(sptr0);
                    float32x4_t _p10 = vld1q_f32(sptr1);
                    float32x4_t _p20 = vld1q_f32(sptr2);
                    float32x4_t _p30 = vld1q_f32(sptr3);

                    float32x4_t _p01 = vld1q_f32(sptr0 + 4);
                    float32x4_t _p11 = vld1q_f32(sptr1 + 4);
                    float32x4_t _p21 = vld1q_f32(sptr2 + 4);
                    float32x4_t _p31 = vld1q_f32(sptr3 + 4);

                    float32x4x4_t _s0;
                    _s0.val[0] = vcombine_f32(vget_low_f32(_p00), vget_low_f32(_p01));
                    _s0.val[1] = vcombine_f32(vget_low_f32(_p10), vget_low_f32(_p11));
                    _s0.val[2] = vcombine_f32(vget_low_f32(_p20), vget_low_f32(_p21));
                    _s0.val[3] = vcombine_f32(vget_low_f32(_p30), vget_low_f32(_p31));

                    float32x4x4_t _s1;
                    _s1.val[0] = vcombine_f32(vget_high_f32(_p00), vget_high_f32(_p01));
                    _s1.val[1] = vcombine_f32(vget_high_f32(_p10), vget_high_f32(_p11));
                    _s1.val[2] = vcombine_f32(vget_high_f32(_p20), vget_high_f32(_p21));
                    _s1.val[3] = vcombine_f32(vget_high_f32(_p30), vget_high_f32(_p31));

                    vst4q_f32(outptr0, _s0);
                    vst4q_f32(outptr1, _s1);

                    sptr0 += 8;
                    sptr1 += 8;
                    sptr2 += 8;
                    sptr3 += 8;
                    outptr0 += 16;
                    outptr1 += 16;
                }
                for (; j < w; j++)
                {
                    outptr0[0] = sptr0[0];
                    outptr0[1] = sptr1[0];
                    outptr0[2] = sptr2[0];
                    outptr0[3] = sptr3[0];

                    outptr0[4] = sptr0[1];
                    outptr0[5] = sptr1[1];
                    outptr0[6] = sptr2[1];
                    outptr0[7] = sptr3[1];

                    outptr1[0] = sptr0[2];
                    outptr1[1] = sptr1[2];
                    outptr1[2] = sptr2[2];
                    outptr1[3] = sptr3[2];

                    outptr1[4] = sptr0[3];
                    outptr1[5] = sptr1[3];
                    outptr1[6] = sptr2[3];
                    outptr1[7] = sptr3[3];

                    sptr0 += 4;
                    sptr1 += 4;
                    sptr2 += 4;
                    sptr3 += 4;
                    outptr0 += 8;
                    outptr1 += 8;
                }
            }
        }

        return 0;
    }

    if (elempack == 4 && out_elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc / out_elempack; p++)
        {
            Mat m = top_blob.channel(p);

            const float* sptr = bottom_blob.channel(p);

            for (int i = 0; i < h; i++)
            {
                float* outptr0 = m.row(i * 2);
                float* outptr1 = m.row(i * 2 + 1);

                int j = 0;
                for (; j + 1 < w; j += 2)
                {
                    float32x4_t _p0 = vld1q_f32(sptr);
                    float32x4_t _p1 = vld1q_f32(sptr + 4);

                    float32x4_t _s0 = vcombine_f32(vget_low_f32(_p0), vget_low_f32(_p1));
                    float32x4_t _s1 = vcombine_f32(vget_high_f32(_p0), vget_high_f32(_p1));

                    vst1q_f32(outptr0, _s0);
                    vst1q_f32(outptr1, _s1);

                    sptr += 8;
                    outptr0 += 4;
                    outptr1 += 4;
                }
                for (; j < w; j++)
                {
                    outptr0[0] = sptr[0];
                    outptr0[1] = sptr[1];
                    outptr1[0] = sptr[2];
                    outptr1[1] = sptr[3];

                    sptr += 4;
                    outptr0 += 2;
                    outptr1 += 2;
                }
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    return PixelShuffle::forward(bottom_blob, top_blob, opt);
}

int PixelShuffle_arm::forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = w * upscale_factor;
    int outh = h * upscale_factor;
    int outc = channels * elempack / (upscale_factor * upscale_factor);

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        out_elempack = opt.use_fp16_arithmetic && outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (upscale_factor != 2 || mode != 0)
    {
        Option opt_pack = opt;
        opt_pack.blob_allocator = opt.workspace_allocator;

        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);

        top_blob.create(outw, outh, outc, 2u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            Mat m = top_blob.channel(p);

            for (int sh = 0; sh < upscale_factor; sh++)
            {
                for (int sw = 0; sw < upscale_factor; sw++)
                {
                    int q;
                    if (mode == 0)
                        q = p * upscale_factor * upscale_factor + sh * upscale_factor + sw;
                    else // if (mode == 1)
                        q = (sh * upscale_factor + sw) * outc + p;

                    const unsigned short* sptr = bottom_blob_unpacked.channel(q);

                    for (int i = 0; i < h; i++)
                    {
                        unsigned short* outptr = m.row<unsigned short>(i * upscale_factor + sh) + sw;
                        for (int j = 0; j < w; j++)
                        {
                            outptr[0] = sptr[0];

                            sptr++;
                            outptr += upscale_factor;
                        }
                    }
                }
            }
        }

        return 0;
    }

    top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __ARM_NEON
    if (elempack == 8 && out_elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc / out_elempack; p++)
        {
            Mat m = top_blob.channel(p);

            const unsigned short* sptr0 = bottom_blob.channel(p * 4);
            const unsigned short* sptr1 = bottom_blob.channel(p * 4 + 1);
            const unsigned short* sptr2 = bottom_blob.channel(p * 4 + 2);
            const unsigned short* sptr3 = bottom_blob.channel(p * 4 + 3);

            for (int i = 0; i < h; i++)
            {
                unsigned short* outptr0 = m.row<unsigned short>(i * 2);
                unsigned short* outptr1 = m.row<unsigned short>(i * 2 + 1);

                int j = 0;
                for (; j + 3 < w; j += 4)
                {
                    uint16x8x4_t _p0 = vld4q_u16(sptr0);
                    uint16x8x4_t _p1 = vld4q_u16(sptr1);
                    uint16x8x4_t _p2 = vld4q_u16(sptr2);
                    uint16x8x4_t _p3 = vld4q_u16(sptr3);

                    uint32x4x2_t _s04_0 = vzipq_u32(vreinterpretq_u32_u16(_p0.val[0]), vreinterpretq_u32_u16(_p1.val[0]));
                    uint32x4x2_t _s15_0 = vzipq_u32(vreinterpretq_u32_u16(_p0.val[1]), vreinterpretq_u32_u16(_p1.val[1]));
                    uint32x4x2_t _s26_0 = vzipq_u32(vreinterpretq_u32_u16(_p0.val[2]), vreinterpretq_u32_u16(_p1.val[2]));
                    uint32x4x2_t _s37_0 = vzipq_u32(vreinterpretq_u32_u16(_p0.val[3]), vreinterpretq_u32_u16(_p1.val[3]));
                    uint32x4x2_t _s04_1 = vzipq_u32(vreinterpretq_u32_u16(_p2.val[0]), vreinterpretq_u32_u16(_p3.val[0]));
                    uint32x4x2_t _s15_1 = vzipq_u32(vreinterpretq_u32_u16(_p2.val[1]), vreinterpretq_u32_u16(_p3.val[1]));
                    uint32x4x2_t _s26_1 = vzipq_u32(vreinterpretq_u32_u16(_p2.val[2]), vreinterpretq_u32_u16(_p3.val[2]));
                    uint32x4x2_t _s37_1 = vzipq_u32(vreinterpretq_u32_u16(_p2.val[3]), vreinterpretq_u32_u16(_p3.val[3]));

                    uint16x8_t _s0_0 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_s04_0.val[0]), vget_low_u32(_s04_1.val[0])));
                    uint16x8_t _s0_1 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_s15_0.val[0]), vget_low_u32(_s15_1.val[0])));
                    uint16x8_t _s0_2 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_s04_0.val[0]), vget_high_u32(_s04_1.val[0])));
                    uint16x8_t _s0_3 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_s15_0.val[0]), vget_high_u32(_s15_1.val[0])));
                    uint16x8_t _s0_4 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_s04_0.val[1]), vget_low_u32(_s04_1.val[1])));
                    uint16x8_t _s0_5 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_s15_0.val[1]), vget_low_u32(_s15_1.val[1])));
                    uint16x8_t _s0_6 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_s04_0.val[1]), vget_high_u32(_s04_1.val[1])));
                    uint16x8_t _s0_7 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_s15_0.val[1]), vget_high_u32(_s15_1.val[1])));
                    uint16x8_t _s1_0 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_s26_0.val[0]), vget_low_u32(_s26_1.val[0])));
                    uint16x8_t _s1_1 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_s37_0.val[0]), vget_low_u32(_s37_1.val[0])));
                    uint16x8_t _s1_2 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_s26_0.val[0]), vget_high_u32(_s26_1.val[0])));
                    uint16x8_t _s1_3 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_s37_0.val[0]), vget_high_u32(_s37_1.val[0])));
                    uint16x8_t _s1_4 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_s26_0.val[1]), vget_low_u32(_s26_1.val[1])));
                    uint16x8_t _s1_5 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_s37_0.val[1]), vget_low_u32(_s37_1.val[1])));
                    uint16x8_t _s1_6 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_s26_0.val[1]), vget_high_u32(_s26_1.val[1])));
                    uint16x8_t _s1_7 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_s37_0.val[1]), vget_high_u32(_s37_1.val[1])));

                    vst1q_u16(outptr0, _s0_0);
                    vst1q_u16(outptr0 + 8, _s0_1);
                    vst1q_u16(outptr0 + 16, _s0_2);
                    vst1q_u16(outptr0 + 24, _s0_3);
                    vst1q_u16(outptr0 + 32, _s0_4);
                    vst1q_u16(outptr0 + 40, _s0_5);
                    vst1q_u16(outptr0 + 48, _s0_6);
                    vst1q_u16(outptr0 + 56, _s0_7);
                    vst1q_u16(outptr1, _s1_0);
                    vst1q_u16(outptr1 + 8, _s1_1);
                    vst1q_u16(outptr1 + 16, _s1_2);
                    vst1q_u16(outptr1 + 24, _s1_3);
                    vst1q_u16(outptr1 + 32, _s1_4);
                    vst1q_u16(outptr1 + 40, _s1_5);
                    vst1q_u16(outptr1 + 48, _s1_6);
                    vst1q_u16(outptr1 + 56, _s1_7);

                    sptr0 += 32;
                    sptr1 += 32;
                    sptr2 += 32;
                    sptr3 += 32;
                    outptr0 += 64;
                    outptr1 += 64;
                }
                for (; j < w; j++)
                {
                    outptr0[0] = sptr0[0];
                    outptr0[1] = sptr0[4];
                    outptr0[2] = sptr1[0];
                    outptr0[3] = sptr1[4];
                    outptr0[4] = sptr2[0];
                    outptr0[5] = sptr2[4];
                    outptr0[6] = sptr3[0];
                    outptr0[7] = sptr3[4];

                    outptr0[8] = sptr0[1];
                    outptr0[9] = sptr0[5];
                    outptr0[10] = sptr1[1];
                    outptr0[11] = sptr1[5];
                    outptr0[12] = sptr2[1];
                    outptr0[13] = sptr2[5];
                    outptr0[14] = sptr3[1];
                    outptr0[15] = sptr3[5];

                    outptr1[0] = sptr0[2];
                    outptr1[1] = sptr0[6];
                    outptr1[2] = sptr1[2];
                    outptr1[3] = sptr1[6];
                    outptr1[4] = sptr2[2];
                    outptr1[5] = sptr2[6];
                    outptr1[6] = sptr3[2];
                    outptr1[7] = sptr3[6];

                    outptr1[8] = sptr0[3];
                    outptr1[9] = sptr0[7];
                    outptr1[10] = sptr1[3];
                    outptr1[11] = sptr1[7];
                    outptr1[12] = sptr2[3];
                    outptr1[13] = sptr2[7];
                    outptr1[14] = sptr3[3];
                    outptr1[15] = sptr3[7];

                    sptr0 += 8;
                    sptr1 += 8;
                    sptr2 += 8;
                    sptr3 += 8;
                    outptr0 += 16;
                    outptr1 += 16;
                }
            }
        }

        return 0;
    }

    if (elempack == 8 && out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc / out_elempack; p++)
        {
            Mat m = top_blob.channel(p);

            const unsigned short* sptr0 = bottom_blob.channel(p * 2);
            const unsigned short* sptr1 = bottom_blob.channel(p * 2 + 1);

            for (int i = 0; i < h; i++)
            {
                unsigned short* outptr0 = m.row<unsigned short>(i * 2);
                unsigned short* outptr1 = m.row<unsigned short>(i * 2 + 1);

                int j = 0;
                for (; j + 3 < w; j += 4)
                {
                    uint16x8x4_t _p0 = vld4q_u16(sptr0);
                    uint16x8x4_t _p1 = vld4q_u16(sptr1);

                    uint32x4x2_t _s04 = vzipq_u32(vreinterpretq_u32_u16(_p0.val[0]), vreinterpretq_u32_u16(_p1.val[0]));
                    uint32x4x2_t _s15 = vzipq_u32(vreinterpretq_u32_u16(_p0.val[1]), vreinterpretq_u32_u16(_p1.val[1]));
                    uint32x4x2_t _s26 = vzipq_u32(vreinterpretq_u32_u16(_p0.val[2]), vreinterpretq_u32_u16(_p1.val[2]));
                    uint32x4x2_t _s37 = vzipq_u32(vreinterpretq_u32_u16(_p0.val[3]), vreinterpretq_u32_u16(_p1.val[3]));

                    uint16x8_t _s0_0 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_s04.val[0]), vget_low_u32(_s15.val[0])));
                    uint16x8_t _s0_1 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_s04.val[0]), vget_high_u32(_s15.val[0])));
                    uint16x8_t _s0_2 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_s04.val[1]), vget_low_u32(_s15.val[1])));
                    uint16x8_t _s0_3 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_s04.val[1]), vget_high_u32(_s15.val[1])));
                    uint16x8_t _s1_0 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_s26.val[0]), vget_low_u32(_s37.val[0])));
                    uint16x8_t _s1_1 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_s26.val[0]), vget_high_u32(_s37.val[0])));
                    uint16x8_t _s1_2 = vreinterpretq_u16_u32(vcombine_u32(vget_low_u32(_s26.val[1]), vget_low_u32(_s37.val[1])));
                    uint16x8_t _s1_3 = vreinterpretq_u16_u32(vcombine_u32(vget_high_u32(_s26.val[1]), vget_high_u32(_s37.val[1])));

                    vst1q_u16(outptr0, _s0_0);
                    vst1q_u16(outptr0 + 8, _s0_1);
                    vst1q_u16(outptr0 + 16, _s0_2);
                    vst1q_u16(outptr0 + 24, _s0_3);
                    vst1q_u16(outptr1, _s1_0);
                    vst1q_u16(outptr1 + 8, _s1_1);
                    vst1q_u16(outptr1 + 16, _s1_2);
                    vst1q_u16(outptr1 + 24, _s1_3);

                    sptr0 += 32;
                    sptr1 += 32;
                    outptr0 += 32;
                    outptr1 += 32;
                }
                for (; j < w; j++)
                {
                    outptr0[0] = sptr0[0];
                    outptr0[1] = sptr0[4];
                    outptr0[2] = sptr1[0];
                    outptr0[3] = sptr1[4];

                    outptr0[4] = sptr0[1];
                    outptr0[5] = sptr0[5];
                    outptr0[6] = sptr1[1];
                    outptr0[7] = sptr1[5];

                    outptr1[0] = sptr0[2];
                    outptr1[1] = sptr0[6];
                    outptr1[2] = sptr1[2];
                    outptr1[3] = sptr1[6];

                    outptr1[4] = sptr0[3];
                    outptr1[5] = sptr0[7];
                    outptr1[6] = sptr1[3];
                    outptr1[7] = sptr1[7];

                    sptr0 += 8;
                    sptr1 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                }
            }
        }

        return 0;
    }

    if (elempack == 8 && out_elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc / out_elempack / 2; p++)
        {
            Mat m0 = top_blob.channel(p * 2);
            Mat m1 = top_blob.channel(p * 2 + 1);

            const unsigned short* sptr = bottom_blob.channel(p);

            for (int i = 0; i < h; i++)
            {
                unsigned short* outptr00 = m0.row<unsigned short>(i * 2);
                unsigned short* outptr01 = m0.row<unsigned short>(i * 2 + 1);
                unsigned short* outptr10 = m1.row<unsigned short>(i * 2);
                unsigned short* outptr11 = m1.row<unsigned short>(i * 2 + 1);

                int j = 0;
                for (; j + 3 < w; j += 4)
                {
                    uint32x4x4_t _p = vld4q_u32((unsigned int*)sptr);

                    uint16x8_t _s0 = vreinterpretq_u16_u32(_p.val[0]);
                    uint16x8_t _s1 = vreinterpretq_u16_u32(_p.val[1]);
                    uint16x8_t _s2 = vreinterpretq_u16_u32(_p.val[2]);
                    uint16x8_t _s3 = vreinterpretq_u16_u32(_p.val[3]);

                    vst1q_u16(outptr00, _s0);
                    vst1q_u16(outptr01, _s1);
                    vst1q_u16(outptr10, _s2);
                    vst1q_u16(outptr11, _s3);

                    sptr += 32;
                    outptr00 += 8;
                    outptr01 += 8;
                    outptr10 += 8;
                    outptr11 += 8;
                }
                for (; j < w; j++)
                {
                    outptr00[0] = sptr[0];
                    outptr00[1] = sptr[1];
                    outptr01[0] = sptr[2];
                    outptr01[1] = sptr[3];

                    outptr10[0] = sptr[4];
                    outptr10[1] = sptr[5];
                    outptr11[0] = sptr[6];
                    outptr11[1] = sptr[7];

                    sptr += 8;
                    outptr00 += 2;
                    outptr01 += 2;
                    outptr10 += 2;
                    outptr11 += 2;
                }
            }
        }

        return 0;
    }

    if (elempack == 4 && out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc / out_elempack; p++)
        {
            Mat m = top_blob.channel(p);

            const unsigned short* sptr0 = bottom_blob.channel(p * 4);
            const unsigned short* sptr1 = bottom_blob.channel(p * 4 + 1);
            const unsigned short* sptr2 = bottom_blob.channel(p * 4 + 2);
            const unsigned short* sptr3 = bottom_blob.channel(p * 4 + 3);

            for (int i = 0; i < h; i++)
            {
                unsigned short* outptr0 = m.row<unsigned short>(i * 2);
                unsigned short* outptr1 = m.row<unsigned short>(i * 2 + 1);

                int j = 0;
                for (; j + 3 < w; j += 4)
                {
                    uint16x8_t _p00 = vld1q_u16(sptr0);
                    uint16x8_t _p10 = vld1q_u16(sptr1);
                    uint16x8_t _p20 = vld1q_u16(sptr2);
                    uint16x8_t _p30 = vld1q_u16(sptr3);
                    uint16x8_t _p01 = vld1q_u16(sptr0 + 8);
                    uint16x8_t _p11 = vld1q_u16(sptr1 + 8);
                    uint16x8_t _p21 = vld1q_u16(sptr2 + 8);
                    uint16x8_t _p31 = vld1q_u16(sptr3 + 8);

                    uint32x4x2_t _p0 = vuzpq_u32(vreinterpretq_u32_u16(_p00), vreinterpretq_u32_u16(_p01));
                    uint32x4x2_t _p1 = vuzpq_u32(vreinterpretq_u32_u16(_p10), vreinterpretq_u32_u16(_p11));
                    uint32x4x2_t _p2 = vuzpq_u32(vreinterpretq_u32_u16(_p20), vreinterpretq_u32_u16(_p21));
                    uint32x4x2_t _p3 = vuzpq_u32(vreinterpretq_u32_u16(_p30), vreinterpretq_u32_u16(_p31));

                    uint16x8x4_t _s0;
                    _s0.val[0] = vreinterpretq_u16_u32(_p0.val[0]);
                    _s0.val[1] = vreinterpretq_u16_u32(_p1.val[0]);
                    _s0.val[2] = vreinterpretq_u16_u32(_p2.val[0]);
                    _s0.val[3] = vreinterpretq_u16_u32(_p3.val[0]);

                    uint16x8x4_t _s1;
                    _s1.val[0] = vreinterpretq_u16_u32(_p0.val[1]);
                    _s1.val[1] = vreinterpretq_u16_u32(_p1.val[1]);
                    _s1.val[2] = vreinterpretq_u16_u32(_p2.val[1]);
                    _s1.val[3] = vreinterpretq_u16_u32(_p3.val[1]);

                    vst4q_u16(outptr0, _s0);
                    vst4q_u16(outptr1, _s1);

                    sptr0 += 16;
                    sptr1 += 16;
                    sptr2 += 16;
                    sptr3 += 16;
                    outptr0 += 32;
                    outptr1 += 32;
                }
                for (; j < w; j++)
                {
                    outptr0[0] = sptr0[0];
                    outptr0[1] = sptr1[0];
                    outptr0[2] = sptr2[0];
                    outptr0[3] = sptr3[0];

                    outptr0[4] = sptr0[1];
                    outptr0[5] = sptr1[1];
                    outptr0[6] = sptr2[1];
                    outptr0[7] = sptr3[1];

                    outptr1[0] = sptr0[2];
                    outptr1[1] = sptr1[2];
                    outptr1[2] = sptr2[2];
                    outptr1[3] = sptr3[2];

                    outptr1[4] = sptr0[3];
                    outptr1[5] = sptr1[3];
                    outptr1[6] = sptr2[3];
                    outptr1[7] = sptr3[3];

                    sptr0 += 4;
                    sptr1 += 4;
                    sptr2 += 4;
                    sptr3 += 4;
                    outptr0 += 8;
                    outptr1 += 8;
                }
            }
        }

        return 0;
    }

    if (elempack == 4 && out_elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc / out_elempack; p++)
        {
            Mat m = top_blob.channel(p);

            const unsigned short* sptr = bottom_blob.channel(p);

            for (int i = 0; i < h; i++)
            {
                unsigned short* outptr0 = m.row<unsigned short>(i * 2);
                unsigned short* outptr1 = m.row<unsigned short>(i * 2 + 1);

                int j = 0;
                for (; j + 3 < w; j += 4)
                {
                    uint16x8_t _p0 = vld1q_u16(sptr);
                    uint16x8_t _p1 = vld1q_u16(sptr + 8);

                    uint32x4x2_t _s01 = vuzpq_u32(vreinterpretq_u32_u16(_p0), vreinterpretq_u32_u16(_p1));

                    uint16x8_t _s0 = vreinterpretq_u16_u32(_s01.val[0]);
                    uint16x8_t _s1 = vreinterpretq_u16_u32(_s01.val[1]);

                    vst1q_u16(outptr0, _s0);
                    vst1q_u16(outptr1, _s1);

                    sptr += 16;
                    outptr0 += 8;
                    outptr1 += 8;
                }
                for (; j < w; j++)
                {
                    outptr0[0] = sptr[0];
                    outptr0[1] = sptr[1];
                    outptr1[0] = sptr[2];
                    outptr1[1] = sptr[3];

                    sptr += 4;
                    outptr0 += 2;
                    outptr1 += 2;
                }
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outc; p++)
    {
        Mat m = top_blob.channel(p);

        for (int sh = 0; sh < upscale_factor; sh++)
        {
            for (int sw = 0; sw < upscale_factor; sw++)
            {
                int q = p * upscale_factor * upscale_factor + sh * upscale_factor + sw;

                const unsigned short* sptr = bottom_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    unsigned short* outptr = m.row<unsigned short>(i * upscale_factor + sh) + sw;
                    for (int j = 0; j < w; j++)
                    {
                        outptr[0] = sptr[0];

                        sptr++;
                        outptr += upscale_factor;
                    }
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
