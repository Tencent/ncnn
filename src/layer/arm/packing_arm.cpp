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

#include "packing_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(Packing_arm)

Packing_arm::Packing_arm()
{
    support_packing = true;

    support_bf16_storage = true;
}

int Packing_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    bool elemtype_is_bf16 = (elemsize == 2u && elempack == 1) || (elemsize == 8u && elempack == 4);
    bool elemtype_is_fp32 = (elemsize == 4u && elempack == 1) || (elemsize == 16u && elempack == 4);

    if (opt.use_bf16_storage && elemtype_is_bf16)
        return forward_bf16s(bottom_blob, top_blob, opt);

    if (use_padding)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    if (!elemtype_is_fp32)
    {
        // non-fp32 type
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    if (elempack == out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    bool pack1to4 = elempack == 1 && out_elempack == 4;
    bool pack4to1 = elempack == 4 && out_elempack == 1;

    if (!pack1to4 && !pack4to1)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    if (!use_padding)
    {
        // identity if use_padding not allowed
        if (dims == 1 && w * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 3 && channels * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    if (dims == 1)
    {
        top_blob = bottom_blob;
        top_blob.w = w * elempack / out_elempack;
        top_blob.cstep = w * elempack / out_elempack;
        top_blob.elemsize = elemsize / elempack * out_elempack;
        top_blob.elempack = out_elempack;
        return 0;
    }

    if (dims == 2)
    {
        int outh = h * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<outh; i++)
            {
                const float* r0 = bottom_blob.row(i*4);
                const float* r1 = bottom_blob.row(i*4+1);
                const float* r2 = bottom_blob.row(i*4+2);
                const float* r3 = bottom_blob.row(i*4+3);

                float* outptr = top_blob.row(i);

#if __ARM_NEON
                int nn = w >> 2;
                int remain = w & 3;
#else
                int remain = w;
#endif

#if __ARM_NEON
                for (; nn>0; nn--)
                {
                    float32x4x4_t _p;
                    _p.val[0] = vld1q_f32(r0);
                    _p.val[1] = vld1q_f32(r1);
                    _p.val[2] = vld1q_f32(r2);
                    _p.val[3] = vld1q_f32(r3);
                    vst4q_f32(outptr, _p);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif
                for (; remain>0; remain--)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<h; i++)
            {
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i*4);
                float* outptr1 = top_blob.row(i*4+1);
                float* outptr2 = top_blob.row(i*4+2);
                float* outptr3 = top_blob.row(i*4+3);

#if __ARM_NEON
                int nn = w >> 2;
                int remain = w & 3;
#else
                int remain = w;
#endif

#if __ARM_NEON
                for (; nn>0; nn--)
                {
                    float32x4x4_t _p = vld4q_f32(r0);
                    vst1q_f32(outptr0, _p.val[0]);
                    vst1q_f32(outptr1, _p.val[1]);
                    vst1q_f32(outptr2, _p.val[2]);
                    vst1q_f32(outptr3, _p.val[3]);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
#endif
                for (; remain>0; remain--)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }

        return 0;
    }

    if (dims == 3)
    {
        int size = w * h;
        int outc = channels * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<outc; q++)
            {
                const float* r0 = bottom_blob.channel(q*4);
                const float* r1 = bottom_blob.channel(q*4+1);
                const float* r2 = bottom_blob.channel(q*4+2);
                const float* r3 = bottom_blob.channel(q*4+3);

                float* outptr = top_blob.channel(q);

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size & 3;
#else
                int remain = size;
#endif

#if __ARM_NEON
                for (; nn>0; nn--)
                {
                    float32x4x4_t _p;
                    _p.val[0] = vld1q_f32(r0);
                    _p.val[1] = vld1q_f32(r1);
                    _p.val[2] = vld1q_f32(r2);
                    _p.val[3] = vld1q_f32(r3);
                    vst4q_f32(outptr, _p);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif
                for (; remain>0; remain--)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q*4);
                float* outptr1 = top_blob.channel(q*4+1);
                float* outptr2 = top_blob.channel(q*4+2);
                float* outptr3 = top_blob.channel(q*4+3);

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size & 3;
#else
                int remain = size;
#endif

#if __ARM_NEON
                for (; nn>0; nn--)
                {
                    float32x4x4_t _p = vld4q_f32(r0);
                    vst1q_f32(outptr0, _p.val[0]);
                    vst1q_f32(outptr1, _p.val[1]);
                    vst1q_f32(outptr2, _p.val[2]);
                    vst1q_f32(outptr3, _p.val[3]);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
#endif
                for (; remain>0; remain--)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }

        return 0;
    }

    return 0;
}

int Packing_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (use_padding)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (elempack == out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    bool pack1to4 = elempack == 1 && out_elempack == 4;
    bool pack4to1 = elempack == 4 && out_elempack == 1;

    if (!pack1to4 && !pack4to1)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    if (!use_padding)
    {
        // identity if use_padding not allowed
        if (dims == 1 && w * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 3 && channels * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    if (dims == 1)
    {
        top_blob = bottom_blob;
        top_blob.w = w * elempack / out_elempack;
        top_blob.cstep = w * elempack / out_elempack;
        top_blob.elemsize = elemsize / elempack * out_elempack;
        top_blob.elempack = out_elempack;
        return 0;
    }

    if (dims == 2)
    {
        int outh = h * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<outh; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i*4);
                const unsigned short* r1 = bottom_blob.row<const unsigned short>(i*4+1);
                const unsigned short* r2 = bottom_blob.row<const unsigned short>(i*4+2);
                const unsigned short* r3 = bottom_blob.row<const unsigned short>(i*4+3);

                unsigned short* outptr = top_blob.row<unsigned short>(i);

#if __ARM_NEON
                int nn = w >> 2;
                int remain = w & 3;
#else
                int remain = w;
#endif

#if __ARM_NEON
                for (; nn>0; nn--)
                {
                    uint16x4x4_t _p;
                    _p.val[0] = vld1_u16(r0);
                    _p.val[1] = vld1_u16(r1);
                    _p.val[2] = vld1_u16(r2);
                    _p.val[3] = vld1_u16(r3);
                    vst4_u16(outptr, _p);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif
                for (; remain>0; remain--)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<h; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                unsigned short* outptr0 = top_blob.row<unsigned short>(i*4);
                unsigned short* outptr1 = top_blob.row<unsigned short>(i*4+1);
                unsigned short* outptr2 = top_blob.row<unsigned short>(i*4+2);
                unsigned short* outptr3 = top_blob.row<unsigned short>(i*4+3);

#if __ARM_NEON
                int nn = w >> 2;
                int remain = w & 3;
#else
                int remain = w;
#endif

#if __ARM_NEON
                for (; nn>0; nn--)
                {
                    uint16x4x4_t _p = vld4_u16(r0);
                    vst1_u16(outptr0, _p.val[0]);
                    vst1_u16(outptr1, _p.val[1]);
                    vst1_u16(outptr2, _p.val[2]);
                    vst1_u16(outptr3, _p.val[3]);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
#endif
                for (; remain>0; remain--)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }

        return 0;
    }

    if (dims == 3)
    {
        int size = w * h;
        int outc = channels * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<outc; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q*4);
                const unsigned short* r1 = bottom_blob.channel(q*4+1);
                const unsigned short* r2 = bottom_blob.channel(q*4+2);
                const unsigned short* r3 = bottom_blob.channel(q*4+3);

                unsigned short* outptr = top_blob.channel(q);

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size & 3;
#else
                int remain = size;
#endif

#if __ARM_NEON
                for (; nn>0; nn--)
                {
                    uint16x4x4_t _p;
                    _p.val[0] = vld1_u16(r0);
                    _p.val[1] = vld1_u16(r1);
                    _p.val[2] = vld1_u16(r2);
                    _p.val[3] = vld1_u16(r3);
                    vst4_u16(outptr, _p);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif
                for (; remain>0; remain--)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q);

                unsigned short* outptr0 = top_blob.channel(q*4);
                unsigned short* outptr1 = top_blob.channel(q*4+1);
                unsigned short* outptr2 = top_blob.channel(q*4+2);
                unsigned short* outptr3 = top_blob.channel(q*4+3);

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size & 3;
#else
                int remain = size;
#endif

#if __ARM_NEON
                for (; nn>0; nn--)
                {
                    uint16x4x4_t _p = vld4_u16(r0);
                    vst1_u16(outptr0, _p.val[0]);
                    vst1_u16(outptr1, _p.val[1]);
                    vst1_u16(outptr2, _p.val[2]);
                    vst1_u16(outptr3, _p.val[3]);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
#endif
                for (; remain>0; remain--)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
