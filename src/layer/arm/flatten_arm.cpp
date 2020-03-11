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

#include "flatten_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(Flatten_arm)

Flatten_arm::Flatten_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON

    support_bf16_storage = true;
}

int Flatten_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elemsize / bottom_blob.elempack == 2u)
    {
        return forward_bf16s(bottom_blob, top_blob, opt);
    }

    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

#if __ARM_NEON
    if (opt.use_packing_layout)
    {

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

    int total = size * channels * elempack;

    int out_elempack = total % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (dims == 2 && elempack == 1)
    {
        top_blob = bottom_blob;
        top_blob.dims = 1;
        top_blob.w = total / out_elempack;
        top_blob.h = 1;
        top_blob.cstep = top_blob.w;
        top_blob.elemsize = out_elemsize;
        top_blob.elempack = out_elempack;
        return 0;
    }

    top_blob.create(total / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (dims == 2 && elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=0; i<h; i++)
        {
            const float* ptr = bottom_blob.row(i);
            float* outptr0 = (float*)top_blob + w * i*4;
            float* outptr1 = (float*)top_blob + w * (i*4 + 1);
            float* outptr2 = (float*)top_blob + w * (i*4 + 2);
            float* outptr3 = (float*)top_blob + w * (i*4 + 3);

            int j=0;
            for (; j+3<w; j+=4)
            {
                float32x4x4_t _v4 = vld4q_f32(ptr);
                vst1q_f32(outptr0, _v4.val[0]);
                vst1q_f32(outptr1, _v4.val[1]);
                vst1q_f32(outptr2, _v4.val[2]);
                vst1q_f32(outptr3, _v4.val[3]);

                ptr += 16;
                outptr0 += 4;
                outptr1 += 4;
                outptr2 += 4;
                outptr3 += 4;
            }
            for (; j<w; j++)
            {
                *outptr0++ = ptr[0];
                *outptr1++ = ptr[1];
                *outptr2++ = ptr[2];
                *outptr3++ = ptr[3];

                ptr += 4;
            }
        }

        return 0;
    }

    if (dims == 3 && elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr0 = (float*)top_blob + size * q*4;
            float* outptr1 = (float*)top_blob + size * (q*4 + 1);
            float* outptr2 = (float*)top_blob + size * (q*4 + 2);
            float* outptr3 = (float*)top_blob + size * (q*4 + 3);

            int i=0;
            for (; i+3<size; i+=4)
            {
                float32x4x4_t _v4 = vld4q_f32(ptr);
                vst1q_f32(outptr0, _v4.val[0]);
                vst1q_f32(outptr1, _v4.val[1]);
                vst1q_f32(outptr2, _v4.val[2]);
                vst1q_f32(outptr3, _v4.val[3]);

                ptr += 16;
                outptr0 += 4;
                outptr1 += 4;
                outptr2 += 4;
                outptr3 += 4;
            }
            for (; i<size; i++)
            {
                *outptr0++ = ptr[0];
                *outptr1++ = ptr[1];
                *outptr2++ = ptr[2];
                *outptr3++ = ptr[3];

                ptr += 4;
            }
        }

        return 0;
    }

    if (dims == 3 && elempack == 1 && out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = (float*)top_blob + size * q;

            int i=0;
            for (; i+3<size; i+=4)
            {
                float32x4_t _v = vld1q_f32(ptr);
                vst1q_f32(outptr, _v);
                ptr += 4;
                outptr += 4;
            }
            for (; i<size; i++)
            {
                *outptr++ = *ptr++;
            }
        }

        return 0;
    }

    } // opt.use_packing_layout
#endif // __ARM_NEON

    return Flatten::forward(bottom_blob, top_blob, opt);
}

int Flatten_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

#if __ARM_NEON
    if (opt.use_packing_layout)
    {

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

    int total = size * channels * elempack;

    int out_elempack = total % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (dims == 2 && elempack == 1)
    {
        top_blob = bottom_blob;
        top_blob.dims = 1;
        top_blob.w = total / out_elempack;
        top_blob.h = 1;
        top_blob.cstep = top_blob.w;
        top_blob.elemsize = out_elemsize;
        top_blob.elempack = out_elempack;
        return 0;
    }

    top_blob.create(total / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (dims == 2 && elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=0; i<h; i++)
        {
            const unsigned short* ptr = bottom_blob.row<const unsigned short>(i);
            unsigned short* outptr0 = (unsigned short*)top_blob + w * i*4;
            unsigned short* outptr1 = (unsigned short*)top_blob + w * (i*4 + 1);
            unsigned short* outptr2 = (unsigned short*)top_blob + w * (i*4 + 2);
            unsigned short* outptr3 = (unsigned short*)top_blob + w * (i*4 + 3);

            int j=0;
            for (; j+3<w; j+=4)
            {
                uint16x4x4_t _v4 = vld4_u16(ptr);
                vst1_u16(outptr0, _v4.val[0]);
                vst1_u16(outptr1, _v4.val[1]);
                vst1_u16(outptr2, _v4.val[2]);
                vst1_u16(outptr3, _v4.val[3]);

                ptr += 16;
                outptr0 += 4;
                outptr1 += 4;
                outptr2 += 4;
                outptr3 += 4;
            }
            for (; j<w; j++)
            {
                *outptr0++ = ptr[0];
                *outptr1++ = ptr[1];
                *outptr2++ = ptr[2];
                *outptr3++ = ptr[3];

                ptr += 4;
            }
        }

        return 0;
    }

    if (dims == 3 && elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            unsigned short* outptr0 = (unsigned short*)top_blob + size * q*4;
            unsigned short* outptr1 = (unsigned short*)top_blob + size * (q*4 + 1);
            unsigned short* outptr2 = (unsigned short*)top_blob + size * (q*4 + 2);
            unsigned short* outptr3 = (unsigned short*)top_blob + size * (q*4 + 3);

            int i=0;
            for (; i+3<size; i+=4)
            {
                uint16x4x4_t _v4 = vld4_u16(ptr);
                vst1_u16(outptr0, _v4.val[0]);
                vst1_u16(outptr1, _v4.val[1]);
                vst1_u16(outptr2, _v4.val[2]);
                vst1_u16(outptr3, _v4.val[3]);

                ptr += 16;
                outptr0 += 4;
                outptr1 += 4;
                outptr2 += 4;
                outptr3 += 4;
            }
            for (; i<size; i++)
            {
                *outptr0++ = ptr[0];
                *outptr1++ = ptr[1];
                *outptr2++ = ptr[2];
                *outptr3++ = ptr[3];

                ptr += 4;
            }
        }

        return 0;
    }

    if (dims == 3 && elempack == 1 && out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            unsigned short* outptr = (unsigned short*)top_blob + size * q;

            int i=0;
            for (; i+3<size; i+=4)
            {
                uint16x4_t _v = vld1_u16(ptr);
                vst1_u16(outptr, _v);
                ptr += 4;
                outptr += 4;
            }
            for (; i<size; i++)
            {
                *outptr++ = *ptr++;
            }
        }

        return 0;
    }

    } // opt.use_packing_layout
#endif // __ARM_NEON

    return Flatten::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
