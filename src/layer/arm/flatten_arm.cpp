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

Flatten_arm::Flatten_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

    support_bf16_storage = true;
}

int Flatten_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);
#endif

    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);

    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

    int total = size * channels * elempack;

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        out_elempack = total % 4 == 0 ? 4 : 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (out_elempack == 1)
    {
        return Flatten::forward(bottom_blob, top_blob, opt);
    }

    if (dims == 2 && elempack == 1) // out_elempack == 4
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

    if (dims == 2)
    {
        if (elempack == 4) // out_elempack == 4
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = bottom_blob.row(i);
                float* outptr0 = (float*)top_blob + w * i * 4;
                float* outptr1 = (float*)top_blob + w * (i * 4 + 1);
                float* outptr2 = (float*)top_blob + w * (i * 4 + 2);
                float* outptr3 = (float*)top_blob + w * (i * 4 + 3);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < w; j += 4)
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
#endif
                for (; j < w; j++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }
        }
    }

    if (dims == 3)
    {
        if (elempack == 4) // out_elempack == 4
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr0 = (float*)top_blob + size * q * 4;
                float* outptr1 = (float*)top_blob + size * (q * 4 + 1);
                float* outptr2 = (float*)top_blob + size * (q * 4 + 2);
                float* outptr3 = (float*)top_blob + size * (q * 4 + 3);

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
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
#endif
                for (; i < size; i++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }
        }

        if (elempack == 1) // out_elempack == 4
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = (float*)top_blob + size * q;

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _v = vld1q_f32(ptr);
                    vst1q_f32(outptr, _v);
                    ptr += 4;
                    outptr += 4;
                }
#endif
                for (; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }
        }
    }

    return 0;
}

int Flatten_arm::forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

    int total = size * channels * elempack;

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        out_elempack = opt.use_fp16_arithmetic && total % 8 == 0 ? 8 : total % 4 == 0 ? 4 : 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (out_elempack == 1)
    {
        return Flatten::forward(bottom_blob, top_blob, opt);
    }

    if (dims == 2 && elempack == 1) // out_elempack == 4 || out_elempack == 8
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

    if (dims == 2)
    {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        if (elempack == 8) // out_elempack == 8
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const __fp16* ptr = bottom_blob.row<const __fp16>(i);
                __fp16* outptr0 = (__fp16*)top_blob + w * i * 8;
                __fp16* outptr1 = (__fp16*)top_blob + w * (i * 8 + 1);
                __fp16* outptr2 = (__fp16*)top_blob + w * (i * 8 + 2);
                __fp16* outptr3 = (__fp16*)top_blob + w * (i * 8 + 3);
                __fp16* outptr4 = (__fp16*)top_blob + w * (i * 8 + 4);
                __fp16* outptr5 = (__fp16*)top_blob + w * (i * 8 + 5);
                __fp16* outptr6 = (__fp16*)top_blob + w * (i * 8 + 6);
                __fp16* outptr7 = (__fp16*)top_blob + w * (i * 8 + 7);

                int j = 0;
                for (; j + 3 < w; j += 4)
                {
                    float16x8x4_t _v4 = vld4q_f16(ptr);
                    float16x8_t _v_01 = vuzp1q_f16(_v4.val[0], _v4.val[1]);
                    float16x8_t _v_23 = vuzp1q_f16(_v4.val[2], _v4.val[3]);
                    float16x8_t _v_45 = vuzp2q_f16(_v4.val[0], _v4.val[1]);
                    float16x8_t _v_67 = vuzp2q_f16(_v4.val[2], _v4.val[3]);
                    vst1_f16(outptr0, vget_low_f16(_v_01));
                    vst1_f16(outptr1, vget_high_f16(_v_01));
                    vst1_f16(outptr2, vget_low_f16(_v_23));
                    vst1_f16(outptr3, vget_high_f16(_v_23));
                    vst1_f16(outptr4, vget_low_f16(_v_45));
                    vst1_f16(outptr5, vget_high_f16(_v_45));
                    vst1_f16(outptr6, vget_low_f16(_v_67));
                    vst1_f16(outptr7, vget_high_f16(_v_67));

                    ptr += 32;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                    outptr4 += 4;
                    outptr5 += 4;
                    outptr6 += 4;
                    outptr7 += 4;
                }
                for (; j < w; j++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];
                    *outptr4++ = ptr[4];
                    *outptr5++ = ptr[5];
                    *outptr6++ = ptr[6];
                    *outptr7++ = ptr[7];

                    ptr += 8;
                }
            }
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        if (elempack == 4) // out_elempack == 4 || out_elempack == 8
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const unsigned short* ptr = bottom_blob.row<const unsigned short>(i);
                unsigned short* outptr0 = (unsigned short*)top_blob + w * i * 4;
                unsigned short* outptr1 = (unsigned short*)top_blob + w * (i * 4 + 1);
                unsigned short* outptr2 = (unsigned short*)top_blob + w * (i * 4 + 2);
                unsigned short* outptr3 = (unsigned short*)top_blob + w * (i * 4 + 3);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < w; j += 4)
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
#endif
                for (; j < w; j++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }
        }
    }

    if (dims == 3)
    {
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        if (elempack == 8) // out_elempack == 8
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const __fp16* ptr = bottom_blob.channel(q);
                __fp16* outptr0 = (__fp16*)top_blob + size * q * 8;
                __fp16* outptr1 = (__fp16*)top_blob + size * (q * 8 + 1);
                __fp16* outptr2 = (__fp16*)top_blob + size * (q * 8 + 2);
                __fp16* outptr3 = (__fp16*)top_blob + size * (q * 8 + 3);
                __fp16* outptr4 = (__fp16*)top_blob + size * (q * 8 + 4);
                __fp16* outptr5 = (__fp16*)top_blob + size * (q * 8 + 5);
                __fp16* outptr6 = (__fp16*)top_blob + size * (q * 8 + 6);
                __fp16* outptr7 = (__fp16*)top_blob + size * (q * 8 + 7);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    float16x8x4_t _v4 = vld4q_f16(ptr);
                    float16x8_t _v_01 = vuzp1q_f16(_v4.val[0], _v4.val[1]);
                    float16x8_t _v_23 = vuzp1q_f16(_v4.val[2], _v4.val[3]);
                    float16x8_t _v_45 = vuzp2q_f16(_v4.val[0], _v4.val[1]);
                    float16x8_t _v_67 = vuzp2q_f16(_v4.val[2], _v4.val[3]);
                    vst1_f16(outptr0, vget_low_f16(_v_01));
                    vst1_f16(outptr1, vget_high_f16(_v_01));
                    vst1_f16(outptr2, vget_low_f16(_v_23));
                    vst1_f16(outptr3, vget_high_f16(_v_23));
                    vst1_f16(outptr4, vget_low_f16(_v_45));
                    vst1_f16(outptr5, vget_high_f16(_v_45));
                    vst1_f16(outptr6, vget_low_f16(_v_67));
                    vst1_f16(outptr7, vget_high_f16(_v_67));

                    ptr += 32;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                    outptr4 += 4;
                    outptr5 += 4;
                    outptr6 += 4;
                    outptr7 += 4;
                }
                for (; i < size; i++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];
                    *outptr4++ = ptr[4];
                    *outptr5++ = ptr[5];
                    *outptr6++ = ptr[6];
                    *outptr7++ = ptr[7];

                    ptr += 8;
                }
            }
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        if (elempack == 4) // out_elempack == 4 || out_elempack == 8
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                unsigned short* outptr0 = (unsigned short*)top_blob + size * q * 4;
                unsigned short* outptr1 = (unsigned short*)top_blob + size * (q * 4 + 1);
                unsigned short* outptr2 = (unsigned short*)top_blob + size * (q * 4 + 2);
                unsigned short* outptr3 = (unsigned short*)top_blob + size * (q * 4 + 3);

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
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
#endif
                for (; i < size; i++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }
        }

        if (elempack == 1) // out_elempack == 4 || out_elempack == 8
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                unsigned short* outptr = (unsigned short*)top_blob + size * q;

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    uint16x4_t _v = vld1_u16(ptr);
                    vst1_u16(outptr, _v);
                    ptr += 4;
                    outptr += 4;
                }
#endif
                for (; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
