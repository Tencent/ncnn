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

#include "padding_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(Padding_arm)

Padding_arm::Padding_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

#if __ARM_NEON
static void padding_constant_pack4_neon(const Mat& src, Mat& dst, int top, int bottom, int left, int right, float v)
{
    float32x4_t _v = vdupq_n_f32(v);

    const float* ptr = src;
    float* outptr = dst;

    // fill top
    for (int i = 0; i < top * dst.w; i++)
    {
        vst1q_f32(outptr, _v);
        outptr += 4;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            vst1q_f32(outptr, _v);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            float32x4_t _p = vld1q_f32(ptr);
            vst1q_f32(outptr, _p);
            ptr += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            vst1q_f32(outptr, _v);
            outptr += 4;
        }
    }
    // fill bottom
    for (int i = 0; i < bottom * dst.w; i++)
    {
        vst1q_f32(outptr, _v);
        outptr += 4;
    }
}

static void padding_replicate_pack4_neon(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const float* ptr = src;
    float* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        const float* ptr0 = ptr;
        float32x4_t _p = vld1q_f32(ptr0);
        for (int x = 0; x < left; x++)
        {
            vst1q_f32(outptr, _p);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = vld1q_f32(ptr0);
            vst1q_f32(outptr, _p);
            ptr0 += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            vst1q_f32(outptr, _p);
            outptr += 4;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        float32x4_t _p = vld1q_f32(ptr);
        for (int x = 0; x < left; x++)
        {
            vst1q_f32(outptr, _p);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = vld1q_f32(ptr);
            vst1q_f32(outptr, _p);
            ptr += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            vst1q_f32(outptr, _p);
            outptr += 4;
        }
    }
    // fill bottom
    ptr -= src.w * 4;
    for (int y = 0; y < bottom; y++)
    {
        const float* ptr0 = ptr;
        float32x4_t _p = vld1q_f32(ptr0);
        for (int x = 0; x < left; x++)
        {
            vst1q_f32(outptr, _p);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = vld1q_f32(ptr0);
            vst1q_f32(outptr, _p);
            ptr0 += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            vst1q_f32(outptr, _p);
            outptr += 4;
        }
    }
}
#endif // __ARM_NEON

int Padding_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (top == 0 && bottom == 0 && left == 0 && right == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __ARM_NEON
    if (opt.use_packing_layout)
    {

    if (elempack == 4)
    {
        int outw = w + left + right;

        if (dims == 1)
        {
            top_blob.create(outw, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (type == 0)
                padding_constant_pack4_neon(bottom_blob, top_blob, 0, 0, left, right, value);
            else
                padding_replicate_pack4_neon(bottom_blob, top_blob, 0, 0, left, right);

            return 0;
        }

        int outh = h + top + bottom;

        if (dims == 2)
        {
            top_blob.create(outw, outh, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (type == 0)
                padding_constant_pack4_neon(bottom_blob, top_blob, top, bottom, left, right, value);
            else
                padding_replicate_pack4_neon(bottom_blob, top_blob, top, bottom, left, right);

            return 0;
        }

        if (dims == 3)
        {
            top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                Mat borderm = top_blob.channel(q);

                if (type == 0)
                    padding_constant_pack4_neon(m, borderm, top, bottom, left, right, value);
                else
                    padding_replicate_pack4_neon(m, borderm, top, bottom, left, right);
            }

            return 0;
        }

        return 0;
    }

    } // opt.use_packing_layout
#endif // __ARM_NEON

    return Padding::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
