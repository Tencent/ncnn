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

#include "shufflechannel_arm.h"

#include "layer_type.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

ShuffleChannel_arm::ShuffleChannel_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

    support_bf16_storage = true;
}

int ShuffleChannel_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);
#endif

    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);

    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;

    int _group = reverse ? channels * elempack / group : group;

    if (_group == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

#if __ARM_NEON
    if (elempack == 4)
    {
        if (_group == 2 && channels % _group != 0)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int size = w * h;
            size_t elemsize = bottom_blob.elemsize;

            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            int channels_per_group = channels / _group;

            // TODO unroll me
            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr0 = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                const float* ptr2 = bottom_blob.channel(channels_per_group + q + 1);
                float* outptr0 = top_blob.channel(q * 2);
                float* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p0 = vld1q_f32(ptr0);
                    float32x4_t _p1 = vld1q_f32(ptr1);
                    float32x4_t _p2 = vld1q_f32(ptr2);

                    float32x4_t _p12 = vextq_f32(_p1, _p2, 2);

                    float32x4x2_t _p01 = vzipq_f32(_p0, _p12);

                    vst1q_f32(outptr0, _p01.val[0]);
                    vst1q_f32(outptr1, _p01.val[1]);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }

            // handle the last channel
            {
                const float* ptr0 = bottom_blob.channel(channels_per_group);
                const float* ptr1 = bottom_blob.channel(channels_per_group + channels_per_group);
                float* outptr0 = top_blob.channel(channels_per_group * 2);

                ptr1 += 2;

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p0 = vld1q_f32(ptr0);
                    float32x4_t _p1 = vld1q_f32(ptr1);

                    float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                    vst1q_f32(outptr0, _p01.val[0]);

                    ptr0 += 4;
                    ptr1 += 4;
                    outptr0 += 4;
                }
            }

            return 0;
        }

        if (_group > 4 || channels % _group != 0)
        {
            // slow path for too large group or shuffle inside elempack
            Option opt_pack = opt;
            opt_pack.blob_allocator = opt.workspace_allocator;

            Mat bottom_blob_unpacked;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);

            Mat top_blob_unpacked;
            int ret = ShuffleChannel::forward(bottom_blob_unpacked, top_blob_unpacked, opt_pack);
            if (ret != 0)
                return ret;

            convert_packing(top_blob_unpacked, top_blob, elempack, opt);

            return 0;
        }

        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int size = w * h;
        size_t elemsize = bottom_blob.elemsize;

        top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int channels_per_group = channels / _group;

        if (_group == 2)
        {
            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr0 = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                float* outptr0 = top_blob.channel(q * 2);
                float* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p0 = vld1q_f32(ptr0);
                    float32x4_t _p1 = vld1q_f32(ptr1);

                    float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                    vst1q_f32(outptr0, _p01.val[0]);
                    vst1q_f32(outptr1, _p01.val[1]);

                    ptr0 += 4;
                    ptr1 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }
        }

        if (_group == 3)
        {
            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr0 = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                const float* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                float* outptr0 = top_blob.channel(q * 3);
                float* outptr1 = top_blob.channel(q * 3 + 1);
                float* outptr2 = top_blob.channel(q * 3 + 2);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p0 = vld1q_f32(ptr0);
                    float32x4_t _p1 = vld1q_f32(ptr1);
                    float32x4_t _p2 = vld1q_f32(ptr2);

                    float32x4x2_t _p01 = vzipq_f32(_p0, _p1);
                    float32x4x2_t _p12 = vzipq_f32(_p1, _p2);

                    float32x4_t _0415 = _p01.val[0];
                    float32x4_t _2637 = _p01.val[1];
                    float32x4_t _4859 = _p12.val[0];
                    float32x4_t _6x7y = _p12.val[1];

                    float32x2_t _15 = vget_high_f32(_0415);
                    float32x2_t _37 = vget_high_f32(_2637);
                    float32x2_t _48 = vget_low_f32(_4859);
                    float32x2_t _6x = vget_low_f32(_6x7y);

                    float32x2_t _81 = vext_f32(_48, _15, 1);
                    float32x2_t _x3 = vext_f32(_6x, _37, 1);

                    float32x4_t _0481 = vcombine_f32(vget_low_f32(_0415), _81);
                    float32x4_t _5926 = vextq_f32(_4859, _2637, 2);
                    float32x4_t _x37y = vcombine_f32(_x3, vget_high_f32(_6x7y));

                    vst1q_f32(outptr0, _0481);
                    vst1q_f32(outptr1, _5926);
                    vst1q_f32(outptr2, _x37y);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                }
            }
        }

        if (_group == 4)
        {
            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr0 = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                const float* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                const float* ptr3 = bottom_blob.channel(channels_per_group * 3 + q);
                float* outptr0 = top_blob.channel(q * 4);
                float* outptr1 = top_blob.channel(q * 4 + 1);
                float* outptr2 = top_blob.channel(q * 4 + 2);
                float* outptr3 = top_blob.channel(q * 4 + 3);

                for (int i = 0; i < size; i++)
                {
                    float32x4_t _p0 = vld1q_f32(ptr0);
                    float32x4_t _p1 = vld1q_f32(ptr1);
                    float32x4_t _p2 = vld1q_f32(ptr2);
                    float32x4_t _p3 = vld1q_f32(ptr3);

                    // transpose 4x4
                    float32x4x2_t _p01 = vtrnq_f32(_p0, _p1);
                    float32x4x2_t _p23 = vtrnq_f32(_p2, _p3);
                    _p0 = vcombine_f32(vget_low_f32(_p01.val[0]), vget_low_f32(_p23.val[0]));
                    _p1 = vcombine_f32(vget_low_f32(_p01.val[1]), vget_low_f32(_p23.val[1]));
                    _p2 = vcombine_f32(vget_high_f32(_p01.val[0]), vget_high_f32(_p23.val[0]));
                    _p3 = vcombine_f32(vget_high_f32(_p01.val[1]), vget_high_f32(_p23.val[1]));

                    vst1q_f32(outptr0, _p0);
                    vst1q_f32(outptr1, _p1);
                    vst1q_f32(outptr2, _p2);
                    vst1q_f32(outptr3, _p3);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    return ShuffleChannel::forward(bottom_blob, top_blob, opt);
}

int ShuffleChannel_arm::forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;

    int _group = reverse ? channels * elempack / group : group;

    if (_group == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (elempack == 8)
    {
        if (_group == 2 && channels % _group != 0)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int size = w * h;
            size_t elemsize = bottom_blob.elemsize;

            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            int channels_per_group = channels / _group;

            // TODO unroll me
            for (int q = 0; q < channels_per_group; q++)
            {
                const __fp16* ptr0 = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob.channel(channels_per_group + q);
                const __fp16* ptr2 = bottom_blob.channel(channels_per_group + q + 1);
                __fp16* outptr0 = top_blob.channel(q * 2);
                __fp16* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    float16x8_t _p0 = vld1q_f16(ptr0);
                    float16x8_t _p1 = vld1q_f16(ptr1);
                    float16x8_t _p2 = vld1q_f16(ptr2);

                    float16x8_t _p12 = vextq_f16(_p1, _p2, 4);

                    float16x8x2_t _p01 = vzipq_f16(_p0, _p12);

                    vst1q_f16(outptr0, _p01.val[0]);
                    vst1q_f16(outptr1, _p01.val[1]);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                }
            }

            // handle the last channel
            {
                const __fp16* ptr0 = bottom_blob.channel(channels_per_group);
                const __fp16* ptr1 = bottom_blob.channel(channels_per_group + channels_per_group);
                __fp16* outptr0 = top_blob.channel(channels_per_group * 2);

                ptr1 += 4;

                for (int i = 0; i < size; i++)
                {
                    float16x4_t _p0 = vld1_f16(ptr0);
                    float16x4_t _p1 = vld1_f16(ptr1);

                    float16x4x2_t _p01 = vzip_f16(_p0, _p1);

                    vst1_f16(outptr0, _p01.val[0]);
                    vst1_f16(outptr0 + 4, _p01.val[1]);

                    ptr0 += 8;
                    ptr1 += 8;
                    outptr0 += 8;
                }
            }

            return 0;
        }

        if (_group > 4 || channels % _group != 0)
        {
            // slow path for too large group or shuffle inside elempack
            Option opt_pack = opt;
            opt_pack.blob_allocator = opt.workspace_allocator;

            Mat bottom_blob_unpacked;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);

            Mat top_blob_unpacked;
            int ret = ShuffleChannel::forward(bottom_blob_unpacked, top_blob_unpacked, opt_pack);
            if (ret != 0)
                return ret;

            convert_packing(top_blob_unpacked, top_blob, elempack, opt);

            return 0;
        }

        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int size = w * h;
        size_t elemsize = bottom_blob.elemsize;

        top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int channels_per_group = channels / _group;

        if (_group == 2)
        {
            for (int q = 0; q < channels_per_group; q++)
            {
                const __fp16* ptr0 = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob.channel(channels_per_group + q);
                __fp16* outptr0 = top_blob.channel(q * 2);
                __fp16* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    float16x8_t _p0 = vld1q_f16(ptr0);
                    float16x8_t _p1 = vld1q_f16(ptr1);

                    float16x8x2_t _p01 = vzipq_f16(_p0, _p1);

                    vst1q_f16(outptr0, _p01.val[0]);
                    vst1q_f16(outptr1, _p01.val[1]);

                    ptr0 += 8;
                    ptr1 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                }
            }
        }

        if (_group == 3)
        {
            for (int q = 0; q < channels_per_group; q++)
            {
                const __fp16* ptr0 = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob.channel(channels_per_group + q);
                const __fp16* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                __fp16* outptr0 = top_blob.channel(q * 3);
                __fp16* outptr1 = top_blob.channel(q * 3 + 1);
                __fp16* outptr2 = top_blob.channel(q * 3 + 2);

                for (int i = 0; i < size; i++)
                {
                    float16x8_t _p0 = vld1q_f16(ptr0);
                    float16x8_t _p1 = vld1q_f16(ptr1);
                    float16x8_t _p2 = vld1q_f16(ptr2);

                    // TODO figure out a faster way

                    // 01234567        08g19h2a
                    // 89abcdef   ->   i3bj4ck5
                    // ghijklmn        dl6em7fn

                    float16x8x3_t _p012;
                    _p012.val[0] = _p0;
                    _p012.val[1] = _p1;
                    _p012.val[2] = _p2;

                    __fp16 tmp[24];
                    vst3q_f16(&tmp[0], _p012);

                    _p0 = vld1q_f16(&tmp[0]);
                    _p1 = vld1q_f16(&tmp[8]);
                    _p2 = vld1q_f16(&tmp[16]);

                    vst1q_f16(outptr0, _p0);
                    vst1q_f16(outptr1, _p1);
                    vst1q_f16(outptr2, _p2);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                }
            }
        }

        if (_group == 4)
        {
            for (int q = 0; q < channels_per_group; q++)
            {
                const __fp16* ptr0 = bottom_blob.channel(q);
                const __fp16* ptr1 = bottom_blob.channel(channels_per_group + q);
                const __fp16* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                const __fp16* ptr3 = bottom_blob.channel(channels_per_group * 3 + q);
                __fp16* outptr0 = top_blob.channel(q * 4);
                __fp16* outptr1 = top_blob.channel(q * 4 + 1);
                __fp16* outptr2 = top_blob.channel(q * 4 + 2);
                __fp16* outptr3 = top_blob.channel(q * 4 + 3);

                for (int i = 0; i < size; i++)
                {
                    float16x8_t _p0 = vld1q_f16(ptr0);
                    float16x8_t _p1 = vld1q_f16(ptr1);
                    float16x8_t _p2 = vld1q_f16(ptr2);
                    float16x8_t _p3 = vld1q_f16(ptr3);

                    // transpose 4x4
                    float16x8x2_t _p01 = vtrnq_f16(_p0, _p1);
                    float16x8x2_t _p23 = vtrnq_f16(_p2, _p3);
                    uint32x4x2_t _p02 = vtrnq_u32(vreinterpretq_u32_f16(_p01.val[0]), vreinterpretq_u32_f16(_p23.val[0]));
                    uint32x4x2_t _p13 = vtrnq_u32(vreinterpretq_u32_f16(_p01.val[1]), vreinterpretq_u32_f16(_p23.val[1]));
                    _p0 = vreinterpretq_f16_u32(_p02.val[0]);
                    _p1 = vreinterpretq_f16_u32(_p13.val[0]);
                    _p2 = vreinterpretq_f16_u32(_p02.val[1]);
                    _p3 = vreinterpretq_f16_u32(_p13.val[1]);

                    vst1q_f16(outptr0, vcombine_f16(vget_low_f16(_p0), vget_low_f16(_p1)));
                    vst1q_f16(outptr1, vcombine_f16(vget_low_f16(_p2), vget_low_f16(_p3)));
                    vst1q_f16(outptr2, vcombine_f16(vget_high_f16(_p0), vget_high_f16(_p1)));
                    vst1q_f16(outptr3, vcombine_f16(vget_high_f16(_p2), vget_high_f16(_p3)));

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                    outptr3 += 8;
                }
            }
        }

        return 0;
    }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if __ARM_NEON
    if (elempack == 4)
    {
        if (_group == 2 && channels % _group != 0)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int size = w * h;
            size_t elemsize = bottom_blob.elemsize;

            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            int channels_per_group = channels / _group;

            // TODO unroll me
            for (int q = 0; q < channels_per_group; q++)
            {
                const unsigned short* ptr0 = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + q);
                const unsigned short* ptr2 = bottom_blob.channel(channels_per_group + q + 1);
                unsigned short* outptr0 = top_blob.channel(q * 2);
                unsigned short* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    uint16x4_t _p0 = vld1_u16(ptr0);
                    uint16x4_t _p1 = vld1_u16(ptr1);
                    uint16x4_t _p2 = vld1_u16(ptr2);

                    uint16x4_t _p12 = vext_u16(_p1, _p2, 2);

                    uint16x4x2_t _p01 = vzip_u16(_p0, _p12);

                    vst1_u16(outptr0, _p01.val[0]);
                    vst1_u16(outptr1, _p01.val[1]);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }

            // handle the last channel
            {
                const unsigned short* ptr0 = bottom_blob.channel(channels_per_group);
                const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + channels_per_group);
                unsigned short* outptr0 = top_blob.channel(channels_per_group * 2);

                ptr1 += 2;

                for (int i = 0; i < size; i++)
                {
                    uint16x4_t _p0 = vld1_u16(ptr0);
                    uint16x4_t _p1 = vld1_u16(ptr1);

                    uint16x4x2_t _p01 = vzip_u16(_p0, _p1);

                    vst1_u16(outptr0, _p01.val[0]);

                    ptr0 += 4;
                    ptr1 += 4;
                    outptr0 += 4;
                }
            }

            return 0;
        }

        if (_group > 4 || channels % _group != 0)
        {
            // slow path for too large group or shuffle inside elempack
            Option opt_pack = opt;
            opt_pack.blob_allocator = opt.workspace_allocator;

            Mat bottom_blob_unpacked;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);

            Mat top_blob_unpacked;
            int ret = ShuffleChannel::forward(bottom_blob_unpacked, top_blob_unpacked, opt_pack);
            if (ret != 0)
                return ret;

            convert_packing(top_blob_unpacked, top_blob, elempack, opt);

            return 0;
        }

        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int size = w * h;
        size_t elemsize = bottom_blob.elemsize;

        top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int channels_per_group = channels / _group;

        if (_group == 2)
        {
            for (int q = 0; q < channels_per_group; q++)
            {
                const unsigned short* ptr0 = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + q);
                unsigned short* outptr0 = top_blob.channel(q * 2);
                unsigned short* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    uint16x4_t _p0 = vld1_u16(ptr0);
                    uint16x4_t _p1 = vld1_u16(ptr1);

                    uint16x4x2_t _p01 = vzip_u16(_p0, _p1);

                    vst1_u16(outptr0, _p01.val[0]);
                    vst1_u16(outptr1, _p01.val[1]);

                    ptr0 += 4;
                    ptr1 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }
        }

        if (_group == 3)
        {
            for (int q = 0; q < channels_per_group; q++)
            {
                const unsigned short* ptr0 = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + q);
                const unsigned short* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                unsigned short* outptr0 = top_blob.channel(q * 3);
                unsigned short* outptr1 = top_blob.channel(q * 3 + 1);
                unsigned short* outptr2 = top_blob.channel(q * 3 + 2);

                for (int i = 0; i < size; i++)
                {
                    uint16x4_t _p0 = vld1_u16(ptr0);
                    uint16x4_t _p1 = vld1_u16(ptr1);
                    uint16x4_t _p2 = vld1_u16(ptr2);

                    // TODO figure out a faster way
                    uint16x4x2_t _p01 = vzip_u16(_p0, _p1);
                    uint16x4x2_t _p12 = vzip_u16(_p1, _p2);

                    uint32x2_t _0415 = vreinterpret_u32_u16(_p01.val[0]);
                    uint16x4_t _2637 = _p01.val[1];
                    uint16x4_t _4859 = _p12.val[0];
                    uint32x2_t _6x7y = vreinterpret_u32_u16(_p12.val[1]);

                    uint16x4_t _98yx = vrev32_u16(_p2);
                    uint16x4x2_t _90y281x3 = vtrn_u16(_98yx, _p0);

                    uint32x2_t _81x3 = vreinterpret_u32_u16(_90y281x3.val[1]);

                    uint32x2x2_t _048115x3 = vtrn_u32(_0415, _81x3);
                    uint32x2x2_t _816xx37y = vtrn_u32(_81x3, _6x7y);

                    uint16x4_t _0481 = vreinterpret_u16_u32(_048115x3.val[0]);
                    uint16x4_t _5926 = vext_u16(_4859, _2637, 2);
                    uint16x4_t _x37y = vreinterpret_u16_u32(_816xx37y.val[1]);

                    vst1_u16(outptr0, _0481);
                    vst1_u16(outptr1, _5926);
                    vst1_u16(outptr2, _x37y);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                }
            }
        }

        if (_group == 4)
        {
            for (int q = 0; q < channels_per_group; q++)
            {
                const unsigned short* ptr0 = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + q);
                const unsigned short* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                const unsigned short* ptr3 = bottom_blob.channel(channels_per_group * 3 + q);
                unsigned short* outptr0 = top_blob.channel(q * 4);
                unsigned short* outptr1 = top_blob.channel(q * 4 + 1);
                unsigned short* outptr2 = top_blob.channel(q * 4 + 2);
                unsigned short* outptr3 = top_blob.channel(q * 4 + 3);

                for (int i = 0; i < size; i++)
                {
                    uint16x4_t _p0 = vld1_u16(ptr0);
                    uint16x4_t _p1 = vld1_u16(ptr1);
                    uint16x4_t _p2 = vld1_u16(ptr2);
                    uint16x4_t _p3 = vld1_u16(ptr3);

                    // transpose 4x4
                    uint16x4x2_t _p01 = vtrn_u16(_p0, _p1);
                    uint16x4x2_t _p23 = vtrn_u16(_p2, _p3);
                    uint32x2x2_t _p02 = vtrn_u32(vreinterpret_u32_u16(_p01.val[0]), vreinterpret_u32_u16(_p23.val[0]));
                    uint32x2x2_t _p13 = vtrn_u32(vreinterpret_u32_u16(_p01.val[1]), vreinterpret_u32_u16(_p23.val[1]));
                    _p0 = vreinterpret_u16_u32(_p02.val[0]);
                    _p1 = vreinterpret_u16_u32(_p13.val[0]);
                    _p2 = vreinterpret_u16_u32(_p02.val[1]);
                    _p3 = vreinterpret_u16_u32(_p13.val[1]);

                    vst1_u16(outptr0, _p0);
                    vst1_u16(outptr1, _p1);
                    vst1_u16(outptr2, _p2);
                    vst1_u16(outptr3, _p3);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    return ShuffleChannel::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
