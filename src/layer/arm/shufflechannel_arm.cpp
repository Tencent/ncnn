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

DEFINE_LAYER_CREATOR(ShuffleChannel_arm)

ShuffleChannel_arm::ShuffleChannel_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

int ShuffleChannel_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (group == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int elempack = bottom_blob.elempack;

#if __ARM_NEON
    if (opt.use_packing_layout)
    {

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;
    size_t elemsize = bottom_blob.elemsize;

    if (elempack == 4)
    {
        if (group <= 4 && channels % group == 0)
        {
            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            int channels_per_group = channels / group;

            if (group == 2)
            {
                for (int q=0; q<channels_per_group; q++)
                {
                    const float* ptr0 = bottom_blob.channel(q);
                    const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                    float* outptr0 = top_blob.channel(q*2);
                    float* outptr1 = top_blob.channel(q*2+1);

                    for (int i=0; i<size; i++)
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
            else if (group == 3)
            {
                for (int q=0; q<channels_per_group; q++)
                {
                    const float* ptr0 = bottom_blob.channel(q);
                    const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                    const float* ptr2 = bottom_blob.channel(channels_per_group*2 + q);
                    float* outptr0 = top_blob.channel(q*3);
                    float* outptr1 = top_blob.channel(q*3+1);
                    float* outptr2 = top_blob.channel(q*3+2);

                    for (int i=0; i<size; i++)
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
            else // group == 4
            {
                for (int q=0; q<channels_per_group; q++)
                {
                    const float* ptr0 = bottom_blob.channel(q);
                    const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                    const float* ptr2 = bottom_blob.channel(channels_per_group*2 + q);
                    const float* ptr3 = bottom_blob.channel(channels_per_group*3 + q);
                    float* outptr0 = top_blob.channel(q*4);
                    float* outptr1 = top_blob.channel(q*4+1);
                    float* outptr2 = top_blob.channel(q*4+2);
                    float* outptr3 = top_blob.channel(q*4+3);

                    for (int i=0; i<size; i++)
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
        }
        else
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

            convert_packing(top_blob_unpacked, top_blob, 4, opt);
        }

        return 0;
    }

    } // opt.use_packing_layout
#endif // __ARM_NEON

    return ShuffleChannel::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
