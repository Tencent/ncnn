// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "bias_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

int Bias_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    const float* bias_ptr = bias_data;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        float bias = bias_ptr[q];

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
        float32x4_t _bias = vdupq_n_f32(bias);
        for (; nn > 0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _outp = vaddq_f32(_p, _bias);
            vst1q_f32(ptr, _outp);

            ptr += 4;
        }
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *ptr = *ptr + bias;

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
