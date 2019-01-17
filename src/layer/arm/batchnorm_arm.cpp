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

#include "batchnorm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(BatchNorm_arm)

int BatchNorm_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    if (dims != 3)
        return BatchNorm::forward_inplace(bottom_top_blob, opt);

    // a = bias - slope * mean / sqrt(var)
    // b = slope / sqrt(var)
    // value = b * value + a

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int size = w * h;

    const float* a_data_ptr = a_data;
    const float* b_data_ptr = b_data;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        float a = a_data_ptr[q];
        float b = b_data_ptr[q];

#if __ARM_NEON
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        if (nn > 0)
        {
        asm volatile(
            "dup        v1.4s, %w4             \n"
            "dup        v2.4s, %w5             \n"
            "0:                                \n"
            "prfm       pldl1keep, [%1, #128]  \n"
            "ld1        {v0.4s}, [%1]          \n"
            "orr        v3.16b, v1.16b, v1.16b \n"
            "fmla       v3.4s, v0.4s, v2.4s    \n"
            "subs       %w0, %w0, #1           \n"
            "st1        {v3.4s}, [%1], #16     \n"
            "bne        0b                     \n"
            : "=r"(nn),     // %0
              "=r"(ptr)     // %1
            : "0"(nn),
              "1"(ptr),
              "r"(a),       // %4
              "r"(b)        // %5
            : "cc", "memory", "v0", "v1", "v2", "v3"
        );
        }
#else
        if (nn > 0)
        {
        asm volatile(
            "vdup.f32   q1, %4              \n"
            "vdup.f32   q2, %5              \n"
            "0:                             \n"
            "pld        [%1, #128]          \n"
            "vld1.f32   {d0-d1}, [%1 :128]  \n"
            "vorr.32    q3, q1, q1          \n"
            "vmla.f32   q3, q0, q2          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d6-d7}, [%1 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
              "=r"(ptr)     // %1
            : "0"(nn),
              "1"(ptr),
              "r"(a),       // %4
              "r"(b)        // %5
            : "cc", "memory", "q0", "q1", "q2", "q3"
        );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain>0; remain--)
        {
            *ptr = b * *ptr + a;

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
