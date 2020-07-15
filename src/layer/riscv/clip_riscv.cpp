// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "clip_riscv.h"

namespace ncnn {

Clip_riscv::Clip_riscv()
{
}

int Clip_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int remain = size;

#if __riscv_vector
        asm volatile(
            "L0:                        \n"
            "vsetvli    t0, %1, e32, m8 \n"
            "vle32.v    v0, (%0)        \n"
            "vfmax.vf   v0, v0, %4      \n"
            "vfmin.vf   v0, v0, %5      \n"
            "vse32.v    v0, (%0)        \n"
            "slli       t1, t0, 2       \n"
            "add        %0, %0, t1      \n"
            "sub        %1, %1, t0      \n"
            "bnez       %1, L0          \n"
            : "=r"(ptr),   // %0
            "=r"(remain) // %1
            : "0"(ptr),
            "1"(remain),
            "f"(min), // %4
            "f"(max)  // %5
            : "cc", "memory", "t0", "t1");
#else  // __riscv_vector
        for (; remain > 0; remain--)
        {
            if (*ptr < min)
                *ptr = min;

            if (*ptr > max)
                *ptr = max;

            ptr++;
        }
#endif // __riscv_vector
    }

    return 0;
}

} //namespace ncnn
