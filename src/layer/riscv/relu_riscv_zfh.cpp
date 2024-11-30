// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "relu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
int ReLU_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);
        if (slope == 0.f)
        {
#if __riscv_zvfh
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m8(n);

                vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
                _p = __riscv_vfmax_vf_f16m8(_p, (__fp16)0.f, vl);
                __riscv_vse16_v_f16m8(ptr, _p, vl);

                ptr += vl;
                n -= vl;
            }
#else  // __riscv_zvfh
            for (int i = 0; i < size; i++)
            {
                if (*ptr < (__fp16)0.f)
                    *ptr = (__fp16)0.f;
                ptr++;
            }
#endif // __riscv_zvfh
        }
        else
        {
            __fp16 _slope = (__fp16)slope;
#if __riscv_zvfh
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e16m8(n);

                vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
                _p = __riscv_vfmul_vf_f16m8_mu(__riscv_vmflt_vf_f16m8_b2(_p, (__fp16)0.f, vl), _p, _p, _slope, vl);
                __riscv_vse16_v_f16m8(ptr, _p, vl);

                ptr += vl;
                n -= vl;
            }
#else  // __riscv_zvfh
            for (int i = 0; i < size; i++)
            {
                if (*ptr < (__fp16)0.f)
                    *ptr *= _slope;
                ptr++;
            }
#endif // __riscv_zvfh
        }
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
