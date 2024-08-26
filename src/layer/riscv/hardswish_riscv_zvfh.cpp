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

#include "hardswish_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

#if __riscv_zvfh
int HardSwish_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
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

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m8(n);
            vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);

            vbool2_t _lower = __riscv_vmflt_vf_f16m8_b2(_p, lower, vl);
            vbool2_t _higher = __riscv_vmfgt_vf_f16m8_b2(_p, upper, vl);
            vbool2_t _apply = __riscv_vmnor_mm_b2(_lower, _higher, vl);
            _p = __riscv_vfmerge_vfm_f16m8(_p, .0f, _lower, vl);

            vfloat16m8_t _p0 = __riscv_vfadd_vf_f16m8_m(_apply, __riscv_vfmul_vf_f16m8_m(_apply, _p, alpha, vl), beta, vl);
            _p = __riscv_vfmul_vv_f16m8_m(_apply, _p, _p0, vl);

            __riscv_vse16_v_f16m8(ptr, _p, vl);
            ptr += vl;
            n -= vl;
        }
    }

    return 0;
}
#endif // __riscv_zvfh

} // namespace ncnn
