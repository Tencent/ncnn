// Xavier Hsinyuan is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 Xavier Hsinyuan <thelastlinex@hotmail.com>. All rights reserved.
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

#include "absval_riscv.h"

#if __riscv_vector
#ifdef RVV_SPEC_0_7
#include "riscv_v_071_fix.h"
#else
#include <riscv_vector.h>
#endif

static inline vfloat32m8_t vfabs_v_f32m8_absval(vfloat32m8_t op1, size_t vl)
{
    return vfsgnjx_vv_f32m8(op1, op1, vl);
}
static inline vfloat16m8_t vfabs_v_f16m8_absval(vfloat16m8_t op1, size_t vl)
{
    return vfsgnjx_vv_f16m8(op1, op1, vl);
}
#endif // __riscv_vector

namespace ncnn {

AbsVal_riscv::AbsVal_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector
}

int AbsVal_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        return forward_inplace_fp16s(bottom_top_blob, opt);
    }
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

#if __riscv_vector
        int n = size * elempack;
        while (n > 0)
        {
            word_type vl = vsetvl_e32m8(n);

            vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
            _p = vfabs_v_f32m8_absval(_p, vl);
            vse32_v_f32m8(ptr, _p, vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            *ptr = (*ptr > 0) ? (*ptr) : (-*ptr);

            ptr++;
        }
#endif // __riscv_vector
    }

    return 0;
}

#if __riscv_vector && __riscv_zfh
int AbsVal_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = bottom_top_blob.channel(q);

        int n = size * elempack;
        while (n > 0)
        {
            word_type vl = vsetvl_e16m8(n);

            vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
            _p = vfabs_v_f16m8_absval(_p, vl);
            vse16_v_f16m8(ptr, _p, vl);

            ptr += vl;
            n -= vl;
        }
    }

    return 0;
}
#endif

} // namespace ncnn
