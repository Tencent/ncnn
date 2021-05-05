// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "swish_riscv.h"

#if __riscv_vector
#ifdef RVV_SPEC_0_7
#include "riscv_v_071_fix.h"
#else
#include <riscv_vector.h>
#endif
#include "rvv_mathfun.h"
#include "rvv_mathfun_fp16s.h"
#endif // __riscv_vector

#include <math.h>

namespace ncnn {

Swish_riscv::Swish_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector
}

int Swish_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_inplace_fp16sa(bottom_top_blob, opt);
        else
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
            _p = vfdiv_vv_f32m8(_p, vfadd_vf_f32m8(exp_ps(vfneg_v_f32m8(_p, vl), vl), 1.f, vl), vl);
            vse32_v_f32m8(ptr, _p, vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            *ptr = *ptr / (1.f + exp(-*ptr));
            ptr++;
        }
#endif // __riscv_vector
    }

    return 0;
}

#if __riscv_vector && __riscv_zfh
int Swish_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
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
            word_type vl = vsetvl_e16m4(n);

            vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr, vl), vl);
            _p = vfdiv_vv_f32m8(_p, vfadd_vf_f32m8(exp_ps(vfneg_v_f32m8(_p, vl), vl), 1.f, vl), vl);
            vse16_v_f16m4(ptr, vfncvt_f_f_w_f16m4(_p, vl), vl);

            ptr += vl;
            n -= vl;
        }
    }

    return 0;
}

int Swish_riscv::forward_inplace_fp16sa(Mat& bottom_top_blob, const Option& opt) const
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
            _p = vfdiv_vv_f16m8(_p, vfadd_vf_f16m8(exp_ps(vfneg_v_f16m8(_p, vl), vl), 1.f, vl), vl);
            vse16_v_f16m8(ptr, _p, vl);

            ptr += vl;
            n -= vl;
        }
    }

    return 0;
}
#endif // __riscv_vector && __riscv_zfh

} // namespace ncnn
