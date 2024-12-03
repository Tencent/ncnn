// Xavier Hsinyuan is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 Xavier Hsinyuan <me@lstlx.com>. All rights reserved.
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

#include "gelu_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

namespace ncnn {

GELU_riscv::GELU_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif
}

int GELU_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

#if __riscv_vector
    if (fast_gelu)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m4(n);

                vfloat32m4_t _p = __riscv_vle32_v_f32m4(ptr, vl);

                vfloat32m4_t _arg = __riscv_vfmul_vf_f32m4(
                                        __riscv_vfmul_vv_f32m4(__riscv_vfmul_vv_f32m4(_p, _p, vl), _p, vl), 0.044715f, vl);

                _arg = __riscv_vfadd_vv_f32m4(_p, _arg, vl);
                _arg = __riscv_vfmul_vf_f32m4(_arg, 0.79788452f, vl);
                vfloat32m4_t _tanharg = tanh_ps(_arg, vl);
                _p = __riscv_vfmul_vf_f32m4(
                         __riscv_vfmul_vv_f32m4(_p, __riscv_vfadd_vf_f32m4(_tanharg, 1.f, vl), vl), .5f, vl);

                __riscv_vse32_v_f32m4(ptr, _p, vl);
                n -= vl;
                ptr += vl;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

#if C906
            // FIXME -O3 leads illegal instruction
            for (int i = 0; i < size; i++)
            {
                // y = x * P(X <= x) where X ~ N(0, 1)
                ptr[i] = 0.5f * ptr[i] * erfcf(-0.70710678f * ptr[i]);
            }
#else
            int n = size;
            while (n > 0)
            {
                size_t vl = __riscv_vsetvl_e32m8(n);
                auto _p = __riscv_vle32_v_f32m8(ptr, vl);
                auto _perfc = __riscv_vfmul_vf_f32m8(_p, -.70710678f, vl);
                _p = __riscv_vfmul_vf_f32m8(_p, .5f, vl);
                // y = x * P(X <= x) where X ~ N(0, 1)

                _perfc = erfc_ps(_perfc, vl);

                _p = __riscv_vfmul_vv_f32m8(_p, _perfc, vl);
                __riscv_vse32_v_f32m8(ptr, _p, vl);

                n -= vl;
                ptr += vl;
            }
#endif
        }
    }

    return 0;
#endif

    return GELU::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn
