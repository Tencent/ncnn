// Xavier Hsinyuan is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 Xavier Hsinyuan <me@lstlx.com>. All rights reserved.
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

#include "selu_riscv.h"

#if __riscv_vector
#ifdef RVV_SPEC_0_7
#include "riscv_v_071_fix.h"
#else
#include <riscv_vector.h>
#endif
#include "rvv_mathfun.h"
#endif // __riscv_vector

namespace ncnn {

int SELU_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    float alphaxlambda = alpha * lambda;
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
            vbool4_t _lower = vmflt_vf_f32m8_b4(_p, 0.f, vl);
            vbool4_t _higher = vmnot_m_b4(_lower, vl);

            _p = vfmul_vf_f32m8_m(_higher, _p, /*op1*/ _p, lambda, vl);
            vfloat32m8_t _nps = exp_ps(_p, vl);
            _nps = vfsub_vf_f32m8_m(_lower, _p, /*op1*/ _nps, 1.f, vl);
            _nps = vfmul_vf_f32m8_m(_lower, _p, /*op1*/ _nps, alphaxlambda, vl);

            vse32_v_f32m8(ptr, _nps, vl);
            ptr += vl;
            n -= vl;
        }
#else
        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < 0.f)
                ptr[i] = static_cast<float>((exp(ptr[i]) - 1.f) * alphaxlambda);
            else
                ptr[i] *= lambda;
        }
#endif // __riscv_vector
    }
    return 0;
};

} // namespace ncnn
