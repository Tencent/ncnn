// Copyright (C) 2025 AtomAlpaca <atal@anche.no>. All rights reserved.
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

#include "bnll_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun_fp16s.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

#if __riscv_zvfh
static inline vfloat16m8_t __riscv_vfabs_v_f16m8_bnll(vfloat16m8_t op1, size_t vl)
{
    return __riscv_vfsgnjx_vv_f16m8(op1, op1, vl);

}
static inline vfloat16m8_t __riscv_vfneg_v_f16m8_bnll(vfloat16m8_t op1, size_t vl)
{
    return __riscv_vfsgnjn_vv_f16m8(op1, op1, vl);
}
#endif // __riscv_zvfh

#if NCNN_ZFH
int BNLL_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
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

#if __riscv_zvfh
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m8(n);

            vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
            vbool2_t _mask = __riscv_vmfgt_vf_f16m8_b2(_p, (__fp16)0.f, vl);

            vfloat16m8_t _comm = __riscv_vfabs_v_f16m8_bnll(_p, vl);
            _comm = __riscv_vfneg_v_f16m8_bnll(_comm, vl);
            _comm = exp_ps(_comm, vl);
            _comm = __riscv_vfadd_vf_f16m8(_comm, (__fp16)1.f, vl);
            _comm = log_ps(_comm, vl);
            vfloat16m8_t _res = __riscv_vfadd_vv_f16m8_mu(_mask, _comm, _comm, _p, vl);

            __riscv_vse16_v_f16m8(ptr, _res, vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_zvfh
        for (int i = 0; i < size; i++)
        {
            float v = (float)*ptr;
            if (v > 0)
                *ptr = (__fp16)(v + logf(1.f + expf(v)));
            else
                *ptr = (__fp16)(logf(1.f + expf(v)));
            ++ptr;
        }
#endif // __riscv_zvfh
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
