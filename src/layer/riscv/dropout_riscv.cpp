// Xavier Hsinyuan is pleased to support the open source community by making
// ncnn available.
//
// Copyright (C) 2021 Xavier Hsinyuan <me@lstlx.com>. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "dropout_riscv.h"

#if __riscv_vector
#ifdef RVV_SPEC_0_7
#include "riscv_v_071_fix.h"
#else
#include <riscv_vector.h>
#endif
#endif // __riscv_vector

namespace ncnn {
Dropout_riscv::Dropout_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif
}

int Dropout_riscv::forward_inplace(Mat& bottom_top_blob,
                                   const Option& opt) const
{
    if (scale == 1.f)
    {
        return 0;
    }

#if __riscv_vector
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        int n = w * elempack;
        float* ptr = bottom_top_blob;
        while (n > 0)
        {
            word_type vl = vsetvl_e32m8(n);
            vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
            _p = vfmul_vf_f32m8(_p, scale, vl);

            vse32_v_f32m8(ptr, _p, vl);
            ptr += vl;
            n -= vl;
        }
    }
    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            int n = w * elempack;
            while (n > 0)
            {
                word_type vl = vsetvl_e32m8(n);
                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                _p = vfmul_vf_f32m8(_p, scale, vl);

                vse32_v_f32m8(ptr, _p, vl);
                ptr += vl;
                n -= vl;
            }
        }
    }
    if (dims == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            int n = size * elempack;
            while (n > 0)
            {
                word_type vl = vsetvl_e32m8(n);
                vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
                _p = vfmul_vf_f32m8(_p, scale, vl);

                vse32_v_f32m8(ptr, _p, vl);
                ptr += vl;
                n -= vl;
            }
        }
    }
    return 0;
#endif // __riscv_vector

    return Dropout::forward_inplace(bottom_top_blob, opt);
}
} // namespace ncnn
