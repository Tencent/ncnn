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
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

Dropout_riscv::Dropout_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif
}

int Dropout_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (scale == 1.f)
    {
        return 0;
    }

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

#if __riscv_vector
        int n = size;
        while (n > 0)
        {
            word_type vl = vsetvl_e32m8(n);

            vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
            _p = vfmul_vf_f32m8(_p, scale, vl);
            vse32_v_f32m8(ptr, _p, vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_vector
        for (int i = 0; i < size; i++)
        {
            *ptr = *ptr * scale;

            ptr++;
        }
#endif // __riscv_vector
    }

    return 0;
}

} // namespace ncnn
