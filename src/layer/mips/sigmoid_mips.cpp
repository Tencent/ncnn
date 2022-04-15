// Leo is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 Leo <leo@nullptr.com.cn>. All rights reserved.
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

#include "sigmoid_mips.h"

#if __mips_msa
#include <msa.h>
#include "msa_mathfun.h"
#endif // __mips_msa

#include "mips_usability.h"

#include <math.h>

namespace ncnn {

Sigmoid_mips::Sigmoid_mips()
{
#if __mips_msa
    support_packing = true;
#endif
}

int Sigmoid_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;

#if __mips_msa
    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);

            for (int i = 0; i < size; i++)
            {
                __builtin_prefetch(ptr + 16);
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                _p = (v4f32)__msa_bnegi_w((v4u32)_p, 31);
                _p = exp_ps(_p);
                _p = __msa_fadd_w(_p, _one);
                v4f32 _outp = __msa_fdiv_w(_one, _p);
                __msa_st_w((v4i32)_outp, ptr, 0);

                ptr += 4;
            }
        }

        return 0;
    }
#endif // __mips_msa

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __mips_msa
        v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);

        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            _p = (v4f32)__msa_bnegi_w((v4u32)_p, 31);
            _p = exp_ps(_p);
            _p = __msa_fadd_w(_p, _one);
            v4f32 _outp = __msa_fdiv_w(_one, _p);
            __msa_st_w((v4i32)_outp, ptr, 0);

            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *ptr = 1.f / (1.f + exp(-*ptr));

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
