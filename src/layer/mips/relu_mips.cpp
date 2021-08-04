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

#include "relu_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

ReLU_mips::ReLU_mips()
{
#if __mips_msa
    support_packing = true;
#endif
}

int ReLU_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

            if (slope == 0.f)
            {
                v4f32 _zero = (v4f32)__msa_fill_w(0);

                for (int i = 0; i < size; i++)
                {
                    __builtin_prefetch(ptr + 32);
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    _p = __msa_fmax_w(_p, _zero);
                    __msa_st_w((v4i32)_p, ptr, 0);

                    ptr += 4;
                }
            }
            else
            {
                v4f32 _zero = (v4f32)__msa_fill_w(0);
                v4f32 _slope = (v4f32)__msa_fill_w_f32(slope);

                for (int i = 0; i < size; i++)
                {
                    __builtin_prefetch(ptr + 32);
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4i32_w _lemask = __msa_fcle_w(_p, _zero);
                    v4f32 _ps = __msa_fmul_w(_p, _slope);
                    _p = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_p, (v16u8)_ps);
                    __msa_st_w((v4i32)_p, ptr, 0);

                    ptr += 4;
                }
            }
        }

        return 0;
    }
#endif // __mips_msa

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        if (slope == 0.f)
        {
            int i = 0;
#if __mips_msa
            v4f32 _zero = (v4f32)__msa_fill_w(0);

            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 32);
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                _p = __msa_fmax_w(_p, _zero);
                __msa_st_w((v4i32)_p, ptr, 0);

                ptr += 4;
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr = 0;
                ptr++;
            }
        }
        else
        {
            int i = 0;
#if __mips_msa
            v4f32 _zero = (v4f32)__msa_fill_w(0);
            v4f32 _slope = (v4f32)__msa_fill_w_f32(slope);

            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 32);
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                v4i32_w _lemask = __msa_fcle_w(_p, _zero);
                v4f32 _ps = __msa_fmul_w(_p, _slope);
                _p = (v4f32)__msa_bsel_v((v16u8)_lemask, (v16u8)_p, (v16u8)_ps);
                __msa_st_w((v4i32)_p, ptr, 0);

                ptr += 4;
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr *= slope;
                ptr++;
            }
        }
    }

    return 0;
}

} // namespace ncnn
