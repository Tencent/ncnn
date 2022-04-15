// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 Leo <leo@nullptr.com.cn>. All rights reserved.
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

#include "absval_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

namespace ncnn {

AbsVal_mips::AbsVal_mips()
{
#if __mips_msa
    support_packing = true;
#endif
}

int AbsVal_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

            for (int i = 0; i < size; i++)
            {
                __builtin_prefetch(ptr + 16);
                v4u32 _p = (v4u32)__msa_ld_w(ptr, 0);
                v4f32 _outp = (v4f32)__msa_bclri_w(_p, 31);
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
        for (; i + 3 < size; i += 4)
        {
            __builtin_prefetch(ptr + 16);
            v4u32 _p = (v4u32)__msa_ld_w(ptr, 0);
            v4f32 _outp = (v4f32)__msa_bclri_w(_p, 31);
            __msa_st_w((v4i32)_outp, ptr, 0);

            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *ptr = *ptr > 0 ? *ptr : -*ptr;

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
