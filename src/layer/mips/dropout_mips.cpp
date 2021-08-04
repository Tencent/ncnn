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

#include "dropout_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

Dropout_mips::Dropout_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
}

int Dropout_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (scale == 1.f)
    {
        return 0;
    }

    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

#if __mips_msa
    if (elempack == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        v4f32 _scale = (v4f32)__msa_fill_w_f32(scale);

        if (dims == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                float* ptr = (float*)bottom_top_blob + i * 4;
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                _p = __msa_fmul_w(_p, _scale);
                __msa_st_w((v4i32)_p, ptr, 0);
            }
        }

        if (dims == 2)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);

                for (int j = 0; j < w; j++)
                {
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    _p = __msa_fmul_w(_p, _scale);
                    __msa_st_w((v4i32)_p, ptr, 0);
                    ptr += 4;
                }
            }
        }

        if (dims == 3)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    _p = __msa_fmul_w(_p, _scale);
                    __msa_st_w((v4i32)_p, ptr, 0);
                    ptr += 4;
                }
            }
        }

        return 0;
    }
#endif // __mips_msa

    return Dropout::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn
