// Leo is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 Leo <leo@nullptr.com.cn>. All rights reserved.
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

#include "clip_mips.h"

#if __MIPS_MSA
#include <msa.h>
#endif // __MIPS_MSA

namespace ncnn {

DEFINE_LAYER_CREATOR(Clip_mips)

int Clip_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

#if __MIPS_MSA
        int nn = size >> 2;
        int remain = size & 3;
#else
        int remain = size;
#endif // __MIPS_MSA

#if __MIPS_MSA
        v4f32 _max = (v4f32)__msa_fill_w(max);
        v4f32 _min = (v4f32)__msa_fill_w(min);
        for (; nn>0; nn--)
        {
            v4f32 _ptr = (v4f32)__msa_ld_w(ptr, 0);
            _ptr = __msa_fmax_w(_ptr, _min);
            _ptr = __msa_fmin_w(_ptr, _max);
            __msa_st_w((v4i32)_ptr, ptr, 0);

            ptr += 4;
        }
#endif // __MIPS_MSA

        for (; remain>0; remain--)
        {
            if (*ptr < min)
                *ptr = min;

            if (*ptr > max)
                *ptr = max;

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
