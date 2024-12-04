// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "cast_riscv.h"

namespace ncnn {

void Cast_riscv::cast_fp32_to_fp16(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
#if __riscv_zfh
        __fp16* outptr = top_blob.channel(q);
#else
        unsigned short* outptr = top_blob.channel(q);
#endif

#if __riscv_zvfh
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);

            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            vfloat16m4_t _outp = __riscv_vfncvt_f_f_w_f16m4(_p, vl);
            __riscv_vse16_v_f16m4(outptr, _outp, vl);

            ptr += vl;
            outptr += vl;
            n -= vl;
        }
#else // __riscv_zvfh
        for (int i = 0; i < size; i++)
        {
#if __riscv_zfh
            *outptr++ = (__fp16)(*ptr++);
#else
            *outptr++ = float32_to_float16(*ptr++);
#endif
        }
#endif // __riscv_zvfh
    }
}

void Cast_riscv::cast_fp16_to_fp32(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
#if __riscv_zfh
        const __fp16* ptr = bottom_blob.channel(q);
#else
        const unsigned short* ptr = bottom_blob.channel(q);
#endif
        float* outptr = top_blob.channel(q);

#if __riscv_zvfh
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);

            vfloat16m4_t _p = __riscv_vle16_v_f16m4(ptr, vl);
            vfloat32m8_t _outp = __riscv_vfwcvt_f_f_v_f32m8(_p, vl);
            __riscv_vse32_v_f32m8(outptr, _outp, vl);

            ptr += vl;
            outptr += vl;
            n -= vl;
        }
#else // __riscv_zvfh
        for (int i = 0; i < size; i++)
        {
#if __riscv_zfh
            *outptr++ = (float)(*ptr++);
#else
            *outptr++ = float16_to_float32(*ptr++);
#endif
        }
#endif // __riscv_zvfh
    }
}

} // namespace ncnn
