// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gelu_mips.h"

#if __mips_msa
#include <msa.h>
#include "msa_mathfun.h"
#endif // __mips_msa

namespace ncnn {

GELU_mips::GELU_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
}

int GELU_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __mips_msa
        if (fast_gelu)
        {
            v4f32 _half = (v4f32)__msa_fill_w_f32(0.5f);
            v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
            v4f32 _fast1c = (v4f32)__msa_fill_w_f32(0.79788452f);
            v4f32 _fast2c = (v4f32)__msa_fill_w_f32(0.044715f * 0.79788452f);
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);

                v4f32 _cube = __msa_fmul_w(_p, _p);
                _cube = __msa_fmul_w(_p, _cube);
                v4f32 _blob = __msa_fmul_w(_fast2c, _cube);
                _blob = __msa_fmadd_w(_blob, _fast1c, _p);
                _blob = tanh_ps(_blob);
                _blob = __msa_fadd_w(_one, _blob);
                _blob = __msa_fmul_w(_half, __msa_fmul_w(_blob, _p));
                __msa_st_w((v4i32)_blob, ptr, 0);

                ptr += 4;
            }
        }
        else
        {
            v4f32 _half = (v4f32)__msa_fill_w_f32(0.5f);
            v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
            v4f32 _inv_sqrt2 = (v4f32)__msa_fill_w_f32(0.70710678f);
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);

                v4f32 _blob = __msa_fmul_w(_inv_sqrt2, _p);
                _blob = erf_ps(_blob);
                _blob = __msa_fadd_w(_one, _blob);
                _blob = __msa_fmul_w(_half, __msa_fmul_w(_blob, _p));
                __msa_st_w((v4i32)_blob, ptr, 0);

                ptr += 4;
            }
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            if (fast_gelu)
            {
                *ptr = 0.5f * *ptr * (1.0f + tanhf(0.79788452f * (*ptr + 0.044715f * *ptr * *ptr * *ptr)));
            }
            else
            {
                *ptr = 0.5f * *ptr * (1.0f + erff(0.70710678f * *ptr));
            }

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
