// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "threshold_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Threshold_powerpc::Threshold_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

int Threshold_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
#if __ALTIVEC__
        v4f32 _thr = vec_splats(threshold);
        v4f32 _zero = vec_splats(0.f);
        v4f32 _one = vec_splats(1.f);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = __ncnn_altivec_load(ptr);
            v4i32 _mask = __ncnn_altivec_cmpgt(_p, _thr);
            v4f32 _r = __ncnn_altivec_sel(_mask, _one, _zero);
            __ncnn_altivec_store(ptr, _r);

            ptr += 4;
        }
#endif // __ALTIVEC__
        for (; i < size; i++)
        {
            *ptr = *ptr > threshold ? 1.f : 0.f;
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
