// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "relu_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

ReLU_powerpc::ReLU_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int ReLU_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __ALTIVEC__
            const v4f32 _zero = vec_splats(0.f);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);
                _p = vec_max(_p, _zero);
                __ncnn_altivec_store(ptr, _p);
                ptr += 4;
            }
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                if (*ptr < 0.f)
                    *ptr = 0.f;
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __ALTIVEC__
            const v4f32 _zero = vec_splats(0.f);
            const v4f32 _slope = vec_splats(slope);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);
                v4f32 _pos = vec_max(_p, _zero);
                v4f32 _neg = __ncnn_altivec_mul(vec_min(_p, _zero), _slope);
                _p = vec_add(_pos, _neg);
                __ncnn_altivec_store(ptr, _p);
                ptr += 4;
            }
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                if (*ptr < 0.f)
                    *ptr *= slope;
                ptr++;
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int ReLU_powerpc::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __ALTIVEC__
            const v4f32 _zero = vec_splats(0.f);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_bf16_to_f32(ptr);
                _p = vec_max(_p, _zero);
                __ncnn_altivec_f32_to_bf16(ptr, _p);
                ptr += 4;
            }
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                if (v < 0.f)
                    v = 0.f;
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __ALTIVEC__
            const v4f32 _zero = vec_splats(0.f);
            const v4f32 _slope = vec_splats(slope);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_bf16_to_f32(ptr);
                v4f32 _pos = vec_max(_p, _zero);
                v4f32 _neg = __ncnn_altivec_mul(vec_min(_p, _zero), _slope);
                _p = vec_add(_pos, _neg);
                __ncnn_altivec_f32_to_bf16(ptr, _p);
                ptr += 4;
            }
#endif // __ALTIVEC__
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                if (v < 0.f)
                    v *= slope;
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
