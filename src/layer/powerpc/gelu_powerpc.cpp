// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gelu_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#include "powerpc_mathfun.h"
#endif // __ALTIVEC__

namespace ncnn {

GELU_powerpc::GELU_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
    support_any_packing = true;
#endif // __ALTIVEC__
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int GELU_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __ALTIVEC__
        if (fast_gelu)
        {
            v4f32 _fast1c = vec_splats(0.79788452f);
            v4f32 _fast2c = vec_splats(0.044715f * 0.79788452f);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);

                v4f32 _cube = __ncnn_altivec_mul(_p, _p);
                _cube = __ncnn_altivec_mul(_p, _cube);
                v4f32 _blob = __ncnn_altivec_mul(_fast2c, _cube);
                _blob = __ncnn_altivec_fmadd(_blob, _fast1c, _p);
                // 0.5 * x * (1 + tanh(y)) == x * sigmoid(2 * y)
                // the sigmoid form keeps the tail exact for saturated tanh
                _blob = sigmoid_ps(vec_add(_blob, _blob));
                _blob = __ncnn_altivec_mul(_p, _blob);
                __ncnn_altivec_store(ptr, _blob);

                ptr += 4;
            }
        }
        else
        {
            v4f32 _half = vec_splats(0.5f);
            v4f32 _one = vec_splats(1.f);
            v4f32 _inv_sqrt2 = vec_splats(0.70710678f);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_load(ptr);

                v4f32 _blob = __ncnn_altivec_mul(_inv_sqrt2, _p);
                _blob = erf_ps(_blob);
                _blob = vec_add(_one, _blob);
                _blob = __ncnn_altivec_mul(_half, __ncnn_altivec_mul(_blob, _p));
                __ncnn_altivec_store(ptr, _blob);

                ptr += 4;
            }
        }
#endif // __ALTIVEC__
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

#if NCNN_BF16
int GELU_powerpc::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
        unsigned short* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __ALTIVEC__
        if (fast_gelu)
        {
            v4f32 _fast1c = vec_splats(0.79788452f);
            v4f32 _fast2c = vec_splats(0.044715f * 0.79788452f);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_bf16_to_f32(ptr);

                v4f32 _cube = __ncnn_altivec_mul(_p, _p);
                _cube = __ncnn_altivec_mul(_p, _cube);
                v4f32 _blob = __ncnn_altivec_mul(_fast2c, _cube);
                _blob = __ncnn_altivec_fmadd(_blob, _fast1c, _p);
                // 0.5 * x * (1 + tanh(y)) == x * sigmoid(2 * y)
                // the sigmoid form keeps the tail exact for saturated tanh
                _blob = sigmoid_ps(vec_add(_blob, _blob));
                _blob = __ncnn_altivec_mul(_p, _blob);
                __ncnn_altivec_f32_to_bf16(ptr, _blob);

                ptr += 4;
            }
        }
        else
        {
            v4f32 _half = vec_splats(0.5f);
            v4f32 _one = vec_splats(1.f);
            v4f32 _inv_sqrt2 = vec_splats(0.70710678f);
            for (; i + 3 < size; i += 4)
            {
                v4f32 _p = __ncnn_altivec_bf16_to_f32(ptr);

                v4f32 _blob = __ncnn_altivec_mul(_inv_sqrt2, _p);
                _blob = erf_ps(_blob);
                _blob = vec_add(_one, _blob);
                _blob = __ncnn_altivec_mul(_half, __ncnn_altivec_mul(_blob, _p));
                __ncnn_altivec_f32_to_bf16(ptr, _blob);

                ptr += 4;
            }
        }
#endif // __ALTIVEC__
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (fast_gelu)
            {
                v = 0.5f * v * (1.0f + tanhf(0.79788452f * (v + 0.044715f * v * v * v)));
            }
            else
            {
                v = 0.5f * v * (1.0f + erff(0.70710678f * v));
            }
            *ptr = float32_to_bfloat16(v);

            ptr++;
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
