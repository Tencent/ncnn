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

#include "layernorm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

LayerNorm_arm::LayerNorm_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static void layernorm(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

#if __ARM_NEON
    float32x4_t _mean = vdupq_n_f32(0.f);
#endif // __ARM_NEON
    float mean = 0.f;
    {
        const float* ptr0 = ptr;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr0);
            _mean = vaddq_f32(_mean, _p);
            ptr0 += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            mean += ptr0[0];
            ptr0++;
        }
    }

#if __ARM_NEON
    if (elempack == 4)
    {
        float32x4_t _elemcount = vdupq_n_f32(elemcount);
        _mean = div_ps(_mean, _elemcount);
    }
#endif // __ARM_NEON
    if (elempack == 1)
    {
#if __ARM_NEON
#if __aarch64__
        mean += vaddvq_f32(_mean);
#else
        float32x2_t _s2 = vadd_f32(vget_low_f32(_mean), vget_high_f32(_mean));
        _s2 = vpadd_f32(_s2, _s2);
        mean += vget_lane_f32(_s2, 0);
#endif
#endif // __ARM_NEON

        mean = mean / elemcount;
#if __ARM_NEON
        _mean = vdupq_n_f32(mean);
#endif // __ARM_NEON
    }

#if __ARM_NEON
    float32x4_t _var = vdupq_n_f32(0.f);
#endif // __ARM_NEON
    float var = 0.f;
    {
        const float* ptr0 = ptr;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr0);
            _p = vsubq_f32(_p, _mean);
            _var = vmlaq_f32(_var, _p, _p);
            ptr0 += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = ptr0[0] - mean;
            var += v * v;
            ptr0++;
        }
    }

#if __ARM_NEON
    if (elempack == 4)
    {
        float32x4_t _elemcount = vdupq_n_f32(elemcount);
        float32x4_t _eps = vdupq_n_f32(eps);
        _var = div_ps(_var, _elemcount);
        _var = vaddq_f32(_var, _eps);
        float32x4_t _rsqrt_var = vrsqrteq_f32(_var);
        _rsqrt_var = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var, _rsqrt_var), _rsqrt_var), _rsqrt_var);
        _var = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var, _rsqrt_var), _rsqrt_var), _rsqrt_var);
        _mean = vmulq_f32(_mean, _var);
        _mean = vnegq_f32(_mean);
    }
#endif // __ARM_NEON
    if (elempack == 1)
    {
#if __ARM_NEON
#if __aarch64__
        var += vaddvq_f32(_var);
#else
        float32x2_t _s2 = vadd_f32(vget_low_f32(_var), vget_high_f32(_var));
        _s2 = vpadd_f32(_s2, _s2);
        var += vget_lane_f32(_s2, 0);
#endif
#endif // __ARM_NEON

        var = 1.f / sqrtf(var / elemcount + eps);
        mean = -mean * var;
#if __ARM_NEON
        _var = vdupq_n_f32(var);
        _mean = vdupq_n_f32(mean);
#endif // __ARM_NEON
    }

    if (gamma_ptr && beta_ptr)
    {
        int i = 0;
#if __ARM_NEON
        if (elempack == 4)
        {
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                float32x4_t _gamma = vdupq_n_f32(gamma_ptr[0]);
                float32x4_t _beta = vdupq_n_f32(beta_ptr[0]);
                _p = vmlaq_f32(_mean, _p, _var);
                _p = vmlaq_f32(_beta, _p, _gamma);
                vst1q_f32(ptr, _p);
                ptr += 4;
                gamma_ptr += 1;
                beta_ptr += 1;
            }
        }
        if (elempack == 1)
        {
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                float32x4_t _gamma = vld1q_f32(gamma_ptr);
                float32x4_t _beta = vld1q_f32(beta_ptr);
                _p = vmlaq_f32(_mean, _p, _var);
                _p = vmlaq_f32(_beta, _p, _gamma);
                vst1q_f32(ptr, _p);
                ptr += 4;
                gamma_ptr += 4;
                beta_ptr += 4;
            }
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            ptr[0] = (ptr[0] * var + mean) * gamma_ptr[0] + beta_ptr[0];
            ptr++;
            gamma_ptr++;
            beta_ptr++;
        }
    }
    else
    {
        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vmlaq_f32(_mean, _p, _var);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            ptr[0] = ptr[0] * var + mean;
            ptr++;
        }
    }
}

int LayerNorm_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blob, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        // assert affine_size == w

        float* ptr = bottom_top_blob;
        layernorm(ptr, gamma_data, beta_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        // assert affine_size == w

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            layernorm(ptr, gamma_data, beta_data, eps, w, elempack);
        }
    }

    if (dims == 3)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
                    layernorm(ptr, gamma_data, beta_data, eps, w, elempack);
                }
            }
        }
        else // if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                layernorm(ptr, gamma_data, beta_data, eps, w * h, elempack);
            }
        }
    }

    return 0;
}

#if NCNN_BF16
static void layernorm_bf16s(unsigned short* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

#if __ARM_NEON
    float32x4_t _mean = vdupq_n_f32(0.f);
#endif // __ARM_NEON
    float mean = 0.f;
    {
        const unsigned short* ptr0 = ptr;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr0));
            _mean = vaddq_f32(_mean, _p);
            ptr0 += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            mean += bfloat16_to_float32(ptr0[0]);
            ptr0++;
        }
    }

#if __ARM_NEON
    if (elempack == 4)
    {
        float32x4_t _elemcount = vdupq_n_f32(elemcount);
        _mean = div_ps(_mean, _elemcount);
    }
#endif // __ARM_NEON
    if (elempack == 1)
    {
#if __ARM_NEON
#if __aarch64__
        mean += vaddvq_f32(_mean);
#else
        float32x2_t _s2 = vadd_f32(vget_low_f32(_mean), vget_high_f32(_mean));
        _s2 = vpadd_f32(_s2, _s2);
        mean += vget_lane_f32(_s2, 0);
#endif
#endif // __ARM_NEON

        mean = mean / elemcount;
#if __ARM_NEON
        _mean = vdupq_n_f32(mean);
#endif // __ARM_NEON
    }

#if __ARM_NEON
    float32x4_t _var = vdupq_n_f32(0.f);
#endif // __ARM_NEON
    float var = 0.f;
    {
        const unsigned short* ptr0 = ptr;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr0));
            _p = vsubq_f32(_p, _mean);
            _var = vmlaq_f32(_var, _p, _p);
            ptr0 += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr0[0]) - mean;
            var += v * v;
            ptr0++;
        }
    }

#if __ARM_NEON
    if (elempack == 4)
    {
        float32x4_t _elemcount = vdupq_n_f32(elemcount);
        float32x4_t _eps = vdupq_n_f32(eps);
        _var = div_ps(_var, _elemcount);
        _var = vaddq_f32(_var, _eps);
        float32x4_t _rsqrt_var = vrsqrteq_f32(_var);
        _rsqrt_var = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var, _rsqrt_var), _rsqrt_var), _rsqrt_var);
        _var = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_var, _rsqrt_var), _rsqrt_var), _rsqrt_var);
        _mean = vmulq_f32(_mean, _var);
        _mean = vnegq_f32(_mean);
    }
#endif // __ARM_NEON
    if (elempack == 1)
    {
#if __ARM_NEON
#if __aarch64__
        var += vaddvq_f32(_var);
#else
        float32x2_t _s2 = vadd_f32(vget_low_f32(_var), vget_high_f32(_var));
        _s2 = vpadd_f32(_s2, _s2);
        var += vget_lane_f32(_s2, 0);
#endif
#endif // __ARM_NEON

        var = 1.f / sqrtf(var / elemcount + eps);
        mean = -mean * var;
#if __ARM_NEON
        _var = vdupq_n_f32(var);
        _mean = vdupq_n_f32(mean);
#endif // __ARM_NEON
    }

    if (gamma_ptr && beta_ptr)
    {
        int i = 0;
#if __ARM_NEON
        if (elempack == 4)
        {
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = bfloat2float(vld1_u16(ptr));
                float32x4_t _gamma = vdupq_n_f32(gamma_ptr[0]);
                float32x4_t _beta = vdupq_n_f32(beta_ptr[0]);
                _p = vmlaq_f32(_mean, _p, _var);
                _p = vmlaq_f32(_beta, _p, _gamma);
                vst1_u16(ptr, float2bfloat(_p));
                ptr += 4;
                gamma_ptr += 1;
                beta_ptr += 1;
            }
        }
        if (elempack == 1)
        {
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = bfloat2float(vld1_u16(ptr));
                float32x4_t _gamma = vld1q_f32(gamma_ptr);
                float32x4_t _beta = vld1q_f32(beta_ptr);
                _p = vmlaq_f32(_mean, _p, _var);
                _p = vmlaq_f32(_beta, _p, _gamma);
                vst1_u16(ptr, float2bfloat(_p));
                ptr += 4;
                gamma_ptr += 4;
                beta_ptr += 4;
            }
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr[0]);
            ptr[0] = float32_to_bfloat16((v * var + mean) * gamma_ptr[0] + beta_ptr[0]);
            ptr++;
            gamma_ptr++;
            beta_ptr++;
        }
    }
    else
    {
        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            _p = vmlaq_f32(_mean, _p, _var);
            vst1_u16(ptr, float2bfloat(_p));
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr[0]);
            ptr[0] = float32_to_bfloat16(v * var + mean);
            ptr++;
        }
    }
}

int LayerNorm_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        // assert affine_size == w

        unsigned short* ptr = bottom_top_blob;
        layernorm_bf16s(ptr, gamma_data, beta_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        // assert affine_size == w

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            layernorm_bf16s(ptr, gamma_data, beta_data, eps, w, elempack);
        }
    }

    if (dims == 3)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    unsigned short* ptr = bottom_top_blob.channel(q).row<unsigned short>(i);
                    layernorm_bf16s(ptr, gamma_data, beta_data, eps, w, elempack);
                }
            }
        }
        else // if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr = bottom_top_blob.channel(q);
                layernorm_bf16s(ptr, gamma_data, beta_data, eps, w * h, elempack);
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
