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

#include "rmsnorm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"
#include "cpu.h"

namespace ncnn {

RMSNorm_arm::RMSNorm_arm()
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

static void rmsnorm(float* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

#if __ARM_NEON
    float32x4_t _rms = vdupq_n_f32(0.f);
#endif // __ARM_NEON
    float rms = 0.f;
    {
        const float* ptr0 = ptr;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr0);
            _rms = vmlaq_f32(_rms, _p, _p);
            ptr0 += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            rms += ptr0[0] * ptr0[0];
            ptr0++;
        }
    }

#if __ARM_NEON
    if (elempack == 4)
    {
        float32x4_t _elemcount = vdupq_n_f32(elemcount);
        float32x4_t _eps = vdupq_n_f32(eps);

#if __aarch64__
        _rms = vdivq_f32(_rms, _elemcount);
        _rms = vaddq_f32(_rms, _eps);
#else
        float32x4_t _inv_elemcount = vrecpeq_f32(_elemcount);
        _inv_elemcount = vmulq_f32(vrecpsq_f32(_elemcount, _inv_elemcount), _inv_elemcount);
        _inv_elemcount = vmulq_f32(vrecpsq_f32(_elemcount, _inv_elemcount), _inv_elemcount);
        _rms = vmlaq_f32(_eps, _rms, _inv_elemcount);
#endif

        float32x4_t _rsqrt_rms = vrsqrteq_f32(_rms);
        _rsqrt_rms = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_rms, _rsqrt_rms), _rsqrt_rms), _rsqrt_rms);
        _rms = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_rms, _rsqrt_rms), _rsqrt_rms), _rsqrt_rms);
    }
#endif // __ARM_NEON
    if (elempack == 1)
    {
#if __ARM_NEON
#if __aarch64__
        rms += vaddvq_f32(_rms);
#else
        float32x2_t _s2 = vadd_f32(vget_low_f32(_rms), vget_high_f32(_rms));
        _s2 = vpadd_f32(_s2, _s2);
        rms += vget_lane_f32(_s2, 0);
#endif
#endif // __ARM_NEON

        rms = 1.f / sqrtf(rms / elemcount + eps);
#if __ARM_NEON
        _rms = vdupq_n_f32(rms);
#endif // __ARM_NEON
    }

    if (gamma_ptr)
    {
        int i = 0;
#if __ARM_NEON
        if (elempack == 4)
        {
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                float32x4_t _gamma = vdupq_n_f32(gamma_ptr[0]);
                _p = vmulq_f32(_p, _rms);
                _p = vmulq_f32(_p, _gamma);
                vst1q_f32(ptr, _p);
                ptr += 4;
                gamma_ptr += 1;
            }
        }
        if (elempack == 1)
        {
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vld1q_f32(ptr);
                float32x4_t _gamma = vld1q_f32(gamma_ptr);
                _p = vmulq_f32(_p, _rms);
                _p = vmulq_f32(_p, _gamma);
                vst1q_f32(ptr, _p);
                ptr += 4;
                gamma_ptr += 4;
            }
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            ptr[0] = (ptr[0] * rms) * gamma_ptr[0];
            ptr++;
            gamma_ptr++;
        }
    }
    else
    {
        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = vmulq_f32(_p, _rms);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            ptr[0] = ptr[0] * rms;
            ptr++;
        }
    }
}

int RMSNorm_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
        rmsnorm(ptr, gamma_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        // assert affine_size == w

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            rmsnorm(ptr, gamma_data, eps, w, elempack);
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
                    rmsnorm(ptr, gamma_data, eps, w, elempack);
                }
            }
        }
        else // if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                rmsnorm(ptr, gamma_data, eps, w * h, elempack);
            }
        }
    }

    return 0;
}

#if NCNN_BF16
static void rmsnorm_bf16s(unsigned short* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

#if __ARM_NEON
    float32x4_t _rms = vdupq_n_f32(0.f);
#endif // __ARM_NEON
    float rms = 0.f;
    {
        const unsigned short* ptr0 = ptr;

        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr0));
            _rms = vmlaq_f32(_rms, _p, _p);
            ptr0 += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr0[0]);
            rms += v * v;
            ptr0++;
        }
    }

#if __ARM_NEON
    if (elempack == 4)
    {
        float32x4_t _elemcount = vdupq_n_f32(elemcount);
        float32x4_t _eps = vdupq_n_f32(eps);

#if __aarch64__
        _rms = vdivq_f32(_rms, _elemcount);
        _rms = vaddq_f32(_rms, _eps);
#else
        float32x4_t _inv_elemcount = vrecpeq_f32(_elemcount);
        _inv_elemcount = vmulq_f32(vrecpsq_f32(_elemcount, _inv_elemcount), _inv_elemcount);
        _inv_elemcount = vmulq_f32(vrecpsq_f32(_elemcount, _inv_elemcount), _inv_elemcount);
        _rms = vmlaq_f32(_eps, _rms, _inv_elemcount);
#endif

        float32x4_t _rsqrt_rms = vrsqrteq_f32(_rms);
        _rsqrt_rms = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_rms, _rsqrt_rms), _rsqrt_rms), _rsqrt_rms);
        _rms = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_rms, _rsqrt_rms), _rsqrt_rms), _rsqrt_rms);
    }
#endif // __ARM_NEON
    if (elempack == 1)
    {
#if __ARM_NEON
#if __aarch64__
        rms += vaddvq_f32(_rms);
#else
        float32x2_t _s2 = vadd_f32(vget_low_f32(_rms), vget_high_f32(_rms));
        _s2 = vpadd_f32(_s2, _s2);
        rms += vget_lane_f32(_s2, 0);
#endif
#endif // __ARM_NEON

        rms = 1.f / sqrtf(rms / elemcount + eps);
#if __ARM_NEON
        _rms = vdupq_n_f32(rms);
#endif // __ARM_NEON
    }

    if (gamma_ptr)
    {
        int i = 0;
#if __ARM_NEON
        if (elempack == 4)
        {
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = bfloat2float(vld1_u16(ptr));
                float32x4_t _gamma = vdupq_n_f32(gamma_ptr[0]);
                _p = vmulq_f32(_p, _rms);
                _p = vmulq_f32(_p, _gamma);
                vst1_u16(ptr, float2bfloat(_p));
                ptr += 4;
                gamma_ptr += 1;
            }
        }
        if (elempack == 1)
        {
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = bfloat2float(vld1_u16(ptr));
                float32x4_t _gamma = vld1q_f32(gamma_ptr);
                _p = vmulq_f32(_p, _rms);
                _p = vmulq_f32(_p, _gamma);
                vst1_u16(ptr, float2bfloat(_p));
                ptr += 4;
                gamma_ptr += 4;
            }
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr[0]);
            ptr[0] = float32_to_bfloat16((v * rms) * gamma_ptr[0]);
            ptr++;
            gamma_ptr++;
        }
    }
    else
    {
        int i = 0;
#if __ARM_NEON
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            _p = vmulq_f32(_p, _rms);
            vst1_u16(ptr, float2bfloat(_p));
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr[0]);
            ptr[0] = float32_to_bfloat16(v * rms);
            ptr++;
        }
    }
}

int RMSNorm_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
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
        rmsnorm_bf16s(ptr, gamma_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        // assert affine_size == w

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            rmsnorm_bf16s(ptr, gamma_data, eps, w, elempack);
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
                    rmsnorm_bf16s(ptr, gamma_data, eps, w, elempack);
                }
            }
        }
        else // if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr = bottom_top_blob.channel(q);
                rmsnorm_bf16s(ptr, gamma_data, eps, w * h, elempack);
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
