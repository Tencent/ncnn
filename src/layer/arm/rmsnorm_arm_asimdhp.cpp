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
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static void rmsnorm_fp16s(__fp16* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    float32x4_t _rms0 = vdupq_n_f32(0.f);
    float32x4_t _rms1 = vdupq_n_f32(0.f);
    float rms = 0.f;
    {
        const __fp16* ptr0 = ptr;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr0);
            float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
            float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
            _rms0 = vmlaq_f32(_rms0, _p0, _p0);
            _rms1 = vmlaq_f32(_rms1, _p1, _p1);
            ptr0 += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr0));
            _rms0 = vmlaq_f32(_rms0, _p, _p);
            ptr0 += 4;
        }
        for (; i < size; i++)
        {
            rms += (float)ptr0[0] * (float)ptr0[0];
            ptr0++;
        }
    }

    if (elempack == 8)
    {
        float32x4_t _elemcount = vdupq_n_f32(elemcount);
        float32x4_t _eps = vdupq_n_f32(eps);

        _rms0 = vdivq_f32(_rms0, _elemcount);
        _rms1 = vdivq_f32(_rms1, _elemcount);
        _rms0 = vaddq_f32(_rms0, _eps);
        _rms1 = vaddq_f32(_rms1, _eps);

        float32x4_t _rsqrt_rms0 = vrsqrteq_f32(_rms0);
        float32x4_t _rsqrt_rms1 = vrsqrteq_f32(_rms1);
        _rsqrt_rms0 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_rms0, _rsqrt_rms0), _rsqrt_rms0), _rsqrt_rms0);
        _rsqrt_rms1 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_rms1, _rsqrt_rms1), _rsqrt_rms1), _rsqrt_rms1);
        _rms0 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_rms0, _rsqrt_rms0), _rsqrt_rms0), _rsqrt_rms0);
        _rms1 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_rms1, _rsqrt_rms1), _rsqrt_rms1), _rsqrt_rms1);
    }
    if (elempack == 4)
    {
        _rms0 = vaddq_f32(_rms0, _rms1);

        float32x4_t _elemcount = vdupq_n_f32(elemcount);
        float32x4_t _eps = vdupq_n_f32(eps);

        _rms0 = vdivq_f32(_rms0, _elemcount);
        _rms0 = vaddq_f32(_rms0, _eps);

        float32x4_t _rsqrt_rms0 = vrsqrteq_f32(_rms0);
        _rsqrt_rms0 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_rms0, _rsqrt_rms0), _rsqrt_rms0), _rsqrt_rms0);
        _rms0 = vmulq_f32(vrsqrtsq_f32(vmulq_f32(_rms0, _rsqrt_rms0), _rsqrt_rms0), _rsqrt_rms0);
        _rms1 = _rms0;
    }
    if (elempack == 1)
    {
        _rms0 = vaddq_f32(_rms0, _rms1);
        rms += vaddvq_f32(_rms0);

        rms = 1.f / sqrtf(rms / elemcount + eps);
        _rms0 = vdupq_n_f32(rms);
        _rms1 = _rms0;
    }

    if (gamma_ptr)
    {
        int i = 0;
        if (elempack == 8)
        {
            for (; i + 7 < size; i += 8)
            {
                float16x8_t _p = vld1q_f16(ptr);
                float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
                float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
                float32x4_t _gamma = vdupq_n_f32(gamma_ptr[0]);
                _p0 = vmulq_f32(_p0, _rms0);
                _p1 = vmulq_f32(_p1, _rms1);
                _p0 = vmulq_f32(_p0, _gamma);
                _p1 = vmulq_f32(_p1, _gamma);
                _p = vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1));
                vst1q_f16(ptr, _p);
                ptr += 8;
                gamma_ptr += 1;
            }
        }
        if (elempack == 4)
        {
            for (; i + 7 < size; i += 8)
            {
                float16x8_t _p = vld1q_f16(ptr);
                float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
                float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
                float32x4_t _gamma0 = vdupq_n_f32(gamma_ptr[0]);
                float32x4_t _gamma1 = vdupq_n_f32(gamma_ptr[1]);
                _p0 = vmulq_f32(_p0, _rms0);
                _p1 = vmulq_f32(_p1, _rms1);
                _p0 = vmulq_f32(_p0, _gamma0);
                _p1 = vmulq_f32(_p1, _gamma1);
                _p = vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1));
                vst1q_f16(ptr, _p);
                ptr += 8;
                gamma_ptr += 2;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                float32x4_t _gamma = vdupq_n_f32(gamma_ptr[0]);
                _p = vmulq_f32(_p, _rms0);
                _p = vmulq_f32(_p, _gamma);
                vst1_f16(ptr, vcvt_f16_f32(_p));
                ptr += 4;
                gamma_ptr += 1;
            }
        }
        if (elempack == 1)
        {
            for (; i + 7 < size; i += 8)
            {
                float16x8_t _p = vld1q_f16(ptr);
                float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
                float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
                float32x4_t _gamma0 = vld1q_f32(gamma_ptr);
                float32x4_t _gamma1 = vld1q_f32(gamma_ptr + 4);
                _p0 = vmulq_f32(_p0, _rms0);
                _p1 = vmulq_f32(_p1, _rms1);
                _p0 = vmulq_f32(_p0, _gamma0);
                _p1 = vmulq_f32(_p1, _gamma1);
                _p = vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1));
                vst1q_f16(ptr, _p);
                ptr += 8;
                gamma_ptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
                float32x4_t _gamma = vld1q_f32(gamma_ptr);
                _p = vmulq_f32(_p, _rms0);
                _p = vmulq_f32(_p, _gamma);
                vst1_f16(ptr, vcvt_f16_f32(_p));
                ptr += 4;
                gamma_ptr += 4;
            }
        }
        for (; i < size; i++)
        {
            ptr[0] = (__fp16)(((float)ptr[0] * rms) * gamma_ptr[0]);
            ptr++;
            gamma_ptr++;
        }
    }
    else
    {
        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            float32x4_t _p0 = vcvt_f32_f16(vget_low_f16(_p));
            float32x4_t _p1 = vcvt_f32_f16(vget_high_f16(_p));
            _p0 = vmulq_f32(_p0, _rms0);
            _p1 = vmulq_f32(_p1, _rms1);
            _p = vcombine_f16(vcvt_f16_f32(_p0), vcvt_f16_f32(_p1));
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vcvt_f32_f16(vld1_f16(ptr));
            _p = vmulq_f32(_p, _rms0);
            vst1_f16(ptr, vcvt_f16_f32(_p));
            ptr += 4;
        }
        for (; i < size; i++)
        {
            ptr[0] = (__fp16)((float)ptr[0] * rms);
            ptr++;
        }
    }
}

int RMSNorm_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        // assert affine_size == w

        __fp16* ptr = bottom_top_blob;
        rmsnorm_fp16s(ptr, gamma_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        // assert affine_size == w

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            __fp16* ptr = bottom_top_blob.row<__fp16>(i);
            rmsnorm_fp16s(ptr, gamma_data, eps, w, elempack);
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
                    __fp16* ptr = bottom_top_blob.channel(q).row<__fp16>(i);
                    rmsnorm_fp16s(ptr, gamma_data, eps, w, elempack);
                }
            }
        }
        else // if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                __fp16* ptr = bottom_top_blob.channel(q);
                rmsnorm_fp16s(ptr, gamma_data, eps, w * h, elempack);
            }
        }
    }

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
