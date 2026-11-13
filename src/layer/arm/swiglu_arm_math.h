// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SWIGLU_ARM_MATH_H
#define LAYER_SWIGLU_ARM_MATH_H

#include <arm_neon.h>

#include "neon_mathfun.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "neon_mathfun_fp16s.h"
#endif

namespace ncnn {

// y = silu(scale * min(gate, limit)) * (clamp(up, -limit, limit) + offset)
static NCNN_FORCEINLINE float32x4_t swiglu_f32x4(float32x4_t _g, float32x4_t _u, float32x4_t _scale, float32x4_t _offset, float32x4_t _limit, float32x4_t _mlimit)
{
    _g = vminq_f32(_g, _limit);
    _u = vminq_f32(vmaxq_f32(_u, _mlimit), _limit);

    // silu(scale * g) = g / (1 + exp(-scale * g))
    _g = div_ps(_g, vaddq_f32(vdupq_n_f32(1.f), exp_ps(vnegq_f32(vmulq_f32(_scale, _g)))));

    return vmulq_f32(_g, vaddq_f32(_u, _offset));
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static NCNN_FORCEINLINE float16x4_t swiglu_f16x4(float16x4_t _g, float16x4_t _u, float16x4_t _scale, float16x4_t _offset, float16x4_t _limit, float16x4_t _mlimit, float16x4_t _one)
{
    _g = vmin_f16(_g, _limit);
    _u = vmin_f16(vmax_f16(_u, _mlimit), _limit);

    // silu(scale * g) = g * sigmoid(scale * g)
    _g = vmul_f16(_g, vdiv_f16(_one, vadd_f16(_one, exp_ps_f16(vneg_f16(vmul_f16(_scale, _g))))));

    return vmul_f16(_g, vadd_f16(_u, _offset));
}

static NCNN_FORCEINLINE float16x8_t swiglu_f16x8(float16x8_t _g, float16x8_t _u, float16x8_t _scale, float16x8_t _offset, float16x8_t _limit, float16x8_t _mlimit, float16x8_t _one)
{
    _g = vminq_f16(_g, _limit);
    _u = vminq_f16(vmaxq_f16(_u, _mlimit), _limit);

    // silu(scale * g) = g * sigmoid(scale * g)
    _g = vmulq_f16(_g, vdivq_f16(_one, vaddq_f16(_one, exp_ps_f16(vnegq_f16(vmulq_f16(_scale, _g))))));

    return vmulq_f16(_g, vaddq_f16(_u, _offset));
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn

#endif // LAYER_SWIGLU_ARM_MATH_H
