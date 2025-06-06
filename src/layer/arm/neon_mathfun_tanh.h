// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef NEON_MATHFUN_TANH_H
#define NEON_MATHFUN_TANH_H

#include <arm_neon.h>

#define c_tanh_tiny 1e-4f
#define c_tanh_hi   9.0f
// The monomial coefficients of the numerator polynomial (odd).
#define c_tanh_alpha_1  4.89352455891786e-3f
#define c_tanh_alpha_3  6.37261928875436e-4f
#define c_tanh_alpha_5  1.48572235717979e-5f
#define c_tanh_alpha_7  5.12229709037114e-8f
#define c_tanh_alpha_9  -8.60467152213735e-11f
#define c_tanh_alpha_11 2.00018790482477e-13f
#define c_tanh_alpha_13 -2.76076847742355e-16f
// The monomial coefficients of the denominator polynomial (even).
#define c_tanh_beta_0 4.89352518554385e-3f
#define c_tanh_beta_2 2.26843463243900e-3f
#define c_tanh_beta_4 1.18534705686654e-4f
#define c_tanh_beta_6 1.19825839466702e-6f

/* Single precision hyperbolic tangent computed for 4 simultaneous float */
static inline float32x4_t tanh_ps(float32x4_t x)
{
    float32x4_t x2 = vabsq_f32(x);

    uint32x4_t tiny_mask = vcgeq_f32(x2, vdupq_n_f32(c_tanh_tiny));

    // clamp the inputs to the range [-9, 9] since anything outside
    // this range is -/+1.0f in single-precision.
    x2 = vreinterpretq_f32_u32(vbslq_u32(vcgeq_f32(vdupq_n_f32(c_tanh_hi), x2), vreinterpretq_u32_f32(x2), vreinterpretq_u32_f32(vdupq_n_f32(c_tanh_hi))));

    // since the polynomials are odd/even, we need x**2.
    float32x4_t z = vmulq_f32(x2, x2);

    // evaluate the numerator polynomial y.
    float32x4_t y = vdupq_n_f32(c_tanh_alpha_13);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_11), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_9), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_7), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_5), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_3), y, z);
    y = vmlaq_f32(vdupq_n_f32(c_tanh_alpha_1), y, z);
    y = vmulq_f32(y, x2);

    // evaluate the denominator polynomial w.
    float32x4_t w = vdupq_n_f32(c_tanh_beta_6);
    w = vmlaq_f32(vdupq_n_f32(c_tanh_beta_4), w, z);
    w = vmlaq_f32(vdupq_n_f32(c_tanh_beta_2), w, z);
    w = vmlaq_f32(vdupq_n_f32(c_tanh_beta_0), w, z);

    // divide the numerator by the denominator.
#if __aarch64__
    y = vdivq_f32(y, w);
#else
    y = div_ps(y, w);
#endif

    // reinstate the sign.
    y = vreinterpretq_f32_u32(vbslq_u32(vdupq_n_u32(1u << 31), vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)));

    // when the argument is very small in magnitude it's more accurate to just return it.
    y = vreinterpretq_f32_u32(vbslq_u32(tiny_mask, vreinterpretq_u32_f32(y), vreinterpretq_u32_f32(x)));

    return y;
}

#endif // NEON_MATHFUN_TANH_H
