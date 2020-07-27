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

#include <arm_neon.h>

// tanh neon vector version
// refer the scalar version from Cephes Math Library

#define c_cephes_HALFMAXLOGF 44.014845935754205f
#define c_cephes_tanh_C1     0.625f

#define c_cephes_tanh_p0 -5.70498872745E-3
#define c_cephes_tanh_p1 +2.06390887954E-2
#define c_cephes_tanh_p2 -5.37397155531E-2
#define c_cephes_tanh_p3 +1.33314422036E-1
#define c_cephes_tanh_p4 -3.33332819422E-1

/* Single precision hyperbolic tangent computed for 4 simultaneous float */
static inline float32x4_t tanh_ps(float32x4_t x)
{
    float32x4_t x2 = vabsq_f32(x);

    uint32x4_t mask_l = vcgeq_f32(x2, vdupq_n_f32(c_cephes_tanh_C1));
    uint32x4_t mask_l2 = vcgtq_f32(x2, vdupq_n_f32(c_cephes_HALFMAXLOGF));

    // abs(x) >= 0.625
    // tanh(x) = 1 − 2 / (exp(2x) + 1)
    float32x4_t _one = vdupq_n_f32(1.f);
    float32x4_t _two = vdupq_n_f32(2.f);
    float32x4_t exp_x_x = exp_ps(vaddq_f32(x, x));
#if __aarch64__
    float32x4_t y0 = vsubq_f32(_one, vdivq_f32(_two, vaddq_f32(exp_x_x, _one)));
#else
    float32x4_t y0 = vsubq_f32(_one, div_ps(_two, vaddq_f32(exp_x_x, _one)));
#endif

    // abs(x) < 0.625
    /*
        z = x2 * x2;
        z =
        (((( -5.70498872745E-3 * z
        + 2.06390887954E-2) * z
        - 5.37397155531E-2) * z
        + 1.33314422036E-1) * z
        - 3.33332819422E-1) * z * x
        + x;
    */
    static const float cephes_tanh_p[5] = {c_cephes_tanh_p0, c_cephes_tanh_p1, c_cephes_tanh_p2, c_cephes_tanh_p3, c_cephes_tanh_p4};
    float32x4_t y = vld1q_dup_f32(cephes_tanh_p + 0);
    float32x4_t c1 = vld1q_dup_f32(cephes_tanh_p + 1);
    float32x4_t c2 = vld1q_dup_f32(cephes_tanh_p + 2);
    float32x4_t c3 = vld1q_dup_f32(cephes_tanh_p + 3);
    float32x4_t c4 = vld1q_dup_f32(cephes_tanh_p + 4);

    float32x4_t z = vmulq_f32(x, x);

    y = vmulq_f32(y, z);
    y = vaddq_f32(y, c1);
    y = vmulq_f32(y, z);
    y = vaddq_f32(y, c2);
    y = vmulq_f32(y, z);
    y = vaddq_f32(y, c3);
    y = vmulq_f32(y, z);
    y = vaddq_f32(y, c4);

    y = vmulq_f32(y, z);
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, x);

    // abs(x) > HALFMAXLOGF
    // return 1.0 or -1.0
    uint32x4_t mask_pos = vcgtq_f32(x2, vdupq_n_f32(0.f));
    float32x4_t y1 = vreinterpretq_f32_u32(vbslq_u32(mask_pos, vreinterpretq_u32_f32(vdupq_n_f32(1.f)), vreinterpretq_u32_f32(vdupq_n_f32(-1.f))));

    y = vreinterpretq_f32_u32(vbslq_u32(mask_l, vreinterpretq_u32_f32(y0), vreinterpretq_u32_f32(y)));
    y = vreinterpretq_f32_u32(vbslq_u32(mask_l2, vreinterpretq_u32_f32(y1), vreinterpretq_u32_f32(y)));
    return y;
}
