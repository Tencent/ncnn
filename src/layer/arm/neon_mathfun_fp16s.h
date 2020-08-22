// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

/* NEON implementation of sin, cos, exp and log
 *
 *   Inspired by Intel Approximate Math library, and based on the
 *   corresponding algorithms of the cephes math library
 */

/* Copyright (C) 2011  Julien Pommier
 *
 *  This software is provided 'as-is', without any express or implied
 *  warranty.  In no event will the authors be held liable for any damages
 *  arising from the use of this software.
 *
 *  Permission is granted to anyone to use this software for any purpose,
 *  including commercial applications, and to alter it and redistribute it
 *  freely, subject to the following restrictions:
 *
 *  1. The origin of this software must not be misrepresented; you must not
 *     claim that you wrote the original software. If you use this software
 *     in a product, an acknowledgment in the product documentation would be
 *     appreciated but is not required.
 *  2. Altered source versions must be plainly marked as such, and must not be
 *     misrepresented as being the original software.
 *  3. This notice may not be removed or altered from any source distribution.
 *
 *  (this is the zlib license)
 */

#include <arm_neon.h>

#define c_exp_hi_f16 10.7421875f
#define c_exp_lo_f16 -10.7421875f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

/* exp() computed for 4 float at once */
static inline float16x4_t exp_ps(float16x4_t x)
{
    float16x4_t tmp, fx;

    float16x4_t one = vdup_n_f16(1);
    x = vmin_f16(x, vdup_n_f16(c_exp_hi_f16));
    x = vmax_f16(x, vdup_n_f16(c_exp_lo_f16));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = vfma_f16(vdup_n_f16(0.5f), x, vdup_n_f16(c_cephes_LOG2EF));

    /* perform a floorf */
    tmp = vcvt_f16_s16(vcvt_s16_f16(fx));

    /* if greater, substract 1 */
    uint16x4_t mask = vcgt_f16(tmp, fx);
    mask = vand_u16(mask, vreinterpret_u16_f16(one));

    fx = vsub_f16(tmp, vreinterpret_f16_u16(mask));

    tmp = vmul_f16(fx, vdup_n_f16(c_cephes_exp_C1));
    float16x4_t z = vmul_f16(fx, vdup_n_f16(c_cephes_exp_C2));
    x = vsub_f16(x, tmp);
    x = vsub_f16(x, z);

    static const __fp16 cephes_exp_p[6] = {c_cephes_exp_p0, c_cephes_exp_p1, c_cephes_exp_p2, c_cephes_exp_p3, c_cephes_exp_p4, c_cephes_exp_p5};
    float16x4_t y = vld1_dup_f16(cephes_exp_p + 0);
    float16x4_t c1 = vld1_dup_f16(cephes_exp_p + 1);
    float16x4_t c2 = vld1_dup_f16(cephes_exp_p + 2);
    float16x4_t c3 = vld1_dup_f16(cephes_exp_p + 3);
    float16x4_t c4 = vld1_dup_f16(cephes_exp_p + 4);
    float16x4_t c5 = vld1_dup_f16(cephes_exp_p + 5);

    y = vmul_f16(y, x);
    z = vmul_f16(x, x);

    y = vadd_f16(y, c1);
    y = vmul_f16(y, x);
    y = vadd_f16(y, c2);
    y = vmul_f16(y, x);
    y = vadd_f16(y, c3);
    y = vmul_f16(y, x);
    y = vadd_f16(y, c4);
    y = vmul_f16(y, x);
    y = vadd_f16(y, c5);

    y = vmul_f16(y, z);
    y = vadd_f16(y, x);
    y = vadd_f16(y, one);

    /* build 2^n */
    int16x4_t mm;
    mm = vcvt_s16_f16(fx);
    mm = vadd_s16(mm, vdup_n_s16(0xf));
    mm = vshl_n_s16(mm, 10);
    float16x4_t pow2n = vreinterpret_f16_s16(mm);

    y = vmul_f16(y, pow2n);
    return y;
}

static inline float16x8_t exp_ps(float16x8_t x)
{
    float16x8_t tmp, fx;

    float16x8_t one = vdupq_n_f16(1);
    x = vminq_f16(x, vdupq_n_f16(c_exp_hi_f16));
    x = vmaxq_f16(x, vdupq_n_f16(c_exp_lo_f16));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = vfmaq_f16(vdupq_n_f16(0.5f), x, vdupq_n_f16(c_cephes_LOG2EF));

    /* perform a floorf */
    tmp = vcvtq_f16_s16(vcvtq_s16_f16(fx));

    /* if greater, substract 1 */
    uint16x8_t mask = vcgtq_f16(tmp, fx);
    mask = vandq_u16(mask, vreinterpretq_u16_f16(one));

    fx = vsubq_f16(tmp, vreinterpretq_f16_u16(mask));

    tmp = vmulq_f16(fx, vdupq_n_f16(c_cephes_exp_C1));
    float16x8_t z = vmulq_f16(fx, vdupq_n_f16(c_cephes_exp_C2));
    x = vsubq_f16(x, tmp);
    x = vsubq_f16(x, z);

    static const __fp16 cephes_exp_p[6] = {c_cephes_exp_p0, c_cephes_exp_p1, c_cephes_exp_p2, c_cephes_exp_p3, c_cephes_exp_p4, c_cephes_exp_p5};
    float16x8_t y = vld1q_dup_f16(cephes_exp_p + 0);
    float16x8_t c1 = vld1q_dup_f16(cephes_exp_p + 1);
    float16x8_t c2 = vld1q_dup_f16(cephes_exp_p + 2);
    float16x8_t c3 = vld1q_dup_f16(cephes_exp_p + 3);
    float16x8_t c4 = vld1q_dup_f16(cephes_exp_p + 4);
    float16x8_t c5 = vld1q_dup_f16(cephes_exp_p + 5);

    y = vmulq_f16(y, x);
    z = vmulq_f16(x, x);

    y = vaddq_f16(y, c1);
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, c2);
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, c3);
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, c4);
    y = vmulq_f16(y, x);
    y = vaddq_f16(y, c5);

    y = vmulq_f16(y, z);
    y = vaddq_f16(y, x);
    y = vaddq_f16(y, one);

    /* build 2^n */
    int16x8_t mm;
    mm = vcvtq_s16_f16(fx);
    mm = vaddq_s16(mm, vdupq_n_s16(0xf));
    mm = vshlq_n_s16(mm, 10);
    float16x8_t pow2n = vreinterpretq_f16_s16(mm);

    y = vmulq_f16(y, pow2n);
    return y;
}

static inline float16x4_t sigmoid_ps(float16x4_t _v)
{
    float16x4_t _one = vdup_n_f16(1.f);
    _v = vneg_f16(_v);
    _v = exp_ps(_v);
    _v = vadd_f16(_v, _one);
    return vdiv_f16(_one, _v);
}

static inline float16x8_t sigmoid_ps(float16x8_t _v)
{
    float16x8_t _one = vdupq_n_f16(1.f);
    _v = vnegq_f16(_v);
    _v = exp_ps(_v);
    _v = vaddq_f16(_v, _one);
    return vdivq_f16(_one, _v);
}
