// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "unaryop_arm.h"

// #include <fenv.h>
#include <float.h>

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "neon_mathfun_fp16s.h"
#endif
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template<typename Op>
static int unary_op_inplace_fp16s(Mat& a, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = a.channel(q);

        int i = 0;
        for (; i + 15 < size; i += 16)
        {
            float16x8_t _p0 = vld1q_f16(ptr);
            float16x8_t _p1 = vld1q_f16(ptr + 8);
            _p0 = op.func_pack8(_p0);
            _p1 = op.func_pack8(_p1);
            vst1q_f16(ptr, _p0);
            vst1q_f16(ptr + 8, _p1);
            ptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = op.func_pack8(_p);
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = op.func_pack4(_p);
            vst1_f16(ptr, _p);
            ptr += 4;
        }
        for (; i < size; i++)
        {
            *ptr = op.func(*ptr);
            ptr++;
        }
    }

    return 0;
}

namespace UnaryOp_arm_functor {

struct unary_op_abs_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)fabsf(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return vabs_f16(x);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return vabsq_f16(x);
    }
};

struct unary_op_neg_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return -x;
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return vneg_f16(x);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return vnegq_f16(x);
    }
};

struct unary_op_floor_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)floorf(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return vcvt_f16_s16(vcvtm_s16_f16(x));
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return vcvtq_f16_s16(vcvtmq_s16_f16(x));
    }
};

struct unary_op_ceil_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)ceilf(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return vcvt_f16_s16(vcvtp_s16_f16(x));
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return vcvtq_f16_s16(vcvtpq_s16_f16(x));
    }
};

struct unary_op_square_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return x * x;
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return vmul_f16(x, x);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return vmulq_f16(x, x);
    }
};

struct unary_op_sqrt_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)sqrtf(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return vsqrt_f16(x);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return vsqrtq_f16(x);
    }
};

struct unary_op_rsqrt_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)1.f / (__fp16)sqrtf(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        float16x4_t _reciprocal = vrsqrte_f16(x);
        _reciprocal = vmul_f16(vrsqrts_f16(vmul_f16(x, _reciprocal), _reciprocal), _reciprocal);
        // _reciprocal = vmul_f16(vrsqrts_f16(vmul_f16(x, _reciprocal), _reciprocal), _reciprocal);
        return _reciprocal;
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        float16x8_t _reciprocal = vrsqrteq_f16(x);
        _reciprocal = vmulq_f16(vrsqrtsq_f16(vmulq_f16(x, _reciprocal), _reciprocal), _reciprocal);
        // _reciprocal = vmulq_f16(vrsqrtsq_f16(vmulq_f16(x, _reciprocal), _reciprocal), _reciprocal);
        return _reciprocal;
    }
};

struct unary_op_exp_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)expf(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return exp_ps_f16(x);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return exp_ps_f16(x);
    }
};

struct unary_op_log_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)logf(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return log_ps_f16(x);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return log_ps_f16(x);
    }
};

struct unary_op_sin_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)sinf(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return sin_ps_f16(x);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return sin_ps_f16(x);
    }
};

struct unary_op_cos_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)cosf(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return cos_ps_f16(x);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return cos_ps_f16(x);
    }
};

struct unary_op_tan_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)tanf(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        // TODO neon optimize
        __fp16 tmp[4];
        vst1_f16(tmp, x);
        tmp[0] = tanf(tmp[0]);
        tmp[1] = tanf(tmp[1]);
        tmp[2] = tanf(tmp[2]);
        tmp[3] = tanf(tmp[3]);
        return vld1_f16(tmp);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        // TODO neon optimize
        __fp16 tmp[8];
        vst1q_f16(tmp, x);
        tmp[0] = tanf(tmp[0]);
        tmp[1] = tanf(tmp[1]);
        tmp[2] = tanf(tmp[2]);
        tmp[3] = tanf(tmp[3]);
        tmp[4] = tanf(tmp[4]);
        tmp[5] = tanf(tmp[5]);
        tmp[6] = tanf(tmp[6]);
        tmp[7] = tanf(tmp[7]);
        return vld1q_f16(tmp);
    }
};

struct unary_op_asin_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)asinf(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        // TODO neon optimize
        __fp16 tmp[4];
        vst1_f16(tmp, x);
        tmp[0] = asinf(tmp[0]);
        tmp[1] = asinf(tmp[1]);
        tmp[2] = asinf(tmp[2]);
        tmp[3] = asinf(tmp[3]);
        return vld1_f16(tmp);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        // TODO neon optimize
        __fp16 tmp[8];
        vst1q_f16(tmp, x);
        tmp[0] = asinf(tmp[0]);
        tmp[1] = asinf(tmp[1]);
        tmp[2] = asinf(tmp[2]);
        tmp[3] = asinf(tmp[3]);
        tmp[4] = asinf(tmp[4]);
        tmp[5] = asinf(tmp[5]);
        tmp[6] = asinf(tmp[6]);
        tmp[7] = asinf(tmp[7]);
        return vld1q_f16(tmp);
    }
};

struct unary_op_acos_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)acosf(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        // TODO neon optimize
        __fp16 tmp[4];
        vst1_f16(tmp, x);
        tmp[0] = acosf(tmp[0]);
        tmp[1] = acosf(tmp[1]);
        tmp[2] = acosf(tmp[2]);
        tmp[3] = acosf(tmp[3]);
        return vld1_f16(tmp);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        // TODO neon optimize
        __fp16 tmp[8];
        vst1q_f16(tmp, x);
        tmp[0] = acosf(tmp[0]);
        tmp[1] = acosf(tmp[1]);
        tmp[2] = acosf(tmp[2]);
        tmp[3] = acosf(tmp[3]);
        tmp[4] = acosf(tmp[4]);
        tmp[5] = acosf(tmp[5]);
        tmp[6] = acosf(tmp[6]);
        tmp[7] = acosf(tmp[7]);
        return vld1q_f16(tmp);
    }
};

struct unary_op_atan_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)atanf(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        // TODO neon optimize
        __fp16 tmp[4];
        vst1_f16(tmp, x);
        tmp[0] = atanf(tmp[0]);
        tmp[1] = atanf(tmp[1]);
        tmp[2] = atanf(tmp[2]);
        tmp[3] = atanf(tmp[3]);
        return vld1_f16(tmp);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        // TODO neon optimize
        __fp16 tmp[8];
        vst1q_f16(tmp, x);
        tmp[0] = atanf(tmp[0]);
        tmp[1] = atanf(tmp[1]);
        tmp[2] = atanf(tmp[2]);
        tmp[3] = atanf(tmp[3]);
        tmp[4] = atanf(tmp[4]);
        tmp[5] = atanf(tmp[5]);
        tmp[6] = atanf(tmp[6]);
        tmp[7] = atanf(tmp[7]);
        return vld1q_f16(tmp);
    }
};

struct unary_op_reciprocal_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)1.f / x;
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        float16x4_t _reciprocal = vrecpe_f16(x);
        _reciprocal = vmul_f16(vrecps_f16(x, _reciprocal), _reciprocal);
        // _reciprocal = vmul_f16(vrecps_f16(x, _reciprocal), _reciprocal);
        return _reciprocal;
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        float16x8_t _reciprocal = vrecpeq_f16(x);
        _reciprocal = vmulq_f16(vrecpsq_f16(x, _reciprocal), _reciprocal);
        // _reciprocal = vmulq_f16(vrecpsq_f16(x, _reciprocal), _reciprocal);
        return _reciprocal;
    }
};

struct unary_op_tanh_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)tanhf(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return tanh_ps_f16(x);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return tanh_ps_f16(x);
    }
};

struct unary_op_log10_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)log10f(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return vmul_f16(log_ps_f16(x), vdup_n_f16(0.434294481903));
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return vmulq_f16(log_ps_f16(x), vdupq_n_f16(0.434294481903));
    }
};

struct unary_op_round_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        // round to nearest even
#if NCNN_GNU_INLINE_ASM
        // return (x + 1536.f) - 1536.f;
        __fp16 y;
        const __fp16 magic = 1536.f;
        asm volatile(
            "fadd   %h0, %h1, %h2  \n"
            "fsub   %h0, %h0, %h2  \n"
            : "=w"(y)
            : "w"(x), "w"(magic)
            :);
        return y;
#else
#ifdef FE_TONEAREST
        int old_rm = fegetround();
        fesetround(FE_TONEAREST);
#endif
        __fp16 y = (__fp16)nearbyintf(x);
#ifdef FE_TONEAREST
        fesetround(old_rm);
#endif
        return y;
#endif
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return vrndn_f16(x);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return vrndnq_f16(x);
    }
};

struct unary_op_trunc_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)truncf(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return vrnd_f16(x);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return vrndq_f16(x);
    }
};

} // namespace UnaryOp_arm_functor

int UnaryOp_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    using namespace UnaryOp_arm_functor;

    if (op_type == Operation_ABS)
        return unary_op_inplace_fp16s<unary_op_abs_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_NEG)
        return unary_op_inplace_fp16s<unary_op_neg_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_FLOOR)
        return unary_op_inplace_fp16s<unary_op_floor_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_CEIL)
        return unary_op_inplace_fp16s<unary_op_ceil_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_SQUARE)
        return unary_op_inplace_fp16s<unary_op_square_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_SQRT)
        return unary_op_inplace_fp16s<unary_op_sqrt_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_RSQRT)
        return unary_op_inplace_fp16s<unary_op_rsqrt_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_EXP)
        return unary_op_inplace_fp16s<unary_op_exp_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_LOG)
        return unary_op_inplace_fp16s<unary_op_log_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_SIN)
        return unary_op_inplace_fp16s<unary_op_sin_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_COS)
        return unary_op_inplace_fp16s<unary_op_cos_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_TAN)
        return unary_op_inplace_fp16s<unary_op_tan_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_ASIN)
        return unary_op_inplace_fp16s<unary_op_asin_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_ACOS)
        return unary_op_inplace_fp16s<unary_op_acos_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_ATAN)
        return unary_op_inplace_fp16s<unary_op_atan_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_RECIPROCAL)
        return unary_op_inplace_fp16s<unary_op_reciprocal_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_TANH)
        return unary_op_inplace_fp16s<unary_op_tanh_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_LOG10)
        return unary_op_inplace_fp16s<unary_op_log10_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_ROUND)
        return unary_op_inplace_fp16s<unary_op_round_fp16s>(bottom_top_blob, opt);

    if (op_type == Operation_TRUNC)
        return unary_op_inplace_fp16s<unary_op_trunc_fp16s>(bottom_top_blob, opt);

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
