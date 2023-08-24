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

#include "unaryop_arm.h"

#include <fenv.h>
#include <float.h>
#include <math.h>

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

#include "arm_usability.h"

#include "cpu.h"

namespace ncnn {

UnaryOp_arm::UnaryOp_arm()
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

template<typename Op>
static int unary_op_inplace(Mat& a, const Option& opt)
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
        float* ptr = a.channel(q);

        int i = 0;
#if __ARM_NEON
        for (; i + 7 < size; i += 8)
        {
            float32x4_t _p0 = vld1q_f32(ptr);
            float32x4_t _p1 = vld1q_f32(ptr + 4);
            _p0 = op.func_pack4(_p0);
            _p1 = op.func_pack4(_p1);
            vst1q_f32(ptr, _p0);
            vst1q_f32(ptr + 4, _p1);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = op.func_pack4(_p);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            *ptr = op.func(*ptr);
            ptr++;
        }
    }

    return 0;
}

namespace UnaryOp_arm_functor {

struct unary_op_abs
{
    float func(const float& x) const
    {
        return (float)fabsf(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        return vabsq_f32(x);
    }
#endif // __ARM_NEON
};

struct unary_op_neg
{
    float func(const float& x) const
    {
        return -x;
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        return vnegq_f32(x);
    }
#endif // __ARM_NEON
};

struct unary_op_floor
{
    float func(const float& x) const
    {
        return (float)floorf(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
#if __aarch64__
        return vrndmq_f32(x);
#else  // __aarch64__
        int32x4_t _xi = vcvtq_s32_f32(x);
        uint32x4_t _mask = vcgtq_f32(vcvtq_f32_s32(_xi), x);
        return vcvtq_f32_s32(vaddq_s32(_xi, vreinterpretq_s32_u32(_mask)));
#endif // __aarch64__
    }
#endif // __ARM_NEON
};

struct unary_op_ceil
{
    float func(const float& x) const
    {
        return (float)ceilf(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
#if __aarch64__
        return vrndpq_f32(x);
#else  // __aarch64__
        int32x4_t _xi = vcvtq_s32_f32(x);
        uint32x4_t _mask = vcgtq_f32(x, vcvtq_f32_s32(_xi));
        return vcvtq_f32_s32(vsubq_s32(_xi, vreinterpretq_s32_u32(_mask)));
#endif // __aarch64__
    }
#endif // __ARM_NEON
};

struct unary_op_square
{
    float func(const float& x) const
    {
        return x * x;
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        return vmulq_f32(x, x);
    }
#endif // __ARM_NEON
};

struct unary_op_sqrt
{
    float func(const float& x) const
    {
        return (float)sqrtf(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
#if __aarch64__
        return vsqrtq_f32(x);
#else
        float32x4_t _reciprocal = vrsqrteq_f32(x);
        _reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, _reciprocal), _reciprocal), _reciprocal);
        // _reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, _reciprocal), _reciprocal), _reciprocal);
        return vmulq_f32(x, _reciprocal);
#endif
    }
#endif // __ARM_NEON
};

struct unary_op_rsqrt
{
    float func(const float& x) const
    {
        return (float)(1.f / sqrtf(x));
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        float32x4_t _reciprocal = vrsqrteq_f32(x);
        _reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, _reciprocal), _reciprocal), _reciprocal);
        // _reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, _reciprocal), _reciprocal), _reciprocal);
        return _reciprocal;
    }
#endif // __ARM_NEON
};

struct unary_op_exp
{
    float func(const float& x) const
    {
        return (float)expf(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        return exp_ps(x);
    }
#endif // __ARM_NEON
};

struct unary_op_log
{
    float func(const float& x) const
    {
        return (float)logf(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        return log_ps(x);
    }
#endif // __ARM_NEON
};

struct unary_op_sin
{
    float func(const float& x) const
    {
        return (float)sinf(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        return sin_ps(x);
    }
#endif // __ARM_NEON
};

struct unary_op_cos
{
    float func(const float& x) const
    {
        return (float)cosf(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        return cos_ps(x);
    }
#endif // __ARM_NEON
};

struct unary_op_tan
{
    float func(const float& x) const
    {
        return (float)tanf(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        return tan_ps(x);
    }
#endif // __ARM_NEON
};

struct unary_op_asin
{
    float func(const float& x) const
    {
        return (float)asinf(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        return asin_ps(x);
    }
#endif // __ARM_NEON
};

struct unary_op_acos
{
    float func(const float& x) const
    {
        return (float)acosf(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        return acos_ps(x);
    }
#endif // __ARM_NEON
};

struct unary_op_atan
{
    float func(const float& x) const
    {
        return (float)atanf(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        // TODO neon optimize
        float tmp[4];
        vst1q_f32(tmp, x);
        tmp[0] = atanf(tmp[0]);
        tmp[1] = atanf(tmp[1]);
        tmp[2] = atanf(tmp[2]);
        tmp[3] = atanf(tmp[3]);
        return vld1q_f32(tmp);
    }
#endif // __ARM_NEON
};

struct unary_op_reciprocal
{
    float func(const float& x) const
    {
        return 1.f / x;
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        float32x4_t _reciprocal = vrecpeq_f32(x);
        _reciprocal = vmulq_f32(vrecpsq_f32(x, _reciprocal), _reciprocal);
        // _reciprocal = vmulq_f32(vrecpsq_f32(x, _reciprocal), _reciprocal);
        return _reciprocal;
    }
#endif // __ARM_NEON
};

struct unary_op_tanh
{
    float func(const float& x) const
    {
        return (float)tanhf(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        return tanh_ps(x);
    }
#endif // __ARM_NEON
};

struct unary_op_log10
{
    float func(const float& x) const
    {
        return (float)log10f(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        return vmulq_f32(log_ps(x), vdupq_n_f32(0.434294481903));
    }
#endif // __ARM_NEON
};

struct unary_op_round
{
    float func(const float& x) const
    {
        // round to nearest even
#if NCNN_GNU_INLINE_ASM && __ARM_NEON
        // return (x + 12582912.f) - 12582912.f;
        float y;
        const float magic = 12582912.f;
#if __aarch64__
        asm volatile(
            "fadd   %s0, %s1, %s2   \n"
            "fsub   %s0, %s0, %s2   \n"
            : "=w"(y)
            : "w"(x), "w"(magic)
            :);
#else
        asm volatile(
            "vadd.f32   %0, %1, %2  \n"
            "vsub.f32   %0, %0, %2  \n"
            : "=t"(y)
            : "t"(x), "t"(magic)
            :);
#endif
        return y;
#else
#ifdef FE_TONEAREST
        int old_rm = fegetround();
        fesetround(FE_TONEAREST);
#endif
        float y = nearbyintf(x);
#ifdef FE_TONEAREST
        fesetround(old_rm);
#endif
        return y;
#endif
    }
#if __ARM_NEON
#if __aarch64__
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        return vrndnq_f32(x);
    }
#else
    float32x4_t func_pack4(const float32x4_t& x) const
    {
#if NCNN_GNU_INLINE_ASM
        float32x4_t y;
        float32x4_t _magic = vdupq_n_f32(12582912.f); // 1.5 * 2^23
        asm volatile(
            "vadd.f32   %q0, %q1, %q2   \n"
            "vsub.f32   %q0, %q0, %q2   \n"
            : "=w"(y)
            : "w"(x), "w"(_magic)
            :);
        return y;
#else
        float tmp[4];
        vst1q_f32(tmp, x);
#ifdef FE_TONEAREST
        int old_rm = fegetround();
        fesetround(FE_TONEAREST);
#endif
        tmp[0] = nearbyintf(tmp[0]);
        tmp[1] = nearbyintf(tmp[1]);
        tmp[2] = nearbyintf(tmp[2]);
        tmp[3] = nearbyintf(tmp[3]);
#ifdef FE_TONEAREST
        fesetround(old_rm);
#endif
        float32x4_t y = vld1q_f32(tmp);
        return y;
#endif
    }
#endif
#endif // __ARM_NEON
};

struct unary_op_trunc
{
    float func(const float& x) const
    {
        return (float)truncf(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
#if __aarch64__
        return vrndq_f32(x);
#else
        return vcvtq_f32_s32(vcvtq_s32_f32(x));
#endif
    }
#endif // __ARM_NEON
};

} // namespace UnaryOp_arm_functor

int UnaryOp_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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

    using namespace UnaryOp_arm_functor;

    if (op_type == Operation_ABS)
        return unary_op_inplace<unary_op_abs>(bottom_top_blob, opt);

    if (op_type == Operation_NEG)
        return unary_op_inplace<unary_op_neg>(bottom_top_blob, opt);

    if (op_type == Operation_FLOOR)
        return unary_op_inplace<unary_op_floor>(bottom_top_blob, opt);

    if (op_type == Operation_CEIL)
        return unary_op_inplace<unary_op_ceil>(bottom_top_blob, opt);

    if (op_type == Operation_SQUARE)
        return unary_op_inplace<unary_op_square>(bottom_top_blob, opt);

    if (op_type == Operation_SQRT)
        return unary_op_inplace<unary_op_sqrt>(bottom_top_blob, opt);

    if (op_type == Operation_RSQRT)
        return unary_op_inplace<unary_op_rsqrt>(bottom_top_blob, opt);

    if (op_type == Operation_EXP)
        return unary_op_inplace<unary_op_exp>(bottom_top_blob, opt);

    if (op_type == Operation_LOG)
        return unary_op_inplace<unary_op_log>(bottom_top_blob, opt);

    if (op_type == Operation_SIN)
        return unary_op_inplace<unary_op_sin>(bottom_top_blob, opt);

    if (op_type == Operation_COS)
        return unary_op_inplace<unary_op_cos>(bottom_top_blob, opt);

    if (op_type == Operation_TAN)
        return unary_op_inplace<unary_op_tan>(bottom_top_blob, opt);

    if (op_type == Operation_ASIN)
        return unary_op_inplace<unary_op_asin>(bottom_top_blob, opt);

    if (op_type == Operation_ACOS)
        return unary_op_inplace<unary_op_acos>(bottom_top_blob, opt);

    if (op_type == Operation_ATAN)
        return unary_op_inplace<unary_op_atan>(bottom_top_blob, opt);

    if (op_type == Operation_RECIPROCAL)
        return unary_op_inplace<unary_op_reciprocal>(bottom_top_blob, opt);

    if (op_type == Operation_TANH)
        return unary_op_inplace<unary_op_tanh>(bottom_top_blob, opt);

    if (op_type == Operation_LOG10)
        return unary_op_inplace<unary_op_log10>(bottom_top_blob, opt);

    if (op_type == Operation_ROUND)
        return unary_op_inplace<unary_op_round>(bottom_top_blob, opt);

    if (op_type == Operation_TRUNC)
        return unary_op_inplace<unary_op_trunc>(bottom_top_blob, opt);

    return 0;
}

#if NCNN_BF16
template<typename Op>
static int unary_op_inplace_bf16s(Mat& a, const Option& opt)
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
        unsigned short* ptr = a.channel(q);

        int i = 0;
#if __ARM_NEON
#if __aarch64__
        for (; i + 15 < size; i += 16)
        {
            uint16x8_t _p01 = vld1q_u16(ptr);
            uint16x8_t _p23 = vld1q_u16(ptr + 8);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p01));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p01));
            float32x4_t _p2 = bfloat2float(vget_low_u16(_p23));
            float32x4_t _p3 = bfloat2float(vget_high_u16(_p23));
            _p0 = op.func_pack4(_p0);
            _p1 = op.func_pack4(_p1);
            _p2 = op.func_pack4(_p2);
            _p3 = op.func_pack4(_p3);
            _p01 = vcombine_u16(float2bfloat(_p0), float2bfloat(_p1));
            _p23 = vcombine_u16(float2bfloat(_p2), float2bfloat(_p3));
            vst1q_u16(ptr, _p01);
            vst1q_u16(ptr + 8, _p23);
            ptr += 16;
        }
#endif // __aarch64__
        for (; i + 7 < size; i += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = bfloat2float(vget_low_u16(_p));
            float32x4_t _p1 = bfloat2float(vget_high_u16(_p));
            _p0 = op.func_pack4(_p0);
            _p1 = op.func_pack4(_p1);
            _p = vcombine_u16(float2bfloat(_p0), float2bfloat(_p1));
            vst1q_u16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = bfloat2float(vld1_u16(ptr));
            _p = op.func_pack4(_p);
            vst1_u16(ptr, float2bfloat(_p));
            ptr += 4;
        }
#endif // __ARM_NEON
        for (; i < size; i++)
        {
            *ptr = float32_to_bfloat16(op.func(bfloat16_to_float32(*ptr)));
            ptr++;
        }
    }

    return 0;
}

int UnaryOp_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    using namespace UnaryOp_arm_functor;

    if (op_type == Operation_ABS)
        return unary_op_inplace_bf16s<unary_op_abs>(bottom_top_blob, opt);

    if (op_type == Operation_NEG)
        return unary_op_inplace_bf16s<unary_op_neg>(bottom_top_blob, opt);

    if (op_type == Operation_FLOOR)
        return unary_op_inplace_bf16s<unary_op_floor>(bottom_top_blob, opt);

    if (op_type == Operation_CEIL)
        return unary_op_inplace_bf16s<unary_op_ceil>(bottom_top_blob, opt);

    if (op_type == Operation_SQUARE)
        return unary_op_inplace_bf16s<unary_op_square>(bottom_top_blob, opt);

    if (op_type == Operation_SQRT)
        return unary_op_inplace_bf16s<unary_op_sqrt>(bottom_top_blob, opt);

    if (op_type == Operation_RSQRT)
        return unary_op_inplace_bf16s<unary_op_rsqrt>(bottom_top_blob, opt);

    if (op_type == Operation_EXP)
        return unary_op_inplace_bf16s<unary_op_exp>(bottom_top_blob, opt);

    if (op_type == Operation_LOG)
        return unary_op_inplace_bf16s<unary_op_log>(bottom_top_blob, opt);

    if (op_type == Operation_SIN)
        return unary_op_inplace_bf16s<unary_op_sin>(bottom_top_blob, opt);

    if (op_type == Operation_COS)
        return unary_op_inplace_bf16s<unary_op_cos>(bottom_top_blob, opt);

    if (op_type == Operation_TAN)
        return unary_op_inplace_bf16s<unary_op_tan>(bottom_top_blob, opt);

    if (op_type == Operation_ASIN)
        return unary_op_inplace_bf16s<unary_op_asin>(bottom_top_blob, opt);

    if (op_type == Operation_ACOS)
        return unary_op_inplace_bf16s<unary_op_acos>(bottom_top_blob, opt);

    if (op_type == Operation_ATAN)
        return unary_op_inplace_bf16s<unary_op_atan>(bottom_top_blob, opt);

    if (op_type == Operation_RECIPROCAL)
        return unary_op_inplace_bf16s<unary_op_reciprocal>(bottom_top_blob, opt);

    if (op_type == Operation_TANH)
        return unary_op_inplace_bf16s<unary_op_tanh>(bottom_top_blob, opt);

    if (op_type == Operation_LOG10)
        return unary_op_inplace_bf16s<unary_op_log10>(bottom_top_blob, opt);

    if (op_type == Operation_ROUND)
        return unary_op_inplace_bf16s<unary_op_round>(bottom_top_blob, opt);

    if (op_type == Operation_TRUNC)
        return unary_op_inplace_bf16s<unary_op_trunc>(bottom_top_blob, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
