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
        return (float)fabs(x);
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
        return (float)floor(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
#if __aarch64__
        return vcvtq_f32_s32(vcvtmq_s32_f32(x));
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
        return (float)ceil(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
#if __aarch64__
        return vcvtq_f32_s32(vcvtpq_s32_f32(x));
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
        return (float)sqrt(x);
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
        return (float)(1.f / sqrt(x));
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
        return (float)exp(x);
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
        return (float)log(x);
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
        return (float)sin(x);
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
        return (float)cos(x);
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
        return (float)tan(x);
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
        return (float)asin(x);
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
        return (float)acos(x);
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
        return (float)atan(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        // TODO neon optimize
        float tmp[4];
        vst1q_f32(tmp, x);
        tmp[0] = atan(tmp[0]);
        tmp[1] = atan(tmp[1]);
        tmp[2] = atan(tmp[2]);
        tmp[3] = atan(tmp[3]);
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
        return (float)tanh(x);
    }
#if __ARM_NEON
    float32x4_t func_pack4(const float32x4_t& x) const
    {
        return tanh_ps(x);
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
            float32x4_t _p0 = float2bfloat(vget_low_u16(_p01));
            float32x4_t _p1 = float2bfloat(vget_high_u16(_p01));
            float32x4_t _p2 = float2bfloat(vget_low_u16(_p23));
            float32x4_t _p3 = float2bfloat(vget_high_u16(_p23));
            _p0 = op.func_pack4(_p0);
            _p1 = op.func_pack4(_p1);
            _p2 = op.func_pack4(_p2);
            _p3 = op.func_pack4(_p3);
            _p01 = vcombine_u16(bfloat2float(_p0), bfloat2float(_p1));
            _p23 = vcombine_u16(bfloat2float(_p2), bfloat2float(_p3));
            vst1q_u16(ptr, _p01);
            vst1q_u16(ptr + 8, _p23);
            ptr += 16;
        }
#endif // __aarch64__
        for (; i + 7 < size; i += 8)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            float32x4_t _p0 = float2bfloat(vget_low_u16(_p));
            float32x4_t _p1 = float2bfloat(vget_high_u16(_p));
            _p0 = op.func_pack4(_p0);
            _p1 = op.func_pack4(_p1);
            _p = vcombine_u16(bfloat2float(_p0), bfloat2float(_p1));
            vst1q_u16(ptr, _p);
            ptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p = float2bfloat(vld1_u16(ptr));
            _p = op.func_pack4(_p);
            vst1_u16(ptr, bfloat2float(_p));
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

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
