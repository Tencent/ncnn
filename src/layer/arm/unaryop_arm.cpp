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
#include "neon_mathfun.h"

#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

UnaryOp_arm::UnaryOp_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

#if __ARM_NEON
template<typename Op>
static int unary_op_inplace(Mat& a, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            float32x4_t _p = vld1q_f32(ptr);
            _p = op(_p);
            vst1q_f32(ptr, _p);
            ptr += 4;
        }
    }

    return 0;
}

struct unary_op_abs_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
    {
        return vabsq_f32(x);
    }
};

struct unary_op_neg_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
    {
        return vnegq_f32(x);
    }
};

struct unary_op_floor_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
    {
#if __aarch64__
        return vcvtq_f32_s32(vcvtmq_s32_f32(x));
#else  // __aarch64__
        int32x4_t _xi = vcvtq_s32_f32(x);
        uint32x4_t _mask = vcgtq_f32(vcvtq_f32_s32(_xi), x);
        return vcvtq_f32_s32(vaddq_s32(_xi, vreinterpretq_s32_u32(_mask)));
#endif // __aarch64__
    }
};

struct unary_op_ceil_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
    {
#if __aarch64__
        return vcvtq_f32_s32(vcvtpq_s32_f32(x));
#else  // __aarch64__
        int32x4_t _xi = vcvtq_s32_f32(x);
        uint32x4_t _mask = vcgtq_f32(x, vcvtq_f32_s32(_xi));
        return vcvtq_f32_s32(vsubq_s32(_xi, vreinterpretq_s32_u32(_mask)));
#endif // __aarch64__
    }
};

struct unary_op_square_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
    {
        return vmulq_f32(x, x);
    }
};

struct unary_op_sqrt_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
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
};

struct unary_op_rsqrt_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
    {
        float32x4_t _reciprocal = vrsqrteq_f32(x);
        _reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, _reciprocal), _reciprocal), _reciprocal);
        // _reciprocal = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, _reciprocal), _reciprocal), _reciprocal);
        return _reciprocal;
    }
};

struct unary_op_exp_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
    {
        return exp_ps(x);
    }
};

struct unary_op_log_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
    {
        return log_ps(x);
    }
};

struct unary_op_sin_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
    {
        return sin_ps(x);
    }
};

struct unary_op_cos_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
    {
        return cos_ps(x);
    }
};

struct unary_op_tan_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
    {
        // TODO neon optimize
        float tmp[4];
        vst1q_f32(tmp, x);
        tmp[0] = tan(tmp[0]);
        tmp[1] = tan(tmp[1]);
        tmp[2] = tan(tmp[2]);
        tmp[3] = tan(tmp[3]);
        return vld1q_f32(tmp);
    }
};

struct unary_op_asin_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
    {
        // TODO neon optimize
        float tmp[4];
        vst1q_f32(tmp, x);
        tmp[0] = asin(tmp[0]);
        tmp[1] = asin(tmp[1]);
        tmp[2] = asin(tmp[2]);
        tmp[3] = asin(tmp[3]);
        return vld1q_f32(tmp);
    }
};

struct unary_op_acos_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
    {
        // TODO neon optimize
        float tmp[4];
        vst1q_f32(tmp, x);
        tmp[0] = acos(tmp[0]);
        tmp[1] = acos(tmp[1]);
        tmp[2] = acos(tmp[2]);
        tmp[3] = acos(tmp[3]);
        return vld1q_f32(tmp);
    }
};

struct unary_op_atan_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
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
};

struct unary_op_reciprocal_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
    {
        float32x4_t _reciprocal = vrecpeq_f32(x);
        _reciprocal = vmulq_f32(vrecpsq_f32(x, _reciprocal), _reciprocal);
        // _reciprocal = vmulq_f32(vrecpsq_f32(x, _reciprocal), _reciprocal);
        return _reciprocal;
    }
};

struct unary_op_tanh_pack4
{
    float32x4_t operator()(const float32x4_t& x) const
    {
        return tanh_ps(x);
    }
};
#endif // __ARM_NEON

int UnaryOp_arm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        if (op_type == Operation_ABS)
            return unary_op_inplace<unary_op_abs_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_NEG)
            return unary_op_inplace<unary_op_neg_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_FLOOR)
            return unary_op_inplace<unary_op_floor_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_CEIL)
            return unary_op_inplace<unary_op_ceil_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_SQUARE)
            return unary_op_inplace<unary_op_square_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_SQRT)
            return unary_op_inplace<unary_op_sqrt_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_RSQRT)
            return unary_op_inplace<unary_op_rsqrt_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_EXP)
            return unary_op_inplace<unary_op_exp_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_LOG)
            return unary_op_inplace<unary_op_log_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_SIN)
            return unary_op_inplace<unary_op_sin_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_COS)
            return unary_op_inplace<unary_op_cos_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_TAN)
            return unary_op_inplace<unary_op_tan_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_ASIN)
            return unary_op_inplace<unary_op_asin_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_ACOS)
            return unary_op_inplace<unary_op_acos_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_ATAN)
            return unary_op_inplace<unary_op_atan_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_RECIPROCAL)
            return unary_op_inplace<unary_op_reciprocal_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_TANH)
            return unary_op_inplace<unary_op_tanh_pack4>(bottom_top_blob, opt);
    }
#endif // __ARM_NEON

    return UnaryOp::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn
