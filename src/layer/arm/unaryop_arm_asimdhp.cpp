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

#include <math.h>

#if __ARM_NEON
#include <arm_neon.h>
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
        return (__fp16)fabs(x);
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
        return (__fp16)floor(x);
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
        return (__fp16)ceil(x);
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
        return (__fp16)sqrt(x);
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
        return (__fp16)1.f / sqrt(x);
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
        return (__fp16)exp(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return exp_ps(x);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return exp_ps(x);
    }
};

struct unary_op_log_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)log(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return log_ps(x);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return log_ps(x);
    }
};

struct unary_op_sin_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)sin(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return sin_ps(x);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return sin_ps(x);
    }
};

struct unary_op_cos_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)cos(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return cos_ps(x);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return cos_ps(x);
    }
};

struct unary_op_tan_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)tan(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        // TODO neon optimize
        __fp16 tmp[4];
        vst1_f16(tmp, x);
        tmp[0] = tan(tmp[0]);
        tmp[1] = tan(tmp[1]);
        tmp[2] = tan(tmp[2]);
        tmp[3] = tan(tmp[3]);
        return vld1_f16(tmp);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        // TODO neon optimize
        __fp16 tmp[8];
        vst1q_f16(tmp, x);
        tmp[0] = tan(tmp[0]);
        tmp[1] = tan(tmp[1]);
        tmp[2] = tan(tmp[2]);
        tmp[3] = tan(tmp[3]);
        tmp[4] = tan(tmp[4]);
        tmp[5] = tan(tmp[5]);
        tmp[6] = tan(tmp[6]);
        tmp[7] = tan(tmp[7]);
        return vld1q_f16(tmp);
    }
};

struct unary_op_asin_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)asin(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        // TODO neon optimize
        __fp16 tmp[4];
        vst1_f16(tmp, x);
        tmp[0] = asin(tmp[0]);
        tmp[1] = asin(tmp[1]);
        tmp[2] = asin(tmp[2]);
        tmp[3] = asin(tmp[3]);
        return vld1_f16(tmp);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        // TODO neon optimize
        __fp16 tmp[8];
        vst1q_f16(tmp, x);
        tmp[0] = asin(tmp[0]);
        tmp[1] = asin(tmp[1]);
        tmp[2] = asin(tmp[2]);
        tmp[3] = asin(tmp[3]);
        tmp[4] = asin(tmp[4]);
        tmp[5] = asin(tmp[5]);
        tmp[6] = asin(tmp[6]);
        tmp[7] = asin(tmp[7]);
        return vld1q_f16(tmp);
    }
};

struct unary_op_acos_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)acos(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        // TODO neon optimize
        __fp16 tmp[4];
        vst1_f16(tmp, x);
        tmp[0] = acos(tmp[0]);
        tmp[1] = acos(tmp[1]);
        tmp[2] = acos(tmp[2]);
        tmp[3] = acos(tmp[3]);
        return vld1_f16(tmp);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        // TODO neon optimize
        __fp16 tmp[8];
        vst1q_f16(tmp, x);
        tmp[0] = acos(tmp[0]);
        tmp[1] = acos(tmp[1]);
        tmp[2] = acos(tmp[2]);
        tmp[3] = acos(tmp[3]);
        tmp[4] = acos(tmp[4]);
        tmp[5] = acos(tmp[5]);
        tmp[6] = acos(tmp[6]);
        tmp[7] = acos(tmp[7]);
        return vld1q_f16(tmp);
    }
};

struct unary_op_atan_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)atan(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        // TODO neon optimize
        __fp16 tmp[4];
        vst1_f16(tmp, x);
        tmp[0] = atan(tmp[0]);
        tmp[1] = atan(tmp[1]);
        tmp[2] = atan(tmp[2]);
        tmp[3] = atan(tmp[3]);
        return vld1_f16(tmp);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        // TODO neon optimize
        __fp16 tmp[8];
        vst1q_f16(tmp, x);
        tmp[0] = atan(tmp[0]);
        tmp[1] = atan(tmp[1]);
        tmp[2] = atan(tmp[2]);
        tmp[3] = atan(tmp[3]);
        tmp[4] = atan(tmp[4]);
        tmp[5] = atan(tmp[5]);
        tmp[6] = atan(tmp[6]);
        tmp[7] = atan(tmp[7]);
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
        return (__fp16)tanh(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return tanh_ps(x);
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return tanh_ps(x);
    }
};

struct unary_op_log10_fp16s
{
    __fp16 func(const __fp16& x) const
    {
        return (__fp16)log10(x);
    }
    float16x4_t func_pack4(const float16x4_t& x) const
    {
        return vmul_f16(log_ps(x), vdup_n_f16(0.434294481903));
    }
    float16x8_t func_pack8(const float16x8_t& x) const
    {
        return vmulq_f16(log_ps(x), vdupq_n_f16(0.434294481903));
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

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
