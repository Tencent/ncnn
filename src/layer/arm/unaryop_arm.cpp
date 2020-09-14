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
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "neon_mathfun_fp16s.h"
#endif
#endif // __ARM_NEON

namespace ncnn {

UnaryOp_arm::UnaryOp_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

    support_bf16_storage = true;
}

#if __ARM_NEON
template<typename Op>
static int unary_op_inplace_pack4(Mat& a, const Option& opt)
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
    int elembits = bottom_top_blob.elembits();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blob, opt);
#endif

    if (opt.use_bf16_storage && elembits == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);

    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        if (op_type == Operation_ABS)
            return unary_op_inplace_pack4<unary_op_abs_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_NEG)
            return unary_op_inplace_pack4<unary_op_neg_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_FLOOR)
            return unary_op_inplace_pack4<unary_op_floor_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_CEIL)
            return unary_op_inplace_pack4<unary_op_ceil_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_SQUARE)
            return unary_op_inplace_pack4<unary_op_square_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_SQRT)
            return unary_op_inplace_pack4<unary_op_sqrt_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_RSQRT)
            return unary_op_inplace_pack4<unary_op_rsqrt_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_EXP)
            return unary_op_inplace_pack4<unary_op_exp_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_LOG)
            return unary_op_inplace_pack4<unary_op_log_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_SIN)
            return unary_op_inplace_pack4<unary_op_sin_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_COS)
            return unary_op_inplace_pack4<unary_op_cos_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_TAN)
            return unary_op_inplace_pack4<unary_op_tan_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_ASIN)
            return unary_op_inplace_pack4<unary_op_asin_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_ACOS)
            return unary_op_inplace_pack4<unary_op_acos_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_ATAN)
            return unary_op_inplace_pack4<unary_op_atan_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_RECIPROCAL)
            return unary_op_inplace_pack4<unary_op_reciprocal_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_TANH)
            return unary_op_inplace_pack4<unary_op_tanh_pack4>(bottom_top_blob, opt);
    }
#endif // __ARM_NEON

    return UnaryOp::forward_inplace(bottom_top_blob, opt);
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template<typename Op>
static int unary_op_inplace_pack8_fp16s(Mat& a, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            float16x8_t _p = vld1q_f16(ptr);
            _p = op(_p);
            vst1q_f16(ptr, _p);
            ptr += 8;
        }
    }

    return 0;
}

struct unary_op_abs_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
    {
        return vabsq_f16(x);
    }
};

struct unary_op_neg_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
    {
        return vnegq_f16(x);
    }
};

struct unary_op_floor_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
    {
        return vcvtq_f16_s16(vcvtmq_s16_f16(x));
    }
};

struct unary_op_ceil_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
    {
        return vcvtq_f16_s16(vcvtpq_s16_f16(x));
    }
};

struct unary_op_square_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
    {
        return vmulq_f16(x, x);
    }
};

struct unary_op_sqrt_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
    {
        return vsqrtq_f16(x);
    }
};

struct unary_op_rsqrt_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
    {
        float16x8_t _reciprocal = vrsqrteq_f16(x);
        _reciprocal = vmulq_f16(vrsqrtsq_f16(vmulq_f16(x, _reciprocal), _reciprocal), _reciprocal);
        // _reciprocal = vmulq_f16(vrsqrtsq_f16(vmulq_f16(x, _reciprocal), _reciprocal), _reciprocal);
        return _reciprocal;
    }
};

struct unary_op_exp_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
    {
        return exp_ps(x);
    }
};

struct unary_op_log_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
    {
        return log_ps(x);
    }
};

struct unary_op_sin_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
    {
        return sin_ps(x);
    }
};

struct unary_op_cos_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
    {
        return cos_ps(x);
    }
};

struct unary_op_tan_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
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

struct unary_op_asin_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
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

struct unary_op_acos_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
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

struct unary_op_atan_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
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

struct unary_op_reciprocal_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
    {
        float16x8_t _reciprocal = vrecpeq_f16(x);
        _reciprocal = vmulq_f16(vrecpsq_f16(x, _reciprocal), _reciprocal);
        // _reciprocal = vmulq_f16(vrecpsq_f16(x, _reciprocal), _reciprocal);
        return _reciprocal;
    }
};

struct unary_op_tanh_pack8_fp16s
{
    float16x8_t operator()(const float16x8_t& x) const
    {
        return tanh_ps(x);
    }
};

template<typename Op>
static int unary_op_inplace_pack4_fp16s(Mat& a, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            float16x4_t _p = vld1_f16(ptr);
            _p = op(_p);
            vst1_f16(ptr, _p);
            ptr += 4;
        }
    }

    return 0;
}

struct unary_op_abs_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
    {
        return vabs_f16(x);
    }
};

struct unary_op_neg_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
    {
        return vneg_f16(x);
    }
};

struct unary_op_floor_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
    {
        return vcvt_f16_s16(vcvtm_s16_f16(x));
    }
};

struct unary_op_ceil_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
    {
        return vcvt_f16_s16(vcvtp_s16_f16(x));
    }
};

struct unary_op_square_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
    {
        return vmul_f16(x, x);
    }
};

struct unary_op_sqrt_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
    {
        return vsqrt_f16(x);
    }
};

struct unary_op_rsqrt_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
    {
        float16x4_t _reciprocal = vrsqrte_f16(x);
        _reciprocal = vmul_f16(vrsqrts_f16(vmul_f16(x, _reciprocal), _reciprocal), _reciprocal);
        // _reciprocal = vmul_f16(vrsqrts_f16(vmul_f16(x, _reciprocal), _reciprocal), _reciprocal);
        return _reciprocal;
    }
};

struct unary_op_exp_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
    {
        return exp_ps(x);
    }
};

struct unary_op_log_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
    {
        return log_ps(x);
    }
};

struct unary_op_sin_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
    {
        return sin_ps(x);
    }
};

struct unary_op_cos_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
    {
        return cos_ps(x);
    }
};

struct unary_op_tan_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
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
};

struct unary_op_asin_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
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
};

struct unary_op_acos_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
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
};

struct unary_op_atan_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
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
};

struct unary_op_reciprocal_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
    {
        float16x4_t _reciprocal = vrecpe_f16(x);
        _reciprocal = vmul_f16(vrecps_f16(x, _reciprocal), _reciprocal);
        // _reciprocal = vmul_f16(vrecps_f16(x, _reciprocal), _reciprocal);
        return _reciprocal;
    }
};

struct unary_op_tanh_pack4_fp16s
{
    float16x4_t operator()(const float16x4_t& x) const
    {
        return tanh_ps(x);
    }
};

template<typename Op>
static int unary_op_inplace_fp16s(Mat& a, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            ptr[i] = op(ptr[i]);
        }
    }

    return 0;
}

struct unary_op_abs_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)abs(x);
    }
};

struct unary_op_neg_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return -x;
    }
};

struct unary_op_floor_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)floor(x);
    }
};

struct unary_op_ceil_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)ceil(x);
    }
};

struct unary_op_square_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return x * x;
    }
};

struct unary_op_sqrt_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)sqrt(x);
    }
};

struct unary_op_rsqrt_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)1.f / sqrt(x);
    }
};

struct unary_op_exp_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)exp(x);
    }
};

struct unary_op_log_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)log(x);
    }
};

struct unary_op_sin_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)sin(x);
    }
};

struct unary_op_cos_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)cos(x);
    }
};

struct unary_op_tan_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)tan(x);
    }
};

struct unary_op_asin_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)asin(x);
    }
};

struct unary_op_acos_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)acos(x);
    }
};

struct unary_op_atan_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)atan(x);
    }
};

struct unary_op_reciprocal_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)1.f / x;
    }
};

struct unary_op_tanh_fp16s
{
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)tanh(x);
    }
};

int UnaryOp_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    int elempack = bottom_top_blob.elempack;

    if (elempack == 8)
    {
        if (op_type == Operation_ABS)
            return unary_op_inplace_pack8_fp16s<unary_op_abs_pack8_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_NEG)
            return unary_op_inplace_pack8_fp16s<unary_op_neg_pack8_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_FLOOR)
            return unary_op_inplace_pack8_fp16s<unary_op_floor_pack8_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_CEIL)
            return unary_op_inplace_pack8_fp16s<unary_op_ceil_pack8_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_SQUARE)
            return unary_op_inplace_pack8_fp16s<unary_op_square_pack8_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_SQRT)
            return unary_op_inplace_pack8_fp16s<unary_op_sqrt_pack8_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_RSQRT)
            return unary_op_inplace_pack8_fp16s<unary_op_rsqrt_pack8_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_EXP)
            return unary_op_inplace_pack8_fp16s<unary_op_exp_pack8_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_LOG)
            return unary_op_inplace_pack8_fp16s<unary_op_log_pack8_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_SIN)
            return unary_op_inplace_pack8_fp16s<unary_op_sin_pack8_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_COS)
            return unary_op_inplace_pack8_fp16s<unary_op_cos_pack8_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_TAN)
            return unary_op_inplace_pack8_fp16s<unary_op_tan_pack8_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_ASIN)
            return unary_op_inplace_pack8_fp16s<unary_op_asin_pack8_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_ACOS)
            return unary_op_inplace_pack8_fp16s<unary_op_acos_pack8_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_ATAN)
            return unary_op_inplace_pack8_fp16s<unary_op_atan_pack8_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_RECIPROCAL)
            return unary_op_inplace_pack8_fp16s<unary_op_reciprocal_pack8_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_TANH)
            return unary_op_inplace_pack8_fp16s<unary_op_tanh_pack8_fp16s>(bottom_top_blob, opt);
    }

    if (elempack == 4)
    {
        if (op_type == Operation_ABS)
            return unary_op_inplace_pack4_fp16s<unary_op_abs_pack4_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_NEG)
            return unary_op_inplace_pack4_fp16s<unary_op_neg_pack4_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_FLOOR)
            return unary_op_inplace_pack4_fp16s<unary_op_floor_pack4_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_CEIL)
            return unary_op_inplace_pack4_fp16s<unary_op_ceil_pack4_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_SQUARE)
            return unary_op_inplace_pack4_fp16s<unary_op_square_pack4_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_SQRT)
            return unary_op_inplace_pack4_fp16s<unary_op_sqrt_pack4_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_RSQRT)
            return unary_op_inplace_pack4_fp16s<unary_op_rsqrt_pack4_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_EXP)
            return unary_op_inplace_pack4_fp16s<unary_op_exp_pack4_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_LOG)
            return unary_op_inplace_pack4_fp16s<unary_op_log_pack4_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_SIN)
            return unary_op_inplace_pack4_fp16s<unary_op_sin_pack4_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_COS)
            return unary_op_inplace_pack4_fp16s<unary_op_cos_pack4_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_TAN)
            return unary_op_inplace_pack4_fp16s<unary_op_tan_pack4_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_ASIN)
            return unary_op_inplace_pack4_fp16s<unary_op_asin_pack4_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_ACOS)
            return unary_op_inplace_pack4_fp16s<unary_op_acos_pack4_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_ATAN)
            return unary_op_inplace_pack4_fp16s<unary_op_atan_pack4_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_RECIPROCAL)
            return unary_op_inplace_pack4_fp16s<unary_op_reciprocal_pack4_fp16s>(bottom_top_blob, opt);

        if (op_type == Operation_TANH)
            return unary_op_inplace_pack4_fp16s<unary_op_tanh_pack4_fp16s>(bottom_top_blob, opt);
    }

    if (elempack == 1)
    {
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
    }

    return 0;
}
#endif

#if __ARM_NEON
template<typename Op>
static int unary_op_inplace_pack4_bf16s(Mat& a, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            float32x4_t _p = vcvt_f32_bf16(vld1_u16(ptr));
            _p = op(_p);
            vst1_u16(ptr, vcvt_bf16_f32(_p));
            ptr += 4;
        }
    }

    return 0;
}
#endif // __ARM_NEON

template<typename Op>
static int unary_op_inplace_bf16s(Mat& a, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            ptr[i] = float32_to_bfloat16(op(bfloat16_to_float32(ptr[i])));
        }
    }

    return 0;
}

struct unary_op_abs
{
    float operator()(const float& x) const
    {
        return abs(x);
    }
};

struct unary_op_neg
{
    float operator()(const float& x) const
    {
        return -x;
    }
};

struct unary_op_floor
{
    float operator()(const float& x) const
    {
        return floor(x);
    }
};

struct unary_op_ceil
{
    float operator()(const float& x) const
    {
        return ceil(x);
    }
};

struct unary_op_square
{
    float operator()(const float& x) const
    {
        return x * x;
    }
};

struct unary_op_sqrt
{
    float operator()(const float& x) const
    {
        return sqrt(x);
    }
};

struct unary_op_rsqrt
{
    float operator()(const float& x) const
    {
        return 1.f / sqrt(x);
    }
};

struct unary_op_exp
{
    float operator()(const float& x) const
    {
        return exp(x);
    }
};

struct unary_op_log
{
    float operator()(const float& x) const
    {
        return log(x);
    }
};

struct unary_op_sin
{
    float operator()(const float& x) const
    {
        return sin(x);
    }
};

struct unary_op_cos
{
    float operator()(const float& x) const
    {
        return cos(x);
    }
};

struct unary_op_tan
{
    float operator()(const float& x) const
    {
        return tan(x);
    }
};

struct unary_op_asin
{
    float operator()(const float& x) const
    {
        return asin(x);
    }
};

struct unary_op_acos
{
    float operator()(const float& x) const
    {
        return acos(x);
    }
};

struct unary_op_atan
{
    float operator()(const float& x) const
    {
        return atan(x);
    }
};

struct unary_op_reciprocal
{
    float operator()(const float& x) const
    {
        return 1.f / x;
    }
};

struct unary_op_tanh
{
    float operator()(const float& x) const
    {
        return tanh(x);
    }
};

int UnaryOp_arm::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int elempack = bottom_top_blob.elempack;

#if __ARM_NEON
    if (elempack == 4)
    {
        if (op_type == Operation_ABS)
            return unary_op_inplace_pack4_bf16s<unary_op_abs_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_NEG)
            return unary_op_inplace_pack4_bf16s<unary_op_neg_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_FLOOR)
            return unary_op_inplace_pack4_bf16s<unary_op_floor_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_CEIL)
            return unary_op_inplace_pack4_bf16s<unary_op_ceil_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_SQUARE)
            return unary_op_inplace_pack4_bf16s<unary_op_square_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_SQRT)
            return unary_op_inplace_pack4_bf16s<unary_op_sqrt_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_RSQRT)
            return unary_op_inplace_pack4_bf16s<unary_op_rsqrt_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_EXP)
            return unary_op_inplace_pack4_bf16s<unary_op_exp_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_LOG)
            return unary_op_inplace_pack4_bf16s<unary_op_log_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_SIN)
            return unary_op_inplace_pack4_bf16s<unary_op_sin_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_COS)
            return unary_op_inplace_pack4_bf16s<unary_op_cos_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_TAN)
            return unary_op_inplace_pack4_bf16s<unary_op_tan_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_ASIN)
            return unary_op_inplace_pack4_bf16s<unary_op_asin_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_ACOS)
            return unary_op_inplace_pack4_bf16s<unary_op_acos_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_ATAN)
            return unary_op_inplace_pack4_bf16s<unary_op_atan_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_RECIPROCAL)
            return unary_op_inplace_pack4_bf16s<unary_op_reciprocal_pack4>(bottom_top_blob, opt);

        if (op_type == Operation_TANH)
            return unary_op_inplace_pack4_bf16s<unary_op_tanh_pack4>(bottom_top_blob, opt);
    }
#endif // __ARM_NEON

    if (elempack == 1)
    {
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
    }

    return 0;
}

} // namespace ncnn
