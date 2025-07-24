// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "unaryop_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#if __riscv_zvfh
#include "rvv_mathfun_fp16s.h"
#endif
#endif // __riscv_vector

#include <float.h>

namespace ncnn {

#if NCNN_ZFH
template<typename Op>
static int unary_op_inplace_fp16s(Mat& a, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int size = w * h * d;
    int elempack = a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = a.channel(q);

#if __riscv_zvfh
        int n = size * elempack;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m8(n);

            vfloat16m8_t _p = __riscv_vle16_v_f16m8(ptr, vl);
            _p = op(_p, vl);
            __riscv_vse16_v_f16m8(ptr, _p, vl);

            ptr += vl;
            n -= vl;
        }
#else  // __riscv_zvfh
        for (int i = 0; i < size; i++)
        {
            ptr[i] = op(ptr[i]);
        }
#endif // __riscv_zvfh
    }

    return 0;
}

namespace UnaryOp_riscv_functor {

struct unary_op_abs_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        return __riscv_vfsgnj_vf_f16m8(x, (__fp16)1.f, vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)fabsf((float)x);
    }
#endif // __riscv_zvfh
};

struct unary_op_neg_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        return __riscv_vfneg_v_f16m8(x, vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return -x;
    }
#endif // __riscv_zvfh
};

struct unary_op_floor_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        return __riscv_vfcvt_f_x_v_f16m8(__riscv_vfcvt_x_f_v_i16m8_rm(x, __RISCV_FRM_RDN, vl), vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)floorf((float)x);
    }
#endif // __riscv_zvfh
};

struct unary_op_ceil_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        return __riscv_vfcvt_f_x_v_f16m8(__riscv_vfcvt_x_f_v_i16m8_rm(x, __RISCV_FRM_RUP, vl), vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)ceilf((float)x);
    }
#endif // __riscv_zvfh
};

struct unary_op_square_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        return __riscv_vfmul_vv_f16m8(x, x, vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return x * x;
    }
#endif // __riscv_zvfh
};

struct unary_op_sqrt_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        return __riscv_vfsqrt_v_f16m8(x, vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)sqrtf((float)x);
    }
#endif // __riscv_zvfh
};

struct unary_op_rsqrt_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
#if __riscv_xtheadvector
        vfloat16m8_t _reciprocal = __riscv_vfrdiv_vf_f16m8(__riscv_vfsqrt_v_f16m8(x, vl), (__fp16)1.f, vl);
#else
        vfloat16m8_t _reciprocal = __riscv_vfrsqrt7_v_f16m8(x, vl);
        _reciprocal = __riscv_vfmul_vv_f16m8(__riscv_vfrsub_vf_f16m8(__riscv_vfmul_vv_f16m8(__riscv_vfmul_vf_f16m8(x, (__fp16)0.5f, vl), __riscv_vfmul_vv_f16m8(_reciprocal, _reciprocal, vl), vl), (__fp16)1.5f, vl), _reciprocal, vl);
        // _reciprocal = __riscv_vfmul_vv_f16m8(__riscv_vfrsub_vf_f16m8(__riscv_vfmul_vv_f16m8(__riscv_vfmul_vf_f16m8(x, (__fp16)0.5f, vl), __riscv_vfmul_vv_f16m8(_reciprocal, _reciprocal, vl), vl), (__fp16)1.5f, vl), _reciprocal, vl);
#endif
        return _reciprocal;
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)(1.f / sqrtf((float)x));
    }
#endif // __riscv_zvfh
};

struct unary_op_exp_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        return exp_ps(x, vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)expf((float)x);
    }
#endif // __riscv_zvfh
};

struct unary_op_log_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        return log_ps(x, vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)logf((float)x);
    }
#endif // __riscv_zvfh
};

struct unary_op_sin_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        return sin_ps(x, vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)sinf((float)x);
    }
#endif // __riscv_zvfh
};

struct unary_op_cos_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        return cos_ps(x, vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)cosf((float)x);
    }
#endif // __riscv_zvfh
};

struct unary_op_tan_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        // TODO rvv optimize
        std::vector<__fp16> tmp(vl);
        __riscv_vse16_v_f16m8(tmp.data(), x, vl);
        for (size_t i = 0; i < vl; i++)
        {
            tmp[i] = (__fp16)tanf((float)tmp[i]);
        }
        return __riscv_vle16_v_f16m8(tmp.data(), vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)tanf((float)x);
    }
#endif // __riscv_zvfh
};

struct unary_op_asin_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        // TODO rvv optimize
        std::vector<__fp16> tmp(vl);
        __riscv_vse16_v_f16m8(tmp.data(), x, vl);
        for (size_t i = 0; i < vl; i++)
        {
            tmp[i] = (__fp16)asinf((float)tmp[i]);
        }
        return __riscv_vle16_v_f16m8(tmp.data(), vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)asinf((float)x);
    }
#endif // __riscv_zvfh
};

struct unary_op_acos_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        // TODO rvv optimize
        std::vector<__fp16> tmp(vl);
        __riscv_vse16_v_f16m8(tmp.data(), x, vl);
        for (size_t i = 0; i < vl; i++)
        {
            tmp[i] = (__fp16)acosf((float)tmp[i]);
        }
        return __riscv_vle16_v_f16m8(tmp.data(), vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)acosf((float)x);
    }
#endif // __riscv_zvfh
};

struct unary_op_atan_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        // TODO rvv optimize
        std::vector<__fp16> tmp(vl);
        __riscv_vse16_v_f16m8(tmp.data(), x, vl);
        for (size_t i = 0; i < vl; i++)
        {
            tmp[i] = (__fp16)atanf((float)tmp[i]);
        }
        return __riscv_vle16_v_f16m8(tmp.data(), vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)atanf((float)x);
    }
#endif // __riscv_zvfh
};

struct unary_op_reciprocal_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
#if __riscv_xtheadvector
        vfloat16m8_t _reciprocal = __riscv_vfrdiv_vf_f16m8(x, (__fp16)1.f, vl);
#else
        vfloat16m8_t _reciprocal = __riscv_vfrec7_v_f16m8(x, vl);
        _reciprocal = __riscv_vfmul_vv_f16m8(__riscv_vfrsub_vf_f16m8(__riscv_vfmul_vv_f16m8(x, _reciprocal, vl), (__fp16)2.f, vl), _reciprocal, vl);
        // _reciprocal = __riscv_vfmul_vv_f16m8(__riscv_vfrsub_vf_f16m8(__riscv_vfmul_vv_f16m8(x, _reciprocal, vl), (__fp16)2.f, vl), _reciprocal, vl);
#endif
        return _reciprocal;
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)1.f / x;
    }
#endif // __riscv_zvfh
};

struct unary_op_tanh_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        return tanh_ps(x, vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)tanhf((float)x);
    }
#endif // __riscv_zvfh
};

struct unary_op_log10_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        return __riscv_vfmul_vf_f16m8(log_ps(x, vl), (__fp16)0.434294481903, vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)log10f((float)x);
    }
#endif // __riscv_zvfh
};

struct unary_op_round_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
        return __riscv_vfcvt_f_x_v_f16m8(__riscv_vfcvt_x_f_v_i16m8(x, vl), vl);
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)nearbyintf((float)x);
    }
#endif // __riscv_zvfh
};

struct unary_op_trunc_fp16s
{
#if __riscv_zvfh
    vfloat16m8_t operator()(const vfloat16m8_t& x, const size_t& vl) const
    {
#if __riscv_xtheadvector
        // simulate trunc with floor positives and ceil negative
        // xi = round(x)
        // floorx = xi - (xi > x)
        // ceilx = xi + (xi < x)
        // truncx = x >= 0 ? floorx : ceilx
        vint16m8_t _xi = __riscv_vfcvt_x_f_v_i16m8(x, vl);
        vfloat16m8_t _xf = __riscv_vfcvt_f_x_v_f16m8(_xi, vl);
        vbool2_t _floormask = __riscv_vmfgt_vv_f16m8_b2(_xf, x, vl);
        vint16m8_t _floorx = __riscv_vsub_vx_i16m8_mu(_floormask, _xi, _xi, 1, vl);
        vbool2_t _ceilmask = __riscv_vmflt_vv_f16m8_b2(_xf, x, vl);
        vint16m8_t _ceilx = __riscv_vadd_vx_i16m8_mu(_ceilmask, _xi, _xi, 1, vl);
        vbool2_t _negative = __riscv_vmflt_vf_f16m8_b2(x, (__fp16)0.f, vl);
        return __riscv_vfcvt_f_x_v_f16m8(__riscv_vmerge_vvm_i16m8(_floorx, _ceilx, _negative, vl), vl);
#else
        return __riscv_vfcvt_f_x_v_f16m8(__riscv_vfcvt_rtz_x_f_v_i16m8(x, vl), vl);
#endif
    }
#else  // __riscv_zvfh
    __fp16 operator()(const __fp16& x) const
    {
        return (__fp16)truncf((float)x);
    }
#endif // __riscv_zvfh
};

} // namespace UnaryOp_riscv_functor

int UnaryOp_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    using namespace UnaryOp_riscv_functor;

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
    {
        // round to nearest even
#ifdef FE_TONEAREST
        int old_rm = fegetround();
        fesetround(FE_TONEAREST);
#endif
        int ret = unary_op_inplace_fp16s<unary_op_round_fp16s>(bottom_top_blob, opt);
#ifdef FE_TONEAREST
        fesetround(old_rm);
#endif
        return ret;
    }

    if (op_type == Operation_TRUNC)
        return unary_op_inplace_fp16s<unary_op_trunc_fp16s>(bottom_top_blob, opt);

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
