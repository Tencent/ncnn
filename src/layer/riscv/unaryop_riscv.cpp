// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "unaryop_riscv.h"

#include <float.h>

#if __riscv_vector
#include <riscv_vector.h>
#include "rvv_mathfun.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

UnaryOp_riscv::UnaryOp_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif
}

#if __riscv_vector
template<typename Op>
static int unary_op_inplace(Mat& a, const Option& opt)
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
        float* ptr = a.channel(q);

        int n = size * elempack;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m8(n);

            vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
            _p = op(_p, vl);
            __riscv_vse32_v_f32m8(ptr, _p, vl);

            ptr += vl;
            n -= vl;
        }
    }

    return 0;
}

namespace UnaryOp_riscv_functor {

struct unary_op_abs
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return __riscv_vfsgnj_vf_f32m8(x, 1.f, vl);
    }
};

struct unary_op_neg
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return __riscv_vfneg_v_f32m8(x, vl);
    }
};

struct unary_op_floor
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return __riscv_vfcvt_f_x_v_f32m8(__riscv_vfcvt_x_f_v_i32m8_rm(x, __RISCV_FRM_RDN, vl), vl);
    }
};

struct unary_op_ceil
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return __riscv_vfcvt_f_x_v_f32m8(__riscv_vfcvt_x_f_v_i32m8_rm(x, __RISCV_FRM_RUP, vl), vl);
    }
};

struct unary_op_square
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return __riscv_vfmul_vv_f32m8(x, x, vl);
    }
};

struct unary_op_sqrt
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return __riscv_vfsqrt_v_f32m8(x, vl);
    }
};

struct unary_op_rsqrt
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
#if __riscv_xtheadvector
        vfloat32m8_t _reciprocal = __riscv_vfrdiv_vf_f32m8(__riscv_vfsqrt_v_f32m8(x, vl), 1.f, vl);
#else
        vfloat32m8_t _reciprocal = __riscv_vfrsqrt7_v_f32m8(x, vl);
        _reciprocal = __riscv_vfmul_vv_f32m8(__riscv_vfrsub_vf_f32m8(__riscv_vfmul_vv_f32m8(__riscv_vfmul_vf_f32m8(x, 0.5f, vl), __riscv_vfmul_vv_f32m8(_reciprocal, _reciprocal, vl), vl), 1.5f, vl), _reciprocal, vl);
        // _reciprocal = __riscv_vfmul_vv_f32m8(__riscv_vfrsub_vf_f32m8(__riscv_vfmul_vv_f32m8(__riscv_vfmul_vf_f32m8(x, 0.5f, vl), __riscv_vfmul_vv_f32m8(_reciprocal, _reciprocal, vl), vl), 1.5f, vl), _reciprocal, vl);
#endif
        return _reciprocal;
    }
};

struct unary_op_exp
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return exp_ps(x, vl);
    }
};

struct unary_op_log
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return log_ps(x, vl);
    }
};

struct unary_op_sin
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return sin_ps(x, vl);
    }
};

struct unary_op_cos
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return cos_ps(x, vl);
    }
};

struct unary_op_tan
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        // TODO rvv optimize
        std::vector<float> tmp(vl);
        __riscv_vse32_v_f32m8(tmp.data(), x, vl);
        for (size_t i = 0; i < vl; i++)
        {
            tmp[i] = tanf(tmp[i]);
        }
        return __riscv_vle32_v_f32m8(tmp.data(), vl);
    }
};

struct unary_op_asin
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        // TODO rvv optimize
        std::vector<float> tmp(vl);
        __riscv_vse32_v_f32m8(tmp.data(), x, vl);
        for (size_t i = 0; i < vl; i++)
        {
            tmp[i] = asinf(tmp[i]);
        }
        return __riscv_vle32_v_f32m8(tmp.data(), vl);
    }
};

struct unary_op_acos
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        // TODO rvv optimize
        std::vector<float> tmp(vl);
        __riscv_vse32_v_f32m8(tmp.data(), x, vl);
        for (size_t i = 0; i < vl; i++)
        {
            tmp[i] = acosf(tmp[i]);
        }
        return __riscv_vle32_v_f32m8(tmp.data(), vl);
    }
};

struct unary_op_atan
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        // TODO rvv optimize
        std::vector<float> tmp(vl);
        __riscv_vse32_v_f32m8(tmp.data(), x, vl);
        for (size_t i = 0; i < vl; i++)
        {
            tmp[i] = atanf(tmp[i]);
        }
        return __riscv_vle32_v_f32m8(tmp.data(), vl);
    }
};

struct unary_op_reciprocal
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
#if __riscv_xtheadvector
        vfloat32m8_t _reciprocal = __riscv_vfrdiv_vf_f32m8(x, 1.f, vl);
#else
        vfloat32m8_t _reciprocal = __riscv_vfrec7_v_f32m8(x, vl);
        _reciprocal = __riscv_vfmul_vv_f32m8(__riscv_vfrsub_vf_f32m8(__riscv_vfmul_vv_f32m8(x, _reciprocal, vl), 2.f, vl), _reciprocal, vl);
        // _reciprocal = __riscv_vfmul_vv_f32m8(__riscv_vfrsub_vf_f32m8(__riscv_vfmul_vv_f32m8(x, _reciprocal, vl), 2.f, vl), _reciprocal, vl);
#endif
        return _reciprocal;
    }
};

struct unary_op_tanh
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return tanh_ps(x, vl);
    }
};

struct unary_op_log10
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return __riscv_vfmul_vf_f32m8(log_ps(x, vl), 0.434294481903, vl);
    }
};

struct unary_op_round
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return __riscv_vfcvt_f_x_v_f32m8(__riscv_vfcvt_x_f_v_i32m8(x, vl), vl);
    }
};

struct unary_op_trunc
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
#if __riscv_xtheadvector
        // simulate trunc with floor positives and ceil negative
        // xi = round(x)
        // floorx = xi - (xi > x)
        // ceilx = xi + (xi < x)
        // truncx = x >= 0 ? floorx : ceilx
        vint32m8_t _xi = __riscv_vfcvt_x_f_v_i32m8(x, vl);
        vfloat32m8_t _xf = __riscv_vfcvt_f_x_v_f32m8(_xi, vl);
        vbool4_t _floormask = __riscv_vmfgt_vv_f32m8_b4(_xf, x, vl);
        vint32m8_t _floorx = __riscv_vsub_vx_i32m8_mu(_floormask, _xi, _xi, 1, vl);
        vbool4_t _ceilmask = __riscv_vmflt_vv_f32m8_b4(_xf, x, vl);
        vint32m8_t _ceilx = __riscv_vadd_vx_i32m8_mu(_ceilmask, _xi, _xi, 1, vl);
        vbool4_t _negative = __riscv_vmflt_vf_f32m8_b4(x, 0.f, vl);
        return __riscv_vfcvt_f_x_v_f32m8(__riscv_vmerge_vvm_i32m8(_floorx, _ceilx, _negative, vl), vl);
#else
        return __riscv_vfcvt_f_x_v_f32m8(__riscv_vfcvt_rtz_x_f_v_i32m8(x, vl), vl);
#endif
    }
};

} // namespace UnaryOp_riscv_functor
#endif // __riscv_vector

int UnaryOp_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_ZFH
    int elembits = bottom_top_blob.elembits();

    if (opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blob, opt);
#endif

#if __riscv_vector
    using namespace UnaryOp_riscv_functor;

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
    {
        // round to nearest even
#ifdef FE_TONEAREST
        int old_rm = fegetround();
        fesetround(FE_TONEAREST);
#endif
        int ret = unary_op_inplace<unary_op_round>(bottom_top_blob, opt);
#ifdef FE_TONEAREST
        fesetround(old_rm);
#endif
        return ret;
    }

    if (op_type == Operation_TRUNC)
        return unary_op_inplace<unary_op_trunc>(bottom_top_blob, opt);

    return 0;
#else  // __riscv_vector
    return UnaryOp::forward_inplace(bottom_top_blob, opt);
#endif // __riscv_vector
}

} // namespace ncnn
