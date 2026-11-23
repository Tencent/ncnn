// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

struct unary_op_abs
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return __riscv_vfsgnj_vf_f32m8(x, 1.f, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)fabsf(x);
    }
#endif // __riscv_vector
};

struct unary_op_neg
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return __riscv_vfneg_v_f32m8(x, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return -x;
    }
#endif // __riscv_vector
};

struct unary_op_floor
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return __riscv_vfcvt_f_x_v_f32m8(__riscv_vfcvt_x_f_v_i32m8_rm(x, __RISCV_FRM_RDN, vl), vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)floorf(x);
    }
#endif // __riscv_vector
};

struct unary_op_ceil
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return __riscv_vfcvt_f_x_v_f32m8(__riscv_vfcvt_x_f_v_i32m8_rm(x, __RISCV_FRM_RUP, vl), vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)ceilf(x);
    }
#endif // __riscv_vector
};

struct unary_op_square
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return __riscv_vfmul_vv_f32m8(x, x, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return x * x;
    }
#endif // __riscv_vector
};

struct unary_op_sqrt
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return __riscv_vfsqrt_v_f32m8(x, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)sqrtf(x);
    }
#endif // __riscv_vector
};

struct unary_op_rsqrt
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
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
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return 1.f / sqrtf(x);
    }
#endif // __riscv_vector
};

struct unary_op_exp
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return exp_ps(x, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)expf(x);
    }
#endif // __riscv_vector
};

struct unary_op_log
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return log_ps(x, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)logf(x);
    }
#endif // __riscv_vector
};

struct unary_op_sin
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return sin_ps(x, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)sinf(x);
    }
#endif // __riscv_vector
};

struct unary_op_cos
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return cos_ps(x, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)cosf(x);
    }
#endif // __riscv_vector
};

struct unary_op_tan
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
#if __riscv_xtheadvector
        std::vector<float> tmp(vl);
        __riscv_vse32_v_f32m8(tmp.data(), x, vl);
        for (size_t i = 0; i < vl; i++)
        {
            tmp[i] = tanf(tmp[i]);
        }
        return __riscv_vle32_v_f32m8(tmp.data(), vl);
#else
        vfloat32m8_t sin_x, cos_x;
        sincos_ps(x, &sin_x, &cos_x, vl);
        return __riscv_vfdiv_vv_f32m8(sin_x, cos_x, vl);
#endif
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)tanf(x);
    }
#endif // __riscv_vector
};

struct unary_op_asin
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
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
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)asinf(x);
    }
#endif // __riscv_vector
};

struct unary_op_acos
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
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
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)acosf(x);
    }
#endif // __riscv_vector
};

struct unary_op_atan
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
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
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)atanf(x);
    }
#endif // __riscv_vector
};

struct unary_op_reciprocal
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
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
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return 1.f / x;
    }
#endif // __riscv_vector
};

struct unary_op_tanh
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return tanh_ps(x, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)tanhf(x);
    }
#endif // __riscv_vector
};

struct unary_op_log10
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return __riscv_vfmul_vf_f32m8(log_ps(x, vl), 0.434294481903, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)log10f(x);
    }
#endif // __riscv_vector
};

struct unary_op_round
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return __riscv_vfcvt_f_x_v_f32m8(__riscv_vfcvt_x_f_v_i32m8(x, vl), vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return nearbyintf(x);
    }
#endif // __riscv_vector
};

struct unary_op_trunc
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
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
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)truncf(x);
    }
#endif // __riscv_vector
};

struct unary_op_sign
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        vbool4_t pos = __riscv_vmfgt_vf_f32m8_b4(x, 0.f, vl);
        vbool4_t neg = __riscv_vmflt_vf_f32m8_b4(x, 0.f, vl);
        vfloat32m8_t zero = __riscv_vfmv_v_f_f32m8(0.f, vl);
        vfloat32m8_t one = __riscv_vfmv_v_f_f32m8(1.f, vl);
        vfloat32m8_t negone = __riscv_vfmv_v_f_f32m8(-1.f, vl);
        return __riscv_vmerge_vvm_f32m8(__riscv_vmerge_vvm_f32m8(zero, one, pos, vl), negone, neg, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return x > 0.f ? 1.f : x < 0.f ? -1.f : 0.f;
    }
#endif // __riscv_vector
};

struct unary_op_expm1
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return expm1_ps(x, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)expm1f(x);
    }
#endif // __riscv_vector
};

struct unary_op_sinh
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return sinh_ps(x, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)sinhf(x);
    }
#endif // __riscv_vector
};

struct unary_op_asinh
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return asinh_ps(x, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)asinhf(x);
    }
#endif // __riscv_vector
};

struct unary_op_cosh
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return cosh_ps(x, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)coshf(x);
    }
#endif // __riscv_vector
};

struct unary_op_acosh
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return acosh_ps(x, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)acoshf(x);
    }
#endif // __riscv_vector
};

struct unary_op_atanh
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return atanh_ps(x, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)atanhf(x);
    }
#endif // __riscv_vector
};

struct unary_op_log1p
{
#if __riscv_vector
    NCNN_FORCEINLINE vfloat32m8_t operator()(const vfloat32m8_t& x, const size_t& vl) const
    {
        return log1p_ps(x, vl);
    }
#else  // __riscv_vector
    NCNN_FORCEINLINE float operator()(const float& x) const
    {
        return (float)log1pf(x);
    }
#endif // __riscv_vector
};
