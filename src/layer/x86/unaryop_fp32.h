// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

namespace unaryop_x86_functor {

#include "unaryop_functor.h"

} // namespace unaryop_x86_functor

static bool unaryop_use_fma(int op_type)
{
    return op_type == UnaryOp::Operation_SQRT
           || op_type == UnaryOp::Operation_RSQRT
           || op_type == UnaryOp::Operation_EXP
           || op_type == UnaryOp::Operation_LOG
           || op_type == UnaryOp::Operation_SIN
           || op_type == UnaryOp::Operation_COS
           || op_type == UnaryOp::Operation_TAN
           || op_type == UnaryOp::Operation_ASIN
           || op_type == UnaryOp::Operation_ACOS
           || op_type == UnaryOp::Operation_ATAN
           || op_type == UnaryOp::Operation_TANH
           || op_type == UnaryOp::Operation_LOG10
           || op_type == UnaryOp::Operation_EXPM1
           || op_type == UnaryOp::Operation_SINH
           || op_type == UnaryOp::Operation_ASINH
           || op_type == UnaryOp::Operation_COSH
           || op_type == UnaryOp::Operation_ACOSH
           || op_type == UnaryOp::Operation_ATANH
           || op_type == UnaryOp::Operation_LOG1P;
}

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
int unaryop_fp32_fma(Mat& bottom_top_blob, int op_type, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
int unaryop_fp32_fma4(Mat& bottom_top_blob, int op_type, const Option& opt);
#endif

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
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _p = op.func_pack16(_p);
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
        }
        if (i < size)
        {
            const unsigned int remain = size - i;
            __mmask16 _mask = (__mmask16)((1u << remain) - 1);
            __m512 _p = _mm512_maskz_loadu_ps(_mask, ptr);
            _p = op.func_pack16(_p);
            _mm512_mask_storeu_ps(ptr, _mask, _p);
        }
#else // __AVX512F__
#if __SSE2__
#if __AVX__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = op.func_pack8(_p);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_load_ps(ptr);
            _p = op.func_pack4(_p);
            _mm_store_ps(ptr, _p);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = op.func(*ptr);
            ptr++;
        }
#endif // __AVX512F__
    }

    return 0;
}

static int unaryop_fp32(Mat& bottom_top_blob, int op_type, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (unaryop_use_fma(op_type) && ncnn::cpu_support_x86_fma())
        return unaryop_fp32_fma(bottom_top_blob, op_type, opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (unaryop_use_fma(op_type) && ncnn::cpu_support_x86_fma4())
        return unaryop_fp32_fma4(bottom_top_blob, op_type, opt);
#endif

    using namespace unaryop_x86_functor;
    if (op_type == UnaryOp::Operation_ABS)
        return unary_op_inplace<unary_op_abs>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_NEG)
        return unary_op_inplace<unary_op_neg>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_FLOOR)
        return unary_op_inplace<unary_op_floor>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_CEIL)
        return unary_op_inplace<unary_op_ceil>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_SQUARE)
        return unary_op_inplace<unary_op_square>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_SQRT)
        return unary_op_inplace<unary_op_sqrt>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_RSQRT)
        return unary_op_inplace<unary_op_rsqrt>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_EXP)
        return unary_op_inplace<unary_op_exp>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_LOG)
        return unary_op_inplace<unary_op_log>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_SIN)
        return unary_op_inplace<unary_op_sin>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_COS)
        return unary_op_inplace<unary_op_cos>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_TAN)
        return unary_op_inplace<unary_op_tan>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_ASIN)
        return unary_op_inplace<unary_op_asin>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_ACOS)
        return unary_op_inplace<unary_op_acos>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_ATAN)
        return unary_op_inplace<unary_op_atan>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_RECIPROCAL)
        return unary_op_inplace<unary_op_reciprocal>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_TANH)
        return unary_op_inplace<unary_op_tanh>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_LOG10)
        return unary_op_inplace<unary_op_log10>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_ROUND)
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

    if (op_type == UnaryOp::Operation_TRUNC)
        return unary_op_inplace<unary_op_trunc>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_SIGN)
        return unary_op_inplace<unary_op_sign>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_EXPM1)
        return unary_op_inplace<unary_op_expm1>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_SINH)
        return unary_op_inplace<unary_op_sinh>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_ASINH)
        return unary_op_inplace<unary_op_asinh>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_COSH)
        return unary_op_inplace<unary_op_cosh>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_ACOSH)
        return unary_op_inplace<unary_op_acosh>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_ATANH)
        return unary_op_inplace<unary_op_atanh>(bottom_top_blob, opt);

    if (op_type == UnaryOp::Operation_LOG1P)
        return unary_op_inplace<unary_op_log1p>(bottom_top_blob, opt);

    return 0;
}
