// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "unaryop_riscv.h"

#if __riscv_vector
#ifdef RVV_SPEC_0_7
#include "riscv_v_071_fix.h"
#else
#include <riscv_vector.h>
#endif
#include "rvv_mathfun.h"
#include "rvv_mathfun_fp16s.h"
#endif // __riscv_vector

#include <math.h>

namespace ncnn {

UnaryOp_riscv::UnaryOp_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector
}

#if __riscv_vector
template<typename Op>
static int unary_op_inplace(Mat& a, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;
    int elempack = a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        int n = size * elempack;
        while (n > 0)
        {
            word_type vl = vsetvl_e32m8(n);

            vfloat32m8_t _p = vle32_v_f32m8(ptr, vl);
            _p = op(_p, vl);
            vse32_v_f32m8(ptr, _p, vl);

            ptr += vl;
            n -= vl;
        }
    }

    return 0;
}

struct unary_op_abs
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        return vfsgnj_vf_f32m8(x, 1.f, vl);
    }
};

struct unary_op_neg
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        return vfneg_v_f32m8(x, vl);
    }
};

struct unary_op_floor
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        vint32m8_t _xi = vfcvt_x_f_v_i32m8(x, vl);
        vbool4_t _mask = vmfgt_vv_f32m8_b4(vfcvt_f_x_v_f32m8(_xi, vl), x, vl);
        return vfcvt_f_x_v_f32m8(vsub_vx_i32m8_m(_mask, _xi, _xi, 1, vl), vl);
    }
};

struct unary_op_ceil
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        vint32m8_t _xi = vfcvt_x_f_v_i32m8(x, vl);
        vbool4_t _mask = vmflt_vv_f32m8_b4(vfcvt_f_x_v_f32m8(_xi, vl), x, vl);
        return vfcvt_f_x_v_f32m8(vadd_vx_i32m8_m(_mask, _xi, _xi, 1, vl), vl);
    }
};

struct unary_op_square
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        return vfmul_vv_f32m8(x, x, vl);
    }
};

struct unary_op_sqrt
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        return vfsqrt_v_f32m8(x, vl);
    }
};

struct unary_op_rsqrt
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        vfloat32m8_t _reciprocal = vfrsqrt7_v_f32m8(x, vl);
        _reciprocal = vfmul_vv_f32m8(vfrsub_vf_f32m8(vfmul_vv_f32m8(vfmul_vf_f32m8(x, 0.5f, vl), vfmul_vv_f32m8(_reciprocal, _reciprocal, vl), vl), 1.5f, vl), _reciprocal, vl);
        // _reciprocal = vfmul_vv_f32m8(vfrsub_vf_f32m8(vfmul_vv_f32m8(vfmul_vf_f32m8(x, 0.5f, vl), vfmul_vv_f32m8(_reciprocal, _reciprocal, vl), vl), 1.5f, vl), _reciprocal, vl);
        return _reciprocal;
    }
};

struct unary_op_exp
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        return exp_ps(x, vl);
    }
};

struct unary_op_log
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        return log_ps(x, vl);
    }
};

struct unary_op_sin
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        return sin_ps(x, vl);
    }
};

struct unary_op_cos
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        return cos_ps(x, vl);
    }
};

struct unary_op_tan
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        // TODO rvv optimize
        std::vector<float> tmp(vl);
        vse32_v_f32m8(tmp.data(), x, vl);
        for (int i = 0; i < vl; i++)
        {
            tmp[i] = tan(tmp[i]);
        }
        return vle32_v_f32m8(tmp.data(), vl);
    }
};

struct unary_op_asin
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        // TODO rvv optimize
        std::vector<float> tmp(vl);
        vse32_v_f32m8(tmp.data(), x, vl);
        for (int i = 0; i < vl; i++)
        {
            tmp[i] = asin(tmp[i]);
        }
        return vle32_v_f32m8(tmp.data(), vl);
    }
};

struct unary_op_acos
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        // TODO rvv optimize
        std::vector<float> tmp(vl);
        vse32_v_f32m8(tmp.data(), x, vl);
        for (int i = 0; i < vl; i++)
        {
            tmp[i] = acos(tmp[i]);
        }
        return vle32_v_f32m8(tmp.data(), vl);
    }
};

struct unary_op_atan
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        // TODO rvv optimize
        std::vector<float> tmp(vl);
        vse32_v_f32m8(tmp.data(), x, vl);
        for (int i = 0; i < vl; i++)
        {
            tmp[i] = atan(tmp[i]);
        }
        return vle32_v_f32m8(tmp.data(), vl);
    }
};

struct unary_op_reciprocal
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        vfloat32m8_t _reciprocal = vfrec7_v_f32m8(x, vl);
        _reciprocal = vfmul_vv_f32m8(vfrsub_vf_f32m8(vfmul_vv_f32m8(x, _reciprocal, vl), 2.f, vl), _reciprocal, vl);
        // _reciprocal = vfmul_vv_f32m8(vfrsub_vf_f32m8(vfmul_vv_f32m8(x, _reciprocal, vl), 2.f, vl), _reciprocal, vl);
        return _reciprocal;
    }
};

struct unary_op_tanh
{
    vfloat32m8_t operator()(const vfloat32m8_t& x, const word_type& vl) const
    {
        return tanh_ps(x, vl);
    }
};
#endif // __riscv_vector

int UnaryOp_riscv::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
        return forward_inplace_fp16s(bottom_top_blob, opt);
#endif

#if __riscv_vector
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
#else  // __riscv_vector
    return UnaryOp::forward_inplace(bottom_top_blob, opt);
#endif // __riscv_vector
}

#if __riscv_vector && __riscv_zfh
template<typename Op>
static int unary_op_inplace_fp16s(Mat& a, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;
    int elempack = a.elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        __fp16* ptr = a.channel(q);

        int n = size * elempack;
        while (n > 0)
        {
            word_type vl = vsetvl_e16m8(n);

            vfloat16m8_t _p = vle16_v_f16m8(ptr, vl);
            _p = op(_p, vl);
            vse16_v_f16m8(ptr, _p, vl);

            ptr += vl;
            n -= vl;
        }
    }

    return 0;
}

struct unary_op_abs_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        return vfsgnj_vf_f16m8(x, 1.f, vl);
    }
};

struct unary_op_neg_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        return vfneg_v_f16m8(x, vl);
    }
};

struct unary_op_floor_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        vint16m8_t _xi = vfcvt_x_f_v_i16m8(x, vl);
        vbool2_t _mask = vmfgt_vv_f16m8_b2(vfcvt_f_x_v_f16m8(_xi, vl), x, vl);
        return vfcvt_f_x_v_f16m8(vsub_vx_i16m8_m(_mask, _xi, _xi, 1, vl), vl);
    }
};

struct unary_op_ceil_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        vint16m8_t _xi = vfcvt_x_f_v_i16m8(x, vl);
        vbool2_t _mask = vmflt_vv_f16m8_b2(vfcvt_f_x_v_f16m8(_xi, vl), x, vl);
        return vfcvt_f_x_v_f16m8(vadd_vx_i16m8_m(_mask, _xi, _xi, 1, vl), vl);
    }
};

struct unary_op_square_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        return vfmul_vv_f16m8(x, x, vl);
    }
};

struct unary_op_sqrt_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        return vfsqrt_v_f16m8(x, vl);
    }
};

struct unary_op_rsqrt_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        vfloat16m8_t _reciprocal = vfrsqrt7_v_f16m8(x, vl);
        _reciprocal = vfmul_vv_f16m8(vfrsub_vf_f16m8(vfmul_vv_f16m8(vfmul_vf_f16m8(x, 0.5f, vl), vfmul_vv_f16m8(_reciprocal, _reciprocal, vl), vl), 1.5f, vl), _reciprocal, vl);
        // _reciprocal = vfmul_vv_f16m8(vfrsub_vf_f16m8(vfmul_vv_f16m8(vfmul_vf_f16m8(x, 0.5f, vl), vfmul_vv_f16m8(_reciprocal, _reciprocal, vl), vl), 1.5f, vl), _reciprocal, vl);
        return _reciprocal;
    }
};

struct unary_op_exp_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        return exp_ps(x, vl);
    }
};

struct unary_op_log_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        return log_ps(x, vl);
    }
};

struct unary_op_sin_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        return sin_ps(x, vl);
    }
};

struct unary_op_cos_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        return cos_ps(x, vl);
    }
};

struct unary_op_tan_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        // TODO rvv optimize
        std::vector<__fp16> tmp(vl);
        vse16_v_f16m8(tmp.data(), x, vl);
        for (int i = 0; i < vl; i++)
        {
            tmp[i] = tan((float)tmp[i]);
        }
        return vle16_v_f16m8(tmp.data(), vl);
    }
};

struct unary_op_asin_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        // TODO rvv optimize
        std::vector<__fp16> tmp(vl);
        vse16_v_f16m8(tmp.data(), x, vl);
        for (int i = 0; i < vl; i++)
        {
            tmp[i] = asin((float)tmp[i]);
        }
        return vle16_v_f16m8(tmp.data(), vl);
    }
};

struct unary_op_acos_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        // TODO rvv optimize
        std::vector<__fp16> tmp(vl);
        vse16_v_f16m8(tmp.data(), x, vl);
        for (int i = 0; i < vl; i++)
        {
            tmp[i] = acos((float)tmp[i]);
        }
        return vle16_v_f16m8(tmp.data(), vl);
    }
};

struct unary_op_atan_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        // TODO rvv optimize
        std::vector<__fp16> tmp(vl);
        vse16_v_f16m8(tmp.data(), x, vl);
        for (int i = 0; i < vl; i++)
        {
            tmp[i] = atan((float)tmp[i]);
        }
        return vle16_v_f16m8(tmp.data(), vl);
    }
};

struct unary_op_reciprocal_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        vfloat16m8_t _reciprocal = vfrec7_v_f16m8(x, vl);
        _reciprocal = vfmul_vv_f16m8(vfrsub_vf_f16m8(vfmul_vv_f16m8(x, _reciprocal, vl), 2.f, vl), _reciprocal, vl);
        // _reciprocal = vfmul_vv_f16m8(vfrsub_vf_f16m8(vfmul_vv_f16m8(x, _reciprocal, vl), 2.f, vl), _reciprocal, vl);
        return _reciprocal;
    }
};

struct unary_op_tanh_fp16s
{
    vfloat16m8_t operator()(const vfloat16m8_t& x, const word_type& vl) const
    {
        return tanh_ps(x, vl);
    }
};

int UnaryOp_riscv::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
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

    return 0;
}
#endif // __riscv_vector && __riscv_zfh

} // namespace ncnn
