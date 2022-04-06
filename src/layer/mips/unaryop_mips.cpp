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

#include "unaryop_mips.h"

#include <math.h>

#if __mips_msa
#include <msa.h>
#include "msa_mathfun.h"
#endif // __mips_msa

namespace ncnn {

UnaryOp_mips::UnaryOp_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
}

#if __mips_msa
template<typename Op>
static int unary_op_inplace_pack4(Mat& a, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int size = w * h * d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            __builtin_prefetch(ptr + 16);
            v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
            _p = op(_p);
            __msa_st_w((v4i32)_p, ptr, 0);
            ptr += 4;
        }
    }

    return 0;
}

namespace UnaryOp_mips_functor {

struct unary_op_abs_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        return (v4f32)__msa_bclri_w((v4u32)x, 31);
    }
};

struct unary_op_neg_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        return (v4f32)__msa_bnegi_w((v4u32)x, 31);
    }
};

struct unary_op_floor_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __msa_st_w((v4i32)x, tmp, 0);
        tmp[0] = floor(tmp[0]);
        tmp[1] = floor(tmp[1]);
        tmp[2] = floor(tmp[2]);
        tmp[3] = floor(tmp[3]);
        return (v4f32)__msa_ld_w(tmp, 0);
        // int old_msacsr = __msa_cfcmsa_msacsr();
        // __msa_ctcmsa_msacsr(old_msacsr | 3); // round towards -inf
        // v4f32 y = __msa_frint_w(x);
        // __msa_ctcmsa_msacsr(old_msacsr);
        // return y;
    }
};

struct unary_op_ceil_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __msa_st_w((v4i32)x, tmp, 0);
        tmp[0] = ceil(tmp[0]);
        tmp[1] = ceil(tmp[1]);
        tmp[2] = ceil(tmp[2]);
        tmp[3] = ceil(tmp[3]);
        return (v4f32)__msa_ld_w(tmp, 0);
        // int old_msacsr = __msa_cfcmsa_msacsr();
        // __msa_ctcmsa_msacsr((old_msacsr | 3) ^ 1); // round towards +inf
        // v4f32 y = __msa_frint_w(x);
        // __msa_ctcmsa_msacsr(old_msacsr);
        // return y;
    }
};

struct unary_op_square_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        return __msa_fmul_w(x, x);
    }
};

struct unary_op_sqrt_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        return __msa_fsqrt_w(x);
    }
};

struct unary_op_rsqrt_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        return __msa_frsqrt_w(x);
    }
};

struct unary_op_exp_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        return exp_ps(x);
    }
};

struct unary_op_log_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        return log_ps(x);
    }
};

struct unary_op_sin_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __msa_st_w((v4i32)x, tmp, 0);
        tmp[0] = sin(tmp[0]);
        tmp[1] = sin(tmp[1]);
        tmp[2] = sin(tmp[2]);
        tmp[3] = sin(tmp[3]);
        return (v4f32)__msa_ld_w(tmp, 0);
    }
};

struct unary_op_cos_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __msa_st_w((v4i32)x, tmp, 0);
        tmp[0] = cos(tmp[0]);
        tmp[1] = cos(tmp[1]);
        tmp[2] = cos(tmp[2]);
        tmp[3] = cos(tmp[3]);
        return (v4f32)__msa_ld_w(tmp, 0);
    }
};

struct unary_op_tan_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __msa_st_w((v4i32)x, tmp, 0);
        tmp[0] = tan(tmp[0]);
        tmp[1] = tan(tmp[1]);
        tmp[2] = tan(tmp[2]);
        tmp[3] = tan(tmp[3]);
        return (v4f32)__msa_ld_w(tmp, 0);
    }
};

struct unary_op_asin_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __msa_st_w((v4i32)x, tmp, 0);
        tmp[0] = asin(tmp[0]);
        tmp[1] = asin(tmp[1]);
        tmp[2] = asin(tmp[2]);
        tmp[3] = asin(tmp[3]);
        return (v4f32)__msa_ld_w(tmp, 0);
    }
};

struct unary_op_acos_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __msa_st_w((v4i32)x, tmp, 0);
        tmp[0] = acos(tmp[0]);
        tmp[1] = acos(tmp[1]);
        tmp[2] = acos(tmp[2]);
        tmp[3] = acos(tmp[3]);
        return (v4f32)__msa_ld_w(tmp, 0);
    }
};

struct unary_op_atan_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        // TODO msa optimize
        float tmp[4];
        __msa_st_w((v4i32)x, tmp, 0);
        tmp[0] = atan(tmp[0]);
        tmp[1] = atan(tmp[1]);
        tmp[2] = atan(tmp[2]);
        tmp[3] = atan(tmp[3]);
        return (v4f32)__msa_ld_w(tmp, 0);
    }
};

struct unary_op_reciprocal_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        return __msa_frcp_w(x);
    }
};

struct unary_op_tanh_pack4
{
    v4f32 operator()(const v4f32& x) const
    {
        return tanh_ps(x);
    }
};

} // namespace UnaryOp_mips_functor
#endif // __mips_msa

int UnaryOp_mips::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if __mips_msa
    using namespace UnaryOp_mips_functor;

    int elempack = bottom_top_blob.elempack;

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
#endif // __mips_msa

    return UnaryOp::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn
