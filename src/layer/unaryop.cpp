// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "unaryop.h"

#include <math.h>

namespace ncnn {

UnaryOp::UnaryOp()
{
    one_blob_only = true;
    support_inplace = true;
}

int UnaryOp::load_param(const ParamDict& pd)
{
    op_type = pd.get(0, 0);

    return 0;
}

template<typename Op>
static int unary_op_inplace(Mat& a, const Option& opt)
{
    Op op;

    int size = static_cast<int>(a.total());

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < size; i++)
    {
        a[i] = op(a[i]);
    }

    return 0;
}

struct unary_op_abs
{
    float operator()(const float& x) const
    {
        return (float)fabs(x);
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
        return (float)floor(x);
    }
};

struct unary_op_ceil
{
    float operator()(const float& x) const
    {
        return (float)ceil(x);
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
        return (float)sqrt(x);
    }
};

struct unary_op_rsqrt
{
    float operator()(const float& x) const
    {
        return (float)(1.f / sqrt(x));
    }
};

struct unary_op_exp
{
    float operator()(const float& x) const
    {
        return (float)exp(x);
    }
};

struct unary_op_log
{
    float operator()(const float& x) const
    {
        return (float)log(x);
    }
};

struct unary_op_sin
{
    float operator()(const float& x) const
    {
        return (float)sin(x);
    }
};

struct unary_op_cos
{
    float operator()(const float& x) const
    {
        return (float)cos(x);
    }
};

struct unary_op_tan
{
    float operator()(const float& x) const
    {
        return (float)tan(x);
    }
};

struct unary_op_asin
{
    float operator()(const float& x) const
    {
        return (float)asin(x);
    }
};

struct unary_op_acos
{
    float operator()(const float& x) const
    {
        return (float)acos(x);
    }
};

struct unary_op_atan
{
    float operator()(const float& x) const
    {
        return (float)atan(x);
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
        return (float)tanh(x);
    }
};

int UnaryOp::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
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

} // namespace ncnn
