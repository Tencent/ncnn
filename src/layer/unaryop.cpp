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
#include <functional>

namespace ncnn {

DEFINE_LAYER_CREATOR(UnaryOp)

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

    int size = a.total();

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i=0; i<size; i++)
    {
        a[i] = op(a[i]);
    }

    return 0;
}

template<typename T>
struct unary_op_abs : std::unary_function<T,T> {
    T operator() (const T& x) const { return fabs(x); }
};

template<typename T>
struct unary_op_neg : std::unary_function<T,T> {
    T operator() (const T& x) const { return -x; }
};

template<typename T>
struct unary_op_floor : std::unary_function<T,T> {
    T operator() (const T& x) const { return floor(x); }
};

template<typename T>
struct unary_op_ceil : std::unary_function<T,T> {
    T operator() (const T& x) const { return ceil(x); }
};

template<typename T>
struct unary_op_square : std::unary_function<T,T> {
    T operator() (const T& x) const { return x * x; }
};

template<typename T>
struct unary_op_sqrt : std::unary_function<T,T> {
    T operator() (const T& x) const { return sqrt(x); }
};

template<typename T>
struct unary_op_rsqrt : std::unary_function<T,T> {
    T operator() (const T& x) const { return 1.f / sqrt(x); }
};

template<typename T>
struct unary_op_exp : std::unary_function<T,T> {
    T operator() (const T& x) const { return exp(x); }
};

template<typename T>
struct unary_op_log : std::unary_function<T,T> {
    T operator() (const T& x) const { return log(x); }
};

template<typename T>
struct unary_op_sin : std::unary_function<T,T> {
    T operator() (const T& x) const { return sin(x); }
};

template<typename T>
struct unary_op_cos : std::unary_function<T,T> {
    T operator() (const T& x) const { return cos(x); }
};

template<typename T>
struct unary_op_tan : std::unary_function<T,T> {
    T operator() (const T& x) const { return tan(x); }
};

template<typename T>
struct unary_op_asin : std::unary_function<T,T> {
    T operator() (const T& x) const { return asin(x); }
};

template<typename T>
struct unary_op_acos : std::unary_function<T,T> {
    T operator() (const T& x) const { return acos(x); }
};

template<typename T>
struct unary_op_atan : std::unary_function<T,T> {
    T operator() (const T& x) const { return atan(x); }
};

template<typename T>
struct unary_op_reciprocal : std::unary_function<T,T> {
    T operator() (const T& x) const { return 1.f / x; }
};

int UnaryOp::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (op_type == Operation_ABS)
        return unary_op_inplace< unary_op_abs<float> >(bottom_top_blob, opt);

    if (op_type == Operation_NEG)
        return unary_op_inplace< unary_op_neg<float> >(bottom_top_blob, opt);

    if (op_type == Operation_FLOOR)
        return unary_op_inplace< unary_op_floor<float> >(bottom_top_blob, opt);

    if (op_type == Operation_CEIL)
        return unary_op_inplace< unary_op_ceil<float> >(bottom_top_blob, opt);

    if (op_type == Operation_SQUARE)
        return unary_op_inplace< unary_op_square<float> >(bottom_top_blob, opt);

    if (op_type == Operation_SQRT)
        return unary_op_inplace< unary_op_sqrt<float> >(bottom_top_blob, opt);

    if (op_type == Operation_RSQRT)
        return unary_op_inplace< unary_op_rsqrt<float> >(bottom_top_blob, opt);

    if (op_type == Operation_EXP)
        return unary_op_inplace< unary_op_exp<float> >(bottom_top_blob, opt);

    if (op_type == Operation_LOG)
        return unary_op_inplace< unary_op_log<float> >(bottom_top_blob, opt);

    if (op_type == Operation_SIN)
        return unary_op_inplace< unary_op_sin<float> >(bottom_top_blob, opt);

    if (op_type == Operation_COS)
        return unary_op_inplace< unary_op_cos<float> >(bottom_top_blob, opt);

    if (op_type == Operation_TAN)
        return unary_op_inplace< unary_op_tan<float> >(bottom_top_blob, opt);

    if (op_type == Operation_ASIN)
        return unary_op_inplace< unary_op_asin<float> >(bottom_top_blob, opt);

    if (op_type == Operation_ACOS)
        return unary_op_inplace< unary_op_acos<float> >(bottom_top_blob, opt);

    if (op_type == Operation_ATAN)
        return unary_op_inplace< unary_op_atan<float> >(bottom_top_blob, opt);

    if (op_type == Operation_RECIPROCAL)
        return unary_op_inplace< unary_op_reciprocal<float> >(bottom_top_blob, opt);

    return 0;
}

} // namespace ncnn
