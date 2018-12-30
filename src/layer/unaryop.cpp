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
    support_vulkan = true;
}

int UnaryOp::load_param(const ParamDict& pd)
{
    op_type = pd.get(0, 0);

#if NCNN_VULKAN
    if (pd.use_vulkan_compute)
    {
        set_optimal_local_size_xyz();

        specializations.resize(1);
        specializations[0].i = op_type;

        binding_count = 1;
        push_constant_count = 5;
    }
#endif // NCNN_VULKAN

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

#if NCNN_VULKAN
int UnaryOp::forward_inplace(VkMat& bottom_top_blob, Command& cmd, const Option& opt) const
{
    fprintf(stderr, "UnaryOp::forward_inplace %p\n", bottom_top_blob.buffer);

    std::vector<VkMat> bindings(1);
    bindings[0] = bottom_top_blob;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;

    uint32_t group_count_xyz[3];
    group_count_xyz[0] = (bottom_top_blob.w + local_size_x - 1) / local_size_x;
    group_count_xyz[1] = (bottom_top_blob.h + local_size_y - 1) / local_size_y;
    group_count_xyz[2] = (bottom_top_blob.c + local_size_z - 1) / local_size_z;

    // record
    cmd.record_bind_pipeline(pipeline);
    cmd.record_update_bindings(pipeline_layout, descriptorset_layout, descriptor_update_template, bindings);
    cmd.record_push_constants(pipeline_layout, constants);
    cmd.record_dispatch(group_count_xyz);

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
