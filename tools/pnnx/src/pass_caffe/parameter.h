// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_CAFFE_PARAMETER_H
#define PNNX_CAFFE_PARAMETER_H

#include "ir.h"

namespace pnnx {

namespace caffe2pnnx {

static inline bool has_known_shape(const Operand* operand)
{
    // An empty shape can also be a scalar inferred by an earlier Caffe layer.
    return !operand->shape.empty() || operand->params.count("__caffe_shape_known") != 0;
}

static inline Parameter get_parameter(const Operator* op, const char* key, const Parameter& value)
{
    auto it = op->params.find(key);
    return it == op->params.end() ? value : it->second;
}

static inline int get_int(const Operator* op, const char* key, int value)
{
    return get_parameter(op, key, value).i;
}

static inline bool get_bool(const Operator* op, const char* key, bool value)
{
    return get_parameter(op, key, value).b;
}

static inline float get_float(const Operator* op, const char* key, float value)
{
    return get_parameter(op, key, value).f;
}

} // namespace caffe2pnnx

} // namespace pnnx

#endif // PNNX_CAFFE_PARAMETER_H
