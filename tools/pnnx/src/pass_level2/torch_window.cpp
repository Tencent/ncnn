// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "torch_window.h"

#include <math.h>

namespace pnnx {

static std::vector<float> make_window(int window_length, bool periodic, float alpha, float beta)
{
    std::vector<float> data(window_length);
    if (window_length == 1)
    {
        data[0] = 1.f;
        return data;
    }

    const double pi = 3.14159265358979323846;
    const int denominator = periodic ? window_length : window_length - 1;
    for (int i = 0; i < window_length; i++)
    {
        data[i] = alpha - beta * (float)cos(2.0 * pi * i / denominator);
    }

    return data;
}

static int get_static_argument(const Operator* op, const char* key, Parameter& value)
{
    if (op->has_param(key))
    {
        value = op->params.at(key);
        return 1;
    }

    for (size_t i = 0; i < op->inputs.size(); i++)
    {
        if (i >= op->inputnames.size() || op->inputnames[i] != key)
            continue;

        const Operator* producer = op->inputs[i]->producer;
        if (producer->type != "prim::Constant" || !producer->has_param("value"))
            return -1;

        value = producer->params.at("value");
        return 1;
    }

    return 0;
}

static bool get_bool_argument(const Operator* op, const char* key, bool default_value, bool& value)
{
    Parameter param;
    const int state = get_static_argument(op, key, param);
    if (state == 0)
    {
        value = default_value;
        return true;
    }
    if (state == -1 || param.type != 1)
        return false;

    value = param.b;
    return true;
}

static bool get_float_argument(const Operator* op, const char* key, float default_value, float& value)
{
    Parameter param;
    const int state = get_static_argument(op, key, param);
    if (state == 0)
    {
        value = default_value;
        return true;
    }
    if (state == -1)
        return false;

    if (param.type == 2)
    {
        value = param.i;
        return true;
    }
    if (param.type == 3)
    {
        value = param.f;
        return true;
    }

    return false;
}

static bool fold_static_window(Operator* op)
{
    const bool is_hann = op->type == "aten::hann_window";
    const bool is_hamming = op->type == "aten::hamming_window";
    if (!is_hann && !is_hamming)
        return false;

    if (op->outputs.size() != 1 || op->outputs[0]->type != 1)
        return false;

    for (size_t i = 0; i < op->inputs.size(); i++)
    {
        const Operator* producer = op->inputs[i]->producer;
        if (producer->type != "prim::Constant" || !producer->has_param("value"))
            return false;
    }

    Parameter window_length_param;
    if (get_static_argument(op, "window_length", window_length_param) != 1)
        return false;

    if (window_length_param.type != 2 || window_length_param.i < 0)
        return false;

    bool periodic;
    if (!get_bool_argument(op, "periodic", true, periodic))
        return false;

    float alpha = 0.5f;
    float beta = 0.5f;
    if (is_hamming)
    {
        if (!get_float_argument(op, "alpha", 0.54f, alpha))
            return false;
        if (!get_float_argument(op, "beta", 0.46f, beta))
            return false;
    }

    const int window_length = window_length_param.i;
    for (size_t i = 0; i < op->inputs.size(); i++)
        op->inputs[i]->remove_consumer(op);
    op->inputs.clear();
    op->inputnames.clear();
    op->type = "pnnx.Attribute";
    op->params.clear();
    op->attrs.clear();
    op->attrs["data"] = Attribute({window_length}, make_window(window_length, periodic, alpha, beta));
    return true;
}

void fold_static_windows(Graph& graph)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
        fold_static_window(graph.ops[i]);
}

} // namespace pnnx
