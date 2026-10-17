// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "ir.h"
#include "model_stat.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <fstream>
#include <iterator>

static bool same_float(float a, float b)
{
    if (std::isnan(a))
        return std::isnan(b);
    return a == b;
}

static int test_python(float value, bool as_tensor)
{
    pnnx::Graph graph;
    pnnx::Operator* input = graph.new_operator("pnnx.Input", "input");
    pnnx::Operand* x = graph.new_operand("in0");
    x->type = as_tensor ? 9 : 1;
    x->shape = {3};
    x->producer = input;
    input->outputs.push_back(x);
    pnnx::Operator* op = graph.new_operator(as_tensor ? "torch.where" : "torch.add", "op");
    op->inputs.push_back(x);
    op->inputnames.push_back(as_tensor ? "condition" : "input");
    x->consumers.push_back(op);
    if (as_tensor)
        op->params["input"] = value;
    op->params["other"] = as_tensor ? 0.f : value;
    pnnx::Operand* y = graph.new_operand("out0");
    y->type = 1;
    y->shape = {3};
    y->producer = op;
    op->outputs.push_back(y);
    pnnx::Operator* output = graph.new_operator("pnnx.Output", "output");
    output->inputs.push_back(y);
    y->consumers.push_back(output);
    if (graph.save("test_pnnx_ir.param", "test_pnnx_ir.bin") != 0)
        return 1;
    pnnx::Graph restored;
    if (restored.load("test_pnnx_ir.param", "test_pnnx_ir.bin") != 0 || restored.python("test_pnnx_ir.py", "test_pnnx_ir.bin", {}, pnnx::ModelStat()) != 0)
        return 1;
    std::ifstream stream("test_pnnx_ir.py");
    const std::string code((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    const std::string scalar = std::isnan(value) ? "float('nan')" : (value < 0 ? "float('-inf')" : "float('inf')");
    return code.find(as_tensor ? "input=torch.tensor(" + scalar + ")" : "other=" + scalar) == std::string::npos;
}

int main()
{
    const float values[] = {0.f, -0.f, 2.f, 1.25f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
    pnnx::Graph graph;
    pnnx::Operator* op = graph.new_operator("test", "op");
    for (int i = 0; i < 7; i++)
    {
        const pnnx::Parameter original(values[i]);
        const pnnx::Parameter restored = pnnx::Parameter::parse_from_string(pnnx::Parameter::encode_to_string(original));
        if (restored.type != 3 || !same_float(original.f, restored.f))
            return 1;
        op->params[std::to_string(i)] = original;
    }

    const char* strings[] = {"inf", "-inf", "nan", "nan.0", "float('inf')", "float('-inf')", "float('nan')"};
    for (int i = 0; i < 7; i++)
    {
        const pnnx::Parameter original(strings[i]);
        const pnnx::Parameter restored = pnnx::Parameter::parse_from_string(pnnx::Parameter::encode_to_string(original));
        if (restored.type != 4 || original.s != restored.s)
            return 1;
        op->params["string" + std::to_string(i)] = original;
    }

    const pnnx::Parameter array(std::vector<float> {0.f, -2.f, 1.25f});
    const pnnx::Parameter restored_array = pnnx::Parameter::parse_from_string(pnnx::Parameter::encode_to_string(array));
    if (restored_array.type != 6 || restored_array.af != array.af)
        return 1;
    op->params["array"] = array;

    if (graph.save("test_pnnx_ir.param", "test_pnnx_ir.bin") != 0)
        return 1;
    pnnx::Graph restored;
    if (restored.load("test_pnnx_ir.param", "test_pnnx_ir.bin") != 0 || restored.ops.size() != 1)
        return 1;
    for (const auto& param : op->params)
    {
        const pnnx::Parameter& value = restored.ops[0]->params.at(param.first);
        if (value.type != param.second.type || (value.type == 3 && !same_float(value.f, param.second.f)) || (value.type == 4 && value.s != param.second.s) || (value.type == 6 && value.af != param.second.af))
            return 1;
    }
    for (int i = 4; i < 7; i++)
    {
        if (test_python(values[i], false) || test_python(values[i], true))
            return 1;
    }
    return 0;
}
