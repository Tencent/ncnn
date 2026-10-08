// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape_shape.h"

#include <algorithm>
#include <cstdlib>

namespace pnnx {

namespace ncnn {

static int batch_axis(const Operand* operand)
{
    const auto it = operand->params.find("__ncnn_batch_axis");
    int axis = it == operand->params.end() ? 233 : it->second.i;
    if (axis < 0)
        axis += (int)operand->shape.size();
    return axis;
}

std::string logical_dim_reference(const Operand* operand, int reference_index, int axis)
{
    int rank = (int)operand->shape.size();
    if (axis < 0)
        axis += rank;
    if (axis < 0 || axis >= rank || reference_index < 0 || reference_index > 9)
    {
        fprintf(stderr, "reshape %s has unsupported logical dimension %d or reference %d\n", operand->name.c_str(), axis, reference_index);
        return std::string();
    }

    const int native_batch_axis = batch_axis(operand);
    if (axis == native_batch_axis)
        return std::to_string(reference_index) + "n";
    if (native_batch_axis != 233)
    {
        rank--;
        if (axis > native_batch_axis)
            axis--;
    }

    static const char* dimensions[] = {"", "w", "hw", "chw", "cdhw"};
    if (rank < 1 || rank > 4 || axis >= rank)
    {
        fprintf(stderr, "reshape %s has unsupported physical rank %d\n", operand->name.c_str(), rank);
        return std::string();
    }
    return std::to_string(reference_index) + dimensions[rank][axis];
}

std::vector<std::string> logical_shape(const Operand* operand, int reference_index)
{
    const std::string& producer_type = operand->producer->type;
    const bool static_source = producer_type == "pnnx.Input" || producer_type == "Input" || producer_type == "pnnx.Attribute" || producer_type == "MemoryData";
    std::vector<std::string> shape;
    for (int i = 0; i < (int)operand->shape.size(); i++)
    {
        // input specifications and constant tensors establish static extents
        // intermediate metadata only records observations from shape inference
        shape.push_back(reference_index == 0 && static_source && operand->shape[i] > 0 ? std::to_string(operand->shape[i]) : logical_dim_reference(operand, reference_index, i));
    }
    return shape;
}

std::vector<std::string> split_shape_expression(const std::string& expression)
{
    std::vector<std::string> dimensions;
    size_t start = 0;
    int depth = 0;
    for (size_t i = 0; i <= expression.size(); i++)
    {
        if (i == expression.size() || (expression[i] == ',' && depth == 0))
        {
            dimensions.push_back(expression.substr(start, i - start));
            start = i + 1;
        }
        else if (expression[i] == '(')
            depth++;
        else if (expression[i] == ')')
            depth--;
    }
    return dimensions;
}

static bool integer_dimension(const std::string& expression, long long& value)
{
    char* end = 0;
    value = strtoll(expression.c_str(), &end, 10);
    return end != expression.c_str() && *end == 0;
}

static void product_factors(const std::string& expression, long long& constant, std::vector<std::string>& factors)
{
    long long value;
    if (integer_dimension(expression, value) && value > 0)
    {
        constant *= value;
        return;
    }
    if (expression.compare(0, 2, "*(") == 0 && expression.back() == ')')
    {
        const auto operands = split_shape_expression(expression.substr(2, expression.size() - 3));
        if (operands.size() == 2)
        {
            product_factors(operands[0], constant, factors);
            product_factors(operands[1], constant, factors);
            return;
        }
    }
    factors.push_back(expression);
}

static std::string make_product(long long constant, std::vector<std::string> factors)
{
    std::sort(factors.begin(), factors.end());
    std::string expression = constant != 1 || factors.empty() ? std::to_string(constant) : std::string();
    for (const auto& factor : factors)
        expression = expression.empty() ? factor : "*(" + expression + "," + factor + ")";
    return expression;
}

std::string shape_product(const std::vector<std::string>& dimensions)
{
    long long constant = 1;
    std::vector<std::string> factors;
    for (const auto& dim : dimensions)
    {
        if (dim.empty())
            return std::string();
        product_factors(dim, constant, factors);
    }
    return make_product(constant, factors);
}

std::string shape_quotient(const std::string& numerator, const std::string& denominator)
{
    if (numerator.empty() || denominator.empty())
        return std::string();

    long long nc = 1;
    long long dc = 1;
    std::vector<std::string> nf;
    std::vector<std::string> df;
    product_factors(numerator, nc, nf);
    product_factors(denominator, dc, df);
    for (auto it = df.begin(); it != df.end();)
    {
        auto n = std::find(nf.begin(), nf.end(), *it);
        if (n == nf.end())
        {
            ++it;
            continue;
        }
        nf.erase(n);
        it = df.erase(it);
    }
    if (dc > 0 && nc % dc == 0)
    {
        nc /= dc;
        dc = 1;
    }
    const std::string n = make_product(nc, nf);
    const std::string d = make_product(dc, df);
    return d == "1" ? n : "//(" + n + "," + d + ")";
}

enum PartitionRelation
{
    PartitionSame,
    PartitionDifferent,
    PartitionUnknown
};

struct ReshapePlan
{
    int input_axis;
    int output_axis;
    PartitionRelation partition;
    bool full_context;
};

static std::vector<size_t> dimension_references(const std::string& expression)
{
    std::vector<size_t> positions;
    size_t start = 0;
    for (size_t i = 0; i <= expression.size(); i++)
    {
        if (i != expression.size() && expression[i] != '(' && expression[i] != ')' && expression[i] != ',')
            continue;
        if (i - start == 2 && expression[start] >= '0' && expression[start] <= '9')
        {
            const char dim = expression[start + 1];
            if (dim == 'w' || dim == 'h' || dim == 'd' || dim == 'c' || dim == 'n')
                positions.push_back(start);
        }
        start = i + 1;
    }
    return positions;
}

static std::string partition_size(const std::vector<std::string>& shape, int axis)
{
    if (axis == 233)
        return "1";
    return shape[axis];
}

static std::string partition_suffix(const std::vector<std::string>& shape, int axis)
{
    return axis == 233 ? "1" : shape_product(std::vector<std::string>(shape.begin() + axis + 1, shape.end()));
}

static bool dimension_differs(const std::string& a, const std::string& b)
{
    long long av;
    long long bv;
    return integer_dimension(a, av) && integer_dimension(b, bv) && av != bv;
}

void write_reshape_shape(Operator* op, std::vector<std::string> shape)
{
    if (shape.empty())
    {
        fprintf(stderr, "reshape %s: target rank is unknown\n", op->name.c_str());
        return;
    }

    const auto input_shape = logical_shape(op->inputs[0], 0);
    ReshapePlan plan = {batch_axis(op->inputs[0]), batch_axis(op->outputs[0]), PartitionUnknown, false};
    if ((plan.input_axis != 233 && (plan.input_axis < 0 || plan.input_axis >= (int)input_shape.size()))
            || (plan.output_axis != 233 && (plan.output_axis < 0 || plan.output_axis >= (int)shape.size())))
    {
        fprintf(stderr, "reshape %s: batch axis is outside logical rank\n", op->name.c_str());
        return;
    }
    if (std::find(input_shape.begin(), input_shape.end(), std::string()) != input_shape.end()
            || std::find(shape.begin(), shape.end(), std::string()) != shape.end())
    {
        fprintf(stderr, "reshape %s: unsupported dimension reference\n", op->name.c_str());
        return;
    }

    // resolve the single infer dimension from the full logical element count
    int infer = -1;
    std::vector<std::string> known_dimensions;
    for (int i = 0; i < (int)shape.size(); i++)
    {
        if (shape[i] == "-1")
        {
            if (infer != -1)
            {
                fprintf(stderr, "reshape %s: target has multiple infer dimensions\n", op->name.c_str());
                return;
            }
            infer = i;
        }
        else
            known_dimensions.push_back(shape[i]);
    }
    if (infer != -1 && !input_shape.empty())
        shape[infer] = shape_quotient(shape_product(input_shape), shape_product(known_dimensions));

    const std::string input_n = partition_size(input_shape, plan.input_axis);
    const std::string output_n = partition_size(shape, plan.output_axis);
    const std::string input_suffix = partition_suffix(input_shape, plan.input_axis);
    const std::string output_suffix = partition_suffix(shape, plan.output_axis);
    if ((input_n == "1" && output_n == "1") || (input_n == output_n && input_suffix == output_suffix))
        plan.partition = PartitionSame;
    else if (dimension_differs(input_n, output_n) || (input_n == output_n && input_n != "1" && dimension_differs(input_suffix, output_suffix)))
        plan.partition = PartitionDifferent;

    // only the physical target needs to be evaluated by an ordinary reshape
    for (int i = 0; i < (int)shape.size(); i++)
    {
        if (i == plan.output_axis)
            continue;
        for (size_t j : dimension_references(shape[i]))
        {
            if (shape[i][j + 1] == 'n' || shape[i][j] != '0')
                plan.full_context = true;
        }
    }

    const bool explicit_batch = (plan.partition != PartitionSame || plan.full_context) && (plan.input_axis != 233 || plan.output_axis != 233);
    if (!explicit_batch && plan.output_axis != 233)
        shape.erase(shape.begin() + plan.output_axis);

    if (shape.empty() || shape.size() > (explicit_batch && plan.output_axis != 233 ? 5u : 4u))
    {
        fprintf(stderr, "reshape %s: target exceeds ncnn physical rank\n", op->name.c_str());
        return;
    }

    op->params.clear();
    bool static_shape = shape.size() <= 4;
    for (const auto& dim : shape)
    {
        long long value;
        if (!integer_dimension(dim, value))
            static_shape = false;
    }
    if (static_shape)
    {
        op->params["0"] = std::stoi(shape.back());
        if (shape.size() >= 2)
            op->params["1"] = std::stoi(shape[shape.size() - 2]);
        if (shape.size() >= 3)
            op->params["2"] = std::stoi(shape[0]);
        if (shape.size() == 4)
            op->params["11"] = std::stoi(shape[1]);
    }
    else
    {
        std::string expression;
        for (auto it = shape.rbegin(); it != shape.rend(); ++it)
            expression += (expression.empty() ? "" : ",") + *it;
        op->params["6"] = expression;
    }
    if (explicit_batch)
    {
        op->params["12"] = plan.input_axis;
        op->params["13"] = plan.output_axis;
    }

    // retain only shape references used by the emitted target
    std::vector<int> indices(op->inputs.size(), -1);
    indices[0] = 0;
    std::string expression = static_shape ? std::string() : op->params["6"].s;
    const auto references = dimension_references(expression);
    for (size_t i : references)
        indices.at(expression[i] - '0') = 0;
    std::vector<Operand*> inputs;
    for (size_t i = 0; i < indices.size(); i++)
    {
        if (indices[i] == -1)
            op->inputs[i]->remove_consumer(op);
        else
        {
            indices[i] = (int)inputs.size();
            inputs.push_back(op->inputs[i]);
        }
    }
    for (size_t i : references)
        expression[i] = '0' + indices.at(expression[i] - '0');
    if (!static_shape)
        op->params["6"] = expression;
    op->inputs = inputs;
    op->inputnames.clear();
}

} // namespace ncnn

} // namespace pnnx
