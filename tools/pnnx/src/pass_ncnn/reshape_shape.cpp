// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape_shape.h"

#include <algorithm>
#include <cstdlib>
#include <cerrno>
#include <climits>

namespace pnnx {

namespace ncnn {

int get_ncnn_batch_axis(const Operand* operand)
{
    const auto it = operand->params.find("__ncnn_batch_axis");
    int axis = it == operand->params.end() ? 233 : it->second.i;
    if (axis < 0)
        axis += (int)operand->shape.size();
    return axis;
}

static std::string get_logical_dim_expr(int rank, int native_batch_axis, int reference_index, int axis)
{
    if (axis < 0)
        axis += rank;
    if (axis < 0 || axis >= rank || reference_index < 0 || reference_index > 9)
    {
        fprintf(stderr, "reshape has unsupported logical dimension %d or reference %d\n", axis, reference_index);
        return std::string();
    }

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
        fprintf(stderr, "reshape has unsupported physical rank %d\n", rank);
        return std::string();
    }
    return std::to_string(reference_index) + dimensions[rank][axis];
}

std::string get_logical_dim_expr(const Operand* operand, int reference_index, int axis)
{
    return get_logical_dim_expr((int)operand->shape.size(), get_ncnn_batch_axis(operand), reference_index, axis);
}

std::vector<std::string> get_logical_shape_expr(int rank, int native_batch_axis, int reference_index)
{
    std::vector<std::string> shape;
    for (int i = 0; i < rank; i++)
        shape.push_back(get_logical_dim_expr(rank, native_batch_axis, reference_index, i));
    return shape;
}

std::vector<std::string> get_logical_shape_expr(const Operand* operand, int reference_index)
{
    std::vector<std::string> shape;
    for (int i = 0; i < (int)operand->shape.size(); i++)
    {
        shape.push_back(operand->shape[i] > 0 ? std::to_string(operand->shape[i]) : get_logical_dim_expr(operand, reference_index, i));
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

// 1 = integer, 0 = expression, -1 = integer overflow
static int parse_integer_dimension(const std::string& expression, long long& value)
{
    errno = 0;
    char* end = 0;
    value = strtoll(expression.c_str(), &end, 10);
    if (end == expression.c_str() || *end != 0)
        return 0;
    return errno == ERANGE ? -1 : 1;
}

static bool product_factors(const std::string& expression, long long& constant, std::vector<std::string>& factors)
{
    long long value;
    const int integer = parse_integer_dimension(expression, value);
    if (integer < 0 || (integer == 1 && value > 0 && constant > LLONG_MAX / value))
    {
        fprintf(stderr, "reshape dimension product exceeds integer range\n");
        return false;
    }
    if (integer == 1 && value > 0)
    {
        constant *= value;
        return true;
    }
    if (expression.compare(0, 2, "*(") == 0 && expression.back() == ')')
    {
        const auto operands = split_shape_expression(expression.substr(2, expression.size() - 3));
        if (operands.size() == 2)
        {
            return product_factors(operands[0], constant, factors) && product_factors(operands[1], constant, factors);
        }
    }
    factors.push_back(expression);
    return true;
}

static std::string make_product(long long constant, std::vector<std::string> factors)
{
    std::sort(factors.begin(), factors.end());
    std::string expression = constant != 1 || factors.empty() ? std::to_string(constant) : std::string();
    for (const auto& factor : factors)
        expression = expression.empty() ? factor : "*(" + expression + "," + factor + ")";
    return expression;
}

std::string make_shape_product_expr(const std::vector<std::string>& dimensions)
{
    long long constant = 1;
    std::vector<std::string> factors;
    for (const auto& dim : dimensions)
    {
        if (dim.empty())
            return std::string();
        if (!product_factors(dim, constant, factors))
            return std::string();
    }
    return make_product(constant, factors);
}

static std::string make_shape_quotient_expr(const std::string& numerator, const std::string& denominator)
{
    if (numerator.empty() || denominator.empty())
        return std::string();

    long long nc = 1;
    long long dc = 1;
    std::vector<std::string> nf;
    std::vector<std::string> df;
    if (!product_factors(numerator, nc, nf) || !product_factors(denominator, dc, df))
        return std::string();
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

static std::vector<size_t> find_shape_expr_references(const std::string& expression)
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
    return axis == 233 ? "1" : make_shape_product_expr(std::vector<std::string>(shape.begin() + axis + 1, shape.end()));
}

bool resolve_reshape_params(const std::vector<std::string>& input_shape, int input_axis, std::vector<std::string> shape, int output_axis, int input_count, std::map<std::string, Parameter>& params)
{
    if (shape.empty())
    {
        fprintf(stderr, "reshape: target rank is unknown\n");
        return false;
    }

    if ((input_axis != 233 && (input_axis < 0 || input_axis >= (int)input_shape.size()))
            || (output_axis != 233 && (output_axis < 0 || output_axis >= (int)shape.size())))
    {
        fprintf(stderr, "reshape: batch axis is outside logical rank\n");
        return false;
    }
    if (std::find(input_shape.begin(), input_shape.end(), std::string()) != input_shape.end()
            || std::find(shape.begin(), shape.end(), std::string()) != shape.end())
    {
        fprintf(stderr, "reshape: unsupported dimension reference\n");
        return false;
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
                fprintf(stderr, "reshape: target has multiple infer dimensions\n");
                return false;
            }
            infer = i;
        }
        else
            known_dimensions.push_back(shape[i]);
    }
    if (infer != -1 && !input_shape.empty())
    {
        const std::string input_total = make_shape_product_expr(input_shape);
        const std::string known_total = make_shape_product_expr(known_dimensions);
        shape[infer] = make_shape_quotient_expr(input_total, known_total);
        if (shape[infer].empty())
            return false;
    }

    const std::string input_n = partition_size(input_shape, input_axis);
    const std::string output_n = partition_size(shape, output_axis);
    const std::string input_suffix = partition_suffix(input_shape, input_axis);
    const std::string output_suffix = partition_suffix(shape, output_axis);
    if (input_suffix.empty() || output_suffix.empty())
        return false;
    const bool same_batch_partition = (input_n == "1" && output_n == "1") || (input_n == output_n && input_suffix == output_suffix);

    // only the physical target needs to be evaluated by an ordinary reshape
    bool requires_batch_context = false;
    for (int i = 0; i < (int)shape.size(); i++)
    {
        if (i == output_axis)
            continue;
        for (size_t j : find_shape_expr_references(shape[i]))
        {
            if (shape[i][j + 1] == 'n' || shape[i][j] != '0')
                requires_batch_context = true;
        }
    }

    const bool explicit_batch = (!same_batch_partition || requires_batch_context) && (input_axis != 233 || output_axis != 233);
    if (!explicit_batch && output_axis != 233)
        shape.erase(shape.begin() + output_axis);

    if (shape.empty() || shape.size() > (explicit_batch && output_axis != 233 ? 5u : 4u))
    {
        fprintf(stderr, "reshape: target exceeds ncnn physical rank\n");
        return false;
    }

    bool static_shape = shape.size() <= 4;
    std::vector<int> dimensions;
    for (const auto& dim : shape)
    {
        long long value;
        const int integer = parse_integer_dimension(dim, value);
        if (integer < 0 || (integer == 1 && (value < -1 || value == 0 || value > INT_MAX)))
        {
            fprintf(stderr, "reshape: dimension exceeds ncnn integer range\n");
            return false;
        }
        if (integer == 1)
        {
            dimensions.push_back((int)value);
            continue;
        }
        static_shape = false;

        // constants inside a dynamic expression also use ncnn integer dimensions
        size_t start = 0;
        for (size_t i = 0; i <= dim.size(); i++)
        {
            if (i != dim.size() && dim[i] != '(' && dim[i] != ')' && dim[i] != ',')
                continue;
            long long constant;
            const int integer = parse_integer_dimension(dim.substr(start, i - start), constant);
            if (integer < 0 || (integer == 1 && (constant < INT_MIN || constant > INT_MAX)))
            {
                fprintf(stderr, "reshape: expression constant exceeds ncnn integer range\n");
                return false;
            }
            start = i + 1;
        }

        for (size_t i : find_shape_expr_references(dim))
        {
            if (dim[i] - '0' >= input_count)
            {
                fprintf(stderr, "reshape: missing shape reference\n");
                return false;
            }
        }
    }
    params.clear();
    if (static_shape)
    {
        params["0"] = dimensions.back();
        if (dimensions.size() >= 2)
            params["1"] = dimensions[dimensions.size() - 2];
        if (dimensions.size() >= 3)
            params["2"] = dimensions[0];
        if (dimensions.size() == 4)
            params["11"] = dimensions[1];
    }
    else
    {
        std::string expression;
        for (auto it = shape.rbegin(); it != shape.rend(); ++it)
            expression += (expression.empty() ? "" : ",") + *it;
        params["6"] = expression;
    }
    if (explicit_batch)
    {
        params["12"] = input_axis;
        params["13"] = output_axis;
    }

    return true;
}

bool resolve_reshape_params(const Operator* op, const std::vector<std::string>& shape, std::map<std::string, Parameter>& params)
{
    const auto input_shape = get_logical_shape_expr(op->inputs[0], 0);
    const int input_axis = get_ncnn_batch_axis(op->inputs[0]);
    const int output_axis = get_ncnn_batch_axis(op->outputs[0]);
    const int input_count = (int)op->inputs.size();
    return resolve_reshape_params(input_shape, input_axis, shape, output_axis, input_count, params);
}

void write_reshape_params(Operator* op, const std::map<std::string, Parameter>& params)
{
    op->params = params;

    // retain only shape references used by the emitted target
    std::vector<int> indices(op->inputs.size(), -1);
    indices[0] = 0;
    std::string expression = op->has_param("6") ? op->params.at("6").s : std::string();
    const auto references = find_shape_expr_references(expression);
    for (size_t i : references)
        indices[expression[i] - '0'] = 0;
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
        expression[i] = '0' + indices[expression[i] - '0'];
    if (!expression.empty())
        op->params["6"] = expression;
    op->inputs = inputs;
    op->inputnames.clear();
}

} // namespace ncnn

} // namespace pnnx
