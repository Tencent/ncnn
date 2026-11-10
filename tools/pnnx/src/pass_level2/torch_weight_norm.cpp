// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

#include <math.h>

namespace pnnx {

static float weight_norm_sum_squares(const float* v, size_t begin, size_t end, size_t inner_size, int groups)
{
    // Pairwise FP32 reduction retains small terms without changing the
    // overflow/underflow behavior of FP32 squares and squared norms.
    if (end - begin > 32)
    {
        const size_t mid = begin + (end - begin) / 2;
        return weight_norm_sum_squares(v, begin, mid, inner_size, groups)
               + weight_norm_sum_squares(v, mid, end, inner_size, groups);
    }

    float sum = 0.f;
    for (size_t i = begin; i < end; i++)
    {
        const float x = v[(i / inner_size) * inner_size * groups + i % inner_size];
        sum += x * x;
    }
    return sum;
}

static void fold_weight_norm(Attribute& weight, const Attribute& weight_g, int dim)
{
    std::vector<float> v = weight.get_float32_data();
    if (v.empty())
        return;

    const std::vector<float> g = weight_g.get_float32_data();

    int inner_size = 1;
    if (dim >= 0)
    {
        for (size_t i = dim + 1; i < weight.shape.size(); i++)
            inner_size *= weight.shape[i];
    }

    const int groups = dim == -1 ? 1 : weight.shape[dim];
    std::vector<float> norms(groups);
    const size_t reduce_size = v.size() / groups;
    for (int group = 0; group < groups; group++)
    {
        norms[group] = sqrtf(weight_norm_sum_squares(v.data() + group * inner_size, 0, reduce_size, inner_size, groups));
    }

    for (size_t i = 0; i < v.size(); i++)
    {
        const int group = dim == -1 ? 0 : (i / inner_size) % groups;
        v[i] = v[i] * (g[group] / norms[group]);
    }

    weight.set_float32_data(v);
}

class torch_weight_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 v
pnnx.Input              input_1     0 1 g
prim::Constant          op_0        0 1 dim value=%dim
aten::_weight_norm      op_1        3 1 v g dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "pnnx.Attribute";
    }

    const char* name_str() const
    {
        return "weight_norm";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* op = matched_operators.at("op_1");
        for (int i = 0; i < 2; i++)
        {
            if (op->inputs[i]->producer->type != "pnnx.Attribute" || !op->inputs[i]->producer->has_attr("data"))
                return false;
        }

        const Attribute& v = op->inputs[0]->producer->attrs.at("data");
        const Attribute& g = op->inputs[1]->producer->attrs.at("data");
        // Fold supported floating-point payloads using the float32 attribute helpers.
        if ((v.type != 1 && v.type != 2) || g.type != v.type)
            return false;

        const int dim = captured_params.at("dim").i;
        if (dim < -1 || dim >= (int)v.shape.size())
            return false;

        for (int size : v.shape)
        {
            if (size <= 0)
                return false;
        }

        if (dim == -1)
            return g.elemcount() == 1 && g.shape.size() <= v.shape.size();

        std::vector<int> g_shape(v.shape.size(), 1);
        g_shape[dim] = v.shape[dim];
        return g.shape == g_shape;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        Attribute weight = op->inputs[0]->producer->attrs.at("data");
        const Attribute& weight_g = op->inputs[1]->producer->attrs.at("data");
        const int dim = captured_params.at("dim").i;
        fold_weight_norm(weight, weight_g, dim);

        op->attrs["data"] = weight;
        for (Operand* input : op->inputs)
            input->remove_consumer(op);
        op->inputs.clear();
        op->inputnames.clear();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_weight_norm, 100)

} // namespace pnnx
