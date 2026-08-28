// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "torch_weight_norm.h"

#include <math.h>

#include "pass_level2.h"

namespace pnnx {

static bool fold_static_weight_norm(Operator* op)
{
    if (op->type != "aten::_weight_norm")
        return false;

    if (op->inputs.size() != 3 || op->outputs.size() != 1)
        return false;

    const Operator* v_op = op->inputs[0]->producer;
    const Operator* g_op = op->inputs[1]->producer;
    const Operator* dim_op = op->inputs[2]->producer;
    if (v_op->type != "pnnx.Attribute" || g_op->type != "pnnx.Attribute" || dim_op->type != "prim::Constant")
        return false;

    if (!v_op->has_attr("data") || !g_op->has_attr("data") || !dim_op->has_param("value"))
        return false;

    const Attribute& v = v_op->attrs.at("data");
    const Attribute& g = g_op->attrs.at("data");
    const Parameter& dim_param = dim_op->params.at("value");
    if (v.type != 1 || g.type != 1 || dim_param.type != 2)
        return false;

    if (v.shape.empty() || v.data.size() != (size_t)v.elemcount() * sizeof(float) || g.data.size() != (size_t)g.elemcount() * sizeof(float))
        return false;

    const int ndim = (int)v.shape.size();
    int dim = dim_param.i;
    const bool whole_tensor = dim == -1;
    if (!whole_tensor && dim < 0)
        dim += ndim;
    if (!whole_tensor && (dim < 0 || dim >= ndim))
        return false;

    const int norm_count = whole_tensor ? 1 : v.shape[dim];
    if (norm_count <= 0 || g.elemcount() != norm_count)
        return false;

    size_t dim_stride = 1;
    if (!whole_tensor)
    {
        for (int i = dim + 1; i < ndim; i++)
        {
            if (v.shape[i] <= 0)
                return false;
            dim_stride *= (size_t)v.shape[i];
        }
    }

    const std::vector<float> v_data = v.get_float32_data();
    const std::vector<float> g_data = g.get_float32_data();
    std::vector<double> norms(norm_count, 0.0);
    for (size_t i = 0; i < v_data.size(); i++)
    {
        const int norm_index = whole_tensor ? 0 : (int)((i / dim_stride) % (size_t)norm_count);
        const double value = v_data[i];
        norms[norm_index] += value * value;
    }
    for (int i = 0; i < norm_count; i++)
        norms[i] = sqrt(norms[i]);

    std::vector<float> weight_data(v_data.size());
    for (size_t i = 0; i < v_data.size(); i++)
    {
        const int norm_index = whole_tensor ? 0 : (int)((i / dim_stride) % (size_t)norm_count);
        weight_data[i] = (float)(v_data[i] * g_data[norm_index] / norms[norm_index]);
    }

    for (size_t i = 0; i < op->inputs.size(); i++)
        op->inputs[i]->remove_consumer(op);
    op->inputs.clear();
    op->inputnames.clear();
    op->params.clear();
    op->attrs.clear();

    op->type = "pnnx.Attribute";
    op->attrs["data"] = v;
    op->attrs["data"].set_float32_data(weight_data);
    return true;
}

void fold_static_weight_norm(Graph& graph)
{
    for (size_t i = 0; i < graph.ops.size(); i++)
        fold_static_weight_norm(graph.ops[i]);
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
        return "torch._weight_norm";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_weight_norm, 70)

} // namespace pnnx
