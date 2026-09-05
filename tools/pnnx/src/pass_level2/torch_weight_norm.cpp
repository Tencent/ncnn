// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "torch_weight_norm.h"

#include <ATen/ATen.h>

#include <exception>

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

    // Other g shapes may broadcast along a different axis or expand the result.
    if (whole_tensor)
    {
        if (g.shape.size() > v.shape.size())
            return false;
    }
    else
    {
        if (g.shape.size() != v.shape.size())
            return false;
        for (int i = 0; i < ndim; i++)
        {
            if (g.shape[i] != (i == dim ? v.shape[i] : 1))
                return false;
        }
    }

    if (!whole_tensor)
    {
        for (int i = dim + 1; i < ndim; i++)
        {
            if (v.shape[i] <= 0)
                return false;
        }
    }

    std::vector<float> v_data = v.get_float32_data();
    std::vector<float> g_data = g.get_float32_data();
    std::vector<float> weight_data;
    try
    {
        // Use ATen's reduction order and float32 intermediate rounding.
        const std::vector<int64_t> v_shape(v.shape.begin(), v.shape.end());
        const std::vector<int64_t> g_shape(g.shape.begin(), g.shape.end());
        const auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
        const at::Tensor vt = at::from_blob(v_data.data(), v_shape, options);
        const at::Tensor gt = at::from_blob(g_data.data(), g_shape, options);
        const at::Tensor weight = at::_weight_norm(vt, gt, dim_param.i).contiguous();
        if (weight.numel() != 0)
        {
            const float* data = weight.data_ptr<float>();
            weight_data.assign(data, data + weight.numel());
        }
    }
    catch (const std::exception&)
    {
        return false;
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
