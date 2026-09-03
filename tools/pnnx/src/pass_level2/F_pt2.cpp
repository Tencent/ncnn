// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// PT2-specific normalization passes.

#include "pass_level2.h"

#include "utils.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

namespace pnnx {

// Fold the value-independent ones_like + scalar subgraph without libtorch.
class F_pt2_fold_ones_like : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
torch.ones_like         op_0        1 1 input ones_out dtype=%ones_dtype
prim::Constant          op_c        0 1 other value=%other
prim::Constant          op_a        0 1 alpha value=%alpha
aten::add               op_1        3 1 ones_out other alpha out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "pnnx.Attribute";
    }

    const char* name_str() const
    {
        return "pnnx_fold";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators,
               const std::map<std::string, Parameter>& captured_params,
               const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Parameter& other = captured_params.at("other");
        const Parameter& alpha = captured_params.at("alpha");
        if (other.type != 2 && other.type != 3)
            return false;
        if (alpha.type != 2 && alpha.type != 3)
            return false;

        const Operator* add = matched_operators.at("op_1");
        if (add->outputs.empty())
            return false;

        const Operand* out = add->outputs[0];
        if (out->type != 1 || out->shape.empty())
            return false;

        size_t elem_count = 1;
        for (size_t i = 0; i < out->shape.size(); i++)
        {
            if (out->shape[i] <= 0 || elem_count > (size_t)-1 / (size_t)out->shape[i])
                return false;
            elem_count *= (size_t)out->shape[i];
        }
        if (elem_count > (size_t)-1 / sizeof(float))
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params,
               const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Parameter& other = captured_params.at("other");
        const Parameter& alpha = captured_params.at("alpha");

        const float scalar_other = (other.type == 2) ? (float)other.i : other.f;
        const float scalar_alpha = (alpha.type == 2) ? (float)alpha.i : alpha.f;
        const float folded_value = 1.f + scalar_alpha * scalar_other;

        const Operand* out = op->outputs[0];

        Attribute attr;
        attr.type = 1;
        attr.shape = out->shape;

        size_t elem_count = 1;
        for (size_t i = 0; i < attr.shape.size(); i++)
            elem_count *= (size_t)attr.shape[i];

        attr.data.resize(elem_count * sizeof(float));
        float* p = (float*)attr.data.data();
        for (size_t i = 0; i < elem_count; i++)
            p[i] = folded_value;

        op->attrs["data"] = attr;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_fold_ones_like, 90)

// Walk weight norm imperatively because shared parameters defeat pattern matching.
void fold_pt2_weight_norm(Graph& pg)
{
    for (size_t i = 0; i < pg.ops.size(); i++)
    {
        Operator* op = pg.ops[i];
        if (op->type != "aten::_weight_norm")
            continue;

        if (op->inputs.size() != 3 || op->outputs.size() != 1)
            continue;

        Operand* r_v = op->inputs[0];
        Operand* r_g = op->inputs[1];
        Operand* r_dim = op->inputs[2];
        if (!r_v->producer || r_v->producer->type != "pnnx.Attribute"
            || !r_g->producer || r_g->producer->type != "pnnx.Attribute")
            continue;

        float dim_value = -1.f;
        if (r_dim->producer && r_dim->producer->type == "prim::Constant")
        {
            const Parameter& dim = r_dim->producer->params.at("value");
            dim_value = (dim.type == 2) ? (float)dim.i : ((dim.type == 3) ? dim.f : -1.f);
        }
        else if (r_dim->producer && r_dim->producer->type == "pnnx.Expression")
        {
            const Parameter& expr = r_dim->producer->params.at("expr");
            if (expr.type != 4)
                continue;
            bool all_numeric = true;
            for (size_t j = 0; j < expr.s.size(); j++)
            {
                const char c = expr.s[j];
                if ((c < '0' || c > '9') && c != '.')
                {
                    all_numeric = false;
                    break;
                }
            }
            if (!all_numeric || atof(expr.s.c_str()) != 0.f)
                continue;
            dim_value = 0.f;
        }
        if (dim_value != 0.f)
            continue;

        const Attribute& attr_v = r_v->producer->attrs.at("data");
        const Attribute& attr_g = r_g->producer->attrs.at("data");
        if (attr_v.type != 1 || attr_g.type != 1)
            continue;

        if (attr_v.shape.empty())
            continue;

        const int dim0 = attr_v.shape[0];
        if (attr_g.get_float32_data().size() != (size_t)dim0)
            continue;

        std::vector<float> weight = attr_v.get_float32_data();
        const std::vector<float>& weight_g = attr_g.get_float32_data();

        const int size = (int)(weight.size() / dim0);

        apply_weight_norm(weight, weight_g, dim0, size);

        op->type = "pnnx.Attribute";
        op->params.clear();
        for (size_t j = 0; j < op->inputs.size(); j++)
        {
            op->inputs[j]->remove_consumer(op);
        }
        op->inputs.clear();
        op->attrs.clear();
        op->attrs["data"] = Attribute();
        op->attrs["data"].type = attr_v.type;
        op->attrs["data"].shape = attr_v.shape;
        op->attrs["data"].data.resize(weight.size() * sizeof(float));
        memcpy(op->attrs["data"].data.data(), weight.data(), weight.size() * sizeof(float));
    }
}

// Restore adaptive-pool None dimensions materialized by torch.export.
class F_pt2_adaptive_pool_base : public GraphRewriterPass
{
public:
    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const std::string marker_key = "op_0.__pt2_none_axes";
        if (captured_params.find(marker_key) == captured_params.end()
            || captured_params.at(marker_key).type != 4)
            return false;
        const std::string& none_axes = captured_params.at(marker_key).s;
        const Parameter& osz = captured_params.at("output_size");
        if (osz.type != 5)
            return false;

        const std::vector<int>& ishape = matched_operators.at("op_0")->inputs[0]->shape;
        if (ishape.empty())
            return false;

        const int k = (int)osz.ai.size();
        for (int i = 0; i < k; i++)
        {
            if (osz.ai[i] == 0)
                return false;

            const int dim_index = (int)ishape.size() - k + i;
            if (i < (int)none_axes.size() && none_axes[i] == '1' && dim_index >= 0
                && dim_index < (int)ishape.size() && osz.ai[i] == ishape[dim_index])
                return true;
        }

        return false;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        Parameter osz = captured_params.at("output_size");
        const std::vector<int>& ishape = ops.at("op_0")->inputs[0]->shape;
        const std::string& none_axes = captured_params.at("op_0.__pt2_none_axes").s;

        if (!ishape.empty())
        {
            const int k = (int)osz.ai.size();
            for (int i = 0; i < k; i++)
            {
                const int dim_index = (int)ishape.size() - k + i;
                if (i < (int)none_axes.size() && none_axes[i] == '1' && dim_index >= 0
                    && dim_index < (int)ishape.size() && osz.ai[i] == ishape[dim_index])
                {
                    osz.ai[i] = 0;
                }
            }
        }

        ops.at("op_sz")->params["value"] = osz;
    }
};

class F_pt2_adaptive_avg_pool1d : public F_pt2_adaptive_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=%output_size
aten::adaptive_avg_pool1d op_0      2 1 input output_size out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=(0)
aten::adaptive_avg_pool1d op_0      2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "aten::adaptive_avg_pool1d";
    }
};

class F_pt2_adaptive_avg_pool2d : public F_pt2_adaptive_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=%output_size
aten::adaptive_avg_pool2d op_0      2 1 input output_size out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=(0)
aten::adaptive_avg_pool2d op_0      2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "aten::adaptive_avg_pool2d";
    }
};

class F_pt2_adaptive_avg_pool3d : public F_pt2_adaptive_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=%output_size
aten::adaptive_avg_pool3d op_0      2 1 input output_size out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=(0)
aten::adaptive_avg_pool3d op_0      2 1 input output_size out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "aten::adaptive_avg_pool3d";
    }
};

class F_pt2_adaptive_max_pool1d : public F_pt2_adaptive_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=%output_size
aten::adaptive_max_pool1d op_0      2 2 input output_size out indices %*=%*
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=(0)
aten::adaptive_max_pool1d op_0      2 2 input output_size out indices
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "aten::adaptive_max_pool1d";
    }
};

class F_pt2_adaptive_max_pool2d : public F_pt2_adaptive_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=%output_size
aten::adaptive_max_pool2d op_0      2 2 input output_size out indices %*=%*
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=(0)
aten::adaptive_max_pool2d op_0      2 2 input output_size out indices
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "aten::adaptive_max_pool2d";
    }
};

class F_pt2_adaptive_max_pool3d : public F_pt2_adaptive_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=%output_size
aten::adaptive_max_pool3d op_0      2 2 input output_size out indices %*=%*
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 input
prim::Constant          op_sz       0 1 output_size value=(0)
aten::adaptive_max_pool3d op_0      2 2 input output_size out indices
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "aten::adaptive_max_pool3d";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_adaptive_avg_pool1d, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_adaptive_avg_pool2d, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_adaptive_avg_pool3d, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_adaptive_max_pool1d, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_adaptive_max_pool2d, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_adaptive_max_pool3d, 110)

// Restore materialized None dimensions in module-form adaptive pooling.
class F_pt2_nn_adaptive_avg_pool_base : public GraphRewriterPass
{
public:
    bool match(const std::map<std::string, const Operator*>& matched_operators,
               const std::map<std::string, Parameter>& captured_params,
               const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        std::map<std::string, Parameter>::const_iterator it = captured_params.find("op_0.output_size");
        if (it == captured_params.end() || it->second.type != 5)
            return false;
        const std::string marker_key = "op_0.__pt2_none_axes";
        if (captured_params.find(marker_key) == captured_params.end()
            || captured_params.at(marker_key).type != 4)
            return false;
        const std::string& none_axes = captured_params.at(marker_key).s;

        const std::vector<int>& ishape = matched_operators.at("op_0")->inputs[0]->shape;
        if (ishape.empty())
            return false;

        const std::vector<int>& ai = it->second.ai;
        const int k = (int)ai.size();
        for (int i = 0; i < k; i++)
        {
            if (ai[i] == 0)
                return false;

            const int dim_index = (int)ishape.size() - k + i;
            if (i < (int)none_axes.size() && none_axes[i] == '1' && dim_index >= 0
                && dim_index < (int)ishape.size() && ai[i] == ishape[dim_index])
                return true;
        }

        return false;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params,
               const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        Parameter osz = captured_params.at("op_0.output_size");
        const std::vector<int>& ishape = ops.at("op_0")->inputs[0]->shape;
        const std::string& none_axes = captured_params.at("op_0.__pt2_none_axes").s;

        if (!ishape.empty())
        {
            const int k = (int)osz.ai.size();
            for (int i = 0; i < k; i++)
            {
                const int dim_index = (int)ishape.size() - k + i;
                if (i < (int)none_axes.size() && none_axes[i] == '1' && dim_index >= 0
                    && dim_index < (int)ishape.size() && osz.ai[i] == ishape[dim_index])
                {
                    osz.ai[i] = 0;
                }
            }
        }

        ops.at("op_0")->params["output_size"] = osz;
    }
};

class F_pt2_nn_adaptive_avg_pool1d : public F_pt2_nn_adaptive_avg_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
nn.AdaptiveAvgPool1d    op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
nn.AdaptiveAvgPool1d    op_0        1 1 input out output_size=(0)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.AdaptiveAvgPool1d";
    }
};

class F_pt2_nn_adaptive_avg_pool2d : public F_pt2_nn_adaptive_avg_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
nn.AdaptiveAvgPool2d    op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
nn.AdaptiveAvgPool2d    op_0        1 1 input out output_size=(0)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.AdaptiveAvgPool2d";
    }
};

class F_pt2_nn_adaptive_avg_pool3d : public F_pt2_nn_adaptive_avg_pool_base
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
nn.AdaptiveAvgPool3d    op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
nn.AdaptiveAvgPool3d    op_0        1 1 input out output_size=(0)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.AdaptiveAvgPool3d";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_nn_adaptive_avg_pool1d, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_nn_adaptive_avg_pool2d, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_nn_adaptive_avg_pool3d, 110)

// Restore the LocalResponseNorm decomposition emitted by torch.export.
class F_pt2_local_response_norm_base : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
18 17
pnnx.Input              input       0 1 input
prim::Constant          op_shape1   0 1 shape1 value=%shape1
aten::mul               op_0        2 1 input input sq
Tensor.reshape          op_1        2 1 sq shape1 r1
F.pad                   op_2        1 1 r1 r2 mode=constant pad=(0,0,0,0,%pad_left,%pad_right) value=%padzero
F.avg_pool3d            op_3        1 1 r2 r3 ceil_mode=False count_include_pad=True divisor_override=None kernel_size=(%size,1,1) padding=(0,0,0) stride=(1,1,1)
torch.squeeze           op_4        1 1 r3 r4 dim=1
prim::Constant          op_shape2   0 1 shape2 value=%shape2
Tensor.reshape          op_5        2 1 r4 shape2 r5
prim::Constant          op_alpha    0 1 alpha value=%alpha
aten::mul               op_6        2 1 r5 alpha r6
prim::Constant          op_k        0 1 k value=%k
prim::Constant          op_one      0 1 one value=1
aten::add               op_7        3 1 r6 k one r7
prim::Constant          op_beta     0 1 beta value=%beta
aten::pow               op_8        2 1 r7 beta r8
aten::div               op_9        2 1 input r8 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const Parameter& padzero = captured_params.at("padzero");
        if (padzero.type == 0)
        {
            // None
        }
        else if (padzero.type == 2)
        {
            if (padzero.i != 0)
                return false;
        }
        else if (padzero.type == 3)
        {
            if (padzero.f != 0.f)
                return false;
        }
        else
        {
            return false;
        }

        const Parameter& pad_left = captured_params.at("pad_left");
        const Parameter& pad_right = captured_params.at("pad_right");
        if (pad_left.type != 2 || pad_right.type != 2)
            return false;
        if (pad_left.i + pad_right.i + 1 != captured_params.at("size").i)
            return false;

        const Parameter& xs = captured_params.at("__input_shape__");
        const Parameter& shape1 = captured_params.at("shape1");
        const Parameter& shape2 = captured_params.at("shape2");
        if (xs.type != 5 || shape1.type != 5 || shape2.type != 5)
            return false;
        if (xs.ai.size() != 4 || shape1.ai.size() != 5 || shape2.ai.size() != 4)
            return false;

        static const int map1[5] = {0, -1, 1, 2, 3};
        for (int i = 0; i < 5; i++)
        {
            const int want = (map1[i] < 0) ? 1 : xs.ai[map1[i]];
            if (shape1.ai[i] != want && shape1.ai[i] != -1)
                return false;
        }
        for (int i = 0; i < 4; i++)
        {
            if (shape2.ai[i] != xs.ai[i] && shape2.ai[i] != -1)
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["size"] = captured_params.at("size");
        op->params["alpha"] = captured_params.at("alpha");
        op->params["beta"] = captured_params.at("beta");
        op->params["k"] = captured_params.at("k");
    }

protected:
    virtual bool pad_is_symmetric() const = 0;

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const Parameter& pad_left = captured_params.at("pad_left");
        const Parameter& pad_right = captured_params.at("pad_right");
        if (pad_is_symmetric() && pad_left.type == 2 && pad_right.type == 2 && pad_left.i != pad_right.i)
            return false;

        const Operator* op_div = matched_operators.at("op_9");
        if (op_div->inputs[0]->shape.size() != 4)
            return false;

        std::map<std::string, Parameter> tmp = captured_params;
        tmp["__input_shape__"] = op_div->inputs[0]->shape;
        return match(tmp);
    }
};

class F_pt2_local_response_norm : public F_pt2_local_response_norm_base
{
public:
    const char* type_str() const
    {
        return "nn.LocalResponseNorm";
    }

protected:
    bool pad_is_symmetric() const
    {
        return true;
    }
};

class F_pt2_F_local_response_norm : public F_pt2_local_response_norm_base
{
public:
    const char* type_str() const
    {
        return "F.local_response_norm";
    }

protected:
    bool pad_is_symmetric() const
    {
        return false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_local_response_norm, 130)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pt2_F_local_response_norm, 131)

} // namespace pnnx
