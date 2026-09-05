// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

#include <algorithm>
#include <math.h>
#include <string.h>

namespace pnnx {

// pt2 path: the parametrization of nn.utils.weight_norm is expanded by dynamo
// into weight = v * (g / ||v||_2) with the norm over all coords except dim
// (aten::_weight_norm semantics, linalg_vector_norm + div + mul).
// v / g are constants (pnnx.Attribute); fold them into a constant weight
// Attribute so downstream Conv1d/2d/3d / Linear use the folded weight directly.
class torch_weight_norm_fold : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              v           0 1 v
prim::Constant          op_ord      0 1 ord value=%ord
prim::Constant          op_dim      0 1 dim value=%dim
prim::Constant          op_kd       0 1 keepdim value=%keepdim
aten::linalg_vector_norm norm       4 1 v ord dim keepdim norm_out
pnnx.Input              g           0 1 g
aten::div               div         2 1 g norm_out div_out
aten::mul               mul         2 1 v div_out out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              v           0 1 v
pnnx.Input              g           0 1 g
pnnx.Attribute          out         0 1 out @data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "pnnx.Attribute";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        // v and g must be constant parameters (pnnx.Attribute); extract the
        // data now since the matched nodes are removed before write()
        const Operator* v_node = matched_operators.at("mul")->inputs[0]->producer;
        const Operator* g_node = matched_operators.at("div")->inputs[0]->producer;
        if (!v_node || v_node->type != "pnnx.Attribute" || v_node->attrs.find("data") == v_node->attrs.end())
            return false;
        if (!g_node || g_node->type != "pnnx.Attribute" || g_node->attrs.find("data") == g_node->attrs.end())
            return false;

        m_v = v_node->attrs.at("data");
        m_g = g_node->attrs.at("data");

        // reset per-match state (a GraphRewriterPass instance is reused across matches)
        m_dim = 0;
        m_use_reduce_list = false;
        m_reduce_dims.clear();

        // only support ord=2 and keepdim=True L2 normalization (weight_norm standard form)
        int ord = -1;
        bool keepdim = false;
        bool has_keepdim = false;
        for (const auto& x : captured_params)
        {
            std::string key = x.first;
            size_t dot = key.rfind('.');
            if (dot != std::string::npos)
                key = key.substr(dot + 1);
            if (key == "ord" && x.second.type == 2)
                ord = x.second.i;
            // torch 2.13 exports weight_norm with a scalar linalg_vector_norm dim
            // that is already the preserved weight_norm axis, so use it directly.
            // A list-typed dim (the axes actually reduced by linalg_vector_norm)
            // is handled defensively in write() by deriving the complement.
            if (key == "dim")
            {
                if (x.second.type == 2) // scalar int = preserved axis
                {
                    m_dim = x.second.i;
                }
                else if (x.second.type == 5) // int list = reduced axes
                {
                    m_reduce_dims = x.second.ai;
                    m_use_reduce_list = true;
                }
            }
            if (key == "keepdim")
            {
                if (x.second.type == 1) // Parameter bool
                {
                    keepdim = x.second.b;
                    has_keepdim = true;
                }
                else if (x.second.type == 2)
                {
                    keepdim = x.second.i != 0;
                    has_keepdim = true;
                }
            }
        }

        if (ord != 2)
            return false;
        if (!has_keepdim || !keepdim)
            return false;
        // only fold float-family weights that get/set_float32_data can round-trip
        // (f32/f64/f16); a non-float v would leave the replacement pnnx.Attribute
        // without usable weight data (write() would bail out after the graph was
        // already rewritten), and bf16 etc. are not supported by those helpers.
        if (m_v.type != 1 && m_v.type != 2 && m_v.type != 3)
            return false;
        if (m_g.type != 1 && m_g.type != 2 && m_g.type != 3)
            return false;
        // a scalar v cannot be folded (no per-coord norm); reject before rewrite
        if (m_v.shape.empty())
            return false;
        // a scalar (dim=-1) f16 g cannot be decoded without a half helper,
        // and the scalar decoders need full element bytes
        if (m_g.shape.empty() && m_g.type == 3)
            return false;
        if (m_g.shape.empty() && m_g.type == 1 && m_g.data.size() < 4)
            return false;
        if (m_g.shape.empty() && m_g.type == 2 && m_g.data.size() < 8)
            return false;
        // fold would overflow the coord[16] stack buffer for >16-d weights;
        // reject before the graph is rewritten
        if ((int)m_v.shape.size() > 16)
            return false;

        // reduce-list form: the axes reduced by linalg_vector_norm leave the
        // preserved weight_norm axis as their complement. reject when more than
        // one axis is preserved (write() cannot fold those and would bail after
        // the graph was already rewritten, leaving an empty pnnx.Attribute).
        // zero preserved axes is the valid dim=None all-axis case, which folds
        // below with a scalar norm.
        if (m_use_reduce_list)
        {
            int preserved = 0;
            const int vdims = (int)m_v.shape.size();
            for (int d = 0; d < vdims; d++)
            {
                if (std::find(m_reduce_dims.begin(), m_reduce_dims.end(), d) == m_reduce_dims.end())
                    preserved++;
            }
            if (preserved > 1)
                return false;
        }
        return true;
    }

    mutable Attribute m_v;
    mutable Attribute m_g;
    mutable int m_dim = 0;
    mutable bool m_use_reduce_list = false;
    mutable std::vector<int> m_reduce_dims;

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        Operator* op = ops.at("out");

        const Attribute& v_attr = m_v;
        const Attribute& g_attr = m_g;

        const std::vector<int>& v_shape = v_attr.shape;
        if (v_shape.empty() || (v_attr.type != 1 && v_attr.type != 2 && v_attr.type != 3))
            return; // f32/f64/f16 weights only

        int dims = (int)v_shape.size();
        if (dims > 16)
            return; // coord[16] below would overflow the stack buffer

        // the preserved weight_norm axis:
        // - scalar dim (torch 2.13): already the preserved axis
        // - list dim (defensive): linalg reduced axes; the preserved axis is the
        //   single axis NOT in the reduced list (or -1 when every axis reduced)
        int dim = m_dim;
        if (m_use_reduce_list)
        {
            for (auto& d : m_reduce_dims)
                if (d < 0)
                    d += dims; // normalize negative axis

            // the preserved weight_norm axis is the complement of the reduced
            // axes; zero preserved axes (dim=None, every axis reduced by
            // linalg_vector_norm) is valid and folds with a scalar norm over
            // all coords, keeping dim = -1 below
            dim = -1;
            int preserved = 0;
            for (int d = 0; d < dims; d++)
            {
                if (std::find(m_reduce_dims.begin(), m_reduce_dims.end(), d) == m_reduce_dims.end())
                {
                    preserved++;
                    dim = d;
                }
            }
            if (preserved > 1)
                return; // more than one preserved axis (also rejected in match())
            if (preserved == 0)
                dim = -1; // scalar norm over every axis
        }

        if (dim < -1 || dim >= dims)
            return;

        // extract v as f32 (Attribute helpers handle f32/f64/f16 internally)
        std::vector<float> vv = v_attr.get_float32_data();
        if (vv.empty())
            return;
        const float* vp = vv.data();
        const int v_count = (int)vv.size();

        // g broadcasts over the dim axis; scalar g (dim=-1 case) is also supported
        const std::vector<int>& g_shape = g_attr.shape;
        bool g_is_scalar = g_shape.empty();
        if (!g_is_scalar && (int)g_shape.size() != dims)
            return;
        int g_count = 1;
        for (int s : g_shape)
            g_count *= s;
        if (g_count <= 0)
            return;

        std::vector<float> gv;
        if (g_is_scalar)
        {
            // scalar g (weight_norm dim=-1): decode the single element by dtype
            gv.resize(1);
            if (g_attr.type == 1)
                gv[0] = *(const float*)g_attr.data.data();
            else if (g_attr.type == 2)
                gv[0] = (float)*(const double*)g_attr.data.data();
            else
                return;
        }
        else
        {
            gv = g_attr.get_float32_data();
            if ((int)gv.size() != g_count)
                return;
        }
        const float* gp = gv.data();

        // aten::_weight_norm(v, g, dim): weight = v * (g / ||v||_2) where the
        // L2 norm is taken over all coords except dim (norm_except_dim).
        // dim=-1 means the norm over all coords (scalar).
        int norm_count = (dim == -1) ? 1 : v_shape[dim];

        // per-coord L2 norm over all non-dim coords
        std::vector<float> norm_flat(norm_count, 0.f);
        for (int idx = 0; idx < v_count; idx++)
        {
            int rem = idx;
            int coord[16];
            for (int dd = dims - 1; dd >= 0; dd--)
            {
                coord[dd] = rem % v_shape[dd];
                rem /= v_shape[dd];
            }
            int norm_idx = (dim == -1) ? 0 : coord[dim];
            norm_flat[norm_idx] += vp[idx] * vp[idx];
        }
        for (int c = 0; c < norm_count; c++)
            norm_flat[c] = sqrtf(norm_flat[c]);

        // weight = v * (g / ||v||)
        std::vector<float> weight_flat(v_count);
        for (int idx = 0; idx < v_count; idx++)
        {
            // decompose idx into v coordinates
            int rem = idx;
            int coord[16];
            for (int dd = dims - 1; dd >= 0; dd--)
            {
                coord[dd] = rem % v_shape[dd];
                rem /= v_shape[dd];
            }

            int norm_idx = (dim == -1) ? 0 : coord[dim];

            // g index (broadcast)
            int g_idx = 0;
            if (g_is_scalar)
            {
                g_idx = 0;
            }
            else
            {
                for (int dd = 0; dd < dims; dd++)
                {
                    int gc = (g_shape[dd] == 1) ? 0 : coord[dd];
                    g_idx = g_idx * g_shape[dd] + gc;
                }
            }

            weight_flat[idx] = vp[idx] * gp[g_idx] / norm_flat[norm_idx];
        }

        // write back in the original weight dtype (f32/f64/f16)
        Attribute a = v_attr; // keeps type/shape
        a.shape = v_shape;
        a.set_float32_data(weight_flat);
        op->attrs["data"] = a;

        op->outputs[0]->type = v_attr.type;
        op->outputs[0]->shape = v_shape;
        op->params.clear();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_weight_norm_fold, 150)

} // namespace pnnx
