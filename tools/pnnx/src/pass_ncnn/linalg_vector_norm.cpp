// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "linalg_vector_norm.h"

#include <algorithm>
#include <cstdlib>

namespace pnnx {

namespace ncnn {

// weight_norm expansion produces aten::linalg_vector_norm (v, p, dim, keepdim, dtype),
// where p/dim/keepdim are often pnnx.Expression constant nodes (expr=0/2/True),
// so the pass_level2 torch.norm rewrite (expecting prim::Constant) fails to match
// and it remains. Resolve the constant params here in pass_ncnn, rewrite to
// torch.norm and let the following Reduction conversion handle it.
static bool get_const_int(Operand* in, int& val)
{
    if (!in->producer)
        return false;

    const Operator* p = in->producer;

    if (p->type == "pnnx.Expression")
    {
        const std::string& expr = p->params.at("expr").s;
        if (expr == "True")
        {
            val = 1;
            return true;
        }
        if (expr == "False")
        {
            val = 0;
            return true;
        }
        val = atoi(expr.c_str());
        return true;
    }

    if (p->type == "prim::Constant" || p->type == "pnnx.Constant")
    {
        if (p->params.find("value") == p->params.end())
            return false;

        const Parameter& par = p->params.at("value");
        if (par.type == 1)
        {
            val = par.b ? 1 : 0;
            return true;
        }
        if (par.type == 2)
        {
            val = par.i;
            return true;
        }
        if (par.type == 3)
        {
            val = (int)par.f;
            return true;
        }
    }

    return false;
}

static bool get_const_float(Operand* in, float& val)
{
    if (!in->producer)
        return false;

    const Operator* p = in->producer;

    if (p->type == "pnnx.Expression")
    {
        const std::string& expr = p->params.at("expr").s;
        val = (float)atof(expr.c_str());
        return true;
    }

    if (p->type == "prim::Constant" || p->type == "pnnx.Constant")
    {
        if (p->params.find("value") == p->params.end())
            return false;

        const Parameter& par = p->params.at("value");
        if (par.type == 1)
        {
            val = par.b ? 1.f : 0.f;
            return true;
        }
        if (par.type == 2)
        {
            val = (float)par.i;
            return true;
        }
        if (par.type == 3)
        {
            val = par.f;
            return true;
        }
    }

    return false;
}

void convert_aten_linalg_vector_norm(Graph& graph)
{
    for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
    {
        Operator* op = graph.ops[i];

        if (op->type != "aten::linalg_vector_norm")
            continue;

        if (op->inputs.size() < 4)
            continue;

        // inputs: v, p, dim, keepdim, (dtype)
        Operand* v = op->inputs[0];

        float p = 2.f;
        std::vector<int> dims;
        int keepdim = 1;

        if (!get_const_float(op->inputs[1], p))
            p = 2.f;

        get_const_int(op->inputs[3], keepdim);

        // parse dim (Expression expr could be "0" or "[0,1]")
        if (op->inputs[2]->producer && op->inputs[2]->producer->type == "pnnx.Expression")
        {
            const std::string& expr = op->inputs[2]->producer->params.at("expr").s;
            if (!expr.empty() && expr[0] == '[')
            {
                // array like [0,1]
                std::string s = expr.substr(1, expr.size() - 2);
                size_t pos = 0;
                while (pos < s.size())
                {
                    size_t comma = s.find(',', pos);
                    std::string tok = (comma == std::string::npos) ? s.substr(pos) : s.substr(pos, comma - pos);
                    dims.push_back(atoi(tok.c_str()));
                    if (comma == std::string::npos)
                        break;
                    pos = comma + 1;
                }
            }
            else
            {
                dims.push_back(atoi(expr.c_str()));
            }
        }
        else
        {
            int dim = 0;
            if (get_const_int(op->inputs[2], dim))
                dims.push_back(dim);
        }

        Operator* norm = graph.new_operator_before("torch.norm", op->name + "_norm", op);
        norm->inputs.push_back(v);
        v->consumers.push_back(norm);

        norm->params["p"] = p;
        norm->params["dim"] = dims;
        norm->params["keepdim"] = keepdim ? true : false;

        for (Operand* r : op->outputs)
        {
            r->producer = norm;
            norm->outputs.push_back(r);
        }

        for (Operand* in : op->inputs)
        {
            in->consumers.erase(std::find(in->consumers.begin(), in->consumers.end(), op));
        }
        op->inputs.clear();
        op->outputs.clear();

        graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op));
        delete op;
    }
}

} // namespace ncnn

} // namespace pnnx
