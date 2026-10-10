// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_slice_squeeze_to_select.h"

#include "pass_level2.h"

namespace pnnx {

class fuse_slice_squeeze_to_select_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
Tensor.slice            op_0        1 1 input a dim=%dim start=%start end=%end step=1
torch.squeeze           op_1        1 1 a out dim=%squeeze_dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.select";
    }

    const char* name_str() const
    {
        return "select";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int start = captured_params.at("start").i;
        const int end = captured_params.at("end").i;
        if (end != start + 1)
            return false;

        const int dim = captured_params.at("dim").i;
        int squeeze_dim;
        if (captured_params.at("squeeze_dim").type == 2)
        {
            squeeze_dim = captured_params.at("squeeze_dim").i;
        }
        else // if (captured_params.at("squeeze_dim").type == 5)
        {
            if (captured_params.at("squeeze_dim").ai.size() != 1)
                return false;

            squeeze_dim = captured_params.at("squeeze_dim").ai[0];
        }
        if (squeeze_dim != dim)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["dim"] = captured_params.at("dim");
        op->params["index"] = captured_params.at("start");
    }
};

void fuse_slice_squeeze_to_select(Graph& graph)
{
    fuse_slice_squeeze_to_select_pass a;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
}

} // namespace pnnx
