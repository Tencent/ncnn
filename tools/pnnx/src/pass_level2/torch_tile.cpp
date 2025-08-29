// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_tile : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dims
aten::tile              op_0        2 1 input dims out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.tile";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_tile, 60)

class torch_tile_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dims
Tile                    op_0        2 1 input dims out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.tile";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_tile_onnx, 60)

class torch_tile_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Tile                    op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.tile";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.repeats") == captured_params.end())
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("op_0.repeats").type == 5)
        {
            op->params["dims"] = captured_params.at("op_0.repeats");
        }
        else // if (captured_params.at("op_0.repeats").type == 2)
        {
            op->params["dims"] = std::vector<int>{captured_params.at("op_0.repeats").i};
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_tile_onnx_1, 60)

} // namespace pnnx
