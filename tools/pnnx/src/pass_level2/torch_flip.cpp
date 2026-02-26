// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_flip : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dims
aten::flip              op_0        2 1 input dims out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.flip";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_flip, 60)

class torch_flip_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Slice                   op_0        1 1 input out axes=%axes starts=%starts ends=%ends steps=%steps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.flip";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("axes").type == 2)
        {
            int start = captured_params.at("starts").i;
            int end = captured_params.at("ends").i;
            int step = captured_params.at("steps").i;

            if (start == -1 && (end == INT_MIN + 1 || end == INT_MIN) && step == -1)
                return true;
        }
        else // if (captured_params.at("axes").type == 5)
        {
            const std::vector<int>& axes = captured_params.at("axes").ai;
            const std::vector<int>& starts = captured_params.at("starts").ai;
            const std::vector<int>& ends = captured_params.at("ends").ai;
            const std::vector<int>& steps = captured_params.at("steps").ai;

            for (size_t i = 0; i < axes.size(); i++)
            {
                if (starts[i] != -1 || (ends[i] != INT_MIN + 1 && ends[i] != INT_MIN) || steps[i] != -1)
                    return false;
            }

            return true;
        }

        return false;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("axes").type == 2)
        {
            int dim = captured_params.at("axes").i;
            op->params["dims"] = std::vector<int>{dim};
        }
        else // if (captured_params.at("axes").type == 5)
        {
            op->params["dims"] = captured_params.at("axes");
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_flip_onnx, 60)

} // namespace pnnx
