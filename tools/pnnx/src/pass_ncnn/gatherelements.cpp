// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class onnx_GatherElements : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 data
pnnx.Input              input_1     0 1 indices
GatherElements          op_0        2 1 data indices out axis=%axis
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "GatherElements";
    }

    const char* name_str() const
    {
        return "gatherelements";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int axis = 0;
        if (captured_params.find("axis") != captured_params.end())
        {
            const Parameter& axis_p = captured_params.at("axis");
            if (axis_p.type == 2)
                axis = axis_p.i;
            else if (axis_p.type == 5 && !axis_p.ai.empty())
                axis = axis_p.ai[0];
        }

        op->params["0"] = axis;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(onnx_GatherElements, 20)

} // namespace ncnn

} // namespace pnnx
