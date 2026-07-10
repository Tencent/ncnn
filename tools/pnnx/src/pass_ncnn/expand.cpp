// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class onnx_Expand : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 shape
Expand                  op_0        2 1 input shape output
pnnx.Output             output      1 0 output
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Expand";
    }

    const char* name_str() const
    {
        return "expand";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        // No parameters needed - shape comes as second input blob
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(onnx_Expand, 20)

} // namespace ncnn

} // namespace pnnx
