// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class onnx_Tile : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 repeats
Tile                    op_0        2 1 input repeats output
pnnx.Output             output      1 0 output
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tile";
    }

    const char* name_str() const
    {
        return "tile";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        // No parameters needed - repeats comes as second input blob
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(onnx_Tile, 20)

} // namespace ncnn

} // namespace pnnx
