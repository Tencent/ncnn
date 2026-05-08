// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_transpose_matmul.h"

#include "pass_level2.h"

#include <float.h>

namespace pnnx {

namespace ncnn {

class fuse_transpose_matmul_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_a     0 1 a
pnnx.Input              input_b     0 1 b
Permute                 op_0        1 1 b bt 0=1
MatMul                  op_1        2 1 a bt out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "MatMul";
    }

    const char* name_str() const
    {
        return "matmultransb";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        op->params["0"] = 1;
    }
};

void fuse_transpose_matmul(Graph& graph)
{
    fuse_transpose_matmul_pass a;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
}

} // namespace ncnn

} // namespace pnnx
