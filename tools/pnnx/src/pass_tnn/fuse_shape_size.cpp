// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_shape_size.h"

#include "pass_level2.h"

namespace pnnx {

namespace tnn2pnnx {

class fuse_shape_size_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
tnn.Shape               op_0        1 1 input a
pnnx.Attribute          op_1        0 1 index @data=(1)i32
tnn.Gather              op_2        2 1 a index out arg0=0 arg1=0 arg2=1
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
prim::Constant          index       0 1 index
aten::size              size        2 1 input index out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const Attribute& index_data = captured_attrs.at("op_1.data");
        const int index = ((const int*)index_data.data.data())[0];

        Operator* op_index = ops.at("index");
        op_index->params["value"] = index;
    }
};

void fuse_shape_size(Graph& graph)
{
    // TODO unpool tnn.Shape

    fuse_shape_size_pass a;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
}

} // namespace tnn2pnnx

} // namespace pnnx
