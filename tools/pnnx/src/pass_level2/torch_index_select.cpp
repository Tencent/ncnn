// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"
#include <string.h>

namespace pnnx {

class torch_index_select : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
pnnx.Input              input_2     0 1 index
aten::index_select      op_1        3 1 input dim index out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.index_select";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_index_select, 70)

class torch_index_select_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        // clang-format off
        // *INDENT-OFF*

        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_0        0 1 index @data=(?)i32
tnn.Gather              op_1        2 1 input index out arg0=%dim arg1=0 arg2=1
pnnx.Output             output      1 0 out
)PNNXIR";

        // *INDENT-ON*
        // clang-format on
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          index       0 1 index
torch.index_select      select      2 1 input index out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const Attribute& index_data = captured_attrs.at("op_0.data");

        // i32 to i64
        Operator* op_index = ops.at("index");
        const int* p = (const int*)index_data.data.data();
        const int n = index_data.data.size() / 4;
        std::vector<int64_t> indices(n);
        for (int i = 0; i < n; i++)
        {
            indices[i] = p[i];
        }
        op_index->attrs["data"].type = 5; // i64
        op_index->attrs["data"].shape = {n};
        op_index->attrs["data"].data.resize(n * 8);
        memcpy((void*)op_index->attrs["data"].data.data(), (const void*)indices.data(), n * 8);

        Operator* op_gather = ops.at("select");
        op_gather->params["dim"] = captured_params.at("dim");
        op_gather->inputnames = {"input", "index"};
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_index_select_tnn, 70)

} // namespace pnnx
