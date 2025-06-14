// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

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
