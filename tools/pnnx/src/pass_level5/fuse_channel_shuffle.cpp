// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "fuse_channel_shuffle.h"

#include "pass_level2.h"

namespace pnnx {

class fuse_channel_shuffle_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
pnnx.Expression         op_0        1 1 input 13 expr=%expr
Tensor.view             op_1        2 1 input 13 14
pnnx.Expression         op_2        1 1 input 15 expr=%expr2
torch.transpose         op_3        1 1 14 16 dim0=1 dim1=2
Tensor.reshape          op_4        2 1 16 15 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.ChannelShuffle";
    }

    const char* name_str() const
    {
        return "channelshuffle";
    }

    bool match_captured_params_attrs(const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& expr = captured_params.at("expr").s;
        const std::string& expr2 = captured_params.at("expr2").s;

        if (expr2 != "[int(size(@0,0)),-1,int(size(@0,2)),int(size(@0,3))]")
            return false;

        int groups;
        int nscan = sscanf(expr.c_str(), "[int(size(@0,0)),2,int(floor_divide(size(@0,1),%d)),int(size(@0,2)),int(size(@0,3))]", &groups);
        if (nscan != 1)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& expr = captured_params.at("expr").s;

        int groups;
        sscanf(expr.c_str(), "[int(size(@0,0)),2,int(floor_divide(size(@0,1),%d)),int(size(@0,2)),int(size(@0,3))]", &groups);

        op->params["groups"] = groups;
    }
};

class fuse_channel_shuffle_pass_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
Tensor.view             op_0        1 1 input 13 shape=%shape
torch.transpose         op_1        1 1 13 14 dim0=1 dim1=2
Tensor.reshape          op_2        1 1 14 out shape=%shape2
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.ChannelShuffle";
    }

    const char* name_str() const
    {
        return "channelshuffle";
    }

    bool match_captured_params_attrs(const std::map<std::string, Parameter>& captured_params) const
    {
        // (1,2,58,28,28)
        // (1,-1,28,28)
        const std::vector<int>& shape = captured_params.at("shape").ai;
        const std::vector<int>& shape2 = captured_params.at("shape2").ai;

        if (shape[0] != 1 || shape2[0] != 1 || shape2[1] != -1 || shape2[2] != shape[3] || shape2[3] != shape[4])
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& shape = captured_params.at("shape").ai;

        int groups = shape[1];

        op->params["groups"] = groups;
    }
};

void fuse_channel_shuffle(Graph& graph)
{
    fuse_channel_shuffle_pass a;
    fuse_channel_shuffle_pass_1 b;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
}

} // namespace pnnx
