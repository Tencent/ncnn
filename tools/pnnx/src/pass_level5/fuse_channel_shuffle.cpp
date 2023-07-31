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
pnnx.Expression         op_0        1 1 input 13 expr=[int(size(@0,0)),2,int(floor_divide(size(@0,1),%groups)),int(size(@0,2)),int(size(@0,3))]
Tensor.view             op_1        2 1 input 13 14
pnnx.Expression         op_2        1 1 input 15 expr=[int(size(@0,0)),-1,int(size(@0,2)),int(size(@0,3))]
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
};

class fuse_channel_shuffle_pass_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
Tensor.view             op_0        1 1 input 13 shape=(%batch,%groups,%channels_per_group,%h,%w)
torch.transpose         op_1        1 1 13 14 dim0=1 dim1=2
Tensor.reshape          op_2        1 1 14 out shape=(%batch,%channels,%h,%w)
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

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["groups"] = captured_params.at("groups");
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
