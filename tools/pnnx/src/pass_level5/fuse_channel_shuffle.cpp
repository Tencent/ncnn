// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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
Tensor.reshape          op_1        2 1 input 13 14
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
Tensor.reshape          op_0        1 1 input 13 shape=(%batch,%groups,%channels_per_group,%h,%w)
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

class fuse_channel_shuffle_pass_2 : public fuse_channel_shuffle_pass_1
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
Tensor.reshape          op_0        1 1 input 13 shape=(%batch,%groups,%channels_per_group,%h,%w)
Tensor.permute          op_1        1 1 13 14 dims=(0,2,1,3,4)
Tensor.reshape          op_2        1 1 14 out shape=(%batch,%channels,%h,%w)
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

void fuse_channel_shuffle(Graph& graph)
{
    fuse_channel_shuffle_pass a;
    fuse_channel_shuffle_pass_1 b;
    fuse_channel_shuffle_pass_2 c;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &c, opindex);
}

} // namespace pnnx
