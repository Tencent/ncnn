// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "fuse_convert_shufflechannel_slice.h"

#include "pass_level2.h"

namespace pnnx {

namespace ncnn {

// def channel_shuffle(self, x):
//     batchsize, num_channels, height, width = x.data.size()
//     assert (num_channels % 4 == 0)
//     x = x.reshape(batchsize * num_channels // 2, 2, height * width)
//     x = x.permute(1, 0, 2)
//     x = x.reshape(2, -1, num_channels // 2, height, width)
//     return x[0], x[1]

class fuse_shufflechannel_slice_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 6
pnnx.Input              input       0 1 input #input=(%batch,%c,%h,%w)f32
Tensor.reshape          op_0        1 1 input a shape=(%batch_mul_ch_per_group,%groups,%h_mul_w)
torch.permute           op_1        1 1 a b dims=(1,0,2)
Tensor.reshape          op_2        1 1 b c shape=(%groups,%batch,%ch_per_group,%h,%w)
torch.unbind            op_3        1 2 c out0 out1 dim=0
pnnx.Output             output      2 0 out0 out1
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 4
pnnx.Input              input       0 1 input
ShuffleChannel          shufflechannel 1 1 input a 0=%groups 1=1 #a=(%batch,%c,%h,%w)f32
Slice                   slice       1 2 a out0 out1 0=(-233,-233) 1=0
pnnx.Output             output      2 0 out0 out1
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int groups = captured_params.at("groups").i;
        const int batch = captured_params.at("batch").i;
        const int batch_mul_ch_per_group = captured_params.at("batch_mul_ch_per_group").i;
        const int ch_per_group = captured_params.at("ch_per_group").i;
        const int h_mul_w = captured_params.at("h_mul_w").i;
        const int c = captured_params.at("c").i;
        const int h = captured_params.at("h").i;
        const int w = captured_params.at("w").i;

        if (groups != 2 || groups * ch_per_group != c)
            return false;

        if (batch_mul_ch_per_group != batch * ch_per_group)
            return false;

        if (h_mul_w != h * w)
            return false;

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int batch_index = ops.at("shufflechannel")->inputs[0]->params["__batch_index"].i;

        ops.at("slice")->inputs[0]->params["__batch_index"] = batch_index;
        ops.at("slice")->outputs[0]->params["__batch_index"] = batch_index;
        ops.at("slice")->outputs[1]->params["__batch_index"] = batch_index;
    }
};

class fuse_shufflechannel_slice_pass_1 : public fuse_shufflechannel_slice_pass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 6
pnnx.Input              input       0 1 input #input=(%batch,%c,%h,%w)f32
Tensor.reshape          op_0        1 1 input a shape=(%batch_mul_ch_per_group,%groups,%h_mul_w)
Tensor.permute          op_1        1 1 a b dims=(1,0,2)
Tensor.reshape          op_2        1 1 b c shape=(%groups,%batch,%ch_per_group,%h,%w)
torch.unbind            op_3        1 2 c out0 out1 dim=0
pnnx.Output             output      2 0 out0 out1
)PNNXIR";
    }
};

void fuse_convert_shufflechannel_slice(Graph& graph)
{
    fuse_shufflechannel_slice_pass a;
    fuse_shufflechannel_slice_pass_1 b;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
}

} // namespace ncnn

} // namespace pnnx
