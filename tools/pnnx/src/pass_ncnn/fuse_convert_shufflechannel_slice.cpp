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
pnnx.Input              input       0 1 input
Tensor.reshape          op_0        1 1 input a shape=%shape
torch.permute           op_1        1 1 a b dims=%dims
Tensor.reshape          op_2        1 1 b c shape=%shape2
torch.unbind            op_3        1 2 c out0 out1 dim=0
pnnx.Output             output      2 0 out0 out1
)PNNXIR";
    }

    const char* type_str() const
    {
        return "ncnn._shufflechannel_slice";
    }

    const char* name_str() const
    {
        return "shufflechannel_slice";
    }

    bool match_captured_params_attrs(const std::map<std::string, Parameter>& captured_params) const
    {
        // (116,2,1024)
        // (1,0,2)
        // (2,-1,116,32,32)
        const std::vector<int>& shape = captured_params.at("shape").ai;
        const std::vector<int>& dims = captured_params.at("dims").ai;
        const std::vector<int>& shape2 = captured_params.at("shape2").ai;

        if (dims != std::vector<int>{1, 0, 2})
            return false;

        if (shape[0] != shape2[2] || shape[1] != shape2[0] || shape[2] != shape2[3] * shape2[4] || shape[1] != 2 || shape2[1] != -1)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& shape = captured_params.at("shape").ai;

        int groups = shape[1];

        op->params["0"] = groups;
        op->params["1"] = 1;
    }
};

class fuse_shufflechannel_slice_pass_1 : public fuse_shufflechannel_slice_pass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 6
pnnx.Input              input       0 1 input
Tensor.reshape          op_0        1 1 input a shape=%shape
Tensor.permute          op_1        1 1 a b dims=%dims
Tensor.reshape          op_2        1 1 b c shape=%shape2
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

    int op_index = 0;

    while (1)
    {
        bool matched = false;

        for (Operator* op : graph.ops)
        {
            if (op->type != "ncnn._shufflechannel_slice")
                continue;

            matched = true;

            const int batch_index = op->inputs[0]->params["__batch_index"].i;

            op->type = "ShuffleChannel";
            op->name = std::string("shufflechannel_") + std::to_string(op_index++);

            Operand* out0 = op->outputs[0];
            Operand* out1 = op->outputs[1];

            Operator* slice = graph.new_operator_after("Slice", op->name + "_slice", op);

            Operand* slice_in = graph.new_operand(op->name + "_slice_in");

            slice_in->params["__batch_index"] = batch_index;
            out0->params["__batch_index"] = batch_index;
            out1->params["__batch_index"] = batch_index;

            slice->inputs.push_back(slice_in);
            slice->outputs.push_back(out0);
            slice->outputs.push_back(out1);

            op->outputs.clear();
            op->outputs.push_back(slice_in);

            out0->producer = slice;
            out1->producer = slice;
            slice_in->producer = op;
            slice_in->consumers.push_back(slice);

            slice->params["0"] = std::vector<int>{-233, -233};
            slice->params["1"] = 0;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
