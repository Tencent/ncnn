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

#include "pass_level1.h"

// #include "../pass_level3/fuse_expression.h"

#include "../utils.h"

namespace pnnx {

class Conv1d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.conv.Conv1d";
    }

    const char* type_str() const
    {
        return "nn.Conv1d";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod) const
    {
        //         {
        //             pnnx::Graph pnnx_graph;
        //
        //             pnnx_graph.load(mod, graph);
        //
        //             pnnx::fuse_expression(pnnx_graph);
        //
        //             pnnx_graph.save("tmp.param", "tmp.bin");
        //         }

        const torch::jit::Node* convolution = find_node_by_kind(graph, "aten::_convolution");
        const torch::jit::Node* convolution_mode = find_node_by_kind(graph, "aten::_convolution_mode");
        const torch::jit::Node* reflection_pad1d = find_node_by_kind(graph, "aten::reflection_pad1d");
        const torch::jit::Node* replication_pad1d = find_node_by_kind(graph, "aten::replication_pad1d");

        if (convolution_mode)
        {
            convolution = convolution_mode;
        }

        const auto& weight = mod.attr("weight").toTensor();

        op->params["groups"] = convolution->namedInput("groups");
        op->params["in_channels"] = weight.size(1) * op->params["groups"].i;
        op->params["out_channels"] = weight.size(0);
        op->params["kernel_size"] = Parameter{weight.size(2)};
        op->params["stride"] = convolution->namedInput("stride");
        if (reflection_pad1d)
        {
            op->params["padding_mode"] = "reflect";
            op->params["padding"] = reflection_pad1d->namedInput("padding");
            std::vector<int>& padding = op->params["padding"].ai;
            if (padding.size() == 2)
            {
                // Conv1d only accepts tuple of one integer
                if (padding[0] == padding[1])
                {
                    padding.resize(1);
                }
                else if (padding[0] != padding[1])
                {
                    padding.resize(0);
                    op->params["padding"].s = "same";
                }
            }
        }
        else if (replication_pad1d)
        {
            op->params["padding_mode"] = "replicate";
            op->params["padding"] = replication_pad1d->namedInput("padding");
            std::vector<int>& padding = op->params["padding"].ai;
            if (padding.size() == 2)
            {
                // Conv1d only accepts tuple of one integer
                if (padding[0] == padding[1])
                {
                    padding.resize(1);
                }
                else if (padding[0] != padding[1])
                {
                    padding.resize(0);
                    op->params["padding"].s = "same";
                }
            }
        }
        else
        {
            op->params["padding_mode"] = "zeros";
            op->params["padding"] = convolution->namedInput("padding");
        }
        op->params["dilation"] = convolution->namedInput("dilation");
        op->params["bias"] = mod.hasattr("bias");

        op->attrs["weight"] = weight;
        if (mod.hasattr("bias"))
        {
            op->attrs["bias"] = mod.attr("bias").toTensor();
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Conv1d)

} // namespace pnnx
