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

#include "../utils.h"

namespace pnnx {

class ConvTranspose1d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.conv.ConvTranspose1d";
    }

    const char* type_str() const
    {
        return "nn.ConvTranspose1d";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod) const
    {
        const torch::jit::Node* convolution = find_node_by_kind(graph, "aten::_convolution");

        const auto& weight = mod.attr("weight").toTensor();

        op->params["groups"] = convolution->namedInput("groups");
        op->params["in_channels"] = weight.size(0);
        op->params["out_channels"] = weight.size(1) * op->params["groups"].i;
        op->params["kernel_size"] = Parameter{weight.size(2)};
        op->params["stride"] = convolution->namedInput("stride");
        op->params["padding"] = convolution->namedInput("padding");
        op->params["output_padding"] = convolution->namedInput("output_padding");
        op->params["dilation"] = convolution->namedInput("dilation");
        op->params["bias"] = mod.hasattr("bias");

        op->attrs["weight"] = weight;
        if (mod.hasattr("bias"))
        {
            op->attrs["bias"] = mod.attr("bias").toTensor();
        }

        if (op->inputs.size() > 1)
        {
            fprintf(stderr, "ConvTranspose1d arg output_size detected and dropped !\n");

            for (size_t i = 1; i < op->inputs.size(); i++)
            {
                op->inputs[i]->remove_consumer(op);
            }
            op->inputs.resize(1);
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(ConvTranspose1d)

} // namespace pnnx
