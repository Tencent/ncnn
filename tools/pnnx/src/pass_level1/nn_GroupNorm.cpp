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

class GroupNorm : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.normalization.GroupNorm";
    }

    const char* type_str() const
    {
        return "nn.GroupNorm";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod) const
    {
        //         graph->dump();

        const torch::jit::Node* gn = find_node_by_kind(graph, "aten::group_norm");

        //         for (auto aa : gn->schema().arguments())
        //         {
        //             fprintf(stderr, "arg %s\n", aa.name().c_str());
        //         }

        op->params["num_groups"] = gn->namedInput("num_groups");
        op->params["eps"] = gn->namedInput("eps");
        op->params["affine"] = mod.hasattr("weight") && mod.hasattr("bias");

        if (mod.hasattr("weight") && mod.hasattr("bias"))
        {
            const auto& weight = mod.attr("weight").toTensor();

            op->params["num_channels"] = weight.size(0);

            op->attrs["weight"] = weight;
            op->attrs["bias"] = mod.attr("bias").toTensor();
        }
        else
        {
            fprintf(stderr, "Cannot resolve GroupNorm num_channels when affint=False\n");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(GroupNorm)

} // namespace pnnx
