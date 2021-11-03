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

class InstanceNorm1d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.instancenorm.InstanceNorm1d";
    }

    const char* type_str() const
    {
        return "nn.InstanceNorm1d";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod) const
    {
        //         graph->dump();

        const torch::jit::Node* in = find_node_by_kind(graph, "aten::instance_norm");

        //         for (auto aa : in->schema().arguments())
        //         {
        //             fprintf(stderr, "arg %s\n", aa.name().c_str());
        //         }

        op->params["eps"] = in->namedInput("eps");
        op->params["affine"] = mod.hasattr("weight") && mod.hasattr("bias");
        op->params["track_running_stats"] = mod.hasattr("running_mean") && mod.hasattr("running_var");

        if (mod.hasattr("weight") && mod.hasattr("bias"))
        {
            const auto& weight = mod.attr("weight").toTensor();

            op->params["num_features"] = weight.size(0);

            op->attrs["weight"] = weight;
            op->attrs["bias"] = mod.attr("bias").toTensor();
        }

        if (mod.hasattr("running_mean") && mod.hasattr("running_var"))
        {
            const auto& running_mean = mod.attr("running_mean").toTensor();

            op->params["num_features"] = running_mean.size(0);

            op->attrs["running_mean"] = running_mean;
            op->attrs["running_var"] = mod.attr("running_var").toTensor();
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(InstanceNorm1d)

} // namespace pnnx
