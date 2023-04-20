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

class MaxPool2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.pooling.MaxPool2d";
    }

    const char* type_str() const
    {
        return "nn.MaxPool2d";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const
    {
        const torch::jit::Node* max_pool2d = find_node_by_kind(graph, "aten::max_pool2d");
        const torch::jit::Node* max_pool2d_with_indices = find_node_by_kind(graph, "aten::max_pool2d_with_indices");

        if (max_pool2d_with_indices)
        {
            max_pool2d = max_pool2d_with_indices;
        }

        op->params["kernel_size"] = max_pool2d->namedInput("kernel_size");
        op->params["stride"] = max_pool2d->namedInput("stride");
        op->params["padding"] = max_pool2d->namedInput("padding");
        op->params["dilation"] = max_pool2d->namedInput("dilation");
        op->params["ceil_mode"] = max_pool2d->namedInput("ceil_mode");
        op->params["return_indices"] = max_pool2d_with_indices ? true : false;
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(MaxPool2d)

} // namespace pnnx
