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

class LocalResponseNorm : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.normalization.LocalResponseNorm";
    }

    const char* type_str() const
    {
        return "nn.LocalResponseNorm";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const
    {
        const torch::jit::Node* avg_pool = find_node_by_kind(graph, "aten::avg_pool2d");
        const torch::jit::Node* avg_pool3d = find_node_by_kind(graph, "aten::avg_pool3d");

        if (avg_pool3d)
        {
            avg_pool = avg_pool3d;
        }

        op->params["size"] = avg_pool->namedInput("kernel_size")->node()->inputs()[0];

        const torch::jit::Node* pow = find_node_by_kind(graph, "aten::pow");
        op->params["beta"] = pow->inputs()[1];

        const torch::jit::Node* add = pow->inputs()[0]->node();
        op->params["k"] = add->inputs()[1];

        const torch::jit::Node* mul = add->inputs()[0]->node();
        op->params["alpha"] = mul->inputs()[1];
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(LocalResponseNorm)

} // namespace pnnx
