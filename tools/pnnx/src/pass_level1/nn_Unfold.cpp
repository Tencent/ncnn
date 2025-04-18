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

#include "pass_level1.h"

#include "../utils.h"

namespace pnnx {

class Unfold : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.fold.Unfold";
    }

    const char* type_str() const
    {
        return "nn.Unfold";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const
    {
        const torch::jit::Node* im2col = find_node_by_kind(graph, "aten::im2col");

        op->params["kernel_size"] = im2col->namedInput("kernel_size");
        op->params["stride"] = im2col->namedInput("stride");
        op->params["padding"] = im2col->namedInput("padding");
        op->params["dilation"] = im2col->namedInput("dilation");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Unfold)

} // namespace pnnx
