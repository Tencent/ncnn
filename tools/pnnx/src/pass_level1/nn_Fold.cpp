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

class Fold : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.fold.Fold";
    }

    const char* type_str() const
    {
        return "nn.Fold";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const
    {
        const torch::jit::Node* col2im = find_node_by_kind(graph, "aten::col2im");

        op->params["output_size"] = col2im->namedInput("output_size");
        op->params["kernel_size"] = col2im->namedInput("kernel_size");
        op->params["stride"] = col2im->namedInput("stride");
        op->params["padding"] = col2im->namedInput("padding");
        op->params["dilation"] = col2im->namedInput("dilation");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Fold)

} // namespace pnnx
