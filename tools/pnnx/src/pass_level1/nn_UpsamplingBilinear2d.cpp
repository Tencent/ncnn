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

class UpsamplingBilinear2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.upsampling.UpsamplingBilinear2d";
    }

    const char* type_str() const
    {
        return "nn.UpsamplingBilinear2d";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const
    {
        const torch::jit::Node* upsample = find_node_by_kind(graph, "aten::upsample_bilinear2d");

        if (upsample->hasNamedInput("output_size"))
        {
            op->params["size"] = upsample->namedInput("output_size");
        }

        if (upsample->hasNamedInput("scale_factors"))
        {
            op->params["scale_factor"] = upsample->namedInput("scale_factors");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(UpsamplingBilinear2d)

} // namespace pnnx
