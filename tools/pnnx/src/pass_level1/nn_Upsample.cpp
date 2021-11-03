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

class Upsample : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.upsampling.Upsample";
    }

    const char* type_str() const
    {
        return "nn.Upsample";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph) const
    {
        const torch::jit::Node* upsample_nearest1d = find_node_by_kind(graph, "aten::upsample_nearest1d");
        const torch::jit::Node* upsample_linear1d = find_node_by_kind(graph, "aten::upsample_linear1d");

        const torch::jit::Node* upsample_nearest2d = find_node_by_kind(graph, "aten::upsample_nearest2d");
        const torch::jit::Node* upsample_bilinear2d = find_node_by_kind(graph, "aten::upsample_bilinear2d");
        const torch::jit::Node* upsample_bicubic2d = find_node_by_kind(graph, "aten::upsample_bicubic2d");

        const torch::jit::Node* upsample_nearest3d = find_node_by_kind(graph, "aten::upsample_nearest3d");
        const torch::jit::Node* upsample_trilinear3d = find_node_by_kind(graph, "aten::upsample_trilinear3d");

        const torch::jit::Node* upsample = 0;
        if (upsample_nearest1d)
        {
            upsample = upsample_nearest1d;
            op->params["mode"] = "nearest";
        }
        else if (upsample_linear1d)
        {
            upsample = upsample_linear1d;
            op->params["mode"] = "linear";
        }
        else if (upsample_nearest2d)
        {
            upsample = upsample_nearest2d;
            op->params["mode"] = "nearest";
        }
        else if (upsample_bilinear2d)
        {
            upsample = upsample_bilinear2d;
            op->params["mode"] = "bilinear";
        }
        else if (upsample_bicubic2d)
        {
            upsample = upsample_bicubic2d;
            op->params["mode"] = "bicubic";
        }
        else if (upsample_nearest3d)
        {
            upsample = upsample_nearest3d;
            op->params["mode"] = "nearest";
        }
        else if (upsample_trilinear3d)
        {
            upsample = upsample_trilinear3d;
            op->params["mode"] = "trilinear";
        }

        if (upsample->hasNamedInput("output_size"))
        {
            op->params["size"] = upsample->namedInput("output_size");
        }

        if (upsample->hasNamedInput("scale_factors"))
        {
            op->params["scale_factor"] = upsample->namedInput("scale_factors");
        }

        if (upsample->hasNamedInput("align_corners"))
        {
            op->params["align_corners"] = upsample->namedInput("align_corners");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Upsample)

} // namespace pnnx
