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

class RoIAlign : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torchvision.ops.roi_align.RoIAlign";
    }

    const char* type_str() const
    {
        return "torchvision.ops.RoIAlign";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& /*mod*/) const
    {
        const torch::jit::Node* roi_align = find_node_by_kind(graph, "torchvision::roi_align");

        if (roi_align->inputs()[0] == graph->inputs()[2] && roi_align->inputs()[1] == graph->inputs()[1])
        {
            fprintf(stderr, "roi_align inputs swapped detected !\n");
            std::swap(op->inputs[0], op->inputs[1]);
        }

        const Parameter pooled_height = roi_align->namedInput("pooled_height");
        const Parameter pooled_width = roi_align->namedInput("pooled_width");

        op->params["spatial_scale"] = roi_align->namedInput("spatial_scale");
        op->params["sampling_ratio"] = roi_align->namedInput("sampling_ratio");
        op->params["aligned"] = roi_align->namedInput("aligned");
        op->params["output_size"] = {pooled_height.i, pooled_width.i};
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(RoIAlign)

} // namespace pnnx
