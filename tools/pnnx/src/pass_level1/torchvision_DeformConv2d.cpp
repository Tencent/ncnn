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

class DeformConv2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torchvision.ops.deform_conv.DeformConv2d";
    }

    const char* type_str() const
    {
        return "torchvision.ops.DeformConv2d";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod) const
    {
        const torch::jit::Node* deform_conv2d = find_node_by_kind(graph, "torchvision::deform_conv2d");

        const auto& weight = mod.attr("weight").toTensor();

        const Parameter stride_w = deform_conv2d->namedInput("stride_w");
        const Parameter stride_h = deform_conv2d->namedInput("stride_h");
        const Parameter pad_w = deform_conv2d->namedInput("pad_w");
        const Parameter pad_h = deform_conv2d->namedInput("pad_h");
        const Parameter dilation_w = deform_conv2d->namedInput("dilation_w");
        const Parameter dilation_h = deform_conv2d->namedInput("dilation_h");

        op->params["groups"] = deform_conv2d->namedInput("groups");
        op->params["in_channels"] = weight.size(1) * op->params["groups"].i;
        op->params["out_channels"] = weight.size(0);
        op->params["kernel_size"] = Parameter{weight.size(2), weight.size(3)};
        op->params["stride"] = {stride_h.i, stride_w.i};
        op->params["padding"] = {pad_h.i, pad_w.i};
        op->params["dilation"] = {dilation_h.i, dilation_w.i};
        op->params["bias"] = mod.hasattr("bias");

        op->attrs["weight"] = weight;
        if (mod.hasattr("bias"))
        {
            op->attrs["bias"] = mod.attr("bias").toTensor();
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(DeformConv2d)

} // namespace pnnx
