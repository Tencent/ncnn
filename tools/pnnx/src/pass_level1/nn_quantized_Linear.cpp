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

class QuantizedLinear : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.quantized.modules.linear.Linear";
    }

    const char* type_str() const
    {
        return "nn.quantized.Linear";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod) const
    {
        //         mod.dump(true, false, false);

        //         graph->dump();

        const torch::jit::Node* quantized_linear = find_node_by_kind(graph, "quantized::linear");

        //         for (auto aa : quantized_linear->schema().arguments())
        //         {
        //             fprintf(stderr, "arg %s\n", aa.name().c_str());
        //         }

        //         torch::jit::Node* packed_params_node = 0;
        //         for (const auto& n : graph->nodes())
        //         {
        //             if (n->kind() == c10::prim::GetAttr && n->s(torch::jit::attr::name) == "_packed_params")
        //             {
        //                 packed_params_node = n;
        //                 break;
        //             }
        //         }

        const auto& packed_params = mod.attr("_packed_params").toObject();

        //         for (auto aa : torch::jit::script::Object(packed_params).get_methods())
        //         {
        //             fprintf(stderr, "method %s\n", aa.name().c_str());
        //         }

        auto x = torch::jit::script::Object(packed_params).run_method("_weight_bias").toTuple();

        auto weight = x->elements()[0].toTensor();
        auto bias = x->elements()[1].toTensor();

        op->attrs["weight"] = weight;
        op->attrs["bias"] = bias;

        if (weight.qscheme() == c10::kPerChannelAffine)
        {
            op->attrs["weight.q_per_channel_scales"] = weight.q_per_channel_scales();
            op->attrs["weight.q_per_channel_zero_points"] = weight.q_per_channel_zero_points();
            //             op->params["weight.q_per_channel_axis"] = weight.q_per_channel_axis();
        }

        op->params["in_features"] = weight.size(1);
        op->params["out_features"] = weight.size(0);

        op->params["scale"] = quantized_linear->namedInput("Y_scale_i");
        op->params["zero_point"] = quantized_linear->namedInput("Y_zero_point_i");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(QuantizedLinear)

class QuantizedLinearReLU : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.intrinsic.quantized.modules.linear_relu.LinearReLU";
    }

    const char* type_str() const
    {
        return "nn.intrinsic.quantized.LinearReLU";
    }

    void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod) const
    {
        //         mod.dump(true, false, false);

        graph->dump();

        const torch::jit::Node* quantized_linear = find_node_by_kind(graph, "quantized::linear_relu");

        //         for (auto aa : quantized_linear->schema().arguments())
        //         {
        //             fprintf(stderr, "arg %s\n", aa.name().c_str());
        //         }

        //         torch::jit::Node* packed_params_node = 0;
        //         for (const auto& n : graph->nodes())
        //         {
        //             if (n->kind() == c10::prim::GetAttr && n->s(torch::jit::attr::name) == "_packed_params")
        //             {
        //                 packed_params_node = n;
        //                 break;
        //             }
        //         }

        const auto& packed_params = mod.attr("_packed_params").toObject();

        //         for (auto aa : torch::jit::script::Object(packed_params).get_methods())
        //         {
        //             fprintf(stderr, "method %s\n", aa.name().c_str());
        //         }

        auto x = torch::jit::script::Object(packed_params).run_method("_weight_bias").toTuple();

        auto weight = x->elements()[0].toTensor();
        auto bias = x->elements()[1].toTensor();

        op->attrs["weight"] = weight;
        op->attrs["bias"] = bias;

        if (weight.qscheme() == c10::kPerChannelAffine)
        {
            op->attrs["weight.q_per_channel_scales"] = weight.q_per_channel_scales();
            op->attrs["weight.q_per_channel_zero_points"] = weight.q_per_channel_zero_points();
            //             op->params["weight.q_per_channel_axis"] = weight.q_per_channel_axis();
        }

        op->params["in_features"] = weight.size(1);
        op->params["out_features"] = weight.size(0);

        op->params["scale"] = quantized_linear->namedInput("Y_scale_i");
        op->params["zero_point"] = quantized_linear->namedInput("Y_zero_point_i");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(QuantizedLinearReLU)

} // namespace pnnx
