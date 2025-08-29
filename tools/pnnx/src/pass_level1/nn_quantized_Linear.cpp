// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/passes/quantization/helper.h>

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

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& _mod) const
    {
        const auto& mod = _mod.mod;

        //         mod.dump(true, false, false);

        //         graph->dump();

        const TorchNodeProxy* quantized_linear = graph.find_node_by_kind("quantized::linear");

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

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& _mod) const
    {
        const auto& mod = _mod.mod;

        //         mod.dump(true, false, false);

        graph.dump();

        const TorchNodeProxy* quantized_linear = graph.find_node_by_kind("quantized::linear_relu");

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
