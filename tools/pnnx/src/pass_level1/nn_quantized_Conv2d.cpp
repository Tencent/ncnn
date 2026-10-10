// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/passes/quantization/helper.h>

namespace pnnx {

class QuantizedConv2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.quantized.modules.conv.Conv2d";
    }

    const char* type_str() const
    {
        return "nn.quantized.Conv2d";
    }

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& _mod) const
    {
        const auto& mod = _mod.mod;

        //         graph->dump();

        const TorchNodeProxy* quantized_convolution = graph.find_node_by_kind("quantized::conv2d");

        //         for (auto aa : quantized_convolution->schema().arguments())
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

        //         quantized_convolution->namedInput("output_scale");

        const auto& packed_params = mod.attr("_packed_params").toObject();

        //         auto x = torch::jit::script::Object(packed_params).run_method("__getstate__");
        auto x = torch::jit::script::Object(packed_params).run_method("unpack").toTuple();
        //         std::cout << x->elements()[0].toTensor() << std::endl;
        //         std::cout << x->elements()[0].toTensor().quantizer() << std::endl;
        //         std::cout << x->elements()[1] << std::endl;
        //   at::Tensor dequantize() const;
        //   double q_scale() const;
        //   int64_t q_zero_point() const;
        //   at::Tensor q_per_channel_scales() const;
        //   at::Tensor q_per_channel_zero_points() const;
        //   int64_t q_per_channel_axis() const;

        //         auto quantizer = x->elements()[0].toTensor().quantizer();

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

        op->params["in_channels"] = mod.attr("in_channels").toInt();
        op->params["out_channels"] = mod.attr("out_channels").toInt();
        op->params["kernel_size"] = Parameter{mod.attr("kernel_size").toTuple()->elements()[0].toInt(), mod.attr("kernel_size").toTuple()->elements()[1].toInt()};
        op->params["stride"] = Parameter{mod.attr("stride").toTuple()->elements()[0].toInt(), mod.attr("stride").toTuple()->elements()[1].toInt()};
        op->params["padding"] = Parameter{mod.attr("padding").toTuple()->elements()[0].toInt(), mod.attr("padding").toTuple()->elements()[1].toInt()};
        op->params["dilation"] = Parameter{mod.attr("dilation").toTuple()->elements()[0].toInt(), mod.attr("dilation").toTuple()->elements()[1].toInt()};
        op->params["groups"] = mod.attr("groups").toInt();
        op->params["padding_mode"] = "zeros";
        op->params["bias"] = mod.hasattr("bias");

        op->params["scale"] = quantized_convolution->namedInput("output_scale");
        op->params["zero_point"] = quantized_convolution->namedInput("output_zero_point");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(QuantizedConv2d)

class QuantizedConvReLU2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d";
    }

    const char* type_str() const
    {
        return "nn.intrinsic.quantized.ConvReLU2d";
    }

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& _mod) const
    {
        const auto& mod = _mod.mod;

        //         graph->dump();

        const TorchNodeProxy* quantized_convolution = graph.find_node_by_kind("quantized::conv2d_relu");

        //         for (auto aa : quantized_convolution->schema().arguments())
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

        //         quantized_convolution->namedInput("output_scale");

        const auto& packed_params = mod.attr("_packed_params").toObject();

        //         auto x = torch::jit::script::Object(packed_params).run_method("__getstate__");
        auto x = torch::jit::script::Object(packed_params).run_method("unpack").toTuple();
        //         std::cout << x->elements()[0].toTensor() << std::endl;
        //         std::cout << x->elements()[0].toTensor().quantizer() << std::endl;
        //         std::cout << x->elements()[1] << std::endl;
        //   at::Tensor dequantize() const;
        //   double q_scale() const;
        //   int64_t q_zero_point() const;
        //   at::Tensor q_per_channel_scales() const;
        //   at::Tensor q_per_channel_zero_points() const;
        //   int64_t q_per_channel_axis() const;

        //         auto quantizer = x->elements()[0].toTensor().quantizer();

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

        op->params["in_channels"] = mod.attr("in_channels").toInt();
        op->params["out_channels"] = mod.attr("out_channels").toInt();
        op->params["kernel_size"] = Parameter{mod.attr("kernel_size").toTuple()->elements()[0].toInt(), mod.attr("kernel_size").toTuple()->elements()[1].toInt()};
        op->params["stride"] = Parameter{mod.attr("stride").toTuple()->elements()[0].toInt(), mod.attr("stride").toTuple()->elements()[1].toInt()};
        op->params["padding"] = Parameter{mod.attr("padding").toTuple()->elements()[0].toInt(), mod.attr("padding").toTuple()->elements()[1].toInt()};
        op->params["dilation"] = Parameter{mod.attr("dilation").toTuple()->elements()[0].toInt(), mod.attr("dilation").toTuple()->elements()[1].toInt()};
        op->params["groups"] = mod.attr("groups").toInt();
        op->params["padding_mode"] = "zeros";
        op->params["bias"] = mod.hasattr("bias");

        op->params["scale"] = quantized_convolution->namedInput("output_scale");
        op->params["zero_point"] = quantized_convolution->namedInput("output_zero_point");
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(QuantizedConvReLU2d)

} // namespace pnnx
