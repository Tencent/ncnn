// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"
#include "utils.h"

namespace pnnx {

class Conv3d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.conv.Conv3d";
    }

    const char* type_str() const
    {
        return "nn.Conv3d";
    }

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& mod) const
    {
        //         {
        //             pnnx::Graph pnnx_graph;
        //
        //             pnnx_graph.load(mod, graph);
        //
        //             pnnx::fuse_expression(pnnx_graph);
        //
        //             pnnx_graph.save("tmp.param", "tmp.bin");
        //         }

        const TorchNodeProxy* convolution = graph.find_node_by_kind("aten::_convolution");
        const TorchNodeProxy* convolution_mode = graph.find_node_by_kind("aten::_convolution_mode");
        const TorchNodeProxy* pad = graph.find_node_by_kind("aten::pad");
        const TorchNodeProxy* reflection_pad3d = graph.find_node_by_kind("aten::reflection_pad3d");
        const TorchNodeProxy* replication_pad3d = graph.find_node_by_kind("aten::replication_pad3d");

        if (convolution_mode)
        {
            convolution = convolution_mode;
        }

        const TorchTensorProxy& weight = mod.hasattr("weight") ? mod.attr("weight") : mod.attr("weight_v");

        op->params["groups"] = convolution->namedInput("groups");
        op->params["in_channels"] = weight.size(1) * op->params["groups"].i;
        op->params["out_channels"] = weight.size(0);
        op->params["kernel_size"] = Parameter{weight.size(2), weight.size(3), weight.size(4)};
        op->params["stride"] = convolution->namedInput("stride");
        if (pad)
        {
            op->params["padding_mode"] = pad->namedInput("mode");
            op->params["padding"] = pad->namedInput("pad");
            std::vector<int>& padding = op->params["padding"].ai;
            if (padding.size() == 6)
            {
                // Conv3d only accepts tuple of three integers
                if (padding[0] == padding[1] && padding[1] == padding[2] && padding[2] == padding[3] && padding[3] == padding[4] && padding[4] == padding[5])
                {
                    padding.resize(3);
                }
                else if (padding[0] == padding[3] && padding[1] == padding[4] && padding[2] == padding[5] && padding[0] != padding[1] && padding[1] != padding[2])
                {
                    padding.resize(0);
                    op->params["padding"].s = "same";
                }
            }
        }
        else if (reflection_pad3d)
        {
            op->params["padding_mode"] = "reflect";
            op->params["padding"] = reflection_pad3d->namedInput("padding");
            std::vector<int>& padding = op->params["padding"].ai;
            if (padding.size() == 6)
            {
                // Conv3d only accepts tuple of three integers
                if (padding[0] == padding[1] && padding[1] == padding[2] && padding[2] == padding[3] && padding[3] == padding[4] && padding[4] == padding[5])
                {
                    padding.resize(3);
                }
                else if (padding[0] == padding[3] && padding[1] == padding[4] && padding[2] == padding[5] && padding[0] != padding[1] && padding[1] != padding[2])
                {
                    padding.resize(0);
                    op->params["padding"].s = "same";
                }
            }
        }
        else if (replication_pad3d)
        {
            op->params["padding_mode"] = "replicate";
            op->params["padding"] = replication_pad3d->namedInput("padding");
            std::vector<int>& padding = op->params["padding"].ai;
            if (padding.size() == 6)
            {
                // Conv3d only accepts tuple of three integers
                if (padding[0] == padding[1] && padding[1] == padding[2] && padding[2] == padding[3] && padding[3] == padding[4] && padding[4] == padding[5])
                {
                    padding.resize(3);
                }
                else if (padding[0] == padding[3] && padding[1] == padding[4] && padding[2] == padding[5] && padding[0] != padding[1] && padding[1] != padding[2])
                {
                    padding.resize(0);
                    op->params["padding"].s = "same";
                }
            }
        }
        else
        {
            op->params["padding_mode"] = "zeros";
            op->params["padding"] = convolution->namedInput("padding");
        }
        op->params["dilation"] = convolution->namedInput("dilation");
        op->params["bias"] = mod.hasattr("bias");

        op->attrs["weight"] = weight;
        if (!mod.hasattr("weight"))
        {
            // weight norm
            Attribute weight_g = mod.attr("weight_g");
            std::vector<float> weight_data = op->attrs["weight"].get_float32_data();
            std::vector<float> weight_g_data = weight_g.get_float32_data();
            int outch = op->params.at("out_channels").i;
            int inch = op->params.at("in_channels").i * op->params.at("kernel_size").ai[0] * op->params.at("kernel_size").ai[1] * op->params.at("kernel_size").ai[2];
            apply_weight_norm(weight_data, weight_g_data, outch, inch);
            op->attrs["weight"].set_float32_data(weight_data);

            // drop the additional weight input
            op->inputs[1]->remove_consumer(op);
            op->inputs.resize(1);
        }
        if (mod.hasattr("bias"))
        {
            op->attrs["bias"] = mod.attr("bias");
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Conv3d)

} // namespace pnnx
