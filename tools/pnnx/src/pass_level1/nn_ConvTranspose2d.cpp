// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"
#include "utils.h"

namespace pnnx {

class ConvTranspose2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.conv.ConvTranspose2d";
    }

    const char* type_str() const
    {
        return "nn.ConvTranspose2d";
    }

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& mod) const
    {
        const TorchNodeProxy* convolution = graph.find_node_by_kind("aten::_convolution");

        const TorchTensorProxy& weight = mod.hasattr("weight") ? mod.attr("weight") : mod.attr("weight_v");

        op->params["groups"] = convolution->namedInput("groups");
        op->params["in_channels"] = weight.size(0);
        op->params["out_channels"] = weight.size(1) * op->params["groups"].i;
        op->params["kernel_size"] = Parameter{weight.size(2), weight.size(3)};
        op->params["stride"] = convolution->namedInput("stride");
        op->params["padding"] = convolution->namedInput("padding");
        op->params["output_padding"] = convolution->namedInput("output_padding");
        op->params["dilation"] = convolution->namedInput("dilation");
        op->params["bias"] = mod.hasattr("bias");

        op->attrs["weight"] = weight;
        if (!mod.hasattr("weight"))
        {
            // weight norm
            Attribute weight_g = mod.attr("weight_g");
            std::vector<float> weight_data = op->attrs["weight"].get_float32_data();
            std::vector<float> weight_g_data = weight_g.get_float32_data();
            int inch = op->params.at("in_channels").i;
            int outch = op->params.at("out_channels").i * op->params.at("kernel_size").ai[0] * op->params.at("kernel_size").ai[1];
            apply_weight_norm(weight_data, weight_g_data, inch, outch);
            op->attrs["weight"].set_float32_data(weight_data);

            // drop the additional weight input
            op->inputs[1]->remove_consumer(op);
            op->inputs.resize(1);
        }
        if (mod.hasattr("bias"))
        {
            op->attrs["bias"] = mod.attr("bias");
        }

        if (op->inputs.size() > 1)
        {
            fprintf(stderr, "ConvTranspose2d arg output_size detected and dropped !\n");

            for (size_t i = 1; i < op->inputs.size(); i++)
            {
                op->inputs[i]->remove_consumer(op);
            }
            op->inputs.resize(1);
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(ConvTranspose2d)

} // namespace pnnx
