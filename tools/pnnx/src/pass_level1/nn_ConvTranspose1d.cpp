// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class ConvTranspose1d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.conv.ConvTranspose1d";
    }

    const char* type_str() const
    {
        return "nn.ConvTranspose1d";
    }

    void write(Operator* op, const TorchGraphProxy& graph, const TorchModuleProxy& mod) const
    {
        const TorchNodeProxy* convolution = graph.find_node_by_kind("aten::_convolution");

        const TorchTensorProxy& weight = mod.attr("weight");

        op->params["groups"] = convolution->namedInput("groups");
        op->params["in_channels"] = weight.size(0);
        op->params["out_channels"] = weight.size(1) * op->params["groups"].i;
        op->params["kernel_size"] = Parameter{weight.size(2)};
        op->params["stride"] = convolution->namedInput("stride");
        op->params["padding"] = convolution->namedInput("padding");
        op->params["output_padding"] = convolution->namedInput("output_padding");
        op->params["dilation"] = convolution->namedInput("dilation");
        op->params["bias"] = mod.hasattr("bias");

        op->attrs["weight"] = weight;
        if (mod.hasattr("bias"))
        {
            op->attrs["bias"] = mod.attr("bias");
        }

        if (op->inputs.size() > 1)
        {
            fprintf(stderr, "ConvTranspose1d arg output_size detected and dropped !\n");

            for (size_t i = 1; i < op->inputs.size(); i++)
            {
                op->inputs[i]->remove_consumer(op);
            }
            op->inputs.resize(1);
        }
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(ConvTranspose1d)

} // namespace pnnx
