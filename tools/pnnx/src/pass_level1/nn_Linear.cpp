// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"
#include "utils.h"

namespace pnnx {

class Linear : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.linear.Linear";
    }

    const char* type_str() const
    {
        return "nn.Linear";
    }

    void write(Operator* op, const TorchGraphProxy& /*graph*/, const TorchModuleProxy& mod) const
    {
        // const TorchNodeProxy* addmm = graph.find_node_by_kind("aten::addmm");

        const TorchTensorProxy& weight = mod.hasattr("weight") ? mod.attr("weight") : mod.attr("weight_v");

        op->params["in_features"] = weight.size(1);
        op->params["out_features"] = weight.size(0);
        op->params["bias"] = mod.hasattr("bias");

        op->attrs["weight"] = weight;
        if (!mod.hasattr("weight"))
        {
            // weight norm
            Attribute weight_g = mod.attr("weight_g");
            std::vector<float> weight_data = op->attrs["weight"].get_float32_data();
            std::vector<float> weight_g_data = weight_g.get_float32_data();
            int outch = op->params.at("out_features").i;
            int inch = op->params.at("in_features").i;
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

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Linear)

} // namespace pnnx
