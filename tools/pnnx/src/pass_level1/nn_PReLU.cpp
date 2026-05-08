// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class PReLU : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.PReLU";
    }

    const char* type_str() const
    {
        return "nn.PReLU";
    }

    void write(Operator* op, const TorchGraphProxy& /*graph*/, const TorchModuleProxy& mod) const
    {
        const TorchTensorProxy& weight = mod.attr("weight");

        op->params["num_parameters"] = weight.size(0);

        op->attrs["weight"] = weight;
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(PReLU)

} // namespace pnnx
