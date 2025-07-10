// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Hardsigmoid : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.Hardsigmoid";
    }

    const char* type_str() const
    {
        return "nn.Hardsigmoid";
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Hardsigmoid)

} // namespace pnnx
