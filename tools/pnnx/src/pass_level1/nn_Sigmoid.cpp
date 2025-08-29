// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Sigmoid : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.Sigmoid";
    }

    const char* type_str() const
    {
        return "nn.Sigmoid";
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Sigmoid)

} // namespace pnnx
