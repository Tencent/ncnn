// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class ReLU6 : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.ReLU6";
    }

    const char* type_str() const
    {
        return "nn.ReLU6";
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(ReLU6)

} // namespace pnnx
