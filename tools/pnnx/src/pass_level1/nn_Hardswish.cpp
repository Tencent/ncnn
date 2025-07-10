// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Hardswish : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.Hardswish";
    }

    const char* type_str() const
    {
        return "nn.Hardswish";
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Hardswish)

} // namespace pnnx
