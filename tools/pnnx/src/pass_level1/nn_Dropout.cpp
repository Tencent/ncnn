// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Dropout : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.dropout.Dropout";
    }

    const char* type_str() const
    {
        return "nn.Dropout";
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Dropout)

} // namespace pnnx
