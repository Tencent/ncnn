// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_module_pass.h"

namespace pnnx {

class Softmax2d : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.torch.nn.modules.activation.Softmax2d";
    }

    const char* type_str() const
    {
        return "nn.Softmax2d";
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(Softmax2d)

} // namespace pnnx
