// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level4.h"

#include "pass_level4/canonicalize.h"
#include "pass_level4/fuse_custom_op.h"
#include "pass_level4/dead_code_elimination.h"

namespace pnnx {

void pass_level4(Graph& g)
{
    fuse_custom_op(g);

    dead_code_elimination(g);

    //canonicalize(g);
}

} // namespace pnnx
