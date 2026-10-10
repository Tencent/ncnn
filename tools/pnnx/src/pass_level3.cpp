// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level3.h"

#include "pass_level3/assign_unique_name.h"
#include "pass_level3/eliminate_noop_math.h"
#include "pass_level3/eliminate_squeeze_unsqueeze_pair.h"
#include "pass_level3/eliminate_tuple_pair.h"
#include "pass_level3/expand_quantization_modules.h"
#include "pass_level3/fuse_opnto1_tensors.h"
#include "pass_level3/fuse_op1ton_unpack.h"
#include "pass_level3/fuse_dynamic_adaptive_pool.h"
#include "pass_level3/fuse_einsum_operands.h"
#include "pass_level3/fuse_expression.h"
#include "pass_level3/fuse_index_expression.h"
#include "pass_level3/fuse_maxpool_unpack.h"
#include "pass_level3/fuse_multiheadattention_unpack.h"
#include "pass_level3/fuse_rnn_unpack.h"
#include "pass_level3/rename_F_dropoutnd.h"

// #include "pass_level4/canonicalize.h"
// #include "pass_level4/fuse_custom_op.h"
// #include "pass_level4/dead_code_elimination.h"

namespace pnnx {

void pass_level3(Graph& g, const std::set<std::string>& foldable_constants, const std::string& foldable_constants_zippath)
{
    assign_unique_name(g);

    fuse_opnto1_tensors(g);

    fuse_op1ton_unpack(g);

    fuse_einsum_operands(g);

    fuse_maxpool_unpack(g);

    fuse_multiheadattention_unpack(g);

    fuse_rnn_unpack(g);

    fuse_dynamic_adaptive_pool(g);

    expand_quantization_modules(g);

    eliminate_tuple_pair(g);
    eliminate_squeeze_unsqueeze_pair(g);

    rename_F_dropoutnd(g);

    eliminate_noop_math(g);

    fuse_expression(g, foldable_constants, foldable_constants_zippath);

    fuse_index_expression(g);

    //     dead_code_elimination(g);

    //     canonicalize(g);
}

} // namespace pnnx
