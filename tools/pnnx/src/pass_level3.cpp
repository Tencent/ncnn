// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "pass_level3.h"

#include "pass_level3/assign_unique_name.h"
#include "pass_level3/eliminate_noop_math.h"
#include "pass_level3/eliminate_tuple_pair.h"
#include "pass_level3/expand_quantization_modules.h"
#include "pass_level3/fuse_cat_stack_tensors.h"
#include "pass_level3/fuse_chunk_split_unbind_unpack.h"
#include "pass_level3/fuse_expression.h"
#include "pass_level3/fuse_index_expression.h"
#include "pass_level3/fuse_rnn_unpack.h"
#include "pass_level3/rename_F_conv_transposend.h"
#include "pass_level3/rename_F_convmode.h"
#include "pass_level3/rename_F_dropoutnd.h"

// #include "pass_level4/canonicalize.h"
// #include "pass_level4/fuse_custom_op.h"
// #include "pass_level4/dead_code_elimination.h"

namespace pnnx {

void pass_level3(Graph& g)
{
    assign_unique_name(g);

    fuse_cat_stack_tensors(g);

    fuse_chunk_split_unbind_unpack(g);

    fuse_rnn_unpack(g);

    expand_quantization_modules(g);

    eliminate_tuple_pair(g);

    rename_F_conv_transposend(g);

    rename_F_convmode(g);

    rename_F_dropoutnd(g);

    eliminate_noop_math(g);

    fuse_expression(g);

    fuse_index_expression(g);

    //     dead_code_elimination(g);

    //     canonicalize(g);
}

} // namespace pnnx
