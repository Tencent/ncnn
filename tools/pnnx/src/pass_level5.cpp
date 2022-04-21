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

#include "pass_level5.h"

#include "pass_level5/fold_constants.h"
#include "pass_level5/eliminate_dropout.h"
#include "pass_level5/eliminate_identity_operator.h"
#include "pass_level5/eliminate_noop_expression.h"
#include "pass_level5/eliminate_noop_pad.h"
#include "pass_level5/eliminate_slice.h"
#include "pass_level5/eliminate_view_reshape.h"
#include "pass_level5/eval_expression.h"
#include "pass_level5/fuse_channel_shuffle.h"
#include "pass_level5/fuse_constant_expression.h"
#include "pass_level5/fuse_conv1d_batchnorm1d.h"
#include "pass_level5/fuse_conv2d_batchnorm2d.h"
#include "pass_level5/fuse_convtranspose1d_batchnorm1d.h"
#include "pass_level5/fuse_convtranspose2d_batchnorm2d.h"
#include "pass_level5/fuse_contiguous_view.h"
#include "pass_level5/fuse_linear_batchnorm1d.h"
#include "pass_level5/fuse_select_to_unbind.h"
#include "pass_level5/fuse_slice_indices.h"
#include "pass_level4/dead_code_elimination.h"
#include "pass_level4/canonicalize.h"
#include "pass_level3/fuse_index_expression.h"

namespace pnnx {

void pass_level5(Graph& g, const std::map<std::string, Attribute>& foldable_constants)
{
    eval_expression(g);

    fuse_constant_expression(g);

    eliminate_noop_expression(g);

    eliminate_slice(g);

    fuse_slice_indices(g);

    eliminate_identity_operator(g);

    fuse_select_to_unbind(g);

    fuse_conv1d_batchnorm1d(g);

    fuse_conv2d_batchnorm2d(g);

    fuse_convtranspose1d_batchnorm1d(g);

    fuse_convtranspose2d_batchnorm2d(g);

    fuse_linear_batchnorm1d(g);

    eliminate_noop_pad(g);

    eliminate_dropout(g);

    fuse_contiguous_view(g);

    eliminate_view_reshape(g);

    fuse_channel_shuffle(g);

    fold_constants(g, foldable_constants);

    fuse_index_expression(g);

    dead_code_elimination(g);

    canonicalize(g);
}

} // namespace pnnx
