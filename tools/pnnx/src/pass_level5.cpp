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

#include "pass_level5/eliminate_dropout.h"
#include "pass_level5/eliminate_slice.h"
#include "pass_level5/eliminate_view_reshape.h"
#include "pass_level5/eval_expression.h"
#include "pass_level5/fuse_channel_shuffle.h"
#include "pass_level5/fuse_constant_expression.h"
#include "pass_level5/fuse_conv2d_batchnorm2d.h"
#include "pass_level5/fuse_convtranspose2d_batchnorm2d.h"
#include "pass_level5/fuse_contiguous_view.h"
#include "pass_level5/fuse_linear_batchnorm1d.h"
#include "pass_level5/fuse_slice_indices.h"
#include "pass_level4/dead_code_elimination.h"
#include "pass_level4/canonicalize.h"

namespace pnnx {

void pass_level5(Graph& g)
{
    eval_expression(g);

    fuse_constant_expression(g);

    eliminate_slice(g);

    fuse_slice_indices(g);

    fuse_conv2d_batchnorm2d(g);

    fuse_convtranspose2d_batchnorm2d(g);

    fuse_linear_batchnorm1d(g);

    fuse_contiguous_view(g);

    eliminate_view_reshape(g);

    eliminate_dropout(g);

    fuse_channel_shuffle(g);

    dead_code_elimination(g);

    canonicalize(g);
}

} // namespace pnnx
