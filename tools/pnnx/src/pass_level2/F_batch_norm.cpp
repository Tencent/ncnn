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

#include "pass_level2.h"

namespace pnnx {

class F_batch_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
11 10
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 running_mean
pnnx.Input              input_2     0 1 running_var
pnnx.Input              input_3     0 1 weight
pnnx.Input              input_4     0 1 bias
prim::Constant          op_0        0 1 training value=*
prim::Constant          op_1        0 1 momentum value=*
prim::Constant          op_2        0 1 eps value=%eps
prim::Constant          op_3        0 1 cudnn_enabled value=*
aten::batch_norm        op_4        9 1 input weight bias running_mean running_var training momentum eps cudnn_enabled out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.batch_norm";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_batch_norm, 10)

} // namespace pnnx
