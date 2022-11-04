// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

class F_fold : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 output_size
pnnx.Input              input_2     0 1 kernel_size
pnnx.Input              input_3     0 1 dilation
pnnx.Input              input_4     0 1 padding
pnnx.Input              input_5     0 1 stride
aten::col2im            op_0        6 1 input output_size kernel_size dilation padding stride out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.fold";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_fold, 10)

} // namespace pnnx
