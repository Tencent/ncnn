// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

class torch_istft : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
12 11
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 center
pnnx.Input              input_6     0 1 normalized
pnnx.Input              input_7     0 1 onesided
pnnx.Input              input_8     0 1 length
pnnx.Input              input_9     0 1 return_complex
aten::istft             op_0        10 1 input n_fft hop_length win_length window center normalized onesided length return_complex out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.istft";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_istft, 20)

} // namespace pnnx
