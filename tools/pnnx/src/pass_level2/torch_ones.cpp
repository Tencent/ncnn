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

class torch_ones : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 size
prim::Constant          op_0        0 1 dtype value=%dtype
prim::Constant          op_1        0 1 layout value=*
prim::Constant          op_2        0 1 device value=*
prim::Constant          op_3        0 1 requires_grad value=*
aten::ones              op_4        5 1 size dtype layout device requires_grad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.ones";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("dtype").i == 0) op->params["dtype"] = "torch.uint8";
        if (captured_params.at("dtype").i == 1) op->params["dtype"] = "torch.int8";
        if (captured_params.at("dtype").i == 2) op->params["dtype"] = "torch.short";
        if (captured_params.at("dtype").i == 3) op->params["dtype"] = "torch.int";
        if (captured_params.at("dtype").i == 4) op->params["dtype"] = "torch.long";
        if (captured_params.at("dtype").i == 5) op->params["dtype"] = "torch.half";
        if (captured_params.at("dtype").i == 6) op->params["dtype"] = "torch.float";
        if (captured_params.at("dtype").i == 7) op->params["dtype"] = "torch.double";
        if (captured_params.at("dtype").i == 8) op->params["dtype"] = "torch.complex32";
        if (captured_params.at("dtype").i == 9) op->params["dtype"] = "torch.complex64";
        if (captured_params.at("dtype").i == 10) op->params["dtype"] = "torch.complex128";
        if (captured_params.at("dtype").i == 11) op->params["dtype"] = "torch.bool";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_ones, 20)

} // namespace pnnx
