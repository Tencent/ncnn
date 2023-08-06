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

class Tensor_to : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 dtype value=%dtype
prim::Constant          op_1        0 1 non_blocking value=*
prim::Constant          op_2        0 1 copy value=%copy
prim::Constant          op_3        0 1 memory_format value=%memory_format
aten::to                op_4        5 1 input dtype non_blocking copy memory_format out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.to";
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

        op->params["copy"] = captured_params.at("copy");

        if (captured_params.at("memory_format").type == 2)
        {
            if (captured_params.at("memory_format").i == 0)
                op->params["memory_format"] = "torch.contiguous_format";
            if (captured_params.at("memory_format").i == 1)
                op->params["memory_format"] = "torch.preserve_format";
            if (captured_params.at("memory_format").i == 2)
                op->params["memory_format"] = "torch.channels_last";
        }
    }
};

class Tensor_to_1 : public Tensor_to
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 device value=*
prim::Constant          op_1        0 1 dtype value=%dtype
prim::Constant          op_2        0 1 non_blocking value=*
prim::Constant          op_3        0 1 copy value=%copy
prim::Constant          op_4        0 1 memory_format value=%memory_format
aten::to                op_5        6 1 input device dtype non_blocking copy memory_format out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class Tensor_to_2 : public Tensor_to
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
10 9
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 dtype value=%dtype
prim::Constant          op_1        0 1 layout value=*
prim::Constant          op_2        0 1 device value=*
prim::Constant          op_3        0 1 pin_memory value=*
prim::Constant          op_4        0 1 non_blocking value=*
prim::Constant          op_5        0 1 copy value=%copy
prim::Constant          op_6        0 1 memory_format value=%memory_format
aten::to                op_7        8 1 input dtype layout device pin_memory non_blocking copy memory_format out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_to, 20)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_to_1, 20)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_to_2, 20)

} // namespace pnnx
