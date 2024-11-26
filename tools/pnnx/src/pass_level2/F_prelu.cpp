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

class F_prelu : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
aten::prelu             op_0        2 1 input weight out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.prelu";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_prelu, 100)

class F_prelu_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        // clang-format off
        // *INDENT-OFF*

        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight #weight=(?)f32
PRelu                   op_0        2 1 input weight out
pnnx.Output             output      1 0 out
)PNNXIR";

        // *INDENT-ON*
        // clang-format on
    }

    const char* type_str() const
    {
        return "F.prelu";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_prelu_onnx, 100)

class F_prelu_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          slope       0 1 weight @data
PRelu                   op_0        2 1 input weight out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          slope       0 1 weight @data=%slope.data
F.prelu                 prelu       2 1 input weight out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        Operator* op_slope = ops.at("slope");

        // hack slope shape
        int num_slope = op_slope->attrs["data"].shape[0];
        op_slope->attrs["data"].shape = {num_slope};

        op_slope->outputs[0]->shape = {num_slope};
        op_slope->outputs[0]->type = op_slope->attrs["data"].type;

        Operator* op_prelu = ops.at("prelu");
        op_prelu->inputnames = {"input", "weight"};
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_prelu_onnx_1, 100)

class F_prelu_onnx_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        // clang-format off
        // *INDENT-OFF*

        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight #weight=(?)f32
torch.unsqueeze         uqz         1 1 weight w2 dim=%dim
PRelu                   op_0        2 1 input w2 out
pnnx.Output             output      1 0 out
)PNNXIR";

        // *INDENT-ON*
        // clang-format on
    }

    const char* type_str() const
    {
        return "F.prelu";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("dim").type == 5)
        {
            // 1 2 ... N
            const std::vector<int>& dim = captured_params.at("dim").ai;
            for (int i = 0; i < (int)dim.size(); i++)
            {
                if (dim[i] != i + 1)
                    return false;
            }
        }
        else
        {
            int dim = captured_params.at("dim").i;
            if (dim != 1)
                return false;
        }

        return true;
    }

    void write(Operator* /*op*/, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_prelu_onnx_2, 100)

} // namespace pnnx
