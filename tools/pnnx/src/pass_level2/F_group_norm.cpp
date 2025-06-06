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

class F_group_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_1     0 1 input
pnnx.Input              input_2     0 1 weight
pnnx.Input              input_3     0 1 bias
prim::Constant          op_0        0 1 num_groups value=%num_groups
prim::Constant          op_1        0 1 eps value=%eps
prim::Constant          op_2        0 1 cudnn_enabled value=*
aten::group_norm        op_3        6 1 input num_groups weight bias eps cudnn_enabled out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.group_norm";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_group_norm, 130)

class F_group_norm_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
Reshape                 op_0        1 1 input r1 allowzero=0 shape=(0,%num_groups,-1)
pnnx.Attribute          op_1        0 1 ones @data
pnnx.Attribute          op_2        0 1 zeros @data
InstanceNormalization   op_3        3 1 r1 ones zeros in epsilon=%epsilon
Reshape                 op_4        1 1 in out allowzero=0 shape=%shape
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.group_norm";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const Operator* op_reshape = matched_operators.at("op_0");
        const std::vector<int>& inputshape = op_reshape->inputs[0]->shape;
        if (inputshape != captured_params.at("shape").ai)
            return false;

        const int num_groups = captured_params.at("num_groups").i;

        const Attribute& ones = captured_attrs.at("op_1.data");
        const Attribute& zeros = captured_attrs.at("op_2.data");

        if (ones.shape.size() != 1 || ones.shape[0] != num_groups)
            return false;
        if (zeros.shape.size() != 1 || zeros.shape[0] != num_groups)
            return false;

        for (auto x : ones.get_float32_data())
        {
            if (x != 1.f)
                return false;
        }

        for (auto x : zeros.get_float32_data())
        {
            if (x != 0.f)
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["num_groups"] = captured_params.at("num_groups");
        op->params["eps"] = captured_params.at("epsilon");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_group_norm_onnx, 130)

class F_group_norm_onnx_1 : public F_group_norm_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
Reshape                 op_0        1 1 input r1 shape=(0,%num_groups,-1)
pnnx.Attribute          op_1        0 1 ones @data
pnnx.Attribute          op_2        0 1 zeros @data
InstanceNormalization   op_3        3 1 r1 ones zeros in epsilon=%epsilon
Reshape                 op_4        1 1 in out shape=%shape
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_group_norm_onnx_1, 130)

} // namespace pnnx
