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

class F_layer_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
pnnx.Input              input_3     0 1 normalized_shape
prim::Constant          op_0        0 1 eps value=%eps
prim::Constant          op_1        0 1 cudnn_enabled value=*
aten::layer_norm        op_2        6 1 input normalized_shape weight bias eps cudnn_enabled out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.layer_norm";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_layer_norm, 10)

class F_layer_norm_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
LayerNormalization      op_0        3 3 input weight bias out Mean InvStdDev axis=%axis epsilon=%epsilon stash_type=%stash_type
pnnx.Output             output      3 0 out Mean InvStdDev
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.layer_norm";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int input_rank = op->inputs[0]->shape.size();

        int axis = captured_params.at("axis").i;
        if (axis < 0)
        {
            axis = input_rank + axis;
        }

        std::vector<int> normalized_shape;
        for (int i = axis; i < input_rank; i++)
        {
            normalized_shape.push_back(op->inputs[0]->shape[i]);
        }

        op->params["normalized_shape"] = normalized_shape;
        op->params["eps"] = captured_params.at("epsilon");

        // drop Mean and InvStdDev if not used
        if (op->outputs[1]->consumers.empty() && op->outputs[2]->consumers.empty())
        {
            op->outputs[1]->producer = 0;
            op->outputs[2]->producer = 0;
            op->outputs.resize(1);
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_layer_norm_1, 10)

} // namespace pnnx
