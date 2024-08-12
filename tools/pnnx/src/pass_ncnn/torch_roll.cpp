// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_roll : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.roll              op_0        1 1 input out dims=%dims shifts=%shifts
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
Slice                   slice       1 2 input a b
Concat                  concat      2 1 b a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("dims").type != 5)
            return false;

        if (captured_params.at("dims").ai.size() != 1)
            return false;

        if (captured_params.at("shifts").type != 5)
            return false;

        if (captured_params.at("shifts").ai.size() != 1)
            return false;

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const Operand* in = ops.at("slice")->inputs[0];

        const int batch_index = in->params.at("__batch_index").i;

        int axis = captured_params.at("dims").ai[0];
        if (axis == batch_index)
        {
            fprintf(stderr, "roll along batch axis %d is not supported\n", batch_index);
        }

        if (axis < 0)
        {
            int input_rank = in->shape.size();
            axis = input_rank + axis;
        }

        if (axis > batch_index)
            axis -= 1;

        ops.at("slice")->params["1"] = axis;

        ops.at("concat")->params["0"] = axis;

        const int shift = captured_params.at("shifts").ai[0];
        ops.at("slice")->params["2"] = std::vector<int>{-shift};
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_roll, 20)

} // namespace ncnn

} // namespace pnnx
