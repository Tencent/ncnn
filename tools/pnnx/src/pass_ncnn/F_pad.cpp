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

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class F_pad : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.pad                   op_0        1 1 input out pad=%pad mode=%mode value=%value
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Padding";
    }

    const char* name_str() const
    {
        return "pad";
    }

    void write(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs, Operator* op) const
    {
        const std::vector<int>& pad = captured_params.at("pad").ai;
        const std::string& mode = captured_params.at("mode").s;

        op->params["0"] = pad[0];
        op->params["1"] = pad[1];
        if (pad.size() >= 4)
        {
            op->params["2"] = pad[2];
            op->params["3"] = pad[3];
        }
        if (pad.size() >= 6)
        {
            op->params["7"] = pad[4];
            op->params["8"] = pad[5];
        }

        if (mode == "constant")
            op->params["4"] = 0;
        if (mode == "reflect")
            op->params["4"] = 2;
        if (mode == "replicate")
            op->params["4"] = 1;

        op->params["5"] = captured_params.at("value");
        op->params["6"] = 0; // per_channel_pad_data_size
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_pad, 20)

} // namespace ncnn

} // namespace pnnx
