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

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_addmm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 mat1
pnnx.Attribute          op_bias     0 1 bias @qwq
pnnx.Attribute          op_weight   0 1 weight @qwq
torch.addmm             op_0        3 1 bias mat1 weight out alpha=%alpha beta=%beta
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "InnerProduct";
    }

    const char* name_str() const
    {
        return "addmm";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        float alpha = 1.f;
        float beta = 1.f;

        if (captured_params.at("alpha").type == 2)
        {
            alpha = captured_params.at("alpha").i;
        }
        if (captured_params.at("alpha").type == 3)
        {
            alpha = captured_params.at("alpha").f;
        }

        if (captured_params.at("beta").type == 2)
        {
            beta = captured_params.at("beta").i;
        }
        if (captured_params.at("beta").type == 3)
        {
            beta = captured_params.at("beta").f;
        }

        if (alpha != 1.f || beta != 1.f)
            return false;

        Attribute weight;
        Attribute bias;
        for (const auto& x : captured_attrs)
        {
            if (x.first.substr(0, 10) == "op_weight.")
                weight = x.second;
            if (x.first.substr(0, 8) == "op_bias.")
                bias = x.second;
        }

        if (weight.shape.size() != 2 || bias.shape.size() != 1)
            return false;

        if (weight.shape[1] != bias.shape[0])
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        Attribute weight;
        Attribute bias;
        for (const auto& x : captured_attrs)
        {
            if (x.first.substr(0, 10) == "op_weight.")
                weight = x.second;
            if (x.first.substr(0, 8) == "op_bias.")
                bias = x.second;
        }

        // transpose weight inch-outch to outch-inch
        const int inch = weight.shape[0];
        const int outch = weight.shape[1];
        std::vector<float> new_weight;
        {
            const float* w = (const float*)weight.data.data();

            new_weight.resize(outch * inch);
            float* w2 = (float*)new_weight.data();

            // reorder weight from inch-outch to outch-inch
            for (int i = 0; i < outch; i++)
            {
                for (int j = 0; j < inch; j++)
                {
                    w2[i * inch + j] = w[j * outch + i];
                }
            }
        }

        op->params["0"] = outch;
        op->params["1"] = 1;
        op->params["2"] = (int)(weight.data.size() / sizeof(float));

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = Attribute({outch, inch}, new_weight);
        op->attrs["2"] = bias;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_addmm, 20)

class torch_addmm_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 mat1
pnnx.Input              input_2     0 1 mat2
torch.addmm             op_0        3 1 input mat1 mat2 out alpha=%alpha beta=%beta
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Gemm";
    }

    const char* name_str() const
    {
        return "addmm";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        std::swap(op->inputs[0], op->inputs[1]);
        std::swap(op->inputs[1], op->inputs[2]);

        float alpha = 1.f;
        float beta = 1.f;

        if (captured_params.at("alpha").type == 2)
        {
            alpha = captured_params.at("alpha").i;
        }
        if (captured_params.at("alpha").type == 3)
        {
            alpha = captured_params.at("alpha").f;
        }

        if (captured_params.at("beta").type == 2)
        {
            beta = captured_params.at("beta").i;
        }
        if (captured_params.at("beta").type == 3)
        {
            beta = captured_params.at("beta").f;
        }

        op->params["0"] = alpha;
        op->params["1"] = beta / alpha;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_addmm_1, 22)

} // namespace ncnn

} // namespace pnnx
