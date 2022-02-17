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
F.pad                   op_0        1 1 input out pad=%pad mode=constant value=%value
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

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& pad = captured_params.at("pad").ai;
        for (int x : pad)
        {
            if (x < 0)
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& pad = captured_params.at("pad").ai;

        float pad_value = 0.f;
        if (captured_params.at("value").type == 2)
            pad_value = captured_params.at("value").i;
        if (captured_params.at("value").type == 3)
            pad_value = captured_params.at("value").f;

        if (pad.size() == 2)
        {
            op->params["0"] = 0;
            op->params["1"] = 0;
            op->params["2"] = pad[0];
            op->params["3"] = pad[1];
        }
        else if (pad.size() >= 4)
        {
            op->params["0"] = pad[2];
            op->params["1"] = pad[3];
            op->params["2"] = pad[0];
            op->params["3"] = pad[1];
        }
        if (pad.size() >= 6)
        {
            op->params["7"] = pad[4];
            op->params["8"] = pad[5];
        }

        op->params["4"] = 0; // constant
        op->params["5"] = pad_value;
        op->params["6"] = 0; // per_channel_pad_data_size
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_pad, 20)

class F_pad_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.pad                   op_0        1 1 input out pad=%pad mode=%mode
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

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& pad = captured_params.at("pad").ai;
        for (int x : pad)
        {
            if (x < 0)
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& pad = captured_params.at("pad").ai;
        const std::string& mode = captured_params.at("mode").s;

        if (pad.size() == 2)
        {
            op->params["0"] = 0;
            op->params["1"] = 0;
            op->params["2"] = pad[0];
            op->params["3"] = pad[1];
        }
        else if (pad.size() >= 4)
        {
            op->params["0"] = pad[2];
            op->params["1"] = pad[3];
            op->params["2"] = pad[0];
            op->params["3"] = pad[1];
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

        op->params["5"] = 0; // value
        op->params["6"] = 0; // per_channel_pad_data_size
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_pad_1, 20)

class F_pad_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.pad                   op_0        1 1 input out pad=%pad mode=constant value=%value
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Crop";
    }

    const char* name_str() const
    {
        return "pad";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& pad = captured_params.at("pad").ai;
        for (int x : pad)
        {
            if (x > 0)
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& pad = captured_params.at("pad").ai;

        std::vector<int> starts;
        std::vector<int> ends;
        std::vector<int> axes;

        if (pad.size() == 2)
        {
            starts = {-pad[0]};
            ends = {pad[1]};
            axes = {-1};
        }
        else if (pad.size() == 4)
        {
            starts = {-pad[2], -pad[0]};
            ends = {pad[3], pad[1]};
            axes = {-2, -1};
        }
        else if (pad.size() == 6)
        {
            starts = {-pad[4], -pad[2], -pad[0]};
            ends = {pad[5], pad[3], pad[1]};
            axes = {-3, -2, -1};
        }

        op->params["9"] = starts;
        op->params["10"] = ends;
        op->params["11"] = axes;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_pad_2, 20)

class F_pad_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.pad                   op_0        1 1 input out pad=%pad mode=%mode
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

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& pad = captured_params.at("pad").ai;
        for (int x : pad)
        {
            if (x < 0)
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& pad = captured_params.at("pad").ai;

        std::vector<int> starts;
        std::vector<int> ends;
        std::vector<int> axes;

        if (pad.size() == 2)
        {
            starts = {-pad[0]};
            ends = {pad[1]};
            axes = {-1};
        }
        else if (pad.size() == 4)
        {
            starts = {-pad[2], -pad[0]};
            ends = {pad[3], pad[1]};
            axes = {-2, -1};
        }
        else if (pad.size() == 6)
        {
            starts = {-pad[4], -pad[2], -pad[0]};
            ends = {pad[5], pad[3], pad[1]};
            axes = {-3, -2, -1};
        }

        op->params["9"] = starts;
        op->params["10"] = ends;
        op->params["11"] = axes;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_pad_3, 20)

} // namespace ncnn

} // namespace pnnx
