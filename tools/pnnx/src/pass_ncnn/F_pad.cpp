// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

#include <algorithm>

namespace pnnx {

namespace ncnn {

class F_pad_mixed_constant : public GraphRewriterPass
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

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
Crop                    op_0        1 1 input out0
Padding                 op_1        1 1 out0 out
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

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const std::vector<int>& pad = captured_params.at("pad").ai;

        bool has_negative = false;
        bool has_positive = false;
        for (int x : pad)
        {
            if (x < 0)
                has_negative = true;
            if (x > 0)
                has_positive = true;
        }

        if (!has_negative || !has_positive)
            return false;

        if (pad.size() % 2 != 0 || pad.size() > 6)
            return false;

        const std::vector<int>& input_shape = matched_operators.at("op_0")->inputs[0]->shape;
        if (input_shape.empty())
            return false;

        const int rank = (int)input_shape.size();
        const int padded_dims = (int)pad.size() / 2;
        if (rank > 4 || padded_dims > rank)
            return false;

        for (int i = 0; i < padded_dims; i++)
        {
            const int start = std::max(-pad[i * 2], 0);
            const int end = std::max(-pad[i * 2 + 1], 0);
            if (start == 0 && end == 0)
                continue;

            const int dim = input_shape[rank - 1 - i];
            if (dim <= 0)
                return false;

            if (start + end >= dim)
                return false;
        }

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& pad = captured_params.at("pad").ai;

        float pad_value = 0.f;
        if (captured_params.at("value").type == 2)
            pad_value = captured_params.at("value").i;
        if (captured_params.at("value").type == 3)
            pad_value = captured_params.at("value").f;

        std::vector<int> starts;
        std::vector<int> ends;
        std::vector<int> axes;

        const int padded_dims = (int)pad.size() / 2;
        for (int i = 0; i < padded_dims; i++)
        {
            const int start = std::max(-pad[i * 2], 0);
            const int end = std::min(pad[i * 2 + 1], 0);
            if (start == 0 && end == 0)
                continue;

            starts.push_back(start);
            ends.push_back(end);
            axes.push_back(-(i + 1));
        }

        Operator* crop = ops.at("op_0");
        crop->params["9"] = starts;
        crop->params["10"] = ends;
        crop->params["11"] = axes;

        Operator* padding = ops.at("op_1");

        if (pad.size() == 2)
        {
            padding->params["0"] = 0;
            padding->params["1"] = 0;
            padding->params["2"] = std::max(pad[0], 0);
            padding->params["3"] = std::max(pad[1], 0);
        }
        else if (pad.size() >= 4)
        {
            padding->params["0"] = std::max(pad[2], 0);
            padding->params["1"] = std::max(pad[3], 0);
            padding->params["2"] = std::max(pad[0], 0);
            padding->params["3"] = std::max(pad[1], 0);
        }
        if (pad.size() >= 6)
        {
            padding->params["7"] = std::max(pad[4], 0);
            padding->params["8"] = std::max(pad[5], 0);
        }

        padding->params["4"] = 0;
        padding->params["5"] = pad_value;
        padding->params["6"] = 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_pad_mixed_constant, 20)

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
F.pad                   op_0        1 1 input out pad=%pad mode=%mode value=None
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
        if (mode != "constant" && mode != "replicate" && mode != "reflect")
            fprintf(stderr, "unsupported F.pad mode %s\n", mode.c_str());

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
F.pad                   op_0        1 1 input out pad=%pad mode=%mode value=None
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
        const std::string& mode = captured_params.at("mode").s;
        if (mode != "constant" && mode != "replicate" && mode != "reflect")
            fprintf(stderr, "unsupported F.pad mode %s\n", mode.c_str());

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
