// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class Tensor_slice : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
pnnx.Input              input_2     0 1 start
pnnx.Input              input_3     0 1 end
pnnx.Input              input_4     0 1 step
aten::slice             op_0        5 1 input dim start end step out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_slice, 70)

class Tensor_slice_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Slice                   op_0        1 1 input out axes=%axes starts=%starts ends=%ends
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("axes").type == 2)
        {
            op->params["dim"] = captured_params.at("axes");
            op->params["start"] = captured_params.at("starts");
            op->params["end"] = captured_params.at("ends");
            op->params["step"] = 1;
        }
        else // if (captured_params.at("axes").type == 5)
        {
            const std::vector<int>& axes = captured_params.at("axes").ai;
            const std::vector<int>& starts = captured_params.at("starts").ai;
            const std::vector<int>& ends = captured_params.at("ends").ai;

            if (axes.size() == 1)
            {
                op->params["dim"] = axes[0];
                op->params["start"] = starts[0];
                op->params["end"] = ends[0];
                op->params["step"] = 1;
            }
            else
            {
                op->params["dims"] = axes;
                op->params["starts"] = starts;
                op->params["ends"] = ends;
                op->params["steps"] = std::vector<int>(axes.size(), 1);
                op->params["selects"] = std::vector<int>(axes.size(), INT_MAX);
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_slice_onnx, 70)

class Tensor_slice_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Slice                   op_0        1 1 input out axes=%axes starts=%starts ends=%ends steps=%steps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("axes").type == 2)
        {
            op->params["dim"] = captured_params.at("axes");
            op->params["start"] = captured_params.at("starts");
            op->params["end"] = captured_params.at("ends");
            op->params["step"] = captured_params.at("steps");
        }
        else // if (captured_params.at("axes").type == 5)
        {
            const std::vector<int>& axes = captured_params.at("axes").ai;
            const std::vector<int>& starts = captured_params.at("starts").ai;
            const std::vector<int>& ends = captured_params.at("ends").ai;
            const std::vector<int>& steps = captured_params.at("steps").ai;

            if (axes.size() == 1)
            {
                op->params["dim"] = axes[0];
                op->params["start"] = starts[0];
                op->params["end"] = ends[0];
                op->params["step"] = steps[0];
            }
            else
            {
                op->params["dims"] = axes;
                op->params["starts"] = starts;
                op->params["ends"] = ends;
                op->params["steps"] = steps;
                op->params["selects"] = std::vector<int>(axes.size(), INT_MAX);
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_slice_onnx_1, 70)

class Tensor_slice_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
tnn.StridedSliceV2      op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.slice";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int nbegins = captured_params.at("op_0.arg0").i;
        std::vector<int> begins(nbegins);
        for (int i = 0; i < nbegins; i++)
        {
            begins[i] = captured_params.at("op_0.arg" + std::to_string(i + 1)).i;
        }
        const int nends = captured_params.at("op_0.arg" + std::to_string(nbegins + 1)).i;
        std::vector<int> ends(nends);
        for (int i = 0; i < nends; i++)
        {
            ends[i] = captured_params.at("op_0.arg" + std::to_string(i + 2 + nbegins)).i;
        }
        const int naxes = captured_params.at("op_0.arg" + std::to_string(nbegins + nends + 2)).i;
        std::vector<int> axes(naxes);
        for (int i = 0; i < naxes; i++)
        {
            axes[i] = captured_params.at("op_0.arg" + std::to_string(i + 3 + nbegins + nends)).i;
        }

        std::vector<int> strides;
        if (captured_params.find("op_0.arg" + std::to_string(nbegins + nends + naxes + 3)) != captured_params.end())
        {
            const int nstrides = captured_params.at("op_0.arg" + std::to_string(nbegins + nends + naxes + 3)).i;
            strides.resize(nstrides);
            for (int i = 0; i < nstrides; i++)
            {
                strides[i] = captured_params.at("op_0.arg" + std::to_string(i + 4 + nbegins + nends + naxes)).i;
            }
        }
        else
        {
            strides.resize(naxes, 1);
        }

        if (axes.size() == 1)
        {
            op->params["dim"] = axes[0];
            op->params["start"] = begins[0];
            op->params["end"] = ends[0];
            op->params["step"] = strides[0];
        }
        else
        {
            op->params["dims"] = axes;
            op->params["starts"] = begins;
            op->params["ends"] = ends;
            op->params["steps"] = strides;
            op->params["selects"] = std::vector<int>(axes.size(), INT_MAX);
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_slice_tnn, 70)

} // namespace pnnx
