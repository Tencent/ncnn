// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_flatten : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 start_dim
pnnx.Input              input_2     0 1 end_dim
aten::flatten           op_0        3 1 input start_dim end_dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.flatten";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_flatten, 60)

class torch_flatten_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
Flatten                 op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.flatten";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* op = matched_operators.at("op_0");

        const int input_rank = op->inputs[0]->shape.size();
        if (input_rank == 0)
            return false;

        int axis = 1;
        if (captured_params.find("op_0.axis") != captured_params.end())
            axis = captured_params.at("op_0.axis").i;

        if (axis < 0)
            axis += input_rank;

        return axis > 1 && axis == input_rank - 1;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int axis = 1;
        if (captured_params.find("op_0.axis") != captured_params.end())
            axis = captured_params.at("op_0.axis").i;

        const int input_rank = op->inputs[0]->shape.size();
        if (axis < 0)
            axis += input_rank;

        op->params["start_dim"] = 0;
        op->params["end_dim"] = axis - 1;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_flatten_onnx_1, 59)

class torch_flatten_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
Flatten                 op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.reshape";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int axis = 1;
        if (captured_params.find("op_0.axis") != captured_params.end())
            axis = captured_params.at("op_0.axis").i;

        const int input_rank = op->inputs[0]->shape.size();
        if (axis < 0 && input_rank != 0)
            axis += input_rank;

        if (axis == 1)
        {
            op->params["shape"] = std::vector<int>{0, -1};
            return;
        }

        if (!op->outputs[0]->shape.empty())
        {
            int dynamic_count = 0;
            for (int x : op->outputs[0]->shape)
            {
                if (x == -1)
                    dynamic_count++;
            }

            if (dynamic_count <= 1)
            {
                op->params["shape"] = op->outputs[0]->shape;
                return;
            }
        }

        if (input_rank != 0 && 0 <= axis && axis <= input_rank)
        {
            int size0 = 1;
            for (int i = 0; i < axis; i++)
            {
                if (op->inputs[0]->shape[i] == -1)
                {
                    size0 = -1;
                    break;
                }
                size0 *= op->inputs[0]->shape[i];
            }

            int size1 = 1;
            for (int i = axis; i < input_rank; i++)
            {
                if (op->inputs[0]->shape[i] == -1)
                {
                    size1 = -1;
                    break;
                }
                size1 *= op->inputs[0]->shape[i];
            }

            if (size0 == -1 && size1 == -1)
            {
                fprintf(stderr, "onnx Flatten with dynamic shape on both sides of axis %d is not supported yet, fallback to flatten all\n", axis);
                op->params["shape"] = std::vector<int>{-1};
                return;
            }

            op->params["shape"] = std::vector<int>{size0, size1};
            return;
        }

        fprintf(stderr, "onnx Flatten with unknown output shape is not supported yet, fallback to flatten all\n");
        op->params["shape"] = std::vector<int>{-1};
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_flatten_onnx, 60)

class torch_flatten_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
tnn.Flatten             op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.flatten";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        // TNN Flatten has arg0 parameter representing the axis to flatten from
        if (captured_params.find("op_0.arg0") != captured_params.end())
        {
            op->params["start_dim"] = captured_params.at("op_0.arg0");
        }
        else
        {
            op->params["start_dim"] = 1;
        }
        op->params["end_dim"] = -1;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_flatten_tnn, 60)

} // namespace pnnx
