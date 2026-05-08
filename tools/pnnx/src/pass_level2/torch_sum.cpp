// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_sum : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
prim::Constant          op_0        0 1 keepdim value=%keepdim
prim::Constant          op_1        0 1 dtype value=*
aten::sum               op_2        4 1 input dim keepdim dtype out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.sum";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_sum, 50)

class torch_sum_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 dtype value=*
aten::sum               op_1        2 1 input dtype out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.sum";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_sum_1, 50)

class torch_sum_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
ReduceSum               op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.sum";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.axes") != captured_params.end())
        {
            op->params["dim"] = captured_params.at("op_0.axes");
        }
        else
        {
            // reduce all
            const int input_rank = (int)op->inputs[0]->shape.size();
            std::vector<int> dim(input_rank);
            for (int i = 0; i < input_rank; i++)
            {
                dim[i] = i;
            }
            op->params["dim"] = dim;
        }

        if (captured_params.find("op_0.keepdims") != captured_params.end())
        {
            op->params["keepdim"] = captured_params.at("op_0.keepdims").i ? true : false;
        }
        else
        {
            op->params["keepdim"] = true;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_sum_onnx, 50)

class torch_sum_onnx_reducelogsum : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
ReduceLogSum            op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
torch.sum               sum         1 1 input a
aten::log               log         1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const
    {
        Operator* op = ops.at("sum");

        if (captured_params.find("op_0.axes") != captured_params.end())
        {
            op->params["dim"] = captured_params.at("op_0.axes");
        }
        else
        {
            // reduce all
            const int input_rank = (int)op->inputs[0]->shape.size();
            std::vector<int> dim(input_rank);
            for (int i = 0; i < input_rank; i++)
            {
                dim[i] = i;
            }
            op->params["dim"] = dim;
        }

        if (captured_params.find("op_0.keepdims") != captured_params.end())
        {
            op->params["keepdim"] = captured_params.at("op_0.keepdims").i ? true : false;
        }
        else
        {
            op->params["keepdim"] = true;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_sum_onnx_reducelogsum, 50)

class torch_sum_onnx_reducesumsquare : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
ReduceSumSquare         op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
aten::square            square      1 1 input a
torch.sum               sum         1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const
    {
        Operator* op = ops.at("sum");

        if (captured_params.find("op_0.axes") != captured_params.end())
        {
            op->params["dim"] = captured_params.at("op_0.axes");
        }
        else
        {
            // reduce all
            const int input_rank = (int)ops.at("square")->inputs[0]->shape.size();
            std::vector<int> dim(input_rank);
            for (int i = 0; i < input_rank; i++)
            {
                dim[i] = i;
            }
            op->params["dim"] = dim;
        }

        if (captured_params.find("op_0.keepdims") != captured_params.end())
        {
            op->params["keepdim"] = captured_params.at("op_0.keepdims").i ? true : false;
        }
        else
        {
            op->params["keepdim"] = true;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_sum_onnx_reducesumsquare, 50)

class torch_sum_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
tnn.ReduceSum           op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.sum";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        std::vector<int> dim;
        for (int i = 1;; i++)
        {
            if (captured_params.find("op_0.arg" + std::to_string(i)) == captured_params.end())
                break;

            dim.push_back(captured_params.at("op_0.arg" + std::to_string(i)).i);
        }

        op->params["dim"] = dim;
        op->params["keepdim"] = captured_params.at("op_0.arg0").i ? true : false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_sum_tnn, 50)

} // namespace pnnx
