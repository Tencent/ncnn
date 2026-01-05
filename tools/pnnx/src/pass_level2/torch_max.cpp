// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torch_max : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
prim::Constant          op_0        0 1 keepdim value=%keepdim
aten::max               op_1        3 2 input dim keepdim out indices
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.max";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        GraphRewriterPass::write(op, captured_params);

        // drop indices if not used
        if (op->outputs[1]->consumers.empty())
        {
            op->outputs[1]->producer = 0;
            op->outputs.resize(1);
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_max, 50)

class torch_max_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
aten::max               op_1        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.max";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_max_1, 50)

class torch_max_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
ReduceMax               op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.max";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.axes") != captured_params.end())
        {
            if (captured_params.at("op_0.axes").type != 5 || captured_params.at("op_0.axes").ai.size() != 1)
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.axes") != captured_params.end())
        {
            op->params["dim"] = captured_params.at("op_0.axes").ai[0];

            if (captured_params.find("op_0.keepdims") != captured_params.end())
            {
                op->params["keepdim"] = captured_params.at("op_0.keepdims").i ? true : false;
            }
            else
            {
                op->params["keepdim"] = true;
            }
        }
        else
        {
            // reduce all
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_max_onnx, 51)

class torch_max_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
ReduceMax               op_0        1 1 input out %*=%*
ArgMax                  op_1        1 1 input indices %*=%*
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.max";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.axes") == captured_params.end())
            return false;

        if (captured_params.find("op_0.keepdims") == captured_params.end())
            return false;

        if (captured_params.find("op_1.axis") == captured_params.end())
            return false;

        if (captured_params.find("op_1.keepdims") == captured_params.end())
            return false;

        if (captured_params.at("op_0.axes").type != 2 && (captured_params.at("op_0.axes").type != 5 || captured_params.at("op_0.axes").ai.size() != 1))
            return false;

        if (captured_params.at("op_1.axis").type != 2)
            return false;

        if (captured_params.at("op_0.axes").type == 2 && captured_params.at("op_0.axes").i != captured_params.at("op_1.axis").i)
            return false;

        if (captured_params.at("op_0.axes").type == 5 && captured_params.at("op_0.axes").ai[0] != captured_params.at("op_1.axis").i)
            return false;

        if (captured_params.at("op_0.keepdims").i != captured_params.at("op_1.keepdims").i)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("op_0.axes").type == 2)
            op->params["dim"] = captured_params.at("op_0.axes").i;
        else
            op->params["dim"] = captured_params.at("op_0.axes").ai[0];
        op->params["keepdim"] = captured_params.at("op_0.keepdims").i ? true : false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_max_onnx_1, 50)

class torch_max_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
tnn.ReduceMax           op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.max";
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

        if (dim.size() == 1)
        {
            op->params["dim"] = dim[0];
        }
        else
        {
            fprintf(stderr, "fallback to reduce max all\n");
        }
        op->params["keepdim"] = captured_params.at("op_0.arg0").i ? true : false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_max_tnn, 50)

} // namespace pnnx
