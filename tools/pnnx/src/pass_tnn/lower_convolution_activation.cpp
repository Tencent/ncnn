// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lower_convolution_activation.h"

#include "pass_level2.h"

namespace pnnx {

namespace tnn2pnnx {

class lower_convolution_activation_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
tnn.Convolution         op_0        3 1 input weight bias out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        if (this->activation == 1)
        {
            return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
tnn.Convolution         conv2d      3 1 input weight bias a
aten::relu              relu        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
        }
        else if (this->activation == 2)
        {
            return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
tnn.Convolution         conv2d      3 1 input weight bias a
aten::relu6             relu6       1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
        }
        else // if (this->activation == 256)
        {
            return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
tnn.Convolution         conv2d      3 1 input weight bias a
aten::silu              silu        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
        }
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.arg13") == captured_params.end())
            return false;

        this->activation = captured_params.at("op_0.arg13").i;
        return activation != 0;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const
    {
        for (int i = 0; i < 13; i++)
        {
            std::string argN = std::string("arg") + std::to_string(i);
            ops.at("conv2d")->params[argN] = captured_params.at("op_0." + argN);
        }

        ops.at("conv2d")->params["arg13"] = 0;
    }

protected:
    mutable int activation;
};

class lower_convolution_activation_pass_1 : public lower_convolution_activation_pass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
tnn.Convolution         op_0        2 1 input weight out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        if (this->activation == 1)
        {
            return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
tnn.Convolution         conv2d      2 1 input weight a
aten::relu              relu        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
        }
        else if (this->activation == 2)
        {
            return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
tnn.Convolution         conv2d      2 1 input weight a
aten::relu6             relu6       1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
        }
        else // if (this->activation == 256)
        {
            return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
tnn.Convolution         conv2d      2 1 input weight a
aten::silu              silu        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
        }
    }
};

class lower_convolution1d_activation_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
tnn.Convolution1D       op_0        3 1 input weight bias out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        if (this->activation == 1)
        {
            return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
tnn.Convolution1D       conv1d      3 1 input weight bias a
aten::relu              relu        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
        }
        else if (this->activation == 2)
        {
            return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
tnn.Convolution1D       conv1d      3 1 input weight bias a
aten::relu6             relu6       1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
        }
        else // if (this->activation == 256)
        {
            return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
tnn.Convolution1D       conv1d      3 1 input weight bias a
aten::silu              silu        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
        }
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.arg9") == captured_params.end())
            return false;

        this->activation = captured_params.at("op_0.arg9").i;
        return activation != 0;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const
    {
        for (int i = 0; i < 9; i++)
        {
            std::string argN = std::string("arg") + std::to_string(i);
            ops.at("conv1d")->params[argN] = captured_params.at("op_0." + argN);
        }

        ops.at("conv1d")->params["arg9"] = 0;
    }

protected:
    mutable int activation;
};

class lower_convolution1d_activation_pass_1 : public lower_convolution1d_activation_pass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
tnn.Convolution1D       op_0        2 1 input weight out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        if (this->activation == 1)
        {
            return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
tnn.Convolution1D       conv1d      2 1 input weight a
aten::relu              relu        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
        }
        else if (this->activation == 2)
        {
            return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
tnn.Convolution1D       conv1d      2 1 input weight a
aten::relu6             relu6       1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
        }
        else // if (this->activation == 256)
        {
            return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
tnn.Convolution1D       conv1d      2 1 input weight a
aten::silu              silu        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
        }
    }
};

void lower_convolution_activation(Graph& graph)
{
    lower_convolution_activation_pass a;
    lower_convolution_activation_pass_1 a1;
    lower_convolution1d_activation_pass b;
    lower_convolution1d_activation_pass_1 b1;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &a1, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &b1, opindex);
}

} // namespace tnn2pnnx

} // namespace pnnx
