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

#include "pass_level2.h"

namespace pnnx {

class nn_RNN_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
RNN                     rnn         3 1 input W R out %*=%*
Squeeze                 sqz         1 1 out out1 axes=%axes
pnnx.Output             output      1 0 out1
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.RNN";
    }

    const char* name_str() const
    {
        return "rnn";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        if (captured_params.find("rnn.hidden_size") == captured_params.end())
            return false;

        const int hidden_size = captured_params.at("rnn.hidden_size").i;

        std::string direction = "forward";
        if (captured_params.find("rnn.direction") != captured_params.end())
        {
            direction = captured_params.at("rnn.direction").s;
        }

        if (direction != "forward" && direction != "bidirectional")
            return false;

        const int num_directions = direction == "bidirectional" ? 2 : 1;

        if (captured_params.find("rnn.activations") != captured_params.end())
        {
            const std::vector<std::string>& acts = captured_params.at("rnn.activations").as;

            if (num_directions == 1)
            {
                if (acts != std::vector<std::string>{"Tanh"} && acts != std::vector<std::string>{"Relu"})
                    return false;
            }
            else // if (num_directions == 2)
            {
                if (acts != std::vector<std::string>{"Tanh", "Tanh"} && acts != std::vector<std::string>{"Relu", "Relu"})
                    return false;
            }
        }

        if (captured_params.find("axes") != captured_params.end())
        {
            if (captured_params.at("axes").type == 2 && captured_params.at("axes").i != 1)
                return false;

            if (captured_params.at("axes").type == 5 && captured_params.at("axes").ai != std::vector<int>{1})
                return false;
        }

        const auto& W = captured_attrs.at("W.data");
        const auto& R = captured_attrs.at("R.data");

        if (W.shape.size() != 3 || W.shape[0] != num_directions || W.shape[1] != hidden_size)
            return false;

        if (R.shape.size() != 3 || R.shape[0] != num_directions || R.shape[1] != hidden_size || R.shape[2] != hidden_size)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        std::string direction = "forward";
        if (captured_params.find("rnn.direction") != captured_params.end())
        {
            direction = captured_params.at("rnn.direction").s;
        }

        std::string act = "Tanh";
        if (captured_params.find("rnn.activations") != captured_params.end())
        {
            act = captured_params.at("rnn.activations").as[0];
        }

        const auto& W = captured_attrs.at("W.data");
        const auto& R = captured_attrs.at("R.data");

        bool batch_first = false;
        if (captured_params.find("rnn.layout") != captured_params.end())
        {
            const int layout = captured_params.at("rnn.layout").i;
            batch_first = layout == 1;
        }

        const int hidden_size = captured_params.at("rnn.hidden_size").i;

        const int input_size = W.shape[2];

        op->params["input_size"] = input_size;
        op->params["hidden_size"] = hidden_size;
        op->params["num_layers"] = 1;
        op->params["nonlinearity"] = act == "Relu" ? "relu" : "tanh";
        op->params["bias"] = false;
        op->params["batch_first"] = batch_first;
        op->params["bidirectional"] = direction == "bidirectional" ? true : false;

        // split W R
        auto W_data = W.get_float32_data();
        auto R_data = R.get_float32_data();

        if (direction == "bidirectional")
        {
            op->attrs["weight_ih_l0"] = Attribute({hidden_size, input_size}, std::vector<float>(&W_data[0], &W_data[hidden_size * input_size]));
            op->attrs["weight_hh_l0"] = Attribute({hidden_size, hidden_size}, std::vector<float>(&R_data[0], &R_data[hidden_size * hidden_size]));

            op->attrs["weight_ih_l0_reverse"] = Attribute({hidden_size, input_size}, std::vector<float>(&W_data[hidden_size * input_size], &W_data[hidden_size * input_size * 2]));
            op->attrs["weight_hh_l0_reverse"] = Attribute({hidden_size, hidden_size}, std::vector<float>(&R_data[hidden_size * hidden_size], &R_data[hidden_size * hidden_size * 2]));
        }
        else
        {
            op->attrs["weight_ih_l0"] = Attribute({hidden_size, input_size}, W_data);
            op->attrs["weight_hh_l0"] = Attribute({hidden_size, hidden_size}, R_data);
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_RNN_onnx, 10)

class nn_RNN_onnx_B : public nn_RNN_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
pnnx.Attribute          B           0 1 B @data
RNN                     rnn         4 1 input W R B out %*=%*
Squeeze                 sqz         1 1 out out1 axes=%axes
pnnx.Output             output      1 0 out1
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        if (!nn_RNN_onnx::match(captured_params, captured_attrs))
            return false;

        const int hidden_size = captured_params.at("rnn.hidden_size").i;

        std::string direction = "forward";
        if (captured_params.find("rnn.direction") != captured_params.end())
        {
            direction = captured_params.at("rnn.direction").s;
        }

        const int num_directions = direction == "bidirectional" ? 2 : 1;

        const auto& B = captured_attrs.at("B.data");

        if (B.shape.size() != 2 || B.shape[0] != num_directions || B.shape[1] != 2 * hidden_size)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        nn_RNN_onnx::write(op, captured_params, captured_attrs);

        const auto& B = captured_attrs.at("B.data");

        bool has_bias = false;
        for (auto b : B.get_float32_data())
        {
            if (b != 0.f)
            {
                has_bias = true;
                break;
            }
        }

        op->params["bias"] = has_bias;

        if (has_bias)
        {
            // split B
            auto B_data = B.get_float32_data();

            const int hidden_size = captured_params.at("rnn.hidden_size").i;

            std::string direction = "forward";
            if (captured_params.find("rnn.direction") != captured_params.end())
            {
                direction = captured_params.at("rnn.direction").s;
            }

            if (direction == "bidirectional")
            {
                op->attrs["bias_ih_l0"] = Attribute({hidden_size}, std::vector<float>(&B_data[0], &B_data[hidden_size]));
                op->attrs["bias_hh_l0"] = Attribute({hidden_size}, std::vector<float>(&B_data[hidden_size], &B_data[hidden_size * 2]));

                op->attrs["bias_ih_l0_reverse"] = Attribute({hidden_size}, std::vector<float>(&B_data[hidden_size * 2], &B_data[hidden_size * 3]));
                op->attrs["bias_hh_l0_reverse"] = Attribute({hidden_size}, std::vector<float>(&B_data[hidden_size * 3], &B_data[hidden_size * 4]));
            }
            else
            {
                op->attrs["bias_ih_l0"] = Attribute({hidden_size}, std::vector<float>(&B_data[0], &B_data[hidden_size]));
                op->attrs["bias_hh_l0"] = Attribute({hidden_size}, std::vector<float>(&B_data[hidden_size], &B_data[hidden_size * 2]));
            }
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_RNN_onnx_B, 10)

class nn_RNN_onnx_1 : public nn_RNN_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 initial_h
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
RNN                     rnn         4 2 input W R initial_h out outh %*=%*
Squeeze                 sqz         1 1 out out1 axes=%axes
pnnx.Output             output      2 0 out1 outh
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_RNN_onnx_1, 10)

class nn_RNN_onnx_B1 : public nn_RNN_onnx_B
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 8
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 initial_h
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
pnnx.Attribute          B           0 1 B @data
RNN                     rnn         5 2 input W R B initial_h out outh %*=%*
Squeeze                 sqz         1 1 out out1 axes=%axes
pnnx.Output             output      2 0 out1 outh
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_RNN_onnx_B1, 10)

class nn_RNN_onnx_2 : public nn_RNN_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 initial_h
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
RNN                     rnn         4 1 input W R initial_h out %*=%*
Squeeze                 sqz         1 1 out out1 axes=%axes
pnnx.Output             output      1 0 out1
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_RNN_onnx_2, 10)

class nn_RNN_onnx_B2 : public nn_RNN_onnx_B
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 initial_h
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
pnnx.Attribute          B           0 1 B @data
RNN                     rnn         5 1 input W R B initial_h out %*=%*
Squeeze                 sqz         1 1 out out1 axes=%axes
pnnx.Output             output      1 0 out1
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_RNN_onnx_B2, 10)

class nn_RNN_onnx_3 : public nn_RNN_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
RNN                     rnn         3 1 input W R out %*=%*
Transpose               transpose   1 1 out out1 perm=(0,2,1,3)
Reshape                 reshape     1 1 out1 out2 %*=%*
pnnx.Output             output      1 0 out2
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        if (!nn_RNN_onnx::match(captured_params, captured_attrs))
            return false;

        if (captured_params.at("reshape.shape").ai != std::vector<int>{0, 0, -1})
            return false;

        return true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_RNN_onnx_3, 10)

class nn_RNN_onnx_B3 : public nn_RNN_onnx_B
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
pnnx.Attribute          B           0 1 B @data
RNN                     rnn         4 1 input W R B out %*=%*
Transpose               transpose   1 1 out out1 perm=(0,2,1,3)
Reshape                 reshape     1 1 out1 out2 %*=%*
pnnx.Output             output      1 0 out2
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        if (!nn_RNN_onnx_B::match(captured_params, captured_attrs))
            return false;

        if (captured_params.at("reshape.shape").ai != std::vector<int>{0, 0, -1})
            return false;

        return true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_RNN_onnx_B3, 10)

class nn_RNN_onnx_4 : public nn_RNN_onnx_3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 8
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 initial_h
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
RNN                     rnn         4 2 input W R initial_h out outh %*=%*
Transpose               transpose   1 1 out out1 perm=(0,2,1,3)
Reshape                 reshape     1 1 out1 out2 %*=%*
pnnx.Output             output      2 0 out2 outh
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_RNN_onnx_4, 10)

class nn_RNN_onnx_B4 : public nn_RNN_onnx_B3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 9
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 initial_h
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
pnnx.Attribute          B           0 1 B @data
RNN                     rnn         5 2 input W R B initial_h out outh %*=%*
Transpose               transpose   1 1 out out1 perm=(0,2,1,3)
Reshape                 reshape     1 1 out1 out2 %*=%*
pnnx.Output             output      2 0 out2 outh
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_RNN_onnx_B4, 10)

class nn_RNN_onnx_5 : public nn_RNN_onnx_3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 initial_h
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
RNN                     rnn         4 1 input W R initial_h out %*=%*
Transpose               transpose   1 1 out out1 perm=(0,2,1,3)
Reshape                 reshape     1 1 out1 out2 %*=%*
pnnx.Output             output      1 0 out2
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_RNN_onnx_5, 10)

class nn_RNN_onnx_B5 : public nn_RNN_onnx_B3
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 initial_h
pnnx.Attribute          W           0 1 W @data
pnnx.Attribute          R           0 1 R @data
pnnx.Attribute          B           0 1 B @data
RNN                     rnn         5 1 input W R B initial_h out %*=%*
Transpose               transpose   1 1 out out1 perm=(0,2,1,3)
Reshape                 reshape     1 1 out1 out2 %*=%*
pnnx.Output             output      1 0 out2
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_RNN_onnx_B5, 10)

} // namespace pnnx
