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

#include <math.h>
#include <string.h>

namespace pnnx {

namespace ncnn {

class nn_MultiheadAttention : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.MultiheadAttention   op_0        1 1 input out num_heads=%num_heads batch_first=%batch_first add_zero_attn=%add_zero_attn embed_dim=%embed_dim kdim=%kdim vdim=%vdim bias=%bias add_bias_kv=%add_bias_kv @in_proj_weight @in_proj_bias @bias_k @bias_v @out_proj.weight @out_proj.bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "MultiHeadAttention";
    }

    const char* name_str() const
    {
        return "attention";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = captured_params.at("embed_dim");
        op->params["1"] = captured_params.at("num_heads");

        if (captured_params.at("add_bias_kv").b)
        {
            fprintf(stderr, "MultiheadAttention add_bias_kv=True not supported\n");
        }

        const int embed_dim = captured_params.at("embed_dim").i;
        const int kdim = captured_params.at("kdim").i;
        const int vdim = captured_params.at("vdim").i;

        // split in_proj_weight and in_proj_bias into q k v
        std::vector<float> q_weight(embed_dim * embed_dim);
        std::vector<float> q_bias(embed_dim);
        std::vector<float> k_weight(embed_dim * embed_dim);
        std::vector<float> k_bias(embed_dim);
        std::vector<float> v_weight(embed_dim * embed_dim);
        std::vector<float> v_bias(embed_dim);
        {
            // qkv - embed_dim - embed_dim
            auto w = captured_attrs.at("op_0.in_proj_weight").get_float32_data();
            // qkv - embed_dim
            auto b = captured_attrs.at("op_0.in_proj_bias").get_float32_data();

            const float* wptr = (const float*)w.data();
            const float* bptr = (const float*)b.data();

            {
                memcpy(q_weight.data(), wptr, embed_dim * embed_dim * sizeof(float));
                memcpy(q_bias.data(), bptr, embed_dim * sizeof(float));
                wptr += embed_dim * embed_dim;
                bptr += embed_dim;
            }

            {
                memcpy(k_weight.data(), wptr, embed_dim * embed_dim * sizeof(float));
                memcpy(k_bias.data(), bptr, embed_dim * sizeof(float));
                wptr += embed_dim * embed_dim;
                bptr += embed_dim;
            }

            {
                memcpy(v_weight.data(), wptr, embed_dim * embed_dim * sizeof(float));
                memcpy(v_bias.data(), bptr, embed_dim * sizeof(float));
            }
        }

        op->params["2"] = embed_dim * embed_dim;
        op->params["3"] = kdim;
        op->params["4"] = vdim;

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = Attribute({embed_dim, embed_dim}, q_weight);
        op->attrs["2"] = Attribute({embed_dim}, q_bias);
        op->attrs["3"] = Attribute();
        op->attrs["3"].data = {0, 0, 0, 0};
        op->attrs["4"] = Attribute({embed_dim, embed_dim}, k_weight);
        op->attrs["5"] = Attribute({embed_dim}, k_bias);
        op->attrs["6"] = Attribute();
        op->attrs["6"].data = {0, 0, 0, 0};
        op->attrs["7"] = Attribute({embed_dim, embed_dim}, v_weight);
        op->attrs["8"] = Attribute({embed_dim}, v_bias);
        op->attrs["9"] = Attribute();
        op->attrs["9"].data = {0, 0, 0, 0};
        op->attrs["a"] = captured_attrs.at("op_0.out_proj.weight");
        op->attrs["b"] = captured_attrs.at("op_0.out_proj.bias");
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_MultiheadAttention, 20)

class nn_MultiheadAttention_attn_mask : public nn_MultiheadAttention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 attn_mask
nn.MultiheadAttention   op_0        2 1 input attn_mask out num_heads=%num_heads batch_first=%batch_first add_zero_attn=%add_zero_attn embed_dim=%embed_dim kdim=%kdim vdim=%vdim bias=%bias add_bias_kv=%add_bias_kv @in_proj_weight @in_proj_bias @bias_k @bias_v @out_proj.weight @out_proj.bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* mha = matched_operators.at("op_0");
        return mha->inputnames.size() == 2 && mha->inputnames[1] == "attn_mask";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        nn_MultiheadAttention::write(op, captured_params, captured_attrs);
        op->params["5"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_MultiheadAttention_attn_mask, 19)

class nn_MultiheadAttention_1 : public nn_MultiheadAttention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.MultiheadAttention   op_0        1 1 input out num_heads=%num_heads add_zero_attn=%add_zero_attn embed_dim=%embed_dim kdim=%kdim vdim=%vdim bias=%bias add_bias_kv=%add_bias_kv @in_proj_weight @in_proj_bias @bias_k @bias_v @out_proj.weight @out_proj.bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_MultiheadAttention_1, 20)

class nn_MultiheadAttention_1_attn_mask : public nn_MultiheadAttention
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 attn_mask
nn.MultiheadAttention   op_0        2 1 input attn_mask out num_heads=%num_heads add_zero_attn=%add_zero_attn embed_dim=%embed_dim kdim=%kdim vdim=%vdim bias=%bias add_bias_kv=%add_bias_kv @in_proj_weight @in_proj_bias @bias_k @bias_v @out_proj.weight @out_proj.bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* mha = matched_operators.at("op_0");
        return mha->inputnames.size() == 2 && mha->inputnames[1] == "attn_mask";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        nn_MultiheadAttention::write(op, captured_params, captured_attrs);
        op->params["5"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_MultiheadAttention_1_attn_mask, 19)

class nn_MultiheadAttention_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
nn.MultiheadAttention   op_0        3 1 query key value out num_heads=%num_heads batch_first=%batch_first add_zero_attn=%add_zero_attn embed_dim=%embed_dim kdim=%kdim vdim=%vdim bias=%bias add_bias_kv=%add_bias_kv @in_proj_weight @q_proj_weight @k_proj_weight @v_proj_weight @in_proj_bias @bias_k @bias_v @out_proj.weight @out_proj.bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "MultiHeadAttention";
    }

    const char* name_str() const
    {
        return "attention";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = captured_params.at("embed_dim");
        op->params["1"] = captured_params.at("num_heads");

        if (captured_params.at("add_bias_kv").b)
        {
            fprintf(stderr, "MultiheadAttention add_bias_kv=True not supported\n");
        }

        const int embed_dim = captured_params.at("embed_dim").i;
        const int kdim = captured_params.at("kdim").i;
        const int vdim = captured_params.at("vdim").i;

        // split in_proj_bias into q k v
        std::vector<float> q_bias(embed_dim);
        std::vector<float> k_bias(embed_dim);
        std::vector<float> v_bias(embed_dim);
        {
            // qkv - embed_dim
            auto b = captured_attrs.at("op_0.in_proj_bias").get_float32_data();

            const float* bptr = (const float*)b.data();

            {
                memcpy(q_bias.data(), bptr, embed_dim * sizeof(float));
                bptr += embed_dim;
            }

            {
                memcpy(k_bias.data(), bptr, embed_dim * sizeof(float));
                bptr += embed_dim;
            }

            {
                memcpy(v_bias.data(), bptr, embed_dim * sizeof(float));
            }
        }

        op->params["2"] = embed_dim * embed_dim;
        op->params["3"] = kdim;
        op->params["4"] = vdim;

        if (captured_attrs.find("op_0.in_proj_weight") != captured_attrs.end())
        {
            // split in_proj_weight and in_proj_bias into q k v
            std::vector<float> q_weight(embed_dim * embed_dim);
            std::vector<float> k_weight(embed_dim * kdim);
            std::vector<float> v_weight(embed_dim * vdim);
            {
                // qkv - embed_dim - embed_dim
                auto w = captured_attrs.at("op_0.in_proj_weight").get_float32_data();

                const float* wptr = (const float*)w.data();

                {
                    memcpy(q_weight.data(), wptr, embed_dim * embed_dim * sizeof(float));
                    wptr += embed_dim * embed_dim;
                }

                {
                    memcpy(k_weight.data(), wptr, embed_dim * kdim * sizeof(float));
                    wptr += embed_dim * kdim;
                }

                {
                    memcpy(v_weight.data(), wptr, embed_dim * vdim * sizeof(float));
                }
            }

            op->attrs["0"] = Attribute();
            op->attrs["0"].data = {0, 0, 0, 0};
            op->attrs["1"] = Attribute({embed_dim, embed_dim}, q_weight);
            op->attrs["2"] = Attribute({embed_dim}, q_bias);
            op->attrs["3"] = Attribute();
            op->attrs["3"].data = {0, 0, 0, 0};
            op->attrs["4"] = Attribute({embed_dim, kdim}, k_weight);
            op->attrs["5"] = Attribute({embed_dim}, k_bias);
            op->attrs["6"] = Attribute();
            op->attrs["6"].data = {0, 0, 0, 0};
            op->attrs["7"] = Attribute({embed_dim, vdim}, v_weight);
            op->attrs["8"] = Attribute({embed_dim}, v_bias);
        }
        else
        {
            op->attrs["0"] = Attribute();
            op->attrs["0"].data = {0, 0, 0, 0};
            op->attrs["1"] = captured_attrs.at("op_0.q_proj_weight");
            op->attrs["2"] = Attribute({embed_dim}, q_bias);
            op->attrs["3"] = Attribute();
            op->attrs["3"].data = {0, 0, 0, 0};
            op->attrs["4"] = captured_attrs.at("op_0.k_proj_weight");
            op->attrs["5"] = Attribute({embed_dim}, k_bias);
            op->attrs["6"] = Attribute();
            op->attrs["6"].data = {0, 0, 0, 0};
            op->attrs["7"] = captured_attrs.at("op_0.v_proj_weight");
            op->attrs["8"] = Attribute({embed_dim}, v_bias);
        }

        op->attrs["9"] = Attribute();
        op->attrs["9"].data = {0, 0, 0, 0};
        op->attrs["a"] = captured_attrs.at("op_0.out_proj.weight");
        op->attrs["b"] = captured_attrs.at("op_0.out_proj.bias");
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_MultiheadAttention_2, 20)

class nn_MultiheadAttention_2_attn_mask : public nn_MultiheadAttention_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 attn_mask
nn.MultiheadAttention   op_0        4 1 query key value attn_mask out num_heads=%num_heads batch_first=%batch_first add_zero_attn=%add_zero_attn embed_dim=%embed_dim kdim=%kdim vdim=%vdim bias=%bias add_bias_kv=%add_bias_kv @in_proj_weight @q_proj_weight @k_proj_weight @v_proj_weight @in_proj_bias @bias_k @bias_v @out_proj.weight @out_proj.bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* mha = matched_operators.at("op_0");
        return mha->inputnames.size() == 4 && mha->inputnames[3] == "attn_mask";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        nn_MultiheadAttention_2::write(op, captured_params, captured_attrs);
        op->params["5"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_MultiheadAttention_2_attn_mask, 19)

class nn_MultiheadAttention_3 : public nn_MultiheadAttention_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
nn.MultiheadAttention   op_0        3 1 query key value out num_heads=%num_heads add_zero_attn=%add_zero_attn embed_dim=%embed_dim kdim=%kdim vdim=%vdim bias=%bias add_bias_kv=%add_bias_kv @in_proj_weight @q_proj_weight @k_proj_weight @v_proj_weight @in_proj_bias @bias_k @bias_v @out_proj.weight @out_proj.bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_MultiheadAttention_3, 20)

class nn_MultiheadAttention_3_attn_mask : public nn_MultiheadAttention_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 value
pnnx.Input              input_3     0 1 attn_mask
nn.MultiheadAttention   op_0        4 1 query key value attn_mask out num_heads=%num_heads add_zero_attn=%add_zero_attn embed_dim=%embed_dim kdim=%kdim vdim=%vdim bias=%bias add_bias_kv=%add_bias_kv @in_proj_weight @q_proj_weight @k_proj_weight @v_proj_weight @in_proj_bias @bias_k @bias_v @out_proj.weight @out_proj.bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* mha = matched_operators.at("op_0");
        return mha->inputnames.size() == 4 && mha->inputnames[3] == "attn_mask";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        nn_MultiheadAttention_2::write(op, captured_params, captured_attrs);
        op->params["5"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_MultiheadAttention_3_attn_mask, 19)

class nn_MultiheadAttention_4 : public nn_MultiheadAttention_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
nn.MultiheadAttention   op_0        2 1 query key out num_heads=%num_heads batch_first=%batch_first add_zero_attn=%add_zero_attn embed_dim=%embed_dim kdim=%kdim vdim=%vdim bias=%bias add_bias_kv=%add_bias_kv @in_proj_weight @q_proj_weight @k_proj_weight @v_proj_weight @in_proj_bias @bias_k @bias_v @out_proj.weight @out_proj.bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_MultiheadAttention_4, 20)

class nn_MultiheadAttention_4_attn_mask : public nn_MultiheadAttention_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 attn_mask
nn.MultiheadAttention   op_0        3 1 query key attn_mask out num_heads=%num_heads batch_first=%batch_first add_zero_attn=%add_zero_attn embed_dim=%embed_dim kdim=%kdim vdim=%vdim bias=%bias add_bias_kv=%add_bias_kv @in_proj_weight @q_proj_weight @k_proj_weight @v_proj_weight @in_proj_bias @bias_k @bias_v @out_proj.weight @out_proj.bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* mha = matched_operators.at("op_0");
        return mha->inputnames.size() == 3 && mha->inputnames[2] == "attn_mask";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        nn_MultiheadAttention_2::write(op, captured_params, captured_attrs);
        op->params["5"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_MultiheadAttention_4_attn_mask, 19)

class nn_MultiheadAttention_5 : public nn_MultiheadAttention_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
nn.MultiheadAttention   op_0        2 1 query key out num_heads=%num_heads add_zero_attn=%add_zero_attn embed_dim=%embed_dim kdim=%kdim vdim=%vdim bias=%bias add_bias_kv=%add_bias_kv @in_proj_weight @q_proj_weight @k_proj_weight @v_proj_weight @in_proj_bias @bias_k @bias_v @out_proj.weight @out_proj.bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_MultiheadAttention_5, 20)

class nn_MultiheadAttention_5_attn_mask : public nn_MultiheadAttention_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 query
pnnx.Input              input_1     0 1 key
pnnx.Input              input_2     0 1 attn_mask
nn.MultiheadAttention   op_0        3 1 query key attn_mask out num_heads=%num_heads add_zero_attn=%add_zero_attn embed_dim=%embed_dim kdim=%kdim vdim=%vdim bias=%bias add_bias_kv=%add_bias_kv @in_proj_weight @q_proj_weight @k_proj_weight @v_proj_weight @in_proj_bias @bias_k @bias_v @out_proj.weight @out_proj.bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* mha = matched_operators.at("op_0");
        return mha->inputnames.size() == 3 && mha->inputnames[2] == "attn_mask";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        nn_MultiheadAttention_2::write(op, captured_params, captured_attrs);
        op->params["5"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_MultiheadAttention_5_attn_mask, 19)

} // namespace ncnn

} // namespace pnnx
