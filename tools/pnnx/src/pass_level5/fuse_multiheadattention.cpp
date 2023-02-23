// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "fuse_multiheadattention.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

namespace pnnx {

class fuse_multiheadattention_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
16 15
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 76 bias=True in_features=%in_features out_features=%qkv_out_features @bias @weight
pnnx.Expression         op_1        1 1 input 77 expr=%expr
Tensor.reshape          op_2        2 1 76 77 78
torch.permute           op_3        1 1 78 79 dims=(2,0,3,1,4)
torch.unbind            op_4        1 3 79 80 81 82 dim=0
pnnx.Expression         op_5        1 1 80 83 expr=%expr2
torch.permute           op_6        1 1 81 84 dims=(0,1,3,2)
torch.matmul            op_7        2 1 83 84 85
F.softmax               op_8        1 1 85 86 dim=-1
torch.matmul            op_9        2 1 86 82 87
pnnx.Expression         op_10       1 1 input 88 expr=%expr3
torch.permute           op_11       1 1 87 89 dims=(0,2,1,3)
Tensor.reshape          op_12       2 1 89 88 90
nn.Linear               out_proj    1 1 90 out bias=True in_features=%out_proj_in_features out_features=%out_proj_out_features @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "nn.MultiheadAttention";
    }

    const char* name_str() const
    {
        return "attention";
    }

    bool match_captured_params_attrs(const std::map<std::string, Parameter>& captured_params) const
    {
        // [-1,int(size(@0,1)),3,8,15]   (-1,12,3,8,15)
        // mul(@0,2.581989e-01)
        // [-1,int(size(@0,1)),120]
        const std::string& expr = captured_params.at("expr").s;
        const std::string& expr2 = captured_params.at("expr2").s;
        const std::string& expr3 = captured_params.at("expr3").s;

        // TODO stricter rules here

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::string& expr = captured_params.at("expr").s;
        const std::string& expr2 = captured_params.at("expr2").s;

        int num_heads;
        if (expr[0] == '[')
        {
            sscanf(expr.c_str(), "[-1,int(size(@0,1)),3,%d,8,15))]", &num_heads);
        }
        else
        {
            sscanf(expr.c_str(), "(-1,12,3,%d,15)", &num_heads);
        }

        op->params["num_heads"] = num_heads;
        op->params["batch_first"] = true;
        op->params["add_zero_attn"] = false;
        op->params["add_bias_kv"] = false;
        op->params["bias"] = true;

        int qkv_out_features = captured_params.at("qkv_out_features").i;
        int embed_dim = qkv_out_features / 3;

        op->params["embed_dim"] = embed_dim;
        op->params["kdim"] = embed_dim;
        op->params["vdim"] = embed_dim;

        op->attrs["in_proj_weight"] = captured_attrs.at("op_0.weight");
        op->attrs["in_proj_bias"] = captured_attrs.at("op_0.bias");

        op->attrs["out_proj.weight"] = captured_attrs.at("out_proj.weight");
        op->attrs["out_proj.bias"] = captured_attrs.at("out_proj.bias");
    }
};

class fuse_multiheadattention_pass_1 : public fuse_multiheadattention_pass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
14 13
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input 76 bias=True in_features=%in_features out_features=%qkv_out_features @bias @weight
Tensor.reshape          op_1        1 1 76 77 shape=%expr
torch.permute           op_2        1 1 77 78 dims=(2,0,3,1,4)
torch.unbind            op_3        1 3 78 79 80 81 dim=0
pnnx.Expression         op_4        1 1 79 82 expr=%expr2
torch.permute           op_5        1 1 80 83 dims=(0,1,3,2)
torch.matmul            op_6        2 1 82 83 84
F.softmax               op_7        1 1 84 85 dim=-1
torch.matmul            op_8        2 1 85 81 86
torch.permute           op_9        1 1 86 87 dims=(0,2,1,3)
Tensor.reshape          op_10       1 1 87 88 shape=%expr3
nn.Linear               out_proj    1 1 88 out bias=True in_features=%out_proj_in_features out_features=%out_proj_out_features @bias @weight
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

void fuse_multiheadattention(Graph& graph)
{
    fuse_multiheadattention_pass a;
    fuse_multiheadattention_pass_1 b;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
}

} // namespace pnnx
