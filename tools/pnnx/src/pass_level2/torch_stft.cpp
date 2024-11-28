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

#include "pass_level2.h"

namespace pnnx {

class torch_stft : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
10 9
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 window
prim::Constant          op_0        0 1 n_fft value=%n_fft
prim::Constant          op_1        0 1 hop_length value=%hop_length
prim::Constant          op_2        0 1 win_length value=%win_length
prim::Constant          op_3        0 1 normalized value=%normalized
prim::Constant          op_4        0 1 onesided value=%onesided
prim::Constant          op_5        0 1 return_complex value=%return_complex
aten::stft              op_6        8 1 input n_fft hop_length win_length window normalized onesided return_complex out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.stft";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        GraphRewriterPass::write(op, captured_params);

        op->params["pad_mode"] = "reflect";
        op->params["center"] = false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_stft, 80)

class torch_stft_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
14 13
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 window
Tensor.size             op_0        1 1 input 16 dim=0
Tensor.size             op_1        1 1 input 25 dim=1
prim::Constant          op_2        0 1 153 value=1
prim::ListConstruct     op_3        3 1 153 16 25 26
Tensor.view             op_4        2 1 input 26 input.1
F.pad                   op_5        1 1 input.1 input0.1 mode=%pad_mode pad=(%pad,%pad) value=None
Tensor.size             op_6        1 1 input0.1 39 dim=1
Tensor.size             op_7        1 1 input0.1 48 dim=2
prim::ListConstruct     op_8        2 1 39 48 49
Tensor.view             op_9        2 1 input0.1 49 input1.1
torch.stft              op_10       2 1 input1.1 window out n_fft=%n_fft hop_length=%hop_length win_length=%win_length center=False pad_mode=reflect normalized=%normalized onesided=%onesided return_complex=%return_complex
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.stft";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["n_fft"] = captured_params.at("n_fft");
        op->params["hop_length"] = captured_params.at("hop_length");
        op->params["win_length"] = captured_params.at("win_length");
        op->params["normalized"] = captured_params.at("normalized");
        op->params["onesided"] = captured_params.at("onesided");
        op->params["return_complex"] = captured_params.at("return_complex");
        op->params["pad_mode"] = captured_params.at("pad_mode");
        op->params["center"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_stft_1, 119)

class torch_stft_2 : public torch_stft_1
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
13 12
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 window
Tensor.size             op_0        1 1 input 81 dim=0
prim::Constant          op_1        0 1 172 value=1
prim::Constant          op_2        0 1 173 value=1
prim::ListConstruct     op_3        3 1 172 173 81 82
Tensor.view             op_4        2 1 input 82 input2.1
F.pad                   op_5        1 1 input2.1 input3.1 mode=%pad_mode pad=(%pad,%pad) value=None
Tensor.size             op_6        1 1 input3.1 95 dim=2
prim::ListConstruct     op_7        1 1 95 96
Tensor.view             op_8        2 1 input3.1 96 input4.1
torch.stft              op_9        2 1 input4.1 window out n_fft=%n_fft hop_length=%hop_length win_length=%win_length center=False pad_mode=reflect normalized=%normalized onesided=%onesided return_complex=%return_complex
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_stft_2, 119)

} // namespace pnnx
