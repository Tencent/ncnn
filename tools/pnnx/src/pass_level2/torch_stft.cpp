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
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 normalized
pnnx.Input              input_6     0 1 onesided
pnnx.Input              input_7     0 1 return_complex
aten::stft              op_0        8 1 input n_fft hop_length win_length window normalized onesided return_complex out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.stft";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
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
24 23
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 normalized
pnnx.Input              input_6     0 1 onesided
pnnx.Input              input_7     0 1 return_complex
Tensor.size             op_0        1 1 input 16 dim=0
Tensor.size             op_1        1 1 input 25 dim=1
prim::Constant          op_2        0 1 153 value=1
prim::ListConstruct     op_3        3 1 153 16 25 26
Tensor.view             op_4        2 1 input 26 input.1
prim::Constant          op_5        0 1 29 value=%pad
prim::Constant          op_6        0 1 154 value=%pad
prim::ListConstruct     op_7        2 1 29 154 30
prim::Constant          op_8        0 1 31 value=%pad_mode
F.pad                   op_9        3 1 input.1 30 31 input0.1
Tensor.size             op_10       1 1 input0.1 39 dim=1
Tensor.size             op_11       1 1 input0.1 48 dim=2
prim::ListConstruct     op_12       2 1 39 48 49
Tensor.view             op_13       2 1 input0.1 49 input1.1
torch.stft              op_14       8 1 input1.1 n_fft hop_length win_length window normalized onesided return_complex out center=False pad_mode=reflect
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.stft";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
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
23 22
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 normalized
pnnx.Input              input_6     0 1 onesided
pnnx.Input              input_7     0 1 return_complex
Tensor.size             op_0        1 1 input 81 dim=0
prim::Constant          op_1        0 1 172 value=1
prim::Constant          op_2        0 1 173 value=1
prim::ListConstruct     op_3        3 1 172 173 81 82
Tensor.view             op_4        2 1 input 82 input2.1
prim::Constant          op_5        0 1 174 value=%pad
prim::Constant          op_6        0 1 175 value=%pad
prim::ListConstruct     op_7        2 1 174 175 85
prim::Constant          op_8        0 1 176 value=None
F.pad                   op_9        3 1 input2.1 85 176 input3.1 mode=%pad_mode
Tensor.size             op_10       1 1 input3.1 95 dim=2
prim::ListConstruct     op_11       1 1 95 96
Tensor.view             op_12       2 1 input3.1 96 input4.1
torch.stft              op_13       8 1 input4.1 n_fft hop_length win_length window normalized onesided return_complex out center=False pad_mode=reflect
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_stft_2, 119)

class torch_stft_3 : public torch_stft_1
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
23 22
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 normalized
pnnx.Input              input_6     0 1 onesided
pnnx.Input              input_7     0 1 return_complex
Tensor.size             op_0        1 1 input 111 dim=0
prim::Constant          op_1        0 1 184 value=1
prim::Constant          op_2        0 1 185 value=1
prim::ListConstruct     op_3        3 1 184 185 111 112
Tensor.view             op_4        2 1 input 112 input5.1
prim::Constant          op_5        0 1 186 value=%pad
prim::Constant          op_6        0 1 187 value=%pad
prim::ListConstruct     op_7        2 1 186 187 115
prim::Constant          op_8        0 1 188 value=%pad_mode
F.pad                   op_9        3 1 input5.1 115 188 input6.1
Tensor.size             op_10       1 1 input6.1 125 dim=2
prim::ListConstruct     op_11       1 1 125 126
Tensor.view             op_12       2 1 input6.1 126 input7.1
torch.stft              op_13       8 1 input7.1 n_fft hop_length win_length window normalized onesided return_complex out center=False pad_mode=reflect
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_stft_3, 119)

} // namespace pnnx
