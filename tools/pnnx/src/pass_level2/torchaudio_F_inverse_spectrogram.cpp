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

class torchaudio_F_inverse_spectrogram : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
29 28
pnnx.Input              input_0     0 1 spectrogram
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 center
pnnx.Input              input_6     0 1 onesided
prim::Constant          op_0        0 1 13 value=0
aten::size              op_1        2 1 spectrogram 13 14
prim::NumToTensor       op_2        1 1 14 15
aten::Int               op_3        1 1 15 18
prim::Constant          op_4        0 1 20 value=1
aten::size              op_5        2 1 spectrogram 20 21
prim::NumToTensor       op_6        1 1 21 22
aten::Int               op_7        1 1 22 28
prim::Constant          op_8        0 1 24 value=-1
prim::ListConstruct     op_9        3 1 24 18 28 29
aten::reshape           op_10       2 1 spectrogram 29 spectrogram.1
prim::Constant          op_11       0 1 normalized value=%normalized
prim::Constant          op_12       0 1 length value=None
prim::Constant          op_13       0 1 return_complex value=False
aten::istft             op_14       10 1 spectrogram.1 n_fft hop_length win_length window center normalized onesided length return_complex waveform.1
prim::Constant          op_15       0 1 75 value=1
aten::size              op_16       2 1 waveform.1 75 42
prim::NumToTensor       op_17       1 1 42 43
aten::Int               op_18       1 1 43 47
prim::ListConstruct     op_19       1 1 47 48
aten::reshape           op_20       2 1 waveform.1 48 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchaudio.functional.inverse_spectrogram";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["length"] = Parameter();
        op->params["pad"] = 0;
        if (captured_params.at("normalized").b)
        {
            op->params["normalized"] = "frame_length";
        }
        else
        {
            op->params["normalized"] = false;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_inverse_spectrogram, 6)

class torchaudio_F_inverse_spectrogram_0 : public torchaudio_F_inverse_spectrogram
{
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
33 32
pnnx.Input              input_0     0 1 spectrogram
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 center
pnnx.Input              input_6     0 1 onesided
prim::Constant          op_0        0 1 13 value=0
aten::size              op_1        2 1 spectrogram 13 14
prim::NumToTensor       op_2        1 1 14 15
aten::Int               op_3        1 1 15 18
prim::Constant          op_4        0 1 20 value=1
aten::size              op_5        2 1 spectrogram 20 21
prim::NumToTensor       op_6        1 1 21 22
aten::Int               op_7        1 1 22 25
prim::Constant          op_8        0 1 27 value=2
aten::size              op_9        2 1 spectrogram 27 28
prim::NumToTensor       op_10       1 1 28 29
aten::Int               op_11       1 1 29 35
prim::Constant          op_12       0 1 31 value=-1
prim::ListConstruct     op_13       3 1 31 25 35 36
aten::reshape           op_14       2 1 spectrogram 36 spectrogram.1
prim::Constant          op_15       0 1 normalized value=%normalized
prim::Constant          op_16       0 1 length value=None
prim::Constant          op_17       0 1 return_complex value=False
aten::istft             op_18       10 1 spectrogram.1 n_fft hop_length win_length window center normalized onesided length return_complex waveform.1
prim::Constant          op_19       0 1 83 value=1
aten::size              op_20       2 1 waveform.1 83 49
prim::NumToTensor       op_21       1 1 49 50
aten::Int               op_22       1 1 50 55
prim::ListConstruct     op_23       2 1 18 55 56
aten::reshape           op_24       2 1 waveform.1 56 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_inverse_spectrogram_0, 6)

class torchaudio_F_inverse_spectrogram_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
15 14
pnnx.Input              input_0     0 1 spectrogram
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 center
pnnx.Input              input_6     0 1 onesided
prim::Constant          op_0        0 1 13 value=2.000000e+00
aten::pow               op_1        2 1 window 13 14
prim::Constant          op_2        0 1 87 value=None
aten::sum               op_3        2 1 14 87 16
aten::sqrt              op_4        1 1 16 17
aten::mul               op_5        2 1 spectrogram 17 spectrogram.1
torchaudio.functional.inverse_spectrogram op_6 7 1 spectrogram.1 n_fft hop_length win_length window center onesided out normalized=False length=%length pad=%pad
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchaudio.functional.inverse_spectrogram";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["length"] = captured_params.at("length");
        op->params["pad"] = captured_params.at("pad");
        op->params["normalized"] = "window";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_inverse_spectrogram_1, 7)

} // namespace pnnx
