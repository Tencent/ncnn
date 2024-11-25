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
12 11
pnnx.Input              input_0     0 1 spectrogram
pnnx.Input              input_1     0 1 window
Tensor.size             op_0        1 1 spectrogram 18 dim=0
Tensor.size             op_1        1 1 spectrogram 28 dim=1
prim::Constant          op_2        0 1 24 value=-1
prim::ListConstruct     op_3        3 1 24 18 28 29
Tensor.reshape          op_4        2 1 spectrogram 29 spectrogram.1
torch.istft             op_5        2 1 spectrogram.1 window waveform.1 n_fft=%n_fft hop_length=%hop_length win_length=%win_length center=%center normalized=%normalized onesided=%onesided length=None return_complex=False
Tensor.size             op_6        1 1 waveform.1 47 dim=1
prim::ListConstruct     op_7        1 1 47 48
Tensor.reshape          op_8        2 1 waveform.1 48 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchaudio.functional.inverse_spectrogram";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        GraphRewriterPass::write(op, captured_params);

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

class torchaudio_F_inverse_spectrogram_0 : public torchaudio_F_inverse_spectrogram
{
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
13 12
pnnx.Input              input_0     0 1 spectrogram
pnnx.Input              input_1     0 1 window
Tensor.size             op_0        1 1 spectrogram 18 dim=0
Tensor.size             op_1        1 1 spectrogram 25 dim=1
Tensor.size             op_2        1 1 spectrogram 35 dim=2
prim::Constant          op_3        0 1 31 value=-1
prim::ListConstruct     op_4        3 1 31 25 35 36
Tensor.reshape          op_5        2 1 spectrogram 36 spectrogram.1
torch.istft             op_6        2 1 spectrogram.1 window waveform.1 n_fft=%n_fft hop_length=%hop_length win_length=%win_length center=%center normalized=%normalized onesided=%onesided length=None return_complex=False
Tensor.size             op_7        1 1 waveform.1 55 dim=1
prim::ListConstruct     op_8        2 1 18 55 56
Tensor.reshape          op_9        2 1 waveform.1 56 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_inverse_spectrogram, 140)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_inverse_spectrogram_0, 140)

class torchaudio_F_inverse_spectrogram_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 spectrogram
pnnx.Input              input_1     0 1 window
prim::Constant          op_0        0 1 13 value=2.000000e+00
aten::pow               op_1        2 1 window 13 14
torch.sum               op_2        1 1 14 16
aten::sqrt              op_3        1 1 16 17
aten::mul               op_4        2 1 spectrogram 17 spectrogram.1
torchaudio.functional.inverse_spectrogram op_5 2 1 spectrogram.1 window out n_fft=%n_fft hop_length=%hop_length win_length=%win_length center=%center normalized=False onesided=%onesided length=None pad=%pad
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchaudio.functional.inverse_spectrogram";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        GraphRewriterPass::write(op, captured_params);

        op->params["length"] = Parameter();
        op->params["normalized"] = "window";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_inverse_spectrogram_1, 141)

} // namespace pnnx
