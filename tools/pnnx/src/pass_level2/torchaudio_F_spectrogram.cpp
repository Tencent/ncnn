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

class torchaudio_F_spectrogram : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
27 26
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 onesided
prim::Constant          op_0        0 1 11 value=0
aten::size              op_1        2 1 waveform 11 12
prim::NumToTensor       op_2        1 1 12 13
aten::Int               op_3        1 1 13 18
prim::Constant          op_4        0 1 15 value=-1
prim::ListConstruct     op_5        2 1 15 18 19
aten::reshape           op_6        2 1 waveform 19 waveform.1
prim::Constant          op_7        0 1 normalized value=%normalized
prim::Constant          op_8        0 1 return_complex value=True
aten::stft              op_9        8 1 waveform.1 n_fft hop_length win_length window normalized onesided return_complex spec_f.1
prim::Constant          op_10       0 1 29 value=1
aten::size              op_11       2 1 spec_f.1 29 30
prim::NumToTensor       op_12       1 1 30 31
aten::Int               op_13       1 1 31 34
prim::Constant          op_14       0 1 36 value=2
aten::size              op_15       2 1 spec_f.1 36 37
prim::NumToTensor       op_16       1 1 37 38
aten::Int               op_17       1 1 38 43
prim::ListConstruct     op_18       2 1 34 43 44
aten::reshape           op_19       2 1 spec_f.1 44 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchaudio.functional.spectrogram";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["pad"] = 0;
        op->params["pad_mode"] = "reflect";
        op->params["center"] = false;
        op->params["power"] = Parameter();
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram, 6)

class torchaudio_F_spectrogram_0 : public torchaudio_F_spectrogram
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
31 30
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 onesided
prim::Constant          op_0        0 1 11 value=0
aten::size              op_1        2 1 waveform 11 12
prim::NumToTensor       op_2        1 1 12 13
aten::Int               op_3        1 1 13 16
prim::Constant          op_4        0 1 18 value=1
aten::size              op_5        2 1 waveform 18 19
prim::NumToTensor       op_6        1 1 19 20
aten::Int               op_7        1 1 20 25
prim::Constant          op_8        0 1 22 value=-1
prim::ListConstruct     op_9        2 1 22 25 26
aten::reshape           op_10       2 1 waveform 26 waveform.1
prim::Constant          op_11       0 1 normalized value=%normalized
prim::Constant          op_12       0 1 return_complex value=True
aten::stft              op_13       8 1 waveform.1 n_fft hop_length win_length window normalized onesided return_complex spec_f.1
prim::Constant          op_14       0 1 72 value=1
aten::size              op_15       2 1 spec_f.1 72 36
prim::NumToTensor       op_16       1 1 36 37
aten::Int               op_17       1 1 37 40
prim::Constant          op_18       0 1 42 value=2
aten::size              op_19       2 1 spec_f.1 42 43
prim::NumToTensor       op_20       1 1 43 44
aten::Int               op_21       1 1 44 50
prim::ListConstruct     op_22       3 1 16 40 50 51
aten::reshape           op_23       2 1 spec_f.1 51 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_0, 6)

class torchaudio_F_spectrogram_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
58 57
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 onesided
prim::Constant          op_0        0 1 18 value=1
aten::size              op_1        2 1 waveform 18 19
prim::NumToTensor       op_2        1 1 19 20
aten::Int               op_3        1 1 20 25
prim::Constant          op_4        0 1 22 value=-1
prim::ListConstruct     op_5        2 1 22 25 26
aten::reshape           op_6        2 1 waveform 26 waveform.1
prim::Constant          op_7        0 1 106 value=0
aten::size              op_8        2 1 waveform.1 106 29
prim::NumToTensor       op_9        1 1 29 30
aten::Int               op_10       1 1 30 33
prim::Constant          op_11       0 1 107 value=1
aten::size              op_12       2 1 waveform.1 107 35
prim::NumToTensor       op_13       1 1 35 36
aten::Int               op_14       1 1 36 41
prim::Constant          op_15       0 1 108 value=1
prim::ListConstruct     op_16       3 1 108 33 41 42
aten::view              op_17       2 1 waveform.1 42 input0.1
prim::Constant          op_18       0 1 45 value=%pad_left
prim::Constant          op_19       0 1 109 value=%pad_right
prim::ListConstruct     op_20       2 1 45 109 46
prim::Constant          op_21       0 1 47 value=%pad_mode
prim::Constant          op_22       0 1 110 value=None
aten::pad               op_23       4 1 input0.1 46 47 110 input1.1
prim::Constant          op_24       0 1 111 value=1
aten::size              op_25       2 1 input1.1 111 51
prim::NumToTensor       op_26       1 1 51 52
aten::Int               op_27       1 1 52 55
prim::Constant          op_28       0 1 57 value=2
aten::size              op_29       2 1 input1.1 57 58
prim::NumToTensor       op_30       1 1 58 59
aten::Int               op_31       1 1 59 64
prim::ListConstruct     op_32       2 1 55 64 65
aten::view              op_33       2 1 input1.1 65 input2.1
prim::Constant          op_34       0 1 normalized value=%normalized
prim::Constant          op_35       0 1 return_complex value=True
aten::stft              op_36       8 1 input2.1 n_fft hop_length win_length window normalized onesided return_complex spec_f.1
prim::Constant          op_37       0 1 11 value=0
aten::size              op_38       2 1 waveform 11 12
prim::NumToTensor       op_39       1 1 12 13
aten::Int               op_40       1 1 13 16
prim::Constant          op_41       0 1 116 value=1
aten::size              op_42       2 1 spec_f.1 116 75
prim::NumToTensor       op_43       1 1 75 76
aten::Int               op_44       1 1 76 79
prim::Constant          op_45       0 1 117 value=2
aten::size              op_46       2 1 spec_f.1 117 81
prim::NumToTensor       op_47       1 1 81 82
aten::Int               op_48       1 1 82 88
prim::ListConstruct     op_49       3 1 16 79 88 89
aten::reshape           op_50       2 1 spec_f.1 89 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchaudio.functional.spectrogram";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["pad"] = 0;
        op->params["pad_mode"] = captured_params.at("pad_mode");
        op->params["center"] = true;
        op->params["power"] = Parameter();
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_1, 6)

class torchaudio_F_spectrogram_1_1 : public torchaudio_F_spectrogram_1
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
63 62
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 onesided
prim::Constant          op_0        0 1 11 value=0
aten::size              op_1        2 1 waveform 11 12
prim::NumToTensor       op_2        1 1 12 13
aten::Int               op_3        1 1 13 18
prim::Constant          op_4        0 1 15 value=-1
prim::ListConstruct     op_5        2 1 15 18 19
aten::reshape           op_6        2 1 waveform 19 waveform.1
prim::Constant          op_7        0 1 108 value=0
aten::size              op_8        2 1 waveform.1 108 22
prim::NumToTensor       op_9        1 1 22 23
aten::Int               op_10       1 1 23 26
prim::Constant          op_11       0 1 28 value=1
aten::size              op_12       2 1 waveform.1 28 29
prim::NumToTensor       op_13       1 1 29 30
aten::Int               op_14       1 1 30 35
prim::Constant          op_15       0 1 109 value=1
prim::ListConstruct     op_16       3 1 109 26 35 36
aten::view              op_17       2 1 waveform.1 36 input0.1
prim::Constant          op_18       0 1 39 value=%pad_left
prim::Constant          op_19       0 1 110 value=%pad_right
prim::ListConstruct     op_20       2 1 39 110 40
prim::Constant          op_21       0 1 41 value=%pad_mode
prim::Constant          op_22       0 1 111 value=None
aten::pad               op_23       4 1 input0.1 40 41 111 input1.1
prim::Constant          op_24       0 1 112 value=1
aten::size              op_25       2 1 input1.1 112 45
prim::NumToTensor       op_26       1 1 45 46
aten::Int               op_27       1 1 46 49
prim::Constant          op_28       0 1 51 value=2
aten::size              op_29       2 1 input1.1 51 52
prim::NumToTensor       op_30       1 1 52 53
aten::Int               op_31       1 1 53 58
prim::ListConstruct     op_32       2 1 49 58 59
aten::view              op_33       2 1 input1.1 59 input2.1
prim::Constant          op_34       0 1 normalized value=%normalized
prim::Constant          op_35       0 1 return_complex value=True
aten::stft              op_36       8 1 input2.1 n_fft hop_length win_length window normalized onesided return_complex spec_f.1
prim::Constant          op_37       0 1 117 value=1
aten::size              op_38       2 1 spec_f.1 117 69
prim::NumToTensor       op_39       1 1 69 70
aten::Int               op_40       1 1 70 73
prim::Constant          op_50       0 1 118 value=2
aten::size              op_51       2 1 spec_f.1 118 75
prim::NumToTensor       op_52       1 1 75 76
aten::Int               op_53       1 1 76 81
prim::ListConstruct     op_54       2 1 73 81 82
aten::reshape           op_55       2 1 spec_f.1 82 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_1_1, 6)

class torchaudio_F_spectrogram_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
14 13
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 onesided
torchaudio.functional.spectrogram op_0 6 1 waveform n_fft hop_length win_length window onesided spec power=None normalized=False center=%center pad=%pad pad_mode=%pad_mode
prim::Constant          op_1        0 1 92 value=2.000000e+00
aten::pow               op_2        2 1 window 92 93
prim::Constant          op_3        0 1 127 value=None
aten::sum               op_4        2 1 93 127 95
aten::sqrt              op_5        1 1 95 96
aten::div               op_6        2 1 spec 96 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchaudio.functional.spectrogram";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["pad"] = captured_params.at("pad");
        op->params["pad_mode"] = captured_params.at("pad_mode");
        op->params["center"] = captured_params.at("center");
        op->params["power"] = Parameter();
        op->params["normalized"] = "window";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_2, 7)

class torchaudio_F_spectrogram_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 onesided
torchaudio.functional.spectrogram op_0 6 1 waveform n_fft hop_length win_length window onesided spec power=None normalized=%normalized center=%center pad=%pad pad_mode=%pad_mode
aten::abs               op_1        1 1 spec out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchaudio.functional.spectrogram";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["pad"] = captured_params.at("pad");
        op->params["pad_mode"] = captured_params.at("pad_mode");
        op->params["center"] = captured_params.at("center");
        op->params["normalized"] = captured_params.at("normalized");
        op->params["power"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_3, 8)

class torchaudio_F_spectrogram_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
10 9
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 onesided
torchaudio.functional.spectrogram op_0 6 1 waveform n_fft hop_length win_length window onesided spec power=1 normalized=%normalized center=%center pad=%pad pad_mode=%pad_mode
prim::Constant          op_1        0 1 391 value=2
aten::pow               op_2        2 1 spec 391 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchaudio.functional.spectrogram";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["pad"] = captured_params.at("pad");
        op->params["pad_mode"] = captured_params.at("pad_mode");
        op->params["center"] = captured_params.at("center");
        op->params["normalized"] = captured_params.at("normalized");
        op->params["power"] = 2;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_4, 9)

} // namespace pnnx
