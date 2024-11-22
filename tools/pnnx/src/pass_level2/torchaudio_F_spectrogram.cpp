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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram, 140)

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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_0, 140)

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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_1, 140)

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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_1_1, 140)

class torchaudio_F_spectrogram_1_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
52 51
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 onesided
prim::Constant          op_0        0 1 211 value=0
aten::size              op_1        2 1 waveform 211 107
prim::NumToTensor       op_2        1 1 107 108
aten::Int               op_3        1 1 108 112
prim::Constant          op_4        0 1 212 value=-1
prim::ListConstruct     op_5        2 1 212 112 113
aten::reshape           op_6        2 1 waveform 113 input3.1
prim::Constant          op_7        0 1 213 value=0
aten::size              op_8        2 1 input3.1 213 116
prim::NumToTensor       op_9        1 1 116 117
aten::Int               op_10       1 1 117 120
prim::Constant          op_11       0 1 214 value=1
aten::size              op_12       2 1 input3.1 214 122
prim::NumToTensor       op_13       1 1 122 123
aten::Int               op_14       1 1 123 128
prim::Constant          op_15       0 1 215 value=1
prim::ListConstruct     op_16       3 1 215 120 128 129
aten::view              op_17       2 1 input3.1 129 input4.1
prim::Constant          op_18       0 1 216 value=%pad_left
prim::Constant          op_19       0 1 217 value=%pad_right
prim::ListConstruct     op_20       2 1 216 217 132
aten::reflection_pad1d  op_21       2 1 input4.1 132 input5.1
prim::Constant          op_22       0 1 218 value=1
aten::size              op_23       2 1 input5.1 218 135
prim::NumToTensor       op_24       1 1 135 136
aten::Int               op_25       1 1 136 139
prim::Constant          op_26       0 1 219 value=2
aten::size              op_27       2 1 input5.1 219 141
prim::NumToTensor       op_28       1 1 141 142
aten::Int               op_29       1 1 142 147
prim::ListConstruct     op_30       2 1 139 147 148
aten::view              op_31       2 1 input5.1 148 input6.1
prim::Constant          op_32       0 1 normalized value=%normalized
prim::Constant          op_33       0 1 return_complex value=True
aten::stft              op_34       8 1 input6.1 n_fft hop_length win_length window normalized onesided return_complex spec_f2.1
prim::Constant          op_35       0 1 226 value=1
aten::size              op_36       2 1 spec_f2.1 226 157
prim::NumToTensor       op_37       1 1 157 158
aten::Int               op_38       1 1 158 161
prim::Constant          op_39       0 1 227 value=2
aten::size              op_40       2 1 spec_f2.1 227 163
prim::NumToTensor       op_41       1 1 163 164
aten::Int               op_42       1 1 164 169
prim::ListConstruct     op_43       2 1 161 169 170
aten::reshape           op_44       2 1 spec_f2.1 170 out
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_1_2, 140)

class torchaudio_F_spectrogram_1_3 : public torchaudio_F_spectrogram_1_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
56 55
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
aten::reshape           op_10       2 1 waveform 26 input.1
prim::Constant          op_11       0 1 326 value=0
aten::size              op_12       2 1 input.1 326 29
prim::NumToTensor       op_13       1 1 29 30
aten::Int               op_14       1 1 30 33
prim::Constant          op_15       0 1 327 value=1
aten::size              op_16       2 1 input.1 327 35
prim::NumToTensor       op_17       1 1 35 36
aten::Int               op_18       1 1 36 41
prim::Constant          op_19       0 1 328 value=1
prim::ListConstruct     op_20       3 1 328 33 41 42
aten::view              op_21       2 1 input.1 42 input0.1
prim::Constant          op_22       0 1 45 value=%pad_left
prim::Constant          op_23       0 1 329 value=%pad_right
prim::ListConstruct     op_24       2 1 45 329 46
aten::reflection_pad1d  op_25       2 1 input0.1 46 input1.1
prim::Constant          op_26       0 1 330 value=1
aten::size              op_27       2 1 input1.1 330 49
prim::NumToTensor       op_28       1 1 49 50
aten::Int               op_29       1 1 50 53
prim::Constant          op_30       0 1 55 value=2
aten::size              op_31       2 1 input1.1 55 56
prim::NumToTensor       op_32       1 1 56 57
aten::Int               op_33       1 1 57 62
prim::ListConstruct     op_34       2 1 53 62 63
aten::view              op_35       2 1 input1.1 63 input2.1
prim::Constant          op_36       0 1 normalized value=%normalized
prim::Constant          op_37       0 1 return_complex value=True
aten::stft              op_38       8 1 input2.1 n_fft hop_length win_length window normalized onesided return_complex spec_f.1
prim::Constant          op_39       0 1 334 value=1
aten::size              op_40       2 1 spec_f.1 334 74
prim::NumToTensor       op_41       1 1 74 75
aten::Int               op_42       1 1 75 78
prim::Constant          op_43       0 1 335 value=2
aten::size              op_44       2 1 spec_f.1 335 80
prim::NumToTensor       op_45       1 1 80 81
aten::Int               op_46       1 1 81 87
prim::ListConstruct     op_47       3 1 16 78 87 88
aten::reshape           op_48       2 1 spec_f.1 88 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_1_3, 140)

class torchaudio_F_spectrogram_1_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
53 52
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 n_fft
pnnx.Input              input_2     0 1 hop_length
pnnx.Input              input_3     0 1 win_length
pnnx.Input              input_4     0 1 window
pnnx.Input              input_5     0 1 onesided
prim::Constant          op_0        0 1 211 value=0
aten::size              op_1        2 1 waveform 211 107
prim::NumToTensor       op_2        1 1 107 108
aten::Int               op_3        1 1 108 112
prim::Constant          op_4        0 1 212 value=-1
prim::ListConstruct     op_5        2 1 212 112 113
aten::reshape           op_6        2 1 waveform 113 input3.1
prim::Constant          op_7        0 1 213 value=0
aten::size              op_8        2 1 input3.1 213 116
prim::NumToTensor       op_9        1 1 116 117
aten::Int               op_10       1 1 117 120
prim::Constant          op_11       0 1 214 value=1
aten::size              op_12       2 1 input3.1 214 122
prim::NumToTensor       op_13       1 1 122 123
aten::Int               op_14       1 1 123 128
prim::Constant          op_15       0 1 215 value=1
prim::ListConstruct     op_16       3 1 215 120 128 129
aten::view              op_17       2 1 input3.1 129 input4.1
prim::Constant          op_18       0 1 216 value=%pad_left
prim::Constant          op_19       0 1 217 value=%pad_right
prim::ListConstruct     op_20       2 1 216 217 132
prim::Constant          op_21       0 1 46 value=0.000000e+00
aten::constant_pad_nd   op_22       3 1 input4.1 132 46 input5.1
prim::Constant          op_23       0 1 218 value=1
aten::size              op_24       2 1 input5.1 218 135
prim::NumToTensor       op_25       1 1 135 136
aten::Int               op_26       1 1 136 139
prim::Constant          op_27       0 1 219 value=2
aten::size              op_28       2 1 input5.1 219 141
prim::NumToTensor       op_29       1 1 141 142
aten::Int               op_30       1 1 142 147
prim::ListConstruct     op_31       2 1 139 147 148
aten::view              op_32       2 1 input5.1 148 input6.1
prim::Constant          op_33       0 1 normalized value=%normalized
prim::Constant          op_34       0 1 return_complex value=True
aten::stft              op_35       8 1 input6.1 n_fft hop_length win_length window normalized onesided return_complex spec_f2.1
prim::Constant          op_36       0 1 226 value=1
aten::size              op_37       2 1 spec_f2.1 226 157
prim::NumToTensor       op_38       1 1 157 158
aten::Int               op_39       1 1 158 161
prim::Constant          op_40       0 1 227 value=2
aten::size              op_41       2 1 spec_f2.1 227 163
prim::NumToTensor       op_42       1 1 163 164
aten::Int               op_43       1 1 164 169
prim::ListConstruct     op_44       2 1 161 169 170
aten::reshape           op_45       2 1 spec_f2.1 170 out
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
        op->params["pad_mode"] = "constant";
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_1_4, 140)

class torchaudio_F_spectrogram_1_5 : public torchaudio_F_spectrogram_1_4
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
57 56
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
aten::reshape           op_10       2 1 waveform 26 input.1
prim::Constant          op_11       0 1 326 value=0
aten::size              op_12       2 1 input.1 326 29
prim::NumToTensor       op_13       1 1 29 30
aten::Int               op_14       1 1 30 33
prim::Constant          op_15       0 1 327 value=1
aten::size              op_16       2 1 input.1 327 35
prim::NumToTensor       op_17       1 1 35 36
aten::Int               op_18       1 1 36 41
prim::Constant          op_19       0 1 328 value=1
prim::ListConstruct     op_20       3 1 328 33 41 42
aten::view              op_21       2 1 input.1 42 input0.1
prim::Constant          op_22       0 1 45 value=%pad_left
prim::Constant          op_23       0 1 329 value=%pad_right
prim::ListConstruct     op_24       2 1 45 329 46
prim::Constant          op_25       0 1 47 value=0.000000e+00
aten::constant_pad_nd   op_26       3 1 input0.1 46 47 input1.1
prim::Constant          op_27       0 1 330 value=1
aten::size              op_28       2 1 input1.1 330 49
prim::NumToTensor       op_29       1 1 49 50
aten::Int               op_30       1 1 50 53
prim::Constant          op_31       0 1 55 value=2
aten::size              op_32       2 1 input1.1 55 56
prim::NumToTensor       op_33       1 1 56 57
aten::Int               op_34       1 1 57 62
prim::ListConstruct     op_35       2 1 53 62 63
aten::view              op_36       2 1 input1.1 63 input2.1
prim::Constant          op_37       0 1 normalized value=%normalized
prim::Constant          op_38       0 1 return_complex value=True
aten::stft              op_39       8 1 input2.1 n_fft hop_length win_length window normalized onesided return_complex spec_f.1
prim::Constant          op_40       0 1 334 value=1
aten::size              op_41       2 1 spec_f.1 334 74
prim::NumToTensor       op_42       1 1 74 75
aten::Int               op_43       1 1 75 78
prim::Constant          op_44       0 1 335 value=2
aten::size              op_45       2 1 spec_f.1 335 80
prim::NumToTensor       op_46       1 1 80 81
aten::Int               op_47       1 1 81 87
prim::ListConstruct     op_48       3 1 16 78 87 88
aten::reshape           op_49       2 1 spec_f.1 88 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_1_5, 140)

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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_2, 141)

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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_3, 142)

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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_4, 143)

} // namespace pnnx
