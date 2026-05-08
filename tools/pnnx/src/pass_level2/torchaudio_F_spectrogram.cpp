// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class torchaudio_F_spectrogram : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
12 11
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 window
Tensor.size             op_0        1 1 waveform 18 dim=0
prim::Constant          op_1        0 1 15 value=-1
prim::ListConstruct     op_2        2 1 15 18 19
Tensor.reshape          op_3        2 1 waveform 19 waveform.1
torch.stft              op_4        2 1 waveform.1 window spec_f.1 n_fft=%n_fft hop_length=%hop_length win_length=%win_length normalized=%normalized center=%center pad_mode=%pad_mode onesided=%onesided return_complex=True
Tensor.size             op_5        1 1 spec_f.1 34 dim=1
Tensor.size             op_6        1 1 spec_f.1 43 dim=2
prim::ListConstruct     op_7        2 1 34 43 44
Tensor.reshape          op_8        2 1 spec_f.1 44 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchaudio.functional.spectrogram";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        GraphRewriterPass::write(op, captured_params);

        op->params["pad"] = 0;
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

class torchaudio_F_spectrogram_0 : public torchaudio_F_spectrogram
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
13 12
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 window
Tensor.size             op_0        1 1 waveform 16 dim=0
Tensor.size             op_1        1 1 waveform 25 dim=1
prim::Constant          op_2        0 1 22 value=-1
prim::ListConstruct     op_3        2 1 22 25 26
Tensor.reshape          op_4        2 1 waveform 26 waveform.1
torch.stft              op_5        2 1 waveform.1 window spec_f.1 n_fft=%n_fft hop_length=%hop_length win_length=%win_length normalized=%normalized center=%center pad_mode=%pad_mode onesided=%onesided return_complex=True
Tensor.size             op_6        1 1 spec_f.1 40 dim=1
Tensor.size             op_7        1 1 spec_f.1 50 dim=2
prim::ListConstruct     op_8        3 1 16 40 50 51
Tensor.reshape          op_9        2 1 spec_f.1 51 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram, 140)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_0, 140)

class torchaudio_F_spectrogram_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
23 22
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 window
Tensor.size             op_0        1 1 waveform 153 dim=0
Tensor.size             op_1        1 1 waveform 159 dim=1
prim::Constant          op_2        0 1 655 value=-1
prim::ListConstruct     op_3        2 1 655 159 165
Tensor.reshape          op_4        2 1 waveform 165 input4.1
Tensor.size             op_5        1 1 input4.1 168 dim=0
Tensor.size             op_6        1 1 input4.1 174 dim=1
prim::Constant          op_7        0 1 658 value=1
prim::ListConstruct     op_8        3 1 658 168 174 181
Tensor.reshape          op_9        2 1 input4.1 181 input5.1
F.pad                   op_10       1 1 input5.1 input6.1 mode=constant pad=(%pad,%pad) value=0.000000e+00
Tensor.size             op_11       1 1 input6.1 188 dim=1
Tensor.size             op_12       1 1 input6.1 194 dim=2
prim::ListConstruct     op_13       2 1 188 194 201
Tensor.reshape          op_14       2 1 input6.1 201 input7.1
torch.stft              op_15       2 1 input7.1 window spec_f3.1 n_fft=%n_fft hop_length=%hop_length win_length=%win_length normalized=%normalized center=%center pad_mode=%pad_mode onesided=%onesided return_complex=True
Tensor.size             op_16       1 1 spec_f3.1 211 dim=1
Tensor.size             op_17       1 1 spec_f3.1 217 dim=2
prim::ListConstruct     op_18       3 1 153 211 217 225
Tensor.reshape          op_19       2 1 spec_f3.1 225 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchaudio.functional.spectrogram";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        GraphRewriterPass::write(op, captured_params);

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

class torchaudio_F_spectrogram_1_1 : public torchaudio_F_spectrogram_1
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
22 21
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 window
Tensor.size             op_0        1 1 waveform 445 dim=0
prim::Constant          op_1        0 1 745 value=-1
prim::ListConstruct     op_2        2 1 745 445 451
Tensor.reshape          op_3        2 1 waveform 451 input17.1
Tensor.size             op_4        1 1 input17.1 454 dim=0
Tensor.size             op_5        1 1 input17.1 460 dim=1
prim::Constant          op_6        0 1 748 value=1
prim::ListConstruct     op_7        3 1 748 454 460 467
Tensor.reshape          op_8        2 1 input17.1 467 input18.1
F.pad                   op_9        1 1 input18.1 input19.1 mode=constant pad=(%pad,%pad) value=0.000000e+00
Tensor.size             op_10       1 1 input19.1 473 dim=1
Tensor.size             op_11       1 1 input19.1 479 dim=2
prim::ListConstruct     op_12       2 1 473 479 486
Tensor.reshape          op_13       2 1 input19.1 486 input20.1
torch.stft              op_14       2 1 input20.1 window spec_f12.1 n_fft=%n_fft hop_length=%hop_length win_length=%win_length normalized=%normalized center=%center pad_mode=%pad_mode onesided=%onesided return_complex=True
Tensor.size             op_15       1 1 spec_f12.1 495 dim=1
Tensor.size             op_16       1 1 spec_f12.1 501 dim=2
prim::ListConstruct     op_17       2 1 495 501 508
Tensor.reshape          op_18       2 1 spec_f12.1 508 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_1, 141)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_1_1, 141)

class torchaudio_F_spectrogram_1_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 window
F.pad                   op_0        1 1 waveform waveform.1 mode=constant pad=(%pad,%pad) value=0.000000e+00
torchaudio.functional.spectrogram op_1 2 1 waveform.1 window out n_fft=%n_fft hop_length=%hop_length win_length=%win_length normalized=%normalized center=%center pad=0 pad_mode=%pad_mode onesided=%onesided power=None
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchaudio.functional.spectrogram";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        GraphRewriterPass::write(op, captured_params);

        op->params["power"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_1_2, 141)

class torchaudio_F_spectrogram_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 window
torchaudio.functional.spectrogram op_0 2 1 waveform window spec n_fft=%n_fft hop_length=%hop_length win_length=%win_length normalized=False center=%center pad=%pad pad_mode=%pad_mode onesided=%onesided power=None
prim::Constant          op_1        0 1 92 value=2.000000e+00
aten::pow               op_2        2 1 window 92 93
torch.sum               op_3        1 1 93 95
aten::sqrt              op_4        1 1 95 96
aten::div               op_5        2 1 spec 96 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torchaudio.functional.spectrogram";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        GraphRewriterPass::write(op, captured_params);

        op->params["power"] = Parameter();
        op->params["normalized"] = "window";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_2, 142)

class torchaudio_F_spectrogram_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 window
torchaudio.functional.spectrogram op_0 2 1 waveform window spec n_fft=%n_fft hop_length=%hop_length win_length=%win_length normalized=%normalized center=%center pad=%pad pad_mode=%pad_mode onesided=%onesided power=None
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
        GraphRewriterPass::write(op, captured_params);

        op->params["power"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_3, 143)

class torchaudio_F_spectrogram_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 waveform
pnnx.Input              input_1     0 1 window
torchaudio.functional.spectrogram op_0 2 1 waveform window spec n_fft=%n_fft hop_length=%hop_length win_length=%win_length normalized=%normalized center=%center pad=%pad pad_mode=%pad_mode onesided=%onesided power=1
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
        GraphRewriterPass::write(op, captured_params);

        op->params["power"] = 2;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_4, 144)

} // namespace pnnx
