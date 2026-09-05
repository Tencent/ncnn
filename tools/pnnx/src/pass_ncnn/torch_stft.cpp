// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

static void write_stft_spectrogram(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs, int power, int normalized);
static int detect_window_type(const std::vector<float>& window_data);

// pt2: stft preceded by an explicit reshape + F.pad (center pad expanded); absorb
// into Spectrogram(center=True). Must match before torch_stft_pt2_complex (stft
// with leading structure).
class torch_stft_pt2_pad_complex : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input       0 1 input
pnnx.Attribute          op_0        0 1 window @data
Reshape                 op_r1       1 1 input r1 %*=%*
Padding                 op_pad      1 1 r1 r2 %*=%*
Reshape                 op_r2       1 1 r2 r3 %*=%*
torch.stft              op_1        2 1 r3 window a center=%center hop_length=%hop_length n_fft=%n_fft normalized=%normalized onesided=%onesided pad_mode=%pad_mode return_complex=True win_length=%win_length
torch.view_as_real      op_3        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Spectrogram";
    }

    const char* name_str() const
    {
        return "stft";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<float> window_data = captured_attrs.at("op_0.data").get_float32_data();
        if (detect_window_type(window_data) == -1)
            return false;

        // only absorb the leading pad when it exactly implements STFT
        // centering (pad n_fft//2 on both sides of the time axis, with a pad
        // mode matching torch.stft's pad_mode). dynamo expands center=True into
        // a constant reflect pad; a user-written F.pad of arbitrary width or
        // mode must NOT be reinterpreted as centering, otherwise the frame
        // count and the padded samples both change.
        const int n_fft = captured_params.at("n_fft").i;
        const int half = n_fft / 2;

        // ncnn Padding params captured as "op_pad.N": 0=top 1=bottom 2=left
        // 3=right 4=type (0 constant / 1 replicate / 2 reflect)
        int pad_top = 0;
        int pad_bottom = 0;
        int pad_left = 0;
        int pad_right = 0;
        int pad_type = 0;
        {
            std::map<std::string, Parameter>::const_iterator it;
            if ((it = captured_params.find("op_pad.0")) != captured_params.end())
                pad_top = it->second.i;
            if ((it = captured_params.find("op_pad.1")) != captured_params.end())
                pad_bottom = it->second.i;
            if ((it = captured_params.find("op_pad.2")) != captured_params.end())
                pad_left = it->second.i;
            if ((it = captured_params.find("op_pad.3")) != captured_params.end())
                pad_right = it->second.i;
            if ((it = captured_params.find("op_pad.4")) != captured_params.end())
                pad_type = it->second.i;
        }

        // stft pads on the time (last) axis only, so a single axis may carry
        // the centering pad; the other axes must be unpadded
        const bool left_right_center = (pad_top == 0 && pad_bottom == 0 && pad_left == half && pad_right == half);
        const bool top_bottom_center = (pad_left == 0 && pad_right == 0 && pad_top == half && pad_bottom == half);
        if (!left_right_center && !top_bottom_center)
            return false;

        // pad mode must match the stft pad_mode (dynamo lowers center pad with
        // the same mode torch.stft would have used)
        const std::string& pad_mode = captured_params.at("pad_mode").s;
        int expect_type = 2;
        if (pad_mode == "constant")
            expect_type = 0;
        if (pad_mode == "replicate")
            expect_type = 1;
        if (pad_mode == "reflect")
            expect_type = 2;
        if (pad_type != expect_type)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        int normalized = captured_params.at("normalized").type == 1 && captured_params.at("normalized").b ? 1 : 0;
        write_stft_spectrogram(op, captured_params, captured_attrs, 0, normalized);
        // absorb the leading F.pad (center pad n_fft//2)
        op->params["5"] = 1; // center=True
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_stft_pt2_pad_complex, 20)
class torch_stft : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
torch.stft              op_0        1 1 input a center=%center pad_mode=%pad_mode hop_length=%hop_length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=True win_length=%win_length window=None
torch.view_as_real      op_1        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Spectrogram";
    }

    const char* name_str() const
    {
        return "stft";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& pad_mode = captured_params.at("pad_mode").s;
        int pad_type = 2;
        if (pad_mode == "constant")
            pad_type = 0;
        if (pad_mode == "replicate")
            pad_type = 1;
        if (pad_mode == "reflect")
            pad_type = 2;
        const int onesided = captured_params.at("onesided").type == 1 && captured_params.at("onesided").b == false ? 0 : 1;

        op->params["0"] = captured_params.at("n_fft");
        op->params["1"] = 0; // power
        op->params["2"] = captured_params.at("hop_length");
        op->params["3"] = captured_params.at("win_length");
        op->params["4"] = 0; // all ones
        op->params["5"] = captured_params.at("center").type == 1 && captured_params.at("center").b ? 1 : 0;
        op->params["6"] = pad_type;
        op->params["7"] = captured_params.at("normalized").type == 1 && captured_params.at("normalized").b ? 1 : 0;
        op->params["8"] = onesided;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_stft, 20)

static bool NearlyEqual(float a, float b, float epsilon)
{
    if (a == b)
        return true;

    float diff = (float)fabs(a - b);
    if (diff <= epsilon)
        return true;

    // relative error
    return diff < epsilon * std::max(fabs(a), fabs(b));
}

static int detect_window_type(const std::vector<float>& window_data)
{
    const int winlen = (int)window_data.size();

    bool is_one = true;
    bool is_hann = true;
    bool is_hamming = true;
    for (int i = 0; i < winlen; i++)
    {
        if (!NearlyEqual(window_data[i], 1.f, 0.001))
            is_one = false;

        if (!NearlyEqual(window_data[i], 0.5f * (1 - cos(2 * 3.14159265358979323846 * i / winlen)), 0.001))
            is_hann = false;

        if (!NearlyEqual(window_data[i], 0.54f - 0.46f * cos(2 * 3.14159265358979323846 * i / winlen), 0.001))
            is_hamming = false;
    }

    if (is_one)
        return 0;
    if (is_hann)
        return 1;
    if (is_hamming)
        return 2;

    return -1;
}

class torch_stft_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_0        0 1 window @data
torch.stft              op_1        2 1 input window a center=%center pad_mode=%pad_mode hop_length=%hop_length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=True win_length=%win_length
torch.view_as_real      op_2        1 1 a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Spectrogram";
    }

    const char* name_str() const
    {
        return "stft";
    }

    bool match(const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<float> window_data = captured_attrs.at("op_0.data").get_float32_data();
        const int window_type = detect_window_type(window_data);
        return window_type != -1;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<float> window_data = captured_attrs.at("op_0.data").get_float32_data();
        const int window_type = detect_window_type(window_data);

        const std::string& pad_mode = captured_params.at("pad_mode").s;
        int pad_type = 2;
        if (pad_mode == "constant")
            pad_type = 0;
        if (pad_mode == "replicate")
            pad_type = 1;
        if (pad_mode == "reflect")
            pad_type = 2;
        const int onesided = captured_params.at("onesided").type == 1 && captured_params.at("onesided").b == false ? 0 : 1;

        op->params["0"] = captured_params.at("n_fft");
        op->params["1"] = 0; // power
        op->params["2"] = captured_params.at("hop_length");
        op->params["3"] = captured_params.at("win_length");
        op->params["4"] = window_type;
        op->params["5"] = captured_params.at("center").type == 1 && captured_params.at("center").b ? 1 : 0;
        op->params["6"] = pad_type;
        op->params["7"] = captured_params.at("normalized").type == 1 && captured_params.at("normalized").b ? 1 : 0;
        op->params["8"] = onesided;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_stft_1, 20)

// common write: map stft params to Spectrogram layer params
static void write_stft_spectrogram(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs, int power, int normalized)
{
    const std::vector<float> window_data = captured_attrs.at("op_0.data").get_float32_data();
    const int window_type = detect_window_type(window_data);

    const std::string& pad_mode = captured_params.at("pad_mode").s;
    int pad_type = 2;
    if (pad_mode == "constant")
        pad_type = 0;
    if (pad_mode == "replicate")
        pad_type = 1;
    if (pad_mode == "reflect")
        pad_type = 2;
    const int onesided = captured_params.at("onesided").type == 1 && captured_params.at("onesided").b == false ? 0 : 1;

    op->params["0"] = captured_params.at("n_fft");
    op->params["1"] = power;
    op->params["2"] = captured_params.at("hop_length");
    op->params["3"] = captured_params.at("win_length");
    op->params["4"] = window_type;
    op->params["5"] = captured_params.at("center").type == 1 && captured_params.at("center").b ? 1 : 0;
    op->params["6"] = pad_type;
    op->params["7"] = normalized;
    op->params["8"] = onesided;
}

// pt2: torch.stft + Reshape + torch.view_as_real (complex output, reshape in between)

class torch_stft_pt2_complex : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
pnnx.Attribute          op_0        0 1 window @data
torch.stft              op_1        2 1 input window a center=%center pad_mode=%pad_mode hop_length=%hop_length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=True win_length=%win_length
Reshape                 op_2        1 1 a b %*=%*
torch.view_as_real      op_3        1 1 b out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Spectrogram";
    }

    const char* name_str() const
    {
        return "stft";
    }

    bool match(const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<float> window_data = captured_attrs.at("op_0.data").get_float32_data();
        return detect_window_type(window_data) != -1;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        // propagate the captured normalized flag (must not be hard-coded to 0
        // or the complex spectrum would be scaled wrongly)
        int normalized = captured_params.at("normalized").type == 1 && captured_params.at("normalized").b ? 1 : 0;
        write_stft_spectrogram(op, captured_params, captured_attrs, 0, normalized);
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_stft_pt2_complex, 20)

// pt2: torch.stft + Reshape + UnaryOp abs + UnaryOp square (power=2 spectrum)
class torch_stft_pt2_power : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
pnnx.Attribute          op_0        0 1 window @data
torch.stft              op_1        2 1 input window a center=%center pad_mode=%pad_mode hop_length=%hop_length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=True win_length=%win_length
Reshape                 op_2        1 1 a b %*=%*
UnaryOp                 op_3        1 1 b c 0=0
UnaryOp                 op_4        1 1 c out 0=4
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Spectrogram";
    }

    const char* name_str() const
    {
        return "stft";
    }

    bool match(const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<float> window_data = captured_attrs.at("op_0.data").get_float32_data();
        return detect_window_type(window_data) != -1;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        // frame_length normalization (normalized=True) maps to Spectrogram's 1
        int normalized = captured_params.at("normalized").type == 1 && captured_params.at("normalized").b ? 1 : 0;
        write_stft_spectrogram(op, captured_params, captured_attrs, 2, normalized);
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_stft_pt2_power, 20)

// pt2: torch.stft + Reshape + UnaryOp abs(div(@0,sqrt(@1))) (window-normalized magnitude spectrum)
class torch_stft_pt2_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input       0 1 input
pnnx.Attribute          op_0        0 1 window @data
torch.stft              op_1        2 1 input window a center=%center pad_mode=%pad_mode hop_length=%hop_length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=True win_length=%win_length
Reshape                 op_2        1 1 a b %*=%*
pnnx.Input              norm        0 1 norm
UnaryOp                 op_3        1 1 norm sqrt_out 0=5
BinaryOp                op_4        2 1 b sqrt_out d 0=3
UnaryOp                 op_5        1 1 d out 0=0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Spectrogram";
    }

    const char* name_str() const
    {
        return "stft";
    }

    bool match(const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<float> window_data = captured_attrs.at("op_0.data").get_float32_data();
        return detect_window_type(window_data) != -1;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        // window normalization (normalized='window') maps to Spectrogram's 2
        write_stft_spectrogram(op, captured_params, captured_attrs, 1, 2);
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_stft_pt2_norm, 20)

} // namespace ncnn

} // namespace pnnx
