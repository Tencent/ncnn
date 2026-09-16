// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

static void write_stft_spectrogram(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs, int power, int normalized);
static int detect_window_type(const std::vector<float>& window_data);

// pt2 (torch.export) materializes the torchaudio.functional frontend instead of
// keeping it as one op: the caller's batch pack/unpack appears as a Reshape
// around the stft result and the magnitude/normalization tail is plain
// operators. ncnn's Spectrogram layer only reads bottom_blob.w and only emits
// the natural frequency/frame layout, so such a Reshape may be folded away
// exactly when it adds or removes leading singleton dims (a batch of one). a
// reshape that regroups the frequency/frame axes must keep its own Reshape.
static bool is_batch_singleton_reshape(const std::vector<int>& ishape, const std::vector<int>& oshape)
{
    if (ishape.empty() || oshape.empty())
        return false;

    size_t i = 0;
    size_t o = 0;
    while (i < ishape.size() && ishape[i] == 1)
        i++;
    while (o < oshape.size() && oshape[o] == 1)
        o++;

    if (ishape.size() - i != oshape.size() - o)
        return false;

    for (; i < ishape.size(); i++, o++)
    {
        if (ishape[i] != oshape[o])
            return false;
    }

    return true;
}

static bool match_batch_singleton_reshape(const std::map<std::string, const Operator*>& matched_operators, const char* name)
{
    std::map<std::string, const Operator*>::const_iterator it = matched_operators.find(name);
    if (it == matched_operators.end())
        return false;

    const Operator* op = it->second;
    if (op->inputs.empty() || op->outputs.empty())
        return false;

    return is_batch_singleton_reshape(op->inputs[0]->shape, op->outputs[0]->shape);
}

// the full reduction guard for the window-energy chain below: torch_sum has
// already lowered the sum to a ncnn Reduction by the time the normalized
// variants run, and only sum over every element is the window energy the layer
// pre-computes for normalized=2. a sum that selects axes is a different factor.
static bool match_full_reduction(const std::map<std::string, Parameter>& captured_params, const char* name)
{
    const std::string prefix = std::string(name) + ".";

    // ReductionOp_SUM
    std::map<std::string, Parameter>::const_iterator operation = captured_params.find(prefix + "0");
    if (operation == captured_params.end() || operation->second.type != 2 || operation->second.i != 0)
        return false;

    // reduce_all
    std::map<std::string, Parameter>::const_iterator reduce_all = captured_params.find(prefix + "1");
    if (reduce_all == captured_params.end() || reduce_all->second.type != 2 || reduce_all->second.i != 1)
        return false;

    // a dim-reduced sum carries its axes, and is a different factor
    return captured_params.find(prefix + "3") == captured_params.end();
}

// pt2 (torch.export) inlines torchaudio.functional.spectrogram, so this channel
// sees the expanded stft instead of the single functional op the torchscript
// channel matches. the variants below fold the expansions the frontend produces
// for the covered configs - complex output feeding view_as_real (power=None),
// squared magnitude (power=2), plain magnitude (power=1) and the hand-written
// window-energy normalized magnitude (power=1 with normalized='window') - and
// decline anything else, which then surfaces as a leftover operator instead of
// a silently different layout.
//
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

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<float> window_data = captured_attrs.at("op_0.data").get_float32_data();
        if (detect_window_type(window_data) == -1)
            return false;

        // the two reshapes around the pad are the frontend's batch pack and
        // unpack; absorbing the pad as centering is only equivalent while they
        // merely add or drop leading singleton dims. a reshape that regroups
        // rows changes both the padded samples and the layout, so it must keep
        // its own Reshape.
        if (!match_batch_singleton_reshape(matched_operators, "op_r1"))
            return false;
        if (!match_batch_singleton_reshape(matched_operators, "op_r2"))
            return false;

        // the pad matched here is the centering pad dynamo materialized for
        // center=True, so it may only be absorbed while the stft itself does
        // not pad; otherwise the layer would center a second time
        if (captured_params.at("center").type == 1 && captured_params.at("center").b)
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

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<float> window_data = captured_attrs.at("op_0.data").get_float32_data();
        if (detect_window_type(window_data) == -1)
            return false;

        // only the caller's batch unpack may be folded away
        return match_batch_singleton_reshape(matched_operators, "op_2");
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        // propagate the captured normalized flag (must not be hard-coded to 0
        // or the complex spectrum would be scaled wrongly)
        int normalized = captured_params.at("normalized").type == 1 && captured_params.at("normalized").b ? 1 : 0;
        write_stft_spectrogram(op, captured_params, captured_attrs, 0, normalized);
    }
};

// a real torch.export of torchaudio.functional.spectrogram(power=None) does put
// a Reshape between the complex stft and view_as_real: it is the frontend's
// batch unpack, verified above to be a leading-singleton reshape only. user
// reshapes that regroup frequency/frame axes keep the original operators.
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

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<float> window_data = captured_attrs.at("op_0.data").get_float32_data();
        if (detect_window_type(window_data) == -1)
            return false;

        // power=2 is abs(x) ** 2, so square(abs(complex)) == re*re + im*im is
        // exactly the layer's power=2 output; only the batch unpack reshapes
        return match_batch_singleton_reshape(matched_operators, "op_2");
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        // frame_length normalization (stft normalized=True) maps to Spectrogram's 1
        int normalized = captured_params.at("normalized").type == 1 && captured_params.at("normalized").b ? 1 : 0;
        write_stft_spectrogram(op, captured_params, captured_attrs, 2, normalized);
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_stft_pt2_power, 20)

// pt2: torch.stft + Reshape + BinaryOp div + UnaryOp abs, scaled by the
// torchaudio frontend's window normalization sqrt(sum(window ** 2))
class torch_stft_pt2_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
11 10
pnnx.Input              input       0 1 input
pnnx.Attribute          op_0        0 1 window @data
pnnx.Attribute          op_1        0 1 window2 @data2
torch.stft              op_2        2 1 input window a center=%center pad_mode=%pad_mode hop_length=%hop_length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=True win_length=%win_length
Reshape                 op_3        1 1 a b %*=%*
UnaryOp                 op_4        1 1 window2 square 0=4
Reduction               op_5        1 1 square sqsum %*=%*
UnaryOp                 op_6        1 1 sqsum win_norm 0=5
BinaryOp                op_7        2 1 b win_norm c 0=3
UnaryOp                 op_8        1 1 c out 0=0
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

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<float> window_data = captured_attrs.at("op_0.data").get_float32_data();
        if (detect_window_type(window_data) == -1)
            return false;
        // attribute_unpooling (pass_level5) gives every use of a constant its
        // own pnnx.Attribute, so the stft window and the operand that is
        // squared are separate nodes; the folded factor is the layer's own
        // window energy only when both carry the same window
        if (!(captured_attrs.at("op_0.data") == captured_attrs.at("op_1.data")))
            return false;

        // the layer applies one normalization only, so the hand-written
        // window-energy division can only stand in for it while the stft
        // itself is unnormalized; with normalized=True the graph normalizes by
        // sqrt(n_fft) as well and the fold would drop that factor
        if (captured_params.at("normalized").type != 0 && (captured_params.at("normalized").type != 1 || captured_params.at("normalized").b))
            return false;

        if (!match_full_reduction(captured_params, "op_5"))
            return false;

        return match_batch_singleton_reshape(matched_operators, "op_3");
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        // abs(stft / sqrt(sum(window ** 2))) is the layer's window-normalized
        // magnitude, i.e. normalized=2 with power=1
        write_stft_spectrogram(op, captured_params, captured_attrs, 1, 2);
    }
};

// priority 21: the window-energy chain holds the fully lowered Reduction, which
// only exists after the priority-20 torch_sum pass has run
REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_stft_pt2_norm, 21)

// pt2: torch.stft + Reshape + UnaryOp abs (power=1 magnitude spectrum, the
// shape the frontend produces for the default Spectrogram config). registered
// after the power variant so a squared magnitude is folded with power=2 there,
// and the layer's power=1 already emits the magnitude this tail computes.
class torch_stft_pt2_abs : public GraphRewriterPass
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
UnaryOp                 op_3        1 1 b out 0=0
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

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<float> window_data = captured_attrs.at("op_0.data").get_float32_data();
        if (detect_window_type(window_data) == -1)
            return false;

        // only the caller's batch unpack may be folded away
        return match_batch_singleton_reshape(matched_operators, "op_2");
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        // the graph's only normalization is the stft's own frame-length flag,
        // which the layer's normalized=1 applies; a window-energy division
        // would sit between the reshape and the abs and is folded separately
        int normalized = captured_params.at("normalized").type == 1 && captured_params.at("normalized").b ? 1 : 0;
        write_stft_spectrogram(op, captured_params, captured_attrs, 1, normalized);
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_stft_pt2_abs, 20)

} // namespace ncnn

} // namespace pnnx
