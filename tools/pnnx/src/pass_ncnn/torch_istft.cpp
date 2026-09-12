// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_istft : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
torch.view_as_complex   op_0        1 1 input a
torch.istft             op_1        1 1 a out center=%center hop_length=%hop_length length=%length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=False win_length=%win_length window=None
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "InverseSpectrogram";
    }

    const char* name_str() const
    {
        return "istft";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        // InverseSpectrogram always computes the natural output length, so an
        // explicit length that trims or zero-pads cannot be reproduced; keep
        // the original operators in that case
        return captured_params.at("length").type == 0;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["0"] = captured_params.at("n_fft");
        op->params["1"] = 1; // returns
        op->params["2"] = captured_params.at("hop_length");
        op->params["3"] = captured_params.at("win_length");
        op->params["4"] = 0; // all ones
        op->params["5"] = captured_params.at("center").type == 1 && captured_params.at("center").b ? 1 : 0;
        op->params["7"] = captured_params.at("normalized").type == 1 && captured_params.at("normalized").b ? 1 : 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_istft, 20)

class torch_istft_1 : public torch_istft
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
torch.view_as_complex   op_0        1 1 input a
torch.istft             op_1        1 1 a b center=%center hop_length=%hop_length length=%length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=True win_length=%win_length window=None
torch.view_as_real      op_2        1 1 b out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        torch_istft::write(op, captured_params);

        op->params["1"] = 0; // returns
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_istft_1, 20)

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

// pt2 (torch.export) materializes the torchaudio.functional frontend instead of
// keeping it as one op, so the inverse path carries the caller's batch
// pack/unpack as a Reshape on both sides of the istft, plus the plain operators
// of the window normalization. ncnn's InverseSpectrogram layer consumes the
// (freq, frame, 2) layout and only produces the natural one-dimensional wave,
// so such a Reshape may be folded away exactly when it adds or removes leading
// singleton dims (a batch of one). a reshape that regroups batch/time dims must
// keep its own Reshape.
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

// the window-energy chain below folds sqrt(sum(window ** 2)) into the layer's
// normalized=2, which is exactly what the layer pre-computes. torch_sum has
// already lowered the sum to a ncnn Reduction by the time the normalized
// variants run, and only sum over every element of the squared window is that
// factor; a sum that selects axes is a different one.
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

class torch_istft_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
torch.view_as_complex   op_0        1 1 input a
pnnx.Attribute          op_1        0 1 window @data
torch.istft             op_2        2 1 a window out center=%center hop_length=%hop_length length=%length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=False win_length=%win_length
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "InverseSpectrogram";
    }

    const char* name_str() const
    {
        return "istft";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        // InverseSpectrogram always computes the natural output length, so an
        // explicit length that trims or zero-pads cannot be reproduced; keep
        // the original operators in that case
        if (captured_params.at("length").type != 0)
            return false;

        const std::vector<float> window_data = captured_attrs.at("op_1.data").get_float32_data();
        const int window_type = detect_window_type(window_data);
        return window_type != -1;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<float> window_data = captured_attrs.at("op_1.data").get_float32_data();
        const int window_type = detect_window_type(window_data);

        op->params["0"] = captured_params.at("n_fft");
        op->params["1"] = 1; // returns
        op->params["2"] = captured_params.at("hop_length");
        op->params["3"] = captured_params.at("win_length");
        op->params["4"] = window_type;
        op->params["5"] = captured_params.at("center").type == 1 && captured_params.at("center").b ? 1 : 0;
        op->params["7"] = captured_params.at("normalized").type == 1 && captured_params.at("normalized").b ? 1 : 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_istft_2, 20)

class torch_istft_3 : public torch_istft_2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
torch.view_as_complex   op_0        1 1 input a
pnnx.Attribute          op_1        0 1 window @data
torch.istft             op_2        2 1 a window b center=%center hop_length=%hop_length length=%length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=True win_length=%win_length
torch.view_as_real      op_3        1 1 b out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        torch_istft_2::write(op, captured_params, captured_attrs);

        op->params["1"] = 0; // returns
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_istft_3, 20)

// pt2: torch.view_as_complex + Reshape + torch.istft + Reshape (inverse_spectrogram expansion chain)
class torch_istft_pt2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
torch.view_as_complex   op_0        1 1 input a
Reshape                 op_1        1 1 a b %*=%*
pnnx.Attribute          op_2        0 1 window @data
torch.istft             op_3        2 1 b window c center=%center hop_length=%hop_length length=%length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=False win_length=%win_length
Reshape                 op_4        1 1 c out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "InverseSpectrogram";
    }

    const char* name_str() const
    {
        return "istft";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        // InverseSpectrogram always computes the natural output length, so an
        // explicit length that trims or zero-pads cannot be reproduced; keep
        // the original operators in that case
        if (captured_params.at("length").type != 0)
            return false;

        const std::vector<float> window_data = captured_attrs.at("op_2.data").get_float32_data();
        const int window_type = detect_window_type(window_data);
        if (window_type == -1)
            return false;

        // both reshapes are the frontend's batch pack and unpack
        return match_batch_singleton_reshape(matched_operators, "op_1")
               && match_batch_singleton_reshape(matched_operators, "op_4");
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<float> window_data = captured_attrs.at("op_2.data").get_float32_data();
        const int window_type = detect_window_type(window_data);

        op->params["0"] = captured_params.at("n_fft");
        op->params["1"] = 1; // returns
        op->params["2"] = captured_params.at("hop_length");
        op->params["3"] = captured_params.at("win_length");
        op->params["4"] = window_type;
        op->params["5"] = captured_params.at("center").type == 1 && captured_params.at("center").b ? 1 : 0;
        op->params["7"] = captured_params.at("normalized").type == 1 && captured_params.at("normalized").b ? 1 : 0;
    }
};

// the two reshapes are the caller's batch pack/unpack of the inverse
// spectrogram frontend, verified above to be leading-singleton only
REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_istft_pt2, 20)

// pt2: torch.view_as_complex + BinaryOp mul + Reshape + torch.istft + Reshape,
// where the mul factor is the torchaudio frontend's window denormalization
// sqrt(sum(window ** 2)) built from the stft window attribute
class torch_istft_pt2_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
12 11
pnnx.Input              input       0 1 input
pnnx.Attribute          op_2        0 1 window @data
pnnx.Attribute          op_3        0 1 window2 @data2
torch.view_as_complex   op_0        1 1 input a
UnaryOp                 op_5        1 1 window2 square 0=4
Reduction               op_6        1 1 square sqsum %*=%*
UnaryOp                 op_7        1 1 sqsum win_norm 0=5
BinaryOp                op_mul      2 1 a win_norm b 0=2
Reshape                 op_1        1 1 b c %*=%*
torch.istft             op_4        2 1 c window d center=%center hop_length=%hop_length length=%length n_fft=%n_fft normalized=%normalized onesided=%onesided return_complex=False win_length=%win_length
Reshape                 op_8        1 1 d out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "InverseSpectrogram";
    }

    const char* name_str() const
    {
        return "istft";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        // InverseSpectrogram always computes the natural output length, so an
        // explicit length that trims or zero-pads cannot be reproduced; keep
        // the original operators in that case
        if (captured_params.at("length").type != 0)
            return false;

        const std::vector<float> window_data = captured_attrs.at("op_2.data").get_float32_data();
        if (detect_window_type(window_data) == -1)
            return false;

        // attribute_unpooling (pass_level5) gives every use of a constant its
        // own pnnx.Attribute, so the istft window and the operand that is
        // squared are separate nodes; the denormalization the layer reproduces
        // for normalized=2 uses the istft window, so both must carry the same
        // window
        if (!(captured_attrs.at("op_2.data") == captured_attrs.at("op_3.data")))
            return false;

        if (!match_full_reduction(captured_params, "op_6"))
            return false;

        // the frontend only denormalizes for normalized='window', which it does
        // by hand before calling the (un-normalized) istft
        if (captured_params.at("normalized").type != 0 && (captured_params.at("normalized").type != 1 || captured_params.at("normalized").b))
            return false;

        return match_batch_singleton_reshape(matched_operators, "op_1")
               && match_batch_singleton_reshape(matched_operators, "op_8");
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const std::vector<float> window_data = captured_attrs.at("op_2.data").get_float32_data();
        const int window_type = detect_window_type(window_data);

        op->params["0"] = captured_params.at("n_fft");
        op->params["1"] = 1; // returns
        op->params["2"] = captured_params.at("hop_length");
        op->params["3"] = captured_params.at("win_length");
        op->params["4"] = window_type;
        op->params["5"] = captured_params.at("center").type == 1 && captured_params.at("center").b ? 1 : 0;
        op->params["7"] = 2; // window normalization
    }
};

// priority 21: the window-energy chain holds the fully lowered Reduction, which
// only exists after the priority-20 torch_sum pass has run
REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_istft_pt2_norm, 21)

} // namespace ncnn

} // namespace pnnx
