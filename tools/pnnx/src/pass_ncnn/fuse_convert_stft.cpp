// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_convert_stft.h"

#include "pass_level2.h"

namespace pnnx {

namespace ncnn {

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

class fuse_stft_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
pnnx.Attribute          op_0        0 1 window @data
torch.stft              op_1        2 1 input window a center=%center hop_length=%hop_length n_fft=%n_fft normalized=%normalized onesided=%onesided pad_mode=%pad_mode return_complex=True win_length=%win_length
Tensor.reshape          op_2        1 1 a b shape=(1,1,%freq,%time)
pnnx.Expression         op_3        1 1 b out expr=pow(abs(@0),2.0)
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
        op->params["1"] = 2; // power
        op->params["2"] = captured_params.at("hop_length");
        op->params["3"] = captured_params.at("win_length");
        op->params["4"] = window_type;
        op->params["5"] = captured_params.at("center").type == 1 && captured_params.at("center").b ? 1 : 0;
        op->params["6"] = pad_type;
        op->params["7"] = captured_params.at("normalized").type == 1 && captured_params.at("normalized").b ? 1 : 0;
        op->params["8"] = onesided;
    }
};

void fuse_convert_stft(Graph& graph)
{
    fuse_stft_pass a;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
}

} // namespace ncnn

} // namespace pnnx