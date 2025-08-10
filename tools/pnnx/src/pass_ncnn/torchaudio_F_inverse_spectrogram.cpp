// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

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

class torchaudio_F_inverse_spectrogram : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_0        0 1 window @data
torch.view_as_complex   op_1        1 1 input a
torchaudio.functional.inverse_spectrogram op_2 2 1 a window out center=%center hop_length=%hop_length length=None n_fft=%n_fft normalized=%normalized onesided=%onesided pad=0 win_length=%win_length
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "InverseSpectrogram";
    }

    const char* name_str() const
    {
        return "inverse_spectrogram";
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

        int normalized = 0;
        if (captured_params.at("normalized").type == 1)
        {
            normalized = captured_params.at("normalized").b ? 2 : 0;
        }
        if (captured_params.at("normalized").type == 4)
        {
            if (captured_params.at("normalized").s == "frame_length")
                normalized = 1;
            if (captured_params.at("normalized").s == "window")
                normalized = 2;
        }

        op->params["0"] = captured_params.at("n_fft");
        op->params["1"] = 1; // returns
        op->params["2"] = captured_params.at("hop_length");
        op->params["3"] = captured_params.at("win_length");
        op->params["4"] = window_type;
        op->params["5"] = captured_params.at("center").type == 1 && captured_params.at("center").b ? 1 : 0;
        op->params["7"] = normalized;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torchaudio_F_inverse_spectrogram, 20)

} // namespace ncnn

} // namespace pnnx
