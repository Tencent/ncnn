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

class torchaudio_F_spectrogram_pad : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 window
torchaudio.functional.spectrogram op_1 2 1 input window out n_fft=%n_fft hop_length=%hop_length win_length=%win_length onesided=%onesided power=%power normalized=%normalized center=%center pad=%pad pad_mode=%pad_mode
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 window
F.pad                   op_0        1 1 input a mode=constant pad=(%pad,%pad) value=0.000000e+00
torchaudio.functional.spectrogram op_1 2 1 a window out n_fft=%n_fft hop_length=%hop_length win_length=%win_length onesided=%onesided power=%power normalized=%normalized center=%center pad=0 pad_mode=%pad_mode
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("pad").type == 2 && captured_params.at("pad").i > 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_pad, 10)

class torchaudio_F_spectrogram : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_0        0 1 window @data
torchaudio.functional.spectrogram op_1 2 1 input window a n_fft=%n_fft hop_length=%hop_length win_length=%win_length onesided=%onesided power=%power normalized=%normalized center=%center pad=0 pad_mode=%pad_mode
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
        return "spectrogram";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        if (captured_params.at("power").type != 0)
            return false;

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
        op->params["1"] = 0; // power
        op->params["2"] = captured_params.at("hop_length");
        op->params["3"] = captured_params.at("win_length");
        op->params["4"] = window_type;
        op->params["5"] = captured_params.at("center").type == 1 && captured_params.at("center").b ? 1 : 0;
        op->params["6"] = pad_type;
        op->params["7"] = normalized;
        op->params["8"] = onesided;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram, 20)

class torchaudio_F_spectrogram_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_0        0 1 window @data
torchaudio.functional.spectrogram op_1 2 1 input window out n_fft=%n_fft hop_length=%hop_length win_length=%win_length onesided=%onesided power=%power normalized=%normalized center=%center pad=0 pad_mode=%pad_mode
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Spectrogram";
    }

    const char* name_str() const
    {
        return "spectrogram";
    }

    bool match(const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        if (captured_params.at("power").type == 0)
            return false;

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

        int power = 0;
        if (captured_params.at("power").type == 2)
        {
            power = captured_params.at("power").i;
            if (power != 1 && power != 2)
                fprintf(stderr, "unsupported spectrogram power %d\n", power);
        }
        if (captured_params.at("power").type == 3)
        {
            if (NearlyEqual(captured_params.at("power").f, 1.0, 0.0001))
                power = 1;
            else if (NearlyEqual(captured_params.at("power").f, 2.0, 0.0001))
                power = 2;
            else
                fprintf(stderr, "unsupported spectrogram power %f\n", captured_params.at("power").f);
        }

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
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_1, 20)

} // namespace ncnn

} // namespace pnnx
