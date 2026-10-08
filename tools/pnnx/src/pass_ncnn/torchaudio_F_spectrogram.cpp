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

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const
    {
        GraphRewriterPass::write(ops, captured_params);
        const Operand* input = ops.at("op_0")->inputs[0];
        Operand* padded = ops.at("op_0")->outputs[0];
        padded->type = input->type;
        padded->shape = input->shape;
        padded->params = input->params;
        if (!padded->shape.empty() && padded->shape.back() > 0)
            padded->shape.back() += captured_params.at("pad").i * 2;
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

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const Operand* input = matched_operators.at("op_1")->inputs[0];
        if (input->shape.size() > 2 || (input->shape.size() == 2 && input->params.at("__ncnn_batch_axis").i != 0))
            return false;
        return match(captured_params, captured_attrs);
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

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const Operand* input = matched_operators.at("op_1")->inputs[0];
        if (input->shape.size() > 2 || (input->shape.size() == 2 && input->params.at("__ncnn_batch_axis").i != 0))
            return false;
        return match(captured_params, captured_attrs);
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

// reshape all leading dimensions into the native ncnn batch axis
static void write_spectrogram_batch(const std::map<std::string, Operator*>& ops)
{
    Operator* pack = ops.at("pack");
    Operator* spectrogram = ops.at("spectrogram");
    Operator* unpack = ops.at("unpack");
    const Operand* input = pack->inputs[0];
    const Operand* output = unpack->outputs[0];
    const int input_rank = (int)input->shape.size();
    const int input_batch_axis = input->params.at("__ncnn_batch_axis").i;
    const int output_batch_axis = output->params.at("__ncnn_batch_axis").i;
    const int physical_rank = input_rank - (input_batch_axis != 233 ? 1 : 0);

    pack->params["6"] = input_batch_axis == input_rank - 1 ? "0n,-1" : "0w,-1";
    pack->params["12"] = input_batch_axis;
    pack->params["13"] = 0;

    std::vector<std::string> input_dimensions;
    if (physical_rank == 1)
        input_dimensions = {"1w"};
    if (physical_rank == 2)
        input_dimensions = {"1h", "1w"};
    if (physical_rank == 3)
        input_dimensions = {"1c", "1h", "1w"};
    if (physical_rank == 4)
        input_dimensions = {"1c", "1d", "1h", "1w"};
    if (input_batch_axis != 233)
        input_dimensions.insert(input_dimensions.begin() + input_batch_axis, "1n");

    const bool complex_output = spectrogram->params.at("1").i == 0;
    std::string shape_expr = complex_output ? "0w,0h,0c" : "0w,0h";
    for (int i = input_rank - 2; i >= 0; i--)
        shape_expr += "," + input_dimensions[i];
    unpack->params["6"] = shape_expr;
    unpack->params["12"] = 0;
    unpack->params["13"] = output_batch_axis;

    int batch = 1;
    for (int i = 0; i < input_rank - 1; i++)
    {
        if (input->shape[i] <= 0 || batch > INT_MAX / input->shape[i])
        {
            batch = -1;
            break;
        }
        batch *= input->shape[i];
    }

    Operand* packed = pack->outputs[0];
    packed->type = input->type;
    packed->shape = {batch, input->shape.back()};
    packed->params["__batch_index"] = 0;
    packed->params["__ncnn_batch_axis"] = 0;

    Operand* spec = spectrogram->outputs[0];
    spec->type = output->type;
    spec->shape = {batch};
    spec->shape.insert(spec->shape.end(), output->shape.begin() + input_rank - 1, output->shape.end());
    spec->params["__batch_index"] = 0;
    spec->params["__ncnn_batch_axis"] = 0;
}

class torchaudio_F_spectrogram_batch : public torchaudio_F_spectrogram
{
public:
    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
Reshape                 pack        1 1 input packed
Spectrogram             spectrogram 1 1 packed spec
Reshape                 unpack      2 1 spec input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const int input_rank = (int)matched_operators.at("op_1")->inputs[0]->shape.size();
        const int output_rank = (int)matched_operators.at("op_2")->outputs[0]->shape.size();
        if (input_rank < 2 || input_rank > 4 || output_rank != input_rank + 2 || output_rank > 5)
            return false;
        if (input_rank == 2 && matched_operators.at("op_1")->inputs[0]->params.at("__ncnn_batch_axis").i == 0)
            return false;
        return torchaudio_F_spectrogram::match(captured_params, captured_attrs);
    }

    using torchaudio_F_spectrogram::write;

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        torchaudio_F_spectrogram::write(ops.at("spectrogram"), captured_params, captured_attrs);
        write_spectrogram_batch(ops);
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_batch, 19)

class torchaudio_F_spectrogram_batch_1 : public torchaudio_F_spectrogram_1
{
public:
    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
Reshape                 pack        1 1 input packed
Spectrogram             spectrogram 1 1 packed spec
Reshape                 unpack      2 1 spec input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const int input_rank = (int)matched_operators.at("op_1")->inputs[0]->shape.size();
        const int output_rank = (int)matched_operators.at("op_1")->outputs[0]->shape.size();
        if (input_rank < 2 || input_rank > 4 || output_rank != input_rank + 1 || output_rank > 5)
            return false;
        if (input_rank == 2 && matched_operators.at("op_1")->inputs[0]->params.at("__ncnn_batch_axis").i == 0)
            return false;
        return torchaudio_F_spectrogram_1::match(captured_params, captured_attrs);
    }

    using torchaudio_F_spectrogram_1::write;

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        torchaudio_F_spectrogram_1::write(ops.at("spectrogram"), captured_params, captured_attrs);
        write_spectrogram_batch(ops);
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torchaudio_F_spectrogram_batch_1, 19)

} // namespace ncnn

} // namespace pnnx
