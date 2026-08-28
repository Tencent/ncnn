// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

#include <math.h>

namespace pnnx {

namespace ncnn {

// ncnn Interp derives the source coordinate of an output pixel in two different ways:
//
//   * params 1/2 (height_scale/width_scale) -> coordinate scale is 1/scale_factor
//   * params 3/4 (output_height/output_width) -> coordinate scale is input_size/output_size
//
// which is exactly the difference between F.interpolate(recompute_scale_factor=False), the
// PyTorch default when a scale_factor is given, and recompute_scale_factor=True, where the
// output size is computed first and the scale is then re-derived from it. The two only agree
// when input_size * scale_factor lands exactly on an integer.
//
// So recompute_scale_factor=True has to be lowered to an explicit output size. Resolve that
// size here: prefer the shape pnnx already recorded for the output operand, which is exact,
// and otherwise fall back to PyTorch's floor(input_size * scale_factor).
static bool interp_resolve_output_size(const Operator* op, const std::map<std::string, Parameter>& captured_params, std::vector<int>& output_size)
{
    const Parameter& sf = captured_params.at("scale_factor");

    std::vector<float> scale_factor;
    if (sf.type == 3)
        scale_factor.push_back(sf.f);
    else if (sf.type == 6)
        scale_factor = sf.af;
    else
        return false;

    // only 1d and 2d interpolation is supported by ncnn Interp
    if (scale_factor.size() != 1 && scale_factor.size() != 2)
        return false;

    const size_t dims = scale_factor.size();

    output_size.clear();

    // shapes are in torch layout and include the batch dim, so the interpolated
    // axes are always the trailing ones
    const std::vector<int>& output_shape = op->outputs[0]->shape;
    if (output_shape.size() >= dims + 1)
    {
        bool resolved = true;
        for (size_t i = 0; i < dims; i++)
        {
            // a dynamic or symbolic dim is recorded as -1
            const int s = output_shape[output_shape.size() - dims + i];
            if (s < 1)
            {
                resolved = false;
                break;
            }

            output_size.push_back(s);
        }

        if (resolved)
            return true;

        output_size.clear();
    }

    const std::vector<int>& input_shape = op->inputs[0]->shape;
    if (input_shape.size() < dims + 1)
        return false;

    for (size_t i = 0; i < dims; i++)
    {
        const int s = input_shape[input_shape.size() - dims + i];
        if (s < 1)
            return false;

        const int outs = (int)floor((double)s * (double)scale_factor[i]);
        if (outs < 1)
            return false;

        output_size.push_back(outs);
    }

    return true;
}

static void interp_write_mode(Operator* op, const std::string& mode)
{
    if (mode == "nearest")
        op->params["0"] = 1;
    if (mode == "nearest-exact")
    {
        fprintf(stderr, "unsupported interpolate mode nearest-exact\n");
        op->params["0"] = 1;
    }
    if (mode == "bilinear" || mode == "linear")
        op->params["0"] = 2;
    if (mode == "bicubic")
        op->params["0"] = 3;
}

static void interp_write_output_size(Operator* op, const std::vector<int>& output_size)
{
    if (output_size.size() == 1)
    {
        op->params["3"] = 1.f;
        op->params["4"] = output_size[0];
    }
    else if (output_size.size() == 2)
    {
        op->params["3"] = output_size[0];
        op->params["4"] = output_size[1];
    }
    else
    {
        fprintf(stderr, "unsupported interpolate output_size\n");
    }
}

// runs before the recompute_scale_factor=* passes below (priority 19 vs 20), and declines
// when the output size cannot be resolved so that those passes still handle the op
class F_interpolate_recompute_scale_factor : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.interpolate           op_0        1 1 input out mode=%mode recompute_scale_factor=True scale_factor=%scale_factor
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Interp";
    }

    const char* name_str() const
    {
        return "interpolate";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        std::vector<int> output_size;
        return interp_resolve_output_size(matched_operators.at("op_0"), captured_params, output_size);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& mode = captured_params.at("mode").s;

        std::vector<int> output_size;
        interp_resolve_output_size(op, captured_params, output_size);

        interp_write_mode(op, mode);

        interp_write_output_size(op, output_size);

        op->params["6"] = 0; // align_corners
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_interpolate_recompute_scale_factor, 19)

class F_interpolate_recompute_scale_factor_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.interpolate           op_0        1 1 input out align_corners=%align_corners mode=%mode recompute_scale_factor=True scale_factor=%scale_factor
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Interp";
    }

    const char* name_str() const
    {
        return "interpolate";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        std::vector<int> output_size;
        return interp_resolve_output_size(matched_operators.at("op_0"), captured_params, output_size);
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& mode = captured_params.at("mode").s;

        std::vector<int> output_size;
        interp_resolve_output_size(op, captured_params, output_size);

        interp_write_mode(op, mode);

        interp_write_output_size(op, output_size);

        op->params["6"] = captured_params.at("align_corners").b ? 1 : 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_interpolate_recompute_scale_factor_1, 19)

class F_interpolate : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.interpolate           op_0        1 1 input out mode=%mode recompute_scale_factor=* scale_factor=%scale_factor
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Interp";
    }

    const char* name_str() const
    {
        return "interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& mode = captured_params.at("mode").s;
        std::vector<float> scale_factor;
        if (captured_params.at("scale_factor").type == 3)
        {
            scale_factor.push_back(captured_params.at("scale_factor").f);
        }
        else
        {
            scale_factor = captured_params.at("scale_factor").af;
        }

        if (mode == "nearest")
            op->params["0"] = 1;
        if (mode == "nearest-exact")
        {
            fprintf(stderr, "unsupported interpolate mode nearest-exact\n");
            op->params["0"] = 1;
        }
        if (mode == "bilinear" || mode == "linear")
            op->params["0"] = 2;
        if (mode == "bicubic")
            op->params["0"] = 3;

        if (scale_factor.size() == 1)
        {
            op->params["1"] = 1.f;
            op->params["2"] = scale_factor[0];
        }
        else if (scale_factor.size() == 2)
        {
            op->params["1"] = scale_factor[0];
            op->params["2"] = scale_factor[1];
        }
        else
        {
            fprintf(stderr, "unsupported interpolate scale_factor\n");
        }

        op->params["6"] = 0; // align_corners
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_interpolate, 20)

class F_interpolate_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.interpolate           op_0        1 1 input out align_corners=%align_corners mode=%mode recompute_scale_factor=* scale_factor=%scale_factor
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Interp";
    }

    const char* name_str() const
    {
        return "interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& mode = captured_params.at("mode").s;
        std::vector<float> scale_factor;
        if (captured_params.at("scale_factor").type == 3)
        {
            scale_factor.push_back(captured_params.at("scale_factor").f);
        }
        else
        {
            scale_factor = captured_params.at("scale_factor").af;
        }

        if (mode == "nearest")
            op->params["0"] = 1;
        if (mode == "nearest-exact")
        {
            fprintf(stderr, "unsupported interpolate mode nearest-exact\n");
            op->params["0"] = 1;
        }
        if (mode == "bilinear" || mode == "linear")
            op->params["0"] = 2;
        if (mode == "bicubic")
            op->params["0"] = 3;

        if (scale_factor.size() == 1)
        {
            op->params["1"] = 1.f;
            op->params["2"] = scale_factor[0];
        }
        else if (scale_factor.size() == 2)
        {
            op->params["1"] = scale_factor[0];
            op->params["2"] = scale_factor[1];
        }
        else
        {
            fprintf(stderr, "unsupported interpolate scale_factor\n");
        }

        op->params["6"] = captured_params.at("align_corners").b ? 1 : 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_interpolate_1, 20)

class F_interpolate_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.interpolate           op_0        1 1 input out align_corners=%align_corners mode=%mode size=%size
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Interp";
    }

    const char* name_str() const
    {
        return "interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& mode = captured_params.at("mode").s;
        std::vector<int> output_size;
        if (captured_params.at("size").type == 2)
        {
            output_size.push_back(captured_params.at("size").i);
        }
        else
        {
            output_size = captured_params.at("size").ai;
        }

        if (mode == "nearest")
            op->params["0"] = 1;
        if (mode == "nearest-exact")
        {
            fprintf(stderr, "unsupported interpolate mode nearest-exact\n");
            op->params["0"] = 1;
        }
        if (mode == "bilinear" || mode == "linear")
            op->params["0"] = 2;
        if (mode == "bicubic")
            op->params["0"] = 3;

        if (output_size.size() == 1)
        {
            op->params["3"] = 1.f;
            op->params["4"] = output_size[0];
        }
        else if (output_size.size() == 2)
        {
            op->params["3"] = output_size[0];
            op->params["4"] = output_size[1];
        }
        else
        {
            fprintf(stderr, "unsupported interpolate output_size\n");
        }

        op->params["6"] = captured_params.at("align_corners").b ? 1 : 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_interpolate_2, 20)

class F_interpolate_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.interpolate           op_0        1 1 input out mode=%mode size=%size
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Interp";
    }

    const char* name_str() const
    {
        return "interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& mode = captured_params.at("mode").s;
        std::vector<int> output_size;
        if (captured_params.at("size").type == 2)
        {
            output_size.push_back(captured_params.at("size").i);
        }
        else
        {
            output_size = captured_params.at("size").ai;
        }

        if (mode == "nearest")
            op->params["0"] = 1;
        if (mode == "nearest-exact")
        {
            fprintf(stderr, "unsupported interpolate mode nearest-exact\n");
            op->params["0"] = 1;
        }
        if (mode == "bilinear" || mode == "linear")
            op->params["0"] = 2;
        if (mode == "bicubic")
            op->params["0"] = 3;

        if (output_size.size() == 1)
        {
            op->params["3"] = 1.f;
            op->params["4"] = output_size[0];
        }
        else if (output_size.size() == 2)
        {
            op->params["3"] = output_size[0];
            op->params["4"] = output_size[1];
        }
        else
        {
            fprintf(stderr, "unsupported interpolate output_size\n");
        }

        op->params["6"] = 0; // align_corners
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_interpolate_3, 20)

} // namespace ncnn

} // namespace pnnx
