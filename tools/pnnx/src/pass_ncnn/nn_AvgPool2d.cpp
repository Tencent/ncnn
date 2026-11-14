// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class nn_AvgPool2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.AvgPool2d            op_0        1 1 input out kernel_size=%kernel_size stride=%stride padding=%padding ceil_mode=%ceil_mode count_include_pad=%count_include_pad divisor_override=%divisor_override
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Pooling";
    }

    const char* name_str() const
    {
        return "avgpool2d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("divisor_override").type != 0)
        {
            fprintf(stderr, "unsupported avgpool2d divisor_override\n");
            return;
        }

        const std::vector<int>& padding = captured_params.at("padding").ai;
        const bool ceil_mode = captured_params.at("ceil_mode").b;
        const bool count_include_pad = captured_params.at("count_include_pad").b;

        op->params["0"] = 1;
        op->params["1"] = captured_params.at("kernel_size").ai[1];
        op->params["11"] = captured_params.at("kernel_size").ai[0];
        op->params["2"] = captured_params.at("stride").ai[1];
        op->params["12"] = captured_params.at("stride").ai[0];
        op->params["3"] = padding[1];
        op->params["13"] = padding[0];

        // pad_mode
        // ceil_mode allows the last window to start outside the valid range,
        // which ncnn only reproduces with full padding. torch drops that window
        // only when it starts at or beyond the end of the input, so decide from
        // the known input shape and fall back to the stride heuristic when the
        // shape is dynamic.
        const int kernel_h = captured_params.at("kernel_size").ai[0];
        const int kernel_w = captured_params.at("kernel_size").ai[1];
        const int stride_h = captured_params.at("stride").ai[0];
        const int stride_w = captured_params.at("stride").ai[1];
        const std::vector<int>& in_shape = op->inputs[0]->shape;
        const bool shape_known = in_shape.size() >= 2 && in_shape[in_shape.size() - 2] > 0 && in_shape[in_shape.size() - 1] > 0;
        bool drop_h = false;
        bool drop_w = false;
        int torch_out_h = 0;
        int torch_out_w = 0;
        if (shape_known)
        {
            const int in_h = in_shape[in_shape.size() - 2];
            const int in_w = in_shape[in_shape.size() - 1];
            const int out_h = (in_h - kernel_h + stride_h - 1) / stride_h + 1;
            const int out_w = (in_w - kernel_w + stride_w - 1) / stride_w + 1;
            drop_h = (out_h - 1) * stride_h >= in_h;
            drop_w = (out_w - 1) * stride_w >= in_w;
            torch_out_h = drop_h ? out_h - 1 : out_h;
            torch_out_w = drop_w ? out_w - 1 : out_w;
        }
        const bool empty_window = stride_h > kernel_h || stride_w > kernel_w;
        const bool unpadded = padding[0] == 0 && padding[1] == 0;
        bool use_full = false;
        if (ceil_mode)
        {
            // padded inputs keep the upstream behaviour
            if (!unpadded)
                use_full = true;
            else if (shape_known)
                use_full = !(drop_h || drop_w);
            else
                use_full = !empty_window;
        }
        op->params["5"] = use_full ? 0 : 1;


        // avgpool_count_include_pad
        // with count_include_pad enabled ncnn always divides by the full kernel,
        // while torch divides by the number of real elements when ceil_mode has
        // zero-filled the tail of the last window, so disable it for that case
        const bool ceil_tail = ceil_mode && padding[0] == 0 && padding[1] == 0;
        op->params["6"] = (count_include_pad && !ceil_tail) ? 1 : 0;
    }
};

// ceil_mode can drop the last window on one axis while keeping it on the other,
// for example a 3x4 input with kernel 2 and stride 3 gives a 1x2 output.
// a single Pooling layer cannot express that, ncnn pads and pools both axes
// the same way, so pad both axes like full mode and crop the extra row/column.
static bool avg_pool2d_resolve_mixed(const std::vector<int>& in_shape,
                                     const std::vector<int>& kernel_size,
                                     const std::vector<int>& stride,
                                     const std::vector<int>& padding,
                                     bool& drop_h,
                                     bool& drop_w,
                                     int& out_h,
                                     int& out_w)
{
    if (padding[0] != 0 || padding[1] != 0)
        return false;

    if (in_shape.size() < 2)
        return false;

    const int in_h = in_shape[in_shape.size() - 2];
    const int in_w = in_shape[in_shape.size() - 1];
    if (in_h <= 0 || in_w <= 0)
        return false;

    // window count torch would produce before dropping the out of range ones
    const int ceil_h = (in_h - kernel_size[0] + stride[0] - 1) / stride[0] + 1;
    const int ceil_w = (in_w - kernel_size[1] + stride[1] - 1) / stride[1] + 1;

    drop_h = (ceil_h - 1) * stride[0] >= in_h;
    drop_w = (ceil_w - 1) * stride[1] >= in_w;

    out_h = drop_h ? ceil_h - 1 : ceil_h;
    out_w = drop_w ? ceil_w - 1 : ceil_w;

    return true;
}

class nn_AvgPool2d_mixed : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.AvgPool2d            op_0        1 1 input out kernel_size=%kernel_size stride=%stride padding=%padding ceil_mode=%ceil_mode count_include_pad=%count_include_pad divisor_override=%divisor_override
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
Pooling                 pool        1 1 input pooled
Crop                    crop        1 1 pooled out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        if (captured_params.at("divisor_override").type != 0)
            return false;

        if (!captured_params.at("ceil_mode").b)
            return false;

        bool drop_h = false;
        bool drop_w = false;
        int out_h = 0;
        int out_w = 0;
        if (!avg_pool2d_resolve_mixed(matched_operators.at("op_0")->inputs[0]->shape, captured_params.at("kernel_size").ai, captured_params.at("stride").ai, captured_params.at("padding").ai, drop_h, drop_w, out_h, out_w))
            return false;

        // only the mixed case needs the extra Crop, both dropped or both kept
        // is handled by the plain Pooling conversion
        return drop_h != drop_w;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& kernel_size = captured_params.at("kernel_size").ai;
        const std::vector<int>& stride = captured_params.at("stride").ai;
        const std::vector<int>& padding = captured_params.at("padding").ai;

        bool drop_h = false;
        bool drop_w = false;
        int out_h = 0;
        int out_w = 0;
        avg_pool2d_resolve_mixed(ops.at("pool")->inputs[0]->shape, kernel_size, stride, padding, drop_h, drop_w, out_h, out_w);

        Operator* pool = ops.at("pool");
        pool->params["0"] = 1;
        pool->params["1"] = kernel_size[1];
        pool->params["11"] = kernel_size[0];
        pool->params["2"] = stride[1];
        pool->params["12"] = stride[0];
        pool->params["3"] = padding[1];
        pool->params["13"] = padding[1];
        pool->params["14"] = padding[0];
        pool->params["15"] = padding[0];
        // full padding keeps the zero padded last window on the non dropped axis
        pool->params["5"] = 0;
        // divide by the real element count for the zero padded tail
        pool->params["6"] = 0;

        // crop the window torch drops, it is fully out of the input on this axis
        std::vector<int> starts(1, 0);
        std::vector<int> ends(1);
        std::vector<int> axes(1);
        if (drop_h)
        {
            ends[0] = out_h;
            axes[0] = -2;
        }
        else
        {
            ends[0] = out_w;
            axes[0] = -1;
        }

        Operator* crop = ops.at("crop");
        crop->params["9"] = starts;
        crop->params["10"] = ends;
        crop->params["11"] = axes;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_AvgPool2d, 20)

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_AvgPool2d_mixed, 19)

} // namespace ncnn

} // namespace pnnx
