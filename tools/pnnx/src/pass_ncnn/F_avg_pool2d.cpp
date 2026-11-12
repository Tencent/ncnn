// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class F_avg_pool2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
F.avg_pool2d            op_0        1 1 input out kernel_size=%kernel_size stride=%stride padding=%padding ceil_mode=%ceil_mode count_include_pad=%count_include_pad divisor_override=%divisor_override
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

        std::vector<int> stride;
        if (captured_params.at("stride").type == 0)
        {
            stride = captured_params.at("kernel_size").ai;
        }
        else
        {
            stride = captured_params.at("stride").ai;
        }

        const std::vector<int>& padding = captured_params.at("padding").ai;
        const bool ceil_mode = captured_params.at("ceil_mode").b;
        const bool count_include_pad = captured_params.at("count_include_pad").b;

        op->params["0"] = 1;
        op->params["1"] = captured_params.at("kernel_size").ai[1];
        op->params["11"] = captured_params.at("kernel_size").ai[0];
        op->params["2"] = stride[1];
        op->params["12"] = stride[0];
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
        const int stride_h = stride[0];
        const int stride_w = stride[1];
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

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_avg_pool2d, 20)

} // namespace ncnn

} // namespace pnnx
