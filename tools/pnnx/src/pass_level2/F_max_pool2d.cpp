// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

#include <float.h>

namespace pnnx {

class F_max_pool2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 kernel_size value=%kernel_size
prim::Constant          op_1        0 1 stride value=%stride
prim::Constant          op_2        0 1 padding value=%padding
prim::Constant          op_3        0 1 dilation value=%dilation
prim::Constant          op_4        0 1 ceil_mode value=%ceil_mode
aten::max_pool2d        op_5        6 1 input kernel_size stride padding dilation ceil_mode out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.max_pool2d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        GraphRewriterPass::write(op, captured_params);

        op->params["return_indices"] = false;
    }
};

class F_max_pool2d_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 8
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 kernel_size value=%kernel_size
prim::Constant          op_1        0 1 stride value=%stride
prim::Constant          op_2        0 1 padding value=%padding
prim::Constant          op_3        0 1 dilation value=%dilation
prim::Constant          op_4        0 1 ceil_mode value=%ceil_mode
aten::max_pool2d_with_indices op_5  6 2 input kernel_size stride padding dilation ceil_mode out indices
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.max_pool2d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        GraphRewriterPass::write(op, captured_params);

        op->params["return_indices"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_max_pool2d, 120)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_max_pool2d_2, 120)

class F_max_pool2d_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 3
pnnx.Input              input_0     0 1 input
aten::max_pool_with_indices_onnx op_1 1 2 input out indices kernel_size=%kernel_size stride=%stride padding=%padding dilation=%dilation ceil_mode=%ceil_mode n_dims_axes=* n_dims_one=* n_dims_zero=* unbatched_rank=*
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.max_pool2d";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        std::vector<int> kernel_size = captured_params.at("kernel_size").ai;
        std::vector<int> dilation = captured_params.at("dilation").ai;
        std::vector<int> stride = captured_params.at("stride").ai;
        std::vector<int> padding = captured_params.at("padding").ai;
        int ceil_mode = captured_params.at("ceil_mode").i;

        if (padding.size() == 4)
        {
            padding = {padding[0], padding[1]};
        }

        op->params["kernel_size"] = kernel_size;
        op->params["dilation"] = dilation;
        op->params["stride"] = stride;
        op->params["padding"] = padding;
        op->params["ceil_mode"] = (ceil_mode != 0);
        op->params["return_indices"] = (op->outputs.size() != 1);
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_max_pool2d_3, 120)

// https://github.com/pytorch/pytorch/blob/c263bd43e8e8502d4726643bc6fd046f0130ac0e/torch/onnx/symbolic_opset9.py#L1496
static int get_pool_ceil_padding(int w, int ksize, int stride, int pad)
{
    if (stride == 1)
        return 0;

    int ceiled_output_w = int(ceil((w + pad * 2 - ksize) / float(stride))) + 1;

    if ((ceiled_output_w - 1) * stride >= w + pad)
        ceiled_output_w = ceiled_output_w - 1;

    int ceil_pad = ksize - (w + pad * 2 - ((ceiled_output_w - 1) * stride + 1));

    return ceil_pad;
}

class F_max_pool2d_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
MaxPool                 op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.max_pool2d";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        if (captured_params.find("op_0.kernel_shape") == captured_params.end())
            return false;

        if (captured_params.at("op_0.kernel_shape").type != 5 || captured_params.at("op_0.kernel_shape").ai.size() != 2)
            return false;

        if (captured_params.find("op_0.dilations") != captured_params.end())
        {
            if (captured_params.at("op_0.dilations").type != 5 || captured_params.at("op_0.dilations").ai.size() != 2)
                return false;
        }

        if (captured_params.find("op_0.strides") != captured_params.end())
        {
            if (captured_params.at("op_0.strides").type != 5 || captured_params.at("op_0.strides").ai.size() != 2)
                return false;
        }

        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            if (captured_params.at("op_0.pads").type != 5 || captured_params.at("op_0.pads").ai.size() != 4)
                return false;

            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            const int ceil_mode = captured_params.find("op_0.ceil_mode") != captured_params.end() ? captured_params.at("op_0.ceil_mode").i : 0;
            if (pads[0] != pads[2] || pads[1] != pads[3])
            {
                // ceil_mode for opset9
                const Operator* avgpool = matched_operators.at("op_0");
                const std::vector<int>& in_shape = avgpool->inputs[0]->shape;
                if (in_shape.size() < 2)
                    return false;

                const int inh = in_shape[in_shape.size() - 2];
                const int inw = in_shape[in_shape.size() - 1];
                const int kh = captured_params.at("op_0.kernel_shape").ai[0];
                const int kw = captured_params.at("op_0.kernel_shape").ai[1];
                const int dh = captured_params.find("op_0.dilations") != captured_params.end() ? captured_params.at("op_0.dilations").ai[0] : 1;
                const int dw = captured_params.find("op_0.dilations") != captured_params.end() ? captured_params.at("op_0.dilations").ai[1] : 1;
                const int sh = captured_params.find("op_0.strides") != captured_params.end() ? captured_params.at("op_0.strides").ai[0] : 1;
                const int sw = captured_params.find("op_0.strides") != captured_params.end() ? captured_params.at("op_0.strides").ai[1] : 1;

                const int keh = dh * (kh - 1) + 1;
                const int kew = dw * (kw - 1) + 1;

                int ceil_padh = get_pool_ceil_padding(inh, keh, sh, pads[0]);
                int ceil_padw = get_pool_ceil_padding(inw, kew, sw, pads[1]);

                if (ceil_mode == 0 && pads[0] + ceil_padh == pads[2] && pads[1] + ceil_padw == pads[3])
                {
                    // useless tail padding  :D
                }
                else
                {
                    return false;
                }
            }
        }

        if (captured_params.find("op_0.auto_pad") != captured_params.end())
        {
            if (captured_params.at("op_0.auto_pad").type != 4)
                return false;

            const std::string& auto_pad = captured_params.at("op_0.auto_pad").s;
            if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER")
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["kernel_size"] = captured_params.at("op_0.kernel_shape");

        if (captured_params.find("op_0.dilations") != captured_params.end())
        {
            op->params["dilation"] = captured_params.at("op_0.dilations");
        }

        if (captured_params.find("op_0.strides") != captured_params.end())
        {
            op->params["stride"] = captured_params.at("op_0.strides");
        }
        else
        {
            op->params["stride"] = {1, 1};
        }

        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            op->params["padding"] = {pads[0], pads[1]};
        }
        else
        {
            op->params["padding"] = {0, 0};
        }

        if (captured_params.find("op_0.ceil_mode") != captured_params.end())
        {
            int ceil_mode = captured_params.at("op_0.ceil_mode").i;
            op->params["ceil_mode"] = (ceil_mode != 0);
        }
        else
        {
            op->params["ceil_mode"] = false;
        }

        // ceil_mode for opset9
        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            if (pads[0] != pads[2] || pads[1] != pads[3])
            {
                op->params["ceil_mode"] = true;
            }
        }

        op->params["return_indices"] = false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_max_pool2d_onnx, 120)

class F_max_pool2d_onnx_1 : public F_max_pool2d_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 3
pnnx.Input              input       0 1 input
MaxPool                 op_0        1 2 input out indices %*=%*
pnnx.Output             output      2 0 out indices
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        F_max_pool2d_onnx::write(op, captured_params);

        op->params["return_indices"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_max_pool2d_onnx_1, 120)

class F_max_pool2d_onnx_pad : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
MaxPool                 op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
F.pad                   pad         1 1 input pad
F.max_pool2d            maxpool     1 1 pad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.kernel_shape") == captured_params.end())
            return false;

        if (captured_params.at("op_0.kernel_shape").type != 5 || captured_params.at("op_0.kernel_shape").ai.size() != 2)
            return false;

        if (captured_params.find("op_0.dilations") != captured_params.end())
        {
            if (captured_params.at("op_0.dilations").type != 5 || captured_params.at("op_0.dilations").ai.size() != 2)
                return false;
        }

        if (captured_params.find("op_0.strides") != captured_params.end())
        {
            if (captured_params.at("op_0.strides").type != 5 || captured_params.at("op_0.strides").ai.size() != 2)
                return false;
        }

        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            if (captured_params.at("op_0.pads").type != 5 || captured_params.at("op_0.pads").ai.size() != 4)
                return false;

            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            if (pads[0] == pads[2] && pads[1] == pads[3])
                return false;
        }

        if (captured_params.find("op_0.auto_pad") != captured_params.end())
        {
            if (captured_params.at("op_0.auto_pad").type != 4)
                return false;

            const std::string& auto_pad = captured_params.at("op_0.auto_pad").s;
            if (auto_pad == "VALID")
                return false;
        }

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const
    {
        Operator* op_pad = ops.at("pad");
        Operator* op_maxpool = ops.at("maxpool");

        op_pad->params["mode"] = "constant";
        op_pad->params["value"] = -FLT_MAX;

        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            op_pad->params["pad"] = std::vector<int>{pads[1], pads[3], pads[0], pads[2]};
        }

        op_maxpool->params["kernel_size"] = captured_params.at("op_0.kernel_shape");

        if (captured_params.find("op_0.dilations") != captured_params.end())
        {
            op_maxpool->params["dilation"] = captured_params.at("op_0.dilations");
        }

        if (captured_params.find("op_0.strides") != captured_params.end())
        {
            op_maxpool->params["stride"] = captured_params.at("op_0.strides");
        }
        else
        {
            op_maxpool->params["stride"] = {1, 1};
        }

        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            op_maxpool->params["padding"] = {pads[0], pads[1]};
        }
        else
        {
            op_maxpool->params["padding"] = {0, 0};
        }

        if (captured_params.find("op_0.ceil_mode") != captured_params.end())
        {
            int ceil_mode = captured_params.at("op_0.ceil_mode").i;
            op_maxpool->params["ceil_mode"] = (ceil_mode != 0);
        }
        else
        {
            op_maxpool->params["ceil_mode"] = false;
        }

        // ceil_mode for opset9
        if (captured_params.find("op_0.pads") != captured_params.end())
        {
            const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
            if (pads[0] != pads[2] || pads[1] != pads[3])
            {
                op_maxpool->params["ceil_mode"] = true;
            }
        }

        // resolve auto_pad
        if (captured_params.find("op_0.auto_pad") != captured_params.end())
        {
            const std::string& auto_pad = captured_params.at("op_0.auto_pad").s;

            const int kernel_h = op_maxpool->params.at("kernel_size").ai[0];
            const int kernel_w = op_maxpool->params.at("kernel_size").ai[1];
            const int stride_h = op_maxpool->params.at("stride").ai[0];
            const int stride_w = op_maxpool->params.at("stride").ai[1];

            const int wpad = kernel_w - 1;
            const int hpad = kernel_h - 1;

            if (auto_pad == "SAME_UPPER")
            {
                op_pad->params["pad"] = std::vector<int>{wpad / 2, wpad - wpad / 2, hpad / 2, hpad - hpad / 2};
            }
            if (auto_pad == "SAME_LOWER")
            {
                op_pad->params["pad"] = std::vector<int>{wpad - wpad / 2, wpad / 2, hpad - hpad / 2, hpad / 2};
            }

            if (stride_w != 1 || stride_h != 1)
            {
                fprintf(stderr, "auto_pad %s with stride %d %d may lead to incorrect output shape\n", auto_pad.c_str(), stride_w, stride_h);
            }
        }

        op_maxpool->params["return_indices"] = false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_max_pool2d_onnx_pad, 120)

class F_max_pool2d_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
tnn.Pooling             op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.max_pool2d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int pool_type = captured_params.at("op_0.arg0").i;
        if (pool_type != 0)
            return false;

        const int kernel_h = captured_params.at("op_0.arg1").i;
        const int kernel_w = captured_params.at("op_0.arg2").i;
        if (kernel_h == 0 && kernel_w == 0)
            return false;

        if (captured_params.find("op_0.arg11") != captured_params.end())
        {
            const int is_adaptive = captured_params.at("op_0.arg11").i;
            if (is_adaptive != 0)
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        // captured_params.at("op_0.arg0"); // pool_type
        op->params["kernel_size"] = {captured_params.at("op_0.arg1").i, captured_params.at("op_0.arg2").i};
        op->params["stride"] = {captured_params.at("op_0.arg3").i, captured_params.at("op_0.arg4").i};
        op->params["padding"] = {captured_params.at("op_0.arg5").i, captured_params.at("op_0.arg6").i};

        const int kernel_index_h = captured_params.at("op_0.arg7").i;
        const int kernel_index_w = captured_params.at("op_0.arg8").i;
        if (kernel_index_h != -1 || kernel_index_w != -1)
        {
            fprintf(stderr, "unsupported F.max_pool2d kernel_index %d %d\n", kernel_index_h, kernel_index_w);
        }

        const int pad_type = captured_params.at("op_0.arg9").i;
        if (pad_type > 0)
        {
            fprintf(stderr, "unsupported F.max_pool2d pad_type %d\n", pad_type);
        }

        op->params["ceil_mode"] = captured_params.at("op_0.arg10").i ? true : false;
        // captured_params.at("op_0.arg11"); // is_adaptive
        // captured_params.at("op_0.arg12"); // output_h
        // captured_params.at("op_0.arg13"); // output_w

        op->params["return_indices"] = false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_max_pool2d_tnn, 120)

} // namespace pnnx
