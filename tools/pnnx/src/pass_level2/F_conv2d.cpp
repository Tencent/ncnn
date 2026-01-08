// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_conv2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
15 14
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
prim::Constant          op_0        0 1 stride value=%stride
prim::Constant          op_1        0 1 padding value=%padding
prim::Constant          op_2        0 1 dilation value=%dilation
prim::Constant          op_3        0 1 transposed value=False
prim::Constant          op_4        0 1 output_padding value=(0,0)
prim::Constant          op_5        0 1 groups value=%groups
prim::Constant          op_6        0 1 benchmark value=*
prim::Constant          op_7        0 1 deterministic value=*
prim::Constant          op_8        0 1 cudnn_enabled value=*
prim::Constant          op_9        0 1 allow_tf32 value=*
aten::_convolution      op_10       13 1 input weight bias stride padding dilation transposed output_padding groups benchmark deterministic cudnn_enabled allow_tf32 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.conv2d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return captured_params.at("stride").type == 5 && captured_params.at("stride").ai.size() == 2;
    }
};

class F_conv2d_mode : public F_conv2d
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
prim::Constant          op_0        0 1 stride value=%stride
prim::Constant          op_1        0 1 padding value=%padding
prim::Constant          op_2        0 1 dilation value=%dilation
prim::Constant          op_3        0 1 groups value=%groups
aten::_convolution_mode op_4        7 1 input weight bias stride padding dilation groups out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d, 140)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_mode, 140)

class F_conv2d_bias : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
F.conv2d                op_0        2 1 input weight a bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Attribute          op_1        0 1 bias @data=(1,%out_channels,1,1)f32
aten::add               op_2        2 1 a bias out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Attribute          bias        0 1 bias @data=%op_1.data
F.conv2d                conv        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        Operator* op_conv = ops.at("conv");

        op_conv->inputnames = {"input", "weight", "bias"};

        const int out_channels = captured_params.at("out_channels").i;

        Operator* op_bias = ops.at("bias");
        // fix bias shape
        op_bias->attrs["data"].shape = std::vector<int>{out_channels};
        op_bias->outputs[0]->shape = std::vector<int>{out_channels};
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_bias, 141)

class F_conv2d_cudnn_relu : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
pnnx.Input              input_3     0 1 stride
pnnx.Input              input_4     0 1 padding
pnnx.Input              input_5     0 1 dilation
pnnx.Input              input_6     0 1 groups
aten::cudnn_convolution_relu op_0   7 1 input weight bias stride padding dilation groups out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
10 9
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
pnnx.Input              input_3     0 1 stride
pnnx.Input              input_4     0 1 padding
pnnx.Input              input_5     0 1 dilation
pnnx.Input              input_6     0 1 groups
F.conv2d                conv        7 1 input weight bias stride padding dilation groups conv
F.relu                  relu        1 1 conv out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_cudnn_relu, 140)

class F_conv2d_cudnn_add_relu : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
11 10
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 z
pnnx.Input              input_3     0 1 alpha
pnnx.Input              input_4     0 1 bias
pnnx.Input              input_5     0 1 stride
pnnx.Input              input_6     0 1 padding
pnnx.Input              input_7     0 1 dilation
pnnx.Input              input_8     0 1 groups
aten::cudnn_convolution_add_relu op_0 9 1 input weight z alpha bias stride padding dilation groups out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
13 12
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 z
pnnx.Input              input_3     0 1 alpha
pnnx.Input              input_4     0 1 bias
pnnx.Input              input_5     0 1 stride
pnnx.Input              input_6     0 1 padding
pnnx.Input              input_7     0 1 dilation
pnnx.Input              input_8     0 1 groups
F.conv2d                conv        7 1 input weight bias stride padding dilation groups conv
aten::add               add         3 1 conv z alpha add
F.relu                  relu        1 1 add out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_cudnn_add_relu, 140)

class F_conv2d_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
pnnx.Input              input_3     0 1 stride
pnnx.Input              input_4     0 1 padding
pnnx.Input              input_5     0 1 dilation
pnnx.Input              input_6     0 1 groups
aten::conv2d            op_0        7 1 input weight bias stride padding dilation groups out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.conv2d";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_1, 140)

class F_conv2d_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight #weight=(?,?,?,?)f32
pnnx.Input              input_2     0 1 bias
Conv                    op_0        3 1 input weight bias out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.conv2d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.kernel_shape") != captured_params.end())
        {
            if (captured_params.at("op_0.kernel_shape").type != 5 || captured_params.at("op_0.kernel_shape").ai.size() != 2)
                return false;
        }

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
            if (pads[0] != pads[2] || pads[1] != pads[3])
                return false;
        }

        if (captured_params.find("op_0.auto_pad") != captured_params.end())
        {
            if (captured_params.at("op_0.auto_pad").type != 4)
                return false;

            const std::string& auto_pad = captured_params.at("op_0.auto_pad").s;
            if (auto_pad == "SAME_LOWER")
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.dilations") != captured_params.end())
        {
            op->params["dilation"] = captured_params.at("op_0.dilations");
        }
        else
        {
            op->params["dilation"] = {1, 1};
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

        if (captured_params.find("op_0.auto_pad") != captured_params.end())
        {
            const std::string& auto_pad = captured_params.at("op_0.auto_pad").s;
            if (auto_pad == "VALID")
            {
                op->params["padding"] = "valid";
            }
            if (auto_pad == "SAME_UPPER")
            {
                op->params["padding"] = "same";
            }
        }

        if (captured_params.find("op_0.group") != captured_params.end())
        {
            op->params["groups"] = captured_params.at("op_0.group");
        }
        else
        {
            op->params["groups"] = 1;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_onnx, 140)

class F_conv2d_onnx_1 : public F_conv2d_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight #weight=(?,?,?,?)f32
Conv                    op_0        2 1 input weight out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        F_conv2d_onnx::write(op, captured_params);

        op->params["bias"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_onnx_1, 140)

class F_conv2d_onnx_pad : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight #weight=(?,?,?,?)f32
pnnx.Input              input_2     0 1 bias
Conv                    op_0        3 1 input weight bias out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
F.pad                   pad         1 1 input pad
F.conv2d                conv        3 1 pad weight bias out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.kernel_shape") != captured_params.end())
        {
            if (captured_params.at("op_0.kernel_shape").type != 5 || captured_params.at("op_0.kernel_shape").ai.size() != 2)
                return false;
        }

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

        if (captured_params.find("op_0.pads") == captured_params.end())
            return false;

        if (captured_params.at("op_0.pads").type != 5 || captured_params.at("op_0.pads").ai.size() != 4)
            return false;

        const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
        if (pads[0] == pads[2] && pads[1] == pads[3])
            return false;

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const
    {
        Operator* op_pad = ops.at("pad");
        Operator* op_conv = ops.at("conv");

        const std::vector<int>& pads = captured_params.at("op_0.pads").ai;
        op_pad->params["mode"] = "constant";
        op_pad->params["pad"] = std::vector<int>{pads[1], pads[3], pads[0], pads[2]};
        op_pad->params["value"] = Parameter();

        if (captured_params.find("op_0.dilations") != captured_params.end())
        {
            op_conv->params["dilation"] = captured_params.at("op_0.dilations");
        }
        else
        {
            op_conv->params["dilation"] = {1, 1};
        }

        if (captured_params.find("op_0.strides") != captured_params.end())
        {
            op_conv->params["stride"] = captured_params.at("op_0.strides");
        }
        else
        {
            op_conv->params["stride"] = {1, 1};
        }

        op_conv->params["padding"] = {0, 0};

        if (captured_params.find("op_0.group") != captured_params.end())
        {
            op_conv->params["groups"] = captured_params.at("op_0.group");
        }
        else
        {
            op_conv->params["groups"] = 1;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_onnx_pad, 140)

class F_conv2d_onnx_pad_1 : public F_conv2d_onnx_pad
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight #weight=(?,?,?,?)f32
Conv                    op_0        2 1 input weight out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
F.pad                   pad         1 1 input pad
F.conv2d                conv        2 1 pad weight out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params) const
    {
        F_conv2d_onnx_pad::write(ops, captured_params);

        Operator* op_conv = ops.at("conv");
        op_conv->params["bias"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_onnx_pad_1, 140)

class F_conv2d_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
tnn.Convolution         op_0        3 1 input weight bias out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.conv2d";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.arg13") == captured_params.end())
            return true;

        const int activation = captured_params.at("op_0.arg13").i;
        return activation == 0;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["groups"] = captured_params.at("op_0.arg0");
        // captured_params.at("op_0.arg1"); // inch
        // captured_params.at("op_0.arg2"); // outch
        // captured_params.at("op_0.arg3"); // kernel_h
        // captured_params.at("op_0.arg4"); // kernel_w
        op->params["stride"] = {captured_params.at("op_0.arg5").i, captured_params.at("op_0.arg6").i};
        op->params["padding"] = {captured_params.at("op_0.arg7").i, captured_params.at("op_0.arg8").i};
        // captured_params.at("op_0.arg9"); // bias
        // captured_params.at("op_0.arg10"); // pad_type
        op->params["dilation"] = {captured_params.at("op_0.arg11").i, captured_params.at("op_0.arg12").i};
        if (op->params["dilation"].ai == std::vector{-1, -1})
        {
            op->params["dilation"] = {1, 1};
        }
        // captured_params.at("op_0.arg13"); // activation
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_tnn, 140)

class F_conv2d_tnn_1 : public F_conv2d_tnn
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
tnn.Convolution         op_0        2 1 input weight out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        F_conv2d_tnn::write(op, captured_params);

        op->params["bias"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_tnn_1, 140)

} // namespace pnnx
