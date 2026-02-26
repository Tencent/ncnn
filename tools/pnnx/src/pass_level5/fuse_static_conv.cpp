// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_static_conv.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

namespace pnnx {

class fuse_static_Fconv1d_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_channels,%in_channels_per_group,%kw)f32
F.conv1d                op_0        2 1 input weight out bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Conv1d               conv1d      1 1 input out out_channels=%out_channels kernel_size=(%kw) padding_mode=zeros stride=%stride padding=%padding dilation=%dilation groups=%groups bias=False @weight=%op_weight.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int in_channels_per_group = captured_params.at("in_channels_per_group").i;
        const int groups = captured_params.at("groups").i;

        ops.at("conv1d")->params["in_channels"] = in_channels_per_group * groups;
    }
};

class fuse_static_Fconv1d_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_channels,%in_channels_per_group,%kw)f32
pnnx.Attribute          op_bias     0 1 bias @data=(%out_channels)f32
F.conv1d                op_0        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Conv1d               conv1d      1 1 input out out_channels=%out_channels kernel_size=(%kw) padding_mode=zeros stride=%stride padding=%padding dilation=%dilation groups=%groups bias=True @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int in_channels_per_group = captured_params.at("in_channels_per_group").i;
        const int groups = captured_params.at("groups").i;

        ops.at("conv1d")->params["in_channels"] = in_channels_per_group * groups;
    }
};

class fuse_static_Fconv1d_pass_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_channels,%in_channels_per_group,%kw)f32
pnnx.Attribute          op_bias     0 1 bias @data=(1,%out_channels,1)f32
F.conv1d                op_0        2 1 input weight a bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Expression         op_1        2 1 a bias out expr=add(@0,@1)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Conv1d               conv1d      1 1 input out out_channels=%out_channels kernel_size=(%kw) padding_mode=zeros stride=%stride padding=%padding dilation=%dilation groups=%groups bias=True @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int in_channels_per_group = captured_params.at("in_channels_per_group").i;
        const int groups = captured_params.at("groups").i;
        const int out_channels = captured_params.at("out_channels").i;

        ops.at("conv1d")->params["in_channels"] = in_channels_per_group * groups;
        ops.at("conv1d")->attrs["bias"].shape = {out_channels};
    }
};

class fuse_static_Fconv2d_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_channels,%in_channels_per_group,%kh,%kw)f32
F.conv2d                op_0        2 1 input weight out bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Conv2d               conv2d      1 1 input out out_channels=%out_channels kernel_size=(%kh,%kw) padding_mode=zeros stride=%stride padding=%padding dilation=%dilation groups=%groups bias=False @weight=%op_weight.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int in_channels_per_group = captured_params.at("in_channels_per_group").i;
        const int groups = captured_params.at("groups").i;

        ops.at("conv2d")->params["in_channels"] = in_channels_per_group * groups;
    }
};

class fuse_static_Fconv2d_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_channels,%in_channels_per_group,%kh,%kw)f32
pnnx.Attribute          op_bias     0 1 bias @data=(%out_channels)f32
F.conv2d                op_0        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Conv2d               conv2d      1 1 input out out_channels=%out_channels kernel_size=(%kh,%kw) padding_mode=zeros stride=%stride padding=%padding dilation=%dilation groups=%groups bias=True @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int in_channels_per_group = captured_params.at("in_channels_per_group").i;
        const int groups = captured_params.at("groups").i;

        ops.at("conv2d")->params["in_channels"] = in_channels_per_group * groups;
    }
};

class fuse_static_Fconv2d_pass_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_channels,%in_channels_per_group,%kh,%kw)f32
pnnx.Attribute          op_bias     0 1 bias @data=(1,%out_channels,1,1)f32
F.conv2d                op_0        2 1 input weight a bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Expression         op_1        2 1 a bias out expr=add(@0,@1)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Conv2d               conv2d      1 1 input out out_channels=%out_channels kernel_size=(%kh,%kw) padding_mode=zeros stride=%stride padding=%padding dilation=%dilation groups=%groups bias=True @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int in_channels_per_group = captured_params.at("in_channels_per_group").i;
        const int groups = captured_params.at("groups").i;
        const int out_channels = captured_params.at("out_channels").i;

        ops.at("conv2d")->params["in_channels"] = in_channels_per_group * groups;
        ops.at("conv2d")->attrs["bias"].shape = {out_channels};
    }
};

class fuse_static_Fconv3d_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_channels,%in_channels_per_group,%kd,%kh,%kw)f32
F.conv3d                op_0        2 1 input weight out bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Conv3d               conv3d      1 1 input out out_channels=%out_channels kernel_size=(%kd,%kh,%kw) padding_mode=zeros stride=%stride padding=%padding dilation=%dilation groups=%groups bias=False @weight=%op_weight.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int in_channels_per_group = captured_params.at("in_channels_per_group").i;
        const int groups = captured_params.at("groups").i;

        ops.at("conv3d")->params["in_channels"] = in_channels_per_group * groups;
    }
};

class fuse_static_Fconv3d_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_channels,%in_channels_per_group,%kd,%kh,%kw)f32
pnnx.Attribute          op_bias     0 1 bias @data=(%out_channels)f32
F.conv3d                op_0        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Conv3d               conv3d      1 1 input out out_channels=%out_channels kernel_size=(%kd,%kh,%kw) padding_mode=zeros stride=%stride padding=%padding dilation=%dilation groups=%groups bias=True @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int in_channels_per_group = captured_params.at("in_channels_per_group").i;
        const int groups = captured_params.at("groups").i;

        ops.at("conv3d")->params["in_channels"] = in_channels_per_group * groups;
    }
};

class fuse_static_Fconv3d_pass_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_channels,%in_channels_per_group,%kd,%kh,%kw)f32
pnnx.Attribute          op_bias     0 1 bias @data=(1,%out_channels,1,1,1)f32
F.conv3d                op_0        2 1 input weight a bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Expression         op_1        2 1 a bias out expr=add(@0,@1)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Conv3d               conv3d      1 1 input out out_channels=%out_channels kernel_size=(%kd,%kh,%kw) padding_mode=zeros stride=%stride padding=%padding dilation=%dilation groups=%groups bias=True @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int in_channels_per_group = captured_params.at("in_channels_per_group").i;
        const int groups = captured_params.at("groups").i;
        const int out_channels = captured_params.at("out_channels").i;

        ops.at("conv3d")->params["in_channels"] = in_channels_per_group * groups;
        ops.at("conv3d")->attrs["bias"].shape = {out_channels};
    }
};

class fuse_static_Fconv1d_no_affine_pass_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Input              weight      0 1 weight
pnnx.Attribute          op_bias     0 1 bias @data
F.conv1d                op_0        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Input              weight      0 1 weight
F.conv1d                op_0        2 1 input weight out bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& /*matched_operators*/, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        auto bias_data = captured_attrs.at("op_bias.data");
        std::vector<float> bias_data_fp32 = bias_data.get_float32_data();
        for (auto b : bias_data_fp32)
        {
            if (b != 0.f)
                return false;
        }
        return true;
    }
};

class fuse_static_Fconv2d_no_affine_pass_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Input              weight      0 1 weight
pnnx.Attribute          op_bias     0 1 bias @data
F.conv2d                op_0        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Input              weight      0 1 weight
F.conv2d                op_0        2 1 input weight out bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& /*matched_operators*/, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        auto bias_data = captured_attrs.at("op_bias.data");
        std::vector<float> bias_data_fp32 = bias_data.get_float32_data();
        for (auto b : bias_data_fp32)
        {
            if (b != 0.f)
                return false;
        }
        return true;
    }
};

class fuse_static_Fconv3d_no_affine_pass_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Input              weight      0 1 weight
pnnx.Attribute          op_bias     0 1 bias @data
F.conv3d                op_0        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Input              weight      0 1 weight
F.conv3d                op_0        2 1 input weight out bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& /*matched_operators*/, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        auto bias_data = captured_attrs.at("op_bias.data");
        std::vector<float> bias_data_fp32 = bias_data.get_float32_data();
        for (auto b : bias_data_fp32)
        {
            if (b != 0.f)
                return false;
        }
        return true;
    }
};

void fuse_static_conv(Graph& graph)
{
    fuse_static_Fconv1d_pass_3 a3;
    fuse_static_Fconv2d_pass_3 a4;
    fuse_static_Fconv3d_pass_3 a5;

    fuse_static_Fconv1d_pass a;
    fuse_static_Fconv1d_pass_2 b;
    fuse_static_Fconv2d_pass c;
    fuse_static_Fconv2d_pass_2 d;
    fuse_static_Fconv3d_pass e;
    fuse_static_Fconv3d_pass_2 f;

    fuse_static_Fconv1d_no_affine_pass_onnx z1;
    fuse_static_Fconv2d_no_affine_pass_onnx z2;
    fuse_static_Fconv3d_no_affine_pass_onnx z3;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a3, opindex);
    pnnx_graph_rewrite(graph, &a4, opindex);
    pnnx_graph_rewrite(graph, &a5, opindex);

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &c, opindex);
    pnnx_graph_rewrite(graph, &d, opindex);
    pnnx_graph_rewrite(graph, &e, opindex);
    pnnx_graph_rewrite(graph, &f, opindex);

    pnnx_graph_rewrite(graph, &z1, opindex);
    pnnx_graph_rewrite(graph, &z2, opindex);
    pnnx_graph_rewrite(graph, &z3, opindex);
}

} // namespace pnnx
