// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "fuse_static_convtranspose.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

namespace pnnx {

class fuse_static_Fconvtranspose1d_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%in_channels,%out_channels_per_group,%kw)f32
F.conv_transpose1d      op_0        2 1 input weight out bias=None stride=%stride padding=%padding dilation=%dilation output_padding=%output_padding groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.ConvTranspose1d      conv_transpose1d 1 1 input out in_channels=%in_channels kernel_size=(%kw) stride=%stride padding=%padding output_padding=%output_padding dilation=%dilation groups=%groups bias=False @weight=%op_weight.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int out_channels_per_group = captured_params.at("out_channels_per_group").i;
        const int groups = captured_params.at("groups").i;

        ops.at("conv_transpose1d")->params["out_channels"] = out_channels_per_group * groups;
    }
};

class fuse_static_Fconvtranspose1d_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%in_channels,%out_channels_per_group,%kw)f32
pnnx.Attribute          op_bias     0 1 bias @data=(%out_channels)f32
F.conv_transpose1d      op_0        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation output_padding=%output_padding groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.ConvTranspose1d      conv_transpose1d 1 1 input out in_channels=%in_channels out_channels=%out_channels kernel_size=(%kw) stride=%stride padding=%padding output_padding=%output_padding dilation=%dilation groups=%groups bias=True @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int out_channels_per_group = captured_params.at("out_channels_per_group").i;
        const int out_channels = captured_params.at("out_channels").i;
        const int groups = captured_params.at("groups").i;

        if (out_channels != out_channels_per_group * groups)
            return false;

        return true;
    }
};

class fuse_static_Fconvtranspose2d_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%in_channels,%out_channels_per_group,%kh,%kw)f32
F.conv_transpose2d      op_0        2 1 input weight out bias=None stride=%stride padding=%padding dilation=%dilation output_padding=%output_padding groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.ConvTranspose2d      conv_transpose2d 1 1 input out in_channels=%in_channels kernel_size=(%kh,%kw) stride=%stride padding=%padding output_padding=%output_padding dilation=%dilation groups=%groups bias=False @weight=%op_weight.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int out_channels_per_group = captured_params.at("out_channels_per_group").i;
        const int groups = captured_params.at("groups").i;

        ops.at("conv_transpose2d")->params["out_channels"] = out_channels_per_group * groups;
    }
};

class fuse_static_Fconvtranspose2d_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%in_channels,%out_channels_per_group,%kh,%kw)f32
pnnx.Attribute          op_bias     0 1 bias @data=(%out_channels)f32
F.conv_transpose2d      op_0        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation output_padding=%output_padding groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.ConvTranspose2d      conv_transpose2d 1 1 input out in_channels=%in_channels out_channels=%out_channels kernel_size=(%kh,%kw) stride=%stride padding=%padding output_padding=%output_padding dilation=%dilation groups=%groups bias=True @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int out_channels_per_group = captured_params.at("out_channels_per_group").i;
        const int out_channels = captured_params.at("out_channels").i;
        const int groups = captured_params.at("groups").i;

        if (out_channels != out_channels_per_group * groups)
            return false;

        return true;
    }
};

class fuse_static_Fconvtranspose3d_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%in_channels,%out_channels_per_group,%kd,%kh,%kw)f32
F.conv_transpose3d      op_0        2 1 input weight out bias=None stride=%stride padding=%padding dilation=%dilation output_padding=%output_padding groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.ConvTranspose3d      conv_transpose3d 1 1 input out in_channels=%in_channels kernel_size=(%kd,%kh,%kw) stride=%stride padding=%padding output_padding=%output_padding dilation=%dilation groups=%groups bias=False @weight=%op_weight.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int out_channels_per_group = captured_params.at("out_channels_per_group").i;
        const int groups = captured_params.at("groups").i;

        ops.at("conv_transpose3d")->params["out_channels"] = out_channels_per_group * groups;
    }
};

class fuse_static_Fconvtranspose3d_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%in_channels,%out_channels_per_group,%kd,%kh,%kw)f32
pnnx.Attribute          op_bias     0 1 bias @data=(%out_channels)f32
F.conv_transpose3d      op_0        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation output_padding=%output_padding groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.ConvTranspose3d      conv_transpose3d 1 1 input out in_channels=%in_channels out_channels=%out_channels kernel_size=(%kd,%kh,%kw) stride=%stride padding=%padding output_padding=%output_padding dilation=%dilation groups=%groups bias=True @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int out_channels_per_group = captured_params.at("out_channels_per_group").i;
        const int out_channels = captured_params.at("out_channels").i;
        const int groups = captured_params.at("groups").i;

        if (out_channels != out_channels_per_group * groups)
            return false;

        return true;
    }
};

void fuse_static_convtranspose(Graph& graph)
{
    fuse_static_Fconvtranspose1d_pass a;
    fuse_static_Fconvtranspose1d_pass_2 b;
    fuse_static_Fconvtranspose2d_pass c;
    fuse_static_Fconvtranspose2d_pass_2 d;
    fuse_static_Fconvtranspose3d_pass e;
    fuse_static_Fconvtranspose3d_pass_2 f;
    int opindex = 0;

    pnnx_graph_rewrite(graph, &a, opindex);
    pnnx_graph_rewrite(graph, &b, opindex);
    pnnx_graph_rewrite(graph, &c, opindex);
    pnnx_graph_rewrite(graph, &d, opindex);
    pnnx_graph_rewrite(graph, &e, opindex);
    pnnx_graph_rewrite(graph, &f, opindex);
}

} // namespace pnnx
