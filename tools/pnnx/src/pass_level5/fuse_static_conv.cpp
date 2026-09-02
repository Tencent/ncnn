// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fuse_static_conv.h"

#include "pass_level2.h"

#include <math.h>
#include <string.h>

namespace pnnx {

// *_pad 融合族的公共守卫:mode 限 reflect/replicate(ts level1 只识别这
// 两种),pad 成对相等且长度符合维数,conv 自身无 padding("valid"/全零)。
// 键名 = pattern 里显式 %name 捕获的裸名(带 op 前缀的是 %*=%* 通配捕获)。
static bool conv_pad_match_common(const std::map<std::string, Parameter>& captured_params, int expected_pad_count)
{
    const Parameter& mode = captured_params.at("mode");
    if (mode.type != 4 || (mode.s != "reflect" && mode.s != "replicate"))
        return false;

    const std::vector<int>& pad = captured_params.at("pad").ai;
    if ((int)pad.size() != expected_pad_count)
        return false;
    for (int i = 0; i < expected_pad_count; i += 2)
    {
        if (pad[i] != pad[i + 1])
            return false;
    }

    const Parameter& padding = captured_params.at("padding");
    if (padding.type == 2)
        return padding.i == 0;
    if (padding.type == 4) // "valid"
        return padding.s == "valid";
    if (padding.type == 5)
    {
        for (size_t i = 0; i < padding.ai.size(); i++)
        {
            if (padding.ai[i] != 0)
                return false;
        }
        return true;
    }
    return false;
}

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

// reflect/replicate padding 模块分解的融合:Conv 模块声明 padding_mode 后,
// torch.export 把 padding 物化成模块内的 aten::pad + 零 padding 卷积两个
// 算子;ts 侧 level1 模块转换(nn_Conv*.cpp)把 pad 融回卷积属性,pt2 路径
// 无 level1,在此对齐(fuse_static_* 的静态折叠本就对应 level1 模块转换的
// 权重折层,层次对等)。仅融合成对相等的对称 pad 与 reflect/replicate——
// 非对称 pad、constant pad(显式 F.pad 用法)与自带 padding 的卷积不属于
// 该分解形态,保持原图交由各自既有转换器处理。
class fuse_static_Fconv1d_pad_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_channels,%in_channels_per_group,%kw)f32
F.pad                   op_pad      1 1 input padded mode=%mode pad=%pad value=None
F.conv1d                op_0        2 1 padded weight out bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Conv1d               conv1d      1 1 input out out_channels=%out_channels kernel_size=(%kw) padding_mode=%mode padding=%padding stride=%stride dilation=%dilation groups=%groups bias=False @weight=%op_weight.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return conv_pad_match_common(captured_params, 2);
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int in_channels_per_group = captured_params.at("in_channels_per_group").i;
        const int groups = captured_params.at("groups").i;

        ops.at("conv1d")->params["in_channels"] = in_channels_per_group * groups;
        // torch pad 的成对值从最后一维起排,conv padding 从第一空间维起
        const std::vector<int>& pad = captured_params.at("pad").ai;
        ops.at("conv1d")->params["padding"] = std::vector<int>{pad[0]};
    }
};

class fuse_static_Fconv1d_pad_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_channels,%in_channels_per_group,%kw)f32
pnnx.Attribute          op_bias     0 1 bias @data=(%out_channels)f32
F.pad                   op_pad      1 1 input padded mode=%mode pad=%pad value=None
F.conv1d                op_0        3 1 padded weight bias out stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Conv1d               conv1d      1 1 input out out_channels=%out_channels kernel_size=(%kw) padding_mode=%mode padding=%padding stride=%stride dilation=%dilation groups=%groups bias=True @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return conv_pad_match_common(captured_params, 2);
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int in_channels_per_group = captured_params.at("in_channels_per_group").i;
        const int groups = captured_params.at("groups").i;

        ops.at("conv1d")->params["in_channels"] = in_channels_per_group * groups;
        const std::vector<int>& pad = captured_params.at("pad").ai;
        ops.at("conv1d")->params["padding"] = std::vector<int>{pad[0]};
    }
};

class fuse_static_Fconv2d_pad_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_channels,%in_channels_per_group,%kh,%kw)f32
F.pad                   op_pad      1 1 input padded mode=%mode pad=%pad value=None
F.conv2d                op_0        2 1 padded weight out bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Conv2d               conv2d      1 1 input out out_channels=%out_channels kernel_size=(%kh,%kw) padding_mode=%mode padding=%padding stride=%stride dilation=%dilation groups=%groups bias=False @weight=%op_weight.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return conv_pad_match_common(captured_params, 4);
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int in_channels_per_group = captured_params.at("in_channels_per_group").i;
        const int groups = captured_params.at("groups").i;

        ops.at("conv2d")->params["in_channels"] = in_channels_per_group * groups;
        const std::vector<int>& pad = captured_params.at("pad").ai;
        ops.at("conv2d")->params["padding"] = std::vector<int>{pad[2], pad[0]};
    }
};

class fuse_static_Fconv2d_pad_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_channels,%in_channels_per_group,%kh,%kw)f32
pnnx.Attribute          op_bias     0 1 bias @data=(%out_channels)f32
F.pad                   op_pad      1 1 input padded mode=%mode pad=%pad value=None
F.conv2d                op_0        3 1 padded weight bias out stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Conv2d               conv2d      1 1 input out out_channels=%out_channels kernel_size=(%kh,%kw) padding_mode=%mode padding=%padding stride=%stride dilation=%dilation groups=%groups bias=True @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return conv_pad_match_common(captured_params, 4);
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int in_channels_per_group = captured_params.at("in_channels_per_group").i;
        const int groups = captured_params.at("groups").i;

        ops.at("conv2d")->params["in_channels"] = in_channels_per_group * groups;
        const std::vector<int>& pad = captured_params.at("pad").ai;
        ops.at("conv2d")->params["padding"] = std::vector<int>{pad[2], pad[0]};
    }
};

class fuse_static_Fconv3d_pad_pass : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_channels,%in_channels_per_group,%kd,%kh,%kw)f32
F.pad                   op_pad      1 1 input padded mode=%mode pad=%pad value=None
F.conv3d                op_0        2 1 padded weight out bias=None stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Conv3d               conv3d      1 1 input out out_channels=%out_channels kernel_size=(%kd,%kh,%kw) padding_mode=%mode padding=%padding stride=%stride dilation=%dilation groups=%groups bias=False @weight=%op_weight.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return conv_pad_match_common(captured_params, 6);
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int in_channels_per_group = captured_params.at("in_channels_per_group").i;
        const int groups = captured_params.at("groups").i;

        ops.at("conv3d")->params["in_channels"] = in_channels_per_group * groups;
        const std::vector<int>& pad = captured_params.at("pad").ai;
        ops.at("conv3d")->params["padding"] = std::vector<int>{pad[4], pad[2], pad[0]};
    }
};

class fuse_static_Fconv3d_pad_pass_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
pnnx.Attribute          op_weight   0 1 weight @data=(%out_channels,%in_channels_per_group,%kd,%kh,%kw)f32
pnnx.Attribute          op_bias     0 1 bias @data=(%out_channels)f32
F.pad                   op_pad      1 1 input padded mode=%mode pad=%pad value=None
F.conv3d                op_0        3 1 padded weight bias out stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Conv3d               conv3d      1 1 input out out_channels=%out_channels kernel_size=(%kd,%kh,%kw) padding_mode=%mode padding=%padding stride=%stride dilation=%dilation groups=%groups bias=True @weight=%op_weight.data @bias=%op_bias.data
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        return conv_pad_match_common(captured_params, 6);
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int in_channels_per_group = captured_params.at("in_channels_per_group").i;
        const int groups = captured_params.at("groups").i;

        ops.at("conv3d")->params["in_channels"] = in_channels_per_group * groups;
        const std::vector<int>& pad = captured_params.at("pad").ai;
        ops.at("conv3d")->params["padding"] = std::vector<int>{pad[4], pad[2], pad[0]};
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
    fuse_static_Fconv1d_pad_pass cp1;
    fuse_static_Fconv1d_pad_pass_2 cp2;
    fuse_static_Fconv2d_pad_pass cp3;
    fuse_static_Fconv2d_pad_pass_2 cp4;
    fuse_static_Fconv3d_pad_pass cp5;
    fuse_static_Fconv3d_pad_pass_2 cp6;

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

    // pad 融合必须先于静态折叠:普通折叠把 F.conv*d 写死成
    // padding_mode=zeros 的 nn.Conv*d,pad 就无法再融进卷积属性
    pnnx_graph_rewrite(graph, &cp1, opindex);
    pnnx_graph_rewrite(graph, &cp2, opindex);
    pnnx_graph_rewrite(graph, &cp3, opindex);
    pnnx_graph_rewrite(graph, &cp4, opindex);
    pnnx_graph_rewrite(graph, &cp5, opindex);
    pnnx_graph_rewrite(graph, &cp6, opindex);

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
