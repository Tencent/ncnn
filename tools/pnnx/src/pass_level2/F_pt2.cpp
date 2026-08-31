// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// pt2 路径的形态归一分支。
//
// torch.export 会省略等于默认值的实参(cat 的 dim=0、flatten 的 end_dim、
// conv2d 的 dilation/groups),因此 pt2 图里 aten 算子的实参个数与
// torchscript(trace 不省略)不同。这些 PT2 形态分支在 pass_level2 把
// 缺省形态归一成 ts 等价形态,下游 pass(torch_flatten / pass_ncnn F.conv2d)
// 零改动复用。
//
// 缺省值目前内联在 replace 图里;N3 会改为查离线生成的 aten 默认值静态表。
// priority 55:早于 torch_cat/torch_flatten(60,需要先吃到归一后的形态),
// 早于 F_conv2d_1(140,其 7-input pattern 会误吃 pt2 的 conv2d 形态)。

#include "pass_level2.h"

namespace pnnx {

// aten::flatten(self, start_dim) —— end_dim 缺省(-1)
// 归一:补 end_dim=-1 常量 operand,汇合 torch_flatten 的 3-input 形态
class F_flatten_pt2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
prim::Constant          op_1        0 1 start_dim value=%start_dim
aten::flatten           op_0        2 1 input start_dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
prim::Constant          start_dim   0 1 start_dim value=%start_dim
prim::Constant          end_dim     0 1 end_dim value=-1
aten::flatten           op_0        3 1 input start_dim end_dim out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_flatten_pt2, 55)

// aten::conv2d(input, weight, bias, stride, padding) —— bias=None,缺 dilation/groups
class F_conv2d_pt2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
prim::Constant          op_1        0 1 bias value=None
prim::Constant          op_2        0 1 stride value=%stride
prim::Constant          op_3        0 1 padding value=%padding
aten::conv2d            op_0        5 1 input weight bias stride padding out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
F.conv2d                conv        2 1 input weight out bias=None stride=%stride padding=%padding dilation=(1,1) groups=1
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_pt2, 55)

// aten::conv2d(input, weight, bias, stride, padding, groups) —— bias=None,缺 dilation
class F_conv2d_pt2_groups : public F_conv2d_pt2
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
prim::Constant          op_1        0 1 bias value=None
prim::Constant          op_2        0 1 stride value=%stride
prim::Constant          op_3        0 1 padding value=%padding
prim::Constant          op_4        0 1 groups value=%groups
aten::conv2d            op_0        6 1 input weight bias stride padding groups out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
F.conv2d                conv        2 1 input weight out bias=None stride=%stride padding=%padding dilation=(1,1) groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_pt2_groups, 55)

// aten::conv2d(input, weight, bias, stride, padding) —— bias 为张量,缺 dilation/groups
class F_conv2d_pt2_bias : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
prim::Constant          op_1        0 1 stride value=%stride
prim::Constant          op_2        0 1 padding value=%padding
aten::conv2d            op_0        5 1 input weight bias stride padding out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
F.conv2d                conv        3 1 input weight bias out stride=%stride padding=%padding dilation=(1,1) groups=1
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_pt2_bias, 55)

// aten::conv2d(input, weight, bias, stride, padding, dilation, groups)
// 全参数形态(target 为 conv2d.padding 时 padding 是字符串,如 "same")
class F_conv2d_pt2_full : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
9 8
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
prim::Constant          op_1        0 1 stride value=%stride
prim::Constant          op_2        0 1 padding value=%padding
prim::Constant          op_3        0 1 dilation value=%dilation
prim::Constant          op_4        0 1 groups value=%groups
aten::conv2d            op_0        7 1 input weight bias stride padding dilation groups out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
F.conv2d                conv        3 1 input weight bias out stride=%stride padding=%padding dilation=%dilation groups=%groups
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_pt2_full, 55)

} // namespace pnnx
