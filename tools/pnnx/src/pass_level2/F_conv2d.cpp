// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "pass_level2.h"

namespace pnnx {

class F_conv2d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
17 16
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 weight
pnnx.Input              input_2     0 1 bias
pnnx.Input              input_3     0 1 stride
pnnx.Input              input_4     0 1 padding
pnnx.Input              input_5     0 1 dilation
pnnx.Input              input_6     0 1 groups
prim::Constant          op_0        0 1 transposed value=False
prim::Constant          op_1        0 1 output_padding_h value=0
prim::Constant          op_2        0 1 output_padding_w value=0
prim::ListConstruct     op_3        2 1 output_padding_h output_padding_w output_padding
prim::Constant          op_4        0 1 benchmark value=*
prim::Constant          op_5        0 1 deterministic value=*
prim::Constant          op_6        0 1 cudnn_enabled value=*
prim::Constant          op_7        0 1 allow_tf32 value=*
aten::_convolution      op_8        13 1 input weight bias stride padding dilation transposed output_padding groups benchmark deterministic cudnn_enabled allow_tf32 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.conv2d";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d, 10)

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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_cudnn_relu, 10)

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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_cudnn_add_relu, 10)

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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_conv2d_1, 10)

} // namespace pnnx
