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

class F_upsample : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
prim::Constant          op_0        0 1 scale_factor value=None
aten::upsample_nearest1d op_1       3 1 input size scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["mode"] = "nearest";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample, 11)

class F_upsample_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 scale_factor
prim::Constant          op_0        0 1 size value=None
aten::upsample_nearest1d op_1       3 1 input size scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["mode"] = "nearest";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_1, 11)

class F_upsample_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
prim::Constant          op_0        0 1 align_corners value=%align_corners
prim::Constant          op_1        0 1 scale_factor value=None
aten::upsample_linear1d op_2        4 1 input size align_corners scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "linear";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_2, 11)

class F_upsample_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 scale_factor
prim::Constant          op_0        0 1 size value=None
prim::Constant          op_1        0 1 align_corners value=%align_corners
aten::upsample_linear1d op_2        4 1 input size align_corners scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "linear";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_3, 11)

class F_upsample_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
prim::Constant          op_0        0 1 align_corners value=%align_corners
prim::Constant          op_1        0 1 scale_h value=None
prim::Constant          op_2        0 1 scale_w value=None
aten::upsample_bilinear2d op_3      5 1 input size align_corners scale_h scale_w out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "bilinear";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_4, 11)

class F_upsample_4_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
prim::Constant          op_0        0 1 align_corners value=%align_corners
prim::Constant          op_1        0 1 scale_factor value=None
aten::upsample_bilinear2d op_2      4 1 input size align_corners scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "bilinear";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_4_1, 11)

class F_upsample_5 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 scale_factor
prim::Constant          op_0        0 1 size value=None
prim::Constant          op_1        0 1 align_corners value=%align_corners
aten::upsample_bilinear2d op_2      4 1 input size align_corners scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "bilinear";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_5, 11)

class F_upsample_6 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
prim::Constant          op_0        0 1 align_corners value=%align_corners
prim::Constant          op_1        0 1 scale_h value=None
prim::Constant          op_2        0 1 scale_w value=None
aten::upsample_bicubic2d op_3       5 1 input size align_corners scale_h scale_w out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "bicubic";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_6, 11)

class F_upsample_6_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
prim::Constant          op_0        0 1 align_corners value=%align_corners
prim::Constant          op_1        0 1 scale_factor value=None
aten::upsample_bicubic2d op_2       4 1 input size align_corners scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "bicubic";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_6_1, 11)

class F_upsample_7 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 scale_factor
prim::Constant          op_0        0 1 size value=None
prim::Constant          op_1        0 1 align_corners value=%align_corners
aten::upsample_bicubic2d op_2       4 1 input size align_corners scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "bicubic";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_7, 11)

class F_upsample_8 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
prim::Constant          op_0        0 1 align_corners value=%align_corners
prim::Constant          op_1        0 1 scale_d value=None
prim::Constant          op_2        0 1 scale_h value=None
prim::Constant          op_3        0 1 scale_w value=None
aten::upsample_trilinear3d op_4     6 1 input size align_corners scale_d scale_h scale_w out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "trilinear";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_8, 11)

class F_upsample_8_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
prim::Constant          op_0        0 1 align_corners value=%align_corners
prim::Constant          op_1        0 1 scale_factor value=None
aten::upsample_trilinear3d op_2     4 1 input size align_corners scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "trilinear";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_8_1, 11)

class F_upsample_9 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 scale_factor
prim::Constant          op_0        0 1 size value=None
prim::Constant          op_1        0 1 align_corners value=%align_corners
aten::upsample_trilinear3d op_2     4 1 input size align_corners scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.upsample";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "trilinear";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_upsample_9, 11)

} // namespace pnnx
