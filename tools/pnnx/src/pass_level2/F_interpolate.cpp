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

class F_interpolate : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
15 14
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input 6 dim=2
prim::NumToTensor       op_1        1 1 6 7
Tensor.to               op_2        1 1 7 13 copy=False dtype=torch.float
prim::Constant          op_3        0 1 scale value=%scale
Tensor.to               op_4        1 1 scale 20 copy=False dtype=torch.float
aten::detach            op_5        1 1 20 23
aten::mul               op_6        2 1 13 23 24
Tensor.to               op_7        1 1 24 28 copy=False dtype=torch.float
aten::floor             op_8        1 1 28 31
aten::Int               op_9        1 1 31 33
prim::ListConstruct     op_10       1 1 33 size
prim::Constant          op_11       0 1 scale_factor value=None
aten::upsample_nearest1d op_12      3 1 input size scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["scale_factor"] = captured_params.at("scale");
        op->params["mode"] = "nearest";
        op->params["recompute_scale_factor"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate, 110)

class F_interpolate_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
16 15
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input 6 dim=2
prim::NumToTensor       op_1        1 1 6 7
Tensor.to               op_2        1 1 7 13 copy=False dtype=torch.float
prim::Constant          op_3        0 1 scale value=%scale
Tensor.to               op_4        1 1 scale 20 copy=False dtype=torch.float
aten::detach            op_5        1 1 20 23
aten::mul               op_6        2 1 13 23 24
Tensor.to               op_7        1 1 24 28 copy=False dtype=torch.float
aten::floor             op_8        1 1 28 31
aten::Int               op_9        1 1 31 33
prim::ListConstruct     op_10       1 1 33 size
prim::Constant          op_11       0 1 align_corners value=%align_corners
prim::Constant          op_12       0 1 scale_factor value=None
aten::upsample_linear1d op_13       4 1 input size align_corners scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["scale_factor"] = captured_params.at("scale");
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "linear";
        op->params["recompute_scale_factor"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_1, 110)

class F_interpolate_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
26 25
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input 6 dim=2
prim::NumToTensor       op_1        1 1 6 7
Tensor.to               op_2        1 1 7 13 copy=False dtype=torch.float
prim::Constant          op_3        0 1 scale_h value=%scale_h
Tensor.to               op_4        1 1 scale_h 20 copy=False dtype=torch.float
aten::detach            op_5        1 1 20 23
aten::mul               op_6        2 1 13 23 24
Tensor.to               op_7        1 1 24 28 copy=False dtype=torch.float
aten::floor             op_8        1 1 28 30
aten::Int               op_9        1 1 30 32
Tensor.size             op_10       1 1 input 35 dim=3
prim::NumToTensor       op_11       1 1 35 36
Tensor.to               op_12       1 1 36 41 copy=False dtype=torch.float
prim::Constant          op_13       0 1 scale_w value=%scale_w
Tensor.to               op_14       1 1 scale_w 48 copy=False dtype=torch.float
aten::detach            op_15       1 1 48 51
aten::mul               op_16       2 1 41 51 52
Tensor.to               op_17       1 1 52 56 copy=False dtype=torch.float
aten::floor             op_18       1 1 56 60
aten::Int               op_19       1 1 60 62
prim::ListConstruct     op_20       2 1 32 62 size
prim::Constant          op_21       0 1 scale_h_none value=None
prim::Constant          op_22       0 1 scale_w_none value=None
aten::upsample_nearest2d op_23      4 1 input size scale_h_none scale_w_none out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["scale_factor"] = Parameter{captured_params.at("scale_h").f, captured_params.at("scale_w").f};
        op->params["mode"] = "nearest";
        op->params["recompute_scale_factor"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_2, 110)

class F_interpolate_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
27 26
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input 6 dim=2
prim::NumToTensor       op_1        1 1 6 7
Tensor.to               op_2        1 1 7 13 copy=False dtype=torch.float
prim::Constant          op_3        0 1 scale_h value=%scale_h
Tensor.to               op_4        1 1 scale_h 20 copy=False dtype=torch.float
aten::detach            op_5        1 1 20 23
aten::mul               op_6        2 1 13 23 24
Tensor.to               op_7        1 1 24 28 copy=False dtype=torch.float
aten::floor             op_8        1 1 28 30
aten::Int               op_9        1 1 30 32
Tensor.size             op_10       1 1 input 35 dim=3
prim::NumToTensor       op_11       1 1 35 36
Tensor.to               op_12       1 1 36 41 copy=False dtype=torch.float
prim::Constant          op_13       0 1 scale_w value=%scale_w
Tensor.to               op_14       1 1 scale_w 48 copy=False dtype=torch.float
aten::detach            op_15       1 1 48 51
aten::mul               op_16       2 1 41 51 52
Tensor.to               op_17       1 1 52 56 copy=False dtype=torch.float
aten::floor             op_18       1 1 56 60
aten::Int               op_19       1 1 60 62
prim::ListConstruct     op_20       2 1 32 62 size
prim::Constant          op_21       0 1 align_corners value=%align_corners
prim::Constant          op_22       0 1 scale_h_none value=None
prim::Constant          op_23       0 1 scale_w_none value=None
aten::upsample_bilinear2d op_24     5 1 input size align_corners scale_h_none scale_w_none out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["scale_factor"] = Parameter{captured_params.at("scale_h").f, captured_params.at("scale_w").f};
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "bilinear";
        op->params["recompute_scale_factor"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_3, 110)

class F_interpolate_3_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
26 25
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input 6 dim=2
prim::NumToTensor       op_1        1 1 6 7
Tensor.to               op_2        1 1 7 13 copy=False dtype=torch.float
prim::Constant          op_3        0 1 scale_h value=%scale_h
Tensor.to               op_4        1 1 scale_h 20 copy=False dtype=torch.float
aten::detach            op_5        1 1 20 23
aten::mul               op_6        2 1 13 23 24
Tensor.to               op_7        1 1 24 28 copy=False dtype=torch.float
aten::floor             op_8        1 1 28 30
aten::Int               op_9        1 1 30 32
Tensor.size             op_10       1 1 input 35 dim=3
prim::NumToTensor       op_11       1 1 35 36
Tensor.to               op_12       1 1 36 41 copy=False dtype=torch.float
prim::Constant          op_13       0 1 scale_w value=%scale_w
Tensor.to               op_14       1 1 scale_w 48 copy=False dtype=torch.float
aten::detach            op_15       1 1 48 51
aten::mul               op_16       2 1 41 51 52
Tensor.to               op_17       1 1 52 56 copy=False dtype=torch.float
aten::floor             op_18       1 1 56 60
aten::Int               op_19       1 1 60 62
prim::ListConstruct     op_20       2 1 32 62 size
prim::Constant          op_21       0 1 align_corners value=%align_corners
prim::Constant          op_22       0 1 scale_factor value=None
aten::upsample_bilinear2d op_23     4 1 input size align_corners scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["scale_factor"] = Parameter{captured_params.at("scale_h").f, captured_params.at("scale_w").f};
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "bilinear";
        op->params["recompute_scale_factor"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_3_1, 110)

class F_interpolate_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
27 26
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input 6 dim=2
prim::NumToTensor       op_1        1 1 6 7
Tensor.to               op_2        1 1 7 13 copy=False dtype=torch.float
prim::Constant          op_3        0 1 scale_h value=%scale_h
Tensor.to               op_4        1 1 scale_h 20 copy=False dtype=torch.float
aten::detach            op_5        1 1 20 23
aten::mul               op_6        2 1 13 23 24
Tensor.to               op_7        1 1 24 28 copy=False dtype=torch.float
aten::floor             op_8        1 1 28 30
aten::Int               op_9        1 1 30 32
Tensor.size             op_10       1 1 input 35 dim=3
prim::NumToTensor       op_11       1 1 35 36
Tensor.to               op_12       1 1 36 41 copy=False dtype=torch.float
prim::Constant          op_13       0 1 scale_w value=%scale_w
Tensor.to               op_14       1 1 scale_w 48 copy=False dtype=torch.float
aten::detach            op_15       1 1 48 51
aten::mul               op_16       2 1 41 51 52
Tensor.to               op_17       1 1 52 56 copy=False dtype=torch.float
aten::floor             op_18       1 1 56 60
aten::Int               op_19       1 1 60 62
prim::ListConstruct     op_20       2 1 32 62 size
prim::Constant          op_21       0 1 align_corners value=%align_corners
prim::Constant          op_22       0 1 scale_h_none value=None
prim::Constant          op_23       0 1 scale_w_none value=None
aten::upsample_bicubic2d op_24      5 1 input size align_corners scale_h_none scale_w_none out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["scale_factor"] = Parameter{captured_params.at("scale_h").f, captured_params.at("scale_w").f};
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "bicubic";
        op->params["recompute_scale_factor"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_4, 110)

class F_interpolate_4_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
26 25
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input 6 dim=2
prim::NumToTensor       op_1        1 1 6 7
Tensor.to               op_2        1 1 7 13 copy=False dtype=torch.float
prim::Constant          op_3        0 1 scale_h value=%scale_h
Tensor.to               op_4        1 1 scale_h 20 copy=False dtype=torch.float
aten::detach            op_5        1 1 20 23
aten::mul               op_6        2 1 13 23 24
Tensor.to               op_7        1 1 24 28 copy=False dtype=torch.float
aten::floor             op_8        1 1 28 30
aten::Int               op_9        1 1 30 32
Tensor.size             op_10       1 1 input 35 dim=3
prim::NumToTensor       op_11       1 1 35 36
Tensor.to               op_12       1 1 36 41 copy=False dtype=torch.float
prim::Constant          op_13       0 1 scale_w value=%scale_w
Tensor.to               op_14       1 1 scale_w 48 copy=False dtype=torch.float
aten::detach            op_15       1 1 48 51
aten::mul               op_16       2 1 41 51 52
Tensor.to               op_17       1 1 52 56 copy=False dtype=torch.float
aten::floor             op_18       1 1 56 60
aten::Int               op_19       1 1 60 62
prim::ListConstruct     op_20       2 1 32 62 size
prim::Constant          op_21       0 1 align_corners value=%align_corners
prim::Constant          op_22       0 1 scale_factor value=None
aten::upsample_bicubic2d op_23      4 1 input size align_corners scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["scale_factor"] = Parameter{captured_params.at("scale_h").f, captured_params.at("scale_w").f};
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "bicubic";
        op->params["recompute_scale_factor"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_4_1, 110)

class F_interpolate_5 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
37 36
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input 6 dim=2
prim::NumToTensor       op_1        1 1 6 7
Tensor.to               op_2        1 1 7 13 copy=False dtype=torch.float
prim::Constant          op_3        0 1 scale_d value=%scale_d
Tensor.to               op_4        1 1 scale_d 20 copy=False dtype=torch.float
aten::detach            op_5        1 1 20 23
aten::mul               op_6        2 1 13 23 24
Tensor.to               op_7        1 1 24 28 copy=False dtype=torch.float
aten::floor             op_8        1 1 28 30
aten::Int               op_9        1 1 30 32
Tensor.size             op_10       1 1 input 35 dim=3
prim::NumToTensor       op_11       1 1 35 36
Tensor.to               op_12       1 1 36 41 copy=False dtype=torch.float
prim::Constant          op_13       0 1 scale_h value=%scale_h
Tensor.to               op_14       1 1 scale_h 48 copy=False dtype=torch.float
aten::detach            op_15       1 1 48 51
aten::mul               op_16       2 1 41 51 52
Tensor.to               op_17       1 1 52 56 copy=False dtype=torch.float
aten::floor             op_18       1 1 56 58
aten::Int               op_19       1 1 58 60
Tensor.size             op_20       1 1 input 63 dim=4
prim::NumToTensor       op_21       1 1 63 64
Tensor.to               op_22       1 1 64 69 copy=False dtype=torch.float
prim::Constant          op_23       0 1 scale_w value=%scale_w
Tensor.to               op_24       1 1 scale_w 76 copy=False dtype=torch.float
aten::detach            op_25       1 1 76 79
aten::mul               op_26       2 1 69 79 80
Tensor.to               op_27       1 1 80 84 copy=False dtype=torch.float
aten::floor             op_28       1 1 84 89
aten::Int               op_29       1 1 89 91
prim::ListConstruct     op_30       3 1 32 60 91 size
prim::Constant          op_31       0 1 scale_d_none value=None
prim::Constant          op_32       0 1 scale_h_none value=None
prim::Constant          op_33       0 1 scale_w_none value=None
aten::upsample_nearest3d op_34      5 1 input size scale_d_none scale_h_none scale_w_none out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["scale_factor"] = Parameter{captured_params.at("scale_d").f, captured_params.at("scale_h").f, captured_params.at("scale_w").f};
        op->params["mode"] = "nearest";
        op->params["recompute_scale_factor"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_5, 110)

class F_interpolate_5_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
35 34
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input 6 dim=2
prim::NumToTensor       op_1        1 1 6 7
Tensor.to               op_2        1 1 7 13 copy=False dtype=torch.float
prim::Constant          op_3        0 1 scale_d value=%scale_d
Tensor.to               op_4        1 1 scale_d 20 copy=False dtype=torch.float
aten::detach            op_5        1 1 20 23
aten::mul               op_6        2 1 13 23 24
Tensor.to               op_7        1 1 24 28 copy=False dtype=torch.float
aten::floor             op_8        1 1 28 30
aten::Int               op_9        1 1 30 32
Tensor.size             op_10       1 1 input 35 dim=3
prim::NumToTensor       op_11       1 1 35 36
Tensor.to               op_12       1 1 36 41 copy=False dtype=torch.float
prim::Constant          op_13       0 1 scale_h value=%scale_h
Tensor.to               op_14       1 1 scale_h 48 copy=False dtype=torch.float
aten::detach            op_15       1 1 48 51
aten::mul               op_16       2 1 41 51 52
Tensor.to               op_17       1 1 52 56 copy=False dtype=torch.float
aten::floor             op_18       1 1 56 58
aten::Int               op_19       1 1 58 60
Tensor.size             op_20       1 1 input 63 dim=4
prim::NumToTensor       op_21       1 1 63 64
Tensor.to               op_22       1 1 64 69 copy=False dtype=torch.float
prim::Constant          op_23       0 1 scale_w value=%scale_w
Tensor.to               op_24       1 1 scale_w 76 copy=False dtype=torch.float
aten::detach            op_25       1 1 76 79
aten::mul               op_26       2 1 69 79 80
Tensor.to               op_27       1 1 80 84 copy=False dtype=torch.float
aten::floor             op_28       1 1 84 89
aten::Int               op_29       1 1 89 91
prim::ListConstruct     op_30       3 1 32 60 91 size
prim::Constant          op_31       0 1 scale_factor value=None
aten::upsample_nearest3d op_32      3 1 input size scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["scale_factor"] = Parameter{captured_params.at("scale_d").f, captured_params.at("scale_h").f, captured_params.at("scale_w").f};
        op->params["mode"] = "nearest";
        op->params["recompute_scale_factor"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_5_1, 110)

class F_interpolate_6 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
38 37
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input 6 dim=2
prim::NumToTensor       op_1        1 1 6 7
Tensor.to               op_2        1 1 7 13 copy=False dtype=torch.float
prim::Constant          op_3        0 1 scale_d value=%scale_d
Tensor.to               op_4        1 1 scale_d 20 copy=False dtype=torch.float
aten::detach            op_5        1 1 20 23
aten::mul               op_6        2 1 13 23 24
Tensor.to               op_7        1 1 24 28 copy=False dtype=torch.float
aten::floor             op_8        1 1 28 30
aten::Int               op_9        1 1 30 32
Tensor.size             op_10       1 1 input 35 dim=3
prim::NumToTensor       op_11       1 1 35 36
Tensor.to               op_12       1 1 36 41 copy=False dtype=torch.float
prim::Constant          op_13       0 1 scale_h value=%scale_h
Tensor.to               op_14       1 1 scale_h 48 copy=False dtype=torch.float
aten::detach            op_15       1 1 48 51
aten::mul               op_16       2 1 41 51 52
Tensor.to               op_17       1 1 52 56 copy=False dtype=torch.float
aten::floor             op_18       1 1 56 58
aten::Int               op_19       1 1 58 60
Tensor.size             op_20       1 1 input 63 dim=4
prim::NumToTensor       op_21       1 1 63 64
Tensor.to               op_22       1 1 64 69 copy=False dtype=torch.float
prim::Constant          op_23       0 1 scale_w value=%scale_w
Tensor.to               op_24       1 1 scale_w 76 copy=False dtype=torch.float
aten::detach            op_25       1 1 76 79
aten::mul               op_26       2 1 69 79 80
Tensor.to               op_27       1 1 80 84 copy=False dtype=torch.float
aten::floor             op_28       1 1 84 89
aten::Int               op_29       1 1 89 91
prim::ListConstruct     op_30       3 1 32 60 91 size
prim::Constant          op_31       0 1 align_corners value=%align_corners
prim::Constant          op_32       0 1 scale_d_none value=None
prim::Constant          op_33       0 1 scale_h_none value=None
prim::Constant          op_34       0 1 scale_w_none value=None
aten::upsample_trilinear3d op_35    6 1 input size align_corners scale_d_none scale_h_none scale_w_none out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["scale_factor"] = Parameter{captured_params.at("scale_d").f, captured_params.at("scale_h").f, captured_params.at("scale_w").f};
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "trilinear";
        op->params["recompute_scale_factor"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_6, 110)

class F_interpolate_6_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
36 35
pnnx.Input              input       0 1 input
Tensor.size             op_0        1 1 input 6 dim=2
prim::NumToTensor       op_1        1 1 6 7
Tensor.to               op_2        1 1 7 13 copy=False dtype=torch.float
prim::Constant          op_3        0 1 scale_d value=%scale_d
Tensor.to               op_4        1 1 scale_d 20 copy=False dtype=torch.float
aten::detach            op_5        1 1 20 23
aten::mul               op_6        2 1 13 23 24
Tensor.to               op_7        1 1 24 28 copy=False dtype=torch.float
aten::floor             op_8        1 1 28 30
aten::Int               op_9        1 1 30 32
Tensor.size             op_10       1 1 input 35 dim=3
prim::NumToTensor       op_11       1 1 35 36
Tensor.to               op_12       1 1 36 41 copy=False dtype=torch.float
prim::Constant          op_13       0 1 scale_h value=%scale_h
Tensor.to               op_14       1 1 scale_h 48 copy=False dtype=torch.float
aten::detach            op_15       1 1 48 51
aten::mul               op_16       2 1 41 51 52
Tensor.to               op_17       1 1 52 56 copy=False dtype=torch.float
aten::floor             op_18       1 1 56 58
aten::Int               op_19       1 1 58 60
Tensor.size             op_20       1 1 input 63 dim=4
prim::NumToTensor       op_21       1 1 63 64
Tensor.to               op_22       1 1 64 69 copy=False dtype=torch.float
prim::Constant          op_23       0 1 scale_w value=%scale_w
Tensor.to               op_24       1 1 scale_w 76 copy=False dtype=torch.float
aten::detach            op_25       1 1 76 79
aten::mul               op_26       2 1 69 79 80
Tensor.to               op_27       1 1 80 84 copy=False dtype=torch.float
aten::floor             op_28       1 1 84 89
aten::Int               op_29       1 1 89 91
prim::ListConstruct     op_30       3 1 32 60 91 size
prim::Constant          op_31       0 1 align_corners value=%align_corners
prim::Constant          op_32       0 1 scale_factor value=None
aten::upsample_trilinear3d op_33    4 1 input size align_corners scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["scale_factor"] = Parameter{captured_params.at("scale_d").f, captured_params.at("scale_h").f, captured_params.at("scale_w").f};
        op->params["align_corners"] = captured_params.at("align_corners");
        op->params["mode"] = "trilinear";
        op->params["recompute_scale_factor"] = true;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_6_1, 110)

class F_interpolate_7 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Input              size        0 1 size
aten::upsample_output_size op_0     2 1 input size out coordinate_transformation_mode=%coordinate_transformation_mode mode=%mode
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int input_rank = op->inputs[0]->shape.size();

        const std::string& coordinate_transformation_mode = captured_params.at("coordinate_transformation_mode").s;
        const std::string& mode = captured_params.at("mode").s;

        if (coordinate_transformation_mode == "pytorch_half_pixel")
        {
            op->params["align_corners"] = false;
        }

        if (mode == "nearest")
        {
            op->params["mode"] = "nearest";
        }
        if (mode == "linear")
        {
            if (input_rank == 3)
                op->params["mode"] = "linear";
            else if (input_rank == 5)
                op->params["mode"] = "trilinear";
            else
                op->params["mode"] = "bilinear";
        }
        if (mode == "cubic")
        {
            if (input_rank == 4)
                op->params["mode"] = "bicubic";
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_7, 110)

class F_interpolate_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Resize                  op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        if (captured_params.find("op_0.coordinate_transformation_mode") == captured_params.end())
            return false;

        if (captured_params.at("op_0.coordinate_transformation_mode").type != 4)
            return false;

        if (captured_params.find("op_0.mode") == captured_params.end())
            return false;

        if (captured_params.at("op_0.mode").type != 4)
            return false;

        if (captured_params.find("op_0.nearest_mode") != captured_params.end())
        {
            if (captured_params.at("op_0.nearest_mode").type != 4 || captured_params.at("op_0.nearest_mode").s != "floor")
                return false;
        }

        if (captured_params.find("op_0.roi") != captured_params.end())
        {
            if (captured_params.at("op_0.roi").type != 6 || !captured_params.at("op_0.roi").ai.empty())
                return false;
        }

        if (captured_params.find("op_0.sizes") == captured_params.end() && captured_params.find("op_0.scales") == captured_params.end())
            return false;

        if (captured_params.find("op_0.sizes") != captured_params.end() && captured_params.at("op_0.sizes").type == 5 && !captured_params.at("op_0.sizes").ai.empty())
        {
            const std::vector<int>& sizes = captured_params.at("op_0.sizes").ai;

            if (sizes.size() < 3 || sizes.size() > 5)
                return false;

            const std::vector<int>& input_shape = matched_operators.at("op_0")->inputs[0]->shape;
            if (input_shape.size() < 3 || input_shape.size() > 5)
                return false;

            if (input_shape[0] != sizes[0] || input_shape[1] != sizes[1])
                return false;
        }
        else if (captured_params.find("op_0.scales") != captured_params.end() && captured_params.at("op_0.scales").type == 6 && !captured_params.at("op_0.scales").af.empty())
        {
            const std::vector<float>& scales = captured_params.at("op_0.scales").af;

            if (scales.size() < 3 || scales.size() > 5)
                return false;

            if (scales[0] != 1.f || scales[1] != 1.f)
                return false;
        }
        else
        {
            return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& coordinate_transformation_mode = captured_params.at("op_0.coordinate_transformation_mode").s;
        std::string mode = captured_params.at("op_0.mode").s;

        if (mode == "linear")
        {
            if (coordinate_transformation_mode == "half_pixel")
                op->params["align_corners"] = false;
            if (coordinate_transformation_mode == "align_corners")
                op->params["align_corners"] = true;
        }

        if (mode == "cubic")
        {
            if (coordinate_transformation_mode == "half_pixel")
                op->params["align_corners"] = false;
            if (coordinate_transformation_mode == "align_corners")
                op->params["align_corners"] = true;
        }

        if (captured_params.find("op_0.sizes") != captured_params.end() && captured_params.at("op_0.sizes").type == 5 && !captured_params.at("op_0.sizes").ai.empty())
        {
            const std::vector<int>& sizes = captured_params.at("op_0.sizes").ai;

            if (mode == "linear")
            {
                if (sizes.size() == 4)
                    mode = "bilinear";
                if (sizes.size() == 5)
                    mode = "trilinear";
            }

            if (mode == "cubic")
            {
                mode = "bicubic";
            }

            op->params["mode"] = mode;
            if (sizes.size() == 3)
                op->params["size"] = {sizes[2]};
            if (sizes.size() == 4)
                op->params["size"] = {sizes[2], sizes[3]};
            if (sizes.size() == 5)
                op->params["size"] = {sizes[2], sizes[3], sizes[4]};
        }
        else if (captured_params.find("op_0.scales") != captured_params.end() && captured_params.at("op_0.scales").type == 6 && !captured_params.at("op_0.scales").af.empty())
        {
            const std::vector<float>& scales = captured_params.at("op_0.scales").af;

            if (mode == "linear")
            {
                if (scales.size() == 4)
                    mode = "bilinear";
                if (scales.size() == 5)
                    mode = "trilinear";
            }

            if (mode == "cubic")
            {
                mode = "bicubic";
            }

            op->params["mode"] = mode;
            op->params["recompute_scale_factor"] = false;
            if (scales.size() == 3)
                op->params["scale_factor"] = {scales[2]};
            if (scales.size() == 4)
                op->params["scale_factor"] = {scales[2], scales[3]};
            if (scales.size() == 5)
                op->params["scale_factor"] = {scales[2], scales[3], scales[4]};
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_onnx, 110)

class F_interpolate_onnx_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Upsample                op_0        1 1 input out mode=%mode scales=%scales
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("scales").type != 6)
            return false;

        const std::vector<float>& scales = captured_params.at("scales").af;

        if (scales.size() < 3 || scales.size() > 5)
            return false;

        if (scales[0] != 1.f || scales[1] != 1.f)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        std::string mode = captured_params.at("mode").s;
        const std::vector<float>& scales = captured_params.at("scales").af;

        if (mode == "linear")
        {
            op->params["align_corners"] = false;
            if (scales.size() == 4)
                mode = "bilinear";
            if (scales.size() == 5)
                mode = "trilinear";
        }

        if (mode == "cubic")
        {
            op->params["align_corners"] = false;
            mode = "bicubic";
        }

        op->params["mode"] = mode;
        op->params["recompute_scale_factor"] = false;
        if (scales.size() == 3)
            op->params["scale_factor"] = {scales[2]};
        if (scales.size() == 4)
            op->params["scale_factor"] = {scales[2], scales[3]};
        if (scales.size() == 5)
            op->params["scale_factor"] = {scales[2], scales[3], scales[4]};
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_onnx_2, 110)

} // namespace pnnx
