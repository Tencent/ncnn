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
29 28
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 scale value=%scale
prim::Constant          op_1        0 1 5 value=2
aten::size              op_2        2 1 input 5 6
prim::NumToTensor       op_3        1 1 6 7
prim::Constant          op_4        0 1 9 value=6
prim::Constant          op_5        0 1 10 value=False
prim::Constant          op_6        0 1 52 value=False
prim::Constant          op_7        0 1 12 value=None
aten::to                op_8        5 1 7 9 10 52 12 13
prim::Constant          op_9        0 1 51 value=*
prim::Constant          op_10       0 1 53 value=6
prim::Constant          op_11       0 1 54 value=False
prim::Constant          op_12       0 1 55 value=False
prim::Constant          op_13       0 1 56 value=None
aten::to                op_14       6 1 scale 51 53 54 55 56 20
aten::detach            op_15       1 1 20 23
aten::mul               op_16       2 1 13 23 24
prim::Constant          op_17       0 1 57 value=6
prim::Constant          op_18       0 1 58 value=False
prim::Constant          op_19       0 1 59 value=False
prim::Constant          op_20       0 1 60 value=None
aten::to                op_21       5 1 24 57 58 59 60 28
aten::floor             op_22       1 1 28 31
aten::Int               op_23       1 1 31 33
prim::ListConstruct     op_24       1 1 33 size
prim::Constant          op_25       0 1 scale_factor value=None
aten::upsample_nearest1d op_26      3 1 input size scale_factor out
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate, 10)

class F_interpolate_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
30 29
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 scale value=%scale
prim::Constant          op_1        0 1 5 value=2
aten::size              op_2        2 1 input 5 6
prim::NumToTensor       op_3        1 1 6 7
prim::Constant          op_4        0 1 9 value=6
prim::Constant          op_5        0 1 10 value=False
prim::Constant          op_6        0 1 52 value=False
prim::Constant          op_7        0 1 12 value=None
aten::to                op_8        5 1 7 9 10 52 12 13
prim::Constant          op_9        0 1 51 value=*
prim::Constant          op_10       0 1 53 value=6
prim::Constant          op_11       0 1 54 value=False
prim::Constant          op_12       0 1 55 value=False
prim::Constant          op_13       0 1 56 value=None
aten::to                op_14       6 1 scale 51 53 54 55 56 20
aten::detach            op_15       1 1 20 23
aten::mul               op_16       2 1 13 23 24
prim::Constant          op_17       0 1 57 value=6
prim::Constant          op_18       0 1 58 value=False
prim::Constant          op_19       0 1 59 value=False
prim::Constant          op_20       0 1 60 value=None
aten::to                op_21       5 1 24 57 58 59 60 28
aten::floor             op_22       1 1 28 31
aten::Int               op_23       1 1 31 33
prim::ListConstruct     op_24       1 1 33 size
prim::Constant          op_25       0 1 align_corners value=%align_corners
prim::Constant          op_26       0 1 scale_factor value=None
aten::upsample_linear1d op_27       4 1 input size align_corners scale_factor out
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_1, 10)

class F_interpolate_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
54 53
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 scale_w value=%scale_w
prim::Constant          op_1        0 1 81 value=*
prim::Constant          op_2        0 1 scale_h value=%scale_h
prim::Constant          op_3        0 1 12 value=None
prim::Constant          op_4        0 1 10 value=False
prim::Constant          op_5        0 1 5 value=2
prim::Constant          op_6        0 1 9 value=6
prim::Constant          op_7        0 1 34 value=3
aten::size              op_8        2 1 input 5 6
prim::NumToTensor       op_9        1 1 6 7
prim::Constant          op_10       0 1 83 value=False
aten::to                op_11       5 1 7 9 10 83 12 13
prim::Constant          op_12       0 1 84 value=6
prim::Constant          op_13       0 1 85 value=False
prim::Constant          op_14       0 1 86 value=False
prim::Constant          op_15       0 1 87 value=None
aten::to                op_16       6 1 scale_h 81 84 85 86 87 20
aten::detach            op_17       1 1 20 23
aten::mul               op_18       2 1 13 23 24
prim::Constant          op_19       0 1 88 value=6
prim::Constant          op_20       0 1 89 value=False
prim::Constant          op_21       0 1 90 value=False
prim::Constant          op_22       0 1 91 value=None
aten::to                op_23       5 1 24 88 89 90 91 28
aten::floor             op_24       1 1 28 30
aten::Int               op_25       1 1 30 32
aten::size              op_26       2 1 input 34 35
prim::NumToTensor       op_27       1 1 35 36
prim::Constant          op_28       0 1 92 value=6
prim::Constant          op_29       0 1 93 value=False
prim::Constant          op_30       0 1 94 value=False
prim::Constant          op_31       0 1 95 value=None
aten::to                op_32       5 1 36 92 93 94 95 41
prim::Constant          op_33       0 1 96 value=*
prim::Constant          op_34       0 1 97 value=6
prim::Constant          op_35       0 1 98 value=False
prim::Constant          op_36       0 1 99 value=False
prim::Constant          op_37       0 1 100 value=None
aten::to                op_38       6 1 scale_w 96 97 98 99 100 48
aten::detach            op_39       1 1 48 51
aten::mul               op_40       2 1 41 51 52
prim::Constant          op_41       0 1 101 value=6
prim::Constant          op_42       0 1 102 value=False
prim::Constant          op_43       0 1 103 value=False
prim::Constant          op_44       0 1 104 value=None
aten::to                op_45       5 1 52 101 102 103 104 56
aten::floor             op_46       1 1 56 60
aten::Int               op_47       1 1 60 62
prim::ListConstruct     op_48       2 1 32 62 size
prim::Constant          op_49       0 1 scale_h_none value=None
prim::Constant          op_50       0 1 scale_w_none value=None
aten::upsample_nearest2d op_51      4 1 input size scale_h_none scale_w_none out
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_2, 10)

class F_interpolate_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
55 54
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 scale_w value=%scale_w
prim::Constant          op_1        0 1 82 value=*
prim::Constant          op_2        0 1 scale_h value=%scale_h
prim::Constant          op_3        0 1 12 value=None
prim::Constant          op_4        0 1 10 value=False
prim::Constant          op_5        0 1 5 value=2
prim::Constant          op_6        0 1 9 value=6
prim::Constant          op_7        0 1 34 value=3
aten::size              op_8        2 1 input 5 6
prim::NumToTensor       op_9        1 1 6 7
prim::Constant          op_10       0 1 84 value=False
aten::to                op_11       5 1 7 9 10 84 12 13
prim::Constant          op_12       0 1 85 value=6
prim::Constant          op_13       0 1 86 value=False
prim::Constant          op_14       0 1 87 value=False
prim::Constant          op_15       0 1 88 value=None
aten::to                op_16       6 1 scale_h 82 85 86 87 88 20
aten::detach            op_17       1 1 20 23
aten::mul               op_18       2 1 13 23 24
prim::Constant          op_19       0 1 89 value=6
prim::Constant          op_20       0 1 90 value=False
prim::Constant          op_21       0 1 91 value=False
prim::Constant          op_22       0 1 92 value=None
aten::to                op_23       5 1 24 89 90 91 92 28
aten::floor             op_24       1 1 28 30
aten::Int               op_25       1 1 30 32
aten::size              op_26       2 1 input 34 35
prim::NumToTensor       op_27       1 1 35 36
prim::Constant          op_28       0 1 93 value=6
prim::Constant          op_29       0 1 94 value=False
prim::Constant          op_30       0 1 95 value=False
prim::Constant          op_31       0 1 96 value=None
aten::to                op_32       5 1 36 93 94 95 96 41
prim::Constant          op_33       0 1 97 value=*
prim::Constant          op_34       0 1 98 value=6
prim::Constant          op_35       0 1 99 value=False
prim::Constant          op_36       0 1 100 value=False
prim::Constant          op_37       0 1 101 value=None
aten::to                op_38       6 1 scale_w 97 98 99 100 101 48
aten::detach            op_39       1 1 48 51
aten::mul               op_40       2 1 41 51 52
prim::Constant          op_41       0 1 102 value=6
prim::Constant          op_42       0 1 103 value=False
prim::Constant          op_43       0 1 104 value=False
prim::Constant          op_44       0 1 105 value=None
aten::to                op_45       5 1 52 102 103 104 105 56
aten::floor             op_46       1 1 56 60
aten::Int               op_47       1 1 60 62
prim::ListConstruct     op_48       2 1 32 62 size
prim::Constant          op_49       0 1 align_corners value=%align_corners
prim::Constant          op_50       0 1 scale_h_none value=None
prim::Constant          op_51       0 1 scale_w_none value=None
aten::upsample_bilinear2d op_52     5 1 input size align_corners scale_h_none scale_w_none out
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_3, 10)

class F_interpolate_3_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
54 53
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 scale_w value=%scale_w
prim::Constant          op_1        0 1 82 value=*
prim::Constant          op_2        0 1 scale_h value=%scale_h
prim::Constant          op_3        0 1 12 value=None
prim::Constant          op_4        0 1 10 value=False
prim::Constant          op_5        0 1 5 value=2
prim::Constant          op_6        0 1 9 value=6
prim::Constant          op_7        0 1 34 value=3
aten::size              op_8        2 1 input 5 6
prim::NumToTensor       op_9        1 1 6 7
prim::Constant          op_10       0 1 84 value=False
aten::to                op_11       5 1 7 9 10 84 12 13
prim::Constant          op_12       0 1 85 value=6
prim::Constant          op_13       0 1 86 value=False
prim::Constant          op_14       0 1 87 value=False
prim::Constant          op_15       0 1 88 value=None
aten::to                op_16       6 1 scale_h 82 85 86 87 88 20
aten::detach            op_17       1 1 20 23
aten::mul               op_18       2 1 13 23 24
prim::Constant          op_19       0 1 89 value=6
prim::Constant          op_20       0 1 90 value=False
prim::Constant          op_21       0 1 91 value=False
prim::Constant          op_22       0 1 92 value=None
aten::to                op_23       5 1 24 89 90 91 92 28
aten::floor             op_24       1 1 28 30
aten::Int               op_25       1 1 30 32
aten::size              op_26       2 1 input 34 35
prim::NumToTensor       op_27       1 1 35 36
prim::Constant          op_28       0 1 93 value=6
prim::Constant          op_29       0 1 94 value=False
prim::Constant          op_30       0 1 95 value=False
prim::Constant          op_31       0 1 96 value=None
aten::to                op_32       5 1 36 93 94 95 96 41
prim::Constant          op_33       0 1 97 value=*
prim::Constant          op_34       0 1 98 value=6
prim::Constant          op_35       0 1 99 value=False
prim::Constant          op_36       0 1 100 value=False
prim::Constant          op_37       0 1 101 value=None
aten::to                op_38       6 1 scale_w 97 98 99 100 101 48
aten::detach            op_39       1 1 48 51
aten::mul               op_40       2 1 41 51 52
prim::Constant          op_41       0 1 102 value=6
prim::Constant          op_42       0 1 103 value=False
prim::Constant          op_43       0 1 104 value=False
prim::Constant          op_44       0 1 105 value=None
aten::to                op_45       5 1 52 102 103 104 105 56
aten::floor             op_46       1 1 56 60
aten::Int               op_47       1 1 60 62
prim::ListConstruct     op_48       2 1 32 62 size
prim::Constant          op_49       0 1 align_corners value=%align_corners
prim::Constant          op_50       0 1 scale_factor value=None
aten::upsample_bilinear2d op_51     4 1 input size align_corners scale_factor out
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_3_1, 10)

class F_interpolate_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
55 54
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 scale_w value=%scale_w
prim::Constant          op_1        0 1 82 value=*
prim::Constant          op_2        0 1 scale_h value=%scale_h
prim::Constant          op_3        0 1 12 value=None
prim::Constant          op_4        0 1 10 value=False
prim::Constant          op_5        0 1 5 value=2
prim::Constant          op_6        0 1 9 value=6
prim::Constant          op_7        0 1 34 value=3
aten::size              op_8        2 1 input 5 6
prim::NumToTensor       op_9        1 1 6 7
prim::Constant          op_10       0 1 84 value=False
aten::to                op_11       5 1 7 9 10 84 12 13
prim::Constant          op_12       0 1 85 value=6
prim::Constant          op_13       0 1 86 value=False
prim::Constant          op_14       0 1 87 value=False
prim::Constant          op_15       0 1 88 value=None
aten::to                op_16       6 1 scale_h 82 85 86 87 88 20
aten::detach            op_17       1 1 20 23
aten::mul               op_18       2 1 13 23 24
prim::Constant          op_19       0 1 89 value=6
prim::Constant          op_20       0 1 90 value=False
prim::Constant          op_21       0 1 91 value=False
prim::Constant          op_22       0 1 92 value=None
aten::to                op_23       5 1 24 89 90 91 92 28
aten::floor             op_24       1 1 28 30
aten::Int               op_25       1 1 30 32
aten::size              op_26       2 1 input 34 35
prim::NumToTensor       op_27       1 1 35 36
prim::Constant          op_28       0 1 93 value=6
prim::Constant          op_29       0 1 94 value=False
prim::Constant          op_30       0 1 95 value=False
prim::Constant          op_31       0 1 96 value=None
aten::to                op_32       5 1 36 93 94 95 96 41
prim::Constant          op_33       0 1 97 value=*
prim::Constant          op_34       0 1 98 value=6
prim::Constant          op_35       0 1 99 value=False
prim::Constant          op_36       0 1 100 value=False
prim::Constant          op_37       0 1 101 value=None
aten::to                op_38       6 1 scale_w 97 98 99 100 101 48
aten::detach            op_39       1 1 48 51
aten::mul               op_40       2 1 41 51 52
prim::Constant          op_41       0 1 102 value=6
prim::Constant          op_42       0 1 103 value=False
prim::Constant          op_43       0 1 104 value=False
prim::Constant          op_44       0 1 105 value=None
aten::to                op_45       5 1 52 102 103 104 105 56
aten::floor             op_46       1 1 56 60
aten::Int               op_47       1 1 60 62
prim::ListConstruct     op_48       2 1 32 62 size
prim::Constant          op_49       0 1 align_corners value=%align_corners
prim::Constant          op_50       0 1 scale_h_none value=None
prim::Constant          op_51       0 1 scale_w_none value=None
aten::upsample_bicubic2d op_52      5 1 input size align_corners scale_h_none scale_w_none out
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_4, 10)

class F_interpolate_4_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
54 53
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 scale_w value=%scale_w
prim::Constant          op_1        0 1 82 value=*
prim::Constant          op_2        0 1 scale_h value=%scale_h
prim::Constant          op_3        0 1 12 value=None
prim::Constant          op_4        0 1 10 value=False
prim::Constant          op_5        0 1 5 value=2
prim::Constant          op_6        0 1 9 value=6
prim::Constant          op_7        0 1 34 value=3
aten::size              op_8        2 1 input 5 6
prim::NumToTensor       op_9        1 1 6 7
prim::Constant          op_10       0 1 84 value=False
aten::to                op_11       5 1 7 9 10 84 12 13
prim::Constant          op_12       0 1 85 value=6
prim::Constant          op_13       0 1 86 value=False
prim::Constant          op_14       0 1 87 value=False
prim::Constant          op_15       0 1 88 value=None
aten::to                op_16       6 1 scale_h 82 85 86 87 88 20
aten::detach            op_17       1 1 20 23
aten::mul               op_18       2 1 13 23 24
prim::Constant          op_19       0 1 89 value=6
prim::Constant          op_20       0 1 90 value=False
prim::Constant          op_21       0 1 91 value=False
prim::Constant          op_22       0 1 92 value=None
aten::to                op_23       5 1 24 89 90 91 92 28
aten::floor             op_24       1 1 28 30
aten::Int               op_25       1 1 30 32
aten::size              op_26       2 1 input 34 35
prim::NumToTensor       op_27       1 1 35 36
prim::Constant          op_28       0 1 93 value=6
prim::Constant          op_29       0 1 94 value=False
prim::Constant          op_30       0 1 95 value=False
prim::Constant          op_31       0 1 96 value=None
aten::to                op_32       5 1 36 93 94 95 96 41
prim::Constant          op_33       0 1 97 value=*
prim::Constant          op_34       0 1 98 value=6
prim::Constant          op_35       0 1 99 value=False
prim::Constant          op_36       0 1 100 value=False
prim::Constant          op_37       0 1 101 value=None
aten::to                op_38       6 1 scale_w 97 98 99 100 101 48
aten::detach            op_39       1 1 48 51
aten::mul               op_40       2 1 41 51 52
prim::Constant          op_41       0 1 102 value=6
prim::Constant          op_42       0 1 103 value=False
prim::Constant          op_43       0 1 104 value=False
prim::Constant          op_44       0 1 105 value=None
aten::to                op_45       5 1 52 102 103 104 105 56
aten::floor             op_46       1 1 56 60
aten::Int               op_47       1 1 60 62
prim::ListConstruct     op_48       2 1 32 62 size
prim::Constant          op_49       0 1 align_corners value=%align_corners
prim::Constant          op_50       0 1 scale_factor value=None
aten::upsample_bicubic2d op_51      4 1 input size align_corners scale_factor out
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_4_1, 10)

class F_interpolate_5 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
79 78
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 scale_w value=%scale_w
prim::Constant          op_1        0 1 scale_h value=%scale_h
prim::Constant          op_2        0 1 108 value=*
prim::Constant          op_3        0 1 scale_d value=%scale_d
prim::Constant          op_4        0 1 12 value=None
prim::Constant          op_5        0 1 10 value=False
prim::Constant          op_6        0 1 5 value=2
prim::Constant          op_7        0 1 9 value=6
prim::Constant          op_8        0 1 34 value=3
prim::Constant          op_9        0 1 62 value=4
aten::size              op_10       2 1 input 5 6
prim::NumToTensor       op_11       1 1 6 7
prim::Constant          op_12       0 1 111 value=False
aten::to                op_13       5 1 7 9 10 111 12 13
prim::Constant          op_14       0 1 112 value=6
prim::Constant          op_15       0 1 113 value=False
prim::Constant          op_16       0 1 114 value=False
prim::Constant          op_17       0 1 115 value=None
aten::to                op_18       6 1 scale_d 108 112 113 114 115 20
aten::detach            op_19       1 1 20 23
aten::mul               op_20       2 1 13 23 24
prim::Constant          op_21       0 1 116 value=6
prim::Constant          op_22       0 1 117 value=False
prim::Constant          op_23       0 1 118 value=False
prim::Constant          op_24       0 1 119 value=None
aten::to                op_25       5 1 24 116 117 118 119 28
aten::floor             op_26       1 1 28 30
aten::Int               op_27       1 1 30 32
aten::size              op_28       2 1 input 34 35
prim::NumToTensor       op_29       1 1 35 36
prim::Constant          op_30       0 1 120 value=6
prim::Constant          op_31       0 1 121 value=False
prim::Constant          op_32       0 1 122 value=False
prim::Constant          op_33       0 1 123 value=None
aten::to                op_34       5 1 36 120 121 122 123 41
prim::Constant          op_35       0 1 124 value=*
prim::Constant          op_36       0 1 125 value=6
prim::Constant          op_37       0 1 126 value=False
prim::Constant          op_38       0 1 127 value=False
prim::Constant          op_39       0 1 128 value=None
aten::to                op_40       6 1 scale_h 124 125 126 127 128 48
aten::detach            op_41       1 1 48 51
aten::mul               op_42       2 1 41 51 52
prim::Constant          op_43       0 1 129 value=6
prim::Constant          op_44       0 1 130 value=False
prim::Constant          op_45       0 1 131 value=False
prim::Constant          op_46       0 1 132 value=None
aten::to                op_47       5 1 52 129 130 131 132 56
aten::floor             op_48       1 1 56 58
aten::Int               op_49       1 1 58 60
aten::size              op_50       2 1 input 62 63
prim::NumToTensor       op_51       1 1 63 64
prim::Constant          op_52       0 1 133 value=6
prim::Constant          op_53       0 1 134 value=False
prim::Constant          op_54       0 1 135 value=False
prim::Constant          op_55       0 1 136 value=None
aten::to                op_56       5 1 64 133 134 135 136 69
prim::Constant          op_57       0 1 137 value=*
prim::Constant          op_58       0 1 138 value=6
prim::Constant          op_59       0 1 139 value=False
prim::Constant          op_60       0 1 140 value=False
prim::Constant          op_61       0 1 141 value=None
aten::to                op_62       6 1 scale_w 137 138 139 140 141 76
aten::detach            op_63       1 1 76 79
aten::mul               op_64       2 1 69 79 80
prim::Constant          op_65       0 1 142 value=6
prim::Constant          op_66       0 1 143 value=False
prim::Constant          op_67       0 1 144 value=False
prim::Constant          op_68       0 1 145 value=None
aten::to                op_69       5 1 80 142 143 144 145 84
aten::floor             op_70       1 1 84 89
aten::Int               op_71       1 1 89 91
prim::ListConstruct     op_72       3 1 32 60 91 size
prim::Constant          op_73       0 1 scale_d_none value=None
prim::Constant          op_74       0 1 scale_h_none value=None
prim::Constant          op_75       0 1 scale_w_none value=None
aten::upsample_nearest3d op_76      5 1 input size scale_d_none scale_h_none scale_w_none out
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_5, 10)

class F_interpolate_5_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
77 76
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 scale_w value=%scale_w
prim::Constant          op_1        0 1 scale_h value=%scale_h
prim::Constant          op_2        0 1 108 value=*
prim::Constant          op_3        0 1 scale_d value=%scale_d
prim::Constant          op_4        0 1 12 value=None
prim::Constant          op_5        0 1 10 value=False
prim::Constant          op_6        0 1 5 value=2
prim::Constant          op_7        0 1 9 value=6
prim::Constant          op_8        0 1 34 value=3
prim::Constant          op_9        0 1 62 value=4
aten::size              op_10       2 1 input 5 6
prim::NumToTensor       op_11       1 1 6 7
prim::Constant          op_12       0 1 111 value=False
aten::to                op_13       5 1 7 9 10 111 12 13
prim::Constant          op_14       0 1 112 value=6
prim::Constant          op_15       0 1 113 value=False
prim::Constant          op_16       0 1 114 value=False
prim::Constant          op_17       0 1 115 value=None
aten::to                op_18       6 1 scale_d 108 112 113 114 115 20
aten::detach            op_19       1 1 20 23
aten::mul               op_20       2 1 13 23 24
prim::Constant          op_21       0 1 116 value=6
prim::Constant          op_22       0 1 117 value=False
prim::Constant          op_23       0 1 118 value=False
prim::Constant          op_24       0 1 119 value=None
aten::to                op_25       5 1 24 116 117 118 119 28
aten::floor             op_26       1 1 28 30
aten::Int               op_27       1 1 30 32
aten::size              op_28       2 1 input 34 35
prim::NumToTensor       op_29       1 1 35 36
prim::Constant          op_30       0 1 120 value=6
prim::Constant          op_31       0 1 121 value=False
prim::Constant          op_32       0 1 122 value=False
prim::Constant          op_33       0 1 123 value=None
aten::to                op_34       5 1 36 120 121 122 123 41
prim::Constant          op_35       0 1 124 value=*
prim::Constant          op_36       0 1 125 value=6
prim::Constant          op_37       0 1 126 value=False
prim::Constant          op_38       0 1 127 value=False
prim::Constant          op_39       0 1 128 value=None
aten::to                op_40       6 1 scale_h 124 125 126 127 128 48
aten::detach            op_41       1 1 48 51
aten::mul               op_42       2 1 41 51 52
prim::Constant          op_43       0 1 129 value=6
prim::Constant          op_44       0 1 130 value=False
prim::Constant          op_45       0 1 131 value=False
prim::Constant          op_46       0 1 132 value=None
aten::to                op_47       5 1 52 129 130 131 132 56
aten::floor             op_48       1 1 56 58
aten::Int               op_49       1 1 58 60
aten::size              op_50       2 1 input 62 63
prim::NumToTensor       op_51       1 1 63 64
prim::Constant          op_52       0 1 133 value=6
prim::Constant          op_53       0 1 134 value=False
prim::Constant          op_54       0 1 135 value=False
prim::Constant          op_55       0 1 136 value=None
aten::to                op_56       5 1 64 133 134 135 136 69
prim::Constant          op_57       0 1 137 value=*
prim::Constant          op_58       0 1 138 value=6
prim::Constant          op_59       0 1 139 value=False
prim::Constant          op_60       0 1 140 value=False
prim::Constant          op_61       0 1 141 value=None
aten::to                op_62       6 1 scale_w 137 138 139 140 141 76
aten::detach            op_63       1 1 76 79
aten::mul               op_64       2 1 69 79 80
prim::Constant          op_65       0 1 142 value=6
prim::Constant          op_66       0 1 143 value=False
prim::Constant          op_67       0 1 144 value=False
prim::Constant          op_68       0 1 145 value=None
aten::to                op_69       5 1 80 142 143 144 145 84
aten::floor             op_70       1 1 84 89
aten::Int               op_71       1 1 89 91
prim::ListConstruct     op_72       3 1 32 60 91 size
prim::Constant          op_73       0 1 scale_factor value=None
aten::upsample_nearest3d op_74      3 1 input size scale_factor out
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_5_1, 10)

class F_interpolate_6 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
80 79
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 scale_w value=%scale_w
prim::Constant          op_1        0 1 scale_h value=%scale_h
prim::Constant          op_2        0 1 113 value=*
prim::Constant          op_3        0 1 scale_d value=%scale_d
prim::Constant          op_4        0 1 12 value=None
prim::Constant          op_5        0 1 10 value=False
prim::Constant          op_6        0 1 5 value=2
prim::Constant          op_7        0 1 9 value=6
prim::Constant          op_8        0 1 34 value=3
prim::Constant          op_9        0 1 62 value=4
aten::size              op_10       2 1 input 5 6
prim::NumToTensor       op_11       1 1 6 7
prim::Constant          op_12       0 1 116 value=False
aten::to                op_13       5 1 7 9 10 116 12 13
prim::Constant          op_14       0 1 117 value=6
prim::Constant          op_15       0 1 118 value=False
prim::Constant          op_16       0 1 119 value=False
prim::Constant          op_17       0 1 120 value=None
aten::to                op_18       6 1 scale_d 113 117 118 119 120 20
aten::detach            op_19       1 1 20 23
aten::mul               op_20       2 1 13 23 24
prim::Constant          op_21       0 1 121 value=6
prim::Constant          op_22       0 1 122 value=False
prim::Constant          op_23       0 1 123 value=False
prim::Constant          op_24       0 1 124 value=None
aten::to                op_25       5 1 24 121 122 123 124 28
aten::floor             op_26       1 1 28 30
aten::Int               op_27       1 1 30 32
aten::size              op_28       2 1 input 34 35
prim::NumToTensor       op_29       1 1 35 36
prim::Constant          op_30       0 1 125 value=6
prim::Constant          op_31       0 1 126 value=False
prim::Constant          op_32       0 1 127 value=False
prim::Constant          op_33       0 1 128 value=None
aten::to                op_34       5 1 36 125 126 127 128 41
prim::Constant          op_35       0 1 129 value=*
prim::Constant          op_36       0 1 130 value=6
prim::Constant          op_37       0 1 131 value=False
prim::Constant          op_38       0 1 132 value=False
prim::Constant          op_39       0 1 133 value=None
aten::to                op_40       6 1 scale_h 129 130 131 132 133 48
aten::detach            op_41       1 1 48 51
aten::mul               op_42       2 1 41 51 52
prim::Constant          op_43       0 1 134 value=6
prim::Constant          op_44       0 1 135 value=False
prim::Constant          op_45       0 1 136 value=False
prim::Constant          op_46       0 1 137 value=None
aten::to                op_47       5 1 52 134 135 136 137 56
aten::floor             op_48       1 1 56 58
aten::Int               op_49       1 1 58 60
aten::size              op_50       2 1 input 62 63
prim::NumToTensor       op_51       1 1 63 64
prim::Constant          op_52       0 1 138 value=6
prim::Constant          op_53       0 1 139 value=False
prim::Constant          op_54       0 1 140 value=False
prim::Constant          op_55       0 1 141 value=None
aten::to                op_56       5 1 64 138 139 140 141 69
prim::Constant          op_57       0 1 142 value=*
prim::Constant          op_58       0 1 143 value=6
prim::Constant          op_59       0 1 144 value=False
prim::Constant          op_60       0 1 145 value=False
prim::Constant          op_61       0 1 146 value=None
aten::to                op_62       6 1 scale_w 142 143 144 145 146 76
aten::detach            op_63       1 1 76 79
aten::mul               op_64       2 1 69 79 80
prim::Constant          op_65       0 1 147 value=6
prim::Constant          op_66       0 1 148 value=False
prim::Constant          op_67       0 1 149 value=False
prim::Constant          op_68       0 1 150 value=None
aten::to                op_69       5 1 80 147 148 149 150 84
aten::floor             op_70       1 1 84 89
aten::Int               op_71       1 1 89 91
prim::ListConstruct     op_72       3 1 32 60 91 size
prim::Constant          op_73       0 1 align_corners value=%align_corners
prim::Constant          op_74       0 1 scale_d_none value=None
prim::Constant          op_75       0 1 scale_h_none value=None
prim::Constant          op_76       0 1 scale_w_none value=None
aten::upsample_trilinear3d op_77    6 1 input size align_corners scale_d_none scale_h_none scale_w_none out
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_6, 10)

class F_interpolate_6_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
78 77
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 scale_w value=%scale_w
prim::Constant          op_1        0 1 scale_h value=%scale_h
prim::Constant          op_2        0 1 113 value=*
prim::Constant          op_3        0 1 scale_d value=%scale_d
prim::Constant          op_4        0 1 12 value=None
prim::Constant          op_5        0 1 10 value=False
prim::Constant          op_6        0 1 5 value=2
prim::Constant          op_7        0 1 9 value=6
prim::Constant          op_8        0 1 34 value=3
prim::Constant          op_9        0 1 62 value=4
aten::size              op_10       2 1 input 5 6
prim::NumToTensor       op_11       1 1 6 7
prim::Constant          op_12       0 1 116 value=False
aten::to                op_13       5 1 7 9 10 116 12 13
prim::Constant          op_14       0 1 117 value=6
prim::Constant          op_15       0 1 118 value=False
prim::Constant          op_16       0 1 119 value=False
prim::Constant          op_17       0 1 120 value=None
aten::to                op_18       6 1 scale_d 113 117 118 119 120 20
aten::detach            op_19       1 1 20 23
aten::mul               op_20       2 1 13 23 24
prim::Constant          op_21       0 1 121 value=6
prim::Constant          op_22       0 1 122 value=False
prim::Constant          op_23       0 1 123 value=False
prim::Constant          op_24       0 1 124 value=None
aten::to                op_25       5 1 24 121 122 123 124 28
aten::floor             op_26       1 1 28 30
aten::Int               op_27       1 1 30 32
aten::size              op_28       2 1 input 34 35
prim::NumToTensor       op_29       1 1 35 36
prim::Constant          op_30       0 1 125 value=6
prim::Constant          op_31       0 1 126 value=False
prim::Constant          op_32       0 1 127 value=False
prim::Constant          op_33       0 1 128 value=None
aten::to                op_34       5 1 36 125 126 127 128 41
prim::Constant          op_35       0 1 129 value=*
prim::Constant          op_36       0 1 130 value=6
prim::Constant          op_37       0 1 131 value=False
prim::Constant          op_38       0 1 132 value=False
prim::Constant          op_39       0 1 133 value=None
aten::to                op_40       6 1 scale_h 129 130 131 132 133 48
aten::detach            op_41       1 1 48 51
aten::mul               op_42       2 1 41 51 52
prim::Constant          op_43       0 1 134 value=6
prim::Constant          op_44       0 1 135 value=False
prim::Constant          op_45       0 1 136 value=False
prim::Constant          op_46       0 1 137 value=None
aten::to                op_47       5 1 52 134 135 136 137 56
aten::floor             op_48       1 1 56 58
aten::Int               op_49       1 1 58 60
aten::size              op_50       2 1 input 62 63
prim::NumToTensor       op_51       1 1 63 64
prim::Constant          op_52       0 1 138 value=6
prim::Constant          op_53       0 1 139 value=False
prim::Constant          op_54       0 1 140 value=False
prim::Constant          op_55       0 1 141 value=None
aten::to                op_56       5 1 64 138 139 140 141 69
prim::Constant          op_57       0 1 142 value=*
prim::Constant          op_58       0 1 143 value=6
prim::Constant          op_59       0 1 144 value=False
prim::Constant          op_60       0 1 145 value=False
prim::Constant          op_61       0 1 146 value=None
aten::to                op_62       6 1 scale_w 142 143 144 145 146 76
aten::detach            op_63       1 1 76 79
aten::mul               op_64       2 1 69 79 80
prim::Constant          op_65       0 1 147 value=6
prim::Constant          op_66       0 1 148 value=False
prim::Constant          op_67       0 1 149 value=False
prim::Constant          op_68       0 1 150 value=None
aten::to                op_69       5 1 80 147 148 149 150 84
aten::floor             op_70       1 1 84 89
aten::Int               op_71       1 1 89 91
prim::ListConstruct     op_72       3 1 32 60 91 size
prim::Constant          op_73       0 1 align_corners value=%align_corners
prim::Constant          op_74       0 1 scale_factor value=None
aten::upsample_trilinear3d op_75    4 1 input size align_corners scale_factor out
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_6_1, 10)

} // namespace pnnx
