// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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

class F_interpolate_nearest_exact1d : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 size value=None
prim::Constant          op_1        0 1 scale_factor value=%scale_factor
aten::_upsample_nearest_exact1d op_2 3 1 input size scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["scale_factor"] = captured_params.at("scale_factor");
        op->params["mode"] = "nearest-exact";
        op->params["recompute_scale_factor"] = false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_nearest_exact1d, 110)

class F_interpolate_nearest_exact1d_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 size value=%size
prim::Constant          op_1        0 1 scale_factor value=None
aten::_upsample_nearest_exact1d op_2 3 1 input size scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["size"] = captured_params.at("size");
        op->params["mode"] = "nearest-exact";
        op->params["recompute_scale_factor"] = false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_nearest_exact1d_1, 110)

class F_interpolate_nearest_exact2d : public F_interpolate_nearest_exact1d
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 size value=None
prim::Constant          op_1        0 1 scale_factor value=%scale_factor
aten::_upsample_nearest_exact2d op_2 3 1 input size scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_nearest_exact2d, 110)

class F_interpolate_nearest_exact2d_1 : public F_interpolate_nearest_exact1d_1
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 size value=%size
prim::Constant          op_1        0 1 scale_factor_h value=None
prim::Constant          op_2        0 1 scale_factor_w value=None
aten::_upsample_nearest_exact2d op_3 4 1 input size scale_factor_h scale_factor_w out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_nearest_exact2d_1, 110)

class F_interpolate_nearest_exact3d : public F_interpolate_nearest_exact1d
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 size value=None
prim::Constant          op_1        0 1 scale_factor value=%scale_factor
aten::_upsample_nearest_exact3d op_2 3 1 input size scale_factor out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_nearest_exact3d, 110)

class F_interpolate_nearest_exact3d_1 : public F_interpolate_nearest_exact1d_1
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 size value=%size
prim::Constant          op_1        0 1 scale_factor_d value=None
prim::Constant          op_2        0 1 scale_factor_h value=None
prim::Constant          op_3        0 1 scale_factor_w value=None
aten::_upsample_nearest_exact3d op_4 5 1 input size scale_factor_d scale_factor_h scale_factor_w out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_nearest_exact3d_1, 110)

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
        if (captured_params.find("op_0.coordinate_transformation_mode") != captured_params.end())
        {
            if (captured_params.at("op_0.coordinate_transformation_mode").type != 4)
                return false;
        }

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
            if (captured_params.at("op_0.roi").type != 6 || !captured_params.at("op_0.roi").af.empty())
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
        std::string coordinate_transformation_mode = "half_pixel";
        if (captured_params.find("op_0.coordinate_transformation_mode") != captured_params.end())
        {
            coordinate_transformation_mode = captured_params.at("op_0.coordinate_transformation_mode").s;
        }

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

class F_interpolate_onnx_dynamic : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 size
Resize                  op_0        2 1 input size out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    bool match(const std::map<std::string, const Operator*>& /*matched_operators*/, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        if (captured_params.find("op_0.sizes") != captured_params.end())
            return false;

        if (captured_params.find("op_0.scales") != captured_params.end())
        {
            if (captured_params.at("op_0.scales").type != 6 || !captured_params.at("op_0.scales").af.empty())
                return false;
        }

        if (captured_params.find("op_0.roi") != captured_params.end())
        {
            if (captured_params.at("op_0.roi").type != 6 || !captured_params.at("op_0.roi").af.empty())
                return false;
        }

        if (captured_params.find("op_0.coordinate_transformation_mode") != captured_params.end())
        {
            if (captured_params.at("op_0.coordinate_transformation_mode").type != 4)
                return false;
        }

        if (captured_params.find("op_0.mode") == captured_params.end())
            return false;

        if (captured_params.at("op_0.mode").type != 4)
            return false;

        if (captured_params.find("op_0.nearest_mode") != captured_params.end())
        {
            if (captured_params.at("op_0.nearest_mode").type != 4 || captured_params.at("op_0.nearest_mode").s != "floor")
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int input_rank = op->inputs[0]->shape.size();

        std::string coordinate_transformation_mode = "half_pixel";
        if (captured_params.find("op_0.coordinate_transformation_mode") != captured_params.end())
        {
            coordinate_transformation_mode = captured_params.at("op_0.coordinate_transformation_mode").s;
        }

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

        if (mode == "linear")
        {
            if (input_rank == 3)
                mode = "linear";
            else if (input_rank == 5)
                mode = "trilinear";
            else
                mode = "bilinear";
        }

        if (mode == "cubic")
        {
            mode = "bicubic";
        }

        op->params["mode"] = mode;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_onnx_dynamic, 110)

static void linear_coeffs(int w, int outw, bool align_corner, std::vector<int>& ia, std::vector<int>& ib, std::vector<float>& im)
{
    ia.resize(outw);
    ib.resize(outw);
    im.resize(outw);

    double scale = (double)w / outw;
    if (align_corner)
    {
        scale = (double)(w - 1) / (outw - 1);
    }

    for (int dx = 0; dx < outw; dx++)
    {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        if (align_corner)
        {
            fx = (float)(dx * scale);
        }

        int sx = (int)floor(fx);
        fx -= sx;

        ia[dx] = sx;
        ib[dx] = sx + 1;
        im[dx] = fx;

        // fx is non-sense on the border, but we have to follow dynamo onnx style
        if (sx < 0)
        {
            ia[dx] = 0;
            ib[dx] = 1;
            im[dx] = 0.f;
        }
        if (sx >= w - 1)
        {
            ia[dx] = w - 1;
            ib[dx] = w - 1;
            if (align_corner)
                im[dx] = 0.f;
        }
    }
}

static bool NearlyEqual(float a, float b, float epsilon)
{
    if (a == b)
        return true;

    float diff = (float)fabs(a - b);
    if (diff <= epsilon)
        return true;

    // relative error
    return diff < epsilon * std::max(fabs(a), fabs(b));
}

static bool resolve_align_corners(int w, int outw, const int64_t* pa, const int64_t* pb, const float* pm)
{
    std::vector<int> ia;
    std::vector<int> ib;
    std::vector<float> im;

    bool is_no_align_corners = true;
    linear_coeffs(w, outw, false, ia, ib, im);
    for (int i = 0; i < outw; i++)
    {
        if (ia[i] != pa[i] || ib[i] != pb[i])
        {
            is_no_align_corners = false;
            break;
        }

        if (!NearlyEqual(im[i], pm[i], 0.001))
        {
            is_no_align_corners = false;
            break;
        }
    }

    if (is_no_align_corners)
        return false;

    bool is_align_corners = true;
    linear_coeffs(w, outw, true, ia, ib, im);
    for (int i = 0; i < outw; i++)
    {
        if (ia[i] != pa[i] || ib[i] != pb[i])
        {
            is_align_corners = false;
            break;
        }

        if (!NearlyEqual(im[i], pm[i], 0.001))
        {
            is_align_corners = false;
            break;
        }
    }

    if (is_align_corners)
        return true;

    fprintf(stderr, "unsupported interpolate align_corners, assume false\n");
    return false;
}

class F_interpolate_onnx_1d_linear : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
13 12
pnnx.Input              input       0 1 input
Tensor.permute          op_0        1 1 input pnnx_4 dims=(2,0,1)
pnnx.Attribute          op_1        0 1 val_30 @data=(%size,1)i64
GatherND                op_2        2 1 pnnx_4 val_30 pnnx_5 batch_dims=0
Tensor.permute          op_3        1 1 pnnx_5 pnnx_6 dims=(1,2,0)
pnnx.Attribute          op_4        0 1 val_37 @data=(%size,1)i64
GatherND                op_5        2 1 pnnx_4 val_37 pnnx_7 batch_dims=0
Tensor.permute          op_6        1 1 pnnx_7 pnnx_8 dims=(1,2,0)
aten::sub               op_7        2 1 pnnx_8 pnnx_6 pnnx_9
pnnx.Attribute          op_8        0 1 clamp_2 @data=(%size)f32
aten::mul               op_9        2 1 pnnx_9 clamp_2 pnnx_10
aten::add               op_10       2 1 pnnx_6 pnnx_10 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const int size = captured_params.at("size").i;

        auto ia = captured_attrs.at("op_1.data");
        auto ib = captured_attrs.at("op_4.data");
        auto im = captured_attrs.at("op_8.data");

        const int64_t* pa = (const int64_t*)ia.data.data();
        const int64_t* pb = (const int64_t*)ib.data.data();
        const float* pm = (const float*)im.data.data();

        const int w = matched_operators.at("op_0")->inputs[0]->shape[2];

        align_corners = resolve_align_corners(w, size, pa, pb, pm);

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int size = captured_params.at("size").i;

        op->params["size"] = {size};

        op->params["mode"] = "linear";
        op->params["align_corners"] = align_corners;
    }

protected:
    mutable bool align_corners;
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_onnx_1d_linear, 110)

static bool resolve_nearest_exact_1d(int w, int outw, const int64_t* pindex)
{
    double scale = (double)w / outw;
    for (int i = 0; i < outw; i++)
    {
        float fx = (float)((i + 0.5f) * scale);
        int sx = (int)floor(fx);

        if (pindex[i] != sx)
            return false;
    }

    return true;
}

class F_interpolate_onnx_1d_nearest_exact : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
Tensor.permute          op_0        1 1 input pnnx_4 dims=(2,0,1)
pnnx.Attribute          op_1        0 1 index @data=(%size,1)i64
GatherND                op_2        2 1 pnnx_4 index pnnx_5 batch_dims=0
Tensor.permute          op_3        1 1 pnnx_5 out dims=(1,2,0)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const int size = captured_params.at("size").i;

        auto index = captured_attrs.at("op_1.data");

        const int64_t* pindex = (const int64_t*)index.data.data();

        const int w = matched_operators.at("op_0")->inputs[0]->shape[2];

        bool nearest_exact = resolve_nearest_exact_1d(w, size, pindex);

        return nearest_exact;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int size = captured_params.at("size").i;
        op->params["size"] = {size};
        op->params["mode"] = "nearest-exact";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_onnx_1d_nearest_exact, 111)

static bool resolve_nearest_exact_2d(int w, int h, int outw, int outh, const int64_t* pindex)
{
    double scale_w = (double)w / outw;
    double scale_h = (double)h / outh;
    for (int i = 0; i < outh; i++)
    {
        float fy = (float)((i + 0.5f) * scale_h);
        int sy = (int)floor(fy);

        for (int j = 0; j < outw; j++)
        {
            float fx = (float)((j + 0.5f) * scale_w);
            int sx = (int)floor(fx);

            int py = pindex[0];
            int px = pindex[1];
            pindex += 2;

            if (px != sx || py != sy)
                return false;
        }
    }

    return true;
}

class F_interpolate_onnx_2d_nearest_exact : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
Tensor.permute          op_0        1 1 input pnnx_48 dims=(2,3,0,1)
pnnx.Attribute          op_1        0 1 index @data=(%size_h,%size_w,2)i64
GatherND                op_2        2 1 pnnx_48 index pnnx_49 batch_dims=0
Tensor.permute          op_3        1 1 pnnx_49 out dims=(2,3,0,1)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const int size_h = captured_params.at("size_h").i;
        const int size_w = captured_params.at("size_w").i;

        auto index = captured_attrs.at("op_1.data");

        const int64_t* pindex = (const int64_t*)index.data.data();

        const int h = matched_operators.at("op_0")->inputs[0]->shape[2];
        const int w = matched_operators.at("op_0")->inputs[0]->shape[3];

        bool nearest_exact = resolve_nearest_exact_2d(w, h, size_w, size_h, pindex);

        return nearest_exact;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int size_h = captured_params.at("size_h").i;
        const int size_w = captured_params.at("size_w").i;
        op->params["size"] = {size_h, size_w};
        op->params["mode"] = "nearest-exact";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_onnx_2d_nearest_exact, 111)

static bool resolve_nearest_exact_3d(int w, int h, int d, int outw, int outh, int outd, const int64_t* pindex)
{
    double scale_w = (double)w / outw;
    double scale_h = (double)h / outh;
    double scale_d = (double)d / outd;
    for (int i = 0; i < outd; i++)
    {
        float fz = (float)((i + 0.5f) * scale_d);
        int sz = (int)floor(fz);

        for (int j = 0; j < outh; j++)
        {
            float fy = (float)((j + 0.5f) * scale_h);
            int sy = (int)floor(fy);

            for (int k = 0; k < outw; k++)
            {
                float fx = (float)((k + 0.5f) * scale_w);
                int sx = (int)floor(fx);

                int pz = pindex[0];
                int py = pindex[1];
                int px = pindex[2];
                pindex += 3;

                if (px != sx || py != sy || pz != sz)
                    return false;
            }
        }
    }

    return true;
}

class F_interpolate_onnx_3d_nearest_exact : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input       0 1 input
Tensor.permute          op_0        1 1 input pnnx_48 dims=(2,3,4,0,1)
pnnx.Attribute          op_1        0 1 index @data=(%size_d,%size_h,%size_w,3)i64
GatherND                op_2        2 1 pnnx_48 index pnnx_49 batch_dims=0
Tensor.permute          op_3        1 1 pnnx_49 out dims=(3,4,0,1,2)
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.interpolate";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const int size_d = captured_params.at("size_d").i;
        const int size_h = captured_params.at("size_h").i;
        const int size_w = captured_params.at("size_w").i;

        auto index = captured_attrs.at("op_1.data");

        const int64_t* pindex = (const int64_t*)index.data.data();

        const int d = matched_operators.at("op_0")->inputs[0]->shape[2];
        const int h = matched_operators.at("op_0")->inputs[0]->shape[3];
        const int w = matched_operators.at("op_0")->inputs[0]->shape[4];

        bool nearest_exact = resolve_nearest_exact_3d(w, h, d, size_w, size_h, size_d, pindex);

        return nearest_exact;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int size_d = captured_params.at("size_d").i;
        const int size_h = captured_params.at("size_h").i;
        const int size_w = captured_params.at("size_w").i;
        op->params["size"] = {size_d, size_h, size_w};
        op->params["mode"] = "nearest-exact";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_interpolate_onnx_3d_nearest_exact, 111)

} // namespace pnnx
