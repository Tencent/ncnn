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

class F_pad : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 pad
pnnx.Input              input_2     0 1 value
aten::constant_pad_nd   op_0        3 1 input pad value out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pad";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["mode"] = "constant";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad, 10)

class F_pad_01 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 pad
aten::constant_pad_nd   op_0        2 1 input pad out value=%value
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pad";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["mode"] = "constant";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_01, 10)

class F_pad_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 pad
aten::reflection_pad1d  op_0        2 1 input pad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pad";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["mode"] = "reflect";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_1, 10)

class F_pad_2 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 pad
aten::replication_pad1d op_0        2 1 input pad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pad";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["mode"] = "replicate";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_2, 10)

class F_pad_3 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 pad
aten::reflection_pad2d  op_0        2 1 input pad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pad";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["mode"] = "reflect";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_3, 10)

class F_pad_4 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 pad
aten::replication_pad2d op_0        2 1 input pad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pad";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["mode"] = "replicate";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_4, 10)

class F_pad_6 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 pad
aten::replication_pad3d op_0        2 1 input pad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pad";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["mode"] = "replicate";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_6, 10)

class F_pad_7 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 pad
pnnx.Input              input_2     0 1 value
prim::Constant          op_0        0 1 mode value=constant
aten::pad               op_1        4 1 input pad mode value out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pad";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        op->params["mode"] = "constant";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_7, 10)

class F_pad_8 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 pad
pnnx.Input              input_2     0 1 mode
prim::Constant          op_0        0 1 value value=*
aten::pad               op_1        4 1 input pad mode value out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pad";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_8, 10)

class F_pad_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Expression         op_0        0 1 pads expr=[0,%pad_front,%pad_top,%pad_left,0,%pad_back,%pad_bottom,%pad_right]
Pad                     op_1        2 1 input pads out mode=%mode
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pad";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& mode = captured_params.at("mode").s;
        if (mode == "constant") op->params["mode"] = "constant";
        if (mode == "reflect") op->params["mode"] = "reflect";
        if (mode == "edge") op->params["mode"] = "replicate";
        if (mode == "wrap") op->params["mode"] = "circular";

        const int pad_front = captured_params.at("pad_front").i;
        const int pad_top = captured_params.at("pad_top").i;
        const int pad_left = captured_params.at("pad_left").i;
        const int pad_back = captured_params.at("pad_back").i;
        const int pad_bottom = captured_params.at("pad_bottom").i;
        const int pad_right = captured_params.at("pad_right").i;
        op->params["pad"] = std::vector<int>{pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back};
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_onnx, 10)

class F_pad_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Expression         op_0        0 1 pads expr=[0,%pad_top,%pad_left,0,%pad_bottom,%pad_right]
Pad                     op_1        2 1 input pads out mode=%mode
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pad";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& mode = captured_params.at("mode").s;
        if (mode == "constant") op->params["mode"] = "constant";
        if (mode == "reflect") op->params["mode"] = "reflect";
        if (mode == "edge") op->params["mode"] = "replicate";
        if (mode == "wrap") op->params["mode"] = "circular";

        const int pad_top = captured_params.at("pad_top").i;
        const int pad_left = captured_params.at("pad_left").i;
        const int pad_bottom = captured_params.at("pad_bottom").i;
        const int pad_right = captured_params.at("pad_right").i;
        op->params["pad"] = std::vector<int>{pad_left, pad_right, pad_top, pad_bottom};
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_onnx_1, 10)

class F_pad_onnx_2 : public F_pad_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
pnnx.Expression         op_0        0 1 pads expr=[0,0,%pad_front,%pad_top,%pad_left,0,0,%pad_back,%pad_bottom,%pad_right]
Pad                     op_1        2 1 input pads out mode=%mode
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_onnx_2, 10)

class F_pad_onnx_0 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Pad                     op_0        1 1 input out mode=%mode pads=%pads
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pad";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("pads").type != 5)
            return false;

        const std::vector<int>& pads = captured_params.at("pads").ai;
        if (pads.size() == 2 && pads[0] == pads[1])
            return true;

        if (pads.size() == 4 && pads[0] == pads[2] && pads[1] == pads[3])
            return true;

        if (pads.size() == 6 && pads[0] == pads[3] && pads[1] == pads[4] && pads[2] == pads[5])
            return true;

        if (pads.size() == 8 && pads[0] == 0 && pads[4] == 0 && pads[1] == pads[5] && pads[2] == pads[6] && pads[3] == pads[7])
            return true;

        return false;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::string& mode = captured_params.at("mode").s;
        if (mode == "constant") op->params["mode"] = "constant";
        if (mode == "reflect") op->params["mode"] = "reflect";
        if (mode == "edge") op->params["mode"] = "replicate";
        if (mode == "wrap") op->params["mode"] = "circular";

        const std::vector<int>& pads = captured_params.at("pads").ai;

        if (pads.size() == 2)
            op->params["pad"] = pads;

        if (pads.size() == 4)
            op->params["pad"] = std::vector<int>{pads[1], pads[3], pads[0], pads[2]};

        if (pads.size() == 6)
            op->params["pad"] = std::vector<int>{pads[2], pads[5], pads[1], pads[4], pads[0], pads[3]};

        if (pads.size() == 8)
            op->params["pad"] = std::vector<int>{pads[3], pads[7], pads[2], pads[6], pads[1], pads[5]};
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_onnx_0, 10)

} // namespace pnnx
