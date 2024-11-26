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
6 5
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 pad value=%pad
prim::Constant          op_1        0 1 value value=%value
prim::Constant          op_2        0 1 mode value=%mode
aten::pad               op_3        4 1 input pad mode value out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pad";
    }
};

class F_pad_dynamic : public GraphRewriterPass
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

class F_pad_dynamic_2 : public GraphRewriterPass
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_dynamic, 111)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_dynamic_2, 112)

class F_pad_constant : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 pad value=%pad
prim::Constant          op_1        0 1 value value=%value
aten::constant_pad_nd   op_2        3 1 input pad value out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pad";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["pad"] = captured_params.at("pad");
        op->params["value"] = captured_params.at("value");
        op->params["mode"] = "constant";
    }
};

class F_pad_constant_dynamic : public GraphRewriterPass
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

class F_pad_constant_dynamic_1 : public GraphRewriterPass
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

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_constant, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_constant_dynamic, 111)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_constant_dynamic_1, 111)

class F_pad_reflect : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 pad value=%pad
aten::reflection_pad1d  op_1        2 1 input pad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pad";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["pad"] = captured_params.at("pad");
        op->params["mode"] = "reflect";
        op->params["value"] = Parameter();
    }
};

class F_pad_reflect_2 : public F_pad_reflect
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 pad value=%pad
aten::reflection_pad2d  op_1        2 1 input pad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class F_pad_reflect_dynamic : public GraphRewriterPass
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
        op->params["value"] = Parameter();
    }
};

class F_pad_reflect_dynamic_2 : public F_pad_reflect_dynamic
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
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_reflect, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_reflect_2, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_reflect_dynamic, 111)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_reflect_dynamic_2, 111)

class F_pad_replicate : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 pad value=%pad
aten::replication_pad1d op_1        2 1 input pad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.pad";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["pad"] = captured_params.at("pad");
        op->params["mode"] = "replicate";
        op->params["value"] = Parameter();
    }
};

class F_pad_replicate_2 : public F_pad_replicate
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 pad value=%pad
aten::replication_pad2d op_1        2 1 input pad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class F_pad_replicate_3 : public F_pad_replicate
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 pad value=%pad
aten::replication_pad3d op_1        2 1 input pad out
pnnx.Output             output      1 0 out
)PNNXIR";
    }
};

class F_pad_replicate_dynamic : public GraphRewriterPass
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
        op->params["value"] = Parameter();
    }
};

class F_pad_replicate_dynamic_2 : public F_pad_replicate_dynamic
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
};

class F_pad_replicate_dynamic_3 : public F_pad_replicate_dynamic
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
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_replicate, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_replicate_2, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_replicate_3, 110)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_replicate_dynamic, 111)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_replicate_dynamic_2, 111)
REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_replicate_dynamic_3, 111)

class F_pad_onnx : public GraphRewriterPass
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
        {
            if (pads[0] == 0 && pads[2] == 0)
                op->params["pad"] = std::vector<int>{pads[1], pads[3]};
            else
                op->params["pad"] = std::vector<int>{pads[1], pads[3], pads[0], pads[2]};
        }

        if (pads.size() == 6)
        {
            if (pads[1] == 0 && pads[4] == 0 && pads[0] == 0 && pads[3] == 0)
                op->params["pad"] = std::vector<int>{pads[2], pads[5]};
            else if (pads[0] == 0 && pads[3] == 0)
                op->params["pad"] = std::vector<int>{pads[2], pads[5], pads[1], pads[4]};
            else
                op->params["pad"] = std::vector<int>{pads[2], pads[5], pads[1], pads[4], pads[0], pads[3]};
        }

        if (pads.size() == 8)
        {
            if (pads[1] == 0 && pads[5] == 0 && pads[0] == 0 && pads[4] == 0)
                op->params["pad"] = std::vector<int>{pads[3], pads[7], pads[2], pads[6]};
            else if (pads[0] == 0 && pads[4] == 0)
                op->params["pad"] = std::vector<int>{pads[3], pads[7], pads[2], pads[6], pads[1], pads[5]};
            else
                op->params["pad"] = std::vector<int>{pads[3], pads[7], pads[2], pads[6], pads[1], pads[5], pads[0], pads[4]};
        }

        if (pads.size() == 10)
        {
            if (pads[1] == 0 && pads[6] == 0 && pads[0] == 0 && pads[5] == 0)
                op->params["pad"] = std::vector<int>{pads[4], pads[9], pads[3], pads[8], pads[2], pads[7]};
            else if (pads[0] == 0 && pads[5] == 0)
                op->params["pad"] = std::vector<int>{pads[4], pads[9], pads[3], pads[8], pads[2], pads[7], pads[1], pads[6]};
            else
                op->params["pad"] = std::vector<int>{pads[4], pads[9], pads[3], pads[8], pads[2], pads[7], pads[1], pads[6], pads[0], pads[5]};
        }

        op->params["value"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_onnx, 110)

class F_pad_onnx_1 : public F_pad_onnx
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
Pad                     op_0        1 1 input out mode=%mode pads=%pads value=%value
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        F_pad_onnx::write(op, captured_params);

        op->params["value"] = captured_params.at("value");
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_pad_onnx_1, 110)

} // namespace pnnx
