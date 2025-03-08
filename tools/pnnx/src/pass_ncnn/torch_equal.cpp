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

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

// 集成≥0，>1，≤2，<3，==4，!=5
class torch_ge_0 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input               input             0 1 input
pnnx.Attribute           op_other          0 1 other @data
torch.ge                 op_0              2 1 input other out
pnnx.Output              output            1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Equal";
    }

    const char* name_str() const
    {
        return "equal";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = 0; // 区分判断类型

        Attribute other = captured_attrs.at("op_other.data");
        op->attrs["0"] = other;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_ge_0, 20)

class torch_ge_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input               input             0 1 input
pnnx.Input               other             0 1 other
torch.ge                 op_0              2 1 input other out
pnnx.Output              output            1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Equal";
    }

    const char* name_str() const
    {
        return "equal";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = 0;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_ge_1, 20)
class torch_gt_0 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input               input             0 1 input
pnnx.Attribute           op_other          0 1 other @data
torch.gt                 op_0              2 1 input other out
pnnx.Output              output            1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Equal";
    }

    const char* name_str() const
    {
        return "equal";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = 1; // 区分判断类型

        Attribute other = captured_attrs.at("op_other.data");
        op->attrs["0"] = other;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_gt_0, 20)

class torch_gt_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input               input             0 1 input
pnnx.Input               other             0 1 other
torch.gt                 op_0              2 1 input other out
pnnx.Output              output            1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Equal";
    }

    const char* name_str() const
    {
        return "equal";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = 1;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_gt_1, 20)
class torch_le_0 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input               input             0 1 input
pnnx.Attribute           op_other          0 1 other @data
torch.le                 op_0              2 1 input other out
pnnx.Output              output            1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Equal";
    }

    const char* name_str() const
    {
        return "equal";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = 2; // 区分判断类型

        Attribute other = captured_attrs.at("op_other.data");
        op->attrs["0"] = other;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_le_0, 20)
class torch_le_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input               input             0 1 input
pnnx.Input               other             0 1 other
torch.le                 op_0              2 1 input other out
pnnx.Output              output            1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Equal";
    }

    const char* name_str() const
    {
        return "equal";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = 2;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_le_1, 20)
// ---
class torch_lt_0 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input               input             0 1 input
pnnx.Attribute           op_other          0 1 other @data
torch.eq                 op_0              2 1 input other out
pnnx.Output              output            1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Equal";
    }

    const char* name_str() const
    {
        return "equal";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = 3; // 区分判断类型

        Attribute other = captured_attrs.at("op_other.data");
        op->attrs["0"] = other;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_lt_0, 20)

class torch_lt_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input               input             0 1 input
pnnx.Input               other             0 1 other
torch.lt                 op_0              2 1 input other out
pnnx.Output              output            1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Equal";
    }

    const char* name_str() const
    {
        return "equal";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = 3;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_lt_1, 20)
class torch_eq_0 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input               input             0 1 input
pnnx.Attribute           op_other          0 1 other @data
torch.eq                 op_0              2 1 input other out
pnnx.Output              output            1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Equal";
    }

    const char* name_str() const
    {
        return "equal";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = 4; // 区分判断类型

        Attribute other = captured_attrs.at("op_other.data");
        op->attrs["0"] = other;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_eq_0, 20)

class torch_eq_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input               input             0 1 input
pnnx.Input               other             0 1 other
torch.eq                 op_0              2 1 input other out
pnnx.Output              output            1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Equal";
    }

    const char* name_str() const
    {
        return "equal";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = 4;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_eq_1, 20)
class torch_ne_0 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input               input             0 1 input
pnnx.Attribute           op_other          0 1 other @data
torch.ne                 op_0              2 1 input other out
pnnx.Output              output            1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Equal";
    }

    const char* name_str() const
    {
        return "equal";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = 5; // 区分判断类型

        Attribute other = captured_attrs.at("op_other.data");
        op->attrs["0"] = other;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_ne_0, 20)

class torch_ne_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input               input             0 1 input
pnnx.Input               other             0 1 other
torch.ne                 op_0              2 1 input other out
pnnx.Output              output            1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Equal";
    }

    const char* name_str() const
    {
        return "equal";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = 5;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_ne_1, 20)

} // namespace ncnn

} // namespace pnnx
