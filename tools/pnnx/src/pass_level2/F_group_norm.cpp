// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class F_group_norm : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input_1     0 1 input
pnnx.Input              input_2     0 1 weight
pnnx.Input              input_3     0 1 bias
prim::Constant          op_0        0 1 num_groups value=%num_groups
prim::Constant          op_1        0 1 eps value=%eps
prim::Constant          op_2        0 1 cudnn_enabled value=*
aten::group_norm        op_3        6 1 input num_groups weight bias eps cudnn_enabled out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.group_norm";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_group_norm, 130)

class F_group_norm_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input_0     0 1 input
Tensor.reshape          op_0        1 1 input r1 shape=(0,%num_groups,-1)
pnnx.Attribute          op_1        0 1 ones @data
pnnx.Attribute          op_2        0 1 zeros @data
InstanceNormalization   op_3        3 1 r1 ones zeros in epsilon=%epsilon
Tensor.reshape          op_4        1 1 in out shape=%shape
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.group_norm";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        const Operator* op_reshape = matched_operators.at("op_0");
        const std::vector<int>& inputshape = op_reshape->inputs[0]->shape;
        if (inputshape != captured_params.at("shape").ai)
            return false;

        const int num_groups = captured_params.at("num_groups").i;

        const Attribute& ones = captured_attrs.at("op_1.data");
        const Attribute& zeros = captured_attrs.at("op_2.data");

        if (ones.shape.size() != 1 || ones.shape[0] != num_groups)
            return false;
        if (zeros.shape.size() != 1 || zeros.shape[0] != num_groups)
            return false;

        for (auto x : ones.get_float32_data())
        {
            if (x != 1.f)
                return false;
        }

        for (auto x : zeros.get_float32_data())
        {
            if (x != 0.f)
                return false;
        }

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        op->params["num_groups"] = captured_params.at("num_groups");
        op->params["eps"] = captured_params.at("epsilon");

        op->params["weight"] = Parameter();
        op->params["bias"] = Parameter();
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_group_norm_onnx, 130)

class F_group_norm_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
7 6
pnnx.Input              input       0 1 input
pnnx.Attribute          weight      0 1 weight @data
pnnx.Attribute          bias        0 1 bias @data
F.group_norm            op_2        1 1 input a num_groups=%num_groups eps=%eps weight=None bias=None
aten::mul               op_3        2 1 a weight b
aten::add               op_4        2 1 b bias out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
pnnx.Attribute          weight      0 1 weight @data=%weight.data
pnnx.Attribute          bias        0 1 bias @data=%bias.data
F.group_norm            gn_affine   3 1 input weight bias out num_groups=%num_groups eps=%eps
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const Operator* op_group_norm = matched_operators.at("op_2");
        const Operator* op_mul = matched_operators.at("op_3");
        const Operator* op_add = matched_operators.at("op_4");
        const std::vector<int>& inputshape = op_group_norm->inputs[0]->shape;
        const std::vector<int>& weight_shape = op_mul->inputs[1]->shape;
        const std::vector<int>& bias_shape = op_add->inputs[1]->shape;

        if (weight_shape.size() != bias_shape.size())
            return false;
        if (weight_shape.size() + 1 != inputshape.size())
            return false;
        if (weight_shape[0] != inputshape[1])
            return false;
        for (size_t i = 1; i < weight_shape.size(); i++)
        {
            if (weight_shape[i] != 1 || bias_shape[i] != 1)
                return false;
        }

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        // fix weight bias shape
        // (N,1,1) -> (N)
        ops.at("weight")->attrs["data"].shape.resize(1);
        ops.at("bias")->attrs["data"].shape.resize(1);
        ops.at("weight")->outputs[0]->shape.resize(1);
        ops.at("bias")->outputs[0]->shape.resize(1);

        Operator* gn = ops.at("gn_affine");
        gn->inputnames.resize(3);
        gn->inputnames[0] = "input";
        gn->inputnames[1] = "weight";
        gn->inputnames[2] = "bias";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_group_norm_onnx_1, 131)

} // namespace pnnx
