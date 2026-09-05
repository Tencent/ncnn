// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

static int input_ncnn_batch_axis(const Operator* op)
{
    if (op->inputs.empty())
        return 233;

    if (op->inputs[0]->params.find("__ncnn_batch_axis") == op->inputs[0]->params.end())
        return 233;

    return op->inputs[0]->params.at("__ncnn_batch_axis").i;
}

// rank-3 nn.Linear with a non-singleton middle dim cannot use the Gemm N1M
// layout (ncnn Gemm only uses c as M and drops the h dim for 3D input), so
// flatten the leading dims into 2D, run a 2D Gemm with the constant weight,
// and reshape back to the original 3D shape.
void convert_nn_Linear_3d_flatten(Graph& graph)
{
    for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
    {
        Operator* op = graph.ops[i];

        if (op->type != "nn.Linear")
            continue;
        if (op->inputs.size() != 1)
            continue;

        Operand* in = op->inputs[0];
        const std::vector<int>& in_shape = in->shape;
        const int rank = (int)in_shape.size();
        if (rank != 3)
            continue;

        // singleton middle dim stays on the N1M Gemm path (nn_Linear_3D_0)
        if (in_shape.size() >= 2 && in_shape[1] == 1)
            continue;

        // only the no-explicit-batch 3D layout is handled here (like nn_Linear_3D_0)
        int batch_axis = 233;
        if (in->params.find("__ncnn_batch_axis") != in->params.end())
            batch_axis = in->params.at("__ncnn_batch_axis").i;
        if (batch_axis != 233)
            continue;

        // weight is stored as an attribute (constantB); bias optional
        if (op->attrs.find("weight") == op->attrs.end())
            continue;
        const bool has_bias = op->attrs.find("bias") != op->attrs.end();
        const int in_features = op->params.at("in_features").i;
        const int out_features = op->params.at("out_features").i;

        Attribute weight = op->attrs.at("weight");
        Attribute bias;
        if (has_bias)
            bias = op->attrs.at("bias");

        // flatten leading dims into 2D (h,F) -> 2D Gemm -> reshape back to 3D
        Operand* fl_in = in;
        Operand* fl_out = op->outputs[0];

        int reshape_h = 1;
        for (int j = 0; j < rank - 1; j++)
        {
            if (in_shape[j] == -1)
            {
                reshape_h = -1;
                break;
            }
            reshape_h *= in_shape[j];
        }

        Operator* reshape0 = graph.new_operator_before("Tensor.reshape", op->name + "_ncnnreshape0", op);
        Operator* reshape1 = graph.new_operator_after("Tensor.reshape", op->name + "_ncnnreshape1", op);
        Operand* reshape0_out = graph.new_operand(op->name + "_ncnnreshape0_out");
        Operand* reshape1_in = graph.new_operand(op->name + "_ncnnreshape1_in");

        reshape0->inputs.push_back(fl_in);
        reshape0->outputs.push_back(reshape0_out);
        reshape1->inputs.push_back(reshape1_in);
        reshape1->outputs.push_back(fl_out);

        for (size_t j = 0; j < fl_in->consumers.size(); j++)
        {
            if (fl_in->consumers[j] == op)
            {
                fl_in->consumers[j] = reshape0;
                break;
            }
        }
        fl_out->producer = reshape1;
        op->inputs[0] = reshape0_out;
        op->outputs[0] = reshape1_in;

        reshape0_out->producer = reshape0;
        reshape0_out->consumers.push_back(op);
        reshape1_in->producer = op;
        reshape1_in->consumers.push_back(reshape1);

        // 2D has no explicit batch, avoid Gemm emitting 3D (M,1,N)
        reshape0_out->params["__ncnn_batch_axis"] = 233;
        reshape1_in->params["__ncnn_batch_axis"] = 233;

        std::vector<int> reshape0_out_shape = {reshape_h, in_shape[rank - 1]};
        std::vector<int> reshape1_in_shape = {reshape_h, -1};
        if (fl_out->shape.size() >= (size_t)rank)
            reshape1_in_shape = {reshape_h, fl_out->shape[rank - 1]};
        std::vector<int> reshape1_out_shape = fl_out->shape;

        reshape0->params["shape"] = reshape0_out_shape;
        reshape1->params["shape"] = reshape1_out_shape;
        reshape0_out->type = fl_in->type;
        reshape0_out->shape = reshape0_out_shape;
        reshape1_in->type = fl_out->type;
        reshape1_in->shape = reshape1_in_shape;

        // turn op into a 2D Gemm with constant weight/bias
        op->type = "Gemm";
        op->name = std::string("gemm_") + op->name;

        op->params.clear();
        op->params["0"] = 1.f;                // alpha
        op->params["1"] = 1.f;                // beta
        op->params["2"] = 0;                  // transA
        op->params["3"] = 1;                  // transB (weight (out,in) -> (in,out))
        op->params["4"] = 0;                  // constantA
        op->params["5"] = 1;                  // constantB (weight attr)
        op->params["6"] = has_bias ? 1 : 0;   // constantC
        op->params["7"] = 0;                  // constantM (M inferred from A)
        op->params["8"] = out_features;       // N
        op->params["9"] = in_features;        // K
        op->params["10"] = has_bias ? 4 : -1; // C broadcast N
        op->params["11"] = 0;                 // output_N1M
        op->params["12"] = 1;                 // output_elempack

        op->attrs.clear();
        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = weight;
        if (has_bias)
        {
            op->attrs["2"] = Attribute();
            op->attrs["2"].data = {0, 0, 0, 0};
            op->attrs["3"] = bias;
        }
    }
}

class nn_Linear_3D_0 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input #input=(%m,%n,%in_features)f32
nn.Linear               op_0        1 1 input out in_features=%in_features out_features=%out_features bias=%bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Gemm";
    }

    const char* name_str() const
    {
        return "gemm";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        // 3D input (S,B,F) without an explicit batch (__ncnn_batch_axis == 233).
        // The Gemm N1M layout is only valid for a singleton middle (h) dim;
        // rank-3 inputs with a larger middle dim are flattened by
        // convert_nn_Linear_3d_flatten before this pass runs.
        if (input_ncnn_batch_axis(matched_operators.at("op_0")) != 233)
            return false;
        return captured_params.at("n").i == 1;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        // Gemm: A=input 3D (w=F,h=B,c=S), B=weight constantB, C=bias constantC
        // output_N1M=1 -> output 3D (M,1,N)=(S,1,out)
        op->params["2"] = 0; // transA
        op->params["3"] = 1; // transB
        op->params["4"] = 0; // constantA
        op->params["5"] = 1; // constantB
        op->params["6"] = 1; // constantC
        op->params["7"] = 0; // constantM
        op->params["8"] = captured_params.at("out_features");
        op->params["9"] = captured_params.at("in_features");
        op->params["10"] = captured_params.at("bias").b ? 4 : -1; // C broadcast N
        op->params["11"] = 1;                                     // output_N1M

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = captured_attrs.at("op_0.weight");
        if (captured_params.at("bias").b)
        {
            op->attrs["2"] = Attribute();
            op->attrs["2"].data = {0, 0, 0, 0};
            op->attrs["3"] = captured_attrs.at("op_0.bias");
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_Linear_3D_0, 18)

class nn_Linear_0 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input #input=(1,%m,%in_features)f32
nn.Linear               op_0        1 1 input out in_features=%in_features out_features=%out_features bias=%bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Gemm";
    }

    const char* name_str() const
    {
        return "gemm";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const int ncnn_batch_axis = input_ncnn_batch_axis(matched_operators.at("op_0"));
        return ncnn_batch_axis == 233 || ncnn_batch_axis == 0;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        int m = captured_params.at("m").i;
        if (m == -1)
            m = 0;

        op->params["2"] = 0;
        op->params["3"] = 1;
        op->params["4"] = 0;
        op->params["5"] = 1;
        op->params["6"] = 1;
        op->params["7"] = m;
        op->params["8"] = captured_params.at("out_features");
        op->params["9"] = captured_params.at("in_features");
        op->params["10"] = captured_params.at("bias").b ? 4 : -1;

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = captured_attrs.at("op_0.weight");
        if (captured_params.at("bias").b)
        {
            op->attrs["2"] = Attribute();
            op->attrs["2"].data = {0, 0, 0, 0};
            op->attrs["3"] = captured_attrs.at("op_0.bias");
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_Linear_0, 19)

class nn_Linear_01 : public nn_Linear_0
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input #input=(%m,%in_features)f32
nn.Linear               op_0        1 1 input out in_features=%in_features out_features=%out_features bias=%bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int m = captured_params.at("m").i;

        if (m == 1)
            return false;

        return true;
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        if (input_ncnn_batch_axis(matched_operators.at("op_0")) != 233)
            return false;

        return match(captured_params);
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_Linear_01, 19)

class nn_Linear_10 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input #input=(1,%m,%in_features)f32
pnnx.Input              input_1     0 1 bias
nn.Linear               op_0        2 1 input bias out in_features=%in_features out_features=%out_features bias=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Gemm";
    }

    const char* name_str() const
    {
        return "gemm";
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& /*captured_params*/, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        const int ncnn_batch_axis = input_ncnn_batch_axis(matched_operators.at("op_0"));
        return ncnn_batch_axis == 233 || ncnn_batch_axis == 0;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        int m = captured_params.at("m").i;
        if (m == -1)
            m = 0;

        op->params["2"] = 0;
        op->params["3"] = 1;
        op->params["4"] = 0;
        op->params["5"] = 1;
        op->params["6"] = 0;
        op->params["7"] = m;
        op->params["8"] = captured_params.at("out_features");
        op->params["9"] = captured_params.at("in_features");
        op->params["10"] = 4;

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = captured_attrs.at("op_0.weight");
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_Linear_10, 19)

class nn_Linear_11 : public nn_Linear_10
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input #input=(%m,%in_features)f32
pnnx.Input              input_1     0 1 bias
nn.Linear               op_0        2 1 input bias out in_features=%in_features out_features=%out_features bias=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const int m = captured_params.at("m").i;

        if (m == 1)
            return false;

        return true;
    }

    bool match(const std::map<std::string, const Operator*>& matched_operators, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& /*captured_attrs*/) const
    {
        if (input_ncnn_batch_axis(matched_operators.at("op_0")) != 233)
            return false;

        return match(captured_params);
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_Linear_11, 19)

class nn_Linear : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
nn.Linear               op_0        1 1 input out in_features=%in_features out_features=%out_features bias=%bias
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "InnerProduct";
    }

    const char* name_str() const
    {
        return "linear";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        op->params["0"] = captured_params.at("out_features");
        op->params["1"] = captured_params.at("bias").b ? 1 : 0;
        op->params["2"] = captured_attrs.at("op_0.weight").elemcount();

        op->attrs["0"] = Attribute();
        op->attrs["0"].data = {0, 0, 0, 0};
        op->attrs["1"] = captured_attrs.at("op_0.weight");
        if (captured_params.at("bias").b)
            op->attrs["2"] = captured_attrs.at("op_0.bias");
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_Linear, 20)

class nn_Linear_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 bias
nn.Linear               op_0        2 1 input bias out in_features=%in_features out_features=%out_features bias=False
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 bias
InnerProduct            linear      1 1 input a
BinaryOp                bias        2 1 a bias out 0=0
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const int batch_index = ops.at("linear")->inputs[0]->params["__batch_index"].i;
        const int ncnn_batch_axis = ops.at("linear")->inputs[0]->params["__ncnn_batch_axis"].i;

        ops.at("linear")->params["0"] = captured_params.at("out_features");
        ops.at("linear")->params["1"] = 0;
        ops.at("linear")->params["2"] = captured_attrs.at("op_0.weight").elemcount();

        ops.at("linear")->attrs["0"] = Attribute();
        ops.at("linear")->attrs["0"].data = {0, 0, 0, 0};
        ops.at("linear")->attrs["1"] = captured_attrs.at("op_0.weight");

        ops.at("linear")->outputs[0]->params["__batch_index"] = batch_index;
        ops.at("linear")->outputs[0]->params["__ncnn_batch_axis"] = ncnn_batch_axis;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(nn_Linear_1, 20)

} // namespace ncnn

} // namespace pnnx
