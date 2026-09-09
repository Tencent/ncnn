// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

// dtype constants serialize as the torch scalar-type enum; map it to the
// torch.<dtype> literal the generated python needs (None stays unset so the
// op keeps its natural dtype)
static void set_softmax_dtype_param(Operator* op, const std::map<std::string, Parameter>& captured_params)
{
    const std::map<std::string, Parameter>::const_iterator it = captured_params.find("dtype");
    if (it == captured_params.end())
        return;

    const Parameter& dtype = it->second;
    if (dtype.type == 0)
    {
        op->params["dtype"] = Parameter();
        return;
    }
    if (dtype.type != 2)
        return;

    const char* dtype_str = 0;
    switch (dtype.i)
    {
    case 0: dtype_str = "torch.uint8"; break;
    case 1: dtype_str = "torch.int8"; break;
    case 2: dtype_str = "torch.short"; break;
    case 3: dtype_str = "torch.int"; break;
    case 4: dtype_str = "torch.long"; break;
    case 5: dtype_str = "torch.half"; break;
    case 6: dtype_str = "torch.float"; break;
    case 7: dtype_str = "torch.double"; break;
    case 8: dtype_str = "torch.complex32"; break;
    case 9: dtype_str = "torch.complex64"; break;
    case 10: dtype_str = "torch.complex128"; break;
    case 11: dtype_str = "torch.bool"; break;
    case 15: dtype_str = "torch.bfloat16"; break;
    default: break;
    }
    if (dtype_str)
        op->params["dtype"] = dtype_str;
}

class F_softmax : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input       0 1 input
prim::Constant          op_0        0 1 dim value=%dim
prim::Constant          op_1        0 1 dtype value=%dtype
aten::softmax           op_2        3 1 input dim dtype out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softmax";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        // copy the captured defaults (dim etc.) like the default rewriter,
        // then restore the explicit dtype so generated code does not silently
        // compute softmax in the input's dtype (dtype=torch.float64 changes
        // the computation/output type)
        for (std::map<std::string, Parameter>::const_iterator x = captured_params.begin(); x != captured_params.end(); ++x)
        {
            op->params[x->first] = x->second;
        }
        set_softmax_dtype_param(op, captured_params);
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softmax, 100)

class F_softmax_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::softmax_no_dtype  op_0        1 1 input out dim=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softmax";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softmax_1, 100)

class F_softmax_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
Softmax                 op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softmax";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.axis") != captured_params.end())
        {
            op->params["dim"] = captured_params.at("op_0.axis");
        }
        else
        {
            op->params["dim"] = -1;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softmax_onnx, 101)

class F_softmax_onnx_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 4
pnnx.Input              input_0     0 1 input
Tensor.permute          op_0        1 1 input a dims=%dims
Softmax                 op_1        1 1 a b axis=%axis
Tensor.permute          op_2        1 1 b out dims=%dims
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softmax";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& dims = captured_params.at("dims").ai;
        const int axis = captured_params.at("axis").i;

        if (axis >= (int)dims.size())
            return false;

        int excount = 0;
        for (int i = 0; i < (int)dims.size(); i++)
        {
            if (dims[i] != i)
                excount++;
        }

        if (excount != 2)
            return false;

        return true;
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const std::vector<int>& dims = captured_params.at("dims").ai;
        const int axis = captured_params.at("axis").i;

        op->params["dim"] = dims[axis];
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softmax_onnx_1, 100)

class F_softmax_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
tnn.SoftmaxCaffe        op_0        1 1 input out arg0=%dim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "F.softmax";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(F_softmax_tnn, 100)

} // namespace pnnx
