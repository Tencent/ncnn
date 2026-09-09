// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

// dtype constants serialize as the torch scalar-type enum; map it to the
// torch.<dtype> literal the generated python needs (None stays unset so the
// op keeps its natural accumulation dtype)
static void set_mean_dtype_param(Operator* op, const std::map<std::string, Parameter>& captured_params)
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
    case 0:
        dtype_str = "torch.uint8";
        break;
    case 1:
        dtype_str = "torch.int8";
        break;
    case 2:
        dtype_str = "torch.short";
        break;
    case 3:
        dtype_str = "torch.int";
        break;
    case 4:
        dtype_str = "torch.long";
        break;
    case 5:
        dtype_str = "torch.half";
        break;
    case 6:
        dtype_str = "torch.float";
        break;
    case 7:
        dtype_str = "torch.double";
        break;
    case 8:
        dtype_str = "torch.complex32";
        break;
    case 9:
        dtype_str = "torch.complex64";
        break;
    case 10:
        dtype_str = "torch.complex128";
        break;
    case 11:
        dtype_str = "torch.bool";
        break;
    case 15:
        dtype_str = "torch.bfloat16";
        break;
    default:
        break;
    }
    if (dtype_str)
        op->params["dtype"] = dtype_str;
}

class torch_mean : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
6 5
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
prim::Constant          op_0        0 1 keepdim value=%keepdim
prim::Constant          op_1        0 1 dtype value=%dtype
aten::mean              op_2        4 1 input dim keepdim dtype out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.mean";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        // copy the captured defaults (keepdim etc.) like the default rewriter,
        // then restore the explicit dtype so generated code does not silently
        // average in the input's default dtype (dtype=torch.float64 changes the
        // accumulation/output type and can alter overflow behavior)
        for (std::map<std::string, Parameter>::const_iterator x = captured_params.begin(); x != captured_params.end(); ++x)
        {
            op->params[x->first] = x->second;
        }
        set_mean_dtype_param(op, captured_params);
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_mean, 50)

class torch_mean_01 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dim
aten::mean_dim          op_0        2 1 input dim out keepdim=%keepdim
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.mean";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        bool keepdim;
        if (captured_params.at("keepdim").type == 2)
        {
            keepdim = captured_params.at("keepdim").i ? true : false;
        }
        else // if (captured_params.at("keepdim").type == 1)
        {
            keepdim = captured_params.at("keepdim").b;
        }

        op->params["keepdim"] = keepdim;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_mean_01, 50)

class torch_mean_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
prim::Constant          op_0        0 1 dtype value=%dtype
aten::mean              op_1        2 1 input dtype out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.mean";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        set_mean_dtype_param(op, captured_params);
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_mean_1, 50)

class torch_mean_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
ReduceMean              op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.mean";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.find("op_0.axes") != captured_params.end())
        {
            op->params["dim"] = captured_params.at("op_0.axes");
        }
        else
        {
            // reduce all
            const int input_rank = (int)op->inputs[0]->shape.size();
            std::vector<int> dim(input_rank);
            for (int i = 0; i < input_rank; i++)
            {
                dim[i] = i;
            }
            op->params["dim"] = dim;
        }

        if (captured_params.find("op_0.keepdims") != captured_params.end())
        {
            op->params["keepdim"] = captured_params.at("op_0.keepdims").i ? true : false;
        }
        else
        {
            op->params["keepdim"] = true;
        }
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_mean_onnx, 50)

class torch_mean_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
tnn.ReduceMean          op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "torch.mean";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        std::vector<int> dim;
        for (int i = 1;; i++)
        {
            if (captured_params.find("op_0.arg" + std::to_string(i)) == captured_params.end())
                break;

            dim.push_back(captured_params.at("op_0.arg" + std::to_string(i)).i);
        }

        op->params["dim"] = dim;
        op->params["keepdim"] = captured_params.at("op_0.arg0").i ? true : false;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(torch_mean_tnn, 50)

} // namespace pnnx
