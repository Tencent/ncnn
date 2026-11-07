// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_level2.h"

namespace pnnx {

class Tensor_permute : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 dims
aten::permute           op_0        2 1 input dims out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.permute";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_permute, 60)

class Tensor_permute_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
aten::permute           op_0        1 1 input out dims=%dims
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.permute";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_permute_1, 60)

class Tensor_permute_onnx : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input_0     0 1 input
Transpose               op_0        1 1 input out perm=%dims
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.permute";
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_permute_onnx, 60)

class Tensor_permute_tnn : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
tnn.Permute             op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Tensor.permute";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int dims_count = captured_params.at("op_0.arg0").i;
        std::vector<int> dims(dims_count);
        for (int i = 0; i < dims_count; i++)
        {
            dims[i] = captured_params.at("op_0.arg" + std::to_string(i + 1)).i;
        }
        op->params["dims"] = dims;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_permute_tnn, 60)

class Tensor_permute_caffe : public Tensor_permute
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
caffe.Permute           op_0        1 1 input out %*=%*
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        auto it = captured_params.find("op_0.order");
        std::vector<int> order = it == captured_params.end() ? std::vector<int>() : it->second.ai;
        const int rank = op->inputs[0]->shape.size();
        std::vector<bool> used(rank, false);
        for (int dim : order)
            used[dim] = true;
        for (int i = 0; i < rank; i++)
        {
            if (!used[i])
                order.push_back(i);
        }
        op->params["dims"] = order;
    }
};

REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(Tensor_permute_caffe, 60)

} // namespace pnnx
