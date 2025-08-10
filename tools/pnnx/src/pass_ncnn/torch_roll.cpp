// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class torch_roll : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.roll              op_0        1 1 input out dims=%dims shifts=%shifts
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input       0 1 input
Slice                   slice       1 2 input a b
Concat                  concat      2 1 b a out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("dims").type != 5)
            return false;

        if (captured_params.at("dims").ai.size() != 1)
            return false;

        if (captured_params.at("shifts").type != 5)
            return false;

        if (captured_params.at("shifts").ai.size() != 1)
            return false;

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const Operand* in = ops.at("slice")->inputs[0];

        const int batch_index = in->params.at("__batch_index").i;

        int axis = captured_params.at("dims").ai[0];
        if (axis == batch_index)
        {
            fprintf(stderr, "roll along batch axis %d is not supported\n", batch_index);
        }

        if (axis < 0)
        {
            int input_rank = in->shape.size();
            axis = input_rank + axis;
        }

        if (axis > batch_index)
            axis -= 1;

        ops.at("slice")->params["1"] = axis;

        ops.at("concat")->params["0"] = axis;

        const int shift = captured_params.at("shifts").ai[0];
        ops.at("slice")->params["2"] = std::vector<int>{-shift};
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_roll, 20)

class torch_roll_1 : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
torch.roll              op_0        1 1 input out dims=%dims shifts=%shifts
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* replace_pattern_graph() const
    {
        return R"PNNXIR(7767517
8 7
pnnx.Input              input       0 1 input
Slice                   slice       1 2 input a b
Slice                   slice_a     1 2 a a0 a1
Slice                   slice_b     1 2 b b0 b1
Concat                  concat_a    2 1 a1 a0 a10
Concat                  concat_b    2 1 b1 b0 b10
Concat                  concat      2 1 b10 a10 out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    bool match(const std::map<std::string, Parameter>& captured_params) const
    {
        if (captured_params.at("dims").type != 5)
            return false;

        if (captured_params.at("dims").ai.size() != 2)
            return false;

        if (captured_params.at("shifts").type != 5)
            return false;

        if (captured_params.at("shifts").ai.size() != 2)
            return false;

        return true;
    }

    void write(const std::map<std::string, Operator*>& ops, const std::map<std::string, Parameter>& captured_params, const std::map<std::string, Attribute>& captured_attrs) const
    {
        GraphRewriterPass::write(ops, captured_params, captured_attrs);

        const Operand* in = ops.at("slice")->inputs[0];

        const int batch_index = in->params.at("__batch_index").i;

        int axis0 = captured_params.at("dims").ai[0];
        int axis1 = captured_params.at("dims").ai[1];
        if (axis0 == batch_index || axis1 == batch_index)
        {
            fprintf(stderr, "roll along batch axis %d is not supported\n", batch_index);
        }

        if (axis0 < 0)
        {
            int input_rank = in->shape.size();
            axis0 = input_rank + axis0;
        }

        if (axis0 > batch_index)
            axis0 -= 1;

        if (axis1 < 0)
        {
            int input_rank = in->shape.size();
            axis1 = input_rank + axis1;
        }
        if (axis1 > batch_index)
            axis1 -= 1;

        ops.at("slice")->params["1"] = axis0;
        ops.at("slice_a")->params["1"] = axis1;
        ops.at("slice_b")->params["1"] = axis1;

        ops.at("concat_a")->params["0"] = axis1;
        ops.at("concat_b")->params["0"] = axis1;
        ops.at("concat")->params["0"] = axis0;

        const int shift0 = captured_params.at("shifts").ai[0];
        const int shift1 = captured_params.at("shifts").ai[1];
        ops.at("slice")->params["2"] = std::vector<int>{-shift0};
        ops.at("slice_a")->params["2"] = std::vector<int>{-shift1};
        ops.at("slice_b")->params["2"] = std::vector<int>{-shift1};
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(torch_roll_1, 20)

} // namespace ncnn

} // namespace pnnx
