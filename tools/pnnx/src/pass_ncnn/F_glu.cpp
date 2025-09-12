// Copyright 2022 Tencent
// Copyright 2022 Xiaomi Corp.   (author: Fangjun Kuang)
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class F_glu : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input           input          0 1 input
F.glu                op_0           1 1 input out dim=%dim
pnnx.Output          output         1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "GLU";
    }

    const char* name_str() const
    {
        return "glu";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        const int batch_index = op->inputs[0]->params["__batch_index"].i;

        int axis = captured_params.at("dim").i;
        if (axis == batch_index)
        {
            fprintf(stderr, "glu along batch axis %d is not supported\n", batch_index);
            return;
        }

        if (axis < 0)
        {
            int input_rank = op->inputs[0]->shape.size();
            axis = input_rank + axis;
        }

        if (axis > batch_index)
            axis -= 1;

        op->params["0"] = axis;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_glu, 20)

} // namespace ncnn

} // namespace pnnx
