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
        const int ncnn_batch_axis = op->inputs[0]->params["__ncnn_batch_axis"].i;

        int axis = captured_params.at("dim").i;
        if (axis < 0)
        {
            int input_rank = op->inputs[0]->shape.size();
            if (input_rank == 0)
                input_rank = op->outputs[0]->shape.size();
            if (input_rank > 0)
                axis = input_rank + axis;
            else if (ncnn_batch_axis != 233)
                fprintf(stderr, "glu axis around batch axis %d is unknown\n", ncnn_batch_axis);
        }

        bool axis_is_batch = false;
        if (ncnn_batch_axis != 233 && axis == ncnn_batch_axis)
        {
            fprintf(stderr, "glu along batch axis %d is not supported\n", ncnn_batch_axis);
            axis_is_batch = true;
        }

        if (!axis_is_batch && ncnn_batch_axis != 233 && axis > ncnn_batch_axis)
            axis -= 1;

        if (!axis_is_batch)
            op->params["0"] = axis;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(F_glu, 20)

} // namespace ncnn

} // namespace pnnx
