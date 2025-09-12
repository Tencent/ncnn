// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class Tensor_reshape_as : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
4 3
pnnx.Input              input_0     0 1 input
pnnx.Input              input_1     0 1 other
Tensor.reshape_as       op_0        2 1 input other out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Reshape";
    }

    const char* name_str() const
    {
        return "reshape_as";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& /*captured_params*/) const
    {
        const int shape_rank = (int)op->outputs[0]->shape.size();

        const int batch_index = op->outputs[0]->params["__batch_index"].i;

        if (batch_index != 0 && batch_index != 233)
        {
            if (shape_rank == 0 || op->outputs[0]->shape[batch_index] != 1)
                fprintf(stderr, "reshape_as tensor with batch index %d is not supported yet!\n", batch_index);
        }

        if (shape_rank == 1)
        {
            op->params["0"] = -1;
        }
        else if (shape_rank == 2)
        {
            if (batch_index == 233)
            {
                op->params["6"] = "1w,1h";
            }
            else
            {
                op->params["0"] = -1;
            }
        }
        else if (shape_rank == 3)
        {
            if (batch_index == 233)
            {
                op->params["6"] = "1w,1h,1c";
            }
            else
            {
                op->params["6"] = "1w,1h";
            }
        }
        else if (shape_rank == 4)
        {
            if (batch_index == 233)
            {
                op->params["6"] = "1w,1h,1d,1c";
            }
            else
            {
                op->params["6"] = "1w,1h,1c";
            }
        }
        else if (shape_rank == 5)
        {
            if (batch_index == 233)
            {
                fprintf(stderr, "reshape_as tensor with unbatched 5 rank tensor is not supported yet!\n");
            }
            else
            {
                op->params["6"] = "1w,1h,1d,1c";
            }
        }
        else
        {
            fprintf(stderr, "reshape_as tensor with over 5 / unknown rank tensor is not supported yet!\n");
        }
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_reshape_as, 20)

} // namespace ncnn

} // namespace pnnx
