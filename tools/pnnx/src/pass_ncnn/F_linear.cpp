// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "F_linear.h"

namespace pnnx {

namespace ncnn {

void convert_aten_F_linear(Graph& graph)
{
    for (int i = (int)graph.ops.size() - 1; i >= 0; i--)
    {
        Operator* op = graph.ops[i];

        if (op->type != "F.linear")
            continue;

        // 2 or 3 inputs: input, weight[, bias]. a pt2 F.linear exported with
        // bias=None folds the None bias into a bias=None parameter, leaving a
        // two-tensor node (lowered as Gemm without C)
        if (op->inputs.size() != 2 && op->inputs.size() != 3)
        {
            fprintf(stderr, "unsupported F.linear input count %zu\n", op->inputs.size());
            continue;
        }

        const std::vector<int>& in_shape = op->inputs[0]->shape;
        const int rank = (int)in_shape.size();

        // output N1M: a rank-3 input with a singleton middle dim (e.g. [S,1,F])
        // can use the Gemm N1M layout (ncnn Gemm only uses c as M and drops the
        // h dim for 3D input, so it is only valid when h == 1). Any other rank-3
        // input (middle dim > 1) must be flattened like the other ranks, or the
        // h dimension would be silently dropped from the multiplication.
        int n1m = 0;
        if (rank == 3 && in_shape.size() >= 2 && in_shape[1] == 1)
            n1m = 1;

        // non-N1M (1D/2D/4D/5D, plus rank-3 with a non-singleton middle dim):
        // insert reshape0 to flatten into 2D (h,F), Gemm 2D, reshape1 restores
        // the original rank.
        if ((rank != 3 || n1m == 0) && rank >= 1 && rank <= 5)
        {
            Operand* fl_in = op->inputs[0];
            Operand* fl_out = op->outputs[0];

            // reshape_h = product of the first rank-1 dims (1 for 1D input).
            // when the input's logical axis 0 is the ncnn batch axis, that
            // axis has already been extracted from every runtime blob, so it
            // must not be folded into the per-sample flatten height
            int batch_axis = 233; // 233 = no explicit batch axis
            if (fl_in->params.find("__ncnn_batch_axis") != fl_in->params.end())
                batch_axis = fl_in->params.at("__ncnn_batch_axis").i;
            int reshape_h = 1;
            for (int j = 0; j < rank - 1; j++)
            {
                if (j == batch_axis)
                    continue;
                if (in_shape[j] == -1)
                {
                    reshape_h = -1;
                    break;
                }
                if (in_shape[j] > 0 && reshape_h > 2147483647 / in_shape[j])
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

            // reconnect: fl_in consumer op -> reshape0; op input -> reshape0_out; op output -> reshape1_in; fl_out producer -> reshape1
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

            // shapes (output shape may be unknown; guard the out-rank access)
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
        }
        else if (rank == 3 && n1m == 1)
        {
            // N1M layout: a rank-3 [S,1,F] input maps directly onto the ncnn
            // Gemm N1M output (M==1) - no flatten reshape pair is needed and
            // the output keeps its [S,1,N] form through the Gemm N1M flag
        }
        else
        {
            // unknown rank or rank>5 cannot be expressed by ncnn Gemm (it
            // would silently drop the middle dims); keep the op untouched
            fprintf(stderr, "unsupported F.linear input rank %d\n", rank);
            continue;
        }

        op->type = "Gemm";
        op->name = std::string("gemm_") + op->name;

        op->params.clear();
        op->params["0"] = 1.f;  // alpha (must be float: Gemm::load_param reads float get(0))
        op->params["1"] = 1.f;  // beta (must be float: Gemm::load_param reads float get(1))
        op->params["2"] = 0;    // transA
        op->params["3"] = 1;    // transB (weight (out,in) -> (in,out))
        op->params["4"] = 0;    // constantA (A=input blob)
        op->params["5"] = 0;    // constantB (B=weight blob)
        op->params["6"] = 0;    // constantC (C=bias blob)
        op->params["11"] = n1m; // output_N1M
        op->params["12"] = 1;   // output_elempack
        op->params["13"] = 0;   // output_elemtype
        op->params["14"] = 0;   // output_transpose
    }
}

} // namespace ncnn

} // namespace pnnx
