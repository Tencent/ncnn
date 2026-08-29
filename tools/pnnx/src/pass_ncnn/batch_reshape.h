// Copyright 2026 futz12
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PNNX_NCNN_BATCH_RESHAPE_H
#define PNNX_NCNN_BATCH_RESHAPE_H

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

// A reshape needs NCNN's batch-aware path when the logical batch axis moves,
// or when it stays at the same logical axis but its extent changes.  The
// latter is the important case for flatten(0, 1): solve_batch_index may
// correctly mark both operands as axis 0 after the Conv1d consumer is seen,
// while the operation still changes n from B to B*F.
static inline bool is_batch_reshape(const Operand* input, const Operand* output)
{
    const int input_batch_axis = input->params.at("__ncnn_batch_axis").i;
    const int output_batch_axis = output->params.at("__ncnn_batch_axis").i;

    if (input_batch_axis != output_batch_axis)
        return true;

    if (input_batch_axis == 233)
        return false;

    int input_axis = input_batch_axis;
    int output_axis = output_batch_axis;
    if (input_axis < 0 && !input->shape.empty())
        input_axis += (int)input->shape.size();
    if (output_axis < 0 && !output->shape.empty())
        output_axis += (int)output->shape.size();

    if (input_axis < 0 || output_axis < 0
        || input_axis >= (int)input->shape.size()
        || output_axis >= (int)output->shape.size())
        return false;

    const int input_batch_size = input->shape[input_axis];
    const int output_batch_size = output->shape[output_axis];

    // -1 denotes an unresolved extent.  Without two concrete extents there
    // is no safe way to infer that the operation changes the batch size.
    return input_batch_size >= 0 && output_batch_size >= 0
           && input_batch_size != output_batch_size;
}

} // namespace ncnn

} // namespace pnnx

#endif // PNNX_NCNN_BATCH_RESHAPE_H
