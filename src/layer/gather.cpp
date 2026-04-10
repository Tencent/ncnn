// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gather.h"

namespace ncnn {

Gather::Gather()
{
    one_blob_only = false;
    support_inplace = false;
}

int Gather::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);

    return 0;
}

int Gather::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() < 2)
        return -1;

    const Mat& input_blob = bottom_blobs[0];
    const Mat& index_blob = bottom_blobs[1];
    const int dims = input_blob.dims;

    // index_blob should contain int64 or int32 indices
    // For simplicity we treat it as float and cast
    const int index_size = (int)index_blob.total();

    int positive_axis = axis < 0 ? axis + dims : axis;
    if (positive_axis < 0 || positive_axis >= dims)
        return -1;

    int shape[4] = {1, 1, 1, 1};
    shape[0] = input_blob.w;
    if (dims >= 2) shape[1] = input_blob.h;
    if (dims == 3)    shape[2] = input_blob.c;
    if (dims == 4)    shape[2] = input_blob.c; // w*h*c layout

    const int axis_dim_size = shape[positive_axis];

    // Output shape matches index_blob shape
    const Mat& out_shape = index_blob;

    // Allocate output (same dtype as input, shape matches index)
    Mat& top_blob = top_blobs[0];
    top_blob.create(out_shape.w, out_shape.h, out_shape.c, input_blob.elemsize, input_blob.elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* inp = input_blob;
    const int* idx = (const int*)index_blob;
    float* out = top_blob;

    // General case: iterate over all output positions
    // Map flat output index to multi-dimensional coords,
    // then compute corresponding input position with index substitution
    const int total_out = (int)top_blob.total();
    for (int i = 0; i < total_out; i++)
    {
        // Decompose flat index i into coordinates based on top_blob shape
        int rem = i;
        int coord_out[4] = {0, 0, 0, 0};
        if (top_blob.dims == 1) {
            coord_out[0] = rem;
        } else if (top_blob.dims == 2) {
            coord_out[0] = rem % top_blob.w;
            coord_out[1] = rem / top_blob.w;
        } else if (top_blob.dims == 3) {
            int hw = top_blob.w * top_blob.h;
            coord_out[0] = (rem % hw) % top_blob.w;
            coord_out[1] = (rem % hw) / top_blob.w;
            coord_out[2] = rem / hw;
        }

        // Get index value at this output position
        int gather_idx = idx[i];
        // Handle negative indices
        if (gather_idx < 0) gather_idx += axis_dim_size;

        // Build input coordinate (same as output, but axis coord replaced)
        int coord_in[4] = {coord_out[0], coord_out[1], coord_out[2], coord_out[3]};
        coord_in[positive_axis] = gather_idx;

        // Clamp to input bounds
        if (coord_in[positive_axis] >= axis_dim_size) coord_in[positive_axis] = axis_dim_size - 1;
        if (coord_in[positive_axis] < 0) coord_in[positive_axis] = 0;

        // Compute flat input index
        int flat_in = 0;
        if (dims == 1) {
            flat_in = coord_in[0];
        } else if (dims == 2) {
            flat_in = coord_in[0] + coord_in[1] * input_blob.w;
        } else if (dims == 3) {
            // ncnn 3D layout: w * h * c, with cstride padding
            size_t cstep = input_blob.cstep;
            flat_in = coord_in[0] + coord_in[1] * input_blob.w + coord_in[2] * (int)cstep;
        }

        out[i] = inp[flat_in];
    }

    return 0;
}

} // namespace ncnn
