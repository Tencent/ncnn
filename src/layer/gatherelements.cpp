// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gatherelements.h"

namespace ncnn {

GatherElements::GatherElements()
{
    one_blob_only = false;
    support_inplace = false;
}

int GatherElements::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);

    return 0;
}

int GatherElements::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() < 2)
        return -1;

    const Mat& data_blob = bottom_blobs[0];
    const Mat& index_blob = bottom_blobs[1];

    // Output has same shape as index_blob
    Mat& top_blob = top_blobs[0];
    top_blob.create(index_blob.w, index_blob.h, index_blob.c, data_blob.elemsize, data_blob.elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    int dims = data_blob.dims;
    int positive_axis = axis < 0 ? axis + dims : axis;
    if (positive_axis < 0 || positive_axis >= dims)
        return -1;

    const float* data = data_blob;
    const int* indices = (const int*)index_blob;
    float* out = top_blob;

    const int total = (int)top_blob.total();

    // Get axis dimension size
    int axis_dim_size = 1;
    if (dims == 1)
    {
        axis_dim_size = data_blob.w;
    }
    else if (dims == 2)
    {
        axis_dim_size = (positive_axis == 0) ? data_blob.w : data_blob.h;
    }
    else if (dims == 3)
    {
        axis_dim_size = (positive_axis == 0) ? data_blob.w : (positive_axis == 1) ? data_blob.h : data_blob.c;
    }

    for (int i = 0; i < total; i++)
    {
        // Calculate output coordinates from flat index
        int out_coords[3] = {0, 0, 0};
        int rem = i;
        
        if (dims >= 1)
        {
            out_coords[0] = rem % index_blob.w;
            rem /= index_blob.w;
        }
        if (dims >= 2)
        {
            out_coords[1] = rem % index_blob.h;
            rem /= index_blob.h;
        }
        if (dims >= 3)
        {
            out_coords[2] = rem;
        }

        // Get index value at this position
        int gather_idx = indices[i];
        
        // Handle negative indices
        if (gather_idx < 0)
            gather_idx += axis_dim_size;
        
        // Clamp to valid range
        if (gather_idx < 0) gather_idx = 0;
        if (gather_idx >= axis_dim_size) gather_idx = axis_dim_size - 1;

        // Calculate input coordinates (replace axis coordinate with gather_idx)
        int in_coords[3] = {0, 0, 0};
        for (int d = 0; d < 3; d++)
        {
            int data_d = d - (3 - dims);
            if (data_d >= 0 && data_d < 3)
            {
                if (data_d == positive_axis)
                    in_coords[data_d] = gather_idx;
                else
                    in_coords[data_d] = out_coords[d];
            }
        }

        // Calculate flat input index
        int flat_in = in_coords[0] + in_coords[1] * data_blob.w + in_coords[2] * (int)data_blob.cstep;

        out[i] = data[flat_in];
    }

    return 0;
}

} // namespace ncnn
