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
    const Mat& out_shape = index_blob;

    Mat& top_blob = top_blobs[0];
    top_blob.create(out_shape.w, out_shape.h, out_shape.c, data_blob.elemsize, data_blob.elempack, opt.blob_allocator);
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
        if (positive_axis == 0)
            axis_dim_size = data_blob.h;
        else
            axis_dim_size = data_blob.w;
    }
    else if (dims == 3)
    {
        if (positive_axis == 0)
            axis_dim_size = data_blob.c;
        else if (positive_axis == 1)
            axis_dim_size = data_blob.h;
        else
            axis_dim_size = data_blob.w;
    }

    for (int i = 0; i < total; i++)
    {
        // Calculate multi-dimensional coordinates from flat index
        int idx[4] = {0, 0, 0, 0};
        int rem = i;
        
        if (dims == 1)
        {
            idx[0] = rem;
        }
        else if (dims == 2)
        {
            idx[0] = rem % out_shape.w;
            idx[1] = rem / out_shape.w;
        }
        else if (dims == 3)
        {
            int wh = out_shape.w * out_shape.h;
            idx[0] = (rem % wh) % out_shape.w;
            idx[1] = (rem % wh) / out_shape.w;
            idx[2] = rem / wh;
        }

        // Get index value
        int gather_idx = indices[i];
        if (gather_idx < 0)
            gather_idx += axis_dim_size;

        // Clamp to valid range
        if (gather_idx < 0 || gather_idx >= axis_dim_size)
        {
            out[i] = 0.0f;
            continue;
        }

        // Replace coordinate at axis dimension
        idx[positive_axis] = gather_idx;

        // Calculate flat index into data
        int data_idx = 0;
        if (dims == 1)
        {
            data_idx = idx[0];
        }
        else if (dims == 2)
        {
            data_idx = idx[0] + idx[1] * data_blob.w;
        }
        else if (dims == 3)
        {
            size_t cstep = data_blob.cstep;
            data_idx = idx[0] + idx[1] * data_blob.w + idx[2] * (int)cstep;
        }

        out[i] = data[data_idx];
    }

    return 0;
}

} // namespace ncnn
