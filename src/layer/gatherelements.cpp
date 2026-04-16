// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gatherelements.h"

#include <stdint.h>

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

    // Output has same shape as index_blob (preserve rank)
    Mat& top_blob = top_blobs[0];
    if (index_blob.dims == 1)
        top_blob.create(index_blob.w, data_blob.elemsize, data_blob.elempack, opt.blob_allocator);
    else if (index_blob.dims == 2)
        top_blob.create(index_blob.w, index_blob.h, data_blob.elemsize, data_blob.elempack, opt.blob_allocator);
    else
        top_blob.create(index_blob.w, index_blob.h, index_blob.c, data_blob.elemsize, data_blob.elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    int data_dims = data_blob.dims;
    int positive_axis = axis < 0 ? axis + data_dims : axis;
    if (positive_axis < 0 || positive_axis >= data_dims)
        return -1;

    const float* data = data_blob;
    const size_t idx_elemsize = index_blob.elemsize / index_blob.elempack;
    float* out = top_blob;

    const int total = (int)top_blob.total();

    // Get axis dimension size
    int axis_dim_size = 1;
    if (data_dims == 1)
    {
        axis_dim_size = data_blob.w;
    }
    else if (data_dims == 2)
    {
        axis_dim_size = (positive_axis == 0) ? data_blob.w : data_blob.h;
    }
    else if (data_dims == 3)
    {
        axis_dim_size = (positive_axis == 0) ? data_blob.w : (positive_axis == 1) ? data_blob.h : data_blob.c;
    }

    for (int i = 0; i < total; i++)
    {
        // Get index value — handle int32 (elemsize=4) and int64 (elemsize=8)
        int gather_idx;
        if (idx_elemsize == 8)
            gather_idx = (int)((const int64_t*)(const void*)index_blob)[i];
        else
            gather_idx = ((const int*)(const void*)index_blob)[i];

        // Handle negative indices
        if (gather_idx < 0)
            gather_idx += axis_dim_size;

        // Clamp to valid range
        if (gather_idx < 0) gather_idx = 0;
        if (gather_idx >= axis_dim_size) gather_idx = axis_dim_size - 1;

        // Calculate input flat index based on axis
        // For 1D data: flat_in = gather_idx
        // For 2D data with axis=0: flat_in = gather_idx + y * w
        // For 2D data with axis=1: flat_in = x + gather_idx * w
        int flat_in = 0;

        if (data_dims == 1)
        {
            flat_in = gather_idx;
        }
        else if (data_dims == 2)
        {
            // Calculate position in output (which matches index_blob shape)
            int x = i % index_blob.w;
            int y = i / index_blob.w;

            if (positive_axis == 0)
            {
                // Gather along width: output[x,y] = data[gather_idx, y]
                flat_in = gather_idx + y * data_blob.w;
            }
            else
            {
                // Gather along height: output[x,y] = data[x, gather_idx]
                flat_in = x + gather_idx * data_blob.w;
            }
        }
        else if (data_dims == 3)
        {
            int x = i % index_blob.w;
            int tmp = i / index_blob.w;
            int y = tmp % index_blob.h;
            int z = tmp / index_blob.h;
            const int cstep = (int)data_blob.cstep;

            if (positive_axis == 0)
            {
                flat_in = gather_idx + y * data_blob.w + z * cstep;
            }
            else if (positive_axis == 1)
            {
                flat_in = x + gather_idx * data_blob.w + z * cstep;
            }
            else
            {
                flat_in = x + y * data_blob.w + gather_idx * cstep;
            }
        }

        out[i] = data[flat_in];
    }

    return 0;
}

} // namespace ncnn
