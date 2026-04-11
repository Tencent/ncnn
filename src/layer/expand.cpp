// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "expand.h"

namespace ncnn {

Expand::Expand()
{
    one_blob_only = false;
    support_inplace = false;
}

int Expand::load_param(const ParamDict& pd)
{
    return 0;
}

int Expand::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() < 2)
        return -1;

    const Mat& input_blob = bottom_blobs[0];
    const Mat& shape_blob = bottom_blobs[1];

    // shape_blob contains the target shape as int64/int32 values
    const int* target_shape = (const int*)shape_blob;
    int target_dims = (int)shape_blob.total();

    // Get input dimensions
    int in_dims = input_blob.dims;
    int in_shape[4] = {1, 1, 1, 1};
    in_shape[0] = input_blob.w;
    if (in_dims >= 2) in_shape[1] = input_blob.h;
    if (in_dims >= 3) in_shape[2] = input_blob.c;
    // For 4D, we'd need to handle differently but ncnn typically uses 3D blobs

    // Calculate output shape (broadcasting rules)
    int out_shape[4] = {1, 1, 1, 1};
    int max_dims = std::max(in_dims, target_dims);
    
    for (int i = 0; i < max_dims; i++)
    {
        int in_idx = i - (max_dims - in_dims);
        int target_idx = i - (max_dims - target_dims);
        
        int in_dim = (in_idx >= 0 && in_idx < in_dims) ? in_shape[in_idx] : 1;
        int target_dim = (target_idx >= 0 && target_idx < target_dims) ? target_shape[target_idx] : 1;
        
        // Broadcasting: if in_dim is 1, expand to target_dim; otherwise must match
        out_shape[i] = (in_dim == 1) ? target_dim : in_dim;
    }

    Mat& top_blob = top_blobs[0];
    
    if (max_dims == 1)
    {
        top_blob.create(out_shape[0], input_blob.elemsize, input_blob.elempack, opt.blob_allocator);
    }
    else if (max_dims == 2)
    {
        top_blob.create(out_shape[0], out_shape[1], input_blob.elemsize, input_blob.elempack, opt.blob_allocator);
    }
    else if (max_dims == 3)
    {
        top_blob.create(out_shape[0], out_shape[1], out_shape[2], input_blob.elemsize, input_blob.elempack, opt.blob_allocator);
    }
    else
    {
        return -1;
    }
    
    if (top_blob.empty())
        return -100;

    const float* inp = input_blob;
    float* out = top_blob;

    // Fill output by broadcasting input
    int total = (int)top_blob.total();
    
    for (int i = 0; i < total; i++)
    {
        // Calculate multi-dimensional coordinates
        int coords[4] = {0, 0, 0, 0};
        int rem = i;
        
        if (max_dims == 1)
        {
            coords[0] = rem;
        }
        else if (max_dims == 2)
        {
            coords[0] = rem % top_blob.w;
            coords[1] = rem / top_blob.w;
        }
        else if (max_dims == 3)
        {
            int wh = top_blob.w * top_blob.h;
            coords[0] = (rem % wh) % top_blob.w;
            coords[1] = (rem % wh) / top_blob.w;
            coords[2] = rem / wh;
        }

        // Map to input coordinates (modulo for expanded dimensions)
        int in_coords[4] = {0, 0, 0, 0};
        for (int d = 0; d < max_dims; d++)
        {
            int in_idx = d - (max_dims - in_dims);
            if (in_idx >= 0 && in_idx < in_dims)
            {
                int dim_size = (d == 0) ? input_blob.w : (d == 1 && in_dims >= 2) ? input_blob.h : input_blob.c;
                in_coords[in_idx] = coords[d] % dim_size;
            }
        }

        // Calculate flat input index
        int in_idx = 0;
        if (in_dims == 1)
        {
            in_idx = in_coords[0];
        }
        else if (in_dims == 2)
        {
            in_idx = in_coords[0] + in_coords[1] * input_blob.w;
        }
        else if (in_dims == 3)
        {
            size_t cstep = input_blob.cstep;
            in_idx = in_coords[0] + in_coords[1] * input_blob.w + in_coords[2] * (int)cstep;
        }

        out[i] = inp[in_idx];
    }

    return 0;
}

} // namespace ncnn
