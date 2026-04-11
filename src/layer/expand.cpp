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

    // shape_blob contains the target shape as int32/int64 values
    const int* target_shape = (const int*)shape_blob;
    int target_dims = (int)shape_blob.total();

    // Get input dimensions
    int in_dims = input_blob.dims;
    int in_shape[3] = {1, 1, 1};
    in_shape[0] = input_blob.w;
    if (in_dims >= 2) in_shape[1] = input_blob.h;
    if (in_dims >= 3) in_shape[2] = input_blob.c;

    // Calculate output shape using numpy broadcasting rules
    // Shapes are aligned from the right (last dimension)
    int out_shape[3] = {1, 1, 1};
    int out_dims = target_dims;
    if (out_dims > 3) out_dims = 3;
    
    for (int i = 0; i < 3; i++)
    {
        // Calculate index into input and target shapes (aligned from right)
        int in_idx = i - (3 - in_dims);
        int target_idx = i - (3 - target_dims);
        
        int in_dim = (in_idx >= 0 && in_idx < 3) ? in_shape[in_idx] : 1;
        int target_dim = (target_idx >= 0 && target_idx < 3) ? target_shape[target_idx] : 1;
        
        // Broadcasting rules:
        // - If both are 1, output is 1
        // - If one is 1, output is the other
        // - If both are > 1, they must match
        if (in_dim == 1)
        {
            out_shape[i] = target_dim;
        }
        else if (target_dim == 1)
        {
            out_shape[i] = in_dim;
        }
        else
        {
            // Both > 1, should match
            out_shape[i] = target_dim;
        }
    }

    Mat& top_blob = top_blobs[0];

    // Create output blob with correct shape
    if (out_dims == 1)
    {
        top_blob.create(out_shape[0], input_blob.elemsize, input_blob.elempack, opt.blob_allocator);
    }
    else if (out_dims == 2)
    {
        top_blob.create(out_shape[0], out_shape[1], input_blob.elemsize, input_blob.elempack, opt.blob_allocator);
    }
    else if (out_dims == 3)
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
        // Calculate output coordinates from flat index
        int rem = i;
        int out_coords[3] = {0, 0, 0};
        
        if (out_dims >= 1)
        {
            out_coords[0] = rem % top_blob.w;
            rem /= top_blob.w;
        }
        if (out_dims >= 2)
        {
            out_coords[1] = rem % top_blob.h;
            rem /= top_blob.h;
        }
        if (out_dims >= 3)
        {
            out_coords[2] = rem;
        }

        // Map to input coordinates using broadcasting
        int in_coords[3] = {0, 0, 0};
        for (int d = 0; d < 3; d++)
        {
            int in_idx = d - (3 - in_dims);
            if (in_idx >= 0 && in_idx < 3)
            {
                if (in_shape[in_idx] == 1)
                {
                    in_coords[in_idx] = 0;
                }
                else
                {
                    in_coords[in_idx] = out_coords[d] % in_shape[in_idx];
                }
            }
        }

        // Calculate flat input index
        int in_idx = in_coords[0] + in_coords[1] * input_blob.w + in_coords[2] * (int)input_blob.cstep;

        out[i] = inp[in_idx];
    }

    return 0;
}

} // namespace ncnn
