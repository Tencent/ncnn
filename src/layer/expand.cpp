// ARM NEON optimized implementation for Expand
// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "expand.h"
#include <algorithm>

#if __ARM_NEON
#include <arm_neon.h>
#endif

namespace ncnn {

int Expand::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() < 2)
        return -1;

    const Mat& input_blob = bottom_blobs[0];
    const Mat& shape_blob = bottom_blobs[1];

    const int* target_shape = (const int*)shape_blob;
    int target_dims = (shape_blob.dims == 1) ? shape_blob.w : (int)shape_blob.total();

    int in_dims = input_blob.dims;
    int in_shape[3] = {1, 1, 1};
    in_shape[0] = input_blob.w;
    if (in_dims >= 2) in_shape[1] = input_blob.h;
    if (in_dims >= 3) in_shape[2] = input_blob.c;

    int out_dims = std::max(in_dims, target_dims);
    if (out_dims > 3) out_dims = 3;
    
    int out_shape[3] = {1, 1, 1};

    for (int i = 0; i < out_dims; i++)
    {
        int in_idx = i - (out_dims - in_dims);
        int target_idx = i - (out_dims - target_dims);

        int in_dim = (in_idx >= 0 && in_idx < 3) ? in_shape[in_idx] : 1;
        int target_dim = (target_idx >= 0 && target_idx < target_dims) ? target_shape[target_idx] : 1;

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
            out_shape[i] = target_dim;
        }
    }

    Mat& top_blob = top_blobs[0];

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

    int total = (int)top_blob.total();

    // ARM NEON optimized path for simple expansion (broadcast from 1 element)
    #if __ARM_NEON
    if (in_dims == 1 && in_shape[0] == 1 && out_dims == 1 && opt.num_threads > 1)
    {
        float val = inp[0];
        float32x4_t val_vec = vdupq_n_f32(val);
        
        const int nn = total >> 2;
        const int remain = total - (nn << 2);
        
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < nn; i++)
        {
            int idx = i << 2;
            vst1q_f32(out + idx, val_vec);
        }
        
        for (int i = nn << 2; i < total; i++)
        {
            out[i] = val;
        }
        
        return 0;
    }
    #endif

    // General path with OpenMP
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < total; i++)
    {
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

        int in_coords[3] = {0, 0, 0};
        for (int d = 0; d < out_dims; d++)
        {
            int in_idx = d - (out_dims - in_dims);
            if (in_idx >= 0 && in_idx < 3 && in_shape[in_idx] > 1)
            {
                in_coords[in_idx] = out_coords[d] % in_shape[in_idx];
            }
            else if (in_idx >= 0 && in_idx < 3)
            {
                in_coords[in_idx] = 0;
            }
        }

        int in_idx = in_coords[0] + in_coords[1] * input_blob.w + in_coords[2] * (int)input_blob.cstep;
        out[i] = inp[in_idx];
    }

    return 0;
}

} // namespace ncnn
