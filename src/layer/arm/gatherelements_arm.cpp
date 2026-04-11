// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gatherelements_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif

namespace ncnn {

int GatherElements_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

#if __ARM_NEON
    // ARM NEON optimized path - process 4 elements at a time
    const int nn = total >> 2;
    const int remain = total - (nn << 2);

    for (int i = 0; i < nn; i++)
    {
        int idx_base = i << 2;
        
        // Load 4 indices
        int32x4_t idx_vec = vld1q_s32(indices + idx_base);
        
        // Handle negative indices
        int32x4_t neg_mask = vcltq_s32(idx_vec, vdupq_n_s32(0));
        int32x4_t adjusted_idx = vaddq_s32(idx_vec, vdupq_n_s32(axis_dim_size));
        idx_vec = vbslq_s32(neg_mask, adjusted_idx, idx_vec);
        
        // Clamp to valid range
        int32x4_t clamp_mask = vcgtq_s32(idx_vec, vdupq_n_s32(axis_dim_size - 1));
        idx_vec = vbslq_s32(clamp_mask, vdupq_n_s32(axis_dim_size - 1), idx_vec);
        clamp_mask = vcltq_s32(idx_vec, vdupq_n_s32(0));
        idx_vec = vbslq_s32(clamp_mask, vdupq_n_s32(0), idx_vec);
        
        // Extract and gather
        int idx[4];
        vst1q_s32(idx, idx_vec);
        
        float32x4_t out_vec;
        for (int j = 0; j < 4; j++)
        {
            int gather_idx = idx[j];
            if (gather_idx < 0 || gather_idx >= axis_dim_size)
            {
                out[idx_base + j] = 0.0f;
            }
            else
            {
                // Calculate multi-dimensional coordinates
                int out_idx = idx_base + j;
                int coords[4] = {0, 0, 0, 0};
                int rem = out_idx;
                
                if (dims == 1)
                {
                    coords[0] = rem;
                }
                else if (dims == 2)
                {
                    coords[0] = rem % out_shape.w;
                    coords[1] = rem / out_shape.w;
                }
                else if (dims == 3)
                {
                    int wh = out_shape.w * out_shape.h;
                    coords[0] = (rem % wh) % out_shape.w;
                    coords[1] = (rem % wh) / out_shape.w;
                    coords[2] = rem / wh;
                }

                coords[positive_axis] = gather_idx;

                // Calculate flat input index
                int data_idx = 0;
                if (dims == 1)
                {
                    data_idx = coords[0];
                }
                else if (dims == 2)
                {
                    data_idx = coords[0] + coords[1] * data_blob.w;
                }
                else if (dims == 3)
                {
                    size_t cstep = data_blob.cstep;
                    data_idx = coords[0] + coords[1] * data_blob.w + coords[2] * (int)cstep;
                }

                out[idx_base + j] = data[data_idx];
            }
        }
    }

    // Handle remaining elements
    for (int i = 0; i < remain; i++)
    {
        int idx_base = (nn << 2) + i;
        int gather_idx = indices[idx_base];
        
        if (gather_idx < 0) gather_idx += axis_dim_size;
        if (gather_idx < 0 || gather_idx >= axis_dim_size)
        {
            out[idx_base] = 0.0f;
            continue;
        }

        // Calculate coordinates and gather (same as scalar implementation)
        int coords[4] = {0, 0, 0, 0};
        int rem = idx_base;
        
        if (dims == 1)
        {
            coords[0] = rem;
        }
        else if (dims == 2)
        {
            coords[0] = rem % out_shape.w;
            coords[1] = rem / out_shape.w;
        }
        else if (dims == 3)
        {
            int wh = out_shape.w * out_shape.h;
            coords[0] = (rem % wh) % out_shape.w;
            coords[1] = (rem % wh) / out_shape.w;
            coords[2] = rem / wh;
        }

        coords[positive_axis] = gather_idx;

        int data_idx = 0;
        if (dims == 1)
        {
            data_idx = coords[0];
        }
        else if (dims == 2)
        {
            data_idx = coords[0] + coords[1] * data_blob.w;
        }
        else if (dims == 3)
        {
            size_t cstep = data_blob.cstep;
            data_idx = coords[0] + coords[1] * data_blob.w + coords[2] * (int)cstep;
        }

        out[idx_base] = data[data_idx];
    }
#else
    // Scalar fallback - same as base implementation
    for (int i = 0; i < total; i++)
    {
        int gather_idx = indices[i];
        if (gather_idx < 0) gather_idx += axis_dim_size;
        if (gather_idx < 0 || gather_idx >= axis_dim_size)
        {
            out[i] = 0.0f;
            continue;
        }

        // Calculate coordinates
        int coords[4] = {0, 0, 0, 0};
        int rem = i;
        
        if (dims == 1)
        {
            coords[0] = rem;
        }
        else if (dims == 2)
        {
            coords[0] = rem % out_shape.w;
            coords[1] = rem / out_shape.w;
        }
        else if (dims == 3)
        {
            int wh = out_shape.w * out_shape.h;
            coords[0] = (rem % wh) % out_shape.w;
            coords[1] = (rem % wh) / out_shape.w;
            coords[2] = rem / wh;
        }

        coords[positive_axis] = gather_idx;

        int data_idx = 0;
        if (dims == 1)
        {
            data_idx = coords[0];
        }
        else if (dims == 2)
        {
            data_idx = coords[0] + coords[1] * data_blob.w;
        }
        else if (dims == 3)
        {
            size_t cstep = data_blob.cstep;
            data_idx = coords[0] + coords[1] * data_blob.w + coords[2] * (int)cstep;
        }

        out[i] = data[data_idx];
    }
#endif // __ARM_NEON

    return 0;
}

} // namespace ncnn
