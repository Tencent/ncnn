// ARM NEON optimized implementation for GatherElements
// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gatherelements_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif

namespace ncnn {

#if __ARM_NEON
int GatherElements_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() < 2)
        return -1;

    const Mat& data_blob = bottom_blobs[0];
    const Mat& index_blob = bottom_blobs[1];

    Mat& top_blob = top_blobs[0];
    top_blob.create(index_blob.w, index_blob.h, index_blob.c, data_blob.elemsize, data_blob.elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    int data_dims = data_blob.dims;
    int positive_axis = axis < 0 ? axis + data_dims : axis;
    if (positive_axis < 0 || positive_axis >= data_dims)
        return -1;

    const float* data = data_blob;
    const int* indices = (const int*)index_blob;
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

    // ARM NEON optimized path for 1D case
    if (data_dims == 1 && opt.num_threads > 1)
    {
        const int nn = total >> 2;
        const int remain = total - (nn << 2);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < nn; i++)
        {
            int idx = i << 2;
            
            // Load 4 indices
            int32x4_t idx_vec = vld1q_s32(indices + idx);
            
            // Handle negative indices: if idx < 0, idx += axis_dim_size
            int32x4_t neg_mask = vcltq_s32(idx_vec, vdupq_n_s32(0));
            int32x4_t adjusted = vaddq_s32(idx_vec, vdupq_n_s32(axis_dim_size));
            idx_vec = vbslq_s32(neg_mask, adjusted, idx_vec);
            
            // Clamp to [0, axis_dim_size-1]
            int32x4_t upper = vdupq_n_s32(axis_dim_size - 1);
            int32x4_t lower = vdupq_n_s32(0);
            idx_vec = vminq_s32(idx_vec, upper);
            idx_vec = vmaxq_s32(idx_vec, lower);
            
            // Gather values
            float32x4_t out_vec;
            int32_t idx_arr[4];
            vst1q_s32(idx_arr, idx_vec);
            
            for (int j = 0; j < 4; j++)
            {
                ((float*)&out_vec)[j] = data[idx_arr[j]];
            }
            
            vst1q_f32(out + idx, out_vec);
        }

        // Handle remaining elements
        for (int i = nn << 2; i < total; i++)
        {
            int gather_idx = indices[i];
            if (gather_idx < 0) gather_idx += axis_dim_size;
            if (gather_idx < 0) gather_idx = 0;
            if (gather_idx >= axis_dim_size) gather_idx = axis_dim_size - 1;
            out[i] = data[gather_idx];
        }

        return 0;
    }

    // Scalar path with OpenMP
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < total; i++)
    {
        int gather_idx = indices[i];
        if (gather_idx < 0) gather_idx += axis_dim_size;
        if (gather_idx < 0) gather_idx = 0;
        if (gather_idx >= axis_dim_size) gather_idx = axis_dim_size - 1;

        int flat_in = 0;
        if (data_dims == 1)
        {
            flat_in = gather_idx;
        }
        else if (data_dims == 2)
        {
            int x = i % index_blob.w;
            int y = i / index_blob.w;
            if (positive_axis == 0)
                flat_in = gather_idx + y * data_blob.w;
            else
                flat_in = x + gather_idx * data_blob.w;
        }
        else if (data_dims == 3)
        {
            int x = i % index_blob.w;
            int tmp = i / index_blob.w;
            int y = tmp % index_blob.h;
            int z = tmp / index_blob.h;
            if (positive_axis == 0)
                flat_in = gather_idx + (y + z * data_blob.h) * data_blob.w;
            else if (positive_axis == 1)
                flat_in = x + (gather_idx + z * data_blob.h) * data_blob.w;
            else
                flat_in = x + (y + gather_idx * data_blob.h) * data_blob.w;
        }

        out[i] = data[flat_in];
    }

    return 0;
}
#else
int GatherElements_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    return GatherElements::forward(bottom_blobs, top_blobs, opt);
}
#endif

} // namespace ncnn
