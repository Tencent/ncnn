// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "expand.h"
#include <algorithm>
#include <stdint.h>

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

    // shape_blob may be int32 (elemsize=4) or int64 (elemsize=8) from ONNX
    const size_t shape_elemsize = shape_blob.elemsize / shape_blob.elempack;
    const bool shape_is_int64 = (shape_elemsize == 8);
    int target_dims = (shape_blob.dims == 1) ? shape_blob.w : (int)shape_blob.total();
    if (target_dims > 3) target_dims = 3;

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

        // Read target dimension from shape_blob (int32 or int64)
        int target_dim = 1;
        if (target_idx >= 0 && target_idx < target_dims)
        {
            if (shape_is_int64)
                target_dim = (int)((const int64_t*)(const void*)shape_blob)[target_idx];
            else
                target_dim = ((const int*)(const void*)shape_blob)[target_idx];
        }

        if (in_dim == 1)
        {
            out_shape[i] = (target_dim > 0) ? target_dim : 1;
        }
        else if (target_dim == 1 || target_dim == -1)
        {
            out_shape[i] = in_dim;
        }
        else if (target_dim == in_dim)
        {
            out_shape[i] = in_dim;
        }
        else
        {
            // Invalid broadcast: target_dim != in_dim and neither is 1
            return -1;
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

// HOT PATH: Broadcast from single value - highly optimized
#if __ARM_NEON
    if (in_dims == 1 && in_shape[0] == 1 && out_dims == 1 && opt.num_threads > 1)
    {
        float val = inp[0];
        float32x4_t val_vec = vdupq_n_f32(val);

        const int nn = total >> 3; // Process 8 at a time
        const int remain = total - (nn << 3);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < nn; i++)
        {
            int idx = i << 3;
            // Store 8 values at once using 2x float32x4
            vst1q_f32(out + idx, val_vec);
            vst1q_f32(out + idx + 4, val_vec);
        }

        // Handle remaining 4 elements
        for (int i = nn << 3; i < total - 3; i += 4)
        {
            vst1q_f32(out + i, val_vec);
        }

        // Handle remaining 1-3 elements
        for (int i = total - (total % 4); i < total; i++)
        {
            out[i] = val;
        }

        return 0;
    }

    // HOT PATH: Broadcast 1D to 2D (row vector to matrix)
    if (in_dims == 1 && out_dims == 2 && in_shape[0] == out_shape[0] && opt.num_threads > 1)
    {
        const int w = out_shape[0];
        const int h = out_shape[1];
        const int nn = w >> 2;
        const int remain = w - (nn << 2);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int row = 0; row < h; row++)
        {
            float* dst_row = out + row * w;

            // Prefetch next row
            if (row + 1 < h)
            {
                __builtin_prefetch(inp, 0, 3);
            }

            // Copy row with NEON
            for (int j = 0; j < nn; j++)
            {
                float32x4_t v = vld1q_f32(inp + j * 4);
                vst1q_f32(dst_row + j * 4, v);
            }
            for (int j = nn << 2; j < w; j++)
            {
                dst_row[j] = inp[j];
            }
        }

        return 0;
    }
#endif

    // HOT PATH: 2D to 2D with same width (broadcast height)
    if (in_dims == 2 && out_dims == 2 && in_shape[0] == out_shape[0] && opt.num_threads > 1)
    {
        const int w = out_shape[0];
        const int h = out_shape[1];
        const int in_h = in_shape[1];

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int row = 0; row < h; row++)
        {
            int src_row = row % in_h;
            const float* src_ptr = inp + src_row * w;
            float* dst_ptr = out + row * w;

            // Copy entire row
            const int nn = w >> 2;
            for (int j = 0; j < nn; j++)
            {
                float32x4_t v = vld1q_f32(src_ptr + j * 4);
                vst1q_f32(dst_ptr + j * 4, v);
            }
            for (int j = nn << 2; j < w; j++)
            {
                dst_ptr[j] = src_ptr[j];
            }
        }

        return 0;
    }

    // General path with OpenMP and optimized indexing
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
