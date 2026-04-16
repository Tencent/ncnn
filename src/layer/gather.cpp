// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gather.h"

#include <stdint.h>

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

    // Only float32 data supported
    if (input_blob.elemsize / input_blob.elempack != 4)
        return -1;

    // Only dims 1/2/3 supported
    if (dims > 3 || index_blob.dims > 3)
        return -1;

    int positive_axis = axis < 0 ? axis + dims : axis;
    if (positive_axis < 0 || positive_axis >= dims)
        return -1;

    // PyTorch-style axis ordering: axis=0 is outermost (c for 3D, h for 2D, w for 1D)
    // shape[] maps axis -> dimension size in that PyTorch order
    int shape[3] = {1, 1, 1};
    if (dims == 1)
        shape[0] = input_blob.w;
    else if (dims == 2)
    {
        shape[0] = input_blob.h;
        shape[1] = input_blob.w;
    }
    else
    {
        shape[0] = input_blob.c;
        shape[1] = input_blob.h;
        shape[2] = input_blob.w;
    }

    const int axis_dim_size = shape[positive_axis];

    // Output shape matches index_blob shape exactly (preserve rank)
    Mat& top_blob = top_blobs[0];
    if (index_blob.dims == 1)
        top_blob.create(index_blob.w, input_blob.elemsize, input_blob.elempack, opt.blob_allocator);
    else if (index_blob.dims == 2)
        top_blob.create(index_blob.w, index_blob.h, input_blob.elemsize, input_blob.elempack, opt.blob_allocator);
    else
        top_blob.create(index_blob.w, index_blob.h, index_blob.c, input_blob.elemsize, input_blob.elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* inp = input_blob;
    // Indices may be int32 (elemsize=4) or int64 (elemsize=8)
    const size_t idx_elemsize = index_blob.elemsize / index_blob.elempack;
    float* out = top_blob;

    if (dims == 1)
    {
        // axis=0 only: output[x] = input[index[x]]
        for (int x = 0; x < index_blob.w; x++)
        {
            int gather_idx;
            if (idx_elemsize == 8)
                gather_idx = (int)((const int64_t*)(const void*)index_blob)[x];
            else
                gather_idx = ((const int*)(const void*)index_blob)[x];
            if (gather_idx < 0) gather_idx += axis_dim_size;
            if (gather_idx < 0) gather_idx = 0;
            if (gather_idx >= axis_dim_size) gather_idx = axis_dim_size - 1;
            out[x] = inp[gather_idx];
        }
    }
    else if (dims == 2)
    {
        // PyTorch axis=0 -> h (outer), axis=1 -> w (inner)
        // axis=0: output[y,x] = input[index[y,x], x]  ->  flat_in = gather_idx*w + x
        // axis=1: output[y,x] = input[y, index[y,x]]  ->  flat_in = y*w + gather_idx
        const int iw = input_blob.w;
        for (int y = 0; y < index_blob.h; y++)
        {
            for (int x = 0; x < index_blob.w; x++)
            {
                int idx_flat = y * index_blob.w + x;
                int gather_idx;
                if (idx_elemsize == 8)
                    gather_idx = (int)((const int64_t*)(const void*)index_blob)[idx_flat];
                else
                    gather_idx = ((const int*)(const void*)index_blob)[idx_flat];
                if (gather_idx < 0) gather_idx += axis_dim_size;
                if (gather_idx < 0) gather_idx = 0;
                if (gather_idx >= axis_dim_size) gather_idx = axis_dim_size - 1;

                int flat_in;
                if (positive_axis == 0)
                    flat_in = gather_idx * iw + x;
                else
                    flat_in = y * iw + gather_idx;

                out[idx_flat] = inp[flat_in];
            }
        }
    }
    else // dims == 3
    {
        // PyTorch axis=0 -> c (outer), axis=1 -> h, axis=2 -> w (inner)
        // axis=0: output[z,y,x] = input[index[z,y,x], y, x]  ->  flat_in = gather_idx*cstep + y*w + x
        // axis=1: output[z,y,x] = input[z, index[z,y,x], x]  ->  flat_in = z*cstep + gather_idx*w + x
        // axis=2: output[z,y,x] = input[z, y, index[z,y,x]]  ->  flat_in = z*cstep + y*w + gather_idx
        const int iw = input_blob.w;
        const size_t in_cstep = input_blob.cstep;
        const size_t idx_cstep = index_blob.cstep;
        const size_t out_cstep = top_blob.cstep;

        for (int z = 0; z < index_blob.c; z++)
        {
            for (int y = 0; y < index_blob.h; y++)
            {
                for (int x = 0; x < index_blob.w; x++)
                {
                    int idx_flat = (int)(z * idx_cstep) + y * index_blob.w + x;
                    int gather_idx;
                    if (idx_elemsize == 8)
                        gather_idx = (int)((const int64_t*)(const void*)index_blob)[idx_flat];
                    else
                        gather_idx = ((const int*)(const void*)index_blob)[idx_flat];
                    if (gather_idx < 0) gather_idx += axis_dim_size;
                    if (gather_idx < 0) gather_idx = 0;
                    if (gather_idx >= axis_dim_size) gather_idx = axis_dim_size - 1;

                    int flat_in;
                    if (positive_axis == 0)
                        flat_in = (int)(gather_idx * in_cstep) + y * iw + x;
                    else if (positive_axis == 1)
                        flat_in = (int)(z * in_cstep) + gather_idx * iw + x;
                    else
                        flat_in = (int)(z * in_cstep) + y * iw + gather_idx;

                    out[(int)(z * out_cstep) + y * top_blob.w + x] = inp[flat_in];
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
