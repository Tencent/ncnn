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

    const int data_dims = data_blob.dims;
    const int positive_axis = axis < 0 ? axis + data_dims : axis;
    if (positive_axis < 0 || positive_axis >= data_dims)
        return -1;

    const float* data = data_blob;
    const size_t idx_elemsize = index_blob.elemsize / index_blob.elempack;
    float* out = top_blob;

    // PyTorch/ONNX axis ordering: axis=0 is outermost (c for 3D, h for 2D, w for 1D)
    int data_shape[3] = {1, 1, 1};
    if (data_dims == 1)
        data_shape[0] = data_blob.w;
    else if (data_dims == 2)
    {
        data_shape[0] = data_blob.h;
        data_shape[1] = data_blob.w;
    }
    else
    {
        data_shape[0] = data_blob.c;
        data_shape[1] = data_blob.h;
        data_shape[2] = data_blob.w;
    }

    const int axis_dim_size = data_shape[positive_axis];

    if (data_dims == 1)
    {
        // axis=0 only: output[x] = data[index[x]]
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
            out[x] = data[gather_idx];
        }
    }
    else if (data_dims == 2)
    {
        // axis=0 -> h (outer): output[y,x] = data[index[y,x], x]  ->  flat_in = gather_idx*w + x
        // axis=1 -> w (inner): output[y,x] = data[y, index[y,x]]  ->  flat_in = y*w + gather_idx
        const int dw = data_blob.w;
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
                    flat_in = gather_idx * dw + x;
                else
                    flat_in = y * dw + gather_idx;

                out[idx_flat] = data[flat_in];
            }
        }
    }
    else // data_dims == 3
    {
        // axis=0 -> c: output[z,y,x] = data[index[z,y,x], y, x]  ->  flat_in = gather_idx*cstep + y*w + x
        // axis=1 -> h: output[z,y,x] = data[z, index[z,y,x], x]  ->  flat_in = z*cstep + gather_idx*w + x
        // axis=2 -> w: output[z,y,x] = data[z, y, index[z,y,x]]  ->  flat_in = z*cstep + y*w + gather_idx
        const int dw = data_blob.w;
        const size_t in_cstep = data_blob.cstep;
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
                        flat_in = (int)(gather_idx * in_cstep) + y * dw + x;
                    else if (positive_axis == 1)
                        flat_in = (int)(z * in_cstep) + gather_idx * dw + x;
                    else
                        flat_in = (int)(z * in_cstep) + y * dw + gather_idx;

                    out[(int)(z * out_cstep) + y * top_blob.w + x] = data[flat_in];
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
