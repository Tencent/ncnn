// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gatherelements.h"

#include <stddef.h>
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

    // Output has same shape as index_blob (same rank)
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

    // PyTorch/ONNX axis ordering: axis=0 = outermost (c for 3D, h for 2D, w for 1D)
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

    const int64_t* idx_ptr64 = (const int64_t*)(const void*)index_blob;
    const int* idx_ptr32 = (const int*)(const void*)index_blob;

#define READ_IDX(pos) \
    (idx_elemsize == 8 ? (int)idx_ptr64[(pos)] : idx_ptr32[(pos)])

#define CLAMP_IDX(gi)                                     \
    do {                                                  \
        if ((gi) < 0) (gi) += axis_dim_size;              \
        if ((gi) < 0) (gi) = 0;                           \
        if ((gi) >= axis_dim_size) (gi) = axis_dim_size - 1; \
    } while (0)

    if (data_dims == 1)
    {
        for (int x = 0; x < index_blob.w; x++)
        {
            int gi = READ_IDX(x);
            CLAMP_IDX(gi);
            out[x] = data[gi];
        }
    }
    else if (data_dims == 2)
    {
        const int dw = data_blob.w;
        const int idxw = index_blob.w;

        if (positive_axis == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < index_blob.h; y++)
            {
                float* out_row = out + y * top_blob.w;
                for (int x = 0; x < idxw; x++)
                {
                    int gi = READ_IDX(y * idxw + x);
                    CLAMP_IDX(gi);
                    out_row[x] = data[gi * dw + x];
                }
            }
        }
        else // positive_axis == 1
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < index_blob.h; y++)
            {
                const float* data_row = data + y * dw;
                float* out_row = out + y * top_blob.w;
                for (int x = 0; x < idxw; x++)
                {
                    int gi = READ_IDX(y * idxw + x);
                    CLAMP_IDX(gi);
                    out_row[x] = data_row[gi];
                }
            }
        }
    }
    else // data_dims == 3
    {
        const int dw = data_blob.w;
        const size_t in_cstep = data_blob.cstep;
        const size_t idx_cstep = index_blob.cstep;
        const size_t out_cstep = top_blob.cstep;
        const int idxw = index_blob.w;

        if (positive_axis == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int z = 0; z < index_blob.c; z++)
            {
                float* out_chan = out + z * out_cstep;
                for (int y = 0; y < index_blob.h; y++)
                {
                    float* out_row = out_chan + y * top_blob.w;
                    for (int x = 0; x < idxw; x++)
                    {
                        int gi = READ_IDX((int)(z * idx_cstep) + y * idxw + x);
                        CLAMP_IDX(gi);
                        out_row[x] = data[(int)(gi * in_cstep) + y * dw + x];
                    }
                }
            }
        }
        else if (positive_axis == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int z = 0; z < index_blob.c; z++)
            {
                const float* data_chan = data + z * in_cstep;
                float* out_chan = out + z * out_cstep;
                for (int y = 0; y < index_blob.h; y++)
                {
                    float* out_row = out_chan + y * top_blob.w;
                    for (int x = 0; x < idxw; x++)
                    {
                        int gi = READ_IDX((int)(z * idx_cstep) + y * idxw + x);
                        CLAMP_IDX(gi);
                        out_row[x] = data_chan[gi * dw + x];
                    }
                }
            }
        }
        else // positive_axis == 2
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int z = 0; z < index_blob.c; z++)
            {
                const float* data_chan = data + z * in_cstep;
                float* out_chan = out + z * out_cstep;
                for (int y = 0; y < index_blob.h; y++)
                {
                    const float* data_row = data_chan + y * dw;
                    float* out_row = out_chan + y * top_blob.w;
                    for (int x = 0; x < idxw; x++)
                    {
                        int gi = READ_IDX((int)(z * idx_cstep) + y * idxw + x);
                        CLAMP_IDX(gi);
                        out_row[x] = data_row[gi];
                    }
                }
            }
        }
    }

#undef READ_IDX
#undef CLAMP_IDX

    return 0;
}

} // namespace ncnn
