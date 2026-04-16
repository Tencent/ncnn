// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gather.h"

#include <stddef.h>
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

    const int64_t* idx_ptr64 = (const int64_t*)(const void*)index_blob;
    const int* idx_ptr32 = (const int*)(const void*)index_blob;

#define READ_IDX(pos) \
    (idx_elemsize == 8 ? (int)idx_ptr64[(pos)] : idx_ptr32[(pos)])

#define CLAMP_IDX(gi)                                        \
    do                                                       \
    {                                                        \
        if ((gi) < 0) (gi) += axis_dim_size;                 \
        if ((gi) < 0) (gi) = 0;                              \
        if ((gi) >= axis_dim_size) (gi) = axis_dim_size - 1; \
    } while (0)

    if (dims == 1)
    {
        // axis=0 only: output[x] = input[index[x]]
        for (int x = 0; x < index_blob.w; x++)
        {
            int gi = READ_IDX(x);
            CLAMP_IDX(gi);
            out[x] = inp[gi];
        }
    }
    else if (dims == 2)
    {
        // PyTorch axis=0 -> h (outer): output[y,x] = input[index[y,x], x]
        // PyTorch axis=1 -> w (inner): output[y,x] = input[y, index[y,x]]
        const int iw = input_blob.w;
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
                    out_row[x] = inp[gi * iw + x];
                }
            }
        }
        else // positive_axis == 1
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < index_blob.h; y++)
            {
                const float* inp_row = inp + y * iw;
                float* out_row = out + y * top_blob.w;
                for (int x = 0; x < idxw; x++)
                {
                    int gi = READ_IDX(y * idxw + x);
                    CLAMP_IDX(gi);
                    out_row[x] = inp_row[gi];
                }
            }
        }
    }
    else // dims == 3
    {
        // PyTorch axis=0 -> c (outer): output[z,y,x] = input[index[z,y,x], y, x]
        // PyTorch axis=1 -> h:          output[z,y,x] = input[z, index[z,y,x], x]
        // PyTorch axis=2 -> w (inner):  output[z,y,x] = input[z, y, index[z,y,x]]
        const int iw = input_blob.w;
        const size_t in_cstep = input_blob.cstep;
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
                        int gi = READ_IDX(z * idx_cstep + y * idxw + x);
                        CLAMP_IDX(gi);
                        out_row[x] = inp[gi * in_cstep + y * iw + x];
                    }
                }
            }
        }
        else if (positive_axis == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int z = 0; z < index_blob.c; z++)
            {
                const float* inp_chan = inp + z * in_cstep;
                float* out_chan = out + z * out_cstep;
                for (int y = 0; y < index_blob.h; y++)
                {
                    float* out_row = out_chan + y * top_blob.w;
                    for (int x = 0; x < idxw; x++)
                    {
                        int gi = READ_IDX(z * idx_cstep + y * idxw + x);
                        CLAMP_IDX(gi);
                        out_row[x] = inp_chan[gi * iw + x];
                    }
                }
            }
        }
        else // positive_axis == 2
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int z = 0; z < index_blob.c; z++)
            {
                const float* inp_chan = inp + z * in_cstep;
                float* out_chan = out + z * out_cstep;
                for (int y = 0; y < index_blob.h; y++)
                {
                    const float* inp_row = inp_chan + y * iw;
                    float* out_row = out_chan + y * top_blob.w;
                    for (int x = 0; x < idxw; x++)
                    {
                        int gi = READ_IDX(z * idx_cstep + y * idxw + x);
                        CLAMP_IDX(gi);
                        out_row[x] = inp_row[gi];
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
