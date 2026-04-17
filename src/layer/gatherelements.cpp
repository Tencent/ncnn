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

#define CLAMP_IDX(gi)                                        \
    do                                                       \
    {                                                        \
        if ((gi) < 0) (gi) += axis_dim_size;                 \
        if ((gi) < 0) (gi) = 0;                              \
        if ((gi) >= axis_dim_size) (gi) = axis_dim_size - 1; \
    } while (0)

    // use_i32: branch hoisted once per forward() call, not per element
    const bool use_i32 = (idx_elemsize == 4);

    if (data_dims == 1)
    {
        if (use_i32)
        {
            int x = 0;
            for (; x + 4 <= index_blob.w; x += 4)
            {
                int gi0 = idx_ptr32[x];   CLAMP_IDX(gi0);
                int gi1 = idx_ptr32[x+1]; CLAMP_IDX(gi1);
                int gi2 = idx_ptr32[x+2]; CLAMP_IDX(gi2);
                int gi3 = idx_ptr32[x+3]; CLAMP_IDX(gi3);
                out[x]   = data[gi0];
                out[x+1] = data[gi1];
                out[x+2] = data[gi2];
                out[x+3] = data[gi3];
            }
            for (; x < index_blob.w; x++)
            {
                int gi = idx_ptr32[x]; CLAMP_IDX(gi); out[x] = data[gi];
            }
        }
        else
        {
            int x = 0;
            for (; x + 4 <= index_blob.w; x += 4)
            {
                int gi0 = (int)idx_ptr64[x];   CLAMP_IDX(gi0);
                int gi1 = (int)idx_ptr64[x+1]; CLAMP_IDX(gi1);
                int gi2 = (int)idx_ptr64[x+2]; CLAMP_IDX(gi2);
                int gi3 = (int)idx_ptr64[x+3]; CLAMP_IDX(gi3);
                out[x]   = data[gi0];
                out[x+1] = data[gi1];
                out[x+2] = data[gi2];
                out[x+3] = data[gi3];
            }
            for (; x < index_blob.w; x++)
            {
                int gi = (int)idx_ptr64[x]; CLAMP_IDX(gi); out[x] = data[gi];
            }
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
                if (use_i32)
                {
                    const int* ir = idx_ptr32 + y * idxw;
                    int x = 0;
                    for (; x + 4 <= idxw; x += 4)
                    {
                        int gi0 = ir[x];   CLAMP_IDX(gi0);
                        int gi1 = ir[x+1]; CLAMP_IDX(gi1);
                        int gi2 = ir[x+2]; CLAMP_IDX(gi2);
                        int gi3 = ir[x+3]; CLAMP_IDX(gi3);
                        out_row[x]   = data[gi0 * dw + x];
                        out_row[x+1] = data[gi1 * dw + x+1];
                        out_row[x+2] = data[gi2 * dw + x+2];
                        out_row[x+3] = data[gi3 * dw + x+3];
                    }
                    for (; x < idxw; x++)
                    {
                        int gi = ir[x]; CLAMP_IDX(gi); out_row[x] = data[gi * dw + x];
                    }
                }
                else
                {
                    const int64_t* ir = idx_ptr64 + y * idxw;
                    int x = 0;
                    for (; x + 4 <= idxw; x += 4)
                    {
                        int gi0 = (int)ir[x];   CLAMP_IDX(gi0);
                        int gi1 = (int)ir[x+1]; CLAMP_IDX(gi1);
                        int gi2 = (int)ir[x+2]; CLAMP_IDX(gi2);
                        int gi3 = (int)ir[x+3]; CLAMP_IDX(gi3);
                        out_row[x]   = data[gi0 * dw + x];
                        out_row[x+1] = data[gi1 * dw + x+1];
                        out_row[x+2] = data[gi2 * dw + x+2];
                        out_row[x+3] = data[gi3 * dw + x+3];
                    }
                    for (; x < idxw; x++)
                    {
                        int gi = (int)ir[x]; CLAMP_IDX(gi); out_row[x] = data[gi * dw + x];
                    }
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
                if (use_i32)
                {
                    const int* ir = idx_ptr32 + y * idxw;
                    int x = 0;
                    for (; x + 4 <= idxw; x += 4)
                    {
                        int gi0 = ir[x];   CLAMP_IDX(gi0);
                        int gi1 = ir[x+1]; CLAMP_IDX(gi1);
                        int gi2 = ir[x+2]; CLAMP_IDX(gi2);
                        int gi3 = ir[x+3]; CLAMP_IDX(gi3);
                        out_row[x]   = data_row[gi0];
                        out_row[x+1] = data_row[gi1];
                        out_row[x+2] = data_row[gi2];
                        out_row[x+3] = data_row[gi3];
                    }
                    for (; x < idxw; x++)
                    {
                        int gi = ir[x]; CLAMP_IDX(gi); out_row[x] = data_row[gi];
                    }
                }
                else
                {
                    const int64_t* ir = idx_ptr64 + y * idxw;
                    int x = 0;
                    for (; x + 4 <= idxw; x += 4)
                    {
                        int gi0 = (int)ir[x];   CLAMP_IDX(gi0);
                        int gi1 = (int)ir[x+1]; CLAMP_IDX(gi1);
                        int gi2 = (int)ir[x+2]; CLAMP_IDX(gi2);
                        int gi3 = (int)ir[x+3]; CLAMP_IDX(gi3);
                        out_row[x]   = data_row[gi0];
                        out_row[x+1] = data_row[gi1];
                        out_row[x+2] = data_row[gi2];
                        out_row[x+3] = data_row[gi3];
                    }
                    for (; x < idxw; x++)
                    {
                        int gi = (int)ir[x]; CLAMP_IDX(gi); out_row[x] = data_row[gi];
                    }
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
                const int idx_z_base = (int)(z * idx_cstep);
                if (use_i32)
                {
                    for (int y = 0; y < index_blob.h; y++)
                    {
                        float* out_row = out_chan + y * top_blob.w;
                        const int* ir = idx_ptr32 + idx_z_base + y * idxw;
                        const int inp_y_off = y * dw;
                        int x = 0;
                        for (; x + 4 <= idxw; x += 4)
                        {
                            int gi0 = ir[x];   CLAMP_IDX(gi0);
                            int gi1 = ir[x+1]; CLAMP_IDX(gi1);
                            int gi2 = ir[x+2]; CLAMP_IDX(gi2);
                            int gi3 = ir[x+3]; CLAMP_IDX(gi3);
                            out_row[x]   = data[(int)(gi0 * in_cstep) + inp_y_off + x];
                            out_row[x+1] = data[(int)(gi1 * in_cstep) + inp_y_off + x+1];
                            out_row[x+2] = data[(int)(gi2 * in_cstep) + inp_y_off + x+2];
                            out_row[x+3] = data[(int)(gi3 * in_cstep) + inp_y_off + x+3];
                        }
                        for (; x < idxw; x++)
                        {
                            int gi = ir[x]; CLAMP_IDX(gi);
                            out_row[x] = data[(int)(gi * in_cstep) + inp_y_off + x];
                        }
                    }
                }
                else
                {
                    for (int y = 0; y < index_blob.h; y++)
                    {
                        float* out_row = out_chan + y * top_blob.w;
                        const int64_t* ir = idx_ptr64 + idx_z_base + y * idxw;
                        const int inp_y_off = y * dw;
                        int x = 0;
                        for (; x + 4 <= idxw; x += 4)
                        {
                            int gi0 = (int)ir[x];   CLAMP_IDX(gi0);
                            int gi1 = (int)ir[x+1]; CLAMP_IDX(gi1);
                            int gi2 = (int)ir[x+2]; CLAMP_IDX(gi2);
                            int gi3 = (int)ir[x+3]; CLAMP_IDX(gi3);
                            out_row[x]   = data[(int)(gi0 * in_cstep) + inp_y_off + x];
                            out_row[x+1] = data[(int)(gi1 * in_cstep) + inp_y_off + x+1];
                            out_row[x+2] = data[(int)(gi2 * in_cstep) + inp_y_off + x+2];
                            out_row[x+3] = data[(int)(gi3 * in_cstep) + inp_y_off + x+3];
                        }
                        for (; x < idxw; x++)
                        {
                            int gi = (int)ir[x]; CLAMP_IDX(gi);
                            out_row[x] = data[(int)(gi * in_cstep) + inp_y_off + x];
                        }
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
                const int idx_z_base = (int)(z * idx_cstep);
                if (use_i32)
                {
                    for (int y = 0; y < index_blob.h; y++)
                    {
                        float* out_row = out_chan + y * top_blob.w;
                        const int* ir = idx_ptr32 + idx_z_base + y * idxw;
                        int x = 0;
                        for (; x + 4 <= idxw; x += 4)
                        {
                            int gi0 = ir[x];   CLAMP_IDX(gi0);
                            int gi1 = ir[x+1]; CLAMP_IDX(gi1);
                            int gi2 = ir[x+2]; CLAMP_IDX(gi2);
                            int gi3 = ir[x+3]; CLAMP_IDX(gi3);
                            out_row[x]   = data_chan[gi0 * dw + x];
                            out_row[x+1] = data_chan[gi1 * dw + x+1];
                            out_row[x+2] = data_chan[gi2 * dw + x+2];
                            out_row[x+3] = data_chan[gi3 * dw + x+3];
                        }
                        for (; x < idxw; x++)
                        {
                            int gi = ir[x]; CLAMP_IDX(gi);
                            out_row[x] = data_chan[gi * dw + x];
                        }
                    }
                }
                else
                {
                    for (int y = 0; y < index_blob.h; y++)
                    {
                        float* out_row = out_chan + y * top_blob.w;
                        const int64_t* ir = idx_ptr64 + idx_z_base + y * idxw;
                        int x = 0;
                        for (; x + 4 <= idxw; x += 4)
                        {
                            int gi0 = (int)ir[x];   CLAMP_IDX(gi0);
                            int gi1 = (int)ir[x+1]; CLAMP_IDX(gi1);
                            int gi2 = (int)ir[x+2]; CLAMP_IDX(gi2);
                            int gi3 = (int)ir[x+3]; CLAMP_IDX(gi3);
                            out_row[x]   = data_chan[gi0 * dw + x];
                            out_row[x+1] = data_chan[gi1 * dw + x+1];
                            out_row[x+2] = data_chan[gi2 * dw + x+2];
                            out_row[x+3] = data_chan[gi3 * dw + x+3];
                        }
                        for (; x < idxw; x++)
                        {
                            int gi = (int)ir[x]; CLAMP_IDX(gi);
                            out_row[x] = data_chan[gi * dw + x];
                        }
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
                const int idx_z_base = (int)(z * idx_cstep);
                if (use_i32)
                {
                    for (int y = 0; y < index_blob.h; y++)
                    {
                        const float* data_row = data_chan + y * dw;
                        float* out_row = out_chan + y * top_blob.w;
                        const int* ir = idx_ptr32 + idx_z_base + y * idxw;
                        int x = 0;
                        for (; x + 4 <= idxw; x += 4)
                        {
                            int gi0 = ir[x];   CLAMP_IDX(gi0);
                            int gi1 = ir[x+1]; CLAMP_IDX(gi1);
                            int gi2 = ir[x+2]; CLAMP_IDX(gi2);
                            int gi3 = ir[x+3]; CLAMP_IDX(gi3);
                            out_row[x]   = data_row[gi0];
                            out_row[x+1] = data_row[gi1];
                            out_row[x+2] = data_row[gi2];
                            out_row[x+3] = data_row[gi3];
                        }
                        for (; x < idxw; x++)
                        {
                            int gi = ir[x]; CLAMP_IDX(gi); out_row[x] = data_row[gi];
                        }
                    }
                }
                else
                {
                    for (int y = 0; y < index_blob.h; y++)
                    {
                        const float* data_row = data_chan + y * dw;
                        float* out_row = out_chan + y * top_blob.w;
                        const int64_t* ir = idx_ptr64 + idx_z_base + y * idxw;
                        int x = 0;
                        for (; x + 4 <= idxw; x += 4)
                        {
                            int gi0 = (int)ir[x];   CLAMP_IDX(gi0);
                            int gi1 = (int)ir[x+1]; CLAMP_IDX(gi1);
                            int gi2 = (int)ir[x+2]; CLAMP_IDX(gi2);
                            int gi3 = (int)ir[x+3]; CLAMP_IDX(gi3);
                            out_row[x]   = data_row[gi0];
                            out_row[x+1] = data_row[gi1];
                            out_row[x+2] = data_row[gi2];
                            out_row[x+3] = data_row[gi3];
                        }
                        for (; x < idxw; x++)
                        {
                            int gi = (int)ir[x]; CLAMP_IDX(gi); out_row[x] = data_row[gi];
                        }
                    }
                }
            }
        }
    }

#undef CLAMP_IDX

    return 0;
}

} // namespace ncnn
