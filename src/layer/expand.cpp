// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "expand.h"

#include <algorithm>
#include <string.h>

namespace ncnn {

Expand::Expand()
{
    one_blob_only = false;
    support_inplace = false;
}

int Expand::load_param(const ParamDict& /*pd*/)
{
    return 0;
}

int Expand::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (bottom_blobs.size() < 2)
        return -1;

    const Mat& input_blob = bottom_blobs[0];
    const Mat& shape_blob = bottom_blobs[1];

    // shape_blob: 1D tensor of int32 or int64 in ncnn ordering (w, h, c)
    const size_t shape_elemsize = shape_blob.elemsize / shape_blob.elempack;
    const bool shape_is_int64 = (shape_elemsize == 8);
    int target_dims = (shape_blob.dims == 1) ? shape_blob.w : (int)shape_blob.total();
    if (target_dims > 3) target_dims = 3;

    // Input shape in ncnn ordering: index 0=w (innermost), 1=h, 2=c (outermost)
    const int in_dims = input_blob.dims;
    int in_w = input_blob.w;
    int in_h = (in_dims >= 2) ? input_blob.h : 1;
    int in_c = (in_dims >= 3) ? input_blob.c : 1;

    // Read target shape from shape_blob (ncnn ordering)
    int tgt_w = 1, tgt_h = 1, tgt_c = 1;
    auto read_shape_dim = [&](int idx) -> int {
        if (idx < 0 || idx >= target_dims) return 1;
        if (shape_is_int64) return (int)((const int64_t*)(const void*)shape_blob)[idx];
        return ((const int*)(const void*)shape_blob)[idx];
    };
    if (target_dims >= 1) tgt_w = read_shape_dim(0);
    if (target_dims >= 2) tgt_h = read_shape_dim(1);
    if (target_dims >= 3) tgt_c = read_shape_dim(2);

    // Resolve broadcast: -1 means keep input dim; 1 means broadcast
    auto resolve_dim = [](int in_dim, int tgt_dim) -> int {
        if (tgt_dim <= 0) return in_dim; // -1 or 0: keep
        if (in_dim == 1) return tgt_dim;
        return in_dim; // tgt==1 or tgt==in_dim: keep in_dim
    };

    const int out_w = resolve_dim(in_w, tgt_w);
    const int out_h = resolve_dim(in_h, tgt_h);
    const int out_c = resolve_dim(in_c, tgt_c);
    const int out_dims = std::max(in_dims, target_dims);

    // Validate: if neither is 1 and they differ, it's invalid
    if ((in_w != 1 && tgt_w != 1 && tgt_w > 0 && in_w != tgt_w) || (in_h != 1 && tgt_h != 1 && tgt_h > 0 && in_h != tgt_h) || (in_c != 1 && tgt_c != 1 && tgt_c > 0 && in_c != tgt_c))
        return -1;

    Mat& top_blob = top_blobs[0];
    if (out_dims == 1)
        top_blob.create(out_w, input_blob.elemsize, input_blob.elempack, opt.blob_allocator);
    else if (out_dims == 2)
        top_blob.create(out_w, out_h, input_blob.elemsize, input_blob.elempack, opt.blob_allocator);
    else
        top_blob.create(out_w, out_h, out_c, input_blob.elemsize, input_blob.elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* inp = input_blob;
    float* out = top_blob;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int z = 0; z < out_c; z++)
    {
        int sz = (in_c > 1) ? z : 0;
        const float* src_chan = inp + sz * (int)input_blob.cstep;
        float* dst_chan = out + z * (int)top_blob.cstep;

        for (int y = 0; y < out_h; y++)
        {
            int sy = (in_h > 1) ? y : 0;
            const float* src_row = src_chan + sy * in_w;
            float* dst_row = dst_chan + y * out_w;

            if (in_w == out_w)
            {
                memcpy(dst_row, src_row, out_w * sizeof(float));
            }
            else // in_w == 1: broadcast scalar across row
            {
                const float val = src_row[0];
                for (int x = 0; x < out_w; x++)
                    dst_row[x] = val;
            }
        }
    }

    return 0;
}

} // namespace ncnn
