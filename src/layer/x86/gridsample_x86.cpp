// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gridsample_x86.h"

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"
#include "cpu.h"

namespace ncnn {

#include "gridsample_compute_blob.h"
#include "gridsample_bilinear_apply_interpolation.h"
#include "gridsample_bicubic_apply_interpolation.h"
#include "gridsample_nearest_apply_interpolation.h"
#include "gridsample_fp32.h"

GridSample_x86::GridSample_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int GridSample_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& grid = bottom_blobs[1];

    Mat grid_unpacked;
    if (grid.elempack != 1)
    {
        convert_packing(grid, grid_unpacked, 1, opt);
        if (grid_unpacked.empty())
            return -100;
    }
    else
    {
        grid_unpacked = grid;
    }

    Mat& top_blob = top_blobs[0];
    if (bottom_blob.dims == 3)
    {
        const int outw = permute_fusion == 0 ? grid_unpacked.h : grid_unpacked.w;
        const int outh = permute_fusion == 0 ? grid_unpacked.c : grid_unpacked.h;

        top_blob.create(outw, outh, bottom_blob.c, bottom_blob.elemsize, bottom_blob.elempack, opt.blob_allocator);
    }
    else // bottom_blob.dims == 4
    {
        const int outw = permute_fusion == 0 ? grid_unpacked.h : grid_unpacked.w;
        const int outh = permute_fusion == 0 ? grid_unpacked.d : grid_unpacked.h;
        const int outd = permute_fusion == 0 ? grid_unpacked.c : grid_unpacked.d;

        top_blob.create(outw, outh, outd, bottom_blob.c, bottom_blob.elemsize, bottom_blob.elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    return gridsample_fp32(bottom_blob, grid_unpacked, top_blob, sample_type, padding_mode, align_corner, permute_fusion, opt);
}

} // namespace ncnn
