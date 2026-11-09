// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gridsample_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

GridSample_arm::GridSample_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

#include "gridsample_compute_blob.h"

#include "gridsample_apply_interpolation.h"

int GridSample_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& grid = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];
    const int elempack = bottom_blob.elempack;

    const int channels = bottom_blob.c;
    const int dims = bottom_blob.dims;
    const size_t elemsize = bottom_blob.elemsize;

    if (dims != 3 && dims != 4)
        return -100;

    if (padding_mode != GridSample::Padding_ZEROS && padding_mode != GridSample::Padding_BORDER && padding_mode != GridSample::Padding_REFLECTION)
    {
        NCNN_LOGE("gridsample padding_mode error\n");
        return -100;
    }

    if (dims == 4 && sample_type == GridSample::Interpolation_BICUBIC)
    {
        NCNN_LOGE("unsupported bicubic when dims == 4");
        return -100;
    }

    Mat grid_p1;
    if (grid.elempack != 1)
    {
        convert_packing(grid, grid_p1, 1, opt);
        if (grid_p1.empty())
            return -100;
    }
    else
    {
        grid_p1 = grid;
    }

    // the offset value blob stores per output point
    //   bilinear 2d: 4 int offsets + 2 float weights
    //   bilinear 3d: 8 int offsets + 3 float weights
    //   bicubic 2d: 2 float weights + 16 int offsets
    //   nearest: 1 int offset
    Mat offset_value_blob;

    if (dims == 3)
    {
        const int outw = permute_fusion == 0 ? grid_p1.h : grid_p1.w;
        const int outh = permute_fusion == 0 ? grid_p1.c : grid_p1.h;

        top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (sample_type == GridSample::Interpolation_BILINEAR)
        {
            offset_value_blob.create(outw, outh, 6 * sizeof(float), 6, opt.workspace_allocator);
            if (offset_value_blob.empty())
                return -100;

            gridsample_2d_bilinear_compute_blob_arm(bottom_blob, grid_p1, offset_value_blob, padding_mode, align_corner, permute_fusion);
        }
        else if (sample_type == GridSample::Interpolation_NEAREST)
        {
            offset_value_blob.create(outw, outh, 1, sizeof(int), 1, opt.workspace_allocator);
            if (offset_value_blob.empty())
                return -100;

            gridsample_2d_nearest_compute_blob_arm(bottom_blob, grid_p1, offset_value_blob, padding_mode, align_corner, permute_fusion);
        }
        else // if (sample_type == GridSample::Interpolation_BICUBIC)
        {
            offset_value_blob.create(outw, outh, 18 * sizeof(float), 18, opt.workspace_allocator);
            if (offset_value_blob.empty())
                return -100;

            gridsample_2d_bicubic_compute_blob_arm(bottom_blob, grid_p1, offset_value_blob, padding_mode, align_corner, permute_fusion);
        }
    }
    else // if (dims == 4)
    {
        const int outw = permute_fusion == 0 ? grid_p1.h : grid_p1.w;
        const int outh = permute_fusion == 0 ? grid_p1.d : grid_p1.h;
        const int outd = permute_fusion == 0 ? grid_p1.c : grid_p1.d;

        top_blob.create(outw, outh, outd, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (sample_type == GridSample::Interpolation_BILINEAR)
        {
            offset_value_blob.create(outw, outh, outd, 11 * sizeof(float), 11, opt.workspace_allocator);
            if (offset_value_blob.empty())
                return -100;

            gridsample_3d_bilinear_compute_blob_arm(bottom_blob, grid_p1, offset_value_blob, padding_mode, align_corner, permute_fusion);
        }
        else // if (sample_type == GridSample::Interpolation_NEAREST)
        {
            offset_value_blob.create(outw, outh, outd, 1, sizeof(int), 1, opt.workspace_allocator);
            if (offset_value_blob.empty())
                return -100;

            gridsample_3d_nearest_compute_blob_arm(bottom_blob, grid_p1, offset_value_blob, padding_mode, align_corner, permute_fusion);
        }
    }

#if __ARM_NEON
    if (elempack == 4)
    {
        if (sample_type == GridSample::Interpolation_BILINEAR)
        {
            if (dims == 3)
                gridsample_2d_bilinear_apply_interpolation_pack4_neon(bottom_blob, top_blob, offset_value_blob, opt);
            else
                gridsample_3d_bilinear_apply_interpolation_pack4_neon(bottom_blob, top_blob, offset_value_blob, opt);
        }
        else if (sample_type == GridSample::Interpolation_NEAREST)
        {
            gridsample_nearest_apply_interpolation_pack4_neon(bottom_blob, top_blob, offset_value_blob, opt);
        }
        else // if (sample_type == GridSample::Interpolation_BICUBIC)
        {
            gridsample_bicubic_apply_interpolation_pack4_neon(bottom_blob, top_blob, offset_value_blob, opt);
        }

        return 0;
    }
#endif // __ARM_NEON

    if (sample_type == GridSample::Interpolation_BILINEAR)
    {
        if (dims == 3)
            gridsample_2d_bilinear_apply_interpolation_p1_arm(bottom_blob, top_blob, offset_value_blob, opt);
        else
            gridsample_3d_bilinear_apply_interpolation_p1_arm(bottom_blob, top_blob, offset_value_blob, opt);
    }
    else if (sample_type == GridSample::Interpolation_NEAREST)
    {
        gridsample_nearest_apply_interpolation_p1_arm(bottom_blob, top_blob, offset_value_blob, opt);
    }
    else // if (sample_type == GridSample::Interpolation_BICUBIC)
    {
        gridsample_bicubic_apply_interpolation_p1_arm(bottom_blob, top_blob, offset_value_blob, opt);
    }

    return 0;
}

} // namespace ncnn
