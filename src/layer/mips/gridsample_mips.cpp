// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gridsample_mips.h"

#include <math.h>

#if __mips_msa
#include "mips_usability.h"
#endif // __mips_msa

namespace ncnn {

GridSample_mips::GridSample_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
}

#include "gridsample_compute_blob.h"

#include "gridsample_apply_interpolation.h"

int GridSample_mips::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& grid = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];
    int elempack = bottom_blob.elempack;

    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    int outw, outh, outd;
    Mat offset_value_blob;

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

    if (padding_mode != GridSample::Padding_ZEROS && padding_mode != GridSample::Padding_BORDER && padding_mode != GridSample::Padding_REFLECTION)
    {
        NCNN_LOGE("gridsample padding_mode error\n");
        return -100;
    }

    if (dims == 3)
    {
        outw = permute_fusion == 0 ? grid_p1.h : grid_p1.w;
        outh = permute_fusion == 0 ? grid_p1.c : grid_p1.h;

        top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (sample_type == GridSample::Interpolation_BILINEAR)
        {
            offset_value_blob.create(outw, outh, elemsize * 6, 6, opt.workspace_allocator);
            if (offset_value_blob.empty())
                return -100;

            gridsample_2d_bilinear_compute_blob_mips(bottom_blob, grid_p1, offset_value_blob, padding_mode, align_corner, permute_fusion);
        }

        if (sample_type == GridSample::Interpolation_NEAREST)
        {
            offset_value_blob.create(outw, outh, 1, elemsize, 1, opt.workspace_allocator);
            if (offset_value_blob.empty())
                return -100;

            gridsample_2d_nearest_compute_blob_mips(bottom_blob, grid_p1, offset_value_blob, padding_mode, align_corner, permute_fusion);
        }

        if (sample_type == GridSample::Interpolation_BICUBIC)
        {
            offset_value_blob.create(outw, outh, elemsize * 18, 18, opt.workspace_allocator);
            if (offset_value_blob.empty())
                return -100;

            gridsample_2d_bicubic_compute_blob_mips(bottom_blob, grid_p1, offset_value_blob, padding_mode, align_corner, permute_fusion);
        }
    }

    if (dims == 4)
    {
        outw = permute_fusion == 0 ? grid_p1.h : grid_p1.w;
        outh = permute_fusion == 0 ? grid_p1.d : grid_p1.h;
        outd = permute_fusion == 0 ? grid_p1.c : grid_p1.d;

        top_blob.create(outw, outh, outd, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (sample_type == GridSample::Interpolation_BILINEAR)
        {
            offset_value_blob.create(outw, outh, outd, elemsize * 11, 11, opt.workspace_allocator);
            if (offset_value_blob.empty())
                return -100;

            gridsample_3d_bilinear_compute_blob_mips(bottom_blob, grid_p1, offset_value_blob, padding_mode, align_corner, permute_fusion);
        }

        if (sample_type == GridSample::Interpolation_NEAREST)
        {
            offset_value_blob.create(outw, outh, outd, 1, elemsize, 1, opt.workspace_allocator);
            if (offset_value_blob.empty())
                return -100;

            gridsample_3d_nearest_compute_blob_mips(bottom_blob, grid_p1, offset_value_blob, padding_mode, align_corner, permute_fusion);
        }

        if (sample_type == GridSample::Interpolation_BICUBIC)
        {
            NCNN_LOGE("unsupported bicubic when dims == 4");
            return -100;
        }
    }

    if (dims != 3 && dims != 4)
        return -100;

#if __mips_msa
    if (elempack == 4)
    {
        if (dims == 3)
        {
            if (sample_type == GridSample::Interpolation_BILINEAR)
            {
                gridsample_2d_bilinear_apply_interpolation_pack4_msa(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_NEAREST)
            {
                gridsample_nearest_apply_interpolation_pack4_msa(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_BICUBIC)
            {
                gridsample_2d_bicubic_apply_interpolation_pack4_msa(bottom_blob, top_blob, offset_value_blob, opt);
            }
        }
        else if (dims == 4)
        {
            if (sample_type == GridSample::Interpolation_BILINEAR)
            {
                gridsample_3d_bilinear_apply_interpolation_pack4_msa(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_NEAREST)
            {
                gridsample_nearest_apply_interpolation_pack4_msa(bottom_blob, top_blob, offset_value_blob, opt);
            }
        }
    }
#endif // __mips_msa

    if (elempack == 1)
    {
        if (dims == 3)
        {
            if (sample_type == GridSample::Interpolation_BILINEAR)
            {
                gridsample_2d_bilinear_apply_interpolation_p1_mips(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_NEAREST)
            {
                gridsample_nearest_apply_interpolation_p1_mips(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_BICUBIC)
            {
                gridsample_2d_bicubic_apply_interpolation_p1_mips(bottom_blob, top_blob, offset_value_blob, opt);
            }
        }
        else if (dims == 4)
        {
            if (sample_type == GridSample::Interpolation_BILINEAR)
            {
                gridsample_3d_bilinear_apply_interpolation_p1_mips(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_NEAREST)
            {
                gridsample_nearest_apply_interpolation_p1_mips(bottom_blob, top_blob, offset_value_blob, opt);
            }
        }
    }

    return 0;
}

} // namespace ncnn
