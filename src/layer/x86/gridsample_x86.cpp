// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

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
    }
    else
    {
        grid_p1 = grid;
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

            if (padding_mode == GridSample::Padding_ZEROS)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_ZEROS, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_ZEROS, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_BORDER)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_BORDER, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_BORDER, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_REFLECTION)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_REFLECTION, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_REFLECTION, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
            }
            else
            {
                NCNN_LOGE("gridsample padding_mode error\n");
                return -100;
            }
        }

        if (sample_type == GridSample::Interpolation_NEAREST)
        {
            offset_value_blob.create(outw, outh, 1, elemsize, 1, opt.workspace_allocator);
            if (offset_value_blob.empty())
                return -100;

            if (padding_mode == GridSample::Padding_ZEROS)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_ZEROS, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_ZEROS, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_BORDER)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_BORDER, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_BORDER, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_REFLECTION)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_REFLECTION, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_REFLECTION, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
            }
            else
            {
                NCNN_LOGE("gridsample padding_mode error\n");
                return -100;
            }
        }

        if (sample_type == GridSample::Interpolation_BICUBIC)
        {
            offset_value_blob.create(outw, outh, elemsize * 18, 18, opt.workspace_allocator);
            if (offset_value_blob.empty())
                return -100;

            if (padding_mode == GridSample::Padding_ZEROS)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_ZEROS, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_ZEROS, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_BORDER)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_BORDER, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_BORDER, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_REFLECTION)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_REFLECTION, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_REFLECTION, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
            }
            else
            {
                NCNN_LOGE("gridsample padding_mode error\n");
                return -100;
            }
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

            if (padding_mode == GridSample::Padding_ZEROS)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_ZEROS, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_ZEROS, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_BORDER)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_BORDER, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_BORDER, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_REFLECTION)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_REFLECTION, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_REFLECTION, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
            }
            else
            {
                NCNN_LOGE("gridsample padding_mode error\n");
                return -100;
            }
        }

        if (sample_type == GridSample::Interpolation_NEAREST)
        {
            offset_value_blob.create(outw, outh, outd, 1, elemsize, 1, opt.workspace_allocator);
            if (offset_value_blob.empty())
                return -100;

            if (padding_mode == GridSample::Padding_ZEROS)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_ZEROS, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_ZEROS, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_BORDER)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_BORDER, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_BORDER, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_REFLECTION)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_REFLECTION, false>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_REFLECTION, true>(bottom_blob, grid_p1, offset_value_blob, permute_fusion);
                }
            }
            else
            {
                NCNN_LOGE("gridsample padding_mode error\n");
                return -100;
            }
        }

        if (sample_type == 3)
        {
            NCNN_LOGE("unsupported bicubic when dims == 4");
            return -100;
        }
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        if (dims == 3)
        {
            if (sample_type == GridSample::Interpolation_BILINEAR)
            {
                gridsample_2d_bilinear_apply_interpolation_p16(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_NEAREST)
            {
                gridsample_nearest_apply_interpolation_p16(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_BICUBIC)
            {
                gridsample_2d_bicubic_apply_interpolation_p16(bottom_blob, top_blob, offset_value_blob, opt);
            }
        }
        else if (dims == 4)
        {
            if (sample_type == GridSample::Interpolation_BILINEAR)
            {
                gridsample_3d_bilinear_apply_interpolation_p16(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_NEAREST)
            {
                gridsample_nearest_apply_interpolation_p16(bottom_blob, top_blob, offset_value_blob, opt);
            }
        }
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        if (dims == 3)
        {
            if (sample_type == GridSample::Interpolation_BILINEAR)
            {
                gridsample_2d_bilinear_apply_interpolation_p8(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_NEAREST)
            {
                gridsample_nearest_apply_interpolation_p8(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_BICUBIC)
            {
                gridsample_2d_bicubic_apply_interpolation_p8(bottom_blob, top_blob, offset_value_blob, opt);
            }
        }
        else if (dims == 4)
        {
            if (sample_type == GridSample::Interpolation_BILINEAR)
            {
                gridsample_3d_bilinear_apply_interpolation_p8(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_NEAREST)
            {
                gridsample_nearest_apply_interpolation_p8(bottom_blob, top_blob, offset_value_blob, opt);
            }
        }
    }

#endif // __AVX__
    if (elempack == 4)
    {
        if (dims == 3)
        {
            if (sample_type == GridSample::Interpolation_BILINEAR)
            {
                gridsample_2d_bilinear_apply_interpolation_p4(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_NEAREST)
            {
                gridsample_nearest_apply_interpolation_p4(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_BICUBIC)
            {
                gridsample_2d_bicubic_apply_interpolation_p4(bottom_blob, top_blob, offset_value_blob, opt);
            }
        }
        else if (dims == 4)
        {
            if (sample_type == GridSample::Interpolation_BILINEAR)
            {
                gridsample_3d_bilinear_apply_interpolation_p4(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_NEAREST)
            {
                gridsample_nearest_apply_interpolation_p4(bottom_blob, top_blob, offset_value_blob, opt);
            }
        }
    }

#endif // __SSE2__

    if (elempack == 1)
    {
        if (dims == 3)
        {
            if (sample_type == GridSample::Interpolation_BILINEAR)
            {
                gridsample_2d_bilinear_apply_interpolation_p1(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_NEAREST)
            {
                gridsample_nearest_apply_interpolation_p1(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_BICUBIC)
            {
                gridsample_2d_bicubic_apply_interpolation_p1(bottom_blob, top_blob, offset_value_blob, opt);
            }
        }
        else if (dims == 4)
        {
            if (sample_type == GridSample::Interpolation_BILINEAR)
            {
                gridsample_3d_bilinear_apply_interpolation_p1(bottom_blob, top_blob, offset_value_blob, opt);
            }
            else if (sample_type == GridSample::Interpolation_NEAREST)
            {
                gridsample_nearest_apply_interpolation_p1(bottom_blob, top_blob, offset_value_blob, opt);
            }
        }
    }

    return 0;
}

} // namespace ncnn
