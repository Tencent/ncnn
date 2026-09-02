// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
int gridsample_fp32_fma(const Mat& bottom_blob, const Mat& grid, Mat& top_blob, int sample_type, int padding_mode, int align_corner, int permute_fusion, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
int gridsample_fp32_fma4(const Mat& bottom_blob, const Mat& grid, Mat& top_blob, int sample_type, int padding_mode, int align_corner, int permute_fusion, const Option& opt);
#endif

static int gridsample_fp32(const Mat& bottom_blob, const Mat& grid, Mat& top_blob, int sample_type, int padding_mode, int align_corner, int permute_fusion, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
        return gridsample_fp32_fma(bottom_blob, grid, top_blob, sample_type, padding_mode, align_corner, permute_fusion, opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
        return gridsample_fp32_fma4(bottom_blob, grid, top_blob, sample_type, padding_mode, align_corner, permute_fusion, opt);
#endif
    int elempack = bottom_blob.elempack;

    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outd = top_blob.d;
    Mat offset_value_blob;

    if (dims == 3)
    {
        if (sample_type == GridSample::Interpolation_BILINEAR)
        {
            offset_value_blob.create(outw, outh, elemsize * 6, 6, opt.workspace_allocator);
            if (offset_value_blob.empty())
                return -100;

            if (padding_mode == GridSample::Padding_ZEROS)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_ZEROS, false>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_ZEROS, true>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_BORDER)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_BORDER, false>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_BORDER, true>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_REFLECTION)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_REFLECTION, false>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_bilinear_compute_blob<GridSample::Padding_REFLECTION, true>(bottom_blob, grid, offset_value_blob, permute_fusion);
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
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_ZEROS, false>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_ZEROS, true>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_BORDER)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_BORDER, false>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_BORDER, true>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_REFLECTION)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_REFLECTION, false>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_nearest_compute_blob<GridSample::Padding_REFLECTION, true>(bottom_blob, grid, offset_value_blob, permute_fusion);
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
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_ZEROS, false>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_ZEROS, true>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_BORDER)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_BORDER, false>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_BORDER, true>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_REFLECTION)
            {
                if (align_corner == 0)
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_REFLECTION, false>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_2d_bicubic_compute_blob<GridSample::Padding_REFLECTION, true>(bottom_blob, grid, offset_value_blob, permute_fusion);
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
        if (sample_type == GridSample::Interpolation_BILINEAR)
        {
            offset_value_blob.create(outw, outh, outd, elemsize * 11, 11, opt.workspace_allocator);
            if (offset_value_blob.empty())
                return -100;

            if (padding_mode == GridSample::Padding_ZEROS)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_ZEROS, false>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_ZEROS, true>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_BORDER)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_BORDER, false>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_BORDER, true>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_REFLECTION)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_REFLECTION, false>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_3d_bilinear_compute_blob<GridSample::Padding_REFLECTION, true>(bottom_blob, grid, offset_value_blob, permute_fusion);
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
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_ZEROS, false>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_ZEROS, true>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_BORDER)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_BORDER, false>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_BORDER, true>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
            }
            else if (padding_mode == GridSample::Padding_REFLECTION)
            {
                if (align_corner == 0)
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_REFLECTION, false>(bottom_blob, grid, offset_value_blob, permute_fusion);
                }
                else
                {
                    gridsample_3d_nearest_compute_blob<GridSample::Padding_REFLECTION, true>(bottom_blob, grid, offset_value_blob, permute_fusion);
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
