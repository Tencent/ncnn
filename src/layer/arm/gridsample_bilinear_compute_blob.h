// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

template<GridSample::PaddingMode padding_mode, bool align_corner>
static void gridsample_2d_bilinear_compute_blob_arm(const Mat& src, const Mat& grid, Mat& offset_value, int permute_fusion)
{
    const int outw = offset_value.w;
    const int outh = offset_value.h;

    if (permute_fusion == 0)
    {
        for (int y = 0; y < outh; y++)
        {
            const float* gridptr = grid.channel(y);
            float* offset_value_ptr = (float*)offset_value.data + (size_t)y * outw * 6;

            int x = 0;
#if __ARM_NEON
            for (; x + 3 < outw; x += 4)
            {
                float32x4x2_t _xy = vld2q_f32(gridptr);
                float32x4_t _sample_x = grid_sample_unormalize_neon(src.w, _xy.val[0], align_corner);
                float32x4_t _sample_y = grid_sample_unormalize_neon(src.h, _xy.val[1], align_corner);
                _sample_x = compute_coord_neon(_sample_x, src.w, padding_mode, align_corner);
                _sample_y = compute_coord_neon(_sample_y, src.h, padding_mode, align_corner);

                gridsample_store_2d_bilinear_neon(offset_value_ptr, _sample_x, _sample_y, src.w, src.h, src.elempack);

                gridptr += 8;
                offset_value_ptr += 24;
            }
#endif // __ARM_NEON
            for (; x < outw; x++)
            {
                float sample_x = grid_sample_unormalize_arm(src.w, gridptr[0], align_corner);
                float sample_y = grid_sample_unormalize_arm(src.h, gridptr[1], align_corner);
                sample_x = compute_coord_arm(sample_x, src.w, padding_mode, align_corner);
                sample_y = compute_coord_arm(sample_y, src.h, padding_mode, align_corner);

                gridsample_store_2d_bilinear_arm(offset_value_ptr, sample_x, sample_y, src.w, src.h, src.elempack);

                gridptr += 2;
                offset_value_ptr += 6;
            }
        }
    }
    else
    {
        for (int y = 0; y < outh; y++)
        {
            const float* gridptr_x = grid.channel(0).row(y);
            const float* gridptr_y = grid.channel(1).row(y);
            float* offset_value_ptr = (float*)offset_value.data + (size_t)y * outw * 6;

            int x = 0;
#if __ARM_NEON
            for (; x + 3 < outw; x += 4)
            {
                float32x4_t _sample_x = grid_sample_unormalize_neon(src.w, vld1q_f32(gridptr_x), align_corner);
                float32x4_t _sample_y = grid_sample_unormalize_neon(src.h, vld1q_f32(gridptr_y), align_corner);
                _sample_x = compute_coord_neon(_sample_x, src.w, padding_mode, align_corner);
                _sample_y = compute_coord_neon(_sample_y, src.h, padding_mode, align_corner);

                gridsample_store_2d_bilinear_neon(offset_value_ptr, _sample_x, _sample_y, src.w, src.h, src.elempack);

                gridptr_x += 4;
                gridptr_y += 4;
                offset_value_ptr += 24;
            }
#endif // __ARM_NEON
            for (; x < outw; x++)
            {
                float sample_x = grid_sample_unormalize_arm(src.w, *gridptr_x, align_corner);
                float sample_y = grid_sample_unormalize_arm(src.h, *gridptr_y, align_corner);
                sample_x = compute_coord_arm(sample_x, src.w, padding_mode, align_corner);
                sample_y = compute_coord_arm(sample_y, src.h, padding_mode, align_corner);

                gridsample_store_2d_bilinear_arm(offset_value_ptr, sample_x, sample_y, src.w, src.h, src.elempack);

                gridptr_x++;
                gridptr_y++;
                offset_value_ptr += 6;
            }
        }
    }
}

template<GridSample::PaddingMode padding_mode, bool align_corner>
static void gridsample_3d_bilinear_compute_blob_arm(const Mat& src, const Mat& grid, Mat& offset_value, int permute_fusion)
{
    const int outw = offset_value.w;
    const int outh = offset_value.h;
    const int outd = offset_value.c;

    if (permute_fusion == 0)
    {
        for (int zy = 0; zy < outd * outh; zy++)
        {
            const int z = zy / outh;
            const int y = zy % outh;

            const float* gridptr = (const float*)grid.channel(z) + (size_t)y * outw * 3;
            float* offset_value_ptr = (float*)offset_value.data + (size_t)zy * outw * 11;

            int x = 0;
#if __ARM_NEON
            for (; x + 3 < outw; x += 4)
            {
                float32x4x3_t _xyz = vld3q_f32(gridptr);
                float32x4_t _sample_x = grid_sample_unormalize_neon(src.w, _xyz.val[0], align_corner);
                float32x4_t _sample_y = grid_sample_unormalize_neon(src.h, _xyz.val[1], align_corner);
                float32x4_t _sample_z = grid_sample_unormalize_neon(src.d, _xyz.val[2], align_corner);
                _sample_x = compute_coord_neon(_sample_x, src.w, padding_mode, align_corner);
                _sample_y = compute_coord_neon(_sample_y, src.h, padding_mode, align_corner);
                _sample_z = compute_coord_neon(_sample_z, src.d, padding_mode, align_corner);

                gridsample_store_3d_bilinear_neon(offset_value_ptr, _sample_x, _sample_y, _sample_z, src.w, src.h, src.d, src.elempack);

                gridptr += 12;
                offset_value_ptr += 44;
            }
#endif // __ARM_NEON
            for (; x < outw; x++)
            {
                float sample_x = grid_sample_unormalize_arm(src.w, gridptr[0], align_corner);
                float sample_y = grid_sample_unormalize_arm(src.h, gridptr[1], align_corner);
                float sample_z = grid_sample_unormalize_arm(src.d, gridptr[2], align_corner);
                sample_x = compute_coord_arm(sample_x, src.w, padding_mode, align_corner);
                sample_y = compute_coord_arm(sample_y, src.h, padding_mode, align_corner);
                sample_z = compute_coord_arm(sample_z, src.d, padding_mode, align_corner);

                gridsample_store_3d_bilinear_arm(offset_value_ptr, sample_x, sample_y, sample_z, src.w, src.h, src.d, src.elempack);

                gridptr += 3;
                offset_value_ptr += 11;
            }
        }
    }
    else
    {
        for (int zy = 0; zy < outd * outh; zy++)
        {
            const float* gridptr_x = (const float*)grid.channel(0) + (size_t)zy * outw;
            const float* gridptr_y = (const float*)grid.channel(1) + (size_t)zy * outw;
            const float* gridptr_z = (const float*)grid.channel(2) + (size_t)zy * outw;
            float* offset_value_ptr = (float*)offset_value.data + (size_t)zy * outw * 11;

            int x = 0;
#if __ARM_NEON
            for (; x + 3 < outw; x += 4)
            {
                float32x4_t _sample_x = grid_sample_unormalize_neon(src.w, vld1q_f32(gridptr_x), align_corner);
                float32x4_t _sample_y = grid_sample_unormalize_neon(src.h, vld1q_f32(gridptr_y), align_corner);
                float32x4_t _sample_z = grid_sample_unormalize_neon(src.d, vld1q_f32(gridptr_z), align_corner);
                _sample_x = compute_coord_neon(_sample_x, src.w, padding_mode, align_corner);
                _sample_y = compute_coord_neon(_sample_y, src.h, padding_mode, align_corner);
                _sample_z = compute_coord_neon(_sample_z, src.d, padding_mode, align_corner);

                gridsample_store_3d_bilinear_neon(offset_value_ptr, _sample_x, _sample_y, _sample_z, src.w, src.h, src.d, src.elempack);

                gridptr_x += 4;
                gridptr_y += 4;
                gridptr_z += 4;
                offset_value_ptr += 44;
            }
#endif // __ARM_NEON
            for (; x < outw; x++)
            {
                float sample_x = grid_sample_unormalize_arm(src.w, *gridptr_x, align_corner);
                float sample_y = grid_sample_unormalize_arm(src.h, *gridptr_y, align_corner);
                float sample_z = grid_sample_unormalize_arm(src.d, *gridptr_z, align_corner);
                sample_x = compute_coord_arm(sample_x, src.w, padding_mode, align_corner);
                sample_y = compute_coord_arm(sample_y, src.h, padding_mode, align_corner);
                sample_z = compute_coord_arm(sample_z, src.d, padding_mode, align_corner);

                gridsample_store_3d_bilinear_arm(offset_value_ptr, sample_x, sample_y, sample_z, src.w, src.h, src.d, src.elempack);

                gridptr_x++;
                gridptr_y++;
                gridptr_z++;
                offset_value_ptr += 11;
            }
        }
    }
}

static void gridsample_2d_bilinear_compute_blob_arm(const Mat& src, const Mat& grid, Mat& offset_value, int padding_mode, int align_corner, int permute_fusion)
{
    GRIDSAMPLE_COMPUTE_BLOB_DISPATCH(gridsample_2d_bilinear_compute_blob_arm, src, grid, offset_value, padding_mode, align_corner, permute_fusion);
}

static void gridsample_3d_bilinear_compute_blob_arm(const Mat& src, const Mat& grid, Mat& offset_value, int padding_mode, int align_corner, int permute_fusion)
{
    GRIDSAMPLE_COMPUTE_BLOB_DISPATCH(gridsample_3d_bilinear_compute_blob_arm, src, grid, offset_value, padding_mode, align_corner, permute_fusion);
}
