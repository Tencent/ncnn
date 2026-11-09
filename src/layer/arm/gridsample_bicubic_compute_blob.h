// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

template<GridSample::PaddingMode padding_mode, bool align_corner>
static void gridsample_2d_bicubic_compute_blob_arm(const Mat& src, const Mat& grid, Mat& offset_value, int permute_fusion)
{
    const int outw = offset_value.w;
    const int outh = offset_value.h;

    if (permute_fusion == 0)
    {
        for (int y = 0; y < outh; y++)
        {
            const float* gridptr = grid.channel(y);
            float* offset_value_ptr = (float*)offset_value.data + (size_t)y * outw * 18;

            int x = 0;
#if __ARM_NEON
            for (; x + 3 < outw; x += 4)
            {
                float32x4x2_t _xy = vld2q_f32(gridptr);
                float32x4_t _sample_x = grid_sample_unormalize_neon(src.w, _xy.val[0], align_corner);
                float32x4_t _sample_y = grid_sample_unormalize_neon(src.h, _xy.val[1], align_corner);

                gridsample_store_2d_bicubic_neon(offset_value_ptr, _sample_x, _sample_y, src.w, src.h, src.elempack, padding_mode, align_corner);

                gridptr += 8;
                offset_value_ptr += 72;
            }
#endif // __ARM_NEON
            for (; x < outw; x++)
            {
                float sample_x = grid_sample_unormalize_arm(src.w, gridptr[0], align_corner);
                float sample_y = grid_sample_unormalize_arm(src.h, gridptr[1], align_corner);

                gridsample_store_2d_bicubic_arm(offset_value_ptr, sample_x, sample_y, src.w, src.h, src.elempack, padding_mode, align_corner);

                gridptr += 2;
                offset_value_ptr += 18;
            }
        }
    }
    else
    {
        for (int y = 0; y < outh; y++)
        {
            const float* gridptr_x = grid.channel(0).row(y);
            const float* gridptr_y = grid.channel(1).row(y);
            float* offset_value_ptr = (float*)offset_value.data + (size_t)y * outw * 18;

            int x = 0;
#if __ARM_NEON
            for (; x + 3 < outw; x += 4)
            {
                float32x4_t _sample_x = grid_sample_unormalize_neon(src.w, vld1q_f32(gridptr_x), align_corner);
                float32x4_t _sample_y = grid_sample_unormalize_neon(src.h, vld1q_f32(gridptr_y), align_corner);

                gridsample_store_2d_bicubic_neon(offset_value_ptr, _sample_x, _sample_y, src.w, src.h, src.elempack, padding_mode, align_corner);

                gridptr_x += 4;
                gridptr_y += 4;
                offset_value_ptr += 72;
            }
#endif // __ARM_NEON
            for (; x < outw; x++)
            {
                float sample_x = grid_sample_unormalize_arm(src.w, *gridptr_x, align_corner);
                float sample_y = grid_sample_unormalize_arm(src.h, *gridptr_y, align_corner);

                gridsample_store_2d_bicubic_arm(offset_value_ptr, sample_x, sample_y, src.w, src.h, src.elempack, padding_mode, align_corner);

                gridptr_x++;
                gridptr_y++;
                offset_value_ptr += 18;
            }
        }
    }
}

static void gridsample_2d_bicubic_compute_blob_arm(const Mat& src, const Mat& grid, Mat& offset_value, int padding_mode, int align_corner, int permute_fusion)
{
    GRIDSAMPLE_COMPUTE_BLOB_DISPATCH(gridsample_2d_bicubic_compute_blob_arm, src, grid, offset_value, padding_mode, align_corner, permute_fusion);
}
