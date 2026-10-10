// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

template<GridSample::PaddingMode padding_mode, bool align_corner>
static void gridsample_2d_bicubic_compute_blob_loongarch(const Mat& src, const Mat& grid, Mat& offset_value, int permute_fusion)
{
    const int outw = offset_value.w;
    const int outh = offset_value.h;
    float* offset_value_ptr = offset_value.channel(0);

    if (permute_fusion == 0)
    {
        for (int y = 0; y < outh; y++)
        {
            const float* gridptr = grid.channel(y);
            int x = 0;
#if __loongarch_asx
            for (; x + 7 < outw; x += 8)
            {
                __m256 _sample_x = gridsample_set8_ps_lasx(gridptr[0], gridptr[2], gridptr[4], gridptr[6], gridptr[8], gridptr[10], gridptr[12], gridptr[14]);
                __m256 _sample_y = gridsample_set8_ps_lasx(gridptr[1], gridptr[3], gridptr[5], gridptr[7], gridptr[9], gridptr[11], gridptr[13], gridptr[15]);
                _sample_x = grid_sample_unormalize_lasx(src.w, _sample_x, align_corner);
                _sample_y = grid_sample_unormalize_lasx(src.h, _sample_y, align_corner);

                gridsample_store_2d_bicubic_lasx(offset_value_ptr, _sample_x, _sample_y, src.w, src.h, src.elempack, padding_mode, align_corner);

                gridptr += 16;
                offset_value_ptr += 144;
            }
#endif // __loongarch_asx
#if __loongarch_sx
            for (; x + 3 < outw; x += 4)
            {
                __m128 _sample_x = gridsample_set4_ps_lsx(gridptr[0], gridptr[2], gridptr[4], gridptr[6]);
                __m128 _sample_y = gridsample_set4_ps_lsx(gridptr[1], gridptr[3], gridptr[5], gridptr[7]);
                _sample_x = grid_sample_unormalize_lsx(src.w, _sample_x, align_corner);
                _sample_y = grid_sample_unormalize_lsx(src.h, _sample_y, align_corner);

                gridsample_store_2d_bicubic_lsx(offset_value_ptr, _sample_x, _sample_y, src.w, src.h, src.elempack, padding_mode, align_corner);

                gridptr += 8;
                offset_value_ptr += 72;
            }
#endif // __loongarch_sx
            for (; x < outw; x++)
            {
                float sample_x = grid_sample_unormalize_loongarch(src.w, gridptr[0], align_corner);
                float sample_y = grid_sample_unormalize_loongarch(src.h, gridptr[1], align_corner);

                int x1 = (int)floorf(sample_x);
                int y1 = (int)floorf(sample_y);
                int x0 = x1 - 1;
                int x2 = x1 + 1;
                int x3 = x1 + 2;

                offset_value_ptr[0] = sample_x - x1;
                offset_value_ptr[1] = sample_y - y1;

                x0 = (int)compute_coord_loongarch((float)x0, src.w, padding_mode, align_corner);
                x1 = (int)compute_coord_loongarch((float)x1, src.w, padding_mode, align_corner);
                x2 = (int)compute_coord_loongarch((float)x2, src.w, padding_mode, align_corner);
                x3 = (int)compute_coord_loongarch((float)x3, src.w, padding_mode, align_corner);

                bool x0_in_range = x0 >= 0 && x0 < src.w;
                bool x1_in_range = x1 >= 0 && x1 < src.w;
                bool x2_in_range = x2 >= 0 && x2 < src.w;
                bool x3_in_range = x3 >= 0 && x3 < src.w;

                int* offset_ptr = (int*)offset_value_ptr + 2;

                for (int i = 0; i < 4; i++)
                {
                    int gy = y1 + i - 1;
                    gy = (int)compute_coord_loongarch((float)gy, src.h, padding_mode, align_corner);
                    int offset_y = gy * src.w;
                    bool y_in_range = gy >= 0 && gy < src.h;

                    offset_ptr[0] = (x0_in_range && y_in_range) ? (offset_y + x0) * src.elempack : -1;
                    offset_ptr[1] = (x1_in_range && y_in_range) ? (offset_y + x1) * src.elempack : -1;
                    offset_ptr[2] = (x2_in_range && y_in_range) ? (offset_y + x2) * src.elempack : -1;
                    offset_ptr[3] = (x3_in_range && y_in_range) ? (offset_y + x3) * src.elempack : -1;

                    offset_ptr += 4;
                }

                gridptr += 2;
                offset_value_ptr += 18;
            }
        }
    }
    else
    {
        const float* gridptr_x = grid.channel(0);
        const float* gridptr_y = grid.channel(1);

        for (int y = 0; y < outh; y++)
        {
            int x = 0;
#if __loongarch_asx
            for (; x + 7 < outw; x += 8)
            {
                __m256 _sample_x = (__m256)__lasx_xvld(gridptr_x, 0);
                __m256 _sample_y = (__m256)__lasx_xvld(gridptr_y, 0);
                _sample_x = grid_sample_unormalize_lasx(src.w, _sample_x, align_corner);
                _sample_y = grid_sample_unormalize_lasx(src.h, _sample_y, align_corner);

                gridsample_store_2d_bicubic_lasx(offset_value_ptr, _sample_x, _sample_y, src.w, src.h, src.elempack, padding_mode, align_corner);

                gridptr_x += 8;
                gridptr_y += 8;
                offset_value_ptr += 144;
            }
#endif // __loongarch_asx
#if __loongarch_sx
            for (; x + 3 < outw; x += 4)
            {
                __m128 _sample_x = (__m128)__lsx_vld(gridptr_x, 0);
                __m128 _sample_y = (__m128)__lsx_vld(gridptr_y, 0);
                _sample_x = grid_sample_unormalize_lsx(src.w, _sample_x, align_corner);
                _sample_y = grid_sample_unormalize_lsx(src.h, _sample_y, align_corner);

                gridsample_store_2d_bicubic_lsx(offset_value_ptr, _sample_x, _sample_y, src.w, src.h, src.elempack, padding_mode, align_corner);

                gridptr_x += 4;
                gridptr_y += 4;
                offset_value_ptr += 72;
            }
#endif // __loongarch_sx
            for (; x < outw; x++)
            {
                float sample_x = grid_sample_unormalize_loongarch(src.w, *gridptr_x, align_corner);
                float sample_y = grid_sample_unormalize_loongarch(src.h, *gridptr_y, align_corner);

                int x1 = (int)floorf(sample_x);
                int y1 = (int)floorf(sample_y);
                int x0 = x1 - 1;
                int x2 = x1 + 1;
                int x3 = x1 + 2;

                offset_value_ptr[0] = sample_x - x1;
                offset_value_ptr[1] = sample_y - y1;

                x0 = (int)compute_coord_loongarch((float)x0, src.w, padding_mode, align_corner);
                x1 = (int)compute_coord_loongarch((float)x1, src.w, padding_mode, align_corner);
                x2 = (int)compute_coord_loongarch((float)x2, src.w, padding_mode, align_corner);
                x3 = (int)compute_coord_loongarch((float)x3, src.w, padding_mode, align_corner);

                bool x0_in_range = x0 >= 0 && x0 < src.w;
                bool x1_in_range = x1 >= 0 && x1 < src.w;
                bool x2_in_range = x2 >= 0 && x2 < src.w;
                bool x3_in_range = x3 >= 0 && x3 < src.w;

                int* offset_ptr = (int*)offset_value_ptr + 2;

                for (int i = 0; i < 4; i++)
                {
                    int gy = y1 + i - 1;
                    gy = (int)compute_coord_loongarch((float)gy, src.h, padding_mode, align_corner);
                    int offset_y = gy * src.w;
                    bool y_in_range = gy >= 0 && gy < src.h;

                    offset_ptr[0] = (x0_in_range && y_in_range) ? (offset_y + x0) * src.elempack : -1;
                    offset_ptr[1] = (x1_in_range && y_in_range) ? (offset_y + x1) * src.elempack : -1;
                    offset_ptr[2] = (x2_in_range && y_in_range) ? (offset_y + x2) * src.elempack : -1;
                    offset_ptr[3] = (x3_in_range && y_in_range) ? (offset_y + x3) * src.elempack : -1;

                    offset_ptr += 4;
                }

                gridptr_x++;
                gridptr_y++;
                offset_value_ptr += 18;
            }
        }
    }
}

static void gridsample_2d_bicubic_compute_blob_loongarch(const Mat& src, const Mat& grid, Mat& offset_value, int padding_mode, int align_corner, int permute_fusion)
{
    GRIDSAMPLE_COMPUTE_BLOB_DISPATCH(gridsample_2d_bicubic_compute_blob_loongarch, src, grid, offset_value, padding_mode, align_corner, permute_fusion);
}
