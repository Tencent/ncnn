// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

template<GridSample::PaddingMode padding_mode, bool align_corner>
static void gridsample_2d_bilinear_compute_blob_loongarch(const Mat& src, const Mat& grid, Mat& offset_value, int permute_fusion)
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
                _sample_x = compute_coord_lasx(_sample_x, src.w, padding_mode, align_corner);
                _sample_y = compute_coord_lasx(_sample_y, src.h, padding_mode, align_corner);

                gridsample_store_2d_bilinear_lasx(offset_value_ptr, _sample_x, _sample_y, src.w, src.h, src.elempack);

                gridptr += 16;
                offset_value_ptr += 48;
            }
#endif // __loongarch_asx
#if __loongarch_sx
            for (; x + 3 < outw; x += 4)
            {
                __m128 _sample_x = gridsample_set4_ps_lsx(gridptr[0], gridptr[2], gridptr[4], gridptr[6]);
                __m128 _sample_y = gridsample_set4_ps_lsx(gridptr[1], gridptr[3], gridptr[5], gridptr[7]);
                _sample_x = grid_sample_unormalize_lsx(src.w, _sample_x, align_corner);
                _sample_y = grid_sample_unormalize_lsx(src.h, _sample_y, align_corner);
                _sample_x = compute_coord_lsx(_sample_x, src.w, padding_mode, align_corner);
                _sample_y = compute_coord_lsx(_sample_y, src.h, padding_mode, align_corner);

                gridsample_store_2d_bilinear_lsx(offset_value_ptr, _sample_x, _sample_y, src.w, src.h, src.elempack);

                gridptr += 8;
                offset_value_ptr += 24;
            }
#endif // __loongarch_sx
            for (; x < outw; x++)
            {
                float sample_x = grid_sample_unormalize_loongarch(src.w, gridptr[0], align_corner);
                float sample_y = grid_sample_unormalize_loongarch(src.h, gridptr[1], align_corner);
                sample_x = compute_coord_loongarch(sample_x, src.w, padding_mode, align_corner);
                sample_y = compute_coord_loongarch(sample_y, src.h, padding_mode, align_corner);

                int x0 = (int)floorf(sample_x);
                int y0 = (int)floorf(sample_y);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                int* offset_ptr = (int*)offset_value_ptr;
                float* value_ptr = offset_value_ptr + 4;

                offset_ptr[0] = in_bounds_loongarch(x0, y0, src.w, src.h) ? (x0 + y0 * src.w) * src.elempack : -1;
                offset_ptr[1] = in_bounds_loongarch(x1, y0, src.w, src.h) ? (x1 + y0 * src.w) * src.elempack : -1;
                offset_ptr[2] = in_bounds_loongarch(x0, y1, src.w, src.h) ? (x0 + y1 * src.w) * src.elempack : -1;
                offset_ptr[3] = in_bounds_loongarch(x1, y1, src.w, src.h) ? (x1 + y1 * src.w) * src.elempack : -1;

                value_ptr[0] = sample_x - x0;
                value_ptr[1] = sample_y - y0;

                gridptr += 2;
                offset_value_ptr += 6;
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
                _sample_x = compute_coord_lasx(_sample_x, src.w, padding_mode, align_corner);
                _sample_y = compute_coord_lasx(_sample_y, src.h, padding_mode, align_corner);

                gridsample_store_2d_bilinear_lasx(offset_value_ptr, _sample_x, _sample_y, src.w, src.h, src.elempack);

                gridptr_x += 8;
                gridptr_y += 8;
                offset_value_ptr += 48;
            }
#endif // __loongarch_asx
#if __loongarch_sx
            for (; x + 3 < outw; x += 4)
            {
                __m128 _sample_x = (__m128)__lsx_vld(gridptr_x, 0);
                __m128 _sample_y = (__m128)__lsx_vld(gridptr_y, 0);
                _sample_x = grid_sample_unormalize_lsx(src.w, _sample_x, align_corner);
                _sample_y = grid_sample_unormalize_lsx(src.h, _sample_y, align_corner);
                _sample_x = compute_coord_lsx(_sample_x, src.w, padding_mode, align_corner);
                _sample_y = compute_coord_lsx(_sample_y, src.h, padding_mode, align_corner);

                gridsample_store_2d_bilinear_lsx(offset_value_ptr, _sample_x, _sample_y, src.w, src.h, src.elempack);

                gridptr_x += 4;
                gridptr_y += 4;
                offset_value_ptr += 24;
            }
#endif // __loongarch_sx
            for (; x < outw; x++)
            {
                float sample_x = grid_sample_unormalize_loongarch(src.w, *gridptr_x, align_corner);
                float sample_y = grid_sample_unormalize_loongarch(src.h, *gridptr_y, align_corner);
                sample_x = compute_coord_loongarch(sample_x, src.w, padding_mode, align_corner);
                sample_y = compute_coord_loongarch(sample_y, src.h, padding_mode, align_corner);

                int x0 = (int)floorf(sample_x);
                int y0 = (int)floorf(sample_y);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                int* offset_ptr = (int*)offset_value_ptr;
                float* value_ptr = offset_value_ptr + 4;

                offset_ptr[0] = in_bounds_loongarch(x0, y0, src.w, src.h) ? (x0 + y0 * src.w) * src.elempack : -1;
                offset_ptr[1] = in_bounds_loongarch(x1, y0, src.w, src.h) ? (x1 + y0 * src.w) * src.elempack : -1;
                offset_ptr[2] = in_bounds_loongarch(x0, y1, src.w, src.h) ? (x0 + y1 * src.w) * src.elempack : -1;
                offset_ptr[3] = in_bounds_loongarch(x1, y1, src.w, src.h) ? (x1 + y1 * src.w) * src.elempack : -1;

                value_ptr[0] = sample_x - x0;
                value_ptr[1] = sample_y - y0;

                gridptr_x++;
                gridptr_y++;
                offset_value_ptr += 6;
            }
        }
    }
}

template<GridSample::PaddingMode padding_mode, bool align_corner>
static void gridsample_3d_bilinear_compute_blob_loongarch(const Mat& src, const Mat& grid, Mat& offset_value, int permute_fusion)
{
    const int outw = offset_value.w;
    const int outh = offset_value.h;
    const int outd = offset_value.c;
    float* offset_value_ptr = offset_value.channel(0);

    if (permute_fusion == 0)
    {
        for (int z = 0; z < outd; z++)
        {
            const float* gridptr = grid.channel(z);
            for (int y = 0; y < outh; y++)
            {
                int x = 0;
#if __loongarch_asx
                for (; x + 7 < outw; x += 8)
                {
                    __m256 _sample_x = gridsample_set8_ps_lasx(gridptr[0], gridptr[3], gridptr[6], gridptr[9], gridptr[12], gridptr[15], gridptr[18], gridptr[21]);
                    __m256 _sample_y = gridsample_set8_ps_lasx(gridptr[1], gridptr[4], gridptr[7], gridptr[10], gridptr[13], gridptr[16], gridptr[19], gridptr[22]);
                    __m256 _sample_z = gridsample_set8_ps_lasx(gridptr[2], gridptr[5], gridptr[8], gridptr[11], gridptr[14], gridptr[17], gridptr[20], gridptr[23]);
                    _sample_x = grid_sample_unormalize_lasx(src.w, _sample_x, align_corner);
                    _sample_y = grid_sample_unormalize_lasx(src.h, _sample_y, align_corner);
                    _sample_z = grid_sample_unormalize_lasx(src.d, _sample_z, align_corner);
                    _sample_x = compute_coord_lasx(_sample_x, src.w, padding_mode, align_corner);
                    _sample_y = compute_coord_lasx(_sample_y, src.h, padding_mode, align_corner);
                    _sample_z = compute_coord_lasx(_sample_z, src.d, padding_mode, align_corner);

                    gridsample_store_3d_bilinear_lasx(offset_value_ptr, _sample_x, _sample_y, _sample_z, src.w, src.h, src.d, src.elempack);

                    gridptr += 24;
                    offset_value_ptr += 88;
                }
#endif // __loongarch_asx
#if __loongarch_sx
                for (; x + 3 < outw; x += 4)
                {
                    __m128 _sample_x = gridsample_set4_ps_lsx(gridptr[0], gridptr[3], gridptr[6], gridptr[9]);
                    __m128 _sample_y = gridsample_set4_ps_lsx(gridptr[1], gridptr[4], gridptr[7], gridptr[10]);
                    __m128 _sample_z = gridsample_set4_ps_lsx(gridptr[2], gridptr[5], gridptr[8], gridptr[11]);
                    _sample_x = grid_sample_unormalize_lsx(src.w, _sample_x, align_corner);
                    _sample_y = grid_sample_unormalize_lsx(src.h, _sample_y, align_corner);
                    _sample_z = grid_sample_unormalize_lsx(src.d, _sample_z, align_corner);
                    _sample_x = compute_coord_lsx(_sample_x, src.w, padding_mode, align_corner);
                    _sample_y = compute_coord_lsx(_sample_y, src.h, padding_mode, align_corner);
                    _sample_z = compute_coord_lsx(_sample_z, src.d, padding_mode, align_corner);

                    gridsample_store_3d_bilinear_lsx(offset_value_ptr, _sample_x, _sample_y, _sample_z, src.w, src.h, src.d, src.elempack);

                    gridptr += 12;
                    offset_value_ptr += 44;
                }
#endif // __loongarch_sx
                for (; x < outw; x++)
                {
                    float sample_x = grid_sample_unormalize_loongarch(src.w, gridptr[0], align_corner);
                    float sample_y = grid_sample_unormalize_loongarch(src.h, gridptr[1], align_corner);
                    float sample_z = grid_sample_unormalize_loongarch(src.d, gridptr[2], align_corner);
                    sample_x = compute_coord_loongarch(sample_x, src.w, padding_mode, align_corner);
                    sample_y = compute_coord_loongarch(sample_y, src.h, padding_mode, align_corner);
                    sample_z = compute_coord_loongarch(sample_z, src.d, padding_mode, align_corner);

                    int x0 = (int)floorf(sample_x);
                    int y0 = (int)floorf(sample_y);
                    int z0 = (int)floorf(sample_z);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;
                    int z1 = z0 + 1;

                    int* offset_ptr = (int*)offset_value_ptr;
                    float* value_ptr = offset_value_ptr + 8;

                    offset_ptr[0] = in_bounds_loongarch(x0, y0, z0, src.w, src.h, src.d) ? (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[1] = in_bounds_loongarch(x1, y0, z0, src.w, src.h, src.d) ? (x1 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[2] = in_bounds_loongarch(x0, y1, z0, src.w, src.h, src.d) ? (x0 + y1 * src.w + z0 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[3] = in_bounds_loongarch(x1, y1, z0, src.w, src.h, src.d) ? (x1 + y1 * src.w + z0 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[4] = in_bounds_loongarch(x0, y0, z1, src.w, src.h, src.d) ? (x0 + y0 * src.w + z1 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[5] = in_bounds_loongarch(x1, y0, z1, src.w, src.h, src.d) ? (x1 + y0 * src.w + z1 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[6] = in_bounds_loongarch(x0, y1, z1, src.w, src.h, src.d) ? (x0 + y1 * src.w + z1 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[7] = in_bounds_loongarch(x1, y1, z1, src.w, src.h, src.d) ? (x1 + y1 * src.w + z1 * src.w * src.h) * src.elempack : -1;

                    value_ptr[0] = sample_x - x0;
                    value_ptr[1] = sample_y - y0;
                    value_ptr[2] = sample_z - z0;

                    gridptr += 3;
                    offset_value_ptr += 11;
                }
            }
        }
    }
    else
    {
        const float* gridptr_x = grid.channel(0);
        const float* gridptr_y = grid.channel(1);
        const float* gridptr_z = grid.channel(2);

        for (int z = 0; z < outd; z++)
        {
            for (int y = 0; y < outh; y++)
            {
                int x = 0;
#if __loongarch_asx
                for (; x + 7 < outw; x += 8)
                {
                    __m256 _sample_x = (__m256)__lasx_xvld(gridptr_x, 0);
                    __m256 _sample_y = (__m256)__lasx_xvld(gridptr_y, 0);
                    __m256 _sample_z = (__m256)__lasx_xvld(gridptr_z, 0);
                    _sample_x = grid_sample_unormalize_lasx(src.w, _sample_x, align_corner);
                    _sample_y = grid_sample_unormalize_lasx(src.h, _sample_y, align_corner);
                    _sample_z = grid_sample_unormalize_lasx(src.d, _sample_z, align_corner);
                    _sample_x = compute_coord_lasx(_sample_x, src.w, padding_mode, align_corner);
                    _sample_y = compute_coord_lasx(_sample_y, src.h, padding_mode, align_corner);
                    _sample_z = compute_coord_lasx(_sample_z, src.d, padding_mode, align_corner);

                    gridsample_store_3d_bilinear_lasx(offset_value_ptr, _sample_x, _sample_y, _sample_z, src.w, src.h, src.d, src.elempack);

                    gridptr_x += 8;
                    gridptr_y += 8;
                    gridptr_z += 8;
                    offset_value_ptr += 88;
                }
#endif // __loongarch_asx
#if __loongarch_sx
                for (; x + 3 < outw; x += 4)
                {
                    __m128 _sample_x = (__m128)__lsx_vld(gridptr_x, 0);
                    __m128 _sample_y = (__m128)__lsx_vld(gridptr_y, 0);
                    __m128 _sample_z = (__m128)__lsx_vld(gridptr_z, 0);
                    _sample_x = grid_sample_unormalize_lsx(src.w, _sample_x, align_corner);
                    _sample_y = grid_sample_unormalize_lsx(src.h, _sample_y, align_corner);
                    _sample_z = grid_sample_unormalize_lsx(src.d, _sample_z, align_corner);
                    _sample_x = compute_coord_lsx(_sample_x, src.w, padding_mode, align_corner);
                    _sample_y = compute_coord_lsx(_sample_y, src.h, padding_mode, align_corner);
                    _sample_z = compute_coord_lsx(_sample_z, src.d, padding_mode, align_corner);

                    gridsample_store_3d_bilinear_lsx(offset_value_ptr, _sample_x, _sample_y, _sample_z, src.w, src.h, src.d, src.elempack);

                    gridptr_x += 4;
                    gridptr_y += 4;
                    gridptr_z += 4;
                    offset_value_ptr += 44;
                }
#endif // __loongarch_sx
                for (; x < outw; x++)
                {
                    float sample_x = grid_sample_unormalize_loongarch(src.w, *gridptr_x, align_corner);
                    float sample_y = grid_sample_unormalize_loongarch(src.h, *gridptr_y, align_corner);
                    float sample_z = grid_sample_unormalize_loongarch(src.d, *gridptr_z, align_corner);
                    sample_x = compute_coord_loongarch(sample_x, src.w, padding_mode, align_corner);
                    sample_y = compute_coord_loongarch(sample_y, src.h, padding_mode, align_corner);
                    sample_z = compute_coord_loongarch(sample_z, src.d, padding_mode, align_corner);

                    int x0 = (int)floorf(sample_x);
                    int y0 = (int)floorf(sample_y);
                    int z0 = (int)floorf(sample_z);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;
                    int z1 = z0 + 1;

                    int* offset_ptr = (int*)offset_value_ptr;
                    float* value_ptr = offset_value_ptr + 8;

                    offset_ptr[0] = in_bounds_loongarch(x0, y0, z0, src.w, src.h, src.d) ? (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[1] = in_bounds_loongarch(x1, y0, z0, src.w, src.h, src.d) ? (x1 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[2] = in_bounds_loongarch(x0, y1, z0, src.w, src.h, src.d) ? (x0 + y1 * src.w + z0 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[3] = in_bounds_loongarch(x1, y1, z0, src.w, src.h, src.d) ? (x1 + y1 * src.w + z0 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[4] = in_bounds_loongarch(x0, y0, z1, src.w, src.h, src.d) ? (x0 + y0 * src.w + z1 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[5] = in_bounds_loongarch(x1, y0, z1, src.w, src.h, src.d) ? (x1 + y0 * src.w + z1 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[6] = in_bounds_loongarch(x0, y1, z1, src.w, src.h, src.d) ? (x0 + y1 * src.w + z1 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[7] = in_bounds_loongarch(x1, y1, z1, src.w, src.h, src.d) ? (x1 + y1 * src.w + z1 * src.w * src.h) * src.elempack : -1;

                    value_ptr[0] = sample_x - x0;
                    value_ptr[1] = sample_y - y0;
                    value_ptr[2] = sample_z - z0;

                    gridptr_x++;
                    gridptr_y++;
                    gridptr_z++;
                    offset_value_ptr += 11;
                }
            }
        }
    }
}

static void gridsample_2d_bilinear_compute_blob_loongarch(const Mat& src, const Mat& grid, Mat& offset_value, int padding_mode, int align_corner, int permute_fusion)
{
    GRIDSAMPLE_COMPUTE_BLOB_DISPATCH(gridsample_2d_bilinear_compute_blob_loongarch, src, grid, offset_value, padding_mode, align_corner, permute_fusion);
}

static void gridsample_3d_bilinear_compute_blob_loongarch(const Mat& src, const Mat& grid, Mat& offset_value, int padding_mode, int align_corner, int permute_fusion)
{
    GRIDSAMPLE_COMPUTE_BLOB_DISPATCH(gridsample_3d_bilinear_compute_blob_loongarch, src, grid, offset_value, padding_mode, align_corner, permute_fusion);
}
