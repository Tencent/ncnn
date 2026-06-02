// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

template<GridSample::PaddingMode padding_mode, bool align_corner>
static void gridsample_2d_bilinear_compute_blob_mips(const Mat& src, const Mat& grid, Mat& offset_value, int permute_fusion)
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
#if __mips_msa
            for (; x + 3 < outw; x += 4)
            {
                v4f32 _sample_x = gridsample_set4_ps_msa(gridptr[0], gridptr[2], gridptr[4], gridptr[6]);
                v4f32 _sample_y = gridsample_set4_ps_msa(gridptr[1], gridptr[3], gridptr[5], gridptr[7]);
                _sample_x = grid_sample_unormalize_msa(src.w, _sample_x, align_corner);
                _sample_y = grid_sample_unormalize_msa(src.h, _sample_y, align_corner);
                _sample_x = compute_coord_msa(_sample_x, src.w, padding_mode, align_corner);
                _sample_y = compute_coord_msa(_sample_y, src.h, padding_mode, align_corner);

                gridsample_store_2d_bilinear_msa(offset_value_ptr, _sample_x, _sample_y, src.w, src.h, src.elempack);

                gridptr += 8;
                offset_value_ptr += 24;
            }
#endif // __mips_msa
            for (; x < outw; x++)
            {
                float sample_x = grid_sample_unormalize_mips(src.w, gridptr[0], align_corner);
                float sample_y = grid_sample_unormalize_mips(src.h, gridptr[1], align_corner);
                sample_x = compute_coord_mips(sample_x, src.w, padding_mode, align_corner);
                sample_y = compute_coord_mips(sample_y, src.h, padding_mode, align_corner);

                int x0 = (int)floorf(sample_x);
                int y0 = (int)floorf(sample_y);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                int* offset_ptr = (int*)offset_value_ptr;
                float* value_ptr = offset_value_ptr + 4;

                offset_ptr[0] = in_bounds_mips(x0, y0, src.w, src.h) ? (x0 + y0 * src.w) * src.elempack : -1;
                offset_ptr[1] = in_bounds_mips(x1, y0, src.w, src.h) ? (x1 + y0 * src.w) * src.elempack : -1;
                offset_ptr[2] = in_bounds_mips(x0, y1, src.w, src.h) ? (x0 + y1 * src.w) * src.elempack : -1;
                offset_ptr[3] = in_bounds_mips(x1, y1, src.w, src.h) ? (x1 + y1 * src.w) * src.elempack : -1;

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
#if __mips_msa
            for (; x + 3 < outw; x += 4)
            {
                v4f32 _sample_x = (v4f32)__msa_ld_w(gridptr_x, 0);
                v4f32 _sample_y = (v4f32)__msa_ld_w(gridptr_y, 0);
                _sample_x = grid_sample_unormalize_msa(src.w, _sample_x, align_corner);
                _sample_y = grid_sample_unormalize_msa(src.h, _sample_y, align_corner);
                _sample_x = compute_coord_msa(_sample_x, src.w, padding_mode, align_corner);
                _sample_y = compute_coord_msa(_sample_y, src.h, padding_mode, align_corner);

                gridsample_store_2d_bilinear_msa(offset_value_ptr, _sample_x, _sample_y, src.w, src.h, src.elempack);

                gridptr_x += 4;
                gridptr_y += 4;
                offset_value_ptr += 24;
            }
#endif // __mips_msa
            for (; x < outw; x++)
            {
                float sample_x = grid_sample_unormalize_mips(src.w, *gridptr_x, align_corner);
                float sample_y = grid_sample_unormalize_mips(src.h, *gridptr_y, align_corner);
                sample_x = compute_coord_mips(sample_x, src.w, padding_mode, align_corner);
                sample_y = compute_coord_mips(sample_y, src.h, padding_mode, align_corner);

                int x0 = (int)floorf(sample_x);
                int y0 = (int)floorf(sample_y);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                int* offset_ptr = (int*)offset_value_ptr;
                float* value_ptr = offset_value_ptr + 4;

                offset_ptr[0] = in_bounds_mips(x0, y0, src.w, src.h) ? (x0 + y0 * src.w) * src.elempack : -1;
                offset_ptr[1] = in_bounds_mips(x1, y0, src.w, src.h) ? (x1 + y0 * src.w) * src.elempack : -1;
                offset_ptr[2] = in_bounds_mips(x0, y1, src.w, src.h) ? (x0 + y1 * src.w) * src.elempack : -1;
                offset_ptr[3] = in_bounds_mips(x1, y1, src.w, src.h) ? (x1 + y1 * src.w) * src.elempack : -1;

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
static void gridsample_3d_bilinear_compute_blob_mips(const Mat& src, const Mat& grid, Mat& offset_value, int permute_fusion)
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
#if __mips_msa
                for (; x + 3 < outw; x += 4)
                {
                    v4f32 _sample_x = gridsample_set4_ps_msa(gridptr[0], gridptr[3], gridptr[6], gridptr[9]);
                    v4f32 _sample_y = gridsample_set4_ps_msa(gridptr[1], gridptr[4], gridptr[7], gridptr[10]);
                    v4f32 _sample_z = gridsample_set4_ps_msa(gridptr[2], gridptr[5], gridptr[8], gridptr[11]);
                    _sample_x = grid_sample_unormalize_msa(src.w, _sample_x, align_corner);
                    _sample_y = grid_sample_unormalize_msa(src.h, _sample_y, align_corner);
                    _sample_z = grid_sample_unormalize_msa(src.d, _sample_z, align_corner);
                    _sample_x = compute_coord_msa(_sample_x, src.w, padding_mode, align_corner);
                    _sample_y = compute_coord_msa(_sample_y, src.h, padding_mode, align_corner);
                    _sample_z = compute_coord_msa(_sample_z, src.d, padding_mode, align_corner);

                    gridsample_store_3d_bilinear_msa(offset_value_ptr, _sample_x, _sample_y, _sample_z, src.w, src.h, src.d, src.elempack);

                    gridptr += 12;
                    offset_value_ptr += 44;
                }
#endif // __mips_msa
                for (; x < outw; x++)
                {
                    float sample_x = grid_sample_unormalize_mips(src.w, gridptr[0], align_corner);
                    float sample_y = grid_sample_unormalize_mips(src.h, gridptr[1], align_corner);
                    float sample_z = grid_sample_unormalize_mips(src.d, gridptr[2], align_corner);
                    sample_x = compute_coord_mips(sample_x, src.w, padding_mode, align_corner);
                    sample_y = compute_coord_mips(sample_y, src.h, padding_mode, align_corner);
                    sample_z = compute_coord_mips(sample_z, src.d, padding_mode, align_corner);

                    int x0 = (int)floorf(sample_x);
                    int y0 = (int)floorf(sample_y);
                    int z0 = (int)floorf(sample_z);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;
                    int z1 = z0 + 1;

                    int* offset_ptr = (int*)offset_value_ptr;
                    float* value_ptr = offset_value_ptr + 8;

                    offset_ptr[0] = in_bounds_mips(x0, y0, z0, src.w, src.h, src.d) ? (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[1] = in_bounds_mips(x1, y0, z0, src.w, src.h, src.d) ? (x1 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[2] = in_bounds_mips(x0, y1, z0, src.w, src.h, src.d) ? (x0 + y1 * src.w + z0 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[3] = in_bounds_mips(x1, y1, z0, src.w, src.h, src.d) ? (x1 + y1 * src.w + z0 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[4] = in_bounds_mips(x0, y0, z1, src.w, src.h, src.d) ? (x0 + y0 * src.w + z1 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[5] = in_bounds_mips(x1, y0, z1, src.w, src.h, src.d) ? (x1 + y0 * src.w + z1 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[6] = in_bounds_mips(x0, y1, z1, src.w, src.h, src.d) ? (x0 + y1 * src.w + z1 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[7] = in_bounds_mips(x1, y1, z1, src.w, src.h, src.d) ? (x1 + y1 * src.w + z1 * src.w * src.h) * src.elempack : -1;

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
#if __mips_msa
                for (; x + 3 < outw; x += 4)
                {
                    v4f32 _sample_x = (v4f32)__msa_ld_w(gridptr_x, 0);
                    v4f32 _sample_y = (v4f32)__msa_ld_w(gridptr_y, 0);
                    v4f32 _sample_z = (v4f32)__msa_ld_w(gridptr_z, 0);
                    _sample_x = grid_sample_unormalize_msa(src.w, _sample_x, align_corner);
                    _sample_y = grid_sample_unormalize_msa(src.h, _sample_y, align_corner);
                    _sample_z = grid_sample_unormalize_msa(src.d, _sample_z, align_corner);
                    _sample_x = compute_coord_msa(_sample_x, src.w, padding_mode, align_corner);
                    _sample_y = compute_coord_msa(_sample_y, src.h, padding_mode, align_corner);
                    _sample_z = compute_coord_msa(_sample_z, src.d, padding_mode, align_corner);

                    gridsample_store_3d_bilinear_msa(offset_value_ptr, _sample_x, _sample_y, _sample_z, src.w, src.h, src.d, src.elempack);

                    gridptr_x += 4;
                    gridptr_y += 4;
                    gridptr_z += 4;
                    offset_value_ptr += 44;
                }
#endif // __mips_msa
                for (; x < outw; x++)
                {
                    float sample_x = grid_sample_unormalize_mips(src.w, *gridptr_x, align_corner);
                    float sample_y = grid_sample_unormalize_mips(src.h, *gridptr_y, align_corner);
                    float sample_z = grid_sample_unormalize_mips(src.d, *gridptr_z, align_corner);
                    sample_x = compute_coord_mips(sample_x, src.w, padding_mode, align_corner);
                    sample_y = compute_coord_mips(sample_y, src.h, padding_mode, align_corner);
                    sample_z = compute_coord_mips(sample_z, src.d, padding_mode, align_corner);

                    int x0 = (int)floorf(sample_x);
                    int y0 = (int)floorf(sample_y);
                    int z0 = (int)floorf(sample_z);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;
                    int z1 = z0 + 1;

                    int* offset_ptr = (int*)offset_value_ptr;
                    float* value_ptr = offset_value_ptr + 8;

                    offset_ptr[0] = in_bounds_mips(x0, y0, z0, src.w, src.h, src.d) ? (x0 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[1] = in_bounds_mips(x1, y0, z0, src.w, src.h, src.d) ? (x1 + y0 * src.w + z0 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[2] = in_bounds_mips(x0, y1, z0, src.w, src.h, src.d) ? (x0 + y1 * src.w + z0 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[3] = in_bounds_mips(x1, y1, z0, src.w, src.h, src.d) ? (x1 + y1 * src.w + z0 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[4] = in_bounds_mips(x0, y0, z1, src.w, src.h, src.d) ? (x0 + y0 * src.w + z1 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[5] = in_bounds_mips(x1, y0, z1, src.w, src.h, src.d) ? (x1 + y0 * src.w + z1 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[6] = in_bounds_mips(x0, y1, z1, src.w, src.h, src.d) ? (x0 + y1 * src.w + z1 * src.w * src.h) * src.elempack : -1;
                    offset_ptr[7] = in_bounds_mips(x1, y1, z1, src.w, src.h, src.d) ? (x1 + y1 * src.w + z1 * src.w * src.h) * src.elempack : -1;

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

static void gridsample_2d_bilinear_compute_blob_mips(const Mat& src, const Mat& grid, Mat& offset_value, int padding_mode, int align_corner, int permute_fusion)
{
    GRIDSAMPLE_COMPUTE_BLOB_DISPATCH(gridsample_2d_bilinear_compute_blob_mips, src, grid, offset_value, padding_mode, align_corner, permute_fusion);
}

static void gridsample_3d_bilinear_compute_blob_mips(const Mat& src, const Mat& grid, Mat& offset_value, int padding_mode, int align_corner, int permute_fusion)
{
    GRIDSAMPLE_COMPUTE_BLOB_DISPATCH(gridsample_3d_bilinear_compute_blob_mips, src, grid, offset_value, padding_mode, align_corner, permute_fusion);
}
