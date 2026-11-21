// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gridsample_powerpc.h"

#include <algorithm>
#include <math.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

GridSample_powerpc::GridSample_powerpc()
{
}

// Restore normalized location to acutal image location
static float grid_sample_unormalize(int w, float coordx, int align_corner)
{
    return align_corner ? (coordx + 1) / 2.f * (w - 1) : ((coordx + 1) * w - 1) / 2.f;
}

static float border_coord(float x, float border)
{
    return std::min(border, std::max(x, 0.0f));
}

static float reflect_coord(float x, int high)
{
    x = fabs(x);
    x = high - fabs(x - high);
    return x;
}

static float compute_coord(float sx, int w, int padding_mode, int align_corner)
{
    if (padding_mode == 2) // border
    {
        sx = border_coord(sx, w - 1);
    }
    else if (padding_mode == 3) // reflection
    {
        if (align_corner)
        {
            sx = reflect_coord(sx, w - 1);
        }
        else
        {
            sx = reflect_coord(sx + 0.5, w) - 0.5;
            sx = border_coord(sx, w - 1);
        }
    }

    return sx;
}

#if __ALTIVEC__

// gather the same (y, x) position from 4 channel planes, zero when out of bounds
static inline v4f32 gridsample_gather(const Mat* pp, int w, int h, int y, int x)
{
    float v[4];
    if (x >= 0 && x < w && y >= 0 && y < h)
    {
        v[0] = pp[0].row(y)[x];
        v[1] = pp[1].row(y)[x];
        v[2] = pp[2].row(y)[x];
        v[3] = pp[3].row(y)[x];
    }
    else
    {
        v[0] = 0.f;
        v[1] = 0.f;
        v[2] = 0.f;
        v[3] = 0.f;
    }
    return __ncnn_altivec_load(v);
}

// gather the same (z, y, x) position from 4 channel planes (dims 4 blob), zero when out of bounds
static inline v4f32 gridsample_gather_depth(const Mat* pp, int w, int h, int d, int z, int y, int x)
{
    float v[4];
    if (x >= 0 && x < w && y >= 0 && y < h && z >= 0 && z < d)
    {
        v[0] = pp[0].depth(z).row(y)[x];
        v[1] = pp[1].depth(z).row(y)[x];
        v[2] = pp[2].depth(z).row(y)[x];
        v[3] = pp[3].depth(z).row(y)[x];
    }
    else
    {
        v[0] = 0.f;
        v[1] = 0.f;
        v[2] = 0.f;
        v[3] = 0.f;
    }
    return __ncnn_altivec_load(v);
}

static void gridsample_bilinear_3d(const Mat& src, const Mat& offset_blob, Mat& dst, int padding_mode, int align_corner, int w, int h, int outw, int outh, int channels, int num_threads)
{
    const float* offxp = offset_blob.channel(0);
    const float* offyp = offset_blob.channel(1);

    #pragma omp parallel for num_threads(num_threads)
    for (int q = 0; q < channels; q += 4)
    {
        const int n = std::min(4, channels - q);

        Mat pp[4];
        for (int k = 0; k < n; k++)
        {
            pp[k] = src.channel(q + k);
        }
        for (int k = n; k < 4; k++)
        {
            pp[k] = pp[0];
        }

        float* op[4];
        for (int k = 0; k < n; k++)
        {
            op[k] = dst.channel(q + k);
        }

        const float* sxp = offxp;
        const float* syp = offyp;

        for (int y = 0; y < outh; y++)
        {
            float* orow[4];
            for (int k = 0; k < n; k++)
            {
                orow[k] = op[k] + y * outw;
            }

            for (int x = 0; x < outw; x++)
            {
                float sample_x = sxp[x];
                float sample_y = syp[x];

                sample_x = compute_coord(sample_x, w, padding_mode, align_corner);
                sample_y = compute_coord(sample_y, h, padding_mode, align_corner);

                int x0 = floor(sample_x);
                int y0 = floor(sample_y);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                float alpha = sample_x - x0;
                float beta = sample_y - y0;

                v4f32 _v00 = gridsample_gather(pp, w, h, y0, x0);
                v4f32 _v01 = gridsample_gather(pp, w, h, y0, x1);
                v4f32 _v10 = gridsample_gather(pp, w, h, y1, x0);
                v4f32 _v11 = gridsample_gather(pp, w, h, y1, x1);

                v4f32 _a = __ncnn_altivec_fill_f32(alpha);
                v4f32 _b = __ncnn_altivec_fill_f32(beta);
                v4f32 _1ma = __ncnn_altivec_fill_f32(1.f - alpha);
                v4f32 _1mb = __ncnn_altivec_fill_f32(1.f - beta);

                v4f32 _v0 = __ncnn_altivec_fmadd(__ncnn_altivec_mul(_v00, _1ma), _v01, _a);
                v4f32 _v1 = __ncnn_altivec_fmadd(__ncnn_altivec_mul(_v10, _1ma), _v11, _a);
                v4f32 _v = __ncnn_altivec_fmadd(__ncnn_altivec_mul(_v0, _1mb), _v1, _b);

                for (int k = 0; k < n; k++)
                {
                    orow[k][x] = vec_extract(_v, k);
                }
            }

            sxp += outw;
            syp += outw;
        }
    }
}

static void gridsample_nearest_3d(const Mat& src, const Mat& offset_blob, Mat& dst, int padding_mode, int align_corner, int w, int h, int outw, int outh, int channels, int num_threads)
{
    const float* offxp = offset_blob.channel(0);
    const float* offyp = offset_blob.channel(1);

    #pragma omp parallel for num_threads(num_threads)
    for (int q = 0; q < channels; q += 4)
    {
        const int n = std::min(4, channels - q);

        Mat pp[4];
        for (int k = 0; k < n; k++)
        {
            pp[k] = src.channel(q + k);
        }
        for (int k = n; k < 4; k++)
        {
            pp[k] = pp[0];
        }

        float* op[4];
        for (int k = 0; k < n; k++)
        {
            op[k] = dst.channel(q + k);
        }

        const float* sxp = offxp;
        const float* syp = offyp;

        for (int y = 0; y < outh; y++)
        {
            float* orow[4];
            for (int k = 0; k < n; k++)
            {
                orow[k] = op[k] + y * outw;
            }

            for (int x = 0; x < outw; x++)
            {
                float sample_x = compute_coord(sxp[x], w, padding_mode, align_corner);
                float sample_y = compute_coord(syp[x], h, padding_mode, align_corner);

                int x0 = static_cast<int>(floor(sample_x + 0.5f));
                int y0 = static_cast<int>(floor(sample_y + 0.5f));

                v4f32 _v = gridsample_gather(pp, w, h, y0, x0);

                for (int k = 0; k < n; k++)
                {
                    orow[k][x] = vec_extract(_v, k);
                }
            }

            sxp += outw;
            syp += outw;
        }
    }
}

static void gridsample_bilinear_4d(const Mat& src, const Mat& offset_blob, Mat& dst, int w, int h, int d, int outw, int outh, int outd, int channels, int num_threads)
{
    const float* offxp = offset_blob.channel(0);
    const float* offyp = offset_blob.channel(1);
    const float* offzp = offset_blob.channel(2);

    #pragma omp parallel for num_threads(num_threads)
    for (int q = 0; q < channels; q += 4)
    {
        const int n = std::min(4, channels - q);

        Mat pp[4];
        for (int k = 0; k < n; k++)
        {
            pp[k] = src.channel(q + k);
        }
        for (int k = n; k < 4; k++)
        {
            pp[k] = pp[0];
        }

        float* op[4];
        for (int k = 0; k < n; k++)
        {
            op[k] = dst.channel(q + k);
        }

        const float* sxp = offxp;
        const float* syp = offyp;
        const float* szp = offzp;

        for (int z = 0; z < outd; z++)
        {
            for (int y = 0; y < outh; y++)
            {
                float* orow[4];
                for (int k = 0; k < n; k++)
                {
                    orow[k] = op[k] + z * outh * outw + y * outw;
                }

                for (int x = 0; x < outw; x++)
                {
                    float sample_x = sxp[x];
                    float sample_y = syp[x];
                    float sample_z = szp[x];

                    int x0 = (int)floor(sample_x);
                    int y0 = (int)floor(sample_y);
                    int z0 = (int)floor(sample_z);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;
                    int z1 = z0 + 1;

                    float alpha = sample_x - x0;
                    float beta = sample_y - y0;
                    float gamma = sample_z - z0;

                    v4f32 _v000 = gridsample_gather_depth(pp, w, h, d, z0, y0, x0);
                    v4f32 _v001 = gridsample_gather_depth(pp, w, h, d, z0, y0, x1);
                    v4f32 _v010 = gridsample_gather_depth(pp, w, h, d, z0, y1, x0);
                    v4f32 _v011 = gridsample_gather_depth(pp, w, h, d, z0, y1, x1);
                    v4f32 _v100 = gridsample_gather_depth(pp, w, h, d, z1, y0, x0);
                    v4f32 _v101 = gridsample_gather_depth(pp, w, h, d, z1, y0, x1);
                    v4f32 _v110 = gridsample_gather_depth(pp, w, h, d, z1, y1, x0);
                    v4f32 _v111 = gridsample_gather_depth(pp, w, h, d, z1, y1, x1);

                    v4f32 _a = __ncnn_altivec_fill_f32(alpha);
                    v4f32 _b = __ncnn_altivec_fill_f32(beta);
                    v4f32 _g = __ncnn_altivec_fill_f32(gamma);
                    v4f32 _1ma = __ncnn_altivec_fill_f32(1.f - alpha);
                    v4f32 _1mb = __ncnn_altivec_fill_f32(1.f - beta);
                    v4f32 _1mg = __ncnn_altivec_fill_f32(1.f - gamma);

                    v4f32 _v00 = vec_add(__ncnn_altivec_mul(_v000, _1ma), __ncnn_altivec_mul(_v001, _a));
                    v4f32 _v01 = vec_add(__ncnn_altivec_mul(_v010, _1ma), __ncnn_altivec_mul(_v011, _a));
                    v4f32 _v10 = vec_add(__ncnn_altivec_mul(_v100, _1ma), __ncnn_altivec_mul(_v101, _a));
                    v4f32 _v11 = vec_add(__ncnn_altivec_mul(_v110, _1ma), __ncnn_altivec_mul(_v111, _a));

                    v4f32 _v0 = vec_add(__ncnn_altivec_mul(_v00, _1mb), __ncnn_altivec_mul(_v01, _b));
                    v4f32 _v1 = vec_add(__ncnn_altivec_mul(_v10, _1mb), __ncnn_altivec_mul(_v11, _b));

                    v4f32 _v = vec_add(__ncnn_altivec_mul(_v0, _1mg), __ncnn_altivec_mul(_v1, _g));

                    for (int k = 0; k < n; k++)
                    {
                        orow[k][x] = vec_extract(_v, k);
                    }
                }

                sxp += outw;
                syp += outw;
                szp += outw;
            }
        }
    }
}

static void gridsample_nearest_4d(const Mat& src, const Mat& offset_blob, Mat& dst, int w, int h, int d, int outw, int outh, int outd, int channels, int num_threads)
{
    const float* offxp = offset_blob.channel(0);
    const float* offyp = offset_blob.channel(1);
    const float* offzp = offset_blob.channel(2);

    #pragma omp parallel for num_threads(num_threads)
    for (int q = 0; q < channels; q += 4)
    {
        const int n = std::min(4, channels - q);

        Mat pp[4];
        for (int k = 0; k < n; k++)
        {
            pp[k] = src.channel(q + k);
        }
        for (int k = n; k < 4; k++)
        {
            pp[k] = pp[0];
        }

        float* op[4];
        for (int k = 0; k < n; k++)
        {
            op[k] = dst.channel(q + k);
        }

        const float* sxp = offxp;
        const float* syp = offyp;
        const float* szp = offzp;

        for (int z = 0; z < outd; z++)
        {
            for (int y = 0; y < outh; y++)
            {
                float* orow[4];
                for (int k = 0; k < n; k++)
                {
                    orow[k] = op[k] + z * outh * outw + y * outw;
                }

                for (int x = 0; x < outw; x++)
                {
                    int x0 = static_cast<int>(floor(sxp[x] + 0.5f));
                    int y0 = static_cast<int>(floor(syp[x] + 0.5f));
                    int z0 = static_cast<int>(floor(szp[x] + 0.5f));

                    v4f32 _v = gridsample_gather_depth(pp, w, h, d, z0, y0, x0);

                    for (int k = 0; k < n; k++)
                    {
                        orow[k][x] = vec_extract(_v, k);
                    }
                }

                sxp += outw;
                syp += outw;
                szp += outw;
            }
        }
    }
}

#endif // __ALTIVEC__

int GridSample_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& grid = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    // non-fp32 or packed input: delegate to the generic implementation
    if (bottom_blob.elemsize != 4u || bottom_blob.elempack != 1 || grid.elemsize != 4u || grid.elempack != 1)
    {
        std::vector<Mat> bottom_blobs_unpacked(2);
        bottom_blobs_unpacked[0] = bottom_blob;
        bottom_blobs_unpacked[1] = grid;

        if (bottom_blob.elempack != 1)
            convert_packing(bottom_blob, bottom_blobs_unpacked[0], 1, opt);
        if (grid.elempack != 1)
            convert_packing(grid, bottom_blobs_unpacked[1], 1, opt);

        return GridSample::forward(bottom_blobs_unpacked, top_blobs, opt);
    }

#if __ALTIVEC__
    const int dims = bottom_blob.dims;

    if ((dims == 3 || dims == 4) && (sample_type == Interpolation_BILINEAR || sample_type == Interpolation_NEAREST))
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;
        size_t elemsize = bottom_blob.elemsize;

        Mat offset_blob;
        int outw;
        int outh;
        int outd;

        if (dims == 3)
        {
            outw = permute_fusion == 0 ? grid.h : grid.w;
            outh = permute_fusion == 0 ? grid.c : grid.h;
            outd = 1;

            top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
            offset_blob.create(outw, outh, grid.c, elemsize, opt.workspace_allocator);
        }
        else
        {
            outw = permute_fusion == 0 ? grid.h : grid.w;
            outh = permute_fusion == 0 ? grid.d : grid.h;
            outd = permute_fusion == 0 ? grid.c : grid.d;

            top_blob.create(outw, outh, outd, channels, elemsize, opt.blob_allocator);
            offset_blob.create(outw, outh, outd, grid.c, elemsize, opt.workspace_allocator);
        }

        if (top_blob.empty() || offset_blob.empty())
            return -100;

        // pre-calculate all interpolation offsets for each x y, unpack grid on-the-fly
        if (dims == 3)
        {
            float* offsetptr_x = offset_blob.channel(0);
            float* offsetptr_y = offset_blob.channel(1);

            if (permute_fusion == 0)
            {
                for (int y = 0; y < outh; y++)
                {
                    const float* gridptr = grid.channel(y);
                    for (int x = 0; x < outw; x++)
                    {
                        float sample_x = gridptr[0];
                        float sample_y = gridptr[1];

                        sample_x = grid_sample_unormalize(w, sample_x, align_corner);
                        sample_y = grid_sample_unormalize(h, sample_y, align_corner);

                        *offsetptr_x = sample_x;
                        *offsetptr_y = sample_y;

                        gridptr += 2;
                        offsetptr_x++;
                        offsetptr_y++;
                    }
                }
            }
            else
            {
                const float* gridptr_x = grid.channel(0);
                const float* gridptr_y = grid.channel(1);

                for (int y = 0; y < outh; y++)
                {
                    for (int x = 0; x < outw; x++)
                    {
                        float sample_x = *gridptr_x;
                        float sample_y = *gridptr_y;

                        sample_x = grid_sample_unormalize(w, sample_x, align_corner);
                        sample_y = grid_sample_unormalize(h, sample_y, align_corner);

                        *offsetptr_x = sample_x;
                        *offsetptr_y = sample_y;

                        gridptr_x++;
                        gridptr_y++;
                        offsetptr_x++;
                        offsetptr_y++;
                    }
                }
            }
        }
        else
        {
            if (permute_fusion == 0)
            {
                float* offsetptr_x = offset_blob.channel(0);
                float* offsetptr_y = offset_blob.channel(1);
                float* offsetptr_z = offset_blob.channel(2);

                for (int z = 0; z < outd; z++)
                {
                    const float* gridptr = grid.channel(z);
                    for (int y = 0; y < outh; y++)
                    {
                        for (int x = 0; x < outw; x++)
                        {
                            float sample_x = gridptr[0];
                            float sample_y = gridptr[1];
                            float sample_z = gridptr[2];

                            sample_x = grid_sample_unormalize(w, sample_x, align_corner);
                            sample_x = compute_coord(sample_x, w, padding_mode, align_corner);

                            sample_y = grid_sample_unormalize(h, sample_y, align_corner);
                            sample_y = compute_coord(sample_y, h, padding_mode, align_corner);

                            sample_z = grid_sample_unormalize(d, sample_z, align_corner);
                            sample_z = compute_coord(sample_z, d, padding_mode, align_corner);

                            *offsetptr_x = sample_x;
                            *offsetptr_y = sample_y;
                            *offsetptr_z = sample_z;

                            gridptr += 3;
                            offsetptr_x++;
                            offsetptr_y++;
                            offsetptr_z++;
                        }
                    }
                }
            }
            else
            {
                const float* gridptr_x = grid.channel(0);
                const float* gridptr_y = grid.channel(1);
                const float* gridptr_z = grid.channel(2);
                float* offsetptr_x = offset_blob.channel(0);
                float* offsetptr_y = offset_blob.channel(1);
                float* offsetptr_z = offset_blob.channel(2);

                for (int z = 0; z < outd; z++)
                {
                    for (int y = 0; y < outh; y++)
                    {
                        for (int x = 0; x < outw; x++)
                        {
                            float sample_x = *gridptr_x;
                            float sample_y = *gridptr_y;
                            float sample_z = *gridptr_z;

                            sample_x = grid_sample_unormalize(w, sample_x, align_corner);
                            sample_x = compute_coord(sample_x, w, padding_mode, align_corner);

                            sample_y = grid_sample_unormalize(h, sample_y, align_corner);
                            sample_y = compute_coord(sample_y, h, padding_mode, align_corner);

                            sample_z = grid_sample_unormalize(d, sample_z, align_corner);
                            sample_z = compute_coord(sample_z, d, padding_mode, align_corner);

                            *offsetptr_x = sample_x;
                            *offsetptr_y = sample_y;
                            *offsetptr_z = sample_z;

                            gridptr_x++;
                            gridptr_y++;
                            gridptr_z++;
                            offsetptr_x++;
                            offsetptr_y++;
                            offsetptr_z++;
                        }
                    }
                }
            }
        }

        if (dims == 3)
        {
            if (sample_type == Interpolation_BILINEAR)
                gridsample_bilinear_3d(bottom_blob, offset_blob, top_blob, padding_mode, align_corner, w, h, outw, outh, channels, opt.num_threads);
            else
                gridsample_nearest_3d(bottom_blob, offset_blob, top_blob, padding_mode, align_corner, w, h, outw, outh, channels, opt.num_threads);
        }
        else
        {
            if (sample_type == Interpolation_BILINEAR)
                gridsample_bilinear_4d(bottom_blob, offset_blob, top_blob, w, h, d, outw, outh, outd, channels, opt.num_threads);
            else
                gridsample_nearest_4d(bottom_blob, offset_blob, top_blob, w, h, d, outw, outh, outd, channels, opt.num_threads);
        }

        return 0;
    }
#endif // __ALTIVEC__

    // bicubic / dims 2 / anything else: exact generic implementation
    return GridSample::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
