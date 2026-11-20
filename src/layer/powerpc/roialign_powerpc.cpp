// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "roialign_powerpc.h"

#include <algorithm>
#include <math.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

ROIAlign_powerpc::ROIAlign_powerpc()
{
}

static inline float bilinear_interpolate(const float* ptr, int w, int h, float x, float y)
{
    int x0 = (int)x;
    int x1 = x0 + 1;
    int y0 = (int)y;
    int y1 = y0 + 1;

    float a0 = x1 - x;
    float a1 = x - x0;
    float b0 = y1 - y;
    float b1 = y - y0;

    if (x1 >= w)
    {
        x1 = w - 1;
        a0 = 1.f;
        a1 = 0.f;
    }
    if (y1 >= h)
    {
        y1 = h - 1;
        b0 = 1.f;
        b1 = 0.f;
    }

    float r0 = ptr[y0 * w + x0] * a0 + ptr[y0 * w + x1] * a1;
    float r1 = ptr[y1 * w + x0] * a0 + ptr[y1 * w + x1] * a1;

    float v = r0 * b0 + r1 * b1;

    return v;
}

#if __ALTIVEC__
// bilinear interpolate the same (x, y) position from 4 channel planes at once
// channels beyond n contribute zero lanes so reduce_add stays correct
static inline v4f32 bilinear_interpolate_4c(const float* const* pp, int n, int w, int h, float x, float y)
{
    int x0 = (int)x;
    int x1 = x0 + 1;
    int y0 = (int)y;
    int y1 = y0 + 1;

    float a0 = x1 - x;
    float a1 = x - x0;
    float b0 = y1 - y;
    float b1 = y - y0;

    if (x1 >= w)
    {
        x1 = w - 1;
        a0 = 1.f;
        a1 = 0.f;
    }
    if (y1 >= h)
    {
        y1 = h - 1;
        b0 = 1.f;
        b1 = 0.f;
    }

    float t00[4];
    float t01[4];
    float t10[4];
    float t11[4];
    for (int k = 0; k < 4; k++)
    {
        if (k < n)
        {
            t00[k] = pp[k][y0 * w + x0];
            t01[k] = pp[k][y0 * w + x1];
            t10[k] = pp[k][y1 * w + x0];
            t11[k] = pp[k][y1 * w + x1];
        }
        else
        {
            t00[k] = 0.f;
            t01[k] = 0.f;
            t10[k] = 0.f;
            t11[k] = 0.f;
        }
    }

    v4f32 _a0 = __ncnn_altivec_fill_f32(a0);
    v4f32 _a1 = __ncnn_altivec_fill_f32(a1);
    v4f32 _b0 = __ncnn_altivec_fill_f32(b0);
    v4f32 _b1 = __ncnn_altivec_fill_f32(b1);

    v4f32 _r0 = vec_add(__ncnn_altivec_mul(__ncnn_altivec_load(t00), _a0), __ncnn_altivec_mul(__ncnn_altivec_load(t01), _a1));
    v4f32 _r1 = vec_add(__ncnn_altivec_mul(__ncnn_altivec_load(t10), _a0), __ncnn_altivec_mul(__ncnn_altivec_load(t11), _a1));

    return vec_add(__ncnn_altivec_mul(_r0, _b0), __ncnn_altivec_mul(_r1, _b1));
}
#endif // __ALTIVEC__

int ROIAlign_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& roi_blob = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    // non-fp32 or packed input: delegate to the generic implementation
    if (bottom_blob.elemsize != 4u || bottom_blob.elempack != 1 || roi_blob.elemsize != 4u || roi_blob.elempack != 1)
    {
        std::vector<Mat> bottom_blobs_unpacked(2);
        bottom_blobs_unpacked[0] = bottom_blob;
        bottom_blobs_unpacked[1] = roi_blob;

        if (bottom_blob.elempack != 1)
            convert_packing(bottom_blob, bottom_blobs_unpacked[0], 1, opt);
        if (roi_blob.elempack != 1)
            convert_packing(roi_blob, bottom_blobs_unpacked[1], 1, opt);

        return ROIAlign::forward(bottom_blobs_unpacked, top_blobs, opt);
    }

#if __ALTIVEC__
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int channels = bottom_blob.c;

    top_blob.create(pooled_width, pooled_height, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // For each ROI R = [x y w h]: avg pool over R
    const float* roi_ptr = roi_blob;

    float roi_x1 = roi_ptr[0] * spatial_scale;
    float roi_y1 = roi_ptr[1] * spatial_scale;
    float roi_x2 = roi_ptr[2] * spatial_scale;
    float roi_y2 = roi_ptr[3] * spatial_scale;
    if (aligned)
    {
        roi_x1 -= 0.5f;
        roi_y1 -= 0.5f;
        roi_x2 -= 0.5f;
        roi_y2 -= 0.5f;
    }

    float roi_w = roi_x2 - roi_x1;
    float roi_h = roi_y2 - roi_y1;

    if (!aligned)
    {
        roi_w = std::max(roi_w, 1.f);
        roi_h = std::max(roi_h, 1.f);
    }

    float bin_size_w = roi_w / (float)pooled_width;
    float bin_size_h = roi_h / (float)pooled_height;

    if (version == 0)
    {
        // original version
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q += 4)
        {
            const int n = std::min(4, channels - q);

            const float* pp[4];
            for (int k = 0; k < n; k++)
                pp[k] = bottom_blob.channel(q + k);
            for (int k = n; k < 4; k++)
                pp[k] = pp[0];

            float* op[4];
            for (int k = 0; k < n; k++)
                op[k] = top_blob.channel(q + k);

            for (int ph = 0; ph < pooled_height; ph++)
            {
                for (int pw = 0; pw < pooled_width; pw++)
                {
                    float hstart = roi_y1 + ph * bin_size_h;
                    float wstart = roi_x1 + pw * bin_size_w;
                    float hend = roi_y1 + (ph + 1) * bin_size_h;
                    float wend = roi_x1 + (pw + 1) * bin_size_w;

                    hstart = std::min(std::max(hstart, 0.f), (float)h);
                    wstart = std::min(std::max(wstart, 0.f), (float)w);
                    hend = std::min(std::max(hend, 0.f), (float)h);
                    wend = std::min(std::max(wend, 0.f), (float)w);

                    int bin_grid_h = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(hend - hstart));
                    int bin_grid_w = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(wend - wstart));

                    bool is_empty = (hend <= hstart) || (wend <= wstart);
                    int area = bin_grid_h * bin_grid_w;

                    v4f32 _sum = __ncnn_altivec_fill_f32(0.f);
                    for (int by = 0; by < bin_grid_h; by++)
                    {
                        float y = hstart + (by + 0.5f) * bin_size_h / (float)bin_grid_h;

                        for (int bx = 0; bx < bin_grid_w; bx++)
                        {
                            float x = wstart + (bx + 0.5f) * bin_size_w / (float)bin_grid_w;

                            // bilinear interpolate at (x,y)
                            v4f32 _v = bilinear_interpolate_4c(pp, n, w, h, x, y);

                            _sum = vec_add(_sum, _v);
                        }
                    }

                    float sum_area = (float)area;

                    for (int k = 0; k < n; k++)
                    {
                        op[k][ph * pooled_width + pw] = is_empty ? 0.f : (vec_extract(_sum, k) / sum_area);
                    }
                }
            }
        }

        return 0;
    }
    else if (version == 1)
    {
        // the version in detectron 2
        int roi_bin_grid_h = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(roi_h / pooled_height));
        int roi_bin_grid_w = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(roi_w / pooled_width));

        const float count = (float)std::max(roi_bin_grid_h * roi_bin_grid_w, 1);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q += 4)
        {
            const int n = std::min(4, channels - q);

            const float* pp[4];
            for (int k = 0; k < n; k++)
                pp[k] = bottom_blob.channel(q + k);
            for (int k = n; k < 4; k++)
                pp[k] = pp[0];

            float* op[4];
            for (int k = 0; k < n; k++)
                op[k] = top_blob.channel(q + k);

            for (int ph = 0; ph < pooled_height; ph++)
            {
                for (int pw = 0; pw < pooled_width; pw++)
                {
                    v4f32 _sum = __ncnn_altivec_fill_f32(0.f);
                    for (int by = 0; by < roi_bin_grid_h; by++)
                    {
                        float y = roi_y1 + ph * bin_size_h + (by + 0.5f) * bin_size_h / (float)roi_bin_grid_h;

                        for (int bx = 0; bx < roi_bin_grid_w; bx++)
                        {
                            float x = roi_x1 + pw * bin_size_w + (bx + 0.5f) * bin_size_w / (float)roi_bin_grid_w;

                            if (y < -1.0 || y > h || x < -1.0 || x > w)
                            {
                                // empty
                                continue;
                            }
                            else
                            {
                                if (y <= 0) y = 0;
                                if (x <= 0) x = 0;

                                // bilinear interpolate at (x,y)
                                v4f32 _v = bilinear_interpolate_4c(pp, n, w, h, x, y);
                                _sum = vec_add(_sum, _v);
                            }
                        }
                    }

                    v4f32 _out = __ncnn_altivec_div2(_sum, __ncnn_altivec_fill_f32(count));

                    for (int k = 0; k < n; k++)
                    {
                        op[k][ph * pooled_width + pw] = vec_extract(_out, k);
                    }
                }
            }
        }

        return 0;
    }

    return 0;
#endif // __ALTIVEC__

    return ROIAlign::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
