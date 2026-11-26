// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "psroipooling_powerpc.h"

#include <algorithm>
#include <math.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

PSROIPooling_powerpc::PSROIPooling_powerpc()
{
}

int PSROIPooling_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

        return PSROIPooling::forward(bottom_blobs_unpacked, top_blobs, opt);
    }

#if __ALTIVEC__
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int channels = bottom_blob.c;

    if (channels != output_dim * pooled_width * pooled_height)
    {
        // input channel number does not match layer parameters
        return -1;
    }

    top_blob.create(pooled_width, pooled_height, output_dim, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // For each ROI R = [x y w h]: avg pool over R
    const float* roi_ptr = roi_blob;

    float roi_x1 = roundf(roi_ptr[0]) * spatial_scale;
    float roi_y1 = roundf(roi_ptr[1]) * spatial_scale;
    float roi_x2 = roundf(roi_ptr[2] + 1.f) * spatial_scale;
    float roi_y2 = roundf(roi_ptr[3] + 1.f) * spatial_scale;

    float roi_w = std::max(roi_x2 - roi_x1, 0.1f);
    float roi_h = std::max(roi_y2 - roi_y1, 0.1f);

    float bin_size_w = roi_w / (float)pooled_width;
    float bin_size_h = roi_h / (float)pooled_height;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < output_dim; q++)
    {
        float* outptr = top_blob.channel(q);

        for (int ph = 0; ph < pooled_height; ph++)
        {
            for (int pw = 0; pw < pooled_width; pw++)
            {
                const float* ptr = bottom_blob.channel((q * pooled_height + ph) * pooled_width + pw);

                int hstart = static_cast<int>(floorf(roi_y1 + ph * bin_size_h));
                int wstart = static_cast<int>(floorf(roi_x1 + pw * bin_size_w));
                int hend = static_cast<int>(ceilf(roi_y1 + (ph + 1) * bin_size_h));
                int wend = static_cast<int>(ceilf(roi_x1 + (pw + 1) * bin_size_w));

                hstart = std::min(std::max(hstart, 0), h);
                wstart = std::min(std::max(wstart, 0), w);
                hend = std::min(std::max(hend, 0), h);
                wend = std::min(std::max(wend, 0), w);

                bool is_empty = (hend <= hstart) || (wend <= wstart);
                int area = (hend - hstart) * (wend - wstart);

                float sum = 0.f;
                // vectorized sum over the contiguous w runs inside the box
                for (int y = hstart; y < hend; y++)
                {
                    const float* rowp = ptr + y * w + wstart;
                    const int n = wend - wstart;

                    int i = 0;
                    v4f32 _sum = __ncnn_altivec_fill_f32(0.f);
                    for (; i + 3 < n; i += 4)
                    {
                        v4f32 _p = __ncnn_altivec_load(rowp + i);
                        _sum = vec_add(_sum, _p);
                    }
                    sum += __ncnn_altivec_reduce_add(_sum);

                    for (; i < n; i++)
                    {
                        sum += rowp[i];
                    }
                }

                outptr[pw] = is_empty ? 0.f : (sum / (float)area);
            }

            outptr += pooled_width;
        }
    }

    return 0;
#endif // __ALTIVEC__

    return PSROIPooling::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
