// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "roialign.h"

#include <assert.h>
#include <math.h>

namespace ncnn {

ROIAlign::ROIAlign()
{
}

int ROIAlign::load_param(const ParamDict& pd)
{
    pooled_width = pd.get(0, 0);
    pooled_height = pd.get(1, 0);
    spatial_scale = pd.get(2, 1.f);
    sampling_ratio = pd.get(3, 0);
    aligned = pd.get(4, false);
    version = pd.get(5, 0);
    /*
     * version 0:
     *  the original version of ROIAlign in ncnn
     * version 1:
     *  the version in detectron2
     */
    assert(version >= 0 && version <= 1);

    return 0;
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

int ROIAlign::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int channels = bottom_blob.c;

    const Mat& roi_blob = bottom_blobs[1];

    Mat& top_blob = top_blobs[0];
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
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int ph = 0; ph < pooled_height; ph++)
            {
                for (int pw = 0; pw < pooled_width; pw++)
                {
                    // Compute pooling region for this output unit:
                    //  start (included) = ph * roi_height / pooled_height
                    //  end (excluded) = (ph + 1) * roi_height / pooled_height
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

                    float sum = 0.f;
                    for (int by = 0; by < bin_grid_h; by++)
                    {
                        float y = hstart + (by + 0.5f) * bin_size_h / (float)bin_grid_h;

                        for (int bx = 0; bx < bin_grid_w; bx++)
                        {
                            float x = wstart + (bx + 0.5f) * bin_size_w / (float)bin_grid_w;

                            // bilinear interpolate at (x,y)
                            float v = bilinear_interpolate(ptr, w, h, x, y);

                            sum += v;
                        }
                    }

                    outptr[pw] = is_empty ? 0.f : (sum / (float)area);
                }

                outptr += pooled_width;
            }
        }
    }
    else if (version == 1)
    {
        // the version in detectron 2
        int roi_bin_grid_h = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(roi_h / pooled_height));
        int roi_bin_grid_w = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(roi_w / pooled_width));

        const float count = (float)std::max(roi_bin_grid_h * roi_bin_grid_w, 1);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int ph = 0; ph < pooled_height; ph++)
            {
                for (int pw = 0; pw < pooled_width; pw++)
                {
                    float sum = 0.f;
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
                                float v = bilinear_interpolate(ptr, w, h, x, y);
                                sum += v;
                            }
                        }
                    }
                    outptr[pw] = sum / count;
                }

                outptr += pooled_width;
            }
        }
    }

    return 0;
}

} // namespace ncnn
