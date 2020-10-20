// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "roialign_x86.h"

#include <math.h>

namespace ncnn {

// adapted from detectron2
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/ROIAlign/ROIAlign_cpu.cpp
template<typename T>
struct PreCalc
{
    int pos1;
    int pos2;
    int pos3;
    int pos4;
    T w1;
    T w2;
    T w3;
    T w4;
};

template<typename T>
void detectron2_pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int iy_upper,
    const int ix_upper,
    T roi_start_h,
    T roi_start_w,
    T bin_size_h,
    T bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc<T> >& pre_calc)
{
    int pre_calc_index = 0;
    for (int ph = 0; ph < pooled_height; ph++)
    {
        for (int pw = 0; pw < pooled_width; pw++)
        {
            for (int iy = 0; iy < iy_upper; iy++)
            {
                const T yy = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
                for (int ix = 0; ix < ix_upper; ix++)
                {
                    const T xx = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

                    T x = xx;
                    T y = yy;
                    // deal with: inverse elements are out of feature map boundary
                    if (y < -1.0 || y > height || x < -1.0 || x > width)
                    {
                        // empty
                        PreCalc<T> pc;
                        pc.pos1 = 0;
                        pc.pos2 = 0;
                        pc.pos3 = 0;
                        pc.pos4 = 0;
                        pc.w1 = 0;
                        pc.w2 = 0;
                        pc.w3 = 0;
                        pc.w4 = 0;
                        pre_calc[pre_calc_index++] = pc;
                        continue;
                    }

                    if (y <= 0)
                    {
                        y = 0;
                    }
                    if (x <= 0)
                    {
                        x = 0;
                    }

                    int y_low = (int)y;
                    int x_low = (int)x;
                    int y_high;
                    int x_high;

                    if (y_low >= height - 1)
                    {
                        y_high = y_low = height - 1;
                        y = (T)y_low;
                    }
                    else
                    {
                        y_high = y_low + 1;
                    }

                    if (x_low >= width - 1)
                    {
                        x_high = x_low = width - 1;
                        x = (T)x_low;
                    }
                    else
                    {
                        x_high = x_low + 1;
                    }

                    T ly = y - y_low;
                    T lx = x - x_low;
                    T hy = (T)(1. - ly), hx = (T)(1. - lx);
                    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

                    // save weights and indices
                    PreCalc<T> pc;
                    pc.pos1 = y_low * width + x_low;
                    pc.pos2 = y_low * width + x_high;
                    pc.pos3 = y_high * width + x_low;
                    pc.pos4 = y_high * width + x_high;
                    pc.w1 = w1;
                    pc.w2 = w2;
                    pc.w3 = w3;
                    pc.w4 = w4;
                    pre_calc[pre_calc_index++] = pc;
                }
            }
        }
    }
}

template<typename T>
void original_pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    T roi_start_h,
    T roi_start_w,
    T bin_size_h,
    T bin_size_w,
    int sampling_ratio,
    std::vector<PreCalc<T> >& pre_calc)
{
    int pre_calc_index = 0;
    for (int ph = 0; ph < pooled_height; ph++)
    {
        for (int pw = 0; pw < pooled_width; pw++)
        {
            float hstart = roi_start_h + ph * bin_size_h;
            float wstart = roi_start_w + pw * bin_size_w;
            float hend = roi_start_h + (ph + 1) * bin_size_h;
            float wend = roi_start_w + (pw + 1) * bin_size_w;
            hstart = std::min(std::max(hstart, 0.f), (float)height);
            wstart = std::min(std::max(wstart, 0.f), (float)width);
            hend = std::min(std::max(hend, 0.f), (float)height);
            wend = std::min(std::max(wend, 0.f), (float)width);

            int bin_grid_h = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(hend - hstart));
            int bin_grid_w = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(wend - wstart));

            for (int by = 0; by < bin_grid_h; by++)
            {
                float y = hstart + (by + 0.5f) * bin_size_h / (float)bin_grid_h;

                for (int bx = 0; bx < bin_grid_w; bx++)
                {
                    float x = wstart + (bx + 0.5f) * bin_size_w / (float)bin_grid_w;
                    int x0 = (int)x;
                    int x1 = x0 + 1;
                    int y0 = (int)y;
                    int y1 = y0 + 1;

                    float a0 = x1 - x;
                    float a1 = x - x0;
                    float b0 = y1 - y;
                    float b1 = y - y0;

                    if (x1 >= width)
                    {
                        x1 = width - 1;
                        a0 = 1.f;
                        a1 = 0.f;
                    }
                    if (y1 >= height)
                    {
                        y1 = height - 1;
                        b0 = 1.f;
                        b1 = 0.f;
                    }
                    // save weights and indices
                    PreCalc<T> pc;
                    pc.pos1 = y0 * width + x0;
                    pc.pos2 = y0 * width + x1;
                    pc.pos3 = y1 * width + x0;
                    pc.pos4 = y1 * width + x1;
                    pc.w1 = a0 * b0;
                    pc.w2 = a1 * b0;
                    pc.w3 = a0 * b1;
                    pc.w4 = a1 * b1;
                    pre_calc[pre_calc_index++] = pc;
                }
            }
        }
    }
}

ROIAlign_x86::ROIAlign_x86()
{
}

int ROIAlign_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const int width = bottom_blob.w;
    const int height = bottom_blob.h;
    const size_t elemsize = bottom_blob.elemsize;
    const int channels = bottom_blob.c;

    const Mat& roi_blob = bottom_blobs[1];

    Mat& top_blob = top_blobs[0];
    top_blob.create(pooled_width, pooled_height, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // For each ROI R = [x y w h]: max pool over R
    const float* roi_ptr = roi_blob;

    float roi_start_w = roi_ptr[0] * spatial_scale;
    float roi_start_h = roi_ptr[1] * spatial_scale;
    float roi_end_w = roi_ptr[2] * spatial_scale;
    float roi_end_h = roi_ptr[3] * spatial_scale;
    if (aligned)
    {
        roi_start_w -= 0.5f;
        roi_start_h -= 0.5f;
        roi_end_w -= 0.5f;
        roi_end_h -= 0.5f;
    }

    float roi_width = roi_end_w - roi_start_w;
    float roi_height = roi_end_h - roi_start_h;

    if (!aligned)
    {
        roi_width = std::max(roi_width, 1.f);
        roi_height = std::max(roi_height, 1.f);
    }

    float bin_size_w = (float)roi_width / (float)pooled_width;
    float bin_size_h = (float)roi_height / (float)pooled_height;

    if (version == 0)
    {
        // original version
        int roi_bin_grid_h = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(roi_height / pooled_height));
        int roi_bin_grid_w = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(roi_width / pooled_width));
        std::vector<PreCalc<float> > pre_calc(
            (size_t)roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
        original_pre_calc_for_bilinear_interpolate(
            height,
            width,
            pooled_height,
            pooled_width,
            roi_start_h,
            roi_start_w,
            bin_size_h,
            bin_size_w,
            sampling_ratio,
            pre_calc);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);
            int pre_calc_index = 0;

            for (int ph = 0; ph < pooled_height; ph++)
            {
                for (int pw = 0; pw < pooled_width; pw++)
                {
                    // Compute pooling region for this output unit:
                    //  start (included) = ph * roi_height / pooled_height
                    //  end (excluded) = (ph + 1) * roi_height / pooled_height
                    float hstart = roi_start_h + ph * bin_size_h;
                    float wstart = roi_start_w + pw * bin_size_w;
                    float hend = roi_start_h + (ph + 1) * bin_size_h;
                    float wend = roi_start_w + (pw + 1) * bin_size_w;

                    hstart = std::min(std::max(hstart, 0.f), (float)height);
                    wstart = std::min(std::max(wstart, 0.f), (float)width);
                    hend = std::min(std::max(hend, 0.f), (float)height);
                    wend = std::min(std::max(wend, 0.f), (float)width);

                    int bin_grid_h = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(hend - hstart));
                    int bin_grid_w = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(wend - wstart));

                    bool is_empty = (hend <= hstart) || (wend <= wstart);
                    int area = bin_grid_h * bin_grid_w;

                    float sum = 0.f;
                    for (int by = 0; by < bin_grid_h; by++)
                    {
                        for (int bx = 0; bx < bin_grid_w; bx++)
                        {
                            PreCalc<float>& pc = pre_calc[pre_calc_index++];
                            // bilinear interpolate at (x,y)
                            sum += pc.w1 * ptr[pc.pos1] + pc.w2 * ptr[pc.pos2] + pc.w3 * ptr[pc.pos3] + pc.w4 * ptr[pc.pos4];
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
        int roi_bin_grid_h = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(roi_height / pooled_height));
        int roi_bin_grid_w = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(roi_width / pooled_width));

        const float count = (float)std::max(roi_bin_grid_h * roi_bin_grid_w, 1);

        std::vector<PreCalc<float> > pre_calc(
            (size_t)roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
        detectron2_pre_calc_for_bilinear_interpolate(
            height,
            width,
            pooled_height,
            pooled_width,
            roi_bin_grid_h,
            roi_bin_grid_w,
            roi_start_h,
            roi_start_w,
            bin_size_h,
            bin_size_w,
            roi_bin_grid_h,
            roi_bin_grid_w,
            pre_calc);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);
            int pre_calc_index = 0;

            for (int ph = 0; ph < pooled_height; ph++)
            {
                for (int pw = 0; pw < pooled_width; pw++)
                {
                    float output_val = 0.f;
                    for (int iy = 0; iy < roi_bin_grid_h; iy++)
                    {
                        for (int ix = 0; ix < roi_bin_grid_w; ix++)
                        {
                            PreCalc<float>& pc = pre_calc[pre_calc_index++];

                            output_val += pc.w1 * ptr[pc.pos1] + pc.w2 * ptr[pc.pos2] + pc.w3 * ptr[pc.pos3] + pc.w4 * ptr[pc.pos4];
                        }
                    }
                    output_val /= count;
                    outptr[pw] = output_val;
                }
                outptr += pooled_width;
            }
        }
    }

    return 0;
}

} // namespace ncnn
