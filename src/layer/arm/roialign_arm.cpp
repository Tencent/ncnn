// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "roialign_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

// adapted from detectron2
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/ROIAlign/ROIAlign_cpu.cpp
struct PreCalc
{
    int pos1;
    int pos2;
    int pos3;
    int pos4;
    float w1;
    float w2;
    float w3;
    float w4;
};

static void detectron2_pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int iy_upper,
    const int ix_upper,
    float roi_start_h,
    float roi_start_w,
    float bin_size_h,
    float bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc>& pre_calc)
{
    int pre_calc_index = 0;
    for (int ph = 0; ph < pooled_height; ph++)
    {
        for (int pw = 0; pw < pooled_width; pw++)
        {
            for (int iy = 0; iy < iy_upper; iy++)
            {
                const float yy = roi_start_h + ph * bin_size_h + (iy + 0.5f) * bin_size_h / (float)roi_bin_grid_h; // e.g., 0.5, 1.5
                for (int ix = 0; ix < ix_upper; ix++)
                {
                    const float xx = roi_start_w + pw * bin_size_w + (ix + 0.5f) * bin_size_w / (float)roi_bin_grid_w;

                    float x = xx;
                    float y = yy;
                    // deal with: inverse elements are out of feature map boundary
                    if (y < -1.0 || y > height || x < -1.0 || x > width)
                    {
                        // empty
                        PreCalc pc;
                        pc.pos1 = 0;
                        pc.pos2 = 0;
                        pc.pos3 = 0;
                        pc.pos4 = 0;
                        pc.w1 = 0.f;
                        pc.w2 = 0.f;
                        pc.w3 = 0.f;
                        pc.w4 = 0.f;
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
                        y = (float)y_low;
                    }
                    else
                    {
                        y_high = y_low + 1;
                    }

                    if (x_low >= width - 1)
                    {
                        x_high = x_low = width - 1;
                        x = (float)x_low;
                    }
                    else
                    {
                        x_high = x_low + 1;
                    }

                    float ly = y - y_low;
                    float lx = x - x_low;
                    float hy = 1.f - ly, hx = 1.f - lx;
                    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

                    // save weights and indices
                    PreCalc pc;
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

static void original_pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    float roi_start_h,
    float roi_start_w,
    float bin_size_h,
    float bin_size_w,
    int sampling_ratio,
    std::vector<PreCalc>& pre_calc)
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
                    PreCalc pc;
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

static inline float roialign_interpolate(const float* ptr, const PreCalc* pre_calc, int n)
{
    float sum = 0.f;
    for (int i = 0; i < n; i++)
    {
        const PreCalc& pc = pre_calc[i];
        sum += pc.w1 * ptr[pc.pos1] + pc.w2 * ptr[pc.pos2] + pc.w3 * ptr[pc.pos3] + pc.w4 * ptr[pc.pos4];
    }
    return sum;
}

#if __ARM_NEON
static inline float32x4_t roialign_interpolate_pack4(const float* ptr, const PreCalc* pre_calc, int n)
{
    // accumulate each bilinear corner into an independent lane so the four
    // fmla chains can issue in parallel instead of forming one long
    // latency-bound dependency chain
    float32x4_t _acc0 = vdupq_n_f32(0.f);
    float32x4_t _acc1 = vdupq_n_f32(0.f);
    float32x4_t _acc2 = vdupq_n_f32(0.f);
    float32x4_t _acc3 = vdupq_n_f32(0.f);
    for (int i = 0; i < n; i++)
    {
        const PreCalc& pc = pre_calc[i];
        _acc0 = vmlaq_n_f32(_acc0, vld1q_f32(ptr + pc.pos1 * 4), pc.w1);
        _acc1 = vmlaq_n_f32(_acc1, vld1q_f32(ptr + pc.pos2 * 4), pc.w2);
        _acc2 = vmlaq_n_f32(_acc2, vld1q_f32(ptr + pc.pos3 * 4), pc.w3);
        _acc3 = vmlaq_n_f32(_acc3, vld1q_f32(ptr + pc.pos4 * 4), pc.w4);
    }
    return vaddq_f32(vaddq_f32(_acc0, _acc1), vaddq_f32(_acc2, _acc3));
}
#endif // __ARM_NEON

ROIAlign_arm::ROIAlign_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

int ROIAlign_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const int width = bottom_blob.w;
    const int height = bottom_blob.h;
    const size_t elemsize = bottom_blob.elemsize;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const Mat& roi_blob = bottom_blobs[1];

    Mat& top_blob = top_blobs[0];
    top_blob.create(pooled_width, pooled_height, channels, elemsize, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // For each ROI R = [x y w h]: avg pool over R
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
        std::vector<PreCalc> pre_calc(
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

        // the bilinear sampling layout is independent of the channel, so the
        // per-cell sample offset, count and emptiness are computed once and
        // reused for every channel instead of being recomputed inside the
        // channel loop
        const int num_cells = pooled_width * pooled_height;
        std::vector<int> cell_offset(num_cells);
        std::vector<int> cell_area(num_cells);
        std::vector<int> cell_empty(num_cells);
        {
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

                    int c = ph * pooled_width + pw;
                    cell_offset[c] = pre_calc_index;
                    cell_area[c] = bin_grid_h * bin_grid_w;
                    cell_empty[c] = ((hend <= hstart) || (wend <= wstart)) ? 1 : 0;
                    pre_calc_index += bin_grid_h * bin_grid_w;
                }
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int c = 0; c < num_cells; c++)
            {
                const PreCalc* pc = &pre_calc[cell_offset[c]];
                int area = cell_area[c];

#if __ARM_NEON
                if (elempack == 4)
                {
                    float32x4_t _v = roialign_interpolate_pack4(ptr, pc, area);
                    _v = cell_empty[c] ? vdupq_n_f32(0.f) : vmulq_n_f32(_v, 1.f / (float)area);
                    vst1q_f32(outptr + c * 4, _v);
                    continue;
                }
#endif // __ARM_NEON

                float sum = roialign_interpolate(ptr, pc, area);
                outptr[c] = cell_empty[c] ? 0.f : (sum / (float)area);
            }
        }
    }
    else if (version == 1)
    {
        // the version in detectron 2
        int roi_bin_grid_h = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(roi_height / pooled_height));
        int roi_bin_grid_w = (int)(sampling_ratio > 0 ? sampling_ratio : ceil(roi_width / pooled_width));

        const float count = (float)std::max(roi_bin_grid_h * roi_bin_grid_w, 1);

        std::vector<PreCalc> pre_calc(
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
                    const PreCalc* pc = &pre_calc[pre_calc_index];
                    pre_calc_index += roi_bin_grid_h * roi_bin_grid_w;

#if __ARM_NEON
                    if (elempack == 4)
                    {
                        float32x4_t _v = roialign_interpolate_pack4(ptr, pc, roi_bin_grid_h * roi_bin_grid_w);
                        _v = vmulq_n_f32(_v, 1.f / count);
                        vst1q_f32(outptr + pw * 4, _v);
                        continue;
                    }
#endif // __ARM_NEON

                    float sum = roialign_interpolate(ptr, pc, roi_bin_grid_h * roi_bin_grid_w);
                    outptr[pw] = sum / count;
                }

                outptr += pooled_width * elempack;
            }
        }
    }

    return 0;
}

} // namespace ncnn
