// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "pooling3d.h"

#include <float.h>

namespace ncnn {

Pooling3D::Pooling3D()
{
    one_blob_only = true;
    support_inplace = false;
}

int Pooling3D::load_param(const ParamDict& pd)
{
    pooling_type = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    kernel_d = pd.get(21, kernel_w);
    stride_w = pd.get(2, 1);
    stride_h = pd.get(12, stride_w);
    stride_d = pd.get(22, stride_w);
    pad_left = pd.get(3, 0);
    pad_right = pd.get(14, pad_left);
    pad_top = pd.get(13, pad_left);
    pad_bottom = pd.get(15, pad_top);
    pad_front = pd.get(23, pad_left);
    pad_behind = pd.get(16, pad_front);
    global_pooling = pd.get(4, 0);
    pad_mode = pd.get(5, 0);
    avgpool_count_include_pad = pd.get(6, 0);
    adaptive_pooling = pd.get(7, 0);
    out_w = pd.get(8, 0);
    out_h = pd.get(18, out_w);
    out_d = pd.get(28, out_w);

    return 0;
}

int Pooling3D::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // max value in NxN window
    // avg value in NxN window

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    //     NCNN_LOGE("Pooling     input %d x %d  pad = %d %d %d %d  ksize=%d %d  stride=%d %d", w, h, pad_left, pad_right, pad_top, pad_bottom, kernel_w, kernel_h, stride_w, stride_h);
    if (global_pooling)
    {
        top_blob.create(channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = w * h * d;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float max_value = ptr[0];
                for (int i = 0; i < size; i++)
                {
                    max_value = std::max(max_value, ptr[i]);
                }

                top_blob[q] = max_value;
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float sum = 0.f;
                for (int i = 0; i < size; i++)
                {
                    sum += ptr[i];
                }

                top_blob[q] = sum / size;
            }
        }

        return 0;
    }

    if (adaptive_pooling)
    {
        int _out_w = out_w == -233 ? w : out_w;
        int _out_h = out_h == -233 ? h : out_h;
        int _out_d = out_d == -233 ? d : out_d;

        if (_out_w == w && _out_h == h && _out_d == d)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(_out_w, _out_h, _out_d, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* inptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < _out_d; z++)
                {
                    // floor div
                    const int id0 = d * z / _out_d;
                    // ceil div
                    const int id1 = (d * (z + 1) + _out_d - 1) / _out_d;
                    for (int i = 0; i < _out_h; i++)
                    {
                        // floor div
                        const int ih0 = h * i / _out_h;
                        // ceil div
                        const int ih1 = (h * (i + 1) + _out_h - 1) / _out_h;
                        for (int j = 0; j < _out_w; j++)
                        {
                            // floor div
                            const int iw0 = w * j / _out_w;
                            // ceil div
                            const int iw1 = (w * (j + 1) + _out_w - 1) / _out_w;

                            float max_value = inptr[id0 * w * h + ih0 * w + iw0];

                            for (int id = id0; id < id1; id++)
                            {
                                for (int ih = ih0; ih < ih1; ih++)
                                {
                                    for (int iw = iw0; iw < iw1; iw++)
                                    {
                                        max_value = std::max(max_value, inptr[id * w * h + ih * w + iw]);
                                    }
                                }
                            }

                            outptr[j] = max_value;
                        }

                        outptr += _out_w;
                    }
                }
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* inptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < _out_d; z++)
                {
                    // floor div
                    const int id0 = d * z / _out_d;
                    // ceil div
                    const int id1 = (d * (z + 1) + _out_d - 1) / _out_d;
                    int dk = id1 - id0;
                    for (int i = 0; i < _out_h; i++)
                    {
                        // floor div
                        const int ih0 = h * i / _out_h;
                        // ceil div
                        const int ih1 = (h * (i + 1) + _out_h - 1) / _out_h;
                        int hk = ih1 - ih0;
                        for (int j = 0; j < _out_w; j++)
                        {
                            // floor div
                            const int iw0 = w * j / _out_w;
                            // ceil div
                            const int iw1 = (w * (j + 1) + _out_w - 1) / _out_w;
                            int wk = iw1 - iw0;

                            float sum = 0;
                            for (int id = id0; id < id1; id++)
                            {
                                for (int ih = ih0; ih < ih1; ih++)
                                {
                                    for (int iw = iw0; iw < iw1; iw++)
                                    {
                                        sum += inptr[id * w * h + ih * w + iw];
                                    }
                                }
                            }

                            outptr[j] = sum / hk / wk / dk;
                        }

                        outptr += _out_w;
                    }
                }
            }
        }

        return 0;
    }

    Mat bottom_blob_bordered;
    Option opt_pad = opt;
    opt_pad.use_packing_layout = false;
    make_padding(bottom_blob, bottom_blob_bordered, opt_pad);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;
    d = bottom_blob_bordered.d;

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;
    int outd = (d - kernel_d) / stride_d + 1;

    top_blob.create(outw, outh, outd, channels, elemsize);
    if (top_blob.empty())
        return -100;

    const int maxk = kernel_w * kernel_h * kernel_d;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap0 = w - kernel_w;
        int gap1 = h * w - w * kernel_h;
        for (int z = 0; z < kernel_d; z++)
        {
            for (int i = 0; i < kernel_h; i++)
            {
                for (int j = 0; j < kernel_w; j++)
                {
                    space_ofs[p1] = p2;
                    p1++;
                    p2 += 1;
                }
                p2 += gap0;
            }
            p2 += gap1;
        }
    }

    if (pooling_type == PoolMethod_MAX)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);
            for (int z = 0; z < outd; z++)
            {
                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const float* sptr = m.depth(z * stride_d).row(i * stride_h) + j * stride_w;

                        float max_value = sptr[0];

                        for (int l = 0; l < maxk; l++)
                        {
                            float val = sptr[space_ofs[l]];
                            max_value = std::max(max_value, val);
                        }

                        outptr[j] = max_value;
                    }

                    outptr += outw;
                }
            }
        }
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        if (avgpool_count_include_pad == 0)
        {
            int wtailpad = 0;
            int htailpad = 0;
            int dtailpad = 0;

            if (pad_mode == 0) // full padding
            {
                wtailpad = bottom_blob_bordered.w - bottom_blob.w - pad_left - pad_right;
                htailpad = bottom_blob_bordered.h - bottom_blob.h - pad_top - pad_bottom;
                dtailpad = bottom_blob_bordered.d - bottom_blob.d - pad_front - pad_behind;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < outd; z++)
                {
                    int sz0 = z * stride_d;

                    for (int i = 0; i < outh; i++)
                    {
                        int sy0 = i * stride_h;

                        for (int j = 0; j < outw; j++)
                        {
                            int sx0 = j * stride_w;

                            float sum = 0;
                            int area = 0;
                            for (int kd = 0; kd < kernel_d; kd++)
                            {
                                int sz = sz0 + kd;

                                if (sz < pad_front)
                                    continue;

                                if (sz >= d - pad_behind - dtailpad)
                                    break;

                                for (int ki = 0; ki < kernel_h; ki++)
                                {
                                    int sy = sy0 + ki;

                                    if (sy < pad_top)
                                        continue;

                                    if (sy >= h - pad_bottom - htailpad)
                                        break;

                                    for (int kj = 0; kj < kernel_w; kj++)
                                    {
                                        int sx = sx0 + kj;

                                        if (sx < pad_left)
                                            continue;

                                        if (sx >= w - pad_right - wtailpad)
                                            break;

                                        float val = m.depth(sz).row(sy)[sx];
                                        sum += val;
                                        area += 1;
                                    }
                                }
                            }

                            outptr[j] = sum / area;
                        }

                        outptr += outw;
                    }
                }
            }
        }
        else // if (avgpool_count_include_pad == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < outd; z++)
                {
                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            const float* sptr = m.depth(z * stride_d).row(i * stride_h) + j * stride_w;

                            float sum = 0;

                            for (int l = 0; l < maxk; l++)
                            {
                                float val = sptr[space_ofs[l]];
                                sum += val;
                            }

                            outptr[j] = sum / maxk;
                        }

                        outptr += outw;
                    }
                }
            }
        }
    }

    return 0;
}

void Pooling3D::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;

    bottom_blob_bordered = bottom_blob;

    float pad_value = 0.f;
    if (pooling_type == PoolMethod_MAX)
    {
        pad_value = bottom_blob.elemsize == 1 ? -128.f : -FLT_MAX;
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        pad_value = 0.f;
    }

    int wtailpad = 0;
    int htailpad = 0;
    int dtailpad = 0;

    if (pad_mode == 0) // full padding
    {
        int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
        int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;
        int dtail = (d + pad_front + pad_behind - kernel_d) % stride_d;
        if (wtail != 0)
            wtailpad = stride_w - wtail;
        if (htail != 0)
            htailpad = stride_h - htail;
        if (dtail != 0)
            dtailpad = stride_d - dtail;

        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border_3d(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom + htailpad, pad_left, pad_right + wtailpad, pad_front, pad_behind + dtailpad, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_mode == 1) // valid padding
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border_3d(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, pad_front, pad_behind, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_mode == 2) // tensorflow padding=SAME or onnx padding=SAME_UPPER
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        int dpad = kernel_d + (d - 1) / stride_d * stride_d - d;
        if (wpad > 0 || hpad > 0 || dpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border_3d(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, dpad / 2, dpad - dpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_mode == 3) // onnx padding=SAME_LOWER
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        int dpad = kernel_d + (d - 1) / stride_d * stride_d - d;
        if (wpad > 0 || hpad > 0 || dpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border_3d(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, dpad / 2, dpad - dpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}

} //namespace ncnn
