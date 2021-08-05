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

#include "pooling1d.h"

#include "layer_type.h"

#include <float.h>

namespace ncnn {

Pooling1D::Pooling1D()
{
    one_blob_only = true;
    support_inplace = false;
}

int Pooling1D::load_param(const ParamDict& pd)
{
    pooling_type = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    stride_w = pd.get(2, 1);
    pad_left = pd.get(3, 0);
    pad_right = pd.get(14, pad_left);
    global_pooling = pd.get(4, 0);
    pad_mode = pd.get(5, 0);
    avgpool_count_include_pad = pd.get(6, 0);
    adaptive_pooling = pd.get(7, 0);
    out_w = pd.get(8, 0);

    return 0;
}

int Pooling1D::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    //     NCNN_LOGE("Pooling1D     input %d x %d  pad = %d %d  ksize=%d  stride=%d", w, h, pad_left, pad_right, kernel_w, stride_w);
    if (global_pooling)
    {
        top_blob.create(h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                const float* ptr = bottom_blob.row(q);

                float max = ptr[0];
                for (int i = 0; i < w; i++)
                {
                    max = std::max(max, ptr[i]);
                }

                top_blob[q] = max;
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                const float* ptr = bottom_blob.row(q);

                float sum = 0.f;
                for (int i = 0; i < w; i++)
                {
                    sum += ptr[i];
                }

                top_blob[q] = sum / w;
            }
        }

        return 0;
    }

    if (adaptive_pooling)
    {
        top_blob.create(out_w, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                const float* inptr = bottom_blob.row(q);
                float* outptr = top_blob.row(q);

                const int wk = std::max(w - out_w + 1, 1);

                for (int j = 0; j < out_w; j++)
                {
                    int iw0 = out_w == 1 ? 0 : j * (w - wk) / (out_w - 1);
                    int iw1 = iw0 + wk;

                    float max = inptr[iw0];
                    for (int iw = iw0; iw < iw1; iw++)
                    {
                        max = std::max(max, inptr[iw]);
                    }

                    outptr[j] = max;
                }
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                const float* inptr = bottom_blob.row(q);
                float* outptr = top_blob.row(q);

                const int wk = std::max(w - out_w + 1, 1);

                for (int j = 0; j < out_w; j++)
                {
                    int iw0 = out_w == 1 ? 0 : j * (w - wk) / (out_w - 1);
                    int iw1 = iw0 + wk;

                    float sum = 0;
                    for (int iw = iw0; iw < iw1; iw++)
                    {
                        sum += inptr[iw];
                    }

                    outptr[j] = sum / wk;
                }
            }
        }

        return 0;
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_w) / stride_w + 1;

    top_blob.create(outw, h, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (pooling_type == PoolMethod_MAX)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < h; q++)
        {
            const float* ptr = bottom_blob_bordered.row(q);
            float* outptr = top_blob.row(q);

            for (int j = 0; j < outw; j++)
            {
                const float* sptr = ptr + j * stride_w;

                float max = sptr[0];

                for (int k = 0; k < kernel_w; k++)
                {
                    float val = sptr[k];
                    max = std::max(max, val);
                }

                outptr[j] = max;
            }
        }
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        if (avgpool_count_include_pad == 0)
        {
            int wtailpad = 0;

            if (pad_mode == 0) // full padding
            {
                wtailpad = bottom_blob_bordered.w - bottom_blob.w - pad_left - pad_right;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                const float* ptr = bottom_blob_bordered.row(q);
                float* outptr = top_blob.row(q);

                for (int j = 0; j < outw; j++)
                {
                    int sx0 = j * stride_w;

                    float sum = 0;
                    int area = 0;

                    for (int kj = 0; kj < kernel_w; kj++)
                    {
                        int sx = sx0 + kj;

                        if (sx < pad_left)
                            continue;

                        if (sx >= w - pad_right - wtailpad)
                            break;

                        float val = ptr[sx];
                        sum += val;
                        area += 1;
                    }

                    outptr[j] = sum / area;
                }
            }
        }
        else // if (avgpool_count_include_pad == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                const float* ptr = bottom_blob_bordered.row(q);
                float* outptr = top_blob.row(q);

                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = ptr + j * stride_w;

                    float sum = 0;

                    for (int k = 0; k < kernel_w; k++)
                    {
                        float val = sptr[k];
                        sum += val;
                    }

                    outptr[j] = sum / kernel_w;
                }
            }
        }
    }

    return 0;
}

void Pooling1D::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const
{
    int w = bottom_blob.w;

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

    if (pad_mode == 0) // full padding
    {
        int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;

        if (wtail != 0)
            wtailpad = stride_w - wtail;

        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, 0, 0, pad_left, pad_right + wtailpad, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_mode == 1) // valid padding
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, 0, 0, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_mode == 2) // tensorflow padding=SAME or onnx padding=SAME_UPPER
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        if (wpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, 0, 0, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_mode == 3) // onnx padding=SAME_LOWER
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        if (wpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, 0, 0, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}

} // namespace ncnn
