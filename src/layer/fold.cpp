// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "fold.h"

namespace ncnn {

Fold::Fold()
{
    one_blob_only = true;
}

int Fold::load_param(const ParamDict& pd)
{
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_top = pd.get(14, pad_left);
    pad_bottom = pd.get(16, pad_top);
    output_w = pd.get(20, 0);
    output_h = pd.get(21, output_w);

    return 0;
}

int Fold::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int size = bottom_blob.w;
    const int max_channels = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int outw = output_w + pad_left + pad_right;
    const int outh = output_h + pad_top + pad_bottom;

    const int inw = (outw - kernel_extent_w) / stride_w + 1;
    const int inh = (outh - kernel_extent_h) / stride_h + 1;

    // assert inw * inh == size

    const int maxk = kernel_w * kernel_h;
    const int channels = max_channels / maxk;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        top_blob_bordered.create(outw, outh, channels, elemsize, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, channels, elemsize, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    // col2im
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < channels; p++)
    {
        float* ptr = top_blob_bordered.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                const int sx_start = (j < kernel_extent_w) ? 0 : (j - kernel_extent_w) / stride_w + 1;
                const int sx_end = std::min(j / stride_w + 1, inw);

                const int sy_start = (i < kernel_extent_h) ? 0 : (i - kernel_extent_h) / stride_h + 1;
                const int sy_end = std::min(i / stride_h + 1, inh);

                for (int sy = sy_start; sy < sy_end; sy += 1)
                {
                    for (int sx = sx_start; sx < sx_end; sx += 1)
                    {
                        int h_k = (i - sy * stride_h);
                        int w_k = (j - sx * stride_w);

                        if (h_k % dilation_h == 0 && w_k % dilation_w == 0)
                        {
                            h_k /= dilation_h;
                            w_k /= dilation_w;

                            sum += bottom_blob.row(p * maxk + h_k * kernel_w + w_k)[sy * inw + sx];
                        }
                    }
                }

                ptr[0] = sum;
                ptr += 1;
            }
        }
    }

    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_b = opt;
        opt_b.use_packing_layout = false;
        copy_cut_border(top_blob_bordered, top_blob, pad_top, pad_bottom, pad_left, pad_right, opt_b);
        if (top_blob.empty())
            return -100;
    }
    else
    {
        top_blob = top_blob_bordered;
    }

    return 0;
}

} // namespace ncnn
