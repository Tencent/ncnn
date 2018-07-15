// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "deconvolutiondepthwise.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(DeconvolutionDepthWise)

DeconvolutionDepthWise::DeconvolutionDepthWise()
{
    one_blob_only = true;
    support_inplace = false;
}

int DeconvolutionDepthWise::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_w = pd.get(4, 0);
    pad_h = pd.get(14, pad_w);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    group = pd.get(7, 1);

    return 0;
}

int DeconvolutionDepthWise::load_model(const ModelBin& mb)
{
    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

int DeconvolutionDepthWise::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // deconvolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    if (channels % group != 0 || num_output % group != 0)
    {
        // reject invalid group
        return -100;
    }

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;

    Mat top_blob_bordered;
    if (pad_w > 0 || pad_h > 0)
    {
        top_blob_bordered.create(outw, outh, num_output, elemsize, opt.workspace_allocator);
        if (top_blob_bordered.empty())
            return -100;
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output, elemsize, opt.blob_allocator);
        if (top_blob_bordered.empty())
            return -100;
    }

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = outw * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    // depth-wise
    if (channels == group && group == num_output)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g=0; g<group; g++)
        {
            const float* inptr = bottom_blob.channel(g);
            const float* kptr = (const float*)weight_data + maxk * g;
            Mat m = top_blob_bordered.channel(g);

            const float bias = bias_term ? bias_data[g] : 0.f;

            m.fill(bias);

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    float* outptr = m.row(i*stride_h) + j*stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = inptr[i*w + j];
                        float w = kptr[k];
                        outptr[ space_ofs[k] ] += val * w;
                    }
                }
            }
        }
    }
    else
    {
        // num_output
        const int channels_g = channels / group;
        const int num_output_g = num_output / group;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            const float* weight_data_ptr = (const float*)weight_data + maxk * channels_g * num_output_g * g;
            for (int p = 0; p < num_output_g; p++)
            {
                Mat out = top_blob_bordered.channel(g * num_output_g + p);

                const float bias = bias_term ? bias_data[g * num_output_g + p] : 0.f;

                out.fill(bias);

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        float* outptr = out.row(i*stride_h) + j*stride_w;

                        const float* kptr = weight_data_ptr + maxk * channels_g * p;

                        // channels_g
                        for (int q = 0; q < channels_g; q++)
                        {
                            const Mat m = bottom_blob.channel(channels_g * g + q);
                            float val = *(m.row(i) + j);

                            for (int k = 0; k < maxk; k++)
                            {
                                outptr[ space_ofs[k] ] += val * kptr[k];
                            }

                            kptr += maxk;
                        }
                    }
                }
            }
        }
    }

    if (pad_w > 0 || pad_h > 0)
    {
        copy_cut_border(top_blob_bordered, top_blob, pad_h, pad_h, pad_w, pad_w, opt.blob_allocator, opt.num_threads);
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }
    else
    {
        top_blob = top_blob_bordered;
    }

    return 0;
}

} // namespace ncnn
