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

DeconvolutionDepthWise::~DeconvolutionDepthWise()
{
}

int DeconvolutionDepthWise::load_param(const ParamDict& pd)
{
    Deconvolution::load_param(pd);

    group = pd.get(7, 1);

    return 0;
}

int DeconvolutionDepthWise::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    if (group == 1)
    {
        return Deconvolution::forward(bottom_blob, top_blob);
    }

    // deconvolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

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
    top_blob_bordered.create(outw, outh, num_output);
    if (top_blob_bordered.empty())
        return -100;

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
        #pragma omp parallel for
        for (int g=0; g<group; g++)
        {
            const float* inptr = bottom_blob.channel(g);
            const float* kptr = weight_data + maxk * g;
            Mat m = top_blob_bordered.channel(g);

            const float bias = bias_term ? bias_data.data[g] : 0.f;

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

        #pragma omp parallel for
        for (int g = 0; g < group; g++)
        {
            const float* weight_data_ptr = weight_data + maxk * channels_g * num_output_g * g;
            for (int p = 0; p < num_output_g; p++)
            {
                Mat out = top_blob_bordered.channel(g * num_output_g + p);

                const float bias = bias_term ? bias_data.data[g * num_output_g + p] : 0.f;

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
                            float val = *(m.data + w * i + j);

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

    top_blob = top_blob_bordered;

    if (pad > 0)
    {
        copy_cut_border(top_blob_bordered, top_blob, pad, pad, pad, pad);
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }

    return 0;
}

} // namespace ncnn
