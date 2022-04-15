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

#include "fused_activation.h"

namespace ncnn {

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
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_top = pd.get(14, pad_left);
    pad_bottom = pd.get(16, pad_top);
    output_pad_right = pd.get(18, 0);
    output_pad_bottom = pd.get(19, output_pad_right);
    output_w = pd.get(20, 0);
    output_h = pd.get(21, output_w);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    group = pd.get(7, 1);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

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

static int deconvolutiondepthwise(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int group, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

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
    if (inch == group && group == outch)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            const float* inptr = bottom_blob.channel(g);
            const float* kptr = (const float*)weight_data + maxk * g;
            Mat out = top_blob.channel(g);

            const float bias = bias_term ? bias_data[g] : 0.f;

            out.fill(bias);

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    float* outptr = out.row(i * stride_h) + j * stride_w;

                    const float val = inptr[i * w + j];

                    for (int k = 0; k < maxk; k++)
                    {
                        float w = kptr[k];
                        outptr[space_ofs[k]] += val * w;
                    }
                }
            }

            {
                float* outptr = out;
                int size = outw * outh;

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = activation_ss(outptr[i], activation_type, activation_params);
                }
            }
        }
    }
    else
    {
        const int inch_g = inch / group;
        const int outch_g = outch / group;

#ifdef _WIN32
        #pragma omp parallel for num_threads(opt.num_threads)
#else
        #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
#endif
        for (int g = 0; g < group; g++)
        {
            for (int p = 0; p < outch_g; p++)
            {
                Mat out = top_blob.channel(g * outch_g + p);

                const float* weight_data_ptr = (const float*)weight_data + maxk * inch_g * outch_g * g;
                const float bias = bias_term ? bias_data[g * outch_g + p] : 0.f;

                out.fill(bias);

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        float* outptr = out.row(i * stride_h) + j * stride_w;

                        const float* kptr = weight_data_ptr + maxk * inch_g * p;

                        for (int q = 0; q < inch_g; q++)
                        {
                            const float val = bottom_blob.channel(inch_g * g + q).row(i)[j];

                            for (int k = 0; k < maxk; k++)
                            {
                                outptr[space_ofs[k]] += val * kptr[k];
                            }

                            kptr += maxk;
                        }
                    }
                }

                {
                    float* outptr = out;
                    int size = outw * outh;

                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = activation_ss(outptr[i], activation_type, activation_params);
                    }
                }
            }
        }
    }

    return 0;
}

int DeconvolutionDepthWise::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w + output_pad_right;
    int outh = (h - 1) * stride_h + kernel_extent_h + output_pad_bottom;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, num_output, elemsize, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    int ret = deconvolutiondepthwise(bottom_blob, top_blob_bordered, weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, group, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

void DeconvolutionDepthWise::cut_padding(const Mat& top_blob_bordered, Mat& top_blob, const Option& opt) const
{
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        copy_cut_border(top_blob_bordered, top_blob, pad_top, pad_bottom, pad_left, pad_right, opt);
    }
    else if (output_w > 0 && output_h > 0)
    {
        int wcut = top_blob_bordered.w - output_w;
        int hcut = top_blob_bordered.h - output_h;

        if (pad_left == -233 || pad_right == -233 || pad_top == -233 || pad_bottom == -233)
        {
            // onnx padding=SAME_UPPER
            copy_cut_border(top_blob_bordered, top_blob, hcut / 2, hcut - hcut / 2, wcut / 2, wcut - wcut / 2, opt);
        }
        else if (pad_left == -234 || pad_right == -234 || pad_top == -234 || pad_bottom == -234)
        {
            // onnx padding=SAME_LOWER
            copy_cut_border(top_blob_bordered, top_blob, hcut - hcut / 2, hcut / 2, wcut - wcut / 2, wcut / 2, opt);
        }
    }
    else
    {
        top_blob = top_blob_bordered;
    }
}

} // namespace ncnn
