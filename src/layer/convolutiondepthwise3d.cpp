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

#include "convolutiondepthwise3d.h"

#include "fused_activation.h"

namespace ncnn {

ConvolutionDepthWise3D::ConvolutionDepthWise3D()
{
    one_blob_only = true;
    support_inplace = false;
}

int ConvolutionDepthWise3D::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    kernel_d = pd.get(21, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    dilation_d = pd.get(22, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    stride_d = pd.get(23, stride_w);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_top = pd.get(14, pad_left);
    pad_bottom = pd.get(16, pad_top);
    pad_front = pd.get(24, pad_left);
    pad_behind = pd.get(17, pad_front);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    group = pd.get(7, 1);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    return 0;
}

int ConvolutionDepthWise3D::load_model(const ModelBin& mb)
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

int ConvolutionDepthWise3D::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_extend_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extend_h = dilation_h * (kernel_h - 1) + 1;
    const int kernel_extend_d = dilation_d * (kernel_d - 1) + 1;

    Mat bottom_blob_bordered;
    Option opt_pad = opt;
    opt_pad.use_packing_layout = false;
    make_padding(bottom_blob, bottom_blob_bordered, opt_pad);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;
    d = bottom_blob_bordered.d;

    int outw = (w - kernel_extend_w) / stride_w + 1;
    int outh = (h - kernel_extend_h) / stride_h + 1;
    int outd = (d - kernel_extend_d) / stride_d + 1;

    const int maxk = kernel_w * kernel_h * kernel_d;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap0 = w * dilation_h - kernel_w * dilation_w;
        int gap1 = h * w * dilation_d - w * kernel_h * dilation_h;
        for (int z = 0; z < kernel_d; z++)
        {
            for (int i = 0; i < kernel_h; i++)
            {
                for (int j = 0; j < kernel_w; j++)
                {
                    space_ofs[p1] = p2;
                    p1++;
                    p2 += dilation_w;
                }
                p2 += gap0;
            }
            p2 += gap1;
        }
    }

    top_blob.create(outw, outh, outd, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // depth-wise
    if (channels == group && group == num_output)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            float* outptr = top_blob.channel(g);
            const float* kptr = (const float*)weight_data + maxk * g;
            const Mat m = bottom_blob_bordered.channel(g);

            for (int z = 0; z < outd; z++)
            {
                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                            sum = bias_data[g];

                        const float* sptr = m.depth(z * stride_d).row(i * stride_h) + j * stride_w;

                        for (int k = 0; k < maxk; k++)
                        {
                            float val = sptr[space_ofs[k]];
                            float w = kptr[k];
                            sum += val * w;
                        }

                        outptr[j] = activation_ss(sum, activation_type, activation_params);
                    }

                    outptr += outw;
                }
            }
        }
    }
    else
    {
        // group convolution
        const int channels_g = channels / group;
        const int num_output_g = num_output / group;

#ifdef _WIN32
        #pragma omp parallel for num_threads(opt.num_threads)
#else
        #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
#endif
        for (int g = 0; g < group; g++)
        {
            for (int p = 0; p < num_output_g; p++)
            {
                float* outptr = top_blob.channel(g * num_output_g + p);
                const float* weight_data_ptr = (const float*)weight_data + maxk * channels_g * num_output_g * g;

                // shadowed variable for less openmp task args
                const int outw = top_blob.w;
                const int outh = top_blob.h;
                const int outd = top_blob.d;

                for (int z = 0; z < outd; z++)
                {
                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float sum = 0.f;

                            if (bias_term)
                                sum = bias_data[num_output_g * g + p];

                            const float* kptr = weight_data_ptr + maxk * channels_g * p;

                            for (int q = 0; q < channels_g; q++)
                            {
                                const Mat m = bottom_blob_bordered.channel(channels_g * g + q);
                                const float* sptr = m.depth(z * stride_d).row(i * stride_h) + j * stride_w;

                                for (int l = 0; l < maxk; l++)
                                {
                                    float val = sptr[space_ofs[l]];

                                    float wt = kptr[l];
                                    sum += val * wt;
                                }

                                kptr += maxk;
                            }

                            outptr[j] = activation_ss(sum, activation_type, activation_params);
                        }

                        outptr += outw;
                    }
                }
            }
        }
    }

    return 0;
}

void ConvolutionDepthWise3D::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const int kernel_extent_d = dilation_d * (kernel_d - 1) + 1;

    bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || pad_front > 0 || pad_behind > 0)
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border_3d(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, pad_front, pad_behind, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233 && pad_front == -233 && pad_behind == -233)
    {
        // tensorflow padding=SAME or onnx padding=SAME_UPPER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        int dpad = kernel_extent_d + (d - 1) / stride_d * stride_d - d;
        if (wpad > 0 || hpad > 0 || dpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border_3d(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, dpad / 2, dpad - dpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234 && pad_front == -234 && pad_behind == -234)
    {
        // onnx padding=SAME_LOWER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        int dpad = kernel_extent_d + (d - 1) / stride_d * stride_d - d;
        if (wpad > 0 || hpad > 0 || dpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border_3d(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, dpad / 2, dpad - dpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}

} // namespace ncnn
