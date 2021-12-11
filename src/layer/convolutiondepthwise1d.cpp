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

#include "convolutiondepthwise1d.h"

#include "layer_type.h"

#include "fused_activation.h"

namespace ncnn {

ConvolutionDepthWise1D::ConvolutionDepthWise1D()
{
    one_blob_only = true;
    support_inplace = false;
}

int ConvolutionDepthWise1D::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    dilation_w = pd.get(2, 1);
    stride_w = pd.get(3, 1);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    group = pd.get(7, 1);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    dynamic_weight = pd.get(19, 0);

    if (dynamic_weight)
    {
        one_blob_only = false;
    }

    if (num_output % group != 0)
    {
        // reject invalid group
        return -100;
    }

    return 0;
}

int ConvolutionDepthWise1D::load_model(const ModelBin& mb)
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

int ConvolutionDepthWise1D::create_pipeline(const Option& opt)
{
    return 0;
}

static int convolutiondepthwise1d(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int stride_w, int dilation_w, int group, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int h = bottom_blob.h;

    const int outw = top_blob.w;
    const int outh = top_blob.h;

    const int bias_term = bias_data.empty() ? 0 : 1;

    // depth-wise
    if (h == group && group == outh)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            float* outptr = top_blob.row(g);
            const float* kptr = (const float*)weight_data + kernel_w * g;

            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_term)
                    sum = bias_data[g];

                const float* sptr = bottom_blob.row(g) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float val = *sptr;
                    float w = kptr[k];
                    sum += val * w;

                    sptr += dilation_w;
                }

                outptr[j] = activation_ss(sum, activation_type, activation_params);
            }
        }
    }
    else
    {
        // group convolution
        const int h_g = h / group;
        const int outh_g = outh / group;

#ifdef _WIN32
        #pragma omp parallel for num_threads(opt.num_threads)
#else
        #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
#endif
        for (int g = 0; g < group; g++)
        {
            for (int p = 0; p < outh_g; p++)
            {
                float* outptr = top_blob.row(g * outh_g + p);
                const float* weight_data_ptr = (const float*)weight_data + kernel_w * h_g * outh_g * g;

                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                        sum = bias_data[outh_g * g + p];

                    const float* kptr = weight_data_ptr + kernel_w * h_g * p;

                    for (int q = 0; q < h_g; q++)
                    {
                        const float* sptr = bottom_blob.row(h_g * g + q) + j * stride_w;

                        for (int k = 0; k < kernel_w; k++)
                        {
                            float val = *sptr;
                            float w = kptr[k];
                            sum += val * w;

                            sptr += dilation_w;
                        }

                        kptr += kernel_w;
                    }

                    outptr[j] = activation_ss(sum, activation_type, activation_params);
                }
            }
        }
    }

    return 0;
}

int ConvolutionDepthWise1D::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const int w = bottom_blob_bordered.w;
    const size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    const int outw = (w - kernel_extent_w) / stride_w + 1;

    top_blob.create(outw, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    int ret = convolutiondepthwise1d(bottom_blob_bordered, top_blob, weight_data, bias_data, kernel_w, stride_w, dilation_w, group, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    return 0;
}

int ConvolutionDepthWise1D::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _kernel_w = _weight_data.w;
    const int _num_output = _weight_data.c;

    Mat weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

    Mat bias_data_flattened;
    if (bias_term)
    {
        const Mat& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, _kernel_w, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const int w = bottom_blob_bordered.w;
    const size_t elemsize = bottom_blob_bordered.elemsize;

    const int kernel_extent_w = dilation_w * (_kernel_w - 1) + 1;

    const int outw = (w - kernel_extent_w) / stride_w + 1;

    top_blob.create(outw, _num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    int ret = convolutiondepthwise1d(bottom_blob_bordered, top_blob, weight_data_flattened, bias_data_flattened, _kernel_w, stride_w, dilation_w, group, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    return 0;
}

void ConvolutionDepthWise1D::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const
{
    make_padding(bottom_blob, bottom_blob_bordered, kernel_w, opt);
}

void ConvolutionDepthWise1D::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, int _kernel_w, const Option& opt) const
{
    int w = bottom_blob.w;

    const int kernel_extent_w = dilation_w * (_kernel_w - 1) + 1;

    bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0)
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, 0, 0, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_left == -233 && pad_right == -233)
    {
        // tensorflow padding=SAME or onnx padding=SAME_UPPER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        if (wpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, 0, 0, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_left == -234 && pad_right == -234)
    {
        // onnx padding=SAME_LOWER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        if (wpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, 0, 0, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}

} // namespace ncnn
