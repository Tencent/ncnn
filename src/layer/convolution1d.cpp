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

#include "convolution1d.h"

#include "layer_type.h"

namespace ncnn {

Convolution1D::Convolution1D()
{
    one_blob_only = true;
    support_inplace = false;
}

int Convolution1D::load_param(const ParamDict& pd)
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
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    return 0;
}

int Convolution1D::load_model(const ModelBin& mb)
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

int Convolution1D::create_pipeline(const Option& opt)
{
    return 0;
}

int Convolution1D::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    //     NCNN_LOGE("Convolution1D input %d x %d  pad = %d  ksize=%d  stride=%d", w, h, pad_w, kernel_w, stride_w);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;

    // float32
    top_blob.create(outw, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < num_output; p++)
    {
        float* outptr = top_blob.row(p);

        for (int j = 0; j < outw; j++)
        {
            float sum = 0.f;

            if (bias_term)
                sum = bias_data[p];

            const float* kptr = (const float*)weight_data + kernel_w * h * p;

            for (int q = 0; q < h; q++)
            {
                const float* sptr = bottom_blob_bordered.row(q) + j * stride_w;

                for (int k = 0; k < kernel_w; k++)
                {
                    float val = *sptr;
                    float wt = kptr[k];
                    sum += val * wt;

                    sptr += dilation_w;
                }

                kptr += kernel_w;
            }

            if (activation_type == 1)
            {
                sum = std::max(sum, 0.f);
            }
            else if (activation_type == 2)
            {
                float slope = activation_params[0];
                sum = sum > 0.f ? sum : sum * slope;
            }
            else if (activation_type == 3)
            {
                float min = activation_params[0];
                float max = activation_params[1];
                if (sum < min)
                    sum = min;
                if (sum > max)
                    sum = max;
            }
            else if (activation_type == 4)
            {
                sum = static_cast<float>(1.f / (1.f + exp(-sum)));
            }
            else if (activation_type == 5)
            {
                const float MISH_THRESHOLD = 20;
                float x = sum, y;
                if (x > MISH_THRESHOLD)
                    y = x;
                else if (x < -MISH_THRESHOLD)
                    y = expf(x);
                else
                    y = logf(expf(x) + 1);
                sum = static_cast<float>(x * tanh(y));
            }

            outptr[j] = sum;
        }
    }

    return 0;
}

void Convolution1D::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const
{
    int w = bottom_blob.w;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

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
