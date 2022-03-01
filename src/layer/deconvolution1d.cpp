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

#include "deconvolution1d.h"

#include "fused_activation.h"

namespace ncnn {

Deconvolution1D::Deconvolution1D()
{
    one_blob_only = true;
    support_inplace = false;
}

int Deconvolution1D::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    dilation_w = pd.get(2, 1);
    stride_w = pd.get(3, 1);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    output_pad_right = pd.get(18, 0);
    output_w = pd.get(20, 0);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    return 0;
}

int Deconvolution1D::load_model(const ModelBin& mb)
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

static int deconvolution1d(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int stride_w, int dilation_w, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;

    const int outw = top_blob.w;
    const int outh = top_blob.h;

    const int bias_term = bias_data.empty() ? 0 : 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outh; p++)
    {
        Mat out = top_blob.row_range(p, 1);

        const float bias = bias_term ? bias_data[p] : 0.f;

        out.fill(bias);

        for (int j = 0; j < w; j++)
        {
            float* outptr = (float*)out + j * stride_w;

            const float* kptr = (const float*)weight_data + kernel_w * h * p;

            for (int q = 0; q < h; q++)
            {
                const float val = bottom_blob.row(q)[j];

                for (int k = 0; k < kernel_w; k++)
                {
                    float w = kptr[k];
                    outptr[k * dilation_w] += val * w;
                }

                kptr += kernel_w;
            }
        }

        {
            float* outptr = out;

            for (int i = 0; i < outw; i++)
            {
                outptr[i] = activation_ss(outptr[i], activation_type, activation_params);
            }
        }
    }

    return 0;
}

int Deconvolution1D::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w + output_pad_right;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || output_w > 0)
    {
        top_blob_bordered.create(outw, num_output, elemsize, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, num_output, elemsize, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    int ret = deconvolution1d(bottom_blob, top_blob_bordered, weight_data, bias_data, kernel_w, stride_w, dilation_w, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

void Deconvolution1D::cut_padding(const Mat& top_blob_bordered, Mat& top_blob, const Option& opt) const
{
    if (pad_left > 0 || pad_right > 0)
    {
        copy_cut_border(top_blob_bordered, top_blob, 0, 0, pad_left, pad_right, opt);
    }
    else if (output_w > 0)
    {
        int wcut = top_blob_bordered.w - output_w;

        if (pad_left == -233 || pad_right == -233)
        {
            // onnx padding=SAME_UPPER
            copy_cut_border(top_blob_bordered, top_blob, 0, 0, wcut / 2, wcut - wcut / 2, opt);
        }
        else if (pad_left == -234 || pad_right == -234)
        {
            // onnx padding=SAME_LOWER
            copy_cut_border(top_blob_bordered, top_blob, 0, 0, wcut - wcut / 2, wcut / 2, opt);
        }
    }
    else
    {
        top_blob = top_blob_bordered;
    }
}

} // namespace ncnn
