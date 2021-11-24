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

#include "convolution3d.h"

#include "fused_activation.h"

namespace ncnn {

Convolution3D::Convolution3D()
{
    one_blob_only = true;
    support_inplace = false;
}

int Convolution3D::load_param(const ParamDict& pd)
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
    pad_behind = pd.get(17, pad_top);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);

    return 0;
}

int Convolution3D::load_model(const ModelBin& mb)
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

int Convolution3D::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
    make_padding(bottom_blob, bottom_blob_bordered, opt);
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
        int offset0 = dilation_d;
        int offset1 = d * dilation_h - kernel_d * dilation_d;
        int offset2 = (h * d) * dilation_w - h * kernel_h * dilation_h - kernel_h * dilation_h;
        for (int i = 0; i < kernel_w; ++i)
        {
            for (int j = 0; j < kernel_h; ++j)
            {
                for (int k = 0; k < kernel_d; ++k)
                {
                    space_ofs[p1] = p2;
                    p1++;
                    p2 += offset0;
                }
                p2 += offset1;
            }
            p2 += offset2;
        }
    }

    top_blob.create(outw, outh, outd, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < num_output; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int j = 0; j < outw; j++)
        {
            for (int i = 0; i < outh; i++)
            {
                for (int k = 0; k < outd; k++)
                {
                    float sum = 0.f;

                    if (bias_term)
                        sum = bias_data[p];

                    const float* kptr = (const float*)weight_data + maxk * channels * p;

                    for (int q = 0; q < channels; q++)
                    {
                        const Mat m = bottom_blob_bordered.channel(q);
                        // (w*d): offset when you go across one h
                        // (d): offset when you go across one w
                        const float* sptr = (float*)m.data + (h * d) * j * stride_w + (d)*i * stride_h + k * stride_d;

                        for (int l = 0; l < maxk; l++)
                        {
                            float val = sptr[space_ofs[l]];
                            float wt = kptr[l];
                            sum += val * wt;
                        }

                        kptr += maxk;
                    }

                    sum = activation_ss(sum, activation_type, activation_params);

                    outptr[k] = sum;
                }

                // move forward output pointer
                outptr += outd;
            }
        }
    }

    return 0;
}

void Convolution3D::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const
{
    bottom_blob_bordered = bottom_blob;
    return;
}

} // namespace ncnn
