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

#include "deconvolutiondepthwise3d.h"

#include "fused_activation.h"

namespace ncnn {

DeconvolutionDepthWise3D::DeconvolutionDepthWise3D()
{
    one_blob_only = true;
    support_inplace = false;
}

int DeconvolutionDepthWise3D::load_param(const ParamDict& pd)
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
    output_pad_right = pd.get(18, 0);
    output_pad_bottom = pd.get(19, output_pad_right);
    output_pad_behind = pd.get(20, output_pad_right);
    output_w = pd.get(25, 0);
    output_h = pd.get(26, output_w);
    output_d = pd.get(27, output_w);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    group = pd.get(7, 1);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    return 0;
}

int DeconvolutionDepthWise3D::load_model(const ModelBin& mb)
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

static int deconvolutiondepthwise3d(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int kernel_h, int kernel_d, int stride_w, int stride_h, int stride_d, int dilation_w, int dilation_h, int dilation_d, int group, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

    const int maxk = kernel_w * kernel_h * kernel_d;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap0 = outw * dilation_h - kernel_w * dilation_w;
        int gap1 = outh * outw * dilation_d - outw * kernel_h * dilation_h;
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

            // shadowed variable for less openmp task args
            const int w = bottom_blob.w;
            const int h = bottom_blob.h;
            const int d = bottom_blob.d;
            const int outw = top_blob.w;
            const int outh = top_blob.h;
            const int outd = top_blob.d;

            for (int z = 0; z < d; z++)
            {
                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        float* outptr = out.depth(z * stride_d).row(i * stride_h) + j * stride_w;

                        const float val = inptr[z * w * h + i * w + j];

                        for (int k = 0; k < maxk; k++)
                        {
                            float w = kptr[k];
                            outptr[space_ofs[k]] += val * w;
                        }
                    }
                }
            }

            {
                float* outptr = out;
                int size = outw * outh * outd;

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

                // shadowed variable for less openmp task args
                const int w = bottom_blob.w;
                const int h = bottom_blob.h;
                const int d = bottom_blob.d;
                const int outw = top_blob.w;
                const int outh = top_blob.h;
                const int outd = top_blob.d;

                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        for (int j = 0; j < w; j++)
                        {
                            float* outptr = out.depth(z * stride_d).row(i * stride_h) + j * stride_w;

                            const float* kptr = weight_data_ptr + maxk * inch_g * p;

                            for (int q = 0; q < inch_g; q++)
                            {
                                const float val = bottom_blob.channel(inch_g * g + q).depth(z).row(i)[j];

                                for (int k = 0; k < maxk; k++)
                                {
                                    outptr[space_ofs[k]] += val * kptr[k];
                                }

                                kptr += maxk;
                            }
                        }
                    }
                }

                {
                    float* outptr = out;
                    int size = outw * outh * outd;

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

int DeconvolutionDepthWise3D::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const int kernel_extent_d = dilation_d * (kernel_d - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w + output_pad_right;
    int outh = (h - 1) * stride_h + kernel_extent_h + output_pad_bottom;
    int outd = (d - 1) * stride_d + kernel_extent_d + output_pad_behind;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || pad_front > 0 || pad_behind > 0 || (output_w > 0 && output_h > 0 && output_d > 0))
    {
        top_blob_bordered.create(outw, outh, outd, num_output, elemsize, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, outd, num_output, elemsize, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    int ret = deconvolutiondepthwise3d(bottom_blob, top_blob_bordered, weight_data, bias_data, kernel_w, kernel_h, kernel_d, stride_w, stride_h, stride_d, dilation_w, dilation_h, dilation_d, group, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

void DeconvolutionDepthWise3D::cut_padding(const Mat& top_blob_bordered, Mat& top_blob, const Option& opt) const
{
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || pad_front > 0 || pad_behind > 0)
    {
        copy_cut_border_3d(top_blob_bordered, top_blob, pad_top, pad_bottom, pad_left, pad_right, pad_front, pad_behind, opt);
    }
    else if (output_w > 0 && output_h > 0 && output_d > 0)
    {
        int wcut = top_blob_bordered.w - output_w;
        int hcut = top_blob_bordered.h - output_h;
        int dcut = top_blob_bordered.d - output_d;

        if (pad_left == -233 || pad_right == -233 || pad_top == -233 || pad_bottom == -233 || pad_front == -233 || pad_behind == -233)
        {
            // onnx padding=SAME_UPPER
            copy_cut_border_3d(top_blob_bordered, top_blob, hcut / 2, hcut - hcut / 2, wcut / 2, wcut - wcut / 2, dcut / 2, dcut - dcut / 2, opt);
        }
        else if (pad_left == -234 || pad_right == -234 || pad_top == -234 || pad_bottom == -234 || pad_front == -234 || pad_behind == -234)
        {
            // onnx padding=SAME_LOWER
            copy_cut_border_3d(top_blob_bordered, top_blob, hcut - hcut / 2, hcut / 2, wcut - wcut / 2, wcut / 2, dcut - dcut / 2, dcut / 2, opt);
        }
    }
    else
    {
        top_blob = top_blob_bordered;
    }
}

} // namespace ncnn
