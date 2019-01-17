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

#include "convolutiondepthwise.h"

#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(ConvolutionDepthWise)

ConvolutionDepthWise::ConvolutionDepthWise()
{
    one_blob_only = true;
    support_inplace = false;
}

ConvolutionDepthWise::~ConvolutionDepthWise()
{
    for (int i=0; i<(int)quantize_ops.size(); i++)
        delete quantize_ops[i];

    quantize_ops.clear();

    for (int i=0; i<(int)dequantize_ops.size(); i++)
        delete dequantize_ops[i];

    dequantize_ops.clear();
}

int ConvolutionDepthWise::load_param(const ParamDict& pd)
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
    int8_scale_term = pd.get(8, 0);

    use_int8_inference = pd.use_int8_inference;

    if (num_output % group != 0)
    {
        // reject invalid group
        return -100;
    }

    if (int8_scale_term == 0)
        use_int8_inference = false;

    return 0;
}

int ConvolutionDepthWise::load_model(const ModelBin& mb)
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

    if (int8_scale_term == 1)
    {
        weight_data_int8_scales = mb.load(group, 1);
        bottom_blob_int8_scales = mb.load(group, 1);
    }
    else if (int8_scale_term == 2)
    {
        weight_data_int8_scales = mb.load(1, 1);
        bottom_blob_int8_scales = mb.load(1, 1);

        // extend group if only one provided
        float weight_data_int8_scale = weight_data_int8_scales[0];
        weight_data_int8_scales = Mat(group);
        weight_data_int8_scales.fill(weight_data_int8_scale);

        float bottom_blob_int8_scale = bottom_blob_int8_scales[0];
        bottom_blob_int8_scales = Mat(group);
        bottom_blob_int8_scales.fill(bottom_blob_int8_scale);
    }

    for (int i=0; i<(int)quantize_ops.size(); i++)
        delete quantize_ops[i];

    quantize_ops.clear();

    for (int i=0; i<(int)dequantize_ops.size(); i++)
        delete dequantize_ops[i];

    dequantize_ops.clear();

    bool weight_data_is_int8 = (weight_data.elemsize == (size_t)1u);
    bool weight_data_is_float32 = (weight_data.elemsize == (size_t)4u);

    if (weight_data_is_int8 && !use_int8_inference)
    {
        fprintf(stderr, "quantized int8 weight loaded but use_int8_inference disabled\n");
        return -1;
    }

    if (weight_data_is_float32 && use_int8_inference)
    {
        // quantize weight to int8
        Mat int8_weight_data(weight_data_size, (size_t)1u);
        if (int8_weight_data.empty())
            return -100;

        const int weight_data_size_g = weight_data_size / group;

        for (int g=0; g<group; g++)
        {
            Layer* op = ncnn::create_layer(ncnn::LayerType::Quantize);

            ncnn::ParamDict pd;
            pd.set(0, weight_data_int8_scales[g]);// scale

            op->load_param(pd);

            ncnn::Option opt = ncnn::get_default_option();
            opt.blob_allocator = int8_weight_data.allocator;

            const Mat weight_data_g = weight_data.range(weight_data_size_g * g, weight_data_size_g);
            Mat int8_weight_data_g = int8_weight_data.range(weight_data_size_g * g, weight_data_size_g);
            op->forward(weight_data_g, int8_weight_data_g, opt);

            delete op;
        }

        weight_data = int8_weight_data;
    }

    if (use_int8_inference)
    {
        quantize_ops.resize(group);
        dequantize_ops.resize(group);

        for (int g=0; g<group; g++)
        {
            quantize_ops[g] = ncnn::create_layer(ncnn::LayerType::Quantize);

            ncnn::ParamDict pd;
            pd.set(0, bottom_blob_int8_scales[g]);// scale

            quantize_ops[g]->load_param(pd);
        }

        for (int g=0; g<group; g++)
        {
            dequantize_ops[g] = ncnn::create_layer(ncnn::LayerType::Dequantize);

            float top_rescale = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

            ncnn::ParamDict pd;
            pd.set(0, top_rescale);// scale
            pd.set(1, bias_term);// bias_term
            pd.set(2, 1);// bias_data_size

            dequantize_ops[g]->load_param(pd);

            ncnn::Mat weights[1];
            weights[0] = bias_data.range(g, 1);

            dequantize_ops[g]->load_model(ModelBinFromMatArray(weights));
        }
    }

    return 0;
}

int ConvolutionDepthWise::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // convolv with NxN kernel
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

//     fprintf(stderr, "ConvolutionDepthWise input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_unbordered = bottom_blob;
    if (use_int8_inference && elemsize != 1)
    {
        Mat bottom_blob_int8;
        bottom_blob_int8.create(w, h, channels, (size_t)1u, opt.workspace_allocator);
        if (bottom_blob_int8.empty())
            return -100;

        const int channels_g = channels / group;

        // quantize, scale and round to nearest
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g=0; g<group; g++)
        {
            ncnn::Option opt_g = opt;
            opt_g.num_threads = 1;
            opt_g.blob_allocator = bottom_blob_int8.allocator;

            const Mat bottom_blob_g = bottom_blob.channel_range(channels_g * g, channels_g);
            Mat bottom_blob_int8_g = bottom_blob_int8.channel_range(channels_g * g, channels_g);
            quantize_ops[g]->forward(bottom_blob_g, bottom_blob_int8_g, opt_g);
        }

        bottom_blob_unbordered = bottom_blob_int8;
    }

    Mat bottom_blob_bordered = bottom_blob_unbordered;
    if (pad_w > 0 || pad_h > 0)
    {
        copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, pad_h, pad_h, pad_w, pad_w, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_w == -233 && pad_h == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
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

    if (use_int8_inference)
    {
        // depth-wise
        if (channels == group && group == num_output)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g=0; g<group; g++)
            {
                int* outptr = top_blob.channel(g);
                const signed char* kptr = (const signed char*)weight_data + maxk * g;
                const Mat m = bottom_blob_bordered.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        int sum = 0;

                        const signed char* sptr = m.row<signed char>(i*stride_h) + j*stride_w;

                        for (int k = 0; k < maxk; k++)
                        {
                            signed char val = sptr[ space_ofs[k] ];
                            signed char w = kptr[k];
                            sum += val * w;
                        }

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }

                // dequantize, reverse scale inplace
                {
                    ncnn::Option opt_g = opt;
                    opt_g.num_threads = 1;
                    opt_g.blob_allocator = top_blob.allocator;

                    Mat top_blob_g = top_blob.channel_range(g, 1);
                    dequantize_ops[g]->forward_inplace(top_blob_g, opt_g);
                }
            }
        }
        else
        {
            const int channels_g = channels / group;
            const int num_output_g = num_output / group;

#ifdef _WIN32
            #pragma omp parallel for num_threads(opt.num_threads)
#else // _WIN32
            #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
#endif // _WIN32
            for (int g=0; g<group; g++)
            {
                for (int p=0; p<num_output_g; p++)
                {
                    int* outptr = top_blob.channel(g * num_output_g + p);
                    const signed char* weight_data_ptr = (const signed char*)weight_data + maxk * channels_g * num_output_g * g;

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            int sum = 0;

                            const signed char* kptr = weight_data_ptr + maxk * channels_g * p;

                            // channels_g
                            for (int q=0; q<channels_g; q++)
                            {
                                const Mat m = bottom_blob_bordered.channel(channels_g * g + q);
                                const signed char* sptr = m.row<signed char>(i*stride_h) + j*stride_w;

                                for (int k = 0; k < maxk; k++)
                                {
                                    signed char val = sptr[ space_ofs[k] ];
                                    signed char w = kptr[k];
                                    sum += val * w;
                                }

                                kptr += maxk;
                            }

                            outptr[j] = sum;
                        }

                        outptr += outw;
                    }
                }
            }

            // dequantize, reverse scale inplace
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g=0; g<group; g++)
            {
                ncnn::Option opt_g = opt;
                opt_g.num_threads = 1;
                opt_g.blob_allocator = top_blob.allocator;

                Mat top_blob_g = top_blob.channel_range(num_output_g * g, num_output_g);
                dequantize_ops[g]->forward_inplace(top_blob_g, opt_g);
            }
        }

        return 0;
    }

    // depth-wise
    if (channels == group && group == num_output)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g=0; g<group; g++)
        {
            float* outptr = top_blob.channel(g);
            const float* kptr = (const float*)weight_data + maxk * g;
            const Mat m = bottom_blob_bordered.channel(g);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                        sum = bias_data[g];

                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        float w = kptr[k];
                        sum += val * w;
                    }

                    outptr[j] = sum;
                }

                outptr += outw;
            }
        }

        return 0;
    }

    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

#ifdef _WIN32
    #pragma omp parallel for num_threads(opt.num_threads)
#else // _WIN32
    #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
#endif // _WIN32
    for (int g=0; g<group; g++)
    {
        for (int p=0; p<num_output_g; p++)
        {
            float* outptr = top_blob.channel(g * num_output_g + p);
            const float* weight_data_ptr = (const float*)weight_data + maxk * channels_g * num_output_g * g;

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                        sum = bias_data[num_output_g * g + p];

                    const float* kptr = weight_data_ptr + maxk * channels_g * p;

                    // channels_g
                    for (int q=0; q<channels_g; q++)
                    {
                        const Mat m = bottom_blob_bordered.channel(channels_g * g + q);
                        const float* sptr = m.row(i*stride_h) + j*stride_w;

                        for (int k = 0; k < maxk; k++)
                        {
                            float val = sptr[ space_ofs[k] ];
                            float w = kptr[k];
                            sum += val * w;
                        }

                        kptr += maxk;
                    }

                    outptr[j] = sum;
                }

                outptr += outw;
            }
        }
    }

    return 0;
}

} // namespace ncnn
