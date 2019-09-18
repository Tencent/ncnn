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

#include "convolution.h"
#include <algorithm>
#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Convolution)

Convolution::Convolution()
{
    one_blob_only = true;
    support_inplace = false;
    use_int8_requantize = false;

    quantize = 0;
}

int Convolution::load_param(const ParamDict& pd)
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
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    int8_scale_term = pd.get(8, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());
    impl_type = pd.get(17, 0);

    return 0;
}

int Convolution::load_model(const ModelBin& mb)
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

    if (int8_scale_term)
    {
        weight_data_int8_scales = mb.load(num_output, 1);
        bottom_blob_int8_scale = mb.load(1, 1)[0];
    }

    return 0;
}

int Convolution::create_pipeline(const Option& opt)
{
    Option opt_cpu = opt;
    opt_cpu.use_vulkan_compute = false;

    use_int8_inference = opt.use_int8_inference;

    if (int8_scale_term == 0)
        use_int8_inference = false;

    bool weight_data_is_int8 = (weight_data.elemsize == (size_t)1u);
    bool weight_data_is_float32 = (weight_data.elemsize == (size_t)4u);

    if (weight_data_is_int8 && !use_int8_inference)
    {
        fprintf(stderr, "quantized int8 weight loaded but use_int8_inference disabled\n");
        return -1;
    }

    // runtime quantize the weight data
    if (weight_data_is_float32 && use_int8_inference)
    {
        // quantize weight to int8
        Mat int8_weight_data(weight_data_size, (size_t)1u);
        if (int8_weight_data.empty())
            return -100;

        const int weight_data_size_output = weight_data_size / num_output;

        for (int n=0; n<num_output; n++)
        {
            Layer* op = ncnn::create_layer(ncnn::LayerType::Quantize);

            ncnn::ParamDict pd;
            pd.set(0, weight_data_int8_scales[n]);// scale

            op->load_param(pd);

            op->create_pipeline(opt_cpu);

            ncnn::Option opt;
            opt.blob_allocator = int8_weight_data.allocator;

            const Mat weight_data_n = weight_data.range(weight_data_size_output * n, weight_data_size_output);
            Mat int8_weight_data_n = int8_weight_data.range(weight_data_size_output * n, weight_data_size_output);
            op->forward(weight_data_n, int8_weight_data_n, opt);

            delete op;
        }

        weight_data = int8_weight_data;
    }

    // initial the quantize,dequantize op layer
    if (use_int8_inference)
    {
        quantize = ncnn::create_layer(ncnn::LayerType::Quantize);
        {
            ncnn::ParamDict pd;
            pd.set(0, bottom_blob_int8_scale);// scale

            quantize->load_param(pd);

            quantize->create_pipeline(opt_cpu);
        }

        dequantize_ops.resize(num_output);
        for (int n=0; n<num_output; n++)
        {
            dequantize_ops[n] = ncnn::create_layer(ncnn::LayerType::Dequantize);

            float top_rescale = 1.f;

            if (weight_data_int8_scales[n] == 0)
                top_rescale = 0;
            else
                top_rescale = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[n]);

            ncnn::ParamDict pd;
            pd.set(0, top_rescale);// scale
            pd.set(1, bias_term);  // bias_term
            pd.set(2, 1);          // bias_data_size

            dequantize_ops[n]->load_param(pd);

            dequantize_ops[n]->create_pipeline(opt_cpu);

            ncnn::Mat weights[1];
            weights[0] = bias_data.range(n, 1);

            dequantize_ops[n]->load_model(ModelBinFromMatArray(weights));

            dequantize_scales.push_back(top_rescale);
        }
    }

    return 0;
}

int Convolution::destroy_pipeline(const Option& opt)
{
    Option opt_cpu = opt;
    opt_cpu.use_vulkan_compute = false;

    if (quantize)
    {
        quantize->destroy_pipeline(opt_cpu);
        delete quantize;
        quantize = 0;
    }

    for (int i=0; i<(int)dequantize_ops.size(); i++)
    {
        dequantize_ops[i]->destroy_pipeline(opt_cpu);
        delete dequantize_ops[i];
    }
    dequantize_ops.clear();

    for (int i=0; i<(int)requantize_ops.size(); i++)
    {
        requantize_ops[i]->destroy_pipeline(opt_cpu);
        delete requantize_ops[i];
    }
    requantize_ops.clear();

    dequantize_scales.clear();
    requantize_scales.clear();

    return 0;
}

int Convolution::create_requantize_op(void)
{
    if (!use_int8_requantize)
    {
        fprintf(stderr, "requantized op set but use_int8_requantize disabled\n");
        return -1;
    }

    requantize_ops.resize(num_output);
    for (int n=0; n<num_output; n++)
    {
        requantize_ops[n] = ncnn::create_layer(ncnn::LayerType::Requantize);

        float scale_in = 1.f;
        float scale_out = 1.f;

        if (weight_data_int8_scales[n] == 0)
        {
            scale_in = 0;
        }
        else
        {
            scale_in = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[n]);
        }

        scale_out = top_blob_int8_scale;

        ncnn::ParamDict pd;
        pd.set(0, scale_in);   // scale in
        pd.set(1, scale_out);  // scale_out
        pd.set(2, bias_term);  // bias_term
        pd.set(3, 1);          // bias_data_size

        requantize_ops[n]->load_param(pd);

        ncnn::Mat weights[1];
        weights[0] = bias_data.range(n, 1);

        requantize_ops[n]->load_model(ModelBinFromMatArray(weights));

        requantize_scales.push_back(scale_in);
        requantize_scales.push_back(scale_out);
    }

    return 0;
}

int Convolution::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // convolv with NxN kernel
    // value = value + bias

    // flattened blob, implement as InnerProduct
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        int num_input = weight_data_size / num_output;
        if (bottom_blob.w == num_input)
        {
            // call InnerProduct
            ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::InnerProduct);

            // set param
            ncnn::ParamDict pd;
            pd.set(0, num_output);
            pd.set(1, bias_term);
            pd.set(2, weight_data_size);
            pd.set(8, int8_scale_term);

            op->load_param(pd);

            // set weights
            ncnn::Mat weights[4];
            weights[0] = weight_data;
            weights[1] = bias_data;

            if (int8_scale_term)
            {
                weights[2] = weight_data_int8_scales;
                weights[3] = Mat(1, (size_t)4u, (void*)&bottom_blob_int8_scale);
            }

            op->load_model(ModelBinFromMatArray(weights));

            Option opt_cpu = opt;
            opt_cpu.use_vulkan_compute = false;
            op->create_pipeline(opt_cpu);

            // forward
            op->forward(bottom_blob, top_blob, opt);

            delete op;

            return 0;
        }
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

//     fprintf(stderr, "Convolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_unbordered = bottom_blob;
    if (use_int8_inference && elemsize != 1)
    {
        Mat bottom_blob_int8;
        bottom_blob_int8.create(w, h, channels, (size_t)1u, opt.workspace_allocator);
        if (bottom_blob_int8.empty())
            return -100;

        // quantize, scale and round to nearest
        {
            ncnn::Option opt_g = opt;
            opt_g.blob_allocator = bottom_blob_int8.allocator;

            quantize->forward(bottom_blob, bottom_blob_int8, opt_g);
        }

        bottom_blob_unbordered = bottom_blob_int8;
    }

    Mat bottom_blob_bordered = bottom_blob_unbordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        // tensorflow padding=SAME or onnx padding=SAME_UPPER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        // onnx padding=SAME_LOWER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

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

    // int8
    if (use_int8_inference)
    {
        if (use_int8_requantize == true)
        {
            Mat top_blob_tm;
            top_blob_tm.create(outw, outh, num_output, (size_t)4u, opt.workspace_allocator);
            if (top_blob_tm.empty())
                return -100;
            
            top_blob.create(outw, outh, num_output, (size_t)1u, opt.blob_allocator);
            if (top_blob.empty())
                return -100; 

            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p=0; p<num_output; p++)
            {
                int* outptr = top_blob_tm.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        int sum = 0;

                        const signed char* kptr = (const signed char*)weight_data + maxk * channels * p;

                        // channels
                        for (int q=0; q<channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const signed char* sptr = m.row<signed char>(i*stride_h) + j*stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                int val = sptr[ space_ofs[k] ];
                                int w = kptr[k];
                                sum += val * w;
                            }

                            kptr += maxk;
                        }

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }

                // requantize, reverse scale inplace
                {
                    ncnn::Option opt_g = opt;
                    opt_g.num_threads = 1;
                    opt_g.blob_allocator = top_blob.allocator;

                    Mat top_blob_tm_g = top_blob_tm.channel_range(p, 1);
                    Mat top_blob_g = top_blob.channel_range(p, 1);
                    requantize_ops[p]->forward(top_blob_tm_g, top_blob_g, opt_g);
                }       

                // activation relu
                if (activation_type == 1)
                {
                    signed char* outptr_s8 = top_blob.channel(p);

                    for (int i = 0; i < outh*outw; i++)
                    {
                        if (outptr_s8[i] < 0)
                            outptr_s8[i] = 0;
                    }
                }                                 
            }
        }
        else
        {
            top_blob.create(outw, outh, num_output, (size_t)4u, opt.blob_allocator);
            if (top_blob.empty())
                return -100;
      
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p=0; p<num_output; p++)
            {
                int* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        int sum = 0;

                        const signed char* kptr = (const signed char*)weight_data + maxk * channels * p;

                        // channels
                        for (int q=0; q<channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const signed char* sptr = m.row<signed char>(i*stride_h) + j*stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                int val = sptr[ space_ofs[k] ];
                                int w = kptr[k];
                                sum += val * w;
                            }

                            kptr += maxk;
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

                    Mat top_blob_g = top_blob.channel_range(p, 1);
                    dequantize_ops[p]->forward_inplace(top_blob_g, opt_g);
                }

                // activation relu
                if (activation_type == 1)
                {
                    float* outptr_fp32 = top_blob.channel(p);

                    for (int i = 0; i < outh*outw; i++)
                    {
                        outptr_fp32[i] = std::max(outptr_fp32[i], 0.f);
                    }
                }
            }   
        }        

        return 0;
    }

    // float32
    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<num_output; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_term)
                    sum = bias_data[p];

                const float* kptr = (const float*)weight_data + maxk * channels * p;

                // channels
                for (int q=0; q<channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        float val = sptr[ space_ofs[k] ]; // 20.72
                        float w = kptr[k];
                        sum += val * w; // 41.45
                    }

                    kptr += maxk;
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
                    sum = 1.f / (1.f + exp(-sum));
                }

                outptr[j] = sum;
            }

            outptr += outw;
        }
    }

    return 0;
}

} // namespace ncnn
