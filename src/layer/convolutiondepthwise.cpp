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
#include <algorithm>
#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(ConvolutionDepthWise)

ConvolutionDepthWise::ConvolutionDepthWise()
{
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = true;
    use_int8_requantize = false;

#if NCNN_VULKAN
    padding = 0;
    packing_pack1 = 0;
    packing_pack4 = 0;

    pipeline_convolutiondepthwise = 0;
    pipeline_convolutiondepthwise_pack4 = 0;

    pipeline_convolutiondepthwise_group = 0;
    pipeline_convolutiondepthwise_group_pack4 = 0;
    pipeline_convolutiondepthwise_group_pack1to4 = 0;
    pipeline_convolutiondepthwise_group_pack4to1 = 0;
#endif // NCNN_VULKAN
}

ConvolutionDepthWise::~ConvolutionDepthWise()
{
#if NCNN_VULKAN
    delete padding;
    delete packing_pack1;
    delete packing_pack4;
#endif // NCNN_VULKAN

    for (int i=0; i<(int)quantize_ops.size(); i++)
        delete quantize_ops[i];
    quantize_ops.clear();

    for (int i=0; i<(int)dequantize_ops.size(); i++)
        delete dequantize_ops[i];
    dequantize_ops.clear();

    for (int i=0; i<(int)requantize_ops.size(); i++)
        delete requantize_ops[i];
    requantize_ops.clear();  

    dequantize_scales.clear();
    requantize_scales.clear();      
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

    if (pad_w == -233 && pad_h == -233)
    {
        // TODO
        support_vulkan = false;
    }

    use_int8_inference = pd.use_int8_inference;

    if (num_output % group != 0)
    {
        // reject invalid group
        return -100;
    }

    if (int8_scale_term == 0)
        use_int8_inference = false;

#if NCNN_VULKAN
    if (pd.use_vulkan_compute)
    {
        {
        padding = ncnn::create_layer(ncnn::LayerType::Padding);
        padding->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, pad_h);
        pd.set(1, pad_h);
        pd.set(2, pad_w);
        pd.set(3, pad_w);
        pd.set(4, 0);
        pd.set(5, 0.f);

        pd.use_vulkan_compute = 1;

        padding->load_param(pd);
        }

        {
        packing_pack1 = ncnn::create_layer(ncnn::LayerType::Packing);
        packing_pack1->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 1);
        pd.use_vulkan_compute = 1;

        packing_pack1->load_param(pd);
        }

        {
        packing_pack4 = ncnn::create_layer(ncnn::LayerType::Packing);
        packing_pack4->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 4);
        pd.use_vulkan_compute = 1;

        packing_pack4->load_param(pd);
        }
    }
#endif // NCNN_VULKAN

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
        bottom_blob_int8_scales = mb.load(1, 1);

        float bottom_blob_int8_scale = bottom_blob_int8_scales[0];
        bottom_blob_int8_scales = Mat(group);
        bottom_blob_int8_scales.fill(bottom_blob_int8_scale);
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

    for (int i=0; i<(int)requantize_ops.size(); i++)
        delete requantize_ops[i];
    requantize_ops.clear();

    dequantize_scales.clear();
    requantize_scales.clear();     

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

            float top_rescale = 1.f;
            if (weight_data_int8_scales[g] == 0)
                top_rescale = 0;
            else
                top_rescale = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

            ncnn::ParamDict pd;
            pd.set(0, top_rescale);// scale
            pd.set(1, bias_term);// bias_term
            pd.set(2, 1);// bias_data_size

            dequantize_ops[g]->load_param(pd);

            ncnn::Mat weights[1];
            weights[0] = bias_data.range(g, 1);

            dequantize_ops[g]->load_model(ModelBinFromMatArray(weights));

            dequantize_scales.push_back(top_rescale);
        }
    }

    return 0;
}

int ConvolutionDepthWise::create_requantize_op(void)
{
    if (!use_int8_requantize)
    {
        fprintf(stderr, "requantized op set but use_int8_requantize disabled\n");
        return -1;
    }

    requantize_ops.resize(group);
    for (int g=0; g<group; g++)
    {
        requantize_ops[g] = ncnn::create_layer(ncnn::LayerType::Requantize);

        float scale_in = 1.f;
        float scale_out = 1.f;

        if (weight_data_int8_scales[g] == 0)
        {
            scale_in = 0;
        }
        else
        {
            scale_in = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);
        }

        scale_out = top_blob_int8_scale;

        ncnn::ParamDict pd;
        pd.set(0, scale_in);   // scale in
        pd.set(1, scale_out);  // scale_out
        pd.set(2, bias_term);  // bias_term
        pd.set(3, 1);          // bias_data_size

        requantize_ops[g]->load_param(pd);

        ncnn::Mat weights[1];
        weights[0] = bias_data.range(g, 1);

        requantize_ops[g]->load_model(ModelBinFromMatArray(weights));

        requantize_scales.push_back(scale_in);
        requantize_scales.push_back(scale_out);
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

                    // requantize, reverse scale inplace
                    {
                        ncnn::Option opt_g = opt;
                        opt_g.num_threads = 1;
                        opt_g.blob_allocator = top_blob.allocator;

                        Mat top_blob_tm_g = top_blob_tm.channel_range(g, 1);
                        Mat top_blob_g = top_blob.channel_range(g, 1);
                        requantize_ops[g]->forward(top_blob_tm_g, top_blob_g, opt_g);
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

                // requantize, reverse scale inplace
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g=0; g<group; g++)
                {
                    ncnn::Option opt_g = opt;
                    opt_g.num_threads = 1;
                    opt_g.blob_allocator = top_blob.allocator;

                    Mat top_blob_tm_g = top_blob_tm.channel_range(num_output_g * g, num_output_g);
                    Mat top_blob_g = top_blob.channel_range(num_output_g * g, num_output_g);
                    requantize_ops[g]->forward(top_blob_tm_g, top_blob_g, opt_g);                    
                }
            }            
        }
        else
        {
            top_blob.create(outw, outh, num_output, (size_t)4u, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

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
        }

        return 0;
    }

    // float32
    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;
    
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

#if NCNN_VULKAN
int ConvolutionDepthWise::upload_model(VkTransfer& cmd)
{
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        // pack1
        if (num_output % 4 != 0)
        {
            cmd.record_upload(weight_data, weight_data_gpu);
        }

        // pack4
        if (num_output % 4 == 0)
        {
            Mat weight_data_pack4;
            Mat weight_data_r2 = weight_data.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_pack4, 4);

            weight_data_pack4 = weight_data_pack4.reshape(maxk * (group/4));
            cmd.record_upload(weight_data_pack4, weight_data_gpu_pack4);
        }

        if (bias_term)
        {
            if (num_output % 4 != 0)
            {
                cmd.record_upload(bias_data, bias_data_gpu);
            }

            if (num_output % 4 == 0)
            {
                Mat bias_data_pack4;
                convert_packing(bias_data, bias_data_pack4, 4);
                cmd.record_upload(bias_data_pack4, bias_data_gpu_pack4);
            }
        }

        return 0;
    }

    // group convolution
    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    // pack1
    if (channels_g % 4 != 0 && num_output_g % 4 != 0)
    {
        cmd.record_upload(weight_data, weight_data_gpu);
    }

    // pack4
    if (channels_g % 4 == 0 && num_output_g % 4 == 0)
    {
        // src = kw-kh-inch-outch
        // dst = 4a-4b-kw-kh-inch/4a-outch/4b
        Mat weight_data_pack4_groups;
        {
            Mat weight_data_r2_groups = weight_data.reshape(maxk, channels_g, num_output_g * group);

            weight_data_pack4_groups.create(16*maxk, channels_g/4, num_output_g/4 * group);

            for (int g=0; g<group; g++)
            {
                const Mat weight_data_r2 = weight_data_r2_groups.channel_range(num_output_g * g, num_output_g);

                Mat weight_data_pack4 = weight_data_pack4_groups.channel_range(num_output_g/4 * g, num_output_g/4);

                for (int q=0; q+3<num_output_g; q+=4)
                {
                    const Mat k0 = weight_data_r2.channel(q);
                    const Mat k1 = weight_data_r2.channel(q+1);
                    const Mat k2 = weight_data_r2.channel(q+2);
                    const Mat k3 = weight_data_r2.channel(q+3);

                    Mat g0 = weight_data_pack4.channel(q/4);

                    for (int p=0; p+3<channels_g; p+=4)
                    {
                        const float* k00 = k0.row(p);
                        const float* k01 = k0.row(p+1);
                        const float* k02 = k0.row(p+2);
                        const float* k03 = k0.row(p+3);

                        const float* k10 = k1.row(p);
                        const float* k11 = k1.row(p+1);
                        const float* k12 = k1.row(p+2);
                        const float* k13 = k1.row(p+3);

                        const float* k20 = k2.row(p);
                        const float* k21 = k2.row(p+1);
                        const float* k22 = k2.row(p+2);
                        const float* k23 = k2.row(p+3);

                        const float* k30 = k3.row(p);
                        const float* k31 = k3.row(p+1);
                        const float* k32 = k3.row(p+2);
                        const float* k33 = k3.row(p+3);

                        float* g00 = g0.row(p/4);

                        for (int k=0; k<maxk; k++)
                        {
                            g00[0] = k00[k];
                            g00[1] = k01[k];
                            g00[2] = k02[k];
                            g00[3] = k03[k];

                            g00[4] = k10[k];
                            g00[5] = k11[k];
                            g00[6] = k12[k];
                            g00[7] = k13[k];

                            g00[8] = k20[k];
                            g00[9] = k21[k];
                            g00[10] = k22[k];
                            g00[11] = k23[k];

                            g00[12] = k30[k];
                            g00[13] = k31[k];
                            g00[14] = k32[k];
                            g00[15] = k33[k];

                            g00 += 16;
                        }
                    }
                }
            }
        }

        weight_data_pack4_groups = weight_data_pack4_groups.reshape(16*maxk * (channels_g/4) * (num_output_g/4) * group);
        cmd.record_upload(weight_data_pack4_groups, weight_data_gpu_pack4);
    }

    // pack1to4
    if (channels_g % 4 != 0 && num_output_g % 4 == 0)
    {
        // src = kw-kh-inch-outch
        // dst = 4b-kw-kh-inch-outch/4b
        Mat weight_data_pack1to4_groups;
        {
            Mat weight_data_r2_groups = weight_data.reshape(maxk, channels_g, num_output_g * group);

            weight_data_pack1to4_groups.create(4*maxk, channels_g, num_output_g/4 * group);

            for (int g=0; g<group; g++)
            {
                const Mat weight_data_r2 = weight_data_r2_groups.channel_range(num_output_g * g, num_output_g);

                Mat weight_data_pack1to4 = weight_data_pack1to4_groups.channel_range(num_output_g/4 * g, num_output_g/4);

                for (int q=0; q+3<num_output_g; q+=4)
                {
                    const Mat k0 = weight_data_r2.channel(q);
                    const Mat k1 = weight_data_r2.channel(q+1);
                    const Mat k2 = weight_data_r2.channel(q+2);
                    const Mat k3 = weight_data_r2.channel(q+3);

                    Mat g0 = weight_data_pack1to4.channel(q/4);

                    for (int p=0; p<channels_g; p++)
                    {
                        const float* k00 = k0.row(p);
                        const float* k10 = k1.row(p);
                        const float* k20 = k2.row(p);
                        const float* k30 = k3.row(p);

                        float* g00 = g0.row(p);

                        for (int k=0; k<maxk; k++)
                        {
                            g00[0] = k00[k];
                            g00[1] = k10[k];
                            g00[2] = k20[k];
                            g00[3] = k30[k];

                            g00 += 4;
                        }
                    }
                }
            }
        }

        weight_data_pack1to4_groups = weight_data_pack1to4_groups.reshape(4*maxk * channels_g * (num_output_g/4) * group);
        cmd.record_upload(weight_data_pack1to4_groups, weight_data_gpu_pack1to4);
    }

    // pack4to1
    if (channels_g % 4 == 0 && num_output_g % 4 != 0)
    {
        // src = kw-kh-inch-outch
        // dst = 4a-kw-kh-inch/4a-outch
        Mat weight_data_pack4to1_groups;
        {
            Mat weight_data_r2_groups = weight_data.reshape(maxk, channels_g, num_output_g * group);

            weight_data_pack4to1_groups.create(4*maxk, channels_g/4, num_output_g * group);

            for (int g=0; g<group; g++)
            {
                const Mat weight_data_r2 = weight_data_r2_groups.channel_range(num_output_g * g, num_output_g);

                Mat weight_data_pack4to1 = weight_data_pack4to1_groups.channel_range(num_output_g * g, num_output_g);

                for (int q=0; q<num_output_g; q++)
                {
                    const Mat k0 = weight_data_r2.channel(q);
                    Mat g0 = weight_data_pack4to1.channel(q);

                    for (int p=0; p+3<channels_g; p+=4)
                    {
                        const float* k00 = k0.row(p);
                        const float* k01 = k0.row(p+1);
                        const float* k02 = k0.row(p+2);
                        const float* k03 = k0.row(p+3);

                        float* g00 = g0.row(p/4);

                        for (int k=0; k<maxk; k++)
                        {
                            g00[0] = k00[k];
                            g00[1] = k01[k];
                            g00[2] = k02[k];
                            g00[3] = k03[k];

                            g00 += 4;
                        }
                    }
                }
            }
        }

        weight_data_pack4to1_groups = weight_data_pack4to1_groups.reshape(4*maxk * (channels_g/4) * num_output_g * group);
        cmd.record_upload(weight_data_pack4to1_groups, weight_data_gpu_pack4to1);
    }

    if (bias_term)
    {
        if (num_output_g % 4 != 0)
        {
            cmd.record_upload(bias_data, bias_data_gpu);
        }

        if (num_output_g % 4 == 0)
        {
            Mat bias_data_pack4;
            convert_packing(bias_data, bias_data_pack4, 4);
            cmd.record_upload(bias_data_pack4, bias_data_gpu_pack4);
        }
    }

    return 0;
}

int ConvolutionDepthWise::create_pipeline()
{
    padding->create_pipeline();

    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        std::vector<vk_specialization_type> specializations(8);
        specializations[0].i = kernel_w;
        specializations[1].i = kernel_h;
        specializations[2].i = dilation_w;
        specializations[3].i = dilation_h;
        specializations[4].i = stride_w;
        specializations[5].i = stride_h;
        specializations[6].i = bias_term;
        specializations[7].i = group;

        // pack1
        if (num_output % 4 != 0)
        {
            pipeline_convolutiondepthwise = new Pipeline(vkdev);
            pipeline_convolutiondepthwise->set_optimal_local_size_xyz(32, 32, num_output);
            pipeline_convolutiondepthwise->create("convolutiondepthwise", specializations, 4, 10);
        }

        // pack4
        if (num_output % 4 == 0)
        {
            pipeline_convolutiondepthwise_pack4 = new Pipeline(vkdev);
            pipeline_convolutiondepthwise_pack4->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 4));
            pipeline_convolutiondepthwise_pack4->create("convolutiondepthwise_pack4", specializations, 4, 10);
        }

        return 0;
    }

    // group convolution
    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    std::vector<vk_specialization_type> specializations(8);
    specializations[0].i = kernel_w;
    specializations[1].i = kernel_h;
    specializations[2].i = dilation_w;
    specializations[3].i = dilation_h;
    specializations[4].i = stride_w;
    specializations[5].i = stride_h;
    specializations[6].i = bias_term;
    specializations[7].i = group;

    // pack1
    if (channels_g % 4 != 0 && num_output_g % 4 != 0)
    {
        pipeline_convolutiondepthwise_group = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolutiondepthwise_group->create("convolutiondepthwise_group", specializations, 4, 10);
    }

    // pack4
    if (channels_g % 4 == 0 && num_output_g % 4 == 0)
    {
        pipeline_convolutiondepthwise_group_pack4 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack4->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolutiondepthwise_group_pack4->create("convolutiondepthwise_group_pack4", specializations, 4, 10);
    }

    // pack1to4
    if (channels_g % 4 != 0 && num_output_g % 4 == 0)
    {
        pipeline_convolutiondepthwise_group_pack1to4 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack1to4->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolutiondepthwise_group_pack1to4->create("convolutiondepthwise_group_pack1to4", specializations, 4, 10);
    }

    // pack4to1
    if (channels_g % 4 == 0 && num_output_g % 4 != 0)
    {
        pipeline_convolutiondepthwise_group_pack4to1 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_group_pack4to1->set_optimal_local_size_xyz(32, 32, std::max(1, num_output / 8));
        pipeline_convolutiondepthwise_group_pack4to1->create("convolutiondepthwise_group_pack4to1", specializations, 4, 10);
    }

    if (channels % 4 == 0 && channels_g % 4 != 0)
    {
        packing_pack1->create_pipeline();
    }

    if (num_output_g % 4 != 0 && num_output % 4 == 0)
    {
        packing_pack4->create_pipeline();
    }

    return 0;
}

int ConvolutionDepthWise::destroy_pipeline()
{
    if (padding)
        padding->destroy_pipeline();

    if (packing_pack1)
        packing_pack1->destroy_pipeline();

    if (packing_pack4)
        packing_pack4->destroy_pipeline();

    delete pipeline_convolutiondepthwise;
    pipeline_convolutiondepthwise = 0;

    delete pipeline_convolutiondepthwise_pack4;
    pipeline_convolutiondepthwise_pack4 = 0;

    delete pipeline_convolutiondepthwise_group;
    pipeline_convolutiondepthwise_group = 0;

    delete pipeline_convolutiondepthwise_group_pack4;
    pipeline_convolutiondepthwise_group_pack4 = 0;

    delete pipeline_convolutiondepthwise_group_pack1to4;
    pipeline_convolutiondepthwise_group_pack1to4 = 0;

    delete pipeline_convolutiondepthwise_group_pack4to1;
    pipeline_convolutiondepthwise_group_pack4to1 = 0;

    return 0;
}

int ConvolutionDepthWise::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int packing = bottom_blob.packing;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    VkMat bottom_blob_bordered = bottom_blob;
    if (pad_w > 0 || pad_h > 0)
    {
        ncnn::Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_packing = num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / packing * out_packing;

    top_blob.create(outw, outh, num_output / out_packing, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

//     fprintf(stderr, "ConvolutionDepthWise::forward %p %p\n", bottom_blob_bordered.buffer(), top_blob.buffer());

    // depth-wise
    if (channels == group / packing && group / packing == num_output / packing)
    {
        std::vector<VkMat> bindings(4);
        bindings[0] = bottom_blob_bordered;
        bindings[1] = top_blob;
        bindings[2] = packing == 4 ? weight_data_gpu_pack4 : weight_data_gpu;
        bindings[3] = bias_term ? (packing == 4 ? bias_data_gpu_pack4 : bias_data_gpu) : bindings[2];// TODO use dummy buffer

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob_bordered.dims;
        constants[1].i = bottom_blob_bordered.w;
        constants[2].i = bottom_blob_bordered.h;
        constants[3].i = bottom_blob_bordered.c;
        constants[4].i = bottom_blob_bordered.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;

        const Pipeline* pipeline = packing == 4 ? pipeline_convolutiondepthwise_pack4 : pipeline_convolutiondepthwise;

        // record
        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    const int channels_g = channels * packing / group;
    const int num_output_g = num_output / group;

    // unpacking
    VkMat bottom_blob_bordered_unpacked = bottom_blob_bordered;
    if (packing == 4 && channels_g % 4 != 0)
    {
        ncnn::Option opt_pack1 = opt;
        opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

        packing_pack1->forward(bottom_blob_bordered, bottom_blob_bordered_unpacked, cmd, opt_pack1);
    }

    VkMat top_blob_unpacked = top_blob;
    if (num_output_g % 4 != 0 && out_packing == 4)
    {
        top_blob_unpacked.create(outw, outh, num_output, elemsize / packing, 1, opt.workspace_vkallocator, opt.staging_vkallocator);
        if (top_blob_unpacked.empty())
            return -100;
    }

    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob_bordered_unpacked;
    bindings[1] = top_blob_unpacked;
    if (channels_g % 4 != 0 && num_output_g % 4 != 0)
    {
        bindings[2] = weight_data_gpu;
        bindings[3] = bias_term ? bias_data_gpu : bindings[2];// TODO use dummy buffer
    }
    else if (channels_g % 4 == 0 && num_output_g % 4 == 0)
    {
        bindings[2] = weight_data_gpu_pack4;
        bindings[3] = bias_term ? bias_data_gpu_pack4 : bindings[2];// TODO use dummy buffer
    }
    else if (channels_g % 4 != 0 && num_output_g % 4 == 0)
    {
        bindings[2] = weight_data_gpu_pack1to4;
        bindings[3] = bias_term ? bias_data_gpu_pack4 : bindings[2];// TODO use dummy buffer
    }
    else if (channels_g % 4 == 0 && num_output_g % 4 != 0)
    {
        bindings[2] = weight_data_gpu_pack4to1;
        bindings[3] = bias_term ? bias_data_gpu : bindings[2];// TODO use dummy buffer
    }

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_bordered_unpacked.dims;
    constants[1].i = bottom_blob_bordered_unpacked.w;
    constants[2].i = bottom_blob_bordered_unpacked.h;
    constants[3].i = bottom_blob_bordered_unpacked.c;
    constants[4].i = bottom_blob_bordered_unpacked.cstep;
    constants[5].i = top_blob_unpacked.dims;
    constants[6].i = top_blob_unpacked.w;
    constants[7].i = top_blob_unpacked.h;
    constants[8].i = top_blob_unpacked.c;
    constants[9].i = top_blob_unpacked.cstep;

    const Pipeline* pipeline = 0;
    if (channels_g % 4 != 0 && num_output_g % 4 != 0)
    {
        pipeline = pipeline_convolutiondepthwise_group;
    }
    else if (channels_g % 4 == 0 && num_output_g % 4 == 0)
    {
        pipeline = pipeline_convolutiondepthwise_group_pack4;
    }
    else if (channels_g % 4 != 0 && num_output_g % 4 == 0)
    {
        pipeline = pipeline_convolutiondepthwise_group_pack1to4;
    }
    else if (channels_g % 4 == 0 && num_output_g % 4 != 0)
    {
        pipeline = pipeline_convolutiondepthwise_group_pack4to1;
    }

    // record
    cmd.record_pipeline(pipeline, bindings, constants, top_blob_unpacked);

    // packing
    if (num_output_g % 4 != 0 && out_packing == 4)
    {
        packing_pack4->forward(top_blob_unpacked, top_blob, cmd, opt);
    }
    else
    {
        top_blob = top_blob_unpacked;
    }

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
