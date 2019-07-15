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

#include "innerproduct.h"
#include <algorithm>
#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(InnerProduct)

InnerProduct::InnerProduct()
{
    one_blob_only = true;
    support_inplace = false;

    quantize = 0;
}

int InnerProduct::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    bias_term = pd.get(1, 0);
    weight_data_size = pd.get(2, 0);
    int8_scale_term = pd.get(8, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    return 0;
}

int InnerProduct::load_model(const ModelBin& mb)
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

int InnerProduct::create_pipeline(const Option& opt)
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

            ncnn::Mat weights[1];
            weights[0] = bias_data.range(n, 1);

            dequantize_ops[n]->load_model(ModelBinFromMatArray(weights));

            dequantize_ops[n]->create_pipeline(opt_cpu);
        }
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

    return 0;
}

int InnerProduct::destroy_pipeline(const Option& opt)
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

    return 0;
}

int InnerProduct::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int size = w * h;

    top_blob.create(num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (use_int8_inference)
    {
        Mat bottom_blob_tm = bottom_blob;
        if (elemsize != 1)
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

            bottom_blob_tm = bottom_blob_int8;
        }

        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output; p++)
        {
            int sum = 0;
            int* out = top_blob;

            // channels
            for (int q=0; q<channels; q++)
            {
                const signed char* w = (const signed char*)weight_data + size * channels * p + size * q;
                const signed char* m = bottom_blob_tm.channel(q);

                for (int i = 0; i < size; i++)
                {
                    sum += m[i] * w[i];
                }
            }

            out[p] = sum;       
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<num_output; p++)
        {
            int* out_s32 = top_blob;
            float* out_f32 = top_blob;
            float top_rescale = 1.f;
            if (weight_data_int8_scales[p] == 0)
                top_rescale = 0;
            else
                top_rescale = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[p]);

            if (bias_term)
                out_f32[p] = out_s32[p] * top_rescale + bias_data[p];
            else
                out_f32[p] = out_s32[p] * top_rescale;

            if (activation_type == 1)
            {
                out_f32[p] = std::max(out_f32[p], 0.f);
            }                
        }

        return 0;
    }

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<num_output; p++)
    {
        float sum = 0.f;

        if (bias_term)
            sum = bias_data[p];

        // channels
        for (int q=0; q<channels; q++)
        {
            const float* w = (const float*)weight_data + size * channels * p + size * q;
            const float* m = bottom_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                sum += m[i] * w[i];
            }
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

        top_blob[p] = sum;
    }

    return 0;
}

} // namespace ncnn
