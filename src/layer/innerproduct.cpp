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

#include "layer_type.h"

namespace ncnn {

InnerProduct::InnerProduct()
{
    one_blob_only = true;
    support_inplace = false;
}

int InnerProduct::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    bias_term = pd.get(1, 0);
    weight_data_size = pd.get(2, 0);
    int8_scale_term = pd.get(8, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    if (int8_scale_term)
    {
        use_int8_inference = true;
    }

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
    // runtime quantize the weight data
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)4u && int8_scale_term)
    {
        Mat int8_weight_data(weight_data_size, (size_t)1u);
        if (int8_weight_data.empty())
            return -100;

        const int weight_data_size_output = weight_data_size / num_output;

        for (int p = 0; p < num_output; p++)
        {
            Option opt_q = opt;
            opt_q.blob_allocator = int8_weight_data.allocator;

            const Mat weight_data_n = weight_data.range(weight_data_size_output * p, weight_data_size_output);
            Mat int8_weight_data_n = int8_weight_data.range(weight_data_size_output * p, weight_data_size_output);
            quantize_float32_to_int8(weight_data_n, int8_weight_data_n, weight_data_int8_scales[p], opt_q);
        }

        weight_data = int8_weight_data;
    }

    return 0;
}

int InnerProduct::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return InnerProduct::forward_int8(bottom_blob, top_blob, opt);
    }

    const int num_input = weight_data_size / num_output;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int size = w * h;

    if (bottom_blob.dims == 2 && w == num_input && h > 1)
    {
        // gemm
        top_blob.create(num_output, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
            const float* m = bottom_blob.row(j);
            float* outptr = top_blob.row(j);

            for (int p = 0; p < num_output; p++)
            {
                const float* kptr = (const float*)weight_data + w * p;

                float sum = 0.f;

                if (bias_term)
                    sum = bias_data[p];

                for (int i = 0; i < w; i++)
                {
                    sum += m[i] * kptr[i];
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
                    sum = static_cast<float>(sum * tanh(log(exp(sum) + 1.f)));
                }

                outptr[p] = sum;
            }
        }

        return 0;
    }

    top_blob.create(num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < num_output; p++)
    {
        float sum = 0.f;

        if (bias_term)
            sum = bias_data[p];

        // channels
        for (int q = 0; q < channels; q++)
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
            sum = static_cast<float>(1.f / (1.f + exp(-sum)));
        }
        else if (activation_type == 5)
        {
            sum = static_cast<float>(sum * tanh(log(exp(sum) + 1.f)));
        }

        top_blob[p] = sum;
    }

    return 0;
}

int InnerProduct::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int num_input = weight_data_size / num_output;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int size = w * h;

    Mat bottom_blob_tm = bottom_blob;
    if (elemsize != 1)
    {
        Option opt_g = opt;
        opt_g.blob_allocator = opt.workspace_allocator;

        quantize_float32_to_int8(bottom_blob, bottom_blob_tm, bottom_blob_int8_scale, opt_g);
    }

    if (bottom_blob.dims == 2 && w == num_input && h > 1)
    {
        // gemm
        top_blob.create(num_output, h, 4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int j = 0; j < h; j++)
        {
            const signed char* m = bottom_blob_tm.row<signed char>(j);
            const signed char* kptr = weight_data;
            float* outptr = top_blob.row(j);

            for (int p = 0; p < num_output; p++)
            {
                int sum = 0;

                for (int i = 0; i < w; i++)
                {
                    sum += m[i] * kptr[i];
                }

                // dequantize and relu
                float scale_in;
                if (weight_data_int8_scales[p] == 0)
                    scale_in = 0;
                else
                    scale_in = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[p]);

                float sumfp32 = sum * scale_in;

                if (bias_term)
                    sumfp32 += bias_data[p];

                if (activation_type == 1)
                {
                    sumfp32 = std::max(sumfp32, 0.f);
                }

                outptr[p] = sumfp32;
            }
        }

        return 0;
    }

    top_blob.create(num_output, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < num_output; p++)
    {
        float* outptr = top_blob;

        int sum = 0;

        int offset = size * channels * p;
        // channels
        for (int q = 0; q < channels; q++)
        {
            const signed char* w = (const signed char*)weight_data + offset + size * q;
            const signed char* m = bottom_blob_tm.channel(q);

            for (int i = 0; i < size; i++)
            {
                sum += m[i] * w[i];
            }
        }

        // dequantize and relu
        float scale_in;
        if (weight_data_int8_scales[p] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[p]);

        float sumfp32 = sum * scale_in;

        if (bias_term)
            sumfp32 += bias_data[p];

        if (activation_type == 1)
        {
            sumfp32 = std::max(sumfp32, 0.f);
        }

        outptr[p] = sumfp32;
    }

    return 0;
}

} // namespace ncnn
