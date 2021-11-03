// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <algorithm>
#include <map>
#include <set>
#include <vector>

// ncnn public header
#include "datareader.h"
#include "layer.h"
#include "layer_type.h"
#include "net.h"

// ncnn private header
#include "modelwriter.h"

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* format, void* p) const
    {
        return 0;
    }
    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

class NetOptimize : public ModelWriter
{
public:
    NetOptimize();

public:
    int fuse_batchnorm_scale();
    int fuse_convolution_batchnorm();
    int fuse_convolution_mul();
    int fuse_convolution_add();
    int fuse_convolutiondepthwise_batchnorm();
    int fuse_convolutiondepthwise_mul();
    int fuse_convolutiondepthwise_add();
    int fuse_deconvolution_batchnorm();
    int fuse_deconvolution_mul();
    int fuse_deconvolution_add();
    int fuse_deconvolutiondepthwise_batchnorm();
    int fuse_innerproduct_batchnorm();
    int fuse_innerproduct_add();
    int fuse_innerproduct_dropout();
    int fuse_convolution_activation();
    int fuse_convolutiondepthwise_activation();
    int fuse_deconvolution_activation();
    int fuse_deconvolutiondepthwise_activation();
    int fuse_innerproduct_activation();
    int fuse_memorydata_binaryop();
    int fuse_binaryop_eltwise();

    int eliminate_dropout();
    int eliminate_pooling1x1();
    int eliminate_noop();
    int eliminate_split();
    int eliminate_orphaned_memorydata();
    int eliminate_flatten_after_global_pooling();
    int eliminate_reshape_after_global_pooling();
    int eliminate_flatten_after_innerproduct();
    int eliminate_reshape_before_binaryop();

    int replace_reduction_with_global_pooling();
    int replace_prelu_with_leaky_relu();
    int replace_convolution_with_innerproduct_after_global_pooling();
    int replace_convolution_with_innerproduct_after_innerproduct();
};

NetOptimize::NetOptimize()
    : ModelWriter()
{
}

int NetOptimize::fuse_batchnorm_scale()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "BatchNorm")
            continue;

        // BatchNorm - Scale
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "Scale")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse BatchNorm - Scale to BatchNorm
        ncnn::BatchNorm* batchnorm = (ncnn::BatchNorm*)layers[i];
        ncnn::Scale* scale = (ncnn::Scale*)layers[j];

        fprintf(stderr, "fuse_batchnorm_scale %s %s\n", batchnorm->name.c_str(), scale->name.c_str());

        {
            //             v = ((v - mean) / sqrt(var + eps) * slope + bias) * s + b
            //               =  (v - mean) / sqrt(var + eps) * (slope * s) + (bias * s + b)

            int channels = batchnorm->channels;

            float* slope = batchnorm->slope_data;
            float* bias = batchnorm->bias_data;

            for (int q = 0; q < channels; q++)
            {
                slope[q] = slope[q] * scale->scale_data[q];
                if (scale->bias_term)
                    bias[q] = bias[q] * scale->scale_data[q] + scale->bias_data[q];
                else
                    bias[q] = bias[q] * scale->scale_data[q];
            }
        }

        int top_blob_index_final = scale->tops[0];
        batchnorm->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        scale->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_convolution_batchnorm()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Convolution")
            continue;

        // Convolution - BatchNorm
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "BatchNorm")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse Convolution - BatchNorm to Convolution
        ncnn::Convolution* convolution = (ncnn::Convolution*)layers[i];
        ncnn::BatchNorm* batchnorm = (ncnn::BatchNorm*)layers[j];

        fprintf(stderr, "fuse_convolution_batchnorm %s %s\n", convolution->name.c_str(), batchnorm->name.c_str());

        {
            int channels = batchnorm->channels;
            float eps = batchnorm->eps;

            // a = bias - slope * mean / sqrt(var + eps)
            // b = slope / sqrt(var + eps)
            // value = value * b + a

            std::vector<float> a(channels);
            std::vector<float> b(channels);
            for (int i = 0; i < channels; i++)
            {
                float sqrt_var = static_cast<float>(sqrt(batchnorm->var_data[i] + eps));
                a[i] = batchnorm->bias_data[i] - batchnorm->slope_data[i] * batchnorm->mean_data[i] / sqrt_var;
                b[i] = batchnorm->slope_data[i] / sqrt_var;
            }

            if (convolution->bias_term == 0)
            {
                // init bias as zero
                convolution->bias_term = 1;
                convolution->bias_data = ncnn::Mat(channels);
                convolution->bias_data.fill(0.f);
            }

            const int weight_per_outch = convolution->weight_data_size / channels;

            float* weight = convolution->weight_data;
            float* bias = convolution->bias_data;
            for (int i = 0; i < channels; i++)
            {
                float* conv_weight_outch = weight + weight_per_outch * i;
                for (int j = 0; j < weight_per_outch; j++)
                {
                    conv_weight_outch[j] *= b[i];
                }

                bias[i] = bias[i] * b[i] + a[i];
            }
        }

        int top_blob_index_final = batchnorm->tops[0];
        convolution->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        batchnorm->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_convolution_mul()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Convolution")
            continue;

        // Convolution - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "BinaryOp")
                continue;

            if (layers[j]->bottoms.size() != 2)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse Convolution - BinaryOp to Convolution
        ncnn::Convolution* convolution = (ncnn::Convolution*)layers[i];
        ncnn::BinaryOp* binaryop = (ncnn::BinaryOp*)layers[j];

        if (binaryop->op_type != 2 || binaryop->with_scalar)
            continue;

        // MemoryData - ..... - BinaryOp
        size_t k = 0;
        for (; k < j; k++)
        {
            if (layers[k]->type != "MemoryData")
                continue;

            if (layers[k]->tops[0] == binaryop->bottoms[1])
                break;
        }

        if (k == j)
            continue;

        ncnn::MemoryData* memorydata = (ncnn::MemoryData*)layers[k];

        int channels = convolution->num_output;

        if (memorydata->w != channels || memorydata->h != 0 || memorydata->c != 0)
        {
            // not bias-like broadcasting type
            continue;
        }

        fprintf(stderr, "fuse_convolution_mul %s %s\n", convolution->name.c_str(), binaryop->name.c_str());

        {
            const int weight_per_outch = convolution->weight_data_size / channels;

            float* weight = convolution->weight_data;
            float* bias = convolution->bias_data;
            for (int i = 0; i < channels; i++)
            {
                float* conv_weight_outch = weight + weight_per_outch * i;
                for (int j = 0; j < weight_per_outch; j++)
                {
                    conv_weight_outch[j] *= memorydata->data[i];
                }

                if (bias)
                {
                    bias[i] = bias[i] * memorydata->data[i];
                }
            }
        }

        int top_blob_index_final = binaryop->tops[0];
        convolution->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        binaryop->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_convolution_add()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Convolution")
            continue;

        // Convolution - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "BinaryOp")
                continue;

            if (layers[j]->bottoms.size() != 2)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse Convolution - BinaryOp to Convolution
        ncnn::Convolution* convolution = (ncnn::Convolution*)layers[i];
        ncnn::BinaryOp* binaryop = (ncnn::BinaryOp*)layers[j];

        if (binaryop->op_type != 0 || binaryop->with_scalar)
            continue;

        // MemoryData - ..... - BinaryOp
        size_t k = 0;
        for (; k < j; k++)
        {
            if (layers[k]->type != "MemoryData")
                continue;

            if (layers[k]->tops[0] == binaryop->bottoms[1])
                break;
        }

        if (k == j)
            continue;

        ncnn::MemoryData* memorydata = (ncnn::MemoryData*)layers[k];

        int channels = convolution->num_output;

        bool broadcasting_type_ok = false;
        if (memorydata->w == channels && memorydata->h == 0 && memorydata->c == 0)
            broadcasting_type_ok = true;
        if (memorydata->w == 1 && memorydata->h == 1 && memorydata->c == channels)
            broadcasting_type_ok = true;

        if (!broadcasting_type_ok)
        {
            // not bias-like broadcasting type
            continue;
        }

        fprintf(stderr, "fuse_convolution_add %s %s\n", convolution->name.c_str(), binaryop->name.c_str());

        ncnn::Mat bias_data = memorydata->data.reshape(channels);
        {
            if (convolution->bias_term == 0)
            {
                // init bias
                convolution->bias_term = 1;
                convolution->bias_data = bias_data;
            }
            else
            {
                float* bias = convolution->bias_data;
                for (int i = 0; i < channels; i++)
                {
                    bias[i] = bias[i] + bias_data[i];
                }
            }
        }

        int top_blob_index_final = binaryop->tops[0];
        convolution->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        binaryop->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_convolutiondepthwise_batchnorm()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "ConvolutionDepthWise")
            continue;

        // ConvolutionDepthWise - BatchNorm
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "BatchNorm")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse ConvolutionDepthWise - BatchNorm to ConvolutionDepthWise
        ncnn::ConvolutionDepthWise* convolutiondepthwise = (ncnn::ConvolutionDepthWise*)layers[i];
        ncnn::BatchNorm* batchnorm = (ncnn::BatchNorm*)layers[j];

        fprintf(stderr, "fuse_convolutiondepthwise_batchnorm %s %s\n", convolutiondepthwise->name.c_str(), batchnorm->name.c_str());

        {
            int channels = batchnorm->channels;
            float eps = batchnorm->eps;

            // a = bias - slope * mean / sqrt(var + eps)
            // b = slope / sqrt(var + eps)
            // value = value * b + a

            std::vector<float> a(channels);
            std::vector<float> b(channels);
            for (int i = 0; i < channels; i++)
            {
                float sqrt_var = static_cast<float>(sqrt(batchnorm->var_data[i] + eps));
                a[i] = batchnorm->bias_data[i] - batchnorm->slope_data[i] * batchnorm->mean_data[i] / sqrt_var;
                b[i] = batchnorm->slope_data[i] / sqrt_var;
            }

            if (convolutiondepthwise->bias_term == 0)
            {
                // init bias as zero
                convolutiondepthwise->bias_term = 1;
                convolutiondepthwise->bias_data = ncnn::Mat(channels);
                convolutiondepthwise->bias_data.fill(0.f);
            }

            const int weight_per_outch = convolutiondepthwise->weight_data_size / channels;

            float* weight = convolutiondepthwise->weight_data;
            float* bias = convolutiondepthwise->bias_data;
            for (int i = 0; i < channels; i++)
            {
                float* conv_weight_outch = weight + weight_per_outch * i;
                for (int j = 0; j < weight_per_outch; j++)
                {
                    conv_weight_outch[j] *= b[i];
                }

                bias[i] = bias[i] * b[i] + a[i];
            }
        }

        int top_blob_index_final = batchnorm->tops[0];
        convolutiondepthwise->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        batchnorm->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_convolutiondepthwise_mul()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "ConvolutionDepthWise")
            continue;

        // ConvolutionDepthWise - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "BinaryOp")
                continue;

            if (layers[j]->bottoms.size() != 2)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse ConvolutionDepthWise - BinaryOp to ConvolutionDepthWise
        ncnn::ConvolutionDepthWise* convolutiondepthwise = (ncnn::ConvolutionDepthWise*)layers[i];
        ncnn::BinaryOp* binaryop = (ncnn::BinaryOp*)layers[j];

        if (binaryop->op_type != 2 || binaryop->with_scalar)
            continue;

        // MemoryData - ..... - BinaryOp
        size_t k = 0;
        for (; k < j; k++)
        {
            if (layers[k]->type != "MemoryData")
                continue;

            if (layers[k]->tops[0] == binaryop->bottoms[1])
                break;
        }

        if (k == j)
            continue;

        ncnn::MemoryData* memorydata = (ncnn::MemoryData*)layers[k];

        int channels = convolutiondepthwise->num_output;

        if (memorydata->w != channels || memorydata->h != 0 || memorydata->c != 0)
        {
            // not bias-like broadcasting type
            continue;
        }

        fprintf(stderr, "fuse_convolutiondepthwise_mul %s %s\n", convolutiondepthwise->name.c_str(), binaryop->name.c_str());

        {
            const int weight_per_outch = convolutiondepthwise->weight_data_size / channels;

            float* weight = convolutiondepthwise->weight_data;
            float* bias = convolutiondepthwise->bias_data;
            for (int i = 0; i < channels; i++)
            {
                float* conv_weight_outch = weight + weight_per_outch * i;
                for (int j = 0; j < weight_per_outch; j++)
                {
                    conv_weight_outch[j] *= memorydata->data[i];
                }

                if (bias)
                {
                    bias[i] = bias[i] * memorydata->data[i];
                }
            }
        }

        int top_blob_index_final = binaryop->tops[0];
        convolutiondepthwise->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        binaryop->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_convolutiondepthwise_add()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "ConvolutionDepthWise")
            continue;

        // ConvolutionDepthWise - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "BinaryOp")
                continue;

            if (layers[j]->bottoms.size() != 2)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse ConvolutionDepthWise - BinaryOp to ConvolutionDepthWise
        ncnn::ConvolutionDepthWise* convolutiondepthwise = (ncnn::ConvolutionDepthWise*)layers[i];
        ncnn::BinaryOp* binaryop = (ncnn::BinaryOp*)layers[j];

        if (binaryop->op_type != 0 || binaryop->with_scalar)
            continue;

        // MemoryData - ..... - BinaryOp
        size_t k = 0;
        for (; k < j; k++)
        {
            if (layers[k]->type != "MemoryData")
                continue;

            if (layers[k]->tops[0] == binaryop->bottoms[1])
                break;
        }

        if (k == j)
            continue;

        ncnn::MemoryData* memorydata = (ncnn::MemoryData*)layers[k];

        int channels = convolutiondepthwise->num_output;

        bool broadcasting_type_ok = false;
        if (memorydata->w == channels && memorydata->h == 0 && memorydata->c == 0)
            broadcasting_type_ok = true;
        if (memorydata->w == 1 && memorydata->h == 1 && memorydata->c == channels)
            broadcasting_type_ok = true;

        if (!broadcasting_type_ok)
        {
            // not bias-like broadcasting type
            continue;
        }

        fprintf(stderr, "fuse_convolutiondepthwise_add %s %s\n", convolutiondepthwise->name.c_str(), binaryop->name.c_str());

        ncnn::Mat bias_data = memorydata->data.reshape(channels);
        {
            if (convolutiondepthwise->bias_term == 0)
            {
                // init bias
                convolutiondepthwise->bias_term = 1;
                convolutiondepthwise->bias_data = bias_data;
            }
            else
            {
                float* bias = convolutiondepthwise->bias_data;
                for (int i = 0; i < channels; i++)
                {
                    bias[i] = bias[i] + bias_data[i];
                }
            }
        }

        int top_blob_index_final = binaryop->tops[0];
        convolutiondepthwise->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        binaryop->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_deconvolution_batchnorm()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Deconvolution")
            continue;

        // Deconvolution - BatchNorm
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "BatchNorm")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse Deconvolution - BatchNorm to Deconvolution
        ncnn::Deconvolution* deconvolution = (ncnn::Deconvolution*)layers[i];
        ncnn::BatchNorm* batchnorm = (ncnn::BatchNorm*)layers[j];

        fprintf(stderr, "fuse_deconvolution_batchnorm %s %s\n", deconvolution->name.c_str(), batchnorm->name.c_str());

        {
            int channels = batchnorm->channels;
            float eps = batchnorm->eps;

            // a = bias - slope * mean / sqrt(var + eps)
            // b = slope / sqrt(var + eps)
            // value = value * b + a

            std::vector<float> a(channels);
            std::vector<float> b(channels);
            for (int i = 0; i < channels; i++)
            {
                float sqrt_var = static_cast<float>(sqrt(batchnorm->var_data[i] + eps));
                a[i] = batchnorm->bias_data[i] - batchnorm->slope_data[i] * batchnorm->mean_data[i] / sqrt_var;
                b[i] = batchnorm->slope_data[i] / sqrt_var;
            }

            if (deconvolution->bias_term == 0)
            {
                // init bias as zero
                deconvolution->bias_term = 1;
                deconvolution->bias_data = ncnn::Mat(channels);
                deconvolution->bias_data.fill(0.f);
            }

            const int weight_per_outch = deconvolution->weight_data_size / channels;

            float* weight = deconvolution->weight_data;
            float* bias = deconvolution->bias_data;
            for (int i = 0; i < channels; i++)
            {
                float* conv_weight_outch = weight + weight_per_outch * i;
                for (int j = 0; j < weight_per_outch; j++)
                {
                    conv_weight_outch[j] *= b[i];
                }

                bias[i] = bias[i] * b[i] + a[i];
            }
        }

        int top_blob_index_final = batchnorm->tops[0];
        deconvolution->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        batchnorm->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_deconvolution_mul()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Deconvolution")
            continue;

        // Deconvolution - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "BinaryOp")
                continue;

            if (layers[j]->bottoms.size() != 2)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse Deconvolution - BinaryOp to Deconvolution
        ncnn::Deconvolution* deconvolution = (ncnn::Deconvolution*)layers[i];
        ncnn::BinaryOp* binaryop = (ncnn::BinaryOp*)layers[j];

        if (binaryop->op_type != 2 || binaryop->with_scalar)
            continue;

        // MemoryData - ..... - BinaryOp
        size_t k = 0;
        for (; k < j; k++)
        {
            if (layers[k]->type != "MemoryData")
                continue;

            if (layers[k]->tops[0] == binaryop->bottoms[1])
                break;
        }

        if (k == j)
            continue;

        ncnn::MemoryData* memorydata = (ncnn::MemoryData*)layers[k];

        int channels = deconvolution->num_output;

        if (memorydata->w != channels || memorydata->h != 0 || memorydata->c != 0)
        {
            // not bias-like broadcasting type
            continue;
        }

        fprintf(stderr, "fuse_deconvolution_mul %s %s\n", deconvolution->name.c_str(), binaryop->name.c_str());

        {
            const int weight_per_outch = deconvolution->weight_data_size / channels;

            float* weight = deconvolution->weight_data;
            float* bias = deconvolution->bias_data;
            for (int i = 0; i < channels; i++)
            {
                float* conv_weight_outch = weight + weight_per_outch * i;
                for (int j = 0; j < weight_per_outch; j++)
                {
                    conv_weight_outch[j] *= memorydata->data[i];
                }

                if (bias)
                {
                    bias[i] = bias[i] * memorydata->data[i];
                }
            }
        }

        int top_blob_index_final = binaryop->tops[0];
        deconvolution->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        binaryop->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_deconvolution_add()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Deconvolution")
            continue;

        // Deconvolution - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "BinaryOp")
                continue;

            if (layers[j]->bottoms.size() != 2)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse Deconvolution - BinaryOp to Deconvolution
        ncnn::Deconvolution* deconvolution = (ncnn::Deconvolution*)layers[i];
        ncnn::BinaryOp* binaryop = (ncnn::BinaryOp*)layers[j];

        if (binaryop->op_type != 0 || binaryop->with_scalar)
            continue;

        // MemoryData - ..... - BinaryOp
        size_t k = 0;
        for (; k < j; k++)
        {
            if (layers[k]->type != "MemoryData")
                continue;

            if (layers[k]->tops[0] == binaryop->bottoms[1])
                break;
        }

        if (k == j)
            continue;

        ncnn::MemoryData* memorydata = (ncnn::MemoryData*)layers[k];

        int channels = deconvolution->num_output;

        bool broadcasting_type_ok = false;
        if (memorydata->w == channels && memorydata->h == 0 && memorydata->c == 0)
            broadcasting_type_ok = true;
        if (memorydata->w == 1 && memorydata->h == 1 && memorydata->c == channels)
            broadcasting_type_ok = true;

        if (!broadcasting_type_ok)
        {
            // not bias-like broadcasting type
            continue;
        }

        fprintf(stderr, "fuse_deconvolution_add %s %s\n", deconvolution->name.c_str(), binaryop->name.c_str());

        ncnn::Mat bias_data = memorydata->data.reshape(channels);
        {
            if (deconvolution->bias_term == 0)
            {
                // init bias
                deconvolution->bias_term = 1;
                deconvolution->bias_data = bias_data;
            }
            else
            {
                float* bias = deconvolution->bias_data;
                for (int i = 0; i < channels; i++)
                {
                    bias[i] = bias[i] + bias_data[i];
                }
            }
        }

        int top_blob_index_final = binaryop->tops[0];
        deconvolution->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        binaryop->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_deconvolutiondepthwise_batchnorm()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "DeconvolutionDepthWise")
            continue;

        // DeconvolutionDepthWise - BatchNorm
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "BatchNorm")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse DeconvolutionDepthWise - BatchNorm to DeconvolutionDepthWise
        ncnn::DeconvolutionDepthWise* deconvolutiondepthwise = (ncnn::DeconvolutionDepthWise*)layers[i];
        ncnn::BatchNorm* batchnorm = (ncnn::BatchNorm*)layers[j];

        fprintf(stderr, "fuse_deconvolutiondepthwise_batchnorm %s %s\n", deconvolutiondepthwise->name.c_str(), batchnorm->name.c_str());

        {
            int channels = batchnorm->channels;
            float eps = batchnorm->eps;

            // a = bias - slope * mean / sqrt(var + eps)
            // b = slope / sqrt(var + eps)
            // value = value * b + a

            std::vector<float> a(channels);
            std::vector<float> b(channels);
            for (int i = 0; i < channels; i++)
            {
                float sqrt_var = static_cast<float>(sqrt(batchnorm->var_data[i] + eps));
                a[i] = batchnorm->bias_data[i] - batchnorm->slope_data[i] * batchnorm->mean_data[i] / sqrt_var;
                b[i] = batchnorm->slope_data[i] / sqrt_var;
            }

            if (deconvolutiondepthwise->bias_term == 0)
            {
                // init bias as zero
                deconvolutiondepthwise->bias_term = 1;
                deconvolutiondepthwise->bias_data = ncnn::Mat(channels);
                deconvolutiondepthwise->bias_data.fill(0.f);
            }

            const int weight_per_outch = deconvolutiondepthwise->weight_data_size / channels;

            float* weight = deconvolutiondepthwise->weight_data;
            float* bias = deconvolutiondepthwise->bias_data;
            for (int i = 0; i < channels; i++)
            {
                float* conv_weight_outch = weight + weight_per_outch * i;
                for (int j = 0; j < weight_per_outch; j++)
                {
                    conv_weight_outch[j] *= b[i];
                }

                bias[i] = bias[i] * b[i] + a[i];
            }
        }

        int top_blob_index_final = batchnorm->tops[0];
        deconvolutiondepthwise->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        batchnorm->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_innerproduct_batchnorm()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "InnerProduct")
            continue;

        // InnerProduct - BatchNorm
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "BatchNorm")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse InnerProduct - BatchNorm to InnerProduct
        ncnn::InnerProduct* innerproduct = (ncnn::InnerProduct*)layers[i];
        ncnn::BatchNorm* batchnorm = (ncnn::BatchNorm*)layers[j];

        fprintf(stderr, "fuse_innerproduct_batchnorm %s %s\n", innerproduct->name.c_str(), batchnorm->name.c_str());

        {
            int channels = batchnorm->channels;
            float eps = batchnorm->eps;

            // a = bias - slope * mean / sqrt(var + eps)
            // b = slope / sqrt(var + eps)
            // value = value * b + a

            std::vector<float> a(channels);
            std::vector<float> b(channels);
            for (int i = 0; i < channels; i++)
            {
                float sqrt_var = static_cast<float>(sqrt(batchnorm->var_data[i] + eps));
                a[i] = batchnorm->bias_data[i] - batchnorm->slope_data[i] * batchnorm->mean_data[i] / sqrt_var;
                b[i] = batchnorm->slope_data[i] / sqrt_var;
            }

            if (innerproduct->bias_term == 0)
            {
                // init bias as zero
                innerproduct->bias_term = 1;
                innerproduct->bias_data = ncnn::Mat(channels);
                innerproduct->bias_data.fill(0.f);
            }

            const int weight_per_outch = innerproduct->weight_data_size / channels;

            float* weight = innerproduct->weight_data;
            float* bias = innerproduct->bias_data;
            for (int i = 0; i < channels; i++)
            {
                float* conv_weight_outch = weight + weight_per_outch * i;
                for (int j = 0; j < weight_per_outch; j++)
                {
                    conv_weight_outch[j] *= b[i];
                }

                bias[i] = bias[i] * b[i] + a[i];
            }
        }

        int top_blob_index_final = batchnorm->tops[0];
        innerproduct->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        batchnorm->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_innerproduct_add()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "InnerProduct")
            continue;

        // InnerProduct - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "BinaryOp")
                continue;

            if (layers[j]->bottoms.size() != 2)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse InnerProduct - BinaryOp to InnerProduct
        ncnn::InnerProduct* innerproduct = (ncnn::InnerProduct*)layers[i];
        ncnn::BinaryOp* binaryop = (ncnn::BinaryOp*)layers[j];

        if (binaryop->op_type != 0 || binaryop->with_scalar)
            continue;

        // MemoryData - ..... - BinaryOp
        size_t k = 0;
        for (; k < j; k++)
        {
            if (layers[k]->type != "MemoryData")
                continue;

            if (layers[k]->tops[0] == binaryop->bottoms[1])
                break;
        }

        if (k == j)
            continue;

        ncnn::MemoryData* memorydata = (ncnn::MemoryData*)layers[k];

        int channels = innerproduct->num_output;

        bool broadcasting_type_ok = false;
        if (memorydata->w == channels && memorydata->h == 0 && memorydata->c == 0)
            broadcasting_type_ok = true;
        if (memorydata->w == 1 && memorydata->h == 1 && memorydata->c == channels)
            broadcasting_type_ok = true;

        if (!broadcasting_type_ok)
        {
            // not bias-like broadcasting type
            continue;
        }

        fprintf(stderr, "fuse_innerproduct_add %s %s\n", innerproduct->name.c_str(), binaryop->name.c_str());

        ncnn::Mat bias_data = memorydata->data.reshape(channels);
        {
            if (innerproduct->bias_term == 0)
            {
                // init bias
                innerproduct->bias_term = 1;
                innerproduct->bias_data = bias_data;
            }
            else
            {
                float* bias = innerproduct->bias_data;
                for (int i = 0; i < channels; i++)
                {
                    bias[i] = bias[i] + bias_data[i];
                }
            }
        }

        int top_blob_index_final = binaryop->tops[0];
        innerproduct->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        binaryop->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_innerproduct_dropout()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "InnerProduct")
            continue;

        // InnerProduct - Dropout
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "Dropout")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse InnerProduct - Dropout to InnerProduct
        ncnn::InnerProduct* innerproduct = (ncnn::InnerProduct*)layers[i];
        ncnn::Dropout* dropout = (ncnn::Dropout*)layers[j];

        fprintf(stderr, "fuse_innerproduct_dropout %s %s\n", innerproduct->name.c_str(), dropout->name.c_str());

        float scale = dropout->scale;
        if (scale != 1.f)
        {
            const int num_output = innerproduct->num_output;
            const int weight_per_outch = innerproduct->weight_data_size / num_output;

            float* weight = innerproduct->weight_data;
            for (int i = 0; i < num_output; i++)
            {
                float* conv_weight_outch = weight + weight_per_outch * i;
                for (int j = 0; j < weight_per_outch; j++)
                {
                    conv_weight_outch[j] *= scale;
                }
            }

            if (innerproduct->bias_term)
            {
                float* bias = innerproduct->bias_data;
                for (int i = 0; i < num_output; i++)
                {
                    bias[i] *= scale;
                }
            }
        }

        int top_blob_index_final = dropout->tops[0];
        innerproduct->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        dropout->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_convolution_activation()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Convolution")
            continue;

        // Convolution - Activation
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "ReLU" && layers[j]->type != "Clip" && layers[j]->type != "Sigmoid" && layers[j]->type != "Mish" && layers[j]->type != "HardSwish")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse Convolution - Activation to Convolution
        ncnn::Convolution* convolution = (ncnn::Convolution*)layers[i];
        ncnn::Layer* activation = layers[j];

        fprintf(stderr, "fuse_convolution_activation %s %s\n", convolution->name.c_str(), activation->name.c_str());

        if (activation->type == "ReLU")
        {
            ncnn::ReLU* relu = (ncnn::ReLU*)activation;

            if (relu->slope == 0.f)
            {
                convolution->activation_type = 1;
            }
            else
            {
                convolution->activation_type = 2;
                convolution->activation_params = ncnn::Mat(1);
                convolution->activation_params[0] = relu->slope;
            }
        }
        else if (activation->type == "Clip")
        {
            ncnn::Clip* clip = (ncnn::Clip*)activation;

            convolution->activation_type = 3;
            convolution->activation_params = ncnn::Mat(2);
            convolution->activation_params[0] = clip->min;
            convolution->activation_params[1] = clip->max;
        }
        else if (activation->type == "Sigmoid")
        {
            convolution->activation_type = 4;
        }
        else if (activation->type == "Mish")
        {
            convolution->activation_type = 5;
        }
        else if (activation->type == "HardSwish")
        {
            ncnn::HardSwish* hardswish = (ncnn::HardSwish*)activation;

            convolution->activation_type = 6;
            convolution->activation_params = ncnn::Mat(2);
            convolution->activation_params[0] = hardswish->alpha;
            convolution->activation_params[1] = hardswish->beta;
        }

        int top_blob_index_final = activation->tops[0];
        convolution->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        activation->type = "ncnnfused";
    }

    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Convolution1D")
            continue;

        // Convolution1D - Activation
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "ReLU" && layers[j]->type != "Clip" && layers[j]->type != "Sigmoid" && layers[j]->type != "Mish")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse Convolution1D - Activation to Convolution1D
        ncnn::Convolution1D* convolution = (ncnn::Convolution1D*)layers[i];
        ncnn::Layer* activation = layers[j];

        fprintf(stderr, "fuse_convolution1d_activation %s %s\n", convolution->name.c_str(), activation->name.c_str());

        if (activation->type == "ReLU")
        {
            ncnn::ReLU* relu = (ncnn::ReLU*)activation;

            if (relu->slope == 0.f)
            {
                convolution->activation_type = 1;
            }
            else
            {
                convolution->activation_type = 2;
                convolution->activation_params = ncnn::Mat(1);
                convolution->activation_params[0] = relu->slope;
            }
        }
        else if (activation->type == "Clip")
        {
            ncnn::Clip* clip = (ncnn::Clip*)activation;

            convolution->activation_type = 3;
            convolution->activation_params = ncnn::Mat(2);
            convolution->activation_params[0] = clip->min;
            convolution->activation_params[1] = clip->max;
        }
        else if (activation->type == "Sigmoid")
        {
            convolution->activation_type = 4;
        }
        else if (activation->type == "Mish")
        {
            convolution->activation_type = 5;
        }

        int top_blob_index_final = activation->tops[0];
        convolution->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        activation->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_convolutiondepthwise_activation()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "ConvolutionDepthWise")
            continue;

        // ConvolutionDepthWise - Activation
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "ReLU" && layers[j]->type != "Clip" && layers[j]->type != "Sigmoid" && layers[j]->type != "Mish" && layers[j]->type != "HardSwish")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse ConvolutionDepthWise - Activation to ConvolutionDepthWise
        ncnn::ConvolutionDepthWise* convolutiondepthwise = (ncnn::ConvolutionDepthWise*)layers[i];
        ncnn::Layer* activation = layers[j];

        fprintf(stderr, "fuse_convolutiondepthwise_activation %s %s\n", convolutiondepthwise->name.c_str(), activation->name.c_str());

        if (activation->type == "ReLU")
        {
            ncnn::ReLU* relu = (ncnn::ReLU*)activation;

            if (relu->slope == 0.f)
            {
                convolutiondepthwise->activation_type = 1;
            }
            else
            {
                convolutiondepthwise->activation_type = 2;
                convolutiondepthwise->activation_params = ncnn::Mat(1);
                convolutiondepthwise->activation_params[0] = relu->slope;
            }
        }
        else if (activation->type == "Clip")
        {
            ncnn::Clip* clip = (ncnn::Clip*)activation;

            convolutiondepthwise->activation_type = 3;
            convolutiondepthwise->activation_params = ncnn::Mat(2);
            convolutiondepthwise->activation_params[0] = clip->min;
            convolutiondepthwise->activation_params[1] = clip->max;
        }
        else if (activation->type == "Sigmoid")
        {
            convolutiondepthwise->activation_type = 4;
        }
        else if (activation->type == "Mish")
        {
            convolutiondepthwise->activation_type = 5;
        }
        else if (activation->type == "HardSwish")
        {
            ncnn::HardSwish* hardswish = (ncnn::HardSwish*)activation;

            convolutiondepthwise->activation_type = 6;
            convolutiondepthwise->activation_params = ncnn::Mat(2);
            convolutiondepthwise->activation_params[0] = hardswish->alpha;
            convolutiondepthwise->activation_params[1] = hardswish->beta;
        }

        int top_blob_index_final = activation->tops[0];
        convolutiondepthwise->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        activation->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_deconvolution_activation()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Deconvolution")
            continue;

        // Deconvolution - Activation
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "ReLU" && layers[j]->type != "Clip" && layers[j]->type != "Sigmoid")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse Deconvolution - Activation to Deconvolution
        ncnn::Deconvolution* deconvolution = (ncnn::Deconvolution*)layers[i];
        ncnn::Layer* activation = layers[j];

        fprintf(stderr, "fuse_deconvolution_activation %s %s\n", deconvolution->name.c_str(), activation->name.c_str());

        if (activation->type == "ReLU")
        {
            ncnn::ReLU* relu = (ncnn::ReLU*)activation;

            if (relu->slope == 0.f)
            {
                deconvolution->activation_type = 1;
            }
            else
            {
                deconvolution->activation_type = 2;
                deconvolution->activation_params = ncnn::Mat(1);
                deconvolution->activation_params[0] = relu->slope;
            }
        }
        else if (activation->type == "Clip")
        {
            ncnn::Clip* clip = (ncnn::Clip*)activation;

            deconvolution->activation_type = 3;
            deconvolution->activation_params = ncnn::Mat(2);
            deconvolution->activation_params[0] = clip->min;
            deconvolution->activation_params[1] = clip->max;
        }
        else if (activation->type == "Sigmoid")
        {
            deconvolution->activation_type = 4;
        }

        int top_blob_index_final = activation->tops[0];
        deconvolution->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        activation->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_deconvolutiondepthwise_activation()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "DeconvolutionDepthWise")
            continue;

        // DeconvolutionDepthWise - Activation
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "ReLU" && layers[j]->type != "Clip" && layers[j]->type != "Sigmoid")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse DeconvolutionDepthWise - Activation to DeconvolutionDepthWise
        ncnn::DeconvolutionDepthWise* deconvolutiondepthwise = (ncnn::DeconvolutionDepthWise*)layers[i];
        ncnn::Layer* activation = layers[j];

        fprintf(stderr, "fuse_deconvolutiondepthwise_activation %s %s\n", deconvolutiondepthwise->name.c_str(), activation->name.c_str());

        if (activation->type == "ReLU")
        {
            ncnn::ReLU* relu = (ncnn::ReLU*)activation;

            if (relu->slope == 0.f)
            {
                deconvolutiondepthwise->activation_type = 1;
            }
            else
            {
                deconvolutiondepthwise->activation_type = 2;
                deconvolutiondepthwise->activation_params = ncnn::Mat(1);
                deconvolutiondepthwise->activation_params[0] = relu->slope;
            }
        }
        else if (activation->type == "Clip")
        {
            ncnn::Clip* clip = (ncnn::Clip*)activation;

            deconvolutiondepthwise->activation_type = 3;
            deconvolutiondepthwise->activation_params = ncnn::Mat(2);
            deconvolutiondepthwise->activation_params[0] = clip->min;
            deconvolutiondepthwise->activation_params[1] = clip->max;
        }
        else if (activation->type == "Sigmoid")
        {
            deconvolutiondepthwise->activation_type = 4;
        }

        int top_blob_index_final = activation->tops[0];
        deconvolutiondepthwise->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        activation->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_innerproduct_activation()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "InnerProduct")
            continue;

        // InnerProduct - Activation
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "ReLU" && layers[j]->type != "Clip" && layers[j]->type != "Sigmoid" && layers[j]->type != "Mish" && layers[j]->type != "HardSwish")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse InnerProduct - Activation to InnerProduct
        ncnn::InnerProduct* innerproduct = (ncnn::InnerProduct*)layers[i];
        ncnn::Layer* activation = layers[j];

        fprintf(stderr, "fuse_innerproduct_activation %s %s\n", innerproduct->name.c_str(), activation->name.c_str());

        if (activation->type == "ReLU")
        {
            ncnn::ReLU* relu = (ncnn::ReLU*)activation;

            if (relu->slope == 0.f)
            {
                innerproduct->activation_type = 1;
            }
            else
            {
                innerproduct->activation_type = 2;
                innerproduct->activation_params = ncnn::Mat(1);
                innerproduct->activation_params[0] = relu->slope;
            }
        }
        else if (activation->type == "Clip")
        {
            ncnn::Clip* clip = (ncnn::Clip*)activation;

            innerproduct->activation_type = 3;
            innerproduct->activation_params = ncnn::Mat(2);
            innerproduct->activation_params[0] = clip->min;
            innerproduct->activation_params[1] = clip->max;
        }
        else if (activation->type == "Sigmoid")
        {
            innerproduct->activation_type = 4;
        }
        else if (activation->type == "Mish")
        {
            innerproduct->activation_type = 5;
        }
        else if (activation->type == "HardSwish")
        {
            ncnn::HardSwish* hardswish = (ncnn::HardSwish*)activation;

            innerproduct->activation_type = 6;
            innerproduct->activation_params = ncnn::Mat(2);
            innerproduct->activation_params[0] = hardswish->alpha;
            innerproduct->activation_params[1] = hardswish->beta;
        }

        int top_blob_index_final = activation->tops[0];
        innerproduct->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        activation->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::fuse_memorydata_binaryop()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "MemoryData")
            continue;

        // MemoryData - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "BinaryOp")
                continue;

            if (layers[j]->bottoms.size() != 2)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index || layers[j]->bottoms[1] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse MemoryData - BinaryOp to BinaryOp
        ncnn::MemoryData* memorydata = (ncnn::MemoryData*)layers[i];
        ncnn::BinaryOp* binaryop = (ncnn::BinaryOp*)layers[j];

        if (memorydata->w != 1 || memorydata->h != 0 || memorydata->c != 0)
        {
            // not a scalar
            continue;
        }

        int memorydata_index = 1;

        if (binaryop->bottoms[0] == top_blob_index)
        {
            int op_type = binaryop->op_type;

            if (op_type == ncnn::BinaryOp::Operation_ADD
                    || op_type == ncnn::BinaryOp::Operation_MUL
                    || op_type == ncnn::BinaryOp::Operation_MAX
                    || op_type == ncnn::BinaryOp::Operation_MIN)
            {
                memorydata_index = 0;
            }
            else if (op_type == ncnn::BinaryOp::Operation_SUB)
            {
                binaryop->op_type = ncnn::BinaryOp::Operation_RSUB;
                memorydata_index = 0;
            }
            else if (op_type == ncnn::BinaryOp::Operation_DIV)
            {
                binaryop->op_type = ncnn::BinaryOp::Operation_RDIV;
                memorydata_index = 0;
            }
            else
            {
                // non interchangeable binaryop
                continue;
            }
        }

        float scalar = memorydata->data[0];

        binaryop->with_scalar = 1;
        binaryop->b = scalar;

        fprintf(stderr, "fuse_memorydata_binaryop %s %s\n", memorydata->name.c_str(), binaryop->name.c_str());

        binaryop->bottoms.erase(binaryop->bottoms.begin() + memorydata_index);
        memorydata->type = "ncnnfused";
    }

    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "MemoryData")
            continue;

        // MemoryData - Split - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        size_t j0 = i + 1;
        for (; j0 < layer_count; j0++)
        {
            if (layers[j0]->type != "Split")
                continue;

            if (layers[j0]->bottoms.size() != 1)
                continue;

            if (layers[j0]->bottoms[0] == top_blob_index)
                break;
        }

        if (j0 == layer_count)
            continue;

        int split_top_blob_index = -1;

        size_t j1 = j0 + 1;
        for (; j1 < layer_count; j1++)
        {
            if (layers[j1]->type != "BinaryOp")
                continue;

            if (layers[j1]->bottoms.size() != 2)
                continue;

            for (int k = 0; k < (int)layers[j0]->tops.size(); k++)
            {
                if (layers[j1]->bottoms[0] == layers[j0]->tops[k] || layers[j1]->bottoms[1] == layers[j0]->tops[k])
                {
                    split_top_blob_index = k;
                    break;
                }
            }

            if (split_top_blob_index != -1)
                break;
        }

        if (j1 == layer_count)
            continue;

        // fuse MemoryData - Split - BinaryOp to BinaryOp
        ncnn::MemoryData* memorydata = (ncnn::MemoryData*)layers[i];
        ncnn::Split* split = (ncnn::Split*)layers[j0];
        ncnn::BinaryOp* binaryop = (ncnn::BinaryOp*)layers[j1];

        if (memorydata->w != 1 || memorydata->h != 0 || memorydata->c != 0)
        {
            // not a scalar
            continue;
        }

        int memorydata_index = 1;

        if (binaryop->bottoms[0] == split->tops[split_top_blob_index])
        {
            int op_type = binaryop->op_type;

            if (op_type == ncnn::BinaryOp::Operation_ADD
                    || op_type == ncnn::BinaryOp::Operation_MUL
                    || op_type == ncnn::BinaryOp::Operation_MAX
                    || op_type == ncnn::BinaryOp::Operation_MIN)
            {
                memorydata_index = 0;
            }
            else if (op_type == ncnn::BinaryOp::Operation_SUB)
            {
                binaryop->op_type = ncnn::BinaryOp::Operation_RSUB;
                memorydata_index = 0;
            }
            else if (op_type == ncnn::BinaryOp::Operation_DIV)
            {
                binaryop->op_type = ncnn::BinaryOp::Operation_RDIV;
                memorydata_index = 0;
            }
            else
            {
                // non interchangeable binaryop
                continue;
            }
        }

        float scalar = memorydata->data[0];

        binaryop->with_scalar = 1;
        binaryop->b = scalar;

        fprintf(stderr, "fuse_memorydata_binaryop %s %s\n", memorydata->name.c_str(), binaryop->name.c_str());

        binaryop->bottoms.erase(binaryop->bottoms.begin() + memorydata_index);
        split->tops.erase(split->tops.begin() + split_top_blob_index);
        if (split->tops.empty())
        {
            split->type = "ncnnfused";
            memorydata->type = "ncnnfused";
        }

        i--;
    }

    return 0;
}

int NetOptimize::fuse_binaryop_eltwise()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "BinaryOp")
            continue;

        if (layers[i]->bottoms.size() != 2)
            continue;

        ncnn::BinaryOp* binaryop = (ncnn::BinaryOp*)layers[i];

        if (binaryop->op_type != ncnn::BinaryOp::Operation_ADD)
            continue;

        if (binaryop->with_scalar)
            continue;

        // BinaryOp - BinaryOp - BinaryOp
        int bottom_blob_index_0 = binaryop->bottoms[0];
        int bottom_blob_index_1 = binaryop->bottoms[1];

        size_t j0 = 0;
        for (; j0 < i; j0++)
        {
            if (layers[j0]->type != "BinaryOp")
                continue;

            if (layers[j0]->bottoms.size() != 1)
                continue;

            if (((ncnn::BinaryOp*)layers[j0])->op_type != ncnn::BinaryOp::Operation_MUL)
                continue;

            if (layers[j0]->tops[0] == bottom_blob_index_0)
                break;
        }

        size_t j1 = 0;
        for (; j1 < i; j1++)
        {
            if (layers[j1]->type != "BinaryOp")
                continue;

            if (layers[j1]->bottoms.size() != 1)
                continue;

            if (((ncnn::BinaryOp*)layers[j1])->op_type != ncnn::BinaryOp::Operation_MUL)
                continue;

            if (layers[j1]->tops[0] == bottom_blob_index_1)
                break;
        }

        if (j0 == i && j1 == i)
            continue;

        ncnn::BinaryOp* binaryop0 = (ncnn::BinaryOp*)layers[j0];
        ncnn::BinaryOp* binaryop1 = (ncnn::BinaryOp*)layers[j1];

        fprintf(stderr, "fuse_binaryop_eltwise %s %s %s\n", binaryop0->name.c_str(), binaryop1->name.c_str(), binaryop->name.c_str());

        ncnn::Eltwise* eltwise = (ncnn::Eltwise*)ncnn::create_layer("Eltwise");

        eltwise->type = "Eltwise";
        eltwise->name = binaryop->name;
        eltwise->bottoms = binaryop->bottoms;
        eltwise->tops = binaryop->tops;

        ncnn::ParamDict pd;
        eltwise->load_param(pd);

        eltwise->op_type = ncnn::Eltwise::Operation_SUM;

        eltwise->coeffs = ncnn::Mat(2);

        if (j0 != i && j1 != i)
        {
            // fuse BinaryOp - BinaryOp - BinaryOp to Eltwise
            eltwise->coeffs[0] = binaryop0->b;
            eltwise->coeffs[1] = binaryop1->b;

            eltwise->bottoms[0] = binaryop0->bottoms[0];
            eltwise->bottoms[1] = binaryop1->bottoms[0];

            binaryop0->type = "ncnnfused";
            binaryop1->type = "ncnnfused";
        }
        if (j0 != i && j1 == i)
        {
            // fuse BinaryOp - X - BinaryOp to Eltwise
            eltwise->coeffs[0] = binaryop0->b;
            eltwise->coeffs[1] = 1.f;

            eltwise->bottoms[0] = binaryop0->bottoms[0];

            binaryop0->type = "ncnnfused";
        }
        if (j0 == i && j1 != i)
        {
            // fuse X - BinaryOp - BinaryOp to Eltwise
            eltwise->coeffs[0] = 1.f;
            eltwise->coeffs[1] = binaryop1->b;

            eltwise->bottoms[1] = binaryop1->bottoms[0];

            binaryop1->type = "ncnnfused";
        }

        layers[i] = eltwise;
        delete binaryop;
    }

    return 0;
}

int NetOptimize::eliminate_dropout()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Dropout")
            continue;

        ncnn::Dropout* dropout = (ncnn::Dropout*)layers[i];
        if (dropout->scale != 1.f)
            continue;

        // Any - Dropout
        int bottom_blob_index = layers[i]->bottoms[0];

        int j = i - 1;
        for (; j >= 0; j--)
        {
            if (layers[j]->type == "ncnnfused")
                continue;

            if (layers[j]->tops.size() != 1)
                continue;

            if (layers[j]->tops[0] == bottom_blob_index)
                break;
        }

        if (j == -1)
            continue;

        ncnn::Layer* any = layers[j];

        fprintf(stderr, "eliminate_dropout %s %s\n", any->name.c_str(), dropout->name.c_str());

        int top_blob_index_final = dropout->tops[0];
        any->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = j;
        dropout->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::eliminate_pooling1x1()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Pooling")
            continue;

        ncnn::Pooling* pooling = (ncnn::Pooling*)layers[i];
        if (pooling->pad_left != 0 || pooling->pad_right != 0 || pooling->pad_top != 0 || pooling->pad_bottom != 0)
            continue;

        if (pooling->kernel_w != 1 || pooling->kernel_h != 1 || pooling->stride_w != 1 || pooling->stride_h != 1)
            continue;

        if (pooling->global_pooling != 0)
            continue;

        // Any - Pooling
        int bottom_blob_index = layers[i]->bottoms[0];

        int top_i = -1;
        int j = i - 1;
        for (; j >= 0; j--)
        {
            if (layers[j]->type == "ncnnfused")
                continue;

            for (size_t k = 0; k < layers[j]->tops.size(); k++)
            {
                if (layers[j]->tops[k] == bottom_blob_index)
                {
                    top_i = k;
                    break;
                }
            }

            if (top_i != -1)
                break;
        }

        if (j == -1)
            continue;

        ncnn::Layer* any = layers[j];

        fprintf(stderr, "eliminate_pooling1x1 %s %s\n", any->name.c_str(), pooling->name.c_str());

        int top_blob_index_final = pooling->tops[0];
        any->tops[top_i] = top_blob_index_final;
        blobs[top_blob_index_final].producer = j;
        pooling->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::eliminate_noop()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Noop")
            continue;

        ncnn::Layer* noop = layers[i];

        if (noop->bottoms.empty())
        {
            // Noop
            fprintf(stderr, "eliminate_noop %s\n", noop->name.c_str());

            size_t top_blob_count = noop->tops.size();
            for (size_t j = 0; j < top_blob_count; j++)
            {
                int top_blob_index_final = noop->tops[j];
                blobs[top_blob_index_final].producer = -1;
            }
            noop->type = "ncnnfused";

            continue;
        }

        // Any - Noop
        int bottom_blob_index = noop->bottoms[0];

        int j = i - 1;
        int any_k = -1;
        for (; j >= 0; j--)
        {
            if (layers[j]->type == "ncnnfused")
                continue;

            bool link_noop = false;
            size_t top_blob_count = layers[j]->tops.size();
            for (size_t k = 0; k < top_blob_count; k++)
            {
                if (layers[j]->tops[k] == bottom_blob_index)
                {
                    link_noop = true;
                    any_k = k;
                    break;
                }
            }

            if (link_noop)
                break;
        }

        if (j == -1 || any_k == -1)
            continue;

        ncnn::Layer* any = layers[j];

        fprintf(stderr, "eliminate_noop %s %s\n", any->name.c_str(), noop->name.c_str());

        int top_blob_index_final = noop->tops[0];
        any->tops[any_k] = top_blob_index_final;
        blobs[top_blob_index_final].producer = j;

        noop->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::eliminate_split()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Split")
            continue;

        ncnn::Layer* split = layers[i];

        int real_split_output_count = 0;
        int real_split_top_blob_index = -1;
        size_t top_blob_count = split->tops.size();
        for (size_t j = 0; j < top_blob_count; j++)
        {
            int top_blob_index_final = split->tops[j];
            if (blobs[top_blob_index_final].consumer != -1)
            {
                real_split_output_count += 1;
                real_split_top_blob_index = j;
            }
        }

        if (real_split_output_count > 1)
            continue;

        // Any - Pooling
        int bottom_blob_index = split->bottoms[0];

        int top_i = -1;
        int j = i - 1;
        for (; j >= 0; j--)
        {
            if (layers[j]->type == "ncnnfused")
                continue;

            for (size_t k = 0; k < layers[j]->tops.size(); k++)
            {
                if (layers[j]->tops[k] == bottom_blob_index)
                {
                    top_i = k;
                    break;
                }
            }

            if (top_i != -1)
                break;
        }

        if (j == -1)
            continue;

        ncnn::Layer* any = layers[j];

        fprintf(stderr, "eliminate_split %s %s\n", any->name.c_str(), split->name.c_str());

        int top_blob_index_final = split->tops[real_split_top_blob_index];
        any->tops[top_i] = top_blob_index_final;
        blobs[top_blob_index_final].producer = j;
        split->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::eliminate_orphaned_memorydata()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "MemoryData")
            continue;

        // MemoryData - X
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type == "ncnnfused")
                continue;

            bool orphaned = true;
            for (size_t k = 0; k < layers[j]->bottoms.size(); k++)
            {
                if (layers[j]->bottoms[k] == top_blob_index)
                {
                    orphaned = false;
                    break;
                }
            }

            if (!orphaned)
                break;
        }

        if (j < layer_count)
            continue;

        // assert orphaned == true
        fprintf(stderr, "eliminate_orphaned_memorydata %s\n", layers[i]->name.c_str());

        layers[i]->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::eliminate_reshape_after_global_pooling()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Pooling")
            continue;

        ncnn::Pooling* pooling = (ncnn::Pooling*)layers[i];
        if (pooling->global_pooling == 0)
            continue;

        // Pooling - Reshape
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "Reshape")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        ncnn::Reshape* reshape = (ncnn::Reshape*)layers[j];
        if (reshape->h != -233 || reshape->c != -233 || reshape->permute != 0)
            continue;

        fprintf(stderr, "eliminate_reshape_after_global_pooling %s %s\n", pooling->name.c_str(), reshape->name.c_str());

        int top_blob_index_final = reshape->tops[0];
        pooling->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        reshape->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::eliminate_flatten_after_global_pooling()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Pooling")
            continue;

        ncnn::Pooling* pooling = (ncnn::Pooling*)layers[i];
        if (pooling->global_pooling == 0)
            continue;

        // Pooling - Flatten
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "Flatten")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        ncnn::Flatten* flatten = (ncnn::Flatten*)layers[j];

        fprintf(stderr, "eliminate_flatten_after_global_pooling %s %s\n", pooling->name.c_str(), flatten->name.c_str());

        int top_blob_index_final = flatten->tops[0];
        pooling->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        flatten->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::eliminate_flatten_after_innerproduct()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "InnerProduct")
            continue;

        // InnerProduct - Flatten
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "Flatten")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        ncnn::InnerProduct* innerproduct = (ncnn::InnerProduct*)layers[i];
        ncnn::Flatten* flatten = (ncnn::Flatten*)layers[j];

        fprintf(stderr, "eliminate_flatten_after_innerproduct %s %s\n", innerproduct->name.c_str(), flatten->name.c_str());

        int top_blob_index_final = flatten->tops[0];
        innerproduct->tops[0] = top_blob_index_final;
        blobs[top_blob_index_final].producer = i;
        flatten->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::eliminate_reshape_before_binaryop()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Reshape")
            continue;

        ncnn::Reshape* reshape = (ncnn::Reshape*)layers[i];
        if (reshape->w != 1 || reshape->h != 1 || reshape->permute != 0)
            continue;

        // Reshape - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "BinaryOp")
                continue;

            if (layers[j]->bottoms.size() != 2)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index || layers[j]->bottoms[1] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        ncnn::BinaryOp* binaryop = (ncnn::BinaryOp*)layers[j];

        fprintf(stderr, "eliminate_reshape_before_binaryop %s %s\n", reshape->name.c_str(), binaryop->name.c_str());

        int bottom_blob_index_final = reshape->bottoms[0];
        if (layers[j]->bottoms[0] == top_blob_index)
            binaryop->bottoms[0] = bottom_blob_index_final;
        if (layers[j]->bottoms[1] == top_blob_index)
            binaryop->bottoms[1] = bottom_blob_index_final;
        blobs[bottom_blob_index_final].consumer = j;
        reshape->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::replace_reduction_with_global_pooling()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Reduction")
            continue;

        ncnn::Reduction* reduction1 = (ncnn::Reduction*)layers[i];
        if (reduction1->operation != 3 || reduction1->reduce_all != 0 || reduction1->coeff != 1.f)
            continue;

        if (reduction1->axes.w != 1)
            continue;

        const int* axes_ptr = reduction1->axes;
        if (axes_ptr[0] != 2 && axes_ptr[0] != 3)
            continue;

        // Reduction(2/3) - Reduction(2)
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "Reduction")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        ncnn::Reduction* reduction2 = (ncnn::Reduction*)layers[j];
        if (reduction2->operation != 3 || reduction2->reduce_all != 0 || reduction2->coeff != 1.f)
            continue;

        if (reduction2->axes.w != 1)
            continue;

        const int* axes2_ptr = reduction2->axes;
        if (axes2_ptr[0] != 2)
            continue;

        fprintf(stderr, "replace_reduction_with_global_pooling %s %s\n", reduction1->name.c_str(), reduction2->name.c_str());

        ncnn::Pooling* pooling = (ncnn::Pooling*)ncnn::create_layer("Pooling");

        pooling->type = "Pooling";
        pooling->name = reduction2->name;
        pooling->bottoms = reduction2->bottoms;
        pooling->tops = reduction2->tops;

        ncnn::ParamDict pd;
        pooling->load_param(pd);

        pooling->pooling_type = 1;
        pooling->global_pooling = 1;

        layers[j] = pooling;
        delete reduction2;

        int bottom_blob_index_final = reduction1->bottoms[0];
        pooling->bottoms[0] = bottom_blob_index_final;
        blobs[bottom_blob_index_final].consumer = j;
        reduction1->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::replace_prelu_with_leaky_relu()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "PReLU")
            continue;

        ncnn::PReLU* prelu = (ncnn::PReLU*)layers[i];
        if (prelu->num_slope != 1)
            continue;

        fprintf(stderr, "replace_prelu_with_leaky_relu %s\n", prelu->name.c_str());

        ncnn::ReLU* relu = (ncnn::ReLU*)ncnn::create_layer("ReLU");

        relu->type = "ReLU";
        relu->name = prelu->name;
        relu->bottoms = prelu->bottoms;
        relu->tops = prelu->tops;

        ncnn::ParamDict pd;
        relu->load_param(pd);

        relu->slope = prelu->slope_data[0];

        layers[i] = relu;
        delete prelu;
    }

    return 0;
}

int NetOptimize::replace_convolution_with_innerproduct_after_global_pooling()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Pooling")
            continue;

        ncnn::Pooling* pooling = (ncnn::Pooling*)layers[i];
        if (pooling->global_pooling == 0)
            continue;

        // Pooling - Convolution
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "Convolution")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        ncnn::Convolution* convolution = (ncnn::Convolution*)layers[j];

        fprintf(stderr, "replace_convolution_with_innerproduct_after_global_pooling %s %s\n", pooling->name.c_str(), convolution->name.c_str());

        ncnn::InnerProduct* innerproduct = (ncnn::InnerProduct*)ncnn::create_layer("InnerProduct");

        innerproduct->type = "InnerProduct";
        innerproduct->name = convolution->name;
        innerproduct->bottoms = convolution->bottoms;
        innerproduct->tops = convolution->tops;

        ncnn::ParamDict pd;
        innerproduct->load_param(pd);

        innerproduct->num_output = convolution->num_output;
        innerproduct->bias_term = convolution->bias_term;
        innerproduct->weight_data_size = convolution->weight_data_size;
        innerproduct->int8_scale_term = convolution->int8_scale_term;

        innerproduct->weight_data = convolution->weight_data;
        innerproduct->bias_data = convolution->bias_data;
#if NCNN_INT8
        innerproduct->weight_data_int8_scales = convolution->weight_data_int8_scales;
        innerproduct->bottom_blob_int8_scales = convolution->bottom_blob_int8_scales;
#endif

        innerproduct->activation_type = convolution->activation_type;
        innerproduct->activation_params = convolution->activation_params;

        layers[j] = innerproduct;
        delete convolution;
    }

    return 0;
}

int NetOptimize::replace_convolution_with_innerproduct_after_innerproduct()
{
    const size_t layer_count = layers.size();
    for (;;)
    {
        bool replaced = false;

        for (size_t i = 0; i < layer_count; i++)
        {
            if (layers[i]->type != "InnerProduct")
                continue;

            // InnerProduct - Convolution
            int top_blob_index = layers[i]->tops[0];

            size_t j = i + 1;
            for (; j < layer_count; j++)
            {
                if (layers[j]->type != "Convolution")
                    continue;

                if (layers[j]->bottoms.size() != 1)
                    continue;

                if (layers[j]->bottoms[0] == top_blob_index)
                    break;
            }

            if (j == layer_count)
                continue;

            ncnn::InnerProduct* innerproduct = (ncnn::InnerProduct*)layers[i];
            ncnn::Convolution* convolution = (ncnn::Convolution*)layers[j];

            fprintf(stderr, "replace_convolution_with_innerproduct_after_innerproduct %s %s\n", innerproduct->name.c_str(), convolution->name.c_str());

            ncnn::InnerProduct* innerproduct2 = (ncnn::InnerProduct*)ncnn::create_layer("InnerProduct");

            innerproduct2->type = "InnerProduct";
            innerproduct2->name = convolution->name;
            innerproduct2->bottoms = convolution->bottoms;
            innerproduct2->tops = convolution->tops;

            ncnn::ParamDict pd;
            innerproduct2->load_param(pd);

            innerproduct2->num_output = convolution->num_output;
            innerproduct2->bias_term = convolution->bias_term;
            innerproduct2->weight_data_size = convolution->weight_data_size;
            innerproduct->int8_scale_term = convolution->int8_scale_term;

            innerproduct2->weight_data = convolution->weight_data;
            innerproduct2->bias_data = convolution->bias_data;
#if NCNN_INT8
            innerproduct->weight_data_int8_scales = convolution->weight_data_int8_scales;
            innerproduct->bottom_blob_int8_scales = convolution->bottom_blob_int8_scales;
#endif

            innerproduct2->activation_type = convolution->activation_type;
            innerproduct2->activation_params = convolution->activation_params;

            layers[j] = innerproduct2;
            delete convolution;

            replaced = true;
        }

        if (!replaced)
            break;
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 6)
    {
        fprintf(stderr, "usage: %s [inparam] [inbin] [outparam] [outbin] [flag] [cutstart] [cutend]\n", argv[0]);
        return -1;
    }

    const char* inparam = argv[1];
    const char* inbin = argv[2];
    const char* outparam = argv[3];
    const char* outbin = argv[4];
    int flag = atoi(argv[5]);
    const char* cutstartname = nullptr;
    const char* cutendname = nullptr;

    if (argc > 6)
    {
        cutstartname = argv[6];
    }

    if (argc > 7)
    {
        cutendname = argv[7];
    }

    NetOptimize optimizer;

    if (flag == 65536 || flag == 1)
    {
        optimizer.storage_type = 1;
    }
    else
    {
        optimizer.storage_type = 0;
    }

    optimizer.load_param(inparam);

    if (strcmp(inbin, "null") == 0)
    {
        DataReaderFromEmpty dr;
        optimizer.load_model(dr);
        optimizer.gen_random_weight = true;
    }
    else
        optimizer.load_model(inbin);

    if (optimizer.set_cutparam(cutstartname, cutendname) < 0)
    {
        return -1;
    }

    optimizer.fuse_batchnorm_scale();
    optimizer.fuse_convolution_batchnorm();
    optimizer.fuse_convolution_mul();
    optimizer.fuse_convolution_add();
    optimizer.fuse_convolutiondepthwise_batchnorm();
    optimizer.fuse_convolutiondepthwise_mul();
    optimizer.fuse_convolutiondepthwise_add();
    optimizer.fuse_deconvolution_batchnorm();
    optimizer.fuse_deconvolution_mul();
    optimizer.fuse_deconvolution_add();
    optimizer.fuse_deconvolutiondepthwise_batchnorm();
    optimizer.fuse_innerproduct_batchnorm();
    optimizer.fuse_innerproduct_add();
    optimizer.fuse_innerproduct_dropout();

    optimizer.replace_reduction_with_global_pooling();
    optimizer.replace_prelu_with_leaky_relu();

    optimizer.fuse_convolution_activation();
    optimizer.fuse_convolutiondepthwise_activation();
    optimizer.fuse_deconvolution_activation();
    optimizer.fuse_deconvolutiondepthwise_activation();
    optimizer.fuse_innerproduct_activation();
    optimizer.fuse_memorydata_binaryop();
    optimizer.fuse_binaryop_eltwise();

    optimizer.eliminate_dropout();
    optimizer.eliminate_pooling1x1();
    optimizer.eliminate_noop();
    optimizer.eliminate_split();
    optimizer.eliminate_flatten_after_global_pooling();
    optimizer.eliminate_reshape_after_global_pooling();
    optimizer.eliminate_reshape_before_binaryop();

    optimizer.replace_convolution_with_innerproduct_after_global_pooling();
    optimizer.replace_convolution_with_innerproduct_after_innerproduct();

    optimizer.eliminate_flatten_after_innerproduct();
    optimizer.eliminate_orphaned_memorydata();

    optimizer.shape_inference();

    optimizer.estimate_memory_footprint();

    optimizer.save(outparam, outbin);

    return 0;
}
