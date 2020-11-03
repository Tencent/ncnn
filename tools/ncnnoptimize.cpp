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
#include <set>
#include <vector>

// ncnn public header
#include "datareader.h"
#include "layer.h"
#include "net.h"

// ncnn private header
#include "layer/batchnorm.h"
#include "layer/bias.h"
#include "layer/binaryop.h"
#include "layer/clip.h"
#include "layer/concat.h"
#include "layer/convolution.h"
#include "layer/convolutiondepthwise.h"
#include "layer/crop.h"
#include "layer/deconvolution.h"
#include "layer/deconvolutiondepthwise.h"
#include "layer/detectionoutput.h"
#include "layer/dropout.h"
#include "layer/eltwise.h"
#include "layer/elu.h"
#include "layer/exp.h"
#include "layer/expanddims.h"
#include "layer/flatten.h"
#include "layer/gemm.h"
#include "layer/groupnorm.h"
#include "layer/hardsigmoid.h"
#include "layer/hardswish.h"
#include "layer/innerproduct.h"
#include "layer/input.h"
#include "layer/instancenorm.h"
#include "layer/interp.h"
#include "layer/log.h"
#include "layer/lrn.h"
#include "layer/lstm.h"
#include "layer/memorydata.h"
#include "layer/mvn.h"
#include "layer/normalize.h"
#include "layer/padding.h"
#include "layer/permute.h"
#include "layer/pixelshuffle.h"
#include "layer/pooling.h"
#include "layer/power.h"
#include "layer/prelu.h"
#include "layer/priorbox.h"
#include "layer/proposal.h"
#include "layer/psroipooling.h"
#include "layer/quantize.h"
#include "layer/reduction.h"
#include "layer/relu.h"
#include "layer/reorg.h"
#include "layer/requantize.h"
#include "layer/reshape.h"
#include "layer/roialign.h"
#include "layer/roipooling.h"
#include "layer/scale.h"
#include "layer/shufflechannel.h"
#include "layer/slice.h"
#include "layer/softmax.h"
#include "layer/split.h"
#include "layer/squeeze.h"
#include "layer/threshold.h"
#include "layer/unaryop.h"
#include "layer/yolodetectionoutput.h"
#include "layer/yolov3detectionoutput.h"

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* format, void* p) const
    {
        return 0;
    }
    virtual size_t read(void* /*buf*/, size_t size) const
    {
        return size;
    }
};

class NetOptimize : public ncnn::Net
{
public:
    // 0=fp32 1=fp16
    int storage_type;

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
    int eliminate_orphaned_memorydata();
    int eliminate_flatten_after_global_pooling();
    int eliminate_reshape_after_global_pooling();
    int eliminate_flatten_after_innerproduct();
    int eliminate_reshape_before_binaryop();

    int replace_convolution_with_innerproduct_after_global_pooling();
    int replace_convolution_with_innerproduct_after_innerproduct();

    int shape_inference();

public:
    int fprintf_param_int_array(int id, const ncnn::Mat& m, FILE* pp);
    int fprintf_param_float_array(int id, const ncnn::Mat& m, FILE* pp);

    int fwrite_weight_tag_data(int tag, const ncnn::Mat& data, FILE* bp);
    int fwrite_weight_data(const ncnn::Mat& data, FILE* bp);

    int save(const char* parampath, const char* binpath);
};

int NetOptimize::fuse_batchnorm_scale()
{
    const size_t layer_count = layers.size();
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "BatchNorm")
            continue;

        // BatchNorm - Scale
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Convolution")
            continue;

        // Convolution - BatchNorm
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Convolution")
            continue;

        // Convolution - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
        int k = 0;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Convolution")
            continue;

        // Convolution - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
        int k = 0;
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

        fprintf(stderr, "fuse_convolution_add %s %s\n", convolution->name.c_str(), binaryop->name.c_str());

        {
            if (convolution->bias_term == 0)
            {
                // init bias
                convolution->bias_term = 1;
                convolution->bias_data = memorydata->data;
            }
            else
            {
                float* bias = convolution->bias_data;
                for (int i = 0; i < channels; i++)
                {
                    bias[i] = bias[i] + memorydata->data[i];
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "ConvolutionDepthWise")
            continue;

        // ConvolutionDepthWise - BatchNorm
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "ConvolutionDepthWise")
            continue;

        // ConvolutionDepthWise - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
        int k = 0;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "ConvolutionDepthWise")
            continue;

        // ConvolutionDepthWise - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
        int k = 0;
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

        fprintf(stderr, "fuse_convolutiondepthwise_add %s %s\n", convolutiondepthwise->name.c_str(), binaryop->name.c_str());

        {
            if (convolutiondepthwise->bias_term == 0)
            {
                // init bias
                convolutiondepthwise->bias_term = 1;
                convolutiondepthwise->bias_data = memorydata->data;
            }
            else
            {
                float* bias = convolutiondepthwise->bias_data;
                for (int i = 0; i < channels; i++)
                {
                    bias[i] = bias[i] + memorydata->data[i];
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Deconvolution")
            continue;

        // Deconvolution - BatchNorm
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Deconvolution")
            continue;

        // Deconvolution - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
        int k = 0;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Deconvolution")
            continue;

        // Deconvolution - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
        int k = 0;
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

        fprintf(stderr, "fuse_deconvolution_add %s %s\n", deconvolution->name.c_str(), binaryop->name.c_str());

        {
            if (deconvolution->bias_term == 0)
            {
                // init bias
                deconvolution->bias_term = 1;
                deconvolution->bias_data = memorydata->data;
            }
            else
            {
                float* bias = deconvolution->bias_data;
                for (int i = 0; i < channels; i++)
                {
                    bias[i] = bias[i] + memorydata->data[i];
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "DeconvolutionDepthWise")
            continue;

        // DeconvolutionDepthWise - BatchNorm
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "InnerProduct")
            continue;

        // InnerProduct - BatchNorm
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "InnerProduct")
            continue;

        // InnerProduct - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
        int k = 0;
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

        if (memorydata->w != channels || memorydata->h != 0 || memorydata->c != 0)
        {
            // not bias-like broadcasting type
            continue;
        }

        fprintf(stderr, "fuse_innerproduct_add %s %s\n", innerproduct->name.c_str(), binaryop->name.c_str());

        {
            if (innerproduct->bias_term == 0)
            {
                // init bias
                innerproduct->bias_term = 1;
                innerproduct->bias_data = memorydata->data;
            }
            else
            {
                float* bias = innerproduct->bias_data;
                for (int i = 0; i < channels; i++)
                {
                    bias[i] = bias[i] + memorydata->data[i];
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "InnerProduct")
            continue;

        // InnerProduct - Dropout
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Convolution")
            continue;

        // Convolution - Activation
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "ConvolutionDepthWise")
            continue;

        // ConvolutionDepthWise - Activation
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Deconvolution")
            continue;

        // Deconvolution - Activation
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "DeconvolutionDepthWise")
            continue;

        // DeconvolutionDepthWise - Activation
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "InnerProduct")
            continue;

        // InnerProduct - Activation
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "MemoryData")
            continue;

        // MemoryData - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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

    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "MemoryData")
            continue;

        // MemoryData - Split - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        int j0 = i + 1;
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

        int j1 = j0 + 1;
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
    for (int i = 0; i < layer_count; i++)
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

        int j0 = 0;
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

        int j1 = 0;
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
    for (int i = 0; i < layer_count; i++)
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
    for (int i = 0; i < layer_count; i++)
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

            for (int k = 0; k < layers[j]->tops.size(); k++)
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Noop")
            continue;

        ncnn::Layer* noop = layers[i];

        if (noop->bottoms.empty())
        {
            // Noop
            fprintf(stderr, "eliminate_noop %s\n", noop->name.c_str());

            size_t top_blob_count = noop->tops.size();
            for (int k = 0; k < top_blob_count; k++)
            {
                int top_blob_index_final = noop->tops[k];
                blobs[top_blob_index_final].producer = -1;
            }
            noop->type = "ncnnfused";

            continue;
        }

        // Any - Noop
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

        fprintf(stderr, "eliminate_noop %s %s\n", any->name.c_str(), noop->name.c_str());

        size_t top_blob_count = std::min(noop->tops.size(), any->tops.size());
        for (int k = 0; k < top_blob_count; k++)
        {
            int top_blob_index_final = noop->tops[k];
            any->tops[k] = top_blob_index_final;
            blobs[top_blob_index_final].producer = j;
        }
        noop->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::eliminate_orphaned_memorydata()
{
    const size_t layer_count = layers.size();
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "MemoryData")
            continue;

        // MemoryData - X
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type == "ncnnfused")
                continue;

            bool orphaned = true;
            for (int k = 0; k < layers[j]->bottoms.size(); k++)
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Pooling")
            continue;

        ncnn::Pooling* pooling = (ncnn::Pooling*)layers[i];
        if (pooling->global_pooling == 0)
            continue;

        // Pooling - Reshape
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Pooling")
            continue;

        ncnn::Pooling* pooling = (ncnn::Pooling*)layers[i];
        if (pooling->global_pooling == 0)
            continue;

        // Pooling - Flatten
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "InnerProduct")
            continue;

        // InnerProduct - Flatten
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Reshape")
            continue;

        ncnn::Reshape* reshape = (ncnn::Reshape*)layers[i];
        if (reshape->w != 1 || reshape->h != 1 || reshape->permute != 0)
            continue;

        // Reshape - BinaryOp
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
        blobs[bottom_blob_index_final].consumers.erase(std::find(blobs[bottom_blob_index_final].consumers.begin(), blobs[bottom_blob_index_final].consumers.end(), i));
        blobs[bottom_blob_index_final].consumers.push_back(j);
        reshape->type = "ncnnfused";
    }

    return 0;
}

int NetOptimize::replace_convolution_with_innerproduct_after_global_pooling()
{
    const size_t layer_count = layers.size();
    for (int i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Pooling")
            continue;

        ncnn::Pooling* pooling = (ncnn::Pooling*)layers[i];
        if (pooling->global_pooling == 0)
            continue;

        // Pooling - Convolution
        int top_blob_index = layers[i]->tops[0];

        int j = i + 1;
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
        innerproduct->weight_data_int8_scales = convolution->weight_data_int8_scales;
        innerproduct->bottom_blob_int8_scale = convolution->bottom_blob_int8_scale;

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

        for (int i = 0; i < layer_count; i++)
        {
            if (layers[i]->type != "InnerProduct")
                continue;

            // InnerProduct - Convolution
            int top_blob_index = layers[i]->tops[0];

            int j = i + 1;
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
            innerproduct->weight_data_int8_scales = convolution->weight_data_int8_scales;
            innerproduct->bottom_blob_int8_scale = convolution->bottom_blob_int8_scale;

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

int NetOptimize::shape_inference()
{
    const size_t layer_count = layers.size();
    const size_t blob_count = blobs.size();

    ncnn::Extractor ex = create_extractor();

    // prepare Input blobs
    for (size_t i = 0; i < layer_count; i++)
    {
        const ncnn::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        if (layer->type != "Input")
            continue;

        ncnn::Input* input = (ncnn::Input*)layer;

        int w = input->w;
        int h = input->h;
        int c = input->c;

        int dims = 0;
        if (w == 0 && h == 0 && c == 0) dims = 0;
        if (w != 0 && h == 0 && c == 0) dims = 1;
        if (w != 0 && h != 0 && c == 0) dims = 2;
        if (w != 0 && h != 0 && c != 0) dims = 3;

        if (dims == 0)
        {
            fprintf(stderr, "Input layer %s without shape info, shape_inference aborted\n", layer->name.c_str());
            return -1;
        }

        ncnn::Mat m;
        if (dims == 1) m.create(w);
        if (dims == 2) m.create(w, h);
        if (dims == 3) m.create(w, h, c);

        ex.input(layer->tops[0], m);
    }

    // prepare blobs with predefined shape
    for (size_t i = 0; i < blob_count; i++)
    {
        const ncnn::Blob blob = blobs[i];

        int dims = blob.shape.dims;
        int w = blob.shape.w;
        int h = blob.shape.h;
        int c = blob.shape.c;

        if (dims == 0)
            continue;

        ncnn::Mat m;
        if (dims == 1) m.create(w);
        if (dims == 2) m.create(w, h);
        if (dims == 3) m.create(w, h, c);

        ex.input(int(i), m);
    }

    fprintf(stderr, "shape_inference\n");

    // resolve all layer output blob shape
    for (size_t i = 0; i < layer_count; i++)
    {
        const ncnn::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        for (size_t j = 0; j < layer->tops.size(); j++)
        {
            int top_blob_index = layer->tops[j];

            ncnn::Mat m;
            ex.extract(top_blob_index, m);

            blobs[top_blob_index].shape = m;
        }
    }

    // assign all layer blob shape
    for (size_t i = 0; i < layer_count; i++)
    {
        ncnn::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        layer->bottom_shapes.resize(layer->bottoms.size());
        for (size_t j = 0; j < layer->bottoms.size(); j++)
        {
            int bottom_blob_index = layer->bottoms[j];

            layer->bottom_shapes[j] = blobs[bottom_blob_index].shape;
        }

        layer->top_shapes.resize(layer->tops.size());
        for (size_t j = 0; j < layer->tops.size(); j++)
        {
            int top_blob_index = layer->tops[j];

            layer->top_shapes[j] = blobs[top_blob_index].shape;

            //             fprintf(stderr, "%d %4d %4d %4d | %2d %s\n", blobs[top_blob_index].shape.dims, blobs[top_blob_index].shape.w, blobs[top_blob_index].shape.h, blobs[top_blob_index].shape.c, top_blob_index, blobs[top_blob_index].name.c_str());
        }
    }

    return 0;
}

int NetOptimize::fprintf_param_int_array(int id, const ncnn::Mat& m, FILE* pp)
{
    const int count = m.w;
    const int* ptr = m;

    fprintf(pp, " -%d=%d", 23300 + id, count);
    for (int i = 0; i < count; i++)
    {
        fprintf(pp, ",%d", ptr[i]);
    }

    return 0;
}

int NetOptimize::fprintf_param_float_array(int id, const ncnn::Mat& m, FILE* pp)
{
    const int count = m.w;
    const float* ptr = m;

    fprintf(pp, " -%d=%d", 23300 + id, count);
    for (int i = 0; i < count; i++)
    {
        fprintf(pp, ",%e", ptr[i]);
    }

    return 0;
}

static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

int NetOptimize::fwrite_weight_tag_data(int tag, const ncnn::Mat& data, FILE* bp)
{
    int p0 = ftell(bp);

    ncnn::Mat data_flattened = data.reshape(data.w * data.h * data.c);
    if (storage_type == 1 && tag == 0)
    {
        tag = 0x01306B47; // fp16 magic
        fwrite(&tag, sizeof(int), 1, bp);
        ncnn::Mat data_flattened_fp16;
        ncnn::cast_float32_to_float16(data_flattened, data_flattened_fp16);
        fwrite(data_flattened_fp16.data, data_flattened_fp16.elemsize, data_flattened_fp16.w, bp);
    }
    else
    {
        fwrite(&tag, sizeof(int), 1, bp);
        fwrite(data_flattened.data, data_flattened.elemsize, data_flattened.w, bp);
    }

    // padding to 32bit align
    int nwrite = ftell(bp) - p0;
    size_t nalign = alignSize(nwrite, 4);
    unsigned char padding[4] = {0x00, 0x00, 0x00, 0x00};
    fwrite(padding, sizeof(unsigned char), nalign - nwrite, bp);

    return 0;
}

int NetOptimize::fwrite_weight_data(const ncnn::Mat& data, FILE* bp)
{
    int p0 = ftell(bp);

    ncnn::Mat data_flattened = data.reshape(data.w * data.h * data.c);
    fwrite(data_flattened.data, data_flattened.elemsize, data_flattened.w, bp);

    // padding to 32bit align
    int nwrite = ftell(bp) - p0;
    size_t nalign = alignSize(nwrite, 4);
    unsigned char padding[4] = {0x00, 0x00, 0x00, 0x00};
    fwrite(padding, sizeof(unsigned char), nalign - nwrite, bp);

    return 0;
}

int NetOptimize::save(const char* parampath, const char* binpath)
{
    unsigned int mac = 0;

    FILE* pp = fopen(parampath, "wb");
    FILE* bp = fopen(binpath, "wb");

    fprintf(pp, "7767517\n");

    const size_t layer_count = layers.size();

    int layer_count_fused = 0;
    std::set<std::string> blob_names;
    for (int i = 0; i < layer_count; i++)
    {
        const ncnn::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        layer_count_fused++;

        size_t bottom_count = layer->bottoms.size();
        for (int j = 0; j < bottom_count; j++)
        {
            int bottom_blob_index = layer->bottoms[j];
            blob_names.insert(blobs[bottom_blob_index].name);
        }

        size_t top_count = layer->tops.size();
        for (int j = 0; j < top_count; j++)
        {
            int top_blob_index = layer->tops[j];
            blob_names.insert(blobs[top_blob_index].name);
        }
    }

    size_t blob_count_fused = blob_names.size();

    fprintf(pp, "%d %zd\n", layer_count_fused, blob_count_fused);

    for (int i = 0; i < layer_count; i++)
    {
        const ncnn::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        size_t bottom_count = layer->bottoms.size();
        size_t top_count = layer->tops.size();

        fprintf(pp, "%-24s %-24s %zd %zd", layer->type.c_str(), layer->name.c_str(), bottom_count, top_count);

        for (int j = 0; j < bottom_count; j++)
        {
            int bottom_blob_index = layer->bottoms[j];
            fprintf(pp, " %s", blobs[bottom_blob_index].name.c_str());
        }
        for (int j = 0; j < top_count; j++)
        {
            int top_blob_index = layer->tops[j];
            fprintf(pp, " %s", blobs[top_blob_index].name.c_str());
        }

        // write shape hints
        bool shape_ready = true;
        for (int j = 0; j < top_count; j++)
        {
            int top_blob_index = layer->tops[j];

            int dims = blobs[top_blob_index].shape.dims;
            if (dims == 0)
            {
                shape_ready = false;
                break;
            }
        }
        if (shape_ready)
        {
            fprintf(pp, " -23330=%zd", top_count * 4);
            for (int j = 0; j < top_count; j++)
            {
                int top_blob_index = layer->tops[j];

                int dims = blobs[top_blob_index].shape.dims;
                int w = blobs[top_blob_index].shape.w;
                int h = blobs[top_blob_index].shape.h;
                int c = blobs[top_blob_index].shape.c;

                fprintf(pp, ",%d,%d,%d,%d", dims, w, h, c);
            }
        }

        ncnn::Layer* layer_default = ncnn::create_layer(layer->typeindex);

        ncnn::ParamDict pd;
        layer_default->load_param(pd);

#define fprintf_param_value(format, phase)                                  \
    {                                                                       \
        if (op->phase != op_default->phase) fprintf(pp, format, op->phase); \
    }

        if (layer->type == "BatchNorm")
        {
            ncnn::BatchNorm* op = (ncnn::BatchNorm*)layer;
            ncnn::BatchNorm* op_default = (ncnn::BatchNorm*)layer_default;

            fprintf_param_value(" 0=%d", channels)
            fprintf_param_value(" 1=%e", eps)

            fwrite_weight_data(op->slope_data, bp);
            fwrite_weight_data(op->mean_data, bp);
            fwrite_weight_data(op->var_data, bp);
            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "Bias")
        {
            ncnn::Bias* op = (ncnn::Bias*)layer;
            ncnn::Bias* op_default = (ncnn::Bias*)layer_default;

            fprintf_param_value(" 0=%d", bias_data_size)

            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "BinaryOp")
        {
            ncnn::BinaryOp* op = (ncnn::BinaryOp*)layer;
            ncnn::BinaryOp* op_default = (ncnn::BinaryOp*)layer_default;

            fprintf_param_value(" 0=%d", op_type)
            fprintf_param_value(" 1=%d", with_scalar)
            fprintf_param_value(" 2=%e", b)
        }
        else if (layer->type == "Clip")
        {
            ncnn::Clip* op = (ncnn::Clip*)layer;
            ncnn::Clip* op_default = (ncnn::Clip*)layer_default;

            fprintf_param_value(" 0=%e", min)
            fprintf_param_value(" 1=%e", max)
        }
        else if (layer->type == "Concat")
        {
            ncnn::Concat* op = (ncnn::Concat*)layer;
            ncnn::Concat* op_default = (ncnn::Concat*)layer_default;

            fprintf_param_value(" 0=%d", axis)
        }
        else if (layer->type == "Convolution")
        {
            ncnn::Convolution* op = (ncnn::Convolution*)layer;
            ncnn::Convolution* op_default = (ncnn::Convolution*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 14=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 16=%d", op->pad_bottom);
            }
            fprintf_param_value(" 18=%e", pad_value)
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 8=%d", int8_scale_term)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }
            fprintf_param_value(" 17=%d", impl_type)

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            if (shape_ready)
            {
                int inc = blobs[layer->bottoms[0]].shape.c;
                int outw = blobs[layer->tops[0]].shape.w;
                int outh = blobs[layer->tops[0]].shape.h;
                int outc = blobs[layer->tops[0]].shape.c;

                mac += op->kernel_h * op->kernel_w * outw * outh * outc * inc;
            }
        }
        else if (layer->type == "ConvolutionDepthWise")
        {
            ncnn::ConvolutionDepthWise* op = (ncnn::ConvolutionDepthWise*)layer;
            ncnn::ConvolutionDepthWise* op_default = (ncnn::ConvolutionDepthWise*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 14=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 16=%d", op->pad_bottom);
            }
            fprintf_param_value(" 18=%e", pad_value)
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 7=%d", group)
            fprintf_param_value(" 8=%d", int8_scale_term)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            if (shape_ready)
            {
                int inc = blobs[layer->bottoms[0]].shape.c;
                int outw = blobs[layer->tops[0]].shape.w;
                int outh = blobs[layer->tops[0]].shape.h;
                int outc = blobs[layer->tops[0]].shape.c;

                mac += op->kernel_h * op->kernel_w * outw * outh * (outc / op->group) * (inc / op->group) * op->group;
            }
        }
        else if (layer->type == "Crop")
        {
            ncnn::Crop* op = (ncnn::Crop*)layer;
            ncnn::Crop* op_default = (ncnn::Crop*)layer_default;

            fprintf_param_value(" 0=%d", woffset)
            fprintf_param_value(" 1=%d", hoffset)
            fprintf_param_value(" 2=%d", coffset)
            fprintf_param_value(" 3=%d", outw)
            fprintf_param_value(" 4=%d", outh)
            fprintf_param_value(" 5=%d", outc)
            fprintf_param_value(" 6=%d", woffset2)
            fprintf_param_value(" 7=%d", hoffset2)
            fprintf_param_value(" 8=%d", coffset2)
            {
                if (!op->starts.empty()) fprintf_param_int_array(9, op->starts, pp);
            }
            {
                if (!op->ends.empty()) fprintf_param_int_array(10, op->ends, pp);
            }
            {
                if (!op->axes.empty()) fprintf_param_int_array(11, op->axes, pp);
            }
        }
        else if (layer->type == "Deconvolution")
        {
            ncnn::Deconvolution* op = (ncnn::Deconvolution*)layer;
            ncnn::Deconvolution* op_default = (ncnn::Deconvolution*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 14=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 16=%d", op->pad_bottom);
            }
            fprintf_param_value(" 18=%d", output_pad_right)
            {
                if (op->output_pad_bottom != op->output_pad_right) fprintf(pp, " 19=%d", op->output_pad_bottom);
            }
            fprintf_param_value(" 20=%d", output_w)
            {
                if (op->output_h != op->output_w) fprintf(pp, " 21=%d", op->output_h);
            }
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            if (shape_ready)
            {
                int inw = blobs[layer->bottoms[0]].shape.w;
                int inh = blobs[layer->bottoms[0]].shape.h;
                int inc = blobs[layer->bottoms[0]].shape.c;
                int outc = blobs[layer->tops[0]].shape.c;

                mac += op->kernel_h * op->kernel_w * inw * inh * outc * inc;
            }
        }
        else if (layer->type == "DeconvolutionDepthWise")
        {
            ncnn::DeconvolutionDepthWise* op = (ncnn::DeconvolutionDepthWise*)layer;
            ncnn::DeconvolutionDepthWise* op_default = (ncnn::DeconvolutionDepthWise*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", dilation_w)
            {
                if (op->dilation_h != op->dilation_w) fprintf(pp, " 12=%d", op->dilation_h);
            }
            fprintf_param_value(" 3=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 13=%d", op->stride_h);
            }
            fprintf_param_value(" 4=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 14=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 15=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 16=%d", op->pad_bottom);
            }
            fprintf_param_value(" 18=%d", output_pad_right)
            {
                if (op->output_pad_bottom != op->output_pad_right) fprintf(pp, " 19=%d", op->output_pad_bottom);
            }
            fprintf_param_value(" 20=%d", output_w)
            {
                if (op->output_h != op->output_w) fprintf(pp, " 21=%d", op->output_h);
            }
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 7=%d", group)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            if (shape_ready)
            {
                int inw = blobs[layer->bottoms[0]].shape.w;
                int inh = blobs[layer->bottoms[0]].shape.h;
                int inc = blobs[layer->bottoms[0]].shape.c;
                int outc = blobs[layer->tops[0]].shape.c;

                mac += op->kernel_h * op->kernel_w * inw * inh * (outc / op->group) * (inc / op->group) * op->group;
            }
        }
        else if (layer->type == "DetectionOutput")
        {
            ncnn::DetectionOutput* op = (ncnn::DetectionOutput*)layer;
            ncnn::DetectionOutput* op_default = (ncnn::DetectionOutput*)layer_default;

            fprintf_param_value(" 0=%d", num_class)
            fprintf_param_value(" 1=%e", nms_threshold)
            fprintf_param_value(" 2=%d", nms_top_k)
            fprintf_param_value(" 3=%d", keep_top_k)
            fprintf_param_value(" 4=%e", confidence_threshold)
            fprintf_param_value(" 5=%e", variances[0])
            fprintf_param_value(" 6=%e", variances[1])
            fprintf_param_value(" 7=%e", variances[2])
            fprintf_param_value(" 8=%e", variances[3])
        }
        else if (layer->type == "Dropout")
        {
            ncnn::Dropout* op = (ncnn::Dropout*)layer;
            ncnn::Dropout* op_default = (ncnn::Dropout*)layer_default;

            fprintf_param_value(" 0=%e", scale)
        }
        else if (layer->type == "Eltwise")
        {
            ncnn::Eltwise* op = (ncnn::Eltwise*)layer;
            ncnn::Eltwise* op_default = (ncnn::Eltwise*)layer_default;

            fprintf_param_value(" 0=%d", op_type)
            {
                if (!op->coeffs.empty()) fprintf_param_float_array(1, op->coeffs, pp);
            }
        }
        else if (layer->type == "ELU")
        {
            ncnn::ELU* op = (ncnn::ELU*)layer;
            ncnn::ELU* op_default = (ncnn::ELU*)layer_default;

            fprintf_param_value(" 0=%e", alpha)
        }
        else if (layer->type == "Exp")
        {
            ncnn::Exp* op = (ncnn::Exp*)layer;
            ncnn::Exp* op_default = (ncnn::Exp*)layer_default;

            fprintf_param_value(" 0=%e", base)
            fprintf_param_value(" 1=%e", scale)
            fprintf_param_value(" 2=%e", shift)
        }
        else if (layer->type == "ExpandDims")
        {
            ncnn::ExpandDims* op = (ncnn::ExpandDims*)layer;
            ncnn::ExpandDims* op_default = (ncnn::ExpandDims*)layer_default;

            fprintf_param_value(" 0=%d", expand_w)
            fprintf_param_value(" 1=%d", expand_h)
            fprintf_param_value(" 2=%d", expand_c)
            {
                if (!op->axes.empty()) fprintf_param_int_array(0, op->axes, pp);
            }
        }
        else if (layer->type == "Gemm")
        {
            ncnn::Gemm* op = (ncnn::Gemm*)layer;
            ncnn::Gemm* op_default = (ncnn::Gemm*)layer_default;

            fprintf_param_value(" 0=%e", alpha)
            fprintf_param_value(" 1=%e", beta)
            fprintf_param_value(" 2=%d", transA)
            fprintf_param_value(" 3=%d", transB)
        }
        else if (layer->type == "GroupNorm")
        {
            ncnn::GroupNorm* op = (ncnn::GroupNorm*)layer;
            ncnn::GroupNorm* op_default = (ncnn::GroupNorm*)layer_default;

            fprintf_param_value(" 0=%d", group)
            fprintf_param_value(" 1=%d", channels)
            fprintf_param_value(" 2=%e", eps)
            fprintf_param_value(" 3=%d", affine)

            fwrite_weight_data(op->gamma_data, bp);
            fwrite_weight_data(op->beta_data, bp);
        }
        else if (layer->type == "HardSigmoid")
        {
            ncnn::HardSigmoid* op = (ncnn::HardSigmoid*)layer;
            ncnn::HardSigmoid* op_default = (ncnn::HardSigmoid*)layer_default;

            fprintf_param_value(" 0=%e", alpha)
            fprintf_param_value(" 1=%e", beta)
        }
        else if (layer->type == "HardSwish")
        {
            ncnn::HardSwish* op = (ncnn::HardSwish*)layer;
            ncnn::HardSwish* op_default = (ncnn::HardSwish*)layer_default;

            fprintf_param_value(" 0=%e", alpha)
            fprintf_param_value(" 1=%e", beta)
        }
        else if (layer->type == "InnerProduct")
        {
            ncnn::InnerProduct* op = (ncnn::InnerProduct*)layer;
            ncnn::InnerProduct* op_default = (ncnn::InnerProduct*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", bias_term)
            fprintf_param_value(" 2=%d", weight_data_size)
            fprintf_param_value(" 8=%d", int8_scale_term)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            if (shape_ready)
            {
                int inw = blobs[layer->bottoms[0]].shape.w;
                int inh = blobs[layer->bottoms[0]].shape.h;
                int inc = blobs[layer->bottoms[0]].shape.c;
                int outw = blobs[layer->tops[0]].shape.w;

                mac += inw * inh * inc * outw;
            }
        }
        else if (layer->type == "Input")
        {
            ncnn::Input* op = (ncnn::Input*)layer;
            ncnn::Input* op_default = (ncnn::Input*)layer_default;

            fprintf_param_value(" 0=%d", w)
            fprintf_param_value(" 1=%d", h)
            fprintf_param_value(" 2=%d", c)
        }
        else if (layer->type == "InstanceNorm")
        {
            ncnn::InstanceNorm* op = (ncnn::InstanceNorm*)layer;
            ncnn::InstanceNorm* op_default = (ncnn::InstanceNorm*)layer_default;

            fprintf_param_value(" 0=%d", channels)
            fprintf_param_value(" 1=%e", eps)
            fprintf_param_value(" 2=%d", affine)

            fwrite_weight_data(op->gamma_data, bp);
            fwrite_weight_data(op->beta_data, bp);
        }
        else if (layer->type == "Interp")
        {
            ncnn::Interp* op = (ncnn::Interp*)layer;
            ncnn::Interp* op_default = (ncnn::Interp*)layer_default;

            fprintf_param_value(" 0=%d", resize_type)
            fprintf_param_value(" 1=%e", height_scale)
            fprintf_param_value(" 2=%e", width_scale)
            fprintf_param_value(" 3=%d", output_height)
            fprintf_param_value(" 4=%d", output_width)
        }
        else if (layer->type == "Log")
        {
            ncnn::Log* op = (ncnn::Log*)layer;
            ncnn::Log* op_default = (ncnn::Log*)layer_default;

            fprintf_param_value(" 0=%e", base)
            fprintf_param_value(" 1=%e", scale)
            fprintf_param_value(" 2=%e", shift)
        }
        else if (layer->type == "LRN")
        {
            ncnn::LRN* op = (ncnn::LRN*)layer;
            ncnn::LRN* op_default = (ncnn::LRN*)layer_default;

            fprintf_param_value(" 0=%d", region_type)
            fprintf_param_value(" 1=%d", local_size)
            fprintf_param_value(" 2=%e", alpha)
            fprintf_param_value(" 3=%e", beta)
            fprintf_param_value(" 4=%e", bias)
        }
        else if (layer->type == "LSTM")
        {
            ncnn::LSTM* op = (ncnn::LSTM*)layer;
            ncnn::LSTM* op_default = (ncnn::LSTM*)layer_default;

            fprintf_param_value(" 0=%d", num_output)
            fprintf_param_value(" 1=%d", weight_data_size)
            fprintf_param_value(" 2=%d", direction)

            fwrite_weight_tag_data(0, op->weight_xc_data, bp);
            fwrite_weight_tag_data(0, op->bias_c_data, bp);
            fwrite_weight_tag_data(0, op->weight_hc_data, bp);
        }
        else if (layer->type == "MemoryData")
        {
            ncnn::MemoryData* op = (ncnn::MemoryData*)layer;
            ncnn::MemoryData* op_default = (ncnn::MemoryData*)layer_default;

            fprintf_param_value(" 0=%d", w)
            fprintf_param_value(" 1=%d", h)
            fprintf_param_value(" 2=%d", c)
            fwrite_weight_data(op->data, bp);
        }
        else if (layer->type == "MVN")
        {
            ncnn::MVN* op = (ncnn::MVN*)layer;
            ncnn::MVN* op_default = (ncnn::MVN*)layer_default;

            fprintf_param_value(" 0=%d", normalize_variance)
            fprintf_param_value(" 1=%d", across_channels)
            fprintf_param_value(" 2=%e", eps)
        }
        else if (layer->type == "Normalize")
        {
            ncnn::Normalize* op = (ncnn::Normalize*)layer;
            ncnn::Normalize* op_default = (ncnn::Normalize*)layer_default;

            fprintf_param_value(" 0=%d", across_spatial)
            fprintf_param_value(" 1=%d", channel_shared)
            fprintf_param_value(" 2=%e", eps)
            fprintf_param_value(" 3=%d", scale_data_size)
            fprintf_param_value(" 4=%d", across_channel)
            fprintf_param_value(" 9=%d", eps_mode)

            fwrite_weight_data(op->scale_data, bp);
        }
        else if (layer->type == "Padding")
        {
            ncnn::Padding* op = (ncnn::Padding*)layer;
            ncnn::Padding* op_default = (ncnn::Padding*)layer_default;

            fprintf_param_value(" 0=%d", top)
            fprintf_param_value(" 1=%d", bottom)
            fprintf_param_value(" 2=%d", left)
            fprintf_param_value(" 3=%d", right)
            fprintf_param_value(" 4=%d", type)
            fprintf_param_value(" 5=%e", value)
            fprintf_param_value(" 6=%d", per_channel_pad_data_size)
            fprintf_param_value(" 7=%d", front)
            fprintf_param_value(" 8=%d", behind)
        }
        else if (layer->type == "Permute")
        {
            ncnn::Permute* op = (ncnn::Permute*)layer;
            ncnn::Permute* op_default = (ncnn::Permute*)layer_default;

            fprintf_param_value(" 0=%d", order_type)
        }
        else if (layer->type == "PixelShuffle")
        {
            ncnn::PixelShuffle* op = (ncnn::PixelShuffle*)layer;
            ncnn::PixelShuffle* op_default = (ncnn::PixelShuffle*)layer_default;

            fprintf_param_value(" 0=%d", upscale_factor)
        }
        else if (layer->type == "Pooling")
        {
            ncnn::Pooling* op = (ncnn::Pooling*)layer;
            ncnn::Pooling* op_default = (ncnn::Pooling*)layer_default;

            fprintf_param_value(" 0=%d", pooling_type)
            fprintf_param_value(" 1=%d", kernel_w)
            {
                if (op->kernel_h != op->kernel_w) fprintf(pp, " 11=%d", op->kernel_h);
            }
            fprintf_param_value(" 2=%d", stride_w)
            {
                if (op->stride_h != op->stride_w) fprintf(pp, " 12=%d", op->stride_h);
            }
            fprintf_param_value(" 3=%d", pad_left)
            {
                if (op->pad_top != op->pad_left) fprintf(pp, " 13=%d", op->pad_top);
            }
            {
                if (op->pad_right != op->pad_left) fprintf(pp, " 14=%d", op->pad_right);
            }
            {
                if (op->pad_bottom != op->pad_top) fprintf(pp, " 15=%d", op->pad_bottom);
            }
            fprintf_param_value(" 4=%d", global_pooling)
            fprintf_param_value(" 5=%d", pad_mode)
        }
        else if (layer->type == "Power")
        {
            ncnn::Power* op = (ncnn::Power*)layer;
            ncnn::Power* op_default = (ncnn::Power*)layer_default;

            fprintf_param_value(" 0=%e", power)
            fprintf_param_value(" 1=%e", scale)
            fprintf_param_value(" 2=%e", shift)
        }
        else if (layer->type == "PReLU")
        {
            ncnn::PReLU* op = (ncnn::PReLU*)layer;
            ncnn::PReLU* op_default = (ncnn::PReLU*)layer_default;

            fprintf_param_value(" 0=%d", num_slope)

            fwrite_weight_data(op->slope_data, bp);
        }
        else if (layer->type == "PriorBox")
        {
            ncnn::PriorBox* op = (ncnn::PriorBox*)layer;
            ncnn::PriorBox* op_default = (ncnn::PriorBox*)layer_default;

            {
                if (!op->min_sizes.empty()) fprintf_param_float_array(0, op->min_sizes, pp);
            }
            {
                if (!op->max_sizes.empty()) fprintf_param_float_array(1, op->max_sizes, pp);
            }
            {
                if (!op->aspect_ratios.empty()) fprintf_param_float_array(2, op->aspect_ratios, pp);
            }
            fprintf_param_value(" 3=%e", variances[0])
            fprintf_param_value(" 4=%e", variances[1])
            fprintf_param_value(" 5=%e", variances[2])
            fprintf_param_value(" 6=%e", variances[3])
            fprintf_param_value(" 7=%d", flip)
            fprintf_param_value(" 8=%d", clip)
            fprintf_param_value(" 9=%d", image_width)
            fprintf_param_value(" 10=%d", image_height)
            fprintf_param_value(" 11=%e", step_width)
            fprintf_param_value(" 12=%e", step_height)
            fprintf_param_value(" 13=%e", offset)
        }
        else if (layer->type == "Proposal")
        {
            ncnn::Proposal* op = (ncnn::Proposal*)layer;
            ncnn::Proposal* op_default = (ncnn::Proposal*)layer_default;

            fprintf_param_value(" 0=%d", feat_stride)
            fprintf_param_value(" 1=%d", base_size)
            fprintf_param_value(" 2=%d", pre_nms_topN)
            fprintf_param_value(" 3=%d", after_nms_topN)
            fprintf_param_value(" 4=%e", nms_thresh)
            fprintf_param_value(" 5=%d", min_size)
        }
        else if (layer->type == "PSROIPooling")
        {
            ncnn::PSROIPooling* op = (ncnn::PSROIPooling*)layer;
            ncnn::PSROIPooling* op_default = (ncnn::PSROIPooling*)layer_default;

            fprintf_param_value(" 0=%d", pooled_width)
            fprintf_param_value(" 1=%d", pooled_height)
            fprintf_param_value(" 2=%e", spatial_scale)
            fprintf_param_value(" 3=%d", output_dim)
        }
        else if (layer->type == "Quantize")
        {
            ncnn::Quantize* op = (ncnn::Quantize*)layer;
            ncnn::Quantize* op_default = (ncnn::Quantize*)layer_default;

            fprintf_param_value(" 0=%e", scale)
        }
        else if (layer->type == "Reduction")
        {
            ncnn::Reduction* op = (ncnn::Reduction*)layer;
            ncnn::Reduction* op_default = (ncnn::Reduction*)layer_default;

            fprintf_param_value(" 0=%d", operation)
            fprintf_param_value(" 1=%d", reduce_all)
            fprintf_param_value(" 2=%e", coeff)
            {
                if (!op->axes.empty()) fprintf_param_int_array(3, op->axes, pp);
            }
            fprintf_param_value(" 4=%d", keepdims)
        }
        else if (layer->type == "ReLU")
        {
            ncnn::ReLU* op = (ncnn::ReLU*)layer;
            ncnn::ReLU* op_default = (ncnn::ReLU*)layer_default;

            fprintf_param_value(" 0=%e", slope)
        }
        else if (layer->type == "Reorg")
        {
            ncnn::Reorg* op = (ncnn::Reorg*)layer;
            ncnn::Reorg* op_default = (ncnn::Reorg*)layer_default;

            fprintf_param_value(" 0=%d", stride)
        }
        else if (layer->type == "Requantize")
        {
            ncnn::Requantize* op = (ncnn::Requantize*)layer;
            ncnn::Requantize* op_default = (ncnn::Requantize*)layer_default;

            fprintf_param_value(" 0=%e", scale_in)
            fprintf_param_value(" 1=%e", scale_out)
            fprintf_param_value(" 2=%d", bias_term)
            fprintf_param_value(" 3=%d", bias_data_size)
            fprintf_param_value(" 4=%d", fusion_relu)
        }
        else if (layer->type == "Reshape")
        {
            ncnn::Reshape* op = (ncnn::Reshape*)layer;
            ncnn::Reshape* op_default = (ncnn::Reshape*)layer_default;

            fprintf_param_value(" 0=%d", w)
            fprintf_param_value(" 1=%d", h)
            fprintf_param_value(" 2=%d", c)
            fprintf_param_value(" 3=%d", permute)
        }
        else if (layer->type == "ROIAlign")
        {
            ncnn::ROIAlign* op = (ncnn::ROIAlign*)layer;
            ncnn::ROIAlign* op_default = (ncnn::ROIAlign*)layer_default;

            fprintf_param_value(" 0=%d", pooled_width)
            fprintf_param_value(" 1=%d", pooled_height)
            fprintf_param_value(" 2=%e", spatial_scale)
            fprintf_param_value(" 3=%d", sampling_ratio)
            fprintf_param_value(" 4=%d", aligned)
            fprintf_param_value(" 5=%d", version)
        }
        else if (layer->type == "ROIPooling")
        {
            ncnn::ROIPooling* op = (ncnn::ROIPooling*)layer;
            ncnn::ROIPooling* op_default = (ncnn::ROIPooling*)layer_default;

            fprintf_param_value(" 0=%d", pooled_width)
            fprintf_param_value(" 1=%d", pooled_height)
            fprintf_param_value(" 2=%e", spatial_scale)
        }
        else if (layer->type == "Scale")
        {
            ncnn::Scale* op = (ncnn::Scale*)layer;
            ncnn::Scale* op_default = (ncnn::Scale*)layer_default;

            fprintf_param_value(" 0=%d", scale_data_size)
            fprintf_param_value(" 1=%d", bias_term)

            fwrite_weight_data(op->scale_data, bp);
            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "ShuffleChannel")
        {
            ncnn::ShuffleChannel* op = (ncnn::ShuffleChannel*)layer;
            ncnn::ShuffleChannel* op_default = (ncnn::ShuffleChannel*)layer_default;

            fprintf_param_value(" 0=%d", group)
        }
        else if (layer->type == "Slice")
        {
            ncnn::Slice* op = (ncnn::Slice*)layer;
            ncnn::Slice* op_default = (ncnn::Slice*)layer_default;

            {
                if (!op->slices.empty()) fprintf_param_int_array(0, op->slices, pp);
            }
            fprintf_param_value(" 1=%d", axis)
        }
        else if (layer->type == "Softmax")
        {
            ncnn::Softmax* op = (ncnn::Softmax*)layer;
            ncnn::Softmax* op_default = (ncnn::Softmax*)layer_default;

            fprintf_param_value(" 0=%d", axis)

            // HACK
            if (op->axis != 0)
            {
                int fixbug0 = 1;
                fprintf(pp, " 1=%d", fixbug0);
            }
        }
        else if (layer->type == "Squeeze")
        {
            ncnn::Squeeze* op = (ncnn::Squeeze*)layer;
            ncnn::Squeeze* op_default = (ncnn::Squeeze*)layer_default;

            fprintf_param_value(" 0=%d", squeeze_w)
            fprintf_param_value(" 1=%d", squeeze_h)
            fprintf_param_value(" 2=%d", squeeze_c)
            {
                if (!op->axes.empty()) fprintf_param_int_array(0, op->axes, pp);
            }
        }
        else if (layer->type == "Threshold")
        {
            ncnn::Threshold* op = (ncnn::Threshold*)layer;
            ncnn::Threshold* op_default = (ncnn::Threshold*)layer_default;

            fprintf_param_value(" 0=%e", threshold)
        }
        else if (layer->type == "UnaryOp")
        {
            ncnn::UnaryOp* op = (ncnn::UnaryOp*)layer;
            ncnn::UnaryOp* op_default = (ncnn::UnaryOp*)layer_default;

            fprintf_param_value(" 0=%d", op_type)
        }
        else if (layer->type == "YoloDetectionOutput")
        {
            ncnn::YoloDetectionOutput* op = (ncnn::YoloDetectionOutput*)layer;
            ncnn::YoloDetectionOutput* op_default = (ncnn::YoloDetectionOutput*)layer_default;

            fprintf_param_value(" 0=%d", num_class)
            fprintf_param_value(" 1=%d", num_box)
            fprintf_param_value(" 2=%e", confidence_threshold)
            fprintf_param_value(" 3=%e", nms_threshold)
            {
                if (!op->biases.empty()) fprintf_param_float_array(4, op->biases, pp);
            }
        }
        else if (layer->type == "Yolov3DetectionOutput")
        {
            ncnn::Yolov3DetectionOutput* op = (ncnn::Yolov3DetectionOutput*)layer;
            ncnn::Yolov3DetectionOutput* op_default = (ncnn::Yolov3DetectionOutput*)layer_default;

            fprintf_param_value(" 0=%d", num_class)
            fprintf_param_value(" 1=%d", num_box)
            fprintf_param_value(" 2=%e", confidence_threshold)
            fprintf_param_value(" 3=%e", nms_threshold)
            {
                if (!op->biases.empty()) fprintf_param_float_array(4, op->biases, pp);
            }
            {
                if (!op->mask.empty()) fprintf_param_int_array(5, op->mask, pp);
            }
            {
                if (!op->anchors_scale.empty()) fprintf_param_float_array(6, op->anchors_scale, pp);
            }
        }

#undef fprintf_param_value

        fprintf(pp, "\n");

        delete layer_default;
    }

    fclose(pp);
    fclose(bp);

    if (mac)
    {
        fprintf(stderr, "mac = %d = %.2f M\n", mac, mac / 1000000.f);
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 6)
    {
        fprintf(stderr, "usage: %s [inparam] [inbin] [outparam] [outbin] [flag]\n", argv[0]);
        return -1;
    }

    const char* inparam = argv[1];
    const char* inbin = argv[2];
    const char* outparam = argv[3];
    const char* outbin = argv[4];
    int flag = atoi(argv[5]);

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
    }
    else
        optimizer.load_model(inbin);

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
    optimizer.eliminate_flatten_after_global_pooling();
    optimizer.eliminate_reshape_after_global_pooling();
    optimizer.eliminate_reshape_before_binaryop();

    optimizer.replace_convolution_with_innerproduct_after_global_pooling();
    optimizer.replace_convolution_with_innerproduct_after_innerproduct();

    optimizer.eliminate_flatten_after_innerproduct();
    optimizer.eliminate_orphaned_memorydata();

    optimizer.shape_inference();

    optimizer.save(outparam, outbin);

    return 0;
}
