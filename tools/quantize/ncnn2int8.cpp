// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
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

#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <vector>

// ncnn public header
#include "layer.h"
#include "layer_type.h"
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
#include "layer/flatten.h"
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
#include "layer/threshold.h"
#include "layer/unaryop.h"
#include "layer/yolodetectionoutput.h"
#include "layer/yolov3detectionoutput.h"

static bool read_int8scale_table(const char* filepath, std::map<std::string, std::vector<float> >& blob_int8scale_table, std::map<std::string, std::vector<float> >& weight_int8scale_table)
{
    blob_int8scale_table.clear();
    weight_int8scale_table.clear();

    FILE* fp = fopen(filepath, "rb");
    if (!fp)
    {
        fprintf(stderr, "Open %s failed.\n", filepath);
        return false;
    }

    std::string key_str;
    std::vector<float> scales;

    std::vector<char> line(102400);
    char* pch = NULL;
    size_t len = 0;

    while (NULL != std::fgets(line.data(), static_cast<int>(line.size()), fp))
    {
        float scale = 1.f;
        char key[256];
        line[strcspn(line.data(), "\r\n")] = 0;

        pch = strtok(line.data(), " ");

        if (pch == NULL) break;

        bool is_key = true;
        while (pch != NULL)
        {
            if (is_key)
            {
                sscanf(pch, "%255s", key);

                key_str = key;
                is_key = false;
            }
            else
            {
                sscanf(pch, "%f", &scale);

                scales.push_back(scale);
            }

            pch = strtok(NULL, " ");
        }

        // XYZ_param_N pattern
        if (strstr(key_str.c_str(), "_param_"))
        {
            weight_int8scale_table[key_str] = scales;
        }
        else
        {
            blob_int8scale_table[key_str] = scales;
        }
        key_str.clear();
        scales.clear();
    }

    fclose(fp);

    return true;
}

class NetQuantize : public ncnn::Net
{
public:
    // 0=fp32 1=fp16 2=int8
    int storage_type;
    std::map<std::string, std::vector<float> > blob_int8scale_table;
    std::map<std::string, std::vector<float> > weight_int8scale_table;

public:
    int quantize_convolution();
    int quantize_convolutiondepthwise();
    int quantize_innerproduct();

public:
    int fprintf_param_int_array(int id, const ncnn::Mat& m, FILE* pp);
    int fprintf_param_float_array(int id, const ncnn::Mat& m, FILE* pp);

    int fwrite_weight_tag_data(int tag, const ncnn::Mat& data, FILE* bp);
    int fwrite_weight_data(const ncnn::Mat& data, FILE* bp);

    int save(const char* parampath, const char* binpath);
};

int NetQuantize::quantize_convolution()
{
    const int layer_count = static_cast<int>(layers.size());
    for (int i = 0; i < layer_count; i++)
    {
        // find convoultion layer
        if (layers[i]->type != "Convolution")
            continue;

        // find convolution layer
        std::map<std::string, std::vector<float> >::iterator iter_data = blob_int8scale_table.find(layers[i]->name);
        if (iter_data == blob_int8scale_table.end())
            continue;

        char key[256];
        sprintf(key, "%s_param_0", layers[i]->name.c_str());

        std::map<std::string, std::vector<float> >::iterator iter = weight_int8scale_table.find(key);
        if (iter == weight_int8scale_table.end())
        {
            fprintf(stderr, "this layer need to be quantized, but no scale param!\n");
            return -1;
        }

        // Convolution - quantize weight from fp32 to int8
        ncnn::Convolution* convolution = (ncnn::Convolution*)layers[i];

        std::vector<float> weight_data_int8_scales = iter->second;

        fprintf(stderr, "quantize_convolution %s\n", convolution->name.c_str());

        {
            ncnn::Mat int8_weight_data(convolution->weight_data_size, (size_t)1u);
            if (int8_weight_data.empty())
                return -100;

            const int weight_data_size_output = convolution->weight_data_size / convolution->num_output;

            // quantize weight to int8
            for (int n = 0; n < convolution->num_output; n++)
            {
                ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Quantize);

                ncnn::ParamDict pd;
                pd.set(0, weight_data_int8_scales[n]); // scale

                op->load_param(pd);

                ncnn::Option opt;
                opt.blob_allocator = int8_weight_data.allocator;

                const ncnn::Mat weight_data_n = convolution->weight_data.range(weight_data_size_output * n, weight_data_size_output);
                ncnn::Mat int8_weight_data_n = int8_weight_data.range(weight_data_size_output * n, weight_data_size_output);
                op->forward(weight_data_n, int8_weight_data_n, opt);

                delete op;
            }

            convolution->weight_data = int8_weight_data;
        }

        convolution->int8_scale_term = 2;
    }

    return 0;
}

int NetQuantize::quantize_convolutiondepthwise()
{
    const int layer_count = static_cast<int>(layers.size());
    for (int i = 0; i < layer_count; i++)
    {
        // find convoultion layer
        if (layers[i]->type != "ConvolutionDepthWise")
            continue;

        // find convolutiondepthwise layer
        std::map<std::string, std::vector<float> >::iterator iter_data = blob_int8scale_table.find(layers[i]->name);
        if (iter_data == blob_int8scale_table.end())
            continue;

        char key[256];
        sprintf(key, "%s_param_0", layers[i]->name.c_str());

        std::map<std::string, std::vector<float> >::iterator iter = weight_int8scale_table.find(key);
        if (iter == weight_int8scale_table.end())
        {
            fprintf(stderr, "this layer need to be quantized, but no scale param!\n");
            return -1;
        }

        // Convolution - quantize weight from fp32 to int8
        ncnn::ConvolutionDepthWise* convdw = (ncnn::ConvolutionDepthWise*)layers[i];

        std::vector<float> weight_data_int8_scales = iter->second;

        fprintf(stderr, "quantize_convolution %s\n", convdw->name.c_str());

        {
            ncnn::Mat int8_weight_data(convdw->weight_data_size, (size_t)1u);
            if (int8_weight_data.empty())
                return -100;

            const int weight_data_size_output = convdw->weight_data_size / convdw->group;

            // quantize weight to int8
            for (int n = 0; n < convdw->group; n++)
            {
                ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Quantize);

                ncnn::ParamDict pd;
                pd.set(0, weight_data_int8_scales[n]); // scale

                op->load_param(pd);

                ncnn::Option opt;
                opt.blob_allocator = int8_weight_data.allocator;

                const ncnn::Mat weight_data_n = convdw->weight_data.range(weight_data_size_output * n, weight_data_size_output);
                ncnn::Mat int8_weight_data_n = int8_weight_data.range(weight_data_size_output * n, weight_data_size_output);
                op->forward(weight_data_n, int8_weight_data_n, opt);

                delete op;
            }

            convdw->weight_data = int8_weight_data;
        }

        convdw->int8_scale_term = 1;
    }

    return 0;
}

int NetQuantize::quantize_innerproduct()
{
    const int layer_count = static_cast<int>(layers.size());
    for (int i = 0; i < layer_count; i++)
    {
        // find convoultion layer
        if (layers[i]->type != "InnerProduct")
            continue;

        // find InnerProduct layer
        std::map<std::string, std::vector<float> >::iterator iter_data = blob_int8scale_table.find(layers[i]->name);
        if (iter_data == blob_int8scale_table.end())
            continue;

        char key[256];
        sprintf(key, "%s_param_0", layers[i]->name.c_str());

        std::map<std::string, std::vector<float> >::iterator iter = weight_int8scale_table.find(key);
        if (iter == weight_int8scale_table.end())
        {
            fprintf(stderr, "this layer need to be quantized, but no scale param!\n");
            return -1;
        }

        // InnerProduct - quantize weight from fp32 to int8
        ncnn::InnerProduct* fc = (ncnn::InnerProduct*)layers[i];

        std::vector<float> weight_data_int8_scales = iter->second;

        fprintf(stderr, "quantize_convolution %s\n", fc->name.c_str());

        {
            ncnn::Mat int8_weight_data(fc->weight_data_size, (size_t)1u);
            if (int8_weight_data.empty())
                return -100;

            const int weight_data_size_output = fc->weight_data_size / fc->num_output;

            // quantize weight to int8
            for (int n = 0; n < fc->num_output; n++)
            {
                ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Quantize);

                ncnn::ParamDict pd;
                pd.set(0, weight_data_int8_scales[n]); // scale

                op->load_param(pd);

                ncnn::Option opt;
                opt.blob_allocator = int8_weight_data.allocator;

                const ncnn::Mat weight_data_n = fc->weight_data.range(weight_data_size_output * n, weight_data_size_output);
                ncnn::Mat int8_weight_data_n = int8_weight_data.range(weight_data_size_output * n, weight_data_size_output);
                op->forward(weight_data_n, int8_weight_data_n, opt);

                delete op;
            }

            fc->weight_data = int8_weight_data;
        }

        fc->int8_scale_term = 2;
    }

    return 0;
}

int NetQuantize::fprintf_param_int_array(int id, const ncnn::Mat& m, FILE* pp)
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

int NetQuantize::fprintf_param_float_array(int id, const ncnn::Mat& m, FILE* pp)
{
    const int count = m.w;
    const float* ptr = m;

    fprintf(pp, " -%d=%d", 23300 + id, count);
    for (int i = 0; i < count; i++)
    {
        fprintf(pp, ",%f", ptr[i]);
    }

    return 0;
}

static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

int NetQuantize::fwrite_weight_tag_data(int tag, const ncnn::Mat& data, FILE* bp)
{
    int p0 = ftell(bp);

    ncnn::Mat data_flattened = data.reshape(data.w * data.h * data.c);

    if (data.elemsize == 1)
        tag = 0x000D4B38; // int8 magic

    fwrite(&tag, sizeof(int), 1, bp);
    fwrite(data_flattened.data, data_flattened.elemsize, data_flattened.w, bp);

    // padding to 32bit align
    int nwrite = ftell(bp) - p0;
    int nalign = static_cast<int>(alignSize(nwrite, 4));
    unsigned char padding[4] = {0x00, 0x00, 0x00, 0x00};
    fwrite(padding, sizeof(unsigned char), nalign - nwrite, bp);

    return 0;
}

int NetQuantize::fwrite_weight_data(const ncnn::Mat& data, FILE* bp)
{
    int p0 = ftell(bp);

    ncnn::Mat data_flattened = data.reshape(data.w * data.h * data.c);
    fwrite(data_flattened.data, data_flattened.elemsize, data_flattened.w, bp);

    // padding to 32bit align
    int nwrite = ftell(bp) - p0;
    int nalign = static_cast<int>(alignSize(nwrite, 4));
    unsigned char padding[4] = {0x00, 0x00, 0x00, 0x00};
    fwrite(padding, sizeof(unsigned char), nalign - nwrite, bp);

    return 0;
}

int NetQuantize::save(const char* parampath, const char* binpath)
{
    FILE* pp = fopen(parampath, "wb");
    FILE* bp = fopen(binpath, "wb");

    fprintf(pp, "7767517\n");

    const int layer_count = static_cast<int>(layers.size());

    int layer_count_fused = 0;
    std::set<std::string> blob_names;
    for (int i = 0; i < layer_count; i++)
    {
        const ncnn::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        layer_count_fused++;

        int bottom_count = static_cast<int>(layer->bottoms.size());
        for (int j = 0; j < bottom_count; j++)
        {
            int bottom_blob_index = layer->bottoms[j];
            blob_names.insert(blobs[bottom_blob_index].name);
        }

        int top_count = static_cast<int>(layer->tops.size());
        for (int j = 0; j < top_count; j++)
        {
            int top_blob_index = layer->tops[j];
            blob_names.insert(blobs[top_blob_index].name);
        }
    }

    int blob_count_fused = static_cast<int>(blob_names.size());

    fprintf(pp, "%d %d\n", layer_count_fused, blob_count_fused);

    for (int i = 0; i < layer_count; i++)
    {
        const ncnn::Layer* layer = layers[i];
        if (layer->type == "ncnnfused")
            continue;

        int bottom_count = static_cast<int>(layer->bottoms.size());
        int top_count = static_cast<int>(layer->tops.size());

        fprintf(pp, "%-24s %-24s %d %d", layer->type.c_str(), layer->name.c_str(), bottom_count, top_count);

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
            fprintf_param_value(" 1=%f", eps)

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
            fprintf_param_value(" 2=%f", b)
        }
        else if (layer->type == "Clip")
        {
            ncnn::Clip* op = (ncnn::Clip*)layer;
            ncnn::Clip* op_default = (ncnn::Clip*)layer_default;

            fprintf_param_value(" 0=%f", min)
            fprintf_param_value(" 1=%f", max)
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
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 8=%d", int8_scale_term)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);

            // write int8_scale data
            if (op->int8_scale_term)
            {
                std::vector<float> weight_int8scale;
                std::vector<float> blob_int8scale;

                char key[256];
                sprintf(key, "%s_param_0", layers[i]->name.c_str());

                if (weight_int8scale_table.find(std::string(key)) != weight_int8scale_table.end())
                {
                    weight_int8scale = weight_int8scale_table[std::string(key)];
                }

                if (blob_int8scale_table.find(layer->name) != blob_int8scale_table.end())
                {
                    blob_int8scale = blob_int8scale_table[layer->name];
                }

                // write int8_scale data
                fwrite(weight_int8scale.data(), sizeof(float), weight_int8scale.size(), bp);
                fwrite(blob_int8scale.data(), sizeof(float), blob_int8scale.size(), bp);
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

            // write int8_scale data
            if (op->int8_scale_term)
            {
                std::vector<float> weight_int8scale;
                std::vector<float> blob_int8scale;

                char key[256];
                sprintf(key, "%s_param_0", layers[i]->name.c_str());

                if (weight_int8scale_table.find(std::string(key)) != weight_int8scale_table.end())
                {
                    weight_int8scale = weight_int8scale_table[std::string(key)];
                }

                if (blob_int8scale_table.find(layer->name) != blob_int8scale_table.end())
                {
                    blob_int8scale = blob_int8scale_table[layer->name];
                }

                // write int8_scale data
                fwrite(weight_int8scale.data(), sizeof(float), weight_int8scale.size(), bp);
                fwrite(blob_int8scale.data(), sizeof(float), blob_int8scale.size(), bp);
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
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);
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
            fprintf_param_value(" 5=%d", bias_term)
            fprintf_param_value(" 6=%d", weight_data_size)
            fprintf_param_value(" 7=%d", group)
            fprintf_param_value(" 9=%d", activation_type)
            {
                if (!op->activation_params.empty()) fprintf_param_float_array(10, op->activation_params, pp);
            }

            fwrite_weight_tag_data(0, op->weight_data, bp);
            fwrite_weight_data(op->bias_data, bp);
        }
        else if (layer->type == "DetectionOutput")
        {
            ncnn::DetectionOutput* op = (ncnn::DetectionOutput*)layer;
            ncnn::DetectionOutput* op_default = (ncnn::DetectionOutput*)layer_default;

            fprintf_param_value(" 0=%d", num_class)
            fprintf_param_value(" 1=%f", nms_threshold)
            fprintf_param_value(" 2=%d", nms_top_k)
            fprintf_param_value(" 3=%d", keep_top_k)
            fprintf_param_value(" 4=%f", confidence_threshold)
            fprintf_param_value(" 5=%f", variances[0])
            fprintf_param_value(" 6=%f", variances[1])
            fprintf_param_value(" 7=%f", variances[2])
            fprintf_param_value(" 8=%f", variances[3])
        }
        else if (layer->type == "Dropout")
        {
            ncnn::Dropout* op = (ncnn::Dropout*)layer;
            ncnn::Dropout* op_default = (ncnn::Dropout*)layer_default;

            fprintf_param_value(" 0=%f", scale)
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

            fprintf_param_value(" 0=%f", alpha)
        }
        else if (layer->type == "Exp")
        {
            ncnn::Exp* op = (ncnn::Exp*)layer;
            ncnn::Exp* op_default = (ncnn::Exp*)layer_default;

            fprintf_param_value(" 0=%f", base)
            fprintf_param_value(" 1=%f", scale)
            fprintf_param_value(" 2=%f", shift)
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

            // write int8_scale data
            if (op->int8_scale_term)
            {
                std::vector<float> weight_int8scale;
                std::vector<float> blob_int8scale;

                char key[256];
                sprintf(key, "%s_param_0", layers[i]->name.c_str());

                if (weight_int8scale_table.find(std::string(key)) != weight_int8scale_table.end())
                {
                    weight_int8scale = weight_int8scale_table[std::string(key)];
                }

                if (blob_int8scale_table.find(layer->name) != blob_int8scale_table.end())
                {
                    blob_int8scale = blob_int8scale_table[layer->name];
                }

                // write int8_scale data
                fwrite(weight_int8scale.data(), sizeof(float), weight_int8scale.size(), bp);
                fwrite(blob_int8scale.data(), sizeof(float), blob_int8scale.size(), bp);
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
            fprintf_param_value(" 1=%f", eps)
        }
        else if (layer->type == "Interp")
        {
            ncnn::Interp* op = (ncnn::Interp*)layer;
            ncnn::Interp* op_default = (ncnn::Interp*)layer_default;

            fprintf_param_value(" 0=%d", resize_type)
            fprintf_param_value(" 1=%f", height_scale)
            fprintf_param_value(" 2=%f", width_scale)
            fprintf_param_value(" 3=%d", output_height)
            fprintf_param_value(" 4=%d", output_width)
        }
        else if (layer->type == "Log")
        {
            ncnn::Log* op = (ncnn::Log*)layer;
            ncnn::Log* op_default = (ncnn::Log*)layer_default;

            fprintf_param_value(" 0=%f", base)
            fprintf_param_value(" 1=%f", scale)
            fprintf_param_value(" 2=%f", shift)
        }
        else if (layer->type == "LRN")
        {
            ncnn::LRN* op = (ncnn::LRN*)layer;
            ncnn::LRN* op_default = (ncnn::LRN*)layer_default;

            fprintf_param_value(" 0=%d", region_type)
            fprintf_param_value(" 1=%d", local_size)
            fprintf_param_value(" 2=%f", alpha)
            fprintf_param_value(" 3=%f", beta)
            fprintf_param_value(" 4=%f", bias)
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
            fprintf_param_value(" 2=%f", eps)
        }
        else if (layer->type == "Normalize")
        {
            ncnn::Normalize* op = (ncnn::Normalize*)layer;
            ncnn::Normalize* op_default = (ncnn::Normalize*)layer_default;

            fprintf_param_value(" 0=%d", across_spatial)
            fprintf_param_value(" 1=%d", channel_shared)
            fprintf_param_value(" 2=%f", eps)
            fprintf_param_value(" 3=%d", scale_data_size)
            fprintf_param_value(" 4=%d", across_channel)

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
            fprintf_param_value(" 5=%f", value)
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

            fprintf_param_value(" 0=%f", power)
            fprintf_param_value(" 1=%f", scale)
            fprintf_param_value(" 2=%f", shift)
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
            fprintf_param_value(" 3=%f", variances[0])
            fprintf_param_value(" 4=%f", variances[1])
            fprintf_param_value(" 5=%f", variances[2])
            fprintf_param_value(" 6=%f", variances[3])
            fprintf_param_value(" 7=%d", flip)
            fprintf_param_value(" 8=%d", clip)
            fprintf_param_value(" 9=%d", image_width)
            fprintf_param_value(" 10=%d", image_height)
            fprintf_param_value(" 11=%f", step_width)
            fprintf_param_value(" 12=%f", step_height)
            fprintf_param_value(" 13=%f", offset)
        }
        else if (layer->type == "Proposal")
        {
            ncnn::Proposal* op = (ncnn::Proposal*)layer;
            ncnn::Proposal* op_default = (ncnn::Proposal*)layer_default;

            fprintf_param_value(" 0=%d", feat_stride)
            fprintf_param_value(" 1=%d", base_size)
            fprintf_param_value(" 2=%d", pre_nms_topN)
            fprintf_param_value(" 3=%d", after_nms_topN)
            fprintf_param_value(" 4=%f", nms_thresh)
            fprintf_param_value(" 5=%d", min_size)
        }
        else if (layer->type == "PSROIPooling")
        {
            ncnn::PSROIPooling* op = (ncnn::PSROIPooling*)layer;
            ncnn::PSROIPooling* op_default = (ncnn::PSROIPooling*)layer_default;

            fprintf_param_value(" 0=%d", pooled_width)
            fprintf_param_value(" 1=%d", pooled_height)
            fprintf_param_value(" 2=%f", spatial_scale)
            fprintf_param_value(" 3=%d", output_dim)
        }
        else if (layer->type == "Quantize")
        {
            ncnn::Quantize* op = (ncnn::Quantize*)layer;
            ncnn::Quantize* op_default = (ncnn::Quantize*)layer_default;

            fprintf_param_value(" 0=%f", scale)
        }
        else if (layer->type == "Reduction")
        {
            ncnn::Reduction* op = (ncnn::Reduction*)layer;
            ncnn::Reduction* op_default = (ncnn::Reduction*)layer_default;

            fprintf_param_value(" 0=%d", operation)
            fprintf_param_value(" 1=%d", reduce_all)
            fprintf_param_value(" 2=%f", coeff)
            {
                if (!op->axes.empty()) fprintf_param_int_array(3, op->axes, pp);
            }
            fprintf_param_value(" 4=%d", keepdims)
        }
        else if (layer->type == "ReLU")
        {
            ncnn::ReLU* op = (ncnn::ReLU*)layer;
            ncnn::ReLU* op_default = (ncnn::ReLU*)layer_default;

            fprintf_param_value(" 0=%f", slope)
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

            fprintf_param_value(" 0=%f", scale_in)
            fprintf_param_value(" 1=%f", scale_out)
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
            fprintf_param_value(" 2=%f", spatial_scale)
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
            fprintf_param_value(" 2=%f", spatial_scale)
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
        else if (layer->type == "Threshold")
        {
            ncnn::Threshold* op = (ncnn::Threshold*)layer;
            ncnn::Threshold* op_default = (ncnn::Threshold*)layer_default;

            fprintf_param_value(" 0=%f", threshold)
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
            fprintf_param_value(" 2=%f", confidence_threshold)
            fprintf_param_value(" 3=%f", nms_threshold)
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
            fprintf_param_value(" 2=%f", confidence_threshold)
            fprintf_param_value(" 3=%f", nms_threshold)
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

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 6)
    {
        fprintf(stderr, "usage: %s [inparam] [inbin] [outparam] [outbin] [calibration table]\n", argv[0]);
        return -1;
    }

    const char* inparam = argv[1];
    const char* inbin = argv[2];
    const char* outparam = argv[3];
    const char* outbin = argv[4];
    const char* int8scale_table_path = argv[5];

    NetQuantize quantizer;

    // parse the calibration scale table
    if (int8scale_table_path)
    {
        bool s2 = read_int8scale_table(int8scale_table_path, quantizer.blob_int8scale_table, quantizer.weight_int8scale_table);
        if (!s2)
        {
            fprintf(stderr, "read_int8scale_table failed\n");
            return -1;
        }
    }

    quantizer.load_param(inparam);
    quantizer.load_model(inbin);

    quantizer.quantize_convolution();
    quantizer.quantize_convolutiondepthwise();
    quantizer.quantize_innerproduct();

    quantizer.save(outparam, outbin);

    return 0;
}
