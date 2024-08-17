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
#include "datareader.h"
#include "layer.h"
#include "layer_type.h"
#include "net.h"

// ncnn private header
#include "../modelwriter.h"

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

static bool read_int8scale_table(const char* filepath, std::map<std::string, ncnn::Mat>& blob_int8scale_table, std::map<std::string, ncnn::Mat>& weight_int8scale_table)
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

    std::vector<char> line(10240000);
    char* pch = NULL;
    size_t len = 0;

    while (!feof(fp))
    {
        char* s = fgets(line.data(), (int)line.size(), fp);
        if (!s)
            break;

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
            weight_int8scale_table[key_str] = ncnn::Mat((int)scales.size(), (void*)scales.data()).clone();
        }
        else
        {
            blob_int8scale_table[key_str] = ncnn::Mat((int)scales.size(), (void*)scales.data()).clone();
        }
        key_str.clear();
        scales.clear();
    }

    fclose(fp);

    return true;
}

class NetQuantize : public ModelWriter
{
public:
    NetQuantize();

    std::map<std::string, ncnn::Mat> blob_int8scale_table;
    std::map<std::string, ncnn::Mat> weight_int8scale_table;

public:
    int quantize_convolution();
    int quantize_convolutiondepthwise();
    int quantize_innerproduct();

    int quantize_rnn();
    int quantize_lstm();
    int quantize_gru();

    int fuse_requantize();
};

NetQuantize::NetQuantize()
    : ModelWriter()
{
}

int NetQuantize::quantize_convolution()
{
    const int layer_count = static_cast<int>(layers.size());
    for (int i = 0; i < layer_count; i++)
    {
        // find convolution layer
        if (layers[i]->type != "Convolution")
            continue;

        // find convolution layer
        std::map<std::string, ncnn::Mat>::iterator iter_data = blob_int8scale_table.find(layers[i]->name);
        if (iter_data == blob_int8scale_table.end())
            continue;

        char key[256];
        sprintf(key, "%s_param_0", layers[i]->name.c_str());

        std::map<std::string, ncnn::Mat>::iterator iter = weight_int8scale_table.find(key);
        if (iter == weight_int8scale_table.end())
        {
            fprintf(stderr, "this layer need to be quantized, but no scale param!\n");
            return -1;
        }

        // Convolution - quantize weight from fp32 to int8
        ncnn::Convolution* convolution = (ncnn::Convolution*)layers[i];

        ncnn::Mat bottom_blob_int8_scales = iter_data->second;
        ncnn::Mat weight_data_int8_scales = iter->second;

        fprintf(stderr, "quantize_convolution %s\n", convolution->name.c_str());

        {
            const int maxk = convolution->kernel_w * convolution->kernel_h;
            const int num_input = convolution->weight_data_size / convolution->num_output / maxk;

            ncnn::Mat weight_data_r2 = convolution->weight_data.reshape(maxk, num_input, convolution->num_output);

            ncnn::Mat weight_data_int8;

            ncnn::Option opt_q = opt;
            opt_q.blob_allocator = convolution->weight_data.allocator;
            opt_q.use_packing_layout = false;
            ncnn::quantize_to_int8(weight_data_r2, weight_data_int8, weight_data_int8_scales, opt_q);
            if (weight_data_int8.empty())
                return -100;

            convolution->weight_data = weight_data_int8.reshape(convolution->weight_data_size);
        }

        convolution->int8_scale_term = 2;
        convolution->weight_data_int8_scales = weight_data_int8_scales;
        convolution->bottom_blob_int8_scales = bottom_blob_int8_scales;
    }

    return 0;
}

int NetQuantize::quantize_convolutiondepthwise()
{
    const int layer_count = static_cast<int>(layers.size());
    for (int i = 0; i < layer_count; i++)
    {
        // find convolution layer
        if (layers[i]->type != "ConvolutionDepthWise")
            continue;

        // find convolutiondepthwise layer
        std::map<std::string, ncnn::Mat>::iterator iter_data = blob_int8scale_table.find(layers[i]->name);
        if (iter_data == blob_int8scale_table.end())
            continue;

        char key[256];
        sprintf(key, "%s_param_0", layers[i]->name.c_str());

        std::map<std::string, ncnn::Mat>::iterator iter = weight_int8scale_table.find(key);
        if (iter == weight_int8scale_table.end())
        {
            fprintf(stderr, "this layer need to be quantized, but no scale param!\n");
            return -1;
        }

        // Convolution - quantize weight from fp32 to int8
        ncnn::ConvolutionDepthWise* convdw = (ncnn::ConvolutionDepthWise*)layers[i];

        ncnn::Mat bottom_blob_int8_scales = iter_data->second;
        ncnn::Mat weight_data_int8_scales = iter->second;

        fprintf(stderr, "quantize_convolutiondepthwise %s\n", convdw->name.c_str());

        {
            ncnn::Mat int8_weight_data(convdw->weight_data_size, (size_t)1u);
            if (int8_weight_data.empty())
                return -100;

            const int weight_data_size_g = convdw->weight_data_size / convdw->group;

            for (int g = 0; g < convdw->group; g++)
            {
                ncnn::Option opt_q = opt;
                opt_q.blob_allocator = int8_weight_data.allocator;
                opt_q.use_packing_layout = false;

                const ncnn::Mat weight_data_g = convdw->weight_data.range(weight_data_size_g * g, weight_data_size_g);
                ncnn::Mat int8_weight_data_g = int8_weight_data.range(weight_data_size_g * g, weight_data_size_g);
                const ncnn::Mat weight_data_int8_scales_g = weight_data_int8_scales.range(g, 1);
                ncnn::quantize_to_int8(weight_data_g, int8_weight_data_g, weight_data_int8_scales_g, opt_q);
            }

            convdw->weight_data = int8_weight_data;
        }

        convdw->int8_scale_term = 1;
        convdw->weight_data_int8_scales = weight_data_int8_scales;
        convdw->bottom_blob_int8_scales = bottom_blob_int8_scales;
    }

    return 0;
}

int NetQuantize::quantize_innerproduct()
{
    const int layer_count = static_cast<int>(layers.size());
    for (int i = 0; i < layer_count; i++)
    {
        // find convolution layer
        if (layers[i]->type != "InnerProduct")
            continue;

        // find InnerProduct layer
        std::map<std::string, ncnn::Mat>::iterator iter_data = blob_int8scale_table.find(layers[i]->name);
        if (iter_data == blob_int8scale_table.end())
            continue;

        char key[256];
        sprintf(key, "%s_param_0", layers[i]->name.c_str());

        std::map<std::string, ncnn::Mat>::iterator iter = weight_int8scale_table.find(key);
        if (iter == weight_int8scale_table.end())
        {
            fprintf(stderr, "this layer need to be quantized, but no scale param!\n");
            return -1;
        }

        // InnerProduct - quantize weight from fp32 to int8
        ncnn::InnerProduct* fc = (ncnn::InnerProduct*)layers[i];

        ncnn::Mat bottom_blob_int8_scales = iter_data->second;
        ncnn::Mat weight_data_int8_scales = iter->second;

        fprintf(stderr, "quantize_innerproduct %s\n", fc->name.c_str());

        {
            const int num_input = fc->weight_data_size / fc->num_output;

            ncnn::Mat weight_data_r2 = fc->weight_data.reshape(num_input, fc->num_output);

            ncnn::Mat weight_data_int8;
            ncnn::Option opt_q = opt;
            opt_q.use_packing_layout = false;
            ncnn::quantize_to_int8(weight_data_r2, weight_data_int8, weight_data_int8_scales, opt_q);
            if (weight_data_int8.empty())
                return -100;

            fc->weight_data = weight_data_int8.reshape(fc->weight_data_size);
        }

        fc->int8_scale_term = 2;
        fc->weight_data_int8_scales = weight_data_int8_scales;
        fc->bottom_blob_int8_scales = bottom_blob_int8_scales;
    }

    return 0;
}

int NetQuantize::quantize_rnn()
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i]->type != "RNN")
            continue;

        // RNN - quantize weight from fp32 to int8
        ncnn::RNN* rnn = (ncnn::RNN*)layers[i];

        fprintf(stderr, "quantize_rnn %s\n", rnn->name.c_str());

        // TODO move to ncnn2table
        const int num_directions = rnn->direction == 2 ? 2 : 1;
        const int size = rnn->weight_data_size / num_directions / rnn->num_output;

        ncnn::Mat weight_xc_data_int8_scales(rnn->num_output * num_directions);
        ncnn::Mat weight_hc_data_int8_scales(rnn->num_output * num_directions);

        for (int d = 0; d < num_directions; d++)
        {
            for (int q = 0; q < rnn->num_output; q++)
            {
                {
                    const float* weight_xc_ptr = rnn->weight_xc_data.channel(d).row(q);
                    float absmax = 0.f;
                    for (int i = 0; i < size; i++)
                    {
                        absmax = std::max(absmax, (float)fabs(weight_xc_ptr[i]));
                    }
                    weight_xc_data_int8_scales[d * rnn->num_output + q] = 127 / absmax;
                }

                {
                    const float* weight_hc_ptr = rnn->weight_hc_data.channel(d).row(q);
                    float absmax = 0.f;
                    for (int i = 0; i < size; i++)
                    {
                        absmax = std::max(absmax, (float)fabs(weight_hc_ptr[i]));
                    }
                    weight_hc_data_int8_scales[d * rnn->num_output + q] = 127 / absmax;
                }
            }
        }

        {
            ncnn::Mat weight_xc_data_r2 = rnn->weight_xc_data.reshape(size, rnn->num_output * num_directions);

            ncnn::Mat weight_xc_data_int8;

            ncnn::Option opt_q = opt;
            opt_q.blob_allocator = rnn->weight_xc_data.allocator;
            opt_q.use_packing_layout = false;
            ncnn::quantize_to_int8(weight_xc_data_r2, weight_xc_data_int8, weight_xc_data_int8_scales, opt_q);
            if (weight_xc_data_int8.empty())
                return -100;

            rnn->weight_xc_data = weight_xc_data_int8.reshape(size * rnn->num_output * num_directions);
        }
        {
            ncnn::Mat weight_hc_data_r2 = rnn->weight_hc_data.reshape(rnn->num_output, rnn->num_output * num_directions);

            ncnn::Mat weight_hc_data_int8;

            ncnn::Option opt_q = opt;
            opt_q.blob_allocator = rnn->weight_hc_data.allocator;
            opt_q.use_packing_layout = false;
            ncnn::quantize_to_int8(weight_hc_data_r2, weight_hc_data_int8, weight_hc_data_int8_scales, opt_q);
            if (weight_hc_data_int8.empty())
                return -100;

            rnn->weight_hc_data = weight_hc_data_int8.reshape(rnn->num_output * rnn->num_output * num_directions);
        }

        rnn->int8_scale_term = 2;
        rnn->weight_xc_data_int8_scales = weight_xc_data_int8_scales;
        rnn->weight_hc_data_int8_scales = weight_hc_data_int8_scales;
    }

    return 0;
}

int NetQuantize::quantize_lstm()
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i]->type != "LSTM")
            continue;

        // LSTM - quantize weight from fp32 to int8
        ncnn::LSTM* lstm = (ncnn::LSTM*)layers[i];

        fprintf(stderr, "quantize_lstm %s\n", lstm->name.c_str());

        // TODO move to ncnn2table
        const int num_directions = lstm->direction == 2 ? 2 : 1;
        const int size = lstm->weight_data_size / num_directions / lstm->hidden_size / 4;

        ncnn::Mat weight_xc_data_int8_scales(lstm->hidden_size * 4 * num_directions);
        ncnn::Mat weight_hc_data_int8_scales(lstm->hidden_size * 4 * num_directions);

        for (int d = 0; d < num_directions; d++)
        {
            for (int q = 0; q < lstm->hidden_size * 4; q++)
            {
                {
                    const float* weight_xc_ptr = lstm->weight_xc_data.channel(d).row(q);
                    float absmax = 0.f;
                    for (int i = 0; i < size; i++)
                    {
                        absmax = std::max(absmax, (float)fabs(weight_xc_ptr[i]));
                    }
                    weight_xc_data_int8_scales[d * lstm->hidden_size * 4 + q] = 127 / absmax;
                }

                {
                    const float* weight_hc_ptr = lstm->weight_hc_data.channel(d).row(q);
                    float absmax = 0.f;
                    for (int i = 0; i < size; i++)
                    {
                        absmax = std::max(absmax, (float)fabs(weight_hc_ptr[i]));
                    }
                    weight_hc_data_int8_scales[d * lstm->hidden_size * 4 + q] = 127 / absmax;
                }
            }
        }

        {
            ncnn::Mat weight_xc_data_r2 = lstm->weight_xc_data.reshape(size, lstm->hidden_size * 4 * num_directions);

            ncnn::Mat weight_xc_data_int8;

            ncnn::Option opt_q = opt;
            opt_q.blob_allocator = lstm->weight_xc_data.allocator;
            opt_q.use_packing_layout = false;
            ncnn::quantize_to_int8(weight_xc_data_r2, weight_xc_data_int8, weight_xc_data_int8_scales, opt_q);
            if (weight_xc_data_int8.empty())
                return -100;

            lstm->weight_xc_data = weight_xc_data_int8.reshape(size * lstm->hidden_size * 4 * num_directions);
        }
        {
            ncnn::Mat weight_hc_data_r2 = lstm->weight_hc_data.reshape(lstm->num_output, lstm->hidden_size * 4 * num_directions);

            ncnn::Mat weight_hc_data_int8;

            ncnn::Option opt_q = opt;
            opt_q.blob_allocator = lstm->weight_hc_data.allocator;
            opt_q.use_packing_layout = false;
            ncnn::quantize_to_int8(weight_hc_data_r2, weight_hc_data_int8, weight_hc_data_int8_scales, opt_q);
            if (weight_hc_data_int8.empty())
                return -100;

            lstm->weight_hc_data = weight_hc_data_int8.reshape(lstm->num_output * lstm->hidden_size * 4 * num_directions);
        }

        lstm->int8_scale_term = 2;
        lstm->weight_xc_data_int8_scales = weight_xc_data_int8_scales;
        lstm->weight_hc_data_int8_scales = weight_hc_data_int8_scales;
    }

    return 0;
}

int NetQuantize::quantize_gru()
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i]->type != "GRU")
            continue;

        // GRU - quantize weight from fp32 to int8
        ncnn::GRU* gru = (ncnn::GRU*)layers[i];

        fprintf(stderr, "quantize_gru %s\n", gru->name.c_str());

        // TODO move to ncnn2table
        const int num_directions = gru->direction == 2 ? 2 : 1;
        const int size = gru->weight_data_size / num_directions / gru->num_output / 3;

        ncnn::Mat weight_xc_data_int8_scales(gru->num_output * 3 * num_directions);
        ncnn::Mat weight_hc_data_int8_scales(gru->num_output * 3 * num_directions);

        for (int d = 0; d < num_directions; d++)
        {
            for (int q = 0; q < gru->num_output * 3; q++)
            {
                {
                    const float* weight_xc_ptr = gru->weight_xc_data.channel(d).row(q);
                    float absmax = 0.f;
                    for (int i = 0; i < size; i++)
                    {
                        absmax = std::max(absmax, (float)fabs(weight_xc_ptr[i]));
                    }
                    weight_xc_data_int8_scales[d * gru->num_output * 3 + q] = 127 / absmax;
                }

                {
                    const float* weight_hc_ptr = gru->weight_hc_data.channel(d).row(q);
                    float absmax = 0.f;
                    for (int i = 0; i < size; i++)
                    {
                        absmax = std::max(absmax, (float)fabs(weight_hc_ptr[i]));
                    }
                    weight_hc_data_int8_scales[d * gru->num_output * 3 + q] = 127 / absmax;
                }
            }
        }

        {
            ncnn::Mat weight_xc_data_r2 = gru->weight_xc_data.reshape(size, gru->num_output * 3 * num_directions);

            ncnn::Mat weight_xc_data_int8;

            ncnn::Option opt_q = opt;
            opt_q.blob_allocator = gru->weight_xc_data.allocator;
            opt_q.use_packing_layout = false;
            ncnn::quantize_to_int8(weight_xc_data_r2, weight_xc_data_int8, weight_xc_data_int8_scales, opt_q);
            if (weight_xc_data_int8.empty())
                return -100;

            gru->weight_xc_data = weight_xc_data_int8.reshape(size * gru->num_output * 3 * num_directions);
        }
        {
            ncnn::Mat weight_hc_data_r2 = gru->weight_hc_data.reshape(gru->num_output, gru->num_output * 3 * num_directions);

            ncnn::Mat weight_hc_data_int8;

            ncnn::Option opt_q = opt;
            opt_q.blob_allocator = gru->weight_hc_data.allocator;
            opt_q.use_packing_layout = false;
            ncnn::quantize_to_int8(weight_hc_data_r2, weight_hc_data_int8, weight_hc_data_int8_scales, opt_q);
            if (weight_hc_data_int8.empty())
                return -100;

            gru->weight_hc_data = weight_hc_data_int8.reshape(gru->num_output * gru->num_output * 3 * num_directions);
        }

        gru->int8_scale_term = 2;
        gru->weight_xc_data_int8_scales = weight_xc_data_int8_scales;
        gru->weight_hc_data_int8_scales = weight_hc_data_int8_scales;
    }

    return 0;
}

int NetQuantize::fuse_requantize()
{
    const size_t layer_count = layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Convolution" && layers[i]->type != "ConvolutionDepthWise")
            continue;

        // Convolution/ConvolutionDepthWise - Convolution/ConvolutionDepthWise
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "Convolution" && layers[j]->type != "ConvolutionDepthWise")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        // fuse requantize
        fprintf(stderr, "fuse_requantize %s %s\n", layers[i]->name.c_str(), layers[j]->name.c_str());

        if (layers[i]->type == "Convolution" && layers[j]->type == "Convolution")
        {
            ncnn::Convolution* convolution1 = (ncnn::Convolution*)layers[i];
            ncnn::Convolution* convolution2 = (ncnn::Convolution*)layers[j];

            if (convolution1->weight_data.elemsize != 1u || convolution2->weight_data.elemsize != 1u)
                continue;

            convolution1->int8_scale_term += 100;
            convolution1->top_blob_int8_scales = convolution2->bottom_blob_int8_scales;
        }
        if (layers[i]->type == "Convolution" && layers[j]->type == "ConvolutionDepthWise")
        {
            ncnn::Convolution* convolution1 = (ncnn::Convolution*)layers[i];
            ncnn::ConvolutionDepthWise* convolution2 = (ncnn::ConvolutionDepthWise*)layers[j];

            if (convolution1->weight_data.elemsize != 1u || convolution2->weight_data.elemsize != 1u)
                continue;

            convolution1->int8_scale_term += 100;
            convolution1->top_blob_int8_scales = convolution2->bottom_blob_int8_scales;
        }
        if (layers[i]->type == "ConvolutionDepthWise" && layers[j]->type == "Convolution")
        {
            ncnn::ConvolutionDepthWise* convolution1 = (ncnn::ConvolutionDepthWise*)layers[i];
            ncnn::Convolution* convolution2 = (ncnn::Convolution*)layers[j];

            if (convolution1->weight_data.elemsize != 1u || convolution2->weight_data.elemsize != 1u)
                continue;

            convolution1->int8_scale_term += 100;
            convolution1->top_blob_int8_scales = convolution2->bottom_blob_int8_scales;
        }
        if (layers[i]->type == "ConvolutionDepthWise" && layers[j]->type == "ConvolutionDepthWise")
        {
            ncnn::ConvolutionDepthWise* convolution1 = (ncnn::ConvolutionDepthWise*)layers[i];
            ncnn::ConvolutionDepthWise* convolution2 = (ncnn::ConvolutionDepthWise*)layers[j];

            if (convolution1->weight_data.elemsize != 1u || convolution2->weight_data.elemsize != 1u)
                continue;

            convolution1->int8_scale_term += 100;
            convolution1->top_blob_int8_scales = convolution2->bottom_blob_int8_scales;
        }
    }

    for (size_t i = 0; i < layer_count; i++)
    {
        if (layers[i]->type != "Convolution" && layers[i]->type != "ConvolutionDepthWise")
            continue;

        // Convolution/ConvolutionDepthWise - Split - Convolution/ConvolutionDepthWise
        int top_blob_index = layers[i]->tops[0];

        size_t j = i + 1;
        for (; j < layer_count; j++)
        {
            if (layers[j]->type != "Split")
                continue;

            if (layers[j]->bottoms.size() != 1)
                continue;

            if (layers[j]->bottoms[0] == top_blob_index)
                break;
        }

        if (j == layer_count)
            continue;

        ncnn::Split* split = (ncnn::Split*)layers[j];

        bool all_conv = true;
        for (size_t p = 0; p < split->tops.size(); p++)
        {
            int split_top_blob_index = split->tops[p];

            size_t k = j + 1;
            for (; k < layer_count; k++)
            {
                if (layers[k]->type != "Convolution" && layers[k]->type != "ConvolutionDepthWise")
                    continue;

                if (layers[k]->bottoms.size() != 1)
                    continue;

                if (layers[k]->bottoms[0] == split_top_blob_index)
                    break;
            }

            if (k == layer_count)
            {
                all_conv = false;
                break;
            }

            if (layers[k]->type == "Convolution")
            {
                ncnn::Convolution* convolution = (ncnn::Convolution*)layers[k];
                if (convolution->weight_data.elemsize != 1u)
                {
                    all_conv = false;
                    break;
                }
            }
            if (layers[k]->type == "ConvolutionDepthWise")
            {
                ncnn::ConvolutionDepthWise* convolution = (ncnn::ConvolutionDepthWise*)layers[k];
                if (convolution->weight_data.elemsize != 1u)
                {
                    all_conv = false;
                    break;
                }
            }
        }

        if (!all_conv)
            continue;

        j = blobs[split->tops[0]].consumer;

        // fuse requantize
        fprintf(stderr, "fuse_requantize %s %s\n", layers[i]->name.c_str(), split->name.c_str());

        if (layers[i]->type == "Convolution" && layers[j]->type == "Convolution")
        {
            ncnn::Convolution* convolution1 = (ncnn::Convolution*)layers[i];
            ncnn::Convolution* convolution2 = (ncnn::Convolution*)layers[j];

            if (convolution1->weight_data.elemsize != 1u || convolution2->weight_data.elemsize != 1u)
                continue;

            convolution1->int8_scale_term += 100;
            convolution1->top_blob_int8_scales = convolution2->bottom_blob_int8_scales;
        }
        if (layers[i]->type == "Convolution" && layers[j]->type == "ConvolutionDepthWise")
        {
            ncnn::Convolution* convolution1 = (ncnn::Convolution*)layers[i];
            ncnn::ConvolutionDepthWise* convolution2 = (ncnn::ConvolutionDepthWise*)layers[j];

            if (convolution1->weight_data.elemsize != 1u || convolution2->weight_data.elemsize != 1u)
                continue;

            convolution1->int8_scale_term += 100;
            convolution1->top_blob_int8_scales = convolution2->bottom_blob_int8_scales;
        }
        if (layers[i]->type == "ConvolutionDepthWise" && layers[j]->type == "Convolution")
        {
            ncnn::ConvolutionDepthWise* convolution1 = (ncnn::ConvolutionDepthWise*)layers[i];
            ncnn::Convolution* convolution2 = (ncnn::Convolution*)layers[j];

            if (convolution1->weight_data.elemsize != 1u || convolution2->weight_data.elemsize != 1u)
                continue;

            convolution1->int8_scale_term += 100;
            convolution1->top_blob_int8_scales = convolution2->bottom_blob_int8_scales;
        }
        if (layers[i]->type == "ConvolutionDepthWise" && layers[j]->type == "ConvolutionDepthWise")
        {
            ncnn::ConvolutionDepthWise* convolution1 = (ncnn::ConvolutionDepthWise*)layers[i];
            ncnn::ConvolutionDepthWise* convolution2 = (ncnn::ConvolutionDepthWise*)layers[j];

            if (convolution1->weight_data.elemsize != 1u || convolution2->weight_data.elemsize != 1u)
                continue;

            convolution1->int8_scale_term += 100;
            convolution1->top_blob_int8_scales = convolution2->bottom_blob_int8_scales;
        }
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 5 && argc != 6)
    {
        fprintf(stderr, "usage: %s [inparam] [inbin] [outparam] [outbin] [calibration table]\n", argv[0]);
        return -1;
    }

    const char* inparam = argv[1];
    const char* inbin = argv[2];
    const char* outparam = argv[3];
    const char* outbin = argv[4];
    const char* int8scale_table_path = argc == 6 ? argv[5] : NULL;

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
    if (strcmp(inbin, "null") == 0)
    {
        DataReaderFromEmpty dr;
        quantizer.load_model(dr);
        quantizer.gen_random_weight = true;
    }
    else
        quantizer.load_model(inbin);

    quantizer.quantize_convolution();
    quantizer.quantize_convolutiondepthwise();
    quantizer.quantize_innerproduct();

    quantizer.quantize_rnn();
    quantizer.quantize_lstm();
    quantizer.quantize_gru();

    quantizer.fuse_requantize();

    quantizer.save(outparam, outbin);

    return 0;
}
