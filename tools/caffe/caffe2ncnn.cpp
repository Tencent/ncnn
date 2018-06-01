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

#include <stdio.h>
#include <limits.h>
#include <math.h>

#include <fstream>
#include <set>
#include <limits>
#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>

#include "caffe.pb.h"


static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n-1) & -n;
}

// convert float to half precision floating point
static unsigned short float2half(float value)
{
    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;

    tmp.f = value;

    // 1 : 8 : 23
    unsigned short sign = (tmp.u & 0x80000000) >> 31;
    unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
    unsigned int significand = tmp.u & 0x7FFFFF;

//     fprintf(stderr, "%d %d %d\n", sign, exponent, significand);

    // 1 : 5 : 10
    unsigned short fp16;
    if (exponent == 0)
    {
        // zero or denormal, always underflow
        fp16 = (sign << 15) | (0x00 << 10) | 0x00;
    }
    else if (exponent == 0xFF)
    {
        // infinity or NaN
        fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    }
    else
    {
        // normalized
        short newexp = exponent + (- 127 + 15);
        if (newexp >= 31)
        {
            // overflow, return infinity
            fp16 = (sign << 15) | (0x1F << 10) | 0x00;
        }
        else if (newexp <= 0)
        {
            // underflow
            if (newexp >= -10)
            {
                // denormal half-precision
                unsigned short sig = (significand | 0x800000) >> (14 - newexp);
                fp16 = (sign << 15) | (0x00 << 10) | sig;
            }
            else
            {
                // underflow
                fp16 = (sign << 15) | (0x00 << 10) | 0x00;
            }
        }
        else
        {
            fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
        }
    }

    return fp16;
}

static int quantize_weight(float *data, size_t data_length, std::vector<unsigned short>& float16_weights)
{
    float16_weights.resize(data_length);

    for (size_t i = 0; i < data_length; i++)
    {
        float f = data[i];

        unsigned short fp16 = float2half(f);

        float16_weights[i] = fp16;
    }

    // magic tag for half-precision floating point
    return 0x01306B47;
}

static bool quantize_weight(float *data, size_t data_length, int quantize_level, std::vector<float> &quantize_table, std::vector<unsigned char> &quantize_index) {

    assert(quantize_level != 0);
    assert(data != NULL);
    assert(data_length > 0);

    if (data_length < static_cast<size_t>(quantize_level)) {
        fprintf(stderr, "No need quantize,because: data_length < quantize_level");
        return false;
    }

    quantize_table.reserve(quantize_level);
    quantize_index.reserve(data_length);

    // 1. Find min and max value
    float max_value = std::numeric_limits<float>::min();
    float min_value = std::numeric_limits<float>::max();

    for (size_t i = 0; i < data_length; ++i)
    {
        if (max_value < data[i]) max_value = data[i];
        if (min_value > data[i]) min_value = data[i];
    }
    float strides = (max_value - min_value) / quantize_level;

    // 2. Generate quantize table
    for (int i = 0; i < quantize_level; ++i)
    {
        quantize_table.push_back(min_value + i * strides);
    }

    // 3. Align data to the quantized value
    for (size_t i = 0; i < data_length; ++i)
    {
        size_t table_index = int((data[i] - min_value) / strides);
        table_index = std::min<float>(table_index, quantize_level - 1);

        float low_value  = quantize_table[table_index];
        float high_value = low_value + strides;

        // find a nearest value between low and high value.
        float targetValue = data[i] - low_value < high_value - data[i] ? low_value : high_value;

        table_index = int((targetValue - min_value) / strides);
        table_index = std::min<float>(table_index, quantize_level - 1);
        quantize_index.push_back(table_index);
    }

    return true;
}

static bool read_proto_from_text(const char* filepath, google::protobuf::Message* message)
{
    std::ifstream fs(filepath, std::ifstream::in);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    bool success = google::protobuf::TextFormat::Parse(&input, message);

    fs.close();

    return success;
}

static bool read_proto_from_binary(const char* filepath, google::protobuf::Message* message)
{
    std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
    if (!fs.is_open())
    {
        fprintf(stderr, "open failed %s\n", filepath);
        return false;
    }

    google::protobuf::io::IstreamInputStream input(&fs);
    google::protobuf::io::CodedInputStream codedstr(&input);

    codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);

    bool success = message->ParseFromCodedStream(&codedstr);

    fs.close();

    return success;
}

int main(int argc, char** argv)
{
    if (!(argc == 3 || argc == 5 || argc == 6))
    {
        fprintf(stderr, "Usage: %s [caffeproto] [caffemodel] [ncnnproto] [ncnnbin] [quantizelevel]\n", argv[0]);
        return -1;
    }

    const char* caffeproto = argv[1];
    const char* caffemodel = argv[2];
    const char* ncnn_prototxt = argc >= 5 ? argv[3] : "ncnn.proto";
    const char* ncnn_modelbin = argc >= 5 ? argv[4] : "ncnn.bin";
    const char* quantize_param = argc == 6 ? argv[5] : "0";
    int quantize_level = atoi(quantize_param);

    if (quantize_level != 0 && quantize_level != 256 && quantize_level != 65536) {
        fprintf(stderr, "%s: only support quantize level = 0, 256, or 65536", argv[0]);
        return -1;
    }

    caffe::NetParameter proto;
    caffe::NetParameter net;

    // load
    bool s0 = read_proto_from_text(caffeproto, &proto);
    if (!s0)
    {
        fprintf(stderr, "read_proto_from_text failed\n");
        return -1;
    }

    bool s1 = read_proto_from_binary(caffemodel, &net);
    if (!s1)
    {
        fprintf(stderr, "read_proto_from_binary failed\n");
        return -1;
    }

    FILE* pp = fopen(ncnn_prototxt, "wb");
    FILE* bp = fopen(ncnn_modelbin, "wb");

    // magic
    fprintf(pp, "7767517\n");

    // rename mapping for identical bottom top style
    std::map<std::string, std::string> blob_name_decorated;

    // bottom blob reference
    std::map<std::string, int> bottom_reference;

    // global definition line
    // [layer count] [blob count]
    int layer_count = proto.layer_size();
    std::set<std::string> blob_names;
    for (int i=0; i<layer_count; i++)
    {
        const caffe::LayerParameter& layer = proto.layer(i);

        for (int j=0; j<layer.bottom_size(); j++)
        {
            std::string blob_name = layer.bottom(j);
            if (blob_name_decorated.find(blob_name) != blob_name_decorated.end())
            {
                blob_name = blob_name_decorated[blob_name];
            }

            blob_names.insert(blob_name);

            if (bottom_reference.find(blob_name) == bottom_reference.end())
            {
                bottom_reference[blob_name] = 1;
            }
            else
            {
                bottom_reference[blob_name] = bottom_reference[blob_name] + 1;
            }
        }

        if (layer.bottom_size() == 1 && layer.top_size() == 1 && layer.bottom(0) == layer.top(0))
        {
            std::string blob_name = layer.top(0) + "_" + layer.name();
            blob_name_decorated[layer.top(0)] = blob_name;
            blob_names.insert(blob_name);
        }
        else
        {
            for (int j=0; j<layer.top_size(); j++)
            {
                std::string blob_name = layer.top(j);
                blob_names.insert(blob_name);
            }
        }
    }
    // remove bottom_reference entry with reference equals to one
    int splitncnn_blob_count = 0;
    std::map<std::string, int>::iterator it = bottom_reference.begin();
    while (it != bottom_reference.end())
    {
        if (it->second == 1)
        {
            bottom_reference.erase(it++);
        }
        else
        {
            splitncnn_blob_count += it->second;
//             fprintf(stderr, "%s %d\n", it->first.c_str(), it->second);
            ++it;
        }
    }
    fprintf(pp, "%lu %lu\n", layer_count + bottom_reference.size(), blob_names.size() + splitncnn_blob_count);

    // populate
    blob_name_decorated.clear();
    int internal_split = 0;
    for (int i=0; i<layer_count; i++)
    {
        const caffe::LayerParameter& layer = proto.layer(i);

        // layer definition line, repeated
        // [type] [name] [bottom blob count] [top blob count] [bottom blobs] [top blobs] [layer specific params]
        if (layer.type() == "Convolution")
        {
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            if (convolution_param.group() != 1)
                fprintf(pp, "%-16s", "ConvolutionDepthWise");
            else
                fprintf(pp, "%-16s", "Convolution");
        }
        else if (layer.type() == "ConvolutionDepthwise" || layer.type() == "DepthwiseConvolution")
        {
            fprintf(pp, "%-16s", "ConvolutionDepthWise");
        }
        else if (layer.type() == "Deconvolution")
        {
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            if (convolution_param.group() != 1)
                fprintf(pp, "%-16s", "DeconvolutionDepthWise");
            else
                fprintf(pp, "%-16s", "Deconvolution");
        }
        else if (layer.type() == "MemoryData")
        {
            fprintf(pp, "%-16s", "Input");
        }
        else if (layer.type() == "Python")
        {
            const caffe::PythonParameter& python_param = layer.python_param();
            std::string python_layer_name = python_param.layer();
            if (python_layer_name == "ProposalLayer")
                fprintf(pp, "%-16s", "Proposal");
            else
                fprintf(pp, "%-16s", python_layer_name.c_str());
        }
        else
        {
            fprintf(pp, "%-16s", layer.type().c_str());
        }
        fprintf(pp, " %-16s %d %d", layer.name().c_str(), layer.bottom_size(), layer.top_size());

        for (int j=0; j<layer.bottom_size(); j++)
        {
            std::string blob_name = layer.bottom(j);
            if (blob_name_decorated.find(layer.bottom(j)) != blob_name_decorated.end())
            {
                blob_name = blob_name_decorated[layer.bottom(j)];
            }

            if (bottom_reference.find(blob_name) != bottom_reference.end())
            {
                int refidx = bottom_reference[blob_name] - 1;
                bottom_reference[blob_name] = refidx;

                char splitsuffix[256];
                sprintf(splitsuffix, "_splitncnn_%d", refidx);
                blob_name = blob_name + splitsuffix;
            }

            fprintf(pp, " %s", blob_name.c_str());
        }

        // decorated
        if (layer.bottom_size() == 1 && layer.top_size() == 1 && layer.bottom(0) == layer.top(0))
        {
            std::string blob_name = layer.top(0) + "_" + layer.name();
            blob_name_decorated[layer.top(0)] = blob_name;

            fprintf(pp, " %s", blob_name.c_str());
        }
        else
        {
            for (int j=0; j<layer.top_size(); j++)
            {
                std::string blob_name = layer.top(j);
                fprintf(pp, " %s", blob_name.c_str());
            }
        }

        // find blob binary by layer name
        int netidx;
        for (netidx=0; netidx<net.layer_size(); netidx++)
        {
            if (net.layer(netidx).name() == layer.name())
            {
                break;
            }
        }

        // layer specific params
        if (layer.type() == "BatchNorm")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::BlobProto& mean_blob = binlayer.blobs(0);
            const caffe::BlobProto& var_blob = binlayer.blobs(1);
            fprintf(pp, " 0=%d", (int)mean_blob.data_size());

            const caffe::BatchNormParameter& batch_norm_param = layer.batch_norm_param();
            float eps = batch_norm_param.eps();

            std::vector<float> ones(mean_blob.data_size(), 1.f);
            fwrite(ones.data(), sizeof(float), ones.size(), bp);// slope

            if (binlayer.blobs_size() < 3)
            {
                fwrite(mean_blob.data().data(), sizeof(float), mean_blob.data_size(), bp);
                float tmp;
                for (int j=0; j<var_blob.data_size(); j++)
                {
                    tmp = var_blob.data().data()[j] + eps;
                    fwrite(&tmp, sizeof(float), 1, bp);
                }
            }
            else
            {
                float scale_factor = binlayer.blobs(2).data().data()[0] == 0 ? 0 : 1 / binlayer.blobs(2).data().data()[0];
                // premultiply scale_factor to mean and variance
                float tmp;
                for (int j=0; j<mean_blob.data_size(); j++)
                {
                    tmp = mean_blob.data().data()[j] * scale_factor;
                    fwrite(&tmp, sizeof(float), 1, bp);
                }
                for (int j=0; j<var_blob.data_size(); j++)
                {
                    tmp = var_blob.data().data()[j] * scale_factor + eps;
                    fwrite(&tmp, sizeof(float), 1, bp);
                }
            }

            std::vector<float> zeros(mean_blob.data_size(), 0.f);
            fwrite(zeros.data(), sizeof(float), zeros.size(), bp);// bias
        }
        else if (layer.type() == "Concat")
        {
            const caffe::ConcatParameter& concat_param = layer.concat_param();
            int dim = concat_param.axis() - 1;
            fprintf(pp, " 0=%d", dim);
        }
        else if (layer.type() == "Convolution" || layer.type() == "ConvolutionDepthwise" || layer.type() == "DepthwiseConvolution")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::BlobProto& weight_blob = binlayer.blobs(0);
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            fprintf(pp, " 0=%d", convolution_param.num_output());
            if (convolution_param.has_kernel_w() && convolution_param.has_kernel_h())
            {
                fprintf(pp, " 1=%d", convolution_param.kernel_w());
                fprintf(pp, " 11=%d", convolution_param.kernel_h());
            }
            else
            {
                fprintf(pp, " 1=%d", convolution_param.kernel_size(0));
            }
            fprintf(pp, " 2=%d", convolution_param.dilation_size() != 0 ? convolution_param.dilation(0) : 1);
            if (convolution_param.has_stride_w() && convolution_param.has_stride_h())
            {
                fprintf(pp, " 3=%d", convolution_param.stride_w());
                fprintf(pp, " 13=%d", convolution_param.stride_h());
            }
            else
            {
                fprintf(pp, " 3=%d", convolution_param.stride_size() != 0 ? convolution_param.stride(0) : 1);
            }
            if (convolution_param.has_pad_w() && convolution_param.has_pad_h())
            {
                fprintf(pp, " 4=%d", convolution_param.pad_w());
                fprintf(pp, " 14=%d", convolution_param.pad_h());
            }
            else
            {
                fprintf(pp, " 4=%d", convolution_param.pad_size() != 0 ? convolution_param.pad(0) : 0);
            }
            fprintf(pp, " 5=%d", convolution_param.bias_term());
            fprintf(pp, " 6=%d", weight_blob.data_size());

            if (layer.type() == "ConvolutionDepthwise")
            {
                fprintf(pp, " 7=%d", convolution_param.num_output());
            }
            else if (convolution_param.group() != 1)
            {
                fprintf(pp, " 7=%d", convolution_param.group());
            }

            for (int j = 0; j < binlayer.blobs_size(); j++)
            {
                int quantize_tag = 0;
                const caffe::BlobProto& blob = binlayer.blobs(j);

                std::vector<float> quantize_table;
                std::vector<unsigned char> quantize_index;

                std::vector<unsigned short> float16_weights;

                // we will not quantize the bias values
                if (j == 0 && quantize_level != 0)
                {
                    if (quantize_level == 256)
                    {
                    quantize_tag = quantize_weight((float *)blob.data().data(), blob.data_size(), quantize_level, quantize_table, quantize_index);
                    }
                    else if (quantize_level == 65536)
                    {
                    quantize_tag = quantize_weight((float *)blob.data().data(), blob.data_size(), float16_weights);
                    }
                }

                // write quantize tag first
                if (j == 0)
                    fwrite(&quantize_tag, sizeof(int), 1, bp);

                if (quantize_tag)
                {
                    int p0 = ftell(bp);
                    if (quantize_level == 256)
                    {
                    // write quantize table and index
                    fwrite(quantize_table.data(), sizeof(float), quantize_table.size(), bp);
                    fwrite(quantize_index.data(), sizeof(unsigned char), quantize_index.size(), bp);
                    }
                    else if (quantize_level == 65536)
                    {
                    fwrite(float16_weights.data(), sizeof(unsigned short), float16_weights.size(), bp);
                    }
                    // padding to 32bit align
                    int nwrite = ftell(bp) - p0;
                    int nalign = alignSize(nwrite, 4);
                    unsigned char padding[4] = {0x00, 0x00, 0x00, 0x00};
                    fwrite(padding, sizeof(unsigned char), nalign - nwrite, bp);
                }
                else
                {
                    // write original data
                    fwrite(blob.data().data(), sizeof(float), blob.data_size(), bp);
                }
            }

        }
        else if (layer.type() == "Crop")
        {
            const caffe::CropParameter& crop_param = layer.crop_param();
            int num_offset = crop_param.offset_size();
            int woffset = (num_offset == 2) ? crop_param.offset(0) : 0;
            int hoffset = (num_offset == 2) ? crop_param.offset(1) : 0;
            fprintf(pp, " 0=%d", woffset);
            fprintf(pp, " 1=%d", hoffset);
        }
        else if (layer.type() == "Deconvolution")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::BlobProto& weight_blob = binlayer.blobs(0);
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            fprintf(pp, " 0=%d", convolution_param.num_output());
            if (convolution_param.has_kernel_w() && convolution_param.has_kernel_h())
            {
                fprintf(pp, " 1=%d", convolution_param.kernel_w());
                fprintf(pp, " 11=%d", convolution_param.kernel_h());
            }
            else
            {
                fprintf(pp, " 1=%d", convolution_param.kernel_size(0));
            }
            fprintf(pp, " 2=%d", convolution_param.dilation_size() != 0 ? convolution_param.dilation(0) : 1);
            if (convolution_param.has_stride_w() && convolution_param.has_stride_h())
            {
                fprintf(pp, " 3=%d", convolution_param.stride_w());
                fprintf(pp, " 13=%d", convolution_param.stride_h());
            }
            else
            {
                fprintf(pp, " 3=%d", convolution_param.stride_size() != 0 ? convolution_param.stride(0) : 1);
            }
            if (convolution_param.has_pad_w() && convolution_param.has_pad_h())
            {
                fprintf(pp, " 4=%d", convolution_param.pad_w());
                fprintf(pp, " 14=%d", convolution_param.pad_h());
            }
            else
            {
                fprintf(pp, " 4=%d", convolution_param.pad_size() != 0 ? convolution_param.pad(0) : 0);
            }
            fprintf(pp, " 5=%d", convolution_param.bias_term());
            fprintf(pp, " 6=%d", weight_blob.data_size());

            int group = convolution_param.group();
            if (group != 1)
            {
                fprintf(pp, " 7=%d", group);
            }

            int quantized_weight = 0;
            fwrite(&quantized_weight, sizeof(int), 1, bp);

            int maxk = 0;
            if (convolution_param.has_kernel_w() && convolution_param.has_kernel_h())
            {
                maxk = convolution_param.kernel_w() * convolution_param.kernel_h();
            }
            else
            {
                maxk = convolution_param.kernel_size(0) * convolution_param.kernel_size(0);
            }
            for (int g=0; g<group; g++)
            {
            // reorder weight from inch-outch to outch-inch
            int num_output = convolution_param.num_output() / group;
            int num_input = weight_blob.data_size() / maxk / num_output / group;
            const float* weight_data_ptr = weight_blob.data().data() + g * maxk * num_output * num_input;
            for (int k=0; k<num_output; k++)
            {
                for (int j=0; j<num_input; j++)
                {
                    fwrite(weight_data_ptr + (j*num_output + k) * maxk, sizeof(float), maxk, bp);
                }
            }
            }

            for (int j=1; j<binlayer.blobs_size(); j++)
            {
                const caffe::BlobProto& blob = binlayer.blobs(j);
                fwrite(blob.data().data(), sizeof(float), blob.data_size(), bp);
            }
        }
        else if (layer.type() == "DetectionOutput")
        {
            const caffe::DetectionOutputParameter& detection_output_param = layer.detection_output_param();
            const caffe::NonMaximumSuppressionParameter& nms_param = detection_output_param.nms_param();
            fprintf(pp, " 0=%d", detection_output_param.num_classes());
            fprintf(pp, " 1=%f", nms_param.nms_threshold());
            fprintf(pp, " 2=%d", nms_param.top_k());
            fprintf(pp, " 3=%d", detection_output_param.keep_top_k());
            fprintf(pp, " 4=%f", detection_output_param.confidence_threshold());
        }
        else if (layer.type() == "Dropout")
        {
            const caffe::DropoutParameter& dropout_param = layer.dropout_param();
            if (dropout_param.has_scale_train() && !dropout_param.scale_train())
            {
                float scale = 1.f - dropout_param.dropout_ratio();
                fprintf(pp, " 0=%f", scale);
            }
        }
        else if (layer.type() == "Eltwise")
        {
            const caffe::EltwiseParameter& eltwise_param = layer.eltwise_param();
            int coeff_size = eltwise_param.coeff_size();
            fprintf(pp, " 0=%d", (int)eltwise_param.operation());
            fprintf(pp, " -23301=%d", coeff_size);
            for (int j=0; j<coeff_size; j++)
            {
                fprintf(pp, ",%f", eltwise_param.coeff(j));
            }
        }
        else if (layer.type() == "ELU")
        {
            const caffe::ELUParameter& elu_param = layer.elu_param();
            fprintf(pp, " 0=%f", elu_param.alpha());
        }
        else if (layer.type() == "InnerProduct")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::BlobProto& weight_blob = binlayer.blobs(0);
            const caffe::InnerProductParameter& inner_product_param = layer.inner_product_param();
            fprintf(pp, " 0=%d", inner_product_param.num_output());
            fprintf(pp, " 1=%d", inner_product_param.bias_term());
            fprintf(pp, " 2=%d", weight_blob.data_size());

            for (int j=0; j<binlayer.blobs_size(); j++)
            {
                int quantize_tag = 0;
                const caffe::BlobProto& blob = binlayer.blobs(j);

                std::vector<float> quantize_table;
                std::vector<unsigned char> quantize_index;

                std::vector<unsigned short> float16_weights;

                // we will not quantize the bias values
                if (j == 0 && quantize_level != 0)
                {
                    if (quantize_level == 256)
                    {
                    quantize_tag = quantize_weight((float *)blob.data().data(), blob.data_size(), quantize_level, quantize_table, quantize_index);
                    }
                    else if (quantize_level == 65536)
                    {
                    quantize_tag = quantize_weight((float *)blob.data().data(), blob.data_size(), float16_weights);
                    }
                }

                // write quantize tag first
                if (j == 0)
                    fwrite(&quantize_tag, sizeof(int), 1, bp);

                if (quantize_tag)
				{
                    int p0 = ftell(bp);
                    if (quantize_level == 256)
                    {
                    // write quantize table and index
                    fwrite(quantize_table.data(), sizeof(float), quantize_table.size(), bp);
                    fwrite(quantize_index.data(), sizeof(unsigned char), quantize_index.size(), bp);
                    }
                    else if (quantize_level == 65536)
                    {
                    fwrite(float16_weights.data(), sizeof(unsigned short), float16_weights.size(), bp);
                    }
                    // padding to 32bit align
                    int nwrite = ftell(bp) - p0;
                    int nalign = alignSize(nwrite, 4);
                    unsigned char padding[4] = {0x00, 0x00, 0x00, 0x00};
                    fwrite(padding, sizeof(unsigned char), nalign - nwrite, bp);
                }
                else
				{
                    // write original data
                    fwrite(blob.data().data(), sizeof(float), blob.data_size(), bp);
                }
            }
        }
        else if (layer.type() == "Input")
        {
            const caffe::InputParameter& input_param = layer.input_param();
            const caffe::BlobShape& bs = input_param.shape(0);
            if (bs.dim_size() == 4)
            {
                fprintf(pp, " 0=%ld", bs.dim(3));
                fprintf(pp, " 1=%ld", bs.dim(2));
                fprintf(pp, " 2=%ld", bs.dim(1));
            }
            else if (bs.dim_size() == 3)
            {
                fprintf(pp, " 0=%ld", bs.dim(2));
                fprintf(pp, " 1=%ld", bs.dim(1));
                fprintf(pp, " 2=-233");
            }
            else if (bs.dim_size() == 2)
            {
                fprintf(pp, " 0=%ld", bs.dim(1));
                fprintf(pp, " 1=-233");
                fprintf(pp, " 2=-233");
            }
        }
        else if (layer.type() == "Interp")
        {
            const caffe::InterpParameter& interp_param = layer.interp_param();
            fprintf(pp, " 0=%d", 2);
            fprintf(pp, " 1=%f", (float)interp_param.zoom_factor());
            fprintf(pp, " 2=%f", (float)interp_param.zoom_factor());
            fprintf(pp, " 3=%d", interp_param.height());
            fprintf(pp, " 4=%d", interp_param.width());
        }
        else if (layer.type() == "LRN")
        {
            const caffe::LRNParameter& lrn_param = layer.lrn_param();
            fprintf(pp, " 0=%d", lrn_param.norm_region());
            fprintf(pp, " 1=%d", lrn_param.local_size());
            fprintf(pp, " 2=%f", lrn_param.alpha());
            fprintf(pp, " 3=%f", lrn_param.beta());
        }
        else if (layer.type() == "MemoryData")
        {
            const caffe::MemoryDataParameter& memory_data_param = layer.memory_data_param();
            fprintf(pp, " 0=%d", memory_data_param.width());
            fprintf(pp, " 1=%d", memory_data_param.height());
            fprintf(pp, " 2=%d", memory_data_param.channels());
        }
        else if (layer.type() == "MVN")
        {
            const caffe::MVNParameter& mvn_param = layer.mvn_param();
            fprintf(pp, " 0=%d", mvn_param.normalize_variance());
            fprintf(pp, " 1=%d", mvn_param.across_channels());
            fprintf(pp, " 2=%f", mvn_param.eps());
        }
        else if (layer.type() == "Normalize")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);
            const caffe::BlobProto& scale_blob = binlayer.blobs(0);
            const caffe::NormalizeParameter& norm_param = layer.norm_param();
            fprintf(pp, " 0=%d", norm_param.across_spatial());
            fprintf(pp, " 1=%d", norm_param.channel_shared());
            fprintf(pp, " 2=%f", norm_param.eps());
            fprintf(pp, " 3=%d", scale_blob.data_size());

            fwrite(scale_blob.data().data(), sizeof(float), scale_blob.data_size(), bp);
        }
        else if (layer.type() == "Permute")
        {
            const caffe::PermuteParameter& permute_param = layer.permute_param();
            int order_size = permute_param.order_size();
            int order_type = 0;
            if (order_size == 0)
                order_type = 0;
            if (order_size == 1)
            {
                int order0 = permute_param.order(0);
                if (order0 == 0)
                    order_type = 0;
                // permute with N not supported
            }
            if (order_size == 2)
            {
                int order0 = permute_param.order(0);
                int order1 = permute_param.order(1);
                if (order0 == 0)
                {
                    if (order1 == 1) // 0 1 2 3
                        order_type = 0;
                    else if (order1 == 2) // 0 2 1 3
                        order_type = 2;
                    else if (order1 == 3) // 0 3 1 2
                        order_type = 4;
                }
                // permute with N not supported
            }
            if (order_size == 3 || order_size == 4)
            {
                int order0 = permute_param.order(0);
                int order1 = permute_param.order(1);
                int order2 = permute_param.order(2);
                if (order0 == 0)
                {
                    if (order1 == 1)
                    {
                        if (order2 == 2) // 0 1 2 3
                            order_type = 0;
                        if (order2 == 3) // 0 1 3 2
                            order_type = 1;
                    }
                    else if (order1 == 2)
                    {
                        if (order2 == 1) // 0 2 1 3
                            order_type = 2;
                        if (order2 == 3) // 0 2 3 1
                            order_type = 3;
                    }
                    else if (order1 == 3)
                    {
                        if (order2 == 1) // 0 3 1 2
                            order_type = 4;
                        if (order2 == 2) // 0 3 2 1
                            order_type = 5;
                    }
                }
                // permute with N not supported
            }
            fprintf(pp, " 0=%d", order_type);
        }
        else if (layer.type() == "Pooling")
        {
            const caffe::PoolingParameter& pooling_param = layer.pooling_param();
            fprintf(pp, " 0=%d", pooling_param.pool());
            if (pooling_param.has_kernel_w() && pooling_param.has_kernel_h())
            {
                fprintf(pp, " 1=%d", pooling_param.kernel_w());
                fprintf(pp, " 11=%d", pooling_param.kernel_h());
            }
            else
            {
                fprintf(pp, " 1=%d", pooling_param.kernel_size());
            }
            if (pooling_param.has_stride_w() && pooling_param.has_stride_h())
            {
                fprintf(pp, " 2=%d", pooling_param.stride_w());
                fprintf(pp, " 12=%d", pooling_param.stride_h());
            }
            else
            {
                fprintf(pp, " 2=%d", pooling_param.stride());
            }
            if (pooling_param.has_pad_w() && pooling_param.has_pad_h())
            {
                fprintf(pp, " 3=%d", pooling_param.pad_w());
                fprintf(pp, " 13=%d", pooling_param.pad_h());
            }
            else
            {
                fprintf(pp, " 3=%d", pooling_param.pad());
            }
            fprintf(pp, " 4=%d", pooling_param.has_global_pooling() ? pooling_param.global_pooling() : 0);
        }
        else if (layer.type() == "Power")
        {
            const caffe::PowerParameter& power_param = layer.power_param();
            fprintf(pp, " 0=%f", power_param.power());
            fprintf(pp, " 1=%f", power_param.scale());
            fprintf(pp, " 2=%f", power_param.shift());
        }
        else if (layer.type() == "PReLU")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);
            const caffe::BlobProto& slope_blob = binlayer.blobs(0);
            fprintf(pp, " 0=%d", slope_blob.data_size());
            fwrite(slope_blob.data().data(), sizeof(float), slope_blob.data_size(), bp);
        }
        else if (layer.type() == "PriorBox")
        {
            const caffe::PriorBoxParameter& prior_box_param = layer.prior_box_param();

            int num_aspect_ratio = prior_box_param.aspect_ratio_size();
            for (int j=0; j<prior_box_param.aspect_ratio_size(); j++)
            {
                float ar = prior_box_param.aspect_ratio(j);
                if (fabs(ar - 1.) < 1e-6) {
                    num_aspect_ratio--;
                }
            }

            float variances[4] = {0.1f, 0.1f, 0.1f, 0.1f};
            if (prior_box_param.variance_size() == 4)
            {
                variances[0] = prior_box_param.variance(0);
                variances[1] = prior_box_param.variance(1);
                variances[2] = prior_box_param.variance(2);
                variances[3] = prior_box_param.variance(3);
            }
            else if (prior_box_param.variance_size() == 1)
            {
                variances[0] = prior_box_param.variance(0);
                variances[1] = prior_box_param.variance(0);
                variances[2] = prior_box_param.variance(0);
                variances[3] = prior_box_param.variance(0);
            }

            int flip = prior_box_param.has_flip() ? prior_box_param.flip() : 1;
            int clip = prior_box_param.has_clip() ? prior_box_param.clip() : 0;
            int image_width = -233;
            int image_height = -233;
            if (prior_box_param.has_img_size())
            {
                image_width = prior_box_param.img_size();
                image_height = prior_box_param.img_size();
            }
            else if (prior_box_param.has_img_w() && prior_box_param.has_img_h())
            {
                image_width = prior_box_param.img_w();
                image_height = prior_box_param.img_h();
            }

            float step_width = -233;
            float step_height = -233;
            if (prior_box_param.has_step())
            {
                step_width = prior_box_param.step();
                step_height = prior_box_param.step();
            }
            else if (prior_box_param.has_step_w() && prior_box_param.has_step_h())
            {
                step_width = prior_box_param.step_w();
                step_height = prior_box_param.step_h();
            }

            fprintf(pp, " -23300=%d", prior_box_param.min_size_size());
            for (int j=0; j<prior_box_param.min_size_size(); j++)
            {
                fprintf(pp, ",%f", prior_box_param.min_size(j));
            }
            fprintf(pp, " -23301=%d", prior_box_param.max_size_size());
            for (int j=0; j<prior_box_param.max_size_size(); j++)
            {
                fprintf(pp, ",%f", prior_box_param.max_size(j));
            }
            fprintf(pp, " -23302=%d", num_aspect_ratio);
            for (int j=0; j<prior_box_param.aspect_ratio_size(); j++)
            {
                float ar = prior_box_param.aspect_ratio(j);
                if (fabs(ar - 1.) < 1e-6) {
                    continue;
                }
                fprintf(pp, ",%f", ar);
            }
            fprintf(pp, " 3=%f", variances[0]);
            fprintf(pp, " 4=%f", variances[1]);
            fprintf(pp, " 5=%f", variances[2]);
            fprintf(pp, " 6=%f", variances[3]);
            fprintf(pp, " 7=%d", flip);
            fprintf(pp, " 8=%d", clip);
            fprintf(pp, " 9=%d", image_width);
            fprintf(pp, " 10=%d", image_height);
            fprintf(pp, " 11=%f", step_width);
            fprintf(pp, " 12=%f", step_height);
            fprintf(pp, " 13=%f", prior_box_param.offset());
        }
        else if (layer.type() == "Python")
        {
            const caffe::PythonParameter& python_param = layer.python_param();
            std::string python_layer_name = python_param.layer();
            if (python_layer_name == "ProposalLayer")
            {
                int feat_stride = 16;
                sscanf(python_param.param_str().c_str(), "'feat_stride': %d", &feat_stride);

                int base_size = 16;
//                 float ratio;
//                 float scale;
                int pre_nms_topN = 6000;
                int after_nms_topN = 300;
                float nms_thresh = 0.7;
                int min_size = 16;
                fprintf(pp, " 0=%d", feat_stride);
                fprintf(pp, " 1=%d", base_size);
                fprintf(pp, " 2=%d", pre_nms_topN);
                fprintf(pp, " 3=%d", after_nms_topN);
                fprintf(pp, " 4=%f", nms_thresh);
                fprintf(pp, " 5=%d", min_size);
            }
        }
        else if (layer.type() == "ReLU")
        {
            const caffe::ReLUParameter& relu_param = layer.relu_param();
            if (relu_param.has_negative_slope())
            {
                fprintf(pp, " 0=%f", relu_param.negative_slope());
            }
        }
        else if (layer.type() == "Reshape")
        {
            const caffe::ReshapeParameter& reshape_param = layer.reshape_param();
            const caffe::BlobShape& bs = reshape_param.shape();
            if (bs.dim_size() == 1)
            {
                fprintf(pp, " 0=%ld 1=-233 2=-233", bs.dim(0));
            }
            else if (bs.dim_size() == 2)
            {
                fprintf(pp, " 0=%ld 1=%ld 2=-233", bs.dim(1), bs.dim(0));
            }
            else if (bs.dim_size() == 3)
            {
                fprintf(pp, " 0=%ld 1=%ld 2=%ld", bs.dim(2), bs.dim(1), bs.dim(0));
            }
            else // bs.dim_size() == 4
            {
                fprintf(pp, " 0=%ld 1=%ld 2=%ld", bs.dim(3), bs.dim(2), bs.dim(1));
            }
            fprintf(pp, " 3=0");// permute
        }
        else if (layer.type() == "ROIPooling")
        {
            const caffe::ROIPoolingParameter& roi_pooling_param = layer.roi_pooling_param();
            fprintf(pp, " 0=%d", roi_pooling_param.pooled_w());
            fprintf(pp, " 1=%d", roi_pooling_param.pooled_h());
            fprintf(pp, " 2=%f", roi_pooling_param.spatial_scale());
        }
        else if (layer.type() == "Scale")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::ScaleParameter& scale_param = layer.scale_param();
            bool scale_weight = scale_param.bias_term() ? (binlayer.blobs_size() == 2) : (binlayer.blobs_size() == 1);
            if (scale_weight)
            {
                const caffe::BlobProto& weight_blob = binlayer.blobs(0);
                fprintf(pp, " 0=%d", (int)weight_blob.data_size());
            }
            else
            {
                fprintf(pp, " 0=-233");
            }

            fprintf(pp, " 1=%d", scale_param.bias_term());

            for (int j=0; j<binlayer.blobs_size(); j++)
            {
                const caffe::BlobProto& blob = binlayer.blobs(j);
                fwrite(blob.data().data(), sizeof(float), blob.data_size(), bp);
            }
        }
        else if (layer.type() == "ShuffleChannel")
        {
            const caffe::ShuffleChannelParameter&
                    shuffle_channel_param = layer.shuffle_channel_param();
            fprintf(pp, " 0=%d", shuffle_channel_param.group());
        }
        else if (layer.type() == "Slice")
        {
            const caffe::SliceParameter& slice_param = layer.slice_param();
            if (slice_param.slice_point_size() == 0)
            {
                int num_slice = layer.top_size();
                fprintf(pp, " -23300=%d", num_slice);
                for (int j=0; j<num_slice; j++)
                {
                    fprintf(pp, ",-233");
                }
            }
            else
            {
                int num_slice = slice_param.slice_point_size() + 1;
                fprintf(pp, " -23300=%d", num_slice);
                int prev_offset = 0;
                for (int j=0; j<slice_param.slice_point_size(); j++)
                {
                    int offset = slice_param.slice_point(j);
                    fprintf(pp, ",%d", offset - prev_offset);
                    prev_offset = offset;
                }
                fprintf(pp, ",-233");
            }
            int dim = slice_param.axis() - 1;
            fprintf(pp, " 1=%d", dim);
        }
        else if (layer.type() == "Softmax")
        {
            const caffe::SoftmaxParameter& softmax_param = layer.softmax_param();
            int dim = softmax_param.axis() - 1;
            fprintf(pp, " 0=%d", dim);
        }
        else if (layer.type() == "Threshold")
        {
            const caffe::ThresholdParameter& threshold_param = layer.threshold_param();
            fprintf(pp, " 0=%f", threshold_param.threshold());
        }

        fprintf(pp, "\n");

        // add split layer if top reference larger than one
        if (layer.bottom_size() == 1 && layer.top_size() == 1 && layer.bottom(0) == layer.top(0))
        {
            std::string blob_name = blob_name_decorated[layer.top(0)];
            if (bottom_reference.find(blob_name) != bottom_reference.end())
            {
                int refcount = bottom_reference[blob_name];
                if (refcount > 1)
                {
                    char splitname[256];
                    sprintf(splitname, "splitncnn_%d", internal_split);
                    fprintf(pp, "%-16s %-16s %d %d", "Split", splitname, 1, refcount);
                    fprintf(pp, " %s", blob_name.c_str());

                    for (int j=0; j<refcount; j++)
                    {
                        fprintf(pp, " %s_splitncnn_%d", blob_name.c_str(), j);
                    }
                    fprintf(pp, "\n");

                    internal_split++;
                }
            }
        }
        else
        {
            for (int j=0; j<layer.top_size(); j++)
            {
                std::string blob_name = layer.top(j);
                if (bottom_reference.find(blob_name) != bottom_reference.end())
                {
                    int refcount = bottom_reference[blob_name];
                    if (refcount > 1)
                    {
                        char splitname[256];
                        sprintf(splitname, "splitncnn_%d", internal_split);
                        fprintf(pp, "%-16s %-16s %d %d", "Split", splitname, 1, refcount);
                        fprintf(pp, " %s", blob_name.c_str());

                        for (int j=0; j<refcount; j++)
                        {
                            fprintf(pp, " %s_splitncnn_%d", blob_name.c_str(), j);
                        }
                        fprintf(pp, "\n");

                        internal_split++;
                    }
                }
            }
        }

    }

    fclose(pp);
    fclose(bp);

    return 0;
}
