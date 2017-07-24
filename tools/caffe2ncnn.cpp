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
        fprintf(pp, "%-16s %-16s %d %d", layer.type().c_str(), layer.name().c_str(), layer.bottom_size(), layer.top_size());

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
            fprintf(pp, " %d", (int)mean_blob.data_size());

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
                float scale_factor = 1 / binlayer.blobs(2).data().data()[0];
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
        else if (layer.type() == "Convolution")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::BlobProto& weight_blob = binlayer.blobs(0);
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            fprintf(pp, " %d %d %d %d %d %d %d", convolution_param.num_output(), convolution_param.kernel_size(0),
                    convolution_param.dilation_size() != 0 ? convolution_param.dilation(0) : 1,
                    convolution_param.stride_size() != 0 ? convolution_param.stride(0) : 1,
                    convolution_param.pad_size() != 0 ? convolution_param.pad(0) : 0,
                    convolution_param.bias_term(),
                    weight_blob.data_size());

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
            fprintf(pp, " %d %d", woffset, hoffset);
        }
        else if (layer.type() == "Deconvolution")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::BlobProto& weight_blob = binlayer.blobs(0);
            const caffe::ConvolutionParameter& convolution_param = layer.convolution_param();
            fprintf(pp, " %d %d %d %d %d %d %d", convolution_param.num_output(), convolution_param.kernel_size(0),
                    convolution_param.dilation_size() != 0 ? convolution_param.dilation(0) : 1,
                    convolution_param.stride_size() != 0 ? convolution_param.stride(0) : 1,
                    convolution_param.pad_size() != 0 ? convolution_param.pad(0) : 0,
                    convolution_param.bias_term(),
                    weight_blob.data_size());

            int quantized_weight = 0;
            fwrite(&quantized_weight, sizeof(int), 1, bp);

            // reorder weight from inch-outch to outch-inch
            int ksize = convolution_param.kernel_size(0);
            int num_output = convolution_param.num_output();
            int num_input = weight_blob.data_size() / (ksize * ksize) / num_output;
            const float* weight_data_ptr = weight_blob.data().data();
            for (int k=0; k<num_output; k++)
            {
                for (int j=0; j<num_input; j++)
                {
                    fwrite(weight_data_ptr + (j*num_output + k) * ksize * ksize, sizeof(float), ksize * ksize, bp);
                }
            }

            for (int j=1; j<binlayer.blobs_size(); j++)
            {
                const caffe::BlobProto& blob = binlayer.blobs(j);
                fwrite(blob.data().data(), sizeof(float), blob.data_size(), bp);
            }
        }
        else if (layer.type() == "Eltwise")
        {
            const caffe::EltwiseParameter& eltwise_param = layer.eltwise_param();
            int coeff_size = eltwise_param.coeff_size();
            fprintf(pp, " %d %d", (int)eltwise_param.operation(), coeff_size);
            for (int j=0; j<coeff_size; j++)
            {
                fprintf(pp, " %f", eltwise_param.coeff(j));
            }
        }
        else if (layer.type() == "InnerProduct")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::BlobProto& weight_blob = binlayer.blobs(0);
            const caffe::InnerProductParameter& inner_product_param = layer.inner_product_param();
            fprintf(pp, " %d %d %d", inner_product_param.num_output(), inner_product_param.bias_term(),
                    weight_blob.data_size());

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
            for (int j=1; j<std::min((int)bs.dim_size(), 4); j++)
            {
                fprintf(pp, " %lld", bs.dim(j));
            }
            for (int j=bs.dim_size(); j<4; j++)
            {
                fprintf(pp, " -233");
            }
        }
        else if (layer.type() == "LRN")
        {
            const caffe::LRNParameter& lrn_param = layer.lrn_param();
            fprintf(pp, " %d %d %.8f %.8f", lrn_param.norm_region(), lrn_param.local_size(), lrn_param.alpha(), lrn_param.beta());
        }
        else if (layer.type() == "MemoryData")
        {
            const caffe::MemoryDataParameter& memory_data_param = layer.memory_data_param();
            fprintf(pp, " %d %d %d", memory_data_param.channels(), memory_data_param.width(), memory_data_param.height());
        }
        else if (layer.type() == "Pooling")
        {
            const caffe::PoolingParameter& pooling_param = layer.pooling_param();
            fprintf(pp, " %d %d %d %d %d", pooling_param.pool(), pooling_param.kernel_size(), pooling_param.stride(), pooling_param.pad(),
                    pooling_param.has_global_pooling() ? pooling_param.global_pooling() : 0);
        }
        else if (layer.type() == "Power")
        {
            const caffe::PowerParameter& power_param = layer.power_param();
            fprintf(pp, " %f %f %f", power_param.power(), power_param.scale(), power_param.shift());
        }
        else if (layer.type() == "PReLU")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);
            const caffe::BlobProto& slope_blob = binlayer.blobs(0);
            fprintf(pp, " %d", slope_blob.data_size());
            fwrite(slope_blob.data().data(), sizeof(float), slope_blob.data_size(), bp);
        }
        else if (layer.type() == "Proposal")
        {
            const caffe::PythonParameter& python_param = layer.python_param();
            int feat_stride = 16;
            sscanf(python_param.param_str().c_str(), "'feat_stride': %d", &feat_stride);
            int base_size = 16;
//             float ratio;
//             float scale;
            int pre_nms_topN = 6000;
            int after_nms_topN = 5;
            float nms_thresh = 0.7;
            int min_size = 16;
            fprintf(pp, " %d %d %d %d %f %d", feat_stride, base_size, pre_nms_topN, after_nms_topN, nms_thresh, min_size);
        }
        else if (layer.type() == "ReLU")
        {
            const caffe::ReLUParameter& relu_param = layer.relu_param();
            fprintf(pp, " %f", relu_param.negative_slope());
        }
        else if (layer.type() == "Reshape")
        {
            const caffe::ReshapeParameter& reshape_param = layer.reshape_param();
            const caffe::BlobShape& bs = reshape_param.shape();
            for (int j=1; j<std::min((int)bs.dim_size(), 4); j++)
            {
                fprintf(pp, " %lld", bs.dim(j));
            }
            for (int j=bs.dim_size(); j<4; j++)
            {
                fprintf(pp, " -233");
            }
        }
        else if (layer.type() == "ROIPooling")
        {
            const caffe::ROIPoolingParameter& roi_pooling_param = layer.roi_pooling_param();
            fprintf(pp, " %d %d %.8f", roi_pooling_param.pooled_w(), roi_pooling_param.pooled_h(), roi_pooling_param.spatial_scale());
        }
        else if (layer.type() == "Scale")
        {
            const caffe::LayerParameter& binlayer = net.layer(netidx);

            const caffe::BlobProto& weight_blob = binlayer.blobs(0);
            const caffe::ScaleParameter& scale_param = layer.scale_param();
            fprintf(pp, " %d %d", (int)weight_blob.data_size(), scale_param.bias_term());

            for (int j=0; j<binlayer.blobs_size(); j++)
            {
                const caffe::BlobProto& blob = binlayer.blobs(j);
                fwrite(blob.data().data(), sizeof(float), blob.data_size(), bp);
            }
        }
        else if (layer.type() == "Slice")
        {
            const caffe::SliceParameter& slice_param = layer.slice_param();
            if (slice_param.has_slice_dim())
            {
                int num_slice = layer.top_size();
                fprintf(pp, " %d", num_slice);
                for (int j=0; j<num_slice; j++)
                {
                    fprintf(pp, " -233");
                }
            }
            else
            {
                int num_slice = slice_param.slice_point_size() + 1;
                fprintf(pp, " %d", num_slice);
                int prev_offset = 0;
                for (int j=0; j<num_slice; j++)
                {
                    int offset = slice_param.slice_point(j);
                    fprintf(pp, " %d", offset - prev_offset);
                    prev_offset = offset;
                }
                fprintf(pp, " -233");
            }
        }
        else if (layer.type() == "Threshold")
        {
            const caffe::ThresholdParameter& threshold_param = layer.threshold_param();
            fprintf(pp, " %f", threshold_param.threshold());
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
