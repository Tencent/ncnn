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

#include <iostream>

#include <fstream>
#include <set>
#include <limits>
#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/message.h>

#include "graph.pb.h"

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

static bool find_tensor_proto(const std::map<std::string, tensorflow::TensorProto>& weights,
                              const tensorflow::NodeDef& node, tensorflow::TensorProto& tensor)
{
    for (int j=0; j<node.input_size(); j++)
    {
        const std::string& input_name = node.input(j);

        const std::map<std::string, tensorflow::TensorProto>::const_iterator it = weights.find(input_name);
        if (it != weights.end())
        {
            tensor = it->second;
            return true;
        }
    }

    return false;
}

static bool get_tensor_proto(const std::map<std::string, tensorflow::TensorProto>& consts,
                             const tensorflow::NodeDef& node, tensorflow::TensorProto& tensor)
{
    const std::string& output_name = node.name();

    const std::map<std::string, tensorflow::TensorProto>::const_iterator it = consts.find(output_name);
    if (it != consts.end())
    {
        tensor = it->second;
        return true;
    }

    return false;
}

static bool find_attr_value(const tensorflow::NodeDef& node, const char* key, tensorflow::AttrValue& value)
{
    const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node.attr();

    const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.find(key);
    if (it != attr.end())
    {
        value = it->second;
        return true;
    }

    return false;
}

static int parse_tensor_reduction_dim(const tensorflow::TensorProto& tensor)
{
    int dim = 0;

    // dim == 0 // w h c -> X X X
    // dim == 1 // w h c -> X X c
    // dim == 2 // w h c -> X h c
    // dim == -1 // w h c -> w X X
    // dim == -2 // w h c -> w h X

    if (!tensor.tensor_content().empty() && tensor.dtype() == 3)// int32
    {
        const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
        int size = tensor.tensor_content().size() / sizeof(int);

        // n h w c
        // n h w
        // n w
        // TODO investigate two stage / three stage reduction
        if (size == 2)
        {
            if (data[0] == 1 && data[1] == 2)
            {
                dim = 1;
            }
        }
    }
    else
    {
        int axis = tensor.int_val(0);
        if (axis == 1)
            dim = 0;
        else if (axis == 3)
            dim = -2;
    }

    return dim;
}

int main(int argc, char** argv)
{
    const char* tensorflowpb = argv[1];
    const char* ncnn_prototxt = argc >= 4 ? argv[2] : "ncnn.proto";
    const char* ncnn_modelbin = argc >= 4 ? argv[3] : "ncnn.bin";

    tensorflow::GraphDef graph;

    // load
    bool s1 = read_proto_from_binary(tensorflowpb, &graph);
    if (!s1)
    {
        fprintf(stderr, "read_proto_from_binary failed\n");
        return -1;
    }

    FILE* pp = fopen(ncnn_prototxt, "wb");
    FILE* bp = fopen(ncnn_modelbin, "wb");

    // magic
    fprintf(pp, "7767517\n");

    int node_count = graph.node_size();

//     fprintf(stderr, "node_count = %d\n\n", node_count);

    // node reference
    std::map<std::string, int> node_reference;

    // mapping for Const and Const-Identity
    std::map<std::string, tensorflow::TensorProto> weights;

    // Dropout like Identity
    std::set<std::string> dropouts;

    // Const before BinaryOp
    std::map<std::string, tensorflow::TensorProto> binaryop_consts;

    // global definition line
    // [layer count] [blob count]
    std::set<std::string> blob_names;
    for (int i=0; i<node_count; i++)
    {
        const tensorflow::NodeDef& node = graph.node(i);

        const std::string& output_name = node.name();

        if (node.op() == "Const")
        {
            tensorflow::AttrValue value;
            if (find_attr_value(node, "value", value))
            {
                const tensorflow::TensorProto& tensor = value.tensor();
                weights[output_name] = tensor;
            }
            continue;
        }
        else if (node.op() == "Identity")
        {
            const std::string& input_name = node.input(0);
            if (weights.find(input_name) != weights.end())
            {
                weights[output_name] = weights[input_name];
                continue;
            }
            else
            {
                dropouts.insert(output_name);
            }
        }
        else if (node.op() == "NoOp")
        {
            weights[output_name] = tensorflow::TensorProto();
            continue;
        }
        else
        {
            bool isBinaryOp = false;
            if (node.op() == "Add" || node.op() == "BiasAdd" || node.op() == "Div"
                || node.op() == "Mul" || node.op() == "RealDiv" || node.op() == "Sub")
            {
                isBinaryOp = true;
            }
            if (node.op() == "Max" || node.op() == "Maximum" || node.op() == "Min" || node.op() == "Minimum")
            {
                // check weights
                tensorflow::TensorProto tensor;
                if (!find_tensor_proto(weights, node, tensor))
                {
                    isBinaryOp = true;
                }
            }

            if (isBinaryOp)
            {
                // check weights
                for (int j=0; j<node.input_size(); j++)
                {
                    const std::string& input_name = node.input(j);

                    std::map<std::string, tensorflow::TensorProto>::iterator it = weights.find(input_name);
                    if (it != weights.end())
                    {
                        // binary op with const, insert MemoryData layer and const blob
                        binaryop_consts[input_name] = it->second;
                        weights.erase(it);
                    }
                }
            }
        }

        // input
        for (int j=0; j<node.input_size(); j++)
        {
            const std::string& input_name = node.input(j);
//             fprintf(stderr, "input = %s\n", input_name.c_str());

            if (weights.find(input_name) != weights.end())
            {
                continue;
            }

            blob_names.insert(input_name);

            if (node_reference.find(input_name) == node_reference.end())
            {
                node_reference[input_name] = 1;
            }
            else
            {
                node_reference[input_name] = node_reference[input_name] + 1;
            }
        }

        // output
//         fprintf(stderr, "output = %s\n", output_name.c_str());
        blob_names.insert(output_name);
    }

    // remove node_reference entry with reference equals to one
    int splitncnn_blob_count = 0;
    std::map<std::string, int>::iterator it = node_reference.begin();
    while (it != node_reference.end())
    {
        if (it->second == 1)
        {
            node_reference.erase(it++);
        }
        else
        {
            splitncnn_blob_count += it->second;
//             fprintf(stderr, "%s %d\n", it->first.c_str(), it->second);
            ++it;
        }
    }

    fprintf(pp, "%lu %lu\n", node_count + node_reference.size() - weights.size(), blob_names.size() + splitncnn_blob_count);

    int internal_split = 0;

    for (int i=0; i<node_count; i++)
    {
        const tensorflow::NodeDef& node = graph.node(i);

        // layer definition line, repeated
        // [type] [name] [bottom blob count] [top blob count] [bottom blobs] [top blobs] [layer specific params]
//         fprintf(pp, "%-16s %-16s %d %d", layer.type().c_str(), layer.name().c_str(), node.input_size(), layer.top_size());

        if (node.op() == "Add" || node.op() == "BiasAdd")
        {
            fprintf(pp, "%-16s", "BinaryOp");
        }
        else if (node.op() == "AvgPool")
        {
            fprintf(pp, "%-16s", "Pooling");
        }
        else if (node.op() == "Concat" || node.op() == "ConcatV2")
        {
            fprintf(pp, "%-16s", "Concat");
        }
        else if (node.op() == "Const")
        {
            // check before binaryop
            tensorflow::TensorProto tensor;
            if (get_tensor_proto(binaryop_consts, node, tensor))
            {
                fprintf(pp, "%-16s", "MemoryData");
            }
            else
            {
                continue;
            }
        }
        else if (node.op() == "Conv2D")
        {
            fprintf(pp, "%-16s", "Convolution");
        }
        else if (node.op() == "DepthwiseConv2dNative")
        {
            fprintf(pp, "%-16s", "ConvolutionDepthWise");
        }
        else if (node.op() == "Div" || node.op() == "RealDiv")
        {
            fprintf(pp, "%-16s", "BinaryOp");
        }
        else if (node.op() == "Exp")
        {
            fprintf(pp, "%-16s", "UnaryOp");
        }
        else if (node.op() == "ExpandDims")
        {
            fprintf(pp, "%-16s", "ExpandDims");
        }
        else if (node.op() == "Floor")
        {
            fprintf(pp, "%-16s", "UnaryOp");
        }
        else if (node.op() == "Identity")
        {
            // check before binaryop
            tensorflow::TensorProto tensor;
            if (get_tensor_proto(binaryop_consts, node, tensor))
            {
                fprintf(pp, "%-16s", "MemoryData");
            }
            else if (dropouts.find(node.name()) != dropouts.end())
            {
                fprintf(pp, "%-16s", "Dropout");
            }
            else
            {
                continue;
            }
        }
        else if (node.op() == "LRN")
        {
            fprintf(pp, "%-16s", "LRN");
        }
        else if (node.op() == "MatMul")
        {
            fprintf(pp, "%-16s", "InnerProduct");
        }
        else if (node.op() == "Max" || node.op() == "Maximum")
        {
            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                fprintf(pp, "%-16s", "Reduction");
            }
            else
            {
                fprintf(pp, "%-16s", "BinaryOp");
            }
        }
        else if (node.op() == "MaxPool")
        {
            fprintf(pp, "%-16s", "Pooling");
        }
        else if (node.op() == "Min" || node.op() == "Minimum")
        {
            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                fprintf(pp, "%-16s", "Reduction");
            }
            else
            {
                fprintf(pp, "%-16s", "BinaryOp");
            }
        }
        else if (node.op() == "Mul")
        {
            fprintf(pp, "%-16s", "BinaryOp");
        }
        else if (node.op() == "Neg")
        {
            fprintf(pp, "%-16s", "UnaryOp");
        }
        else if (node.op() == "NoOp")
        {
            continue;
        }
        else if (node.op() == "Pad")
        {
            fprintf(pp, "%-16s", "Padding");
        }
        else if (node.op() == "Placeholder")
        {
            fprintf(pp, "%-16s", "Input");
        }
        else if (node.op() == "Prod")
        {
            fprintf(pp, "%-16s", "Reduction");
        }
        else if (node.op() == "Reciprocal")
        {
            fprintf(pp, "%-16s", "UnaryOp");
        }
        else if (node.op() == "Relu")
        {
            fprintf(pp, "%-16s", "ReLU");
        }
        else if (node.op() == "Reshape")
        {
            fprintf(pp, "%-16s", "Reshape");
        }
        else if (node.op() == "Rsqrt")
        {
            fprintf(pp, "%-16s", "UnaryOp");
        }
        else if (node.op() == "Sigmoid")
        {
            fprintf(pp, "%-16s", "Sigmoid");
        }
        else if (node.op() == "Softmax")
        {
            fprintf(pp, "%-16s", "Softmax");
        }
        else if (node.op() == "Square")
        {
            fprintf(pp, "%-16s", "UnaryOp");
        }
        else if (node.op() == "Squeeze")
        {
            fprintf(pp, "%-16s", "Squeeze");
        }
        else if (node.op() == "Sub")
        {
            fprintf(pp, "%-16s", "BinaryOp");
        }
        else if (node.op() == "Sum")
        {
            fprintf(pp, "%-16s", "Reduction");
        }
        else
        {
            fprintf(pp, "%-16s", node.op().c_str());
            fprintf(stderr, "%s not supported yet !\nn", node.op().c_str());
        }

        int input_size = node.input_size();
        for (int j=0; j<node.input_size(); j++)
        {
            const std::string& input_name = node.input(j);
            if (weights.find(input_name) != weights.end())
            {
                input_size--;
            }
        }

        fprintf(pp, " %-32s %d 1", node.name().c_str(), input_size);

        for (int j=0; j<node.input_size(); j++)
        {
            std::string input_name = node.input(j);

            if (weights.find(input_name) != weights.end())
            {
                continue;
            }

            if (node_reference.find(input_name) != node_reference.end())
            {
                int refidx = node_reference[input_name] - 1;
                node_reference[input_name] = refidx;

                char splitsuffix[256];
                sprintf(splitsuffix, "_splitncnn_%d", refidx);
                input_name = input_name + splitsuffix;
            }

            fprintf(pp, " %s", input_name.c_str());
        }

        fprintf(pp, " %s", node.name().c_str());

        if (node.op() == "Add" || node.op() == "BiasAdd")
        {
            int op_type = 0;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "AvgPool")
        {
            int pooling_type = 1;

            int kernel_size_h = 1;
            int kernel_size_w = 1;
            int stride_h = 1;
            int stride_w = 1;
            int pad = 0;

            int global_pooling = 0;

            tensorflow::AttrValue value_ksize;
            if (find_attr_value(node, "ksize", value_ksize))
            {
                // batch, height, width, channels
                kernel_size_h = value_ksize.list().i(1);
                kernel_size_w = value_ksize.list().i(2);
            }

            tensorflow::AttrValue value_strides;
            if (find_attr_value(node, "strides", value_strides))
            {
                // batch, height, width, channels
                stride_h = value_strides.list().i(1);
                stride_w = value_strides.list().i(2);
            }

            tensorflow::AttrValue value_padding;
            if (find_attr_value(node, "padding", value_padding))
            {
                if (value_padding.s() == "VALID")
                {
                    pad = 0;
                }
                else if (value_padding.s() == "SAME")
                {
                    pad = -233;
                }
            }

            fprintf(pp, " 0=%d", pooling_type);
            fprintf(pp, " 1=%d", kernel_size_w);
            fprintf(pp, " 2=%d", stride_w);
            fprintf(pp, " 3=%d", pad);
            fprintf(pp, " 4=%d", global_pooling);
        }
        else if (node.op() == "Concat" || node.op() == "ConcatV2")
        {
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                // TODO
//                 int axis = tensor.int_val(0);
            }
        }
        else if (node.op() == "Const" || node.op() == "Identity")
        {
            // check before binaryop
            tensorflow::TensorProto tensor;
            if (get_tensor_proto(binaryop_consts, node, tensor))
            {
                const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();

                int w = 0;
                int h = 0;
                int c = 0;

                if (shape.dim_size() == 1)
                {
                    w = shape.dim(0).size();
                }
                else if (shape.dim_size() == 2)
                {
                    h = shape.dim(0).size();
                    w = shape.dim(1).size();
                }
                else if (shape.dim_size() == 3)
                {
                    c = shape.dim(2).size();
                    h = shape.dim(0).size();
                    w = shape.dim(1).size();
                }

                int weight_data_size = 0;

                if (!tensor.tensor_content().empty())
                {
                    if (tensor.dtype() == 1)// float
                    {
                        const float* data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                        weight_data_size = tensor.tensor_content().size() / sizeof(float);

                        if (c == 0)
                            fwrite(data, sizeof(float), weight_data_size, bp);
                        else
                        {
                            float tmp;
                            // h-w-c to c-h-w
                            for (int p=0; p<c; p++)
                            {
                                for (int i=0; i<h; i++)
                                {
                                    for (int j=0; j<w; j++)
                                    {
                                        tmp = data[i*w*c + j*c + p];
                                        fwrite(&tmp, sizeof(float), 1, bp);
                                    }
                                }
                            }
                        }
                    }
                    else if (tensor.dtype() == 3)// int32
                    {
                        const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                        weight_data_size = tensor.tensor_content().size() / sizeof(int);

                        float tmp;
                        if (c == 0)
                        {
                            for (int i=0; i<weight_data_size; i++)
                            {
                                tmp = data[i];
                                fwrite(&tmp, sizeof(float), 1, bp);
                            }
                        }
                        else
                        {
                            // h-w-c to c-h-w
                            for (int p=0; p<c; p++)
                            {
                                for (int i=0; i<h; i++)
                                {
                                    for (int j=0; j<w; j++)
                                    {
                                        tmp = data[i*w*c + j*c + p];
                                        fwrite(&tmp, sizeof(float), 1, bp);
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    if (tensor.dtype() == 1)// float
                    {
                        float val = tensor.float_val(0);
                        fwrite(&val, sizeof(float), 1, bp);
                    }
                    else if (tensor.dtype() == 3)// int32
                    {
                        float val = tensor.int_val(0);
                        fwrite(&val, sizeof(float), 1, bp);
                    }
                }

                fprintf(pp, " 0=%d", w);
                fprintf(pp, " 1=%d", h);
                fprintf(pp, " 2=%d", c);
            }
        }
        else if (node.op() == "Conv2D")
        {
            // weights
            tensorflow::TensorProto tensor;
            find_tensor_proto(weights, node, tensor);

            const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();

            int kernel_size_h = shape.dim(0).size();
            int kernel_size_w = shape.dim(1).size();
            int num_input = shape.dim(2).size();
            int num_output = shape.dim(3).size();

            int stride_h = 1;
            int stride_w = 1;
            int dilation_h = 1;
            int dilation_w = 1;
            int pad = 0;

            tensorflow::AttrValue value_strides;
            if (find_attr_value(node, "strides", value_strides))
            {
                // batch, height, width, channels
                stride_h = value_strides.list().i(1);
                stride_w = value_strides.list().i(2);
            }

            tensorflow::AttrValue value_padding;
            if (find_attr_value(node, "padding", value_padding))
            {
                if (value_padding.s() == "VALID")
                {
                    pad = 0;
                }
                else if (value_padding.s() == "SAME")
                {
                    pad = -233;
                }
            }

            tensorflow::AttrValue value_rate;
            if (find_attr_value(node, "rate", value_rate))
            {
                // height, width
                dilation_h = value_rate.list().i(0);
                dilation_w = value_rate.list().i(1);
            }

            int bias_term = 0;
            int weight_data_size = 0;

            // reorder h-w-i-o to o-i-h-w
            if (!tensor.tensor_content().empty())
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                if (tensor.dtype() == 1)// float
                {
                    const float* data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(float);

                    float tmp;
                    for (int p=0; p<num_output; p++)
                    {
                        for (int q=0; q<num_input; q++)
                        {
                            for (int i=0; i<kernel_size_h; i++)
                            {
                                for (int j=0; j<kernel_size_w; j++)
                                {
                                    tmp = data[i*kernel_size_w*num_input*num_output + j*num_input*num_output + q*num_output + p];
                                    fwrite(&tmp, sizeof(float), 1, bp);
                                }
                            }
                        }
                    }
                }
                else if (tensor.dtype() == 3)// int32
                {
                    const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(int);

                    float tmp;
                    for (int p=0; p<num_output; p++)
                    {
                        for (int q=0; q<num_input; q++)
                        {
                            for (int i=0; i<kernel_size_h; i++)
                            {
                                for (int j=0; j<kernel_size_w; j++)
                                {
                                    tmp = data[i*kernel_size_w*num_input*num_output + j*num_input*num_output + q*num_output + p];
                                    fwrite(&tmp, sizeof(float), 1, bp);
                                }
                            }
                        }
                    }
                }
            }

            fprintf(pp, " 0=%d", num_output);
            fprintf(pp, " 1=%d", kernel_size_w);
            fprintf(pp, " 2=%d", dilation_w);
            fprintf(pp, " 3=%d", stride_w);
            fprintf(pp, " 4=%d", pad);
            fprintf(pp, " 5=%d", bias_term);
            fprintf(pp, " 6=%d", weight_data_size);
        }
        else if (node.op() == "DepthwiseConv2dNative")
        {
            // weights
            tensorflow::TensorProto tensor;
            find_tensor_proto(weights, node, tensor);

            const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();

            int kernel_size_h = shape.dim(0).size();
            int kernel_size_w = shape.dim(1).size();
            int num_input = shape.dim(2).size();
            int channel_multiplier = shape.dim(3).size();

            int num_output = num_input * channel_multiplier;
            int group = num_input;

            int stride_h = 1;
            int stride_w = 1;
            int dilation_h = 1;
            int dilation_w = 1;
            int pad = 0;

            tensorflow::AttrValue value_strides;
            if (find_attr_value(node, "strides", value_strides))
            {
                // batch, height, width, channels
                stride_h = value_strides.list().i(1);
                stride_w = value_strides.list().i(2);
            }

            tensorflow::AttrValue value_padding;
            if (find_attr_value(node, "padding", value_padding))
            {
                if (value_padding.s() == "VALID")
                {
                    pad = 0;
                }
                else if (value_padding.s() == "SAME")
                {
                    pad = -233;
                }
            }

            tensorflow::AttrValue value_rate;
            if (find_attr_value(node, "rate", value_rate))
            {
                // height, width
                dilation_h = value_rate.list().i(0);
                dilation_w = value_rate.list().i(1);
            }

            int bias_term = 0;
            int weight_data_size = 0;

            // reorder h-w-i-cm to i-cm-h-w
            if (!tensor.tensor_content().empty())
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                if (tensor.dtype() == 1)// float
                {
                    const float* data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(float);

                    float tmp;
                    for (int p=0; p<num_input; p++)
                    {
                        for (int q=0; q<channel_multiplier; q++)
                        {
                            for (int i=0; i<kernel_size_h; i++)
                            {
                                for (int j=0; j<kernel_size_w; j++)
                                {
                                    tmp = data[i*kernel_size_w*channel_multiplier*num_input + j*channel_multiplier*num_input + p*channel_multiplier + q];
                                    fwrite(&tmp, sizeof(float), 1, bp);
                                }
                            }
                        }
                    }
                }
                else if (tensor.dtype() == 3)// int32
                {
                    const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(int);

                    float tmp;
                    for (int p=0; p<num_input; p++)
                    {
                        for (int q=0; q<channel_multiplier; q++)
                        {
                            for (int i=0; i<kernel_size_h; i++)
                            {
                                for (int j=0; j<kernel_size_w; j++)
                                {
                                    tmp = data[i*kernel_size_w*channel_multiplier*num_input + j*channel_multiplier*num_input + p*channel_multiplier + q];
                                    fwrite(&tmp, sizeof(float), 1, bp);
                                }
                            }
                        }
                    }
                }
            }

            fprintf(pp, " 0=%d", num_output);
            fprintf(pp, " 1=%d", kernel_size_w);
            fprintf(pp, " 2=%d", dilation_w);
            fprintf(pp, " 3=%d", stride_w);
            fprintf(pp, " 4=%d", pad);
            fprintf(pp, " 5=%d", bias_term);
            fprintf(pp, " 6=%d", weight_data_size);
            fprintf(pp, " 7=%d", group);
        }
        else if (node.op() == "Div" || node.op() == "RealDiv")
        {
            int op_type = 3;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "Exp")
        {
            int op_type = 7;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "ExpandDims")
        {
            int expand_w = 0;
            int expand_h = 0;
            int expand_c = 0;

            tensorflow::AttrValue value_dim;
            if (find_attr_value(node, "Tdim", value_dim))
            {
                int dim = value_dim.i();
                if (dim == 0)
                    expand_w = 1;
                if (dim == 1)
                    expand_h = 1;
                if (dim == 2)
                    expand_c = 1;
            }

            fprintf(pp, " 0=%d", expand_w);
            fprintf(pp, " 1=%d", expand_h);
            fprintf(pp, " 2=%d", expand_c);
        }
        else if (node.op() == "Floor")
        {
            int op_type = 2;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "LRN")
        {
            int norm_region = 0;
            int local_size = 1;
            float alpha = 1.f;
            float beta = 0.5f;

            tensorflow::AttrValue value_depth_radius;
            if (find_attr_value(node, "depth_radius", value_depth_radius))
            {
                local_size = value_depth_radius.i() * 2 + 1;
            }

            tensorflow::AttrValue value_alpha;
            if (find_attr_value(node, "alpha", value_alpha))
            {
                alpha = value_alpha.f();
            }

            tensorflow::AttrValue value_beta;
            if (find_attr_value(node, "beta", value_beta))
            {
                beta = value_beta.f();
            }

            // TODO
            float bias = 1.f;
            tensorflow::AttrValue value_bias;
            if (find_attr_value(node, "bias", value_bias))
            {
                bias = value_bias.f();
            }

            fprintf(pp, " 0=%d", norm_region);
            fprintf(pp, " 1=%d", local_size);
            fprintf(pp, " 2=%f", alpha);
            fprintf(pp, " 3=%f", beta);
        }
        else if (node.op() == "MatMul")
        {
            // weights
            tensorflow::TensorProto tensor;
            find_tensor_proto(weights, node, tensor);

            const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();

            int num_input = shape.dim(0).size();
            int num_output = shape.dim(1).size();

            int bias_term = 0;
            int weight_data_size = 0;

            // reorder i-o to o-i
            if (!tensor.tensor_content().empty())
            {
                int quantize_tag = 0;
                fwrite(&quantize_tag, sizeof(int), 1, bp);

                if (tensor.dtype() == 1)// float
                {
                    const float* data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(float);

                    float tmp;
                    for (int p=0; p<num_output; p++)
                    {
                        for (int q=0; q<num_input; q++)
                        {
                            tmp = data[q*num_output + p];
                            fwrite(&tmp, sizeof(float), 1, bp);
                        }
                    }
                }
                else if (tensor.dtype() == 3)// int32
                {
                    const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                    weight_data_size = tensor.tensor_content().size() / sizeof(int);

                    float tmp;
                    for (int p=0; p<num_output; p++)
                    {
                        for (int q=0; q<num_input; q++)
                        {
                            tmp = data[q*num_output + p];
                            fwrite(&tmp, sizeof(float), 1, bp);
                        }
                    }
                }
            }

            fprintf(pp, " 0=%d", num_output);
            fprintf(pp, " 1=%d", bias_term);
            fprintf(pp, " 2=%d", weight_data_size);
        }
        else if (node.op() == "Max" || node.op() == "Maximum")
        {
            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                int operation = 4;
                int dim = 0;
                float coeff = 1.f;

                dim = parse_tensor_reduction_dim(tensor);

                fprintf(pp, " 0=%d", operation);
                fprintf(pp, " 1=%d", dim);
                fprintf(pp, " 2=%f", coeff);
            }
            else
            {
                int op_type = 4;
                fprintf(pp, " 0=%d", op_type);
            }
        }
        else if (node.op() == "MaxPool")
        {
            int pooling_type = 0;

            int kernel_size_h = 1;
            int kernel_size_w = 1;
            int stride_h = 1;
            int stride_w = 1;
            int pad = 0;

            int global_pooling = 0;

            tensorflow::AttrValue value_ksize;
            if (find_attr_value(node, "ksize", value_ksize))
            {
                // batch, height, width, channels
                kernel_size_h = value_ksize.list().i(1);
                kernel_size_w = value_ksize.list().i(2);
            }

            tensorflow::AttrValue value_strides;
            if (find_attr_value(node, "strides", value_strides))
            {
                // batch, height, width, channels
                stride_h = value_strides.list().i(1);
                stride_w = value_strides.list().i(2);
            }

            tensorflow::AttrValue value_padding;
            if (find_attr_value(node, "padding", value_padding))
            {
                if (value_padding.s() == "VALID")
                {
                    pad = -2333;
                }
                else if (value_padding.s() == "SAME")
                {
                    pad = -233;
                }
            }

            fprintf(pp, " 0=%d", pooling_type);
            fprintf(pp, " 1=%d", kernel_size_w);
            fprintf(pp, " 2=%d", stride_w);
            fprintf(pp, " 3=%d", pad);
            fprintf(pp, " 4=%d", global_pooling);
        }
        else if (node.op() == "Min" || node.op() == "Minimum")
        {
            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                int operation = 5;
                int dim = 0;
                float coeff = 1.f;

                dim = parse_tensor_reduction_dim(tensor);

                fprintf(pp, " 0=%d", operation);
                fprintf(pp, " 1=%d", dim);
                fprintf(pp, " 2=%f", coeff);
            }
            else
            {
                int op_type = 5;
                fprintf(pp, " 0=%d", op_type);
            }
        }
        else if (node.op() == "Mul")
        {
            int op_type = 2;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "Neg")
        {
            int op_type = 1;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "NoOp")
        {
        }
        else if (node.op() == "Pad")
        {
            int top = 0;
            int bottom = 0;
            int left = 0;
            int right = 0;
            int type = 0;
            float value = 0.f;

            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                if (!tensor.tensor_content().empty() && tensor.dtype() == 3)// int32
                {
                    const int *data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                    int size = tensor.tensor_content().size() / sizeof(int);

                    if (size == 8)
                    {
                        // n h w c
                        top = data[2];
                        bottom = data[3];
                        left = data[4];
                        right = data[5];
                    }
                }
            }

            tensorflow::AttrValue value_Tpaddings;
            if (find_attr_value(node, "Tpaddings", value_Tpaddings))
            {
                type = value_Tpaddings.i();
            }

            tensorflow::AttrValue value_T;
            if (find_attr_value(node, "T", value_T))
            {
                value = value_T.f();
            }

            fprintf(pp, " 0=%d", top);
            fprintf(pp, " 1=%d", bottom);
            fprintf(pp, " 2=%d", left);
            fprintf(pp, " 3=%d", right);
            fprintf(pp, " 4=%d", type);
            fprintf(pp, " 5=%f", value);
        }
        else if (node.op() == "Placeholder")
        {
            // TODO pass through
            fprintf(pp, " 0=0 1=0 2=0");
        }
        else if (node.op() == "Prod")
        {
            int operation = 6;
            int dim = 0;
            float coeff = 1.f;

            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                dim = parse_tensor_reduction_dim(tensor);
            }

            fprintf(pp, " 0=%d", operation);
            fprintf(pp, " 1=%d", dim);
            fprintf(pp, " 2=%f", coeff);
        }
        else if (node.op() == "Reciprocal")
        {
            int op_type = 15;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "Relu")
        {
            float slope = 0.f;
            fprintf(pp, " 0=%f", slope);
        }
        else if (node.op() == "Reshape")
        {
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                if (!tensor.tensor_content().empty() && tensor.dtype() == 3)// int32
                {
                    const int* data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                    int size = tensor.tensor_content().size() / sizeof(int);

                    // n h w c
                    // n h w
                    // n w
                    if (size == 4)
                    {
                        fprintf(pp, " 0=%d 1=%d 2=%d 3=0", data[2], data[1], data[3]);
                    }
                    if (size == 3)
                    {
                        fprintf(pp, " 0=%d 1=%d 2=-233 3=1", data[2], data[1]);
                    }
                    if (size == 2)
                    {
                        fprintf(pp, " 0=%d 1=-233 2=-233 3=1", data[1]);
                    }
                }
            }
            else
            {
                // pass through
                fprintf(pp, " 0=0 1=0 2=0 3=0");
            }
        }
        else if (node.op() == "Rsqrt")
        {
            int op_type = 6;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "Sigmoid")
        {
        }
        else if (node.op() == "Softmax")
        {
        }
        else if (node.op() == "Square")
        {
            int op_type = 4;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "Squeeze")
        {
            int squeeze_w = 0;
            int squeeze_h = 0;
            int squeeze_c = 0;

            tensorflow::AttrValue value_squeeze_dims;
            if (find_attr_value(node, "squeeze_dims", value_squeeze_dims))
            {
                for (int i = 0; i<value_squeeze_dims.list().i_size(); i++)
                {
                    int dim = value_squeeze_dims.list().i(i);
                    if (dim == 0)
                        squeeze_w = 1;
                    if (dim == 1)
                        squeeze_h = 1;
                    if (dim == 2)
                        squeeze_c = 1;
                }
            }

            fprintf(pp, " 0=%d", squeeze_w);
            fprintf(pp, " 1=%d", squeeze_h);
            fprintf(pp, " 2=%d", squeeze_c);
        }
        else if (node.op() == "Sub")
        {
            int op_type = 1;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (node.op() == "Sum")
        {
            int operation = 0;
            int dim = 0;
            float coeff = 1.f;

            // check weights
            tensorflow::TensorProto tensor;
            if (find_tensor_proto(weights, node, tensor))
            {
                dim = parse_tensor_reduction_dim(tensor);
            }

            fprintf(pp, " 0=%d", operation);
            fprintf(pp, " 1=%d", dim);
            fprintf(pp, " 2=%f", coeff);
        }
        else
        {
            const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node.attr();

            google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.begin();
            for (; it != attr.end(); it++)
            {
                std::cerr << it->first << " #" << it->second.type() << std::endl;
            }
        }

        fprintf(pp, "\n");

        std::string output_name = node.name();
        if (node_reference.find(output_name) != node_reference.end())
        {
            int refcount = node_reference[output_name];
            if (refcount > 1)
            {
                char splitname[256];
                sprintf(splitname, "splitncnn_%d", internal_split);
                fprintf(pp, "%-16s %-32s %d %d", "Split", splitname, 1, refcount);
                fprintf(pp, " %s", output_name.c_str());

                for (int j=0; j<refcount; j++)
                {
                    fprintf(pp, " %s_splitncnn_%d", output_name.c_str(), j);
                }
                fprintf(pp, "\n");

                internal_split++;
            }
        }
    }

    fclose(pp);
    fclose(bp);

    return 0;
}
