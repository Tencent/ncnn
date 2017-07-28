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

static const tensorflow::TensorProto& find_tensor_proto(const std::map<std::string, tensorflow::TensorProto>& weights, const tensorflow::NodeDef& node)
{
    for (int j=0; j<node.input_size(); j++)
    {
        const std::string& input_name = node.input(j);

        const std::map<std::string, tensorflow::TensorProto>::const_iterator it = weights.find(input_name);
        if (it != weights.end())
        {
            const tensorflow::TensorProto& tensor = it->second;
            return tensor;
        }
    }

    static tensorflow::TensorProto null_tensor = tensorflow::TensorProto();
    return null_tensor;
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

    FILE* pp = stderr;//fopen(ncnn_prototxt, "wb");
    FILE* bp = stderr;//fopen(ncnn_modelbin, "wb");

    int node_count = graph.node_size();

//     fprintf(stderr, "node_count = %d\n\n", node_count);

    // node reference
    std::map<std::string, int> node_reference;

    // mapping for Const and Const-Identity
    std::map<std::string, tensorflow::TensorProto> weights;

    // global definition line
    // [layer count] [blob count]
    std::set<std::string> blob_names;
    for (int i=0; i<node_count; i++)
    {
        const tensorflow::NodeDef& node = graph.node(i);

        const std::string& output_name = node.name();

        if (node.op() == "Const")
        {
            const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node.attr();

            const google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.find("value");
            if (it != attr.end())
            {
                const tensorflow::TensorProto& tensor = it->second.tensor();

                weights[output_name] = tensor;
            }

            continue;
        }
        else if (node.op() == "Identity")
        {
            const std::string& input_name = node.input(0);
            weights[output_name] = weights[input_name];
            continue;
        }
        else if (node.op() == "NoOp")
        {
            weights[output_name] = tensorflow::TensorProto();
            continue;
        }

        // input
        for (int j=0; j<node.input_size(); j++)
        {
            const std::string& input_name = node.input(j);
//             fprintf(stderr, "%s\n", input_name.c_str());

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
//         fprintf(stderr, "%s\n", output_name.c_str());
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

        if (node.op() == "Add")
        {
            fprintf(pp, "%-16s", "Eltwise");
        }
        else if (node.op() == "BiasAdd")
        {
            fprintf(pp, "%-16s", "Eltwise");
        }
        else if (node.op() == "Const")
        {
            continue;
        }
        else if (node.op() == "Conv2D")
        {
            fprintf(pp, "%-16s", "Convolution");
        }
        else if (node.op() == "Identity")
        {
            continue;
        }
        else if (node.op() == "MatMul")
        {
            fprintf(pp, "%-16s", "InnerProduct");
        }
        else if (node.op() == "Max")
        {
            fprintf(pp, "%-16s", "Eltwise");
        }
        else if (node.op() == "MaxPool")
        {
            fprintf(pp, "%-16s", "Pooling");
        }
        else if (node.op() == "Mul")
        {
            fprintf(pp, "%-16s", "Eltwise");
        }
        else if (node.op() == "NoOp")
        {
            continue;
        }
        else if (node.op() == "Placeholder")
        {
            fprintf(pp, "%-16s", "Input");
        }
        else if (node.op() == "Relu")
        {
            fprintf(pp, "%-16s", "ReLU");
        }
        else
        {
            fprintf(pp, "%-16s", node.op().c_str());
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

        fprintf(pp, " %-16s %d 1", node.name().c_str(), input_size);

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

        fprintf(pp, " %s\n", node.name().c_str());

        if (node.op() == "Add")
        {
        }
        else if (node.op() == "BiasAdd")
        {
        }
        else if (node.op() == "Const")
        {
        }
        else if (node.op() == "Conv2D")
        {
            // weights
            const tensorflow::TensorProto& tensor = find_tensor_proto(weights, node);

            fprintf(stderr, "[ ");
            const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();
            for (int d = 0; d<shape.dim_size(); d++)
                fprintf(stderr, "%d ", shape.dim(d).size());
            fprintf(stderr, "]\n");

            if (!tensor.tensor_content().empty())
            {
                switch (tensor.dtype())
                {
                    case 1:  // float
                    {
                        const float *data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                        int size = tensor.tensor_content().size() / sizeof(float);
                        fprintf(stderr, "       size = %d\n", size);
                        break;
                    }
                    case 3:  // int32
                    {
                        const int *data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                        int size = tensor.tensor_content().size() / sizeof(int);
                        fprintf(stderr, "       size = %d\n", size);
                        break;
                    }
                    default:
                        fprintf(stderr, "Tensor type is not supported\n");
                        break;
                }
            }
        }
        else if (node.op() == "Identity")
        {
        }
        else if (node.op() == "MatMul")
        {
            // weights
            const tensorflow::TensorProto& tensor = find_tensor_proto(weights, node);

            fprintf(stderr, "[ ");
            const tensorflow::TensorShapeProto& shape = tensor.tensor_shape();
            for (int d = 0; d<shape.dim_size(); d++)
                fprintf(stderr, "%d ", shape.dim(d).size());
            fprintf(stderr, "]\n");

            if (!tensor.tensor_content().empty())
            {
                switch (tensor.dtype())
                {
                    case 1:  // float
                    {
                        const float *data = reinterpret_cast<const float*>(tensor.tensor_content().c_str());
                        int size = tensor.tensor_content().size() / sizeof(float);
                        fprintf(stderr, "       size = %d\n", size);
                        break;
                    }
                    case 3:  // int32
                    {
                        const int *data = reinterpret_cast<const int*>(tensor.tensor_content().c_str());
                        int size = tensor.tensor_content().size() / sizeof(int);
                        fprintf(stderr, "       size = %d\n", size);
                        break;
                    }
                    default:
                        fprintf(stderr, "Tensor type is not supported\n");
                        break;
                }
            }
        }
        else if (node.op() == "Max")
        {
        }
        else if (node.op() == "MaxPool")
        {
        }
        else if (node.op() == "Mul")
        {
        }
        else if (node.op() == "NoOp")
        {
        }
        else if (node.op() == "Placeholder")
        {
        }
        else if (node.op() == "Relu")
        {
        }
        else
        {
            const google::protobuf::Map<std::string, tensorflow::AttrValue>& attr = node.attr();

            google::protobuf::Map<std::string, tensorflow::AttrValue>::const_iterator it = attr.begin();
            for (; it != attr.end(); it++)
            {
                std::cerr << it->first << std::endl;
                std::cerr << it->second.type() << std::endl;
            }
        }

        std::string output_name = node.name();
        if (node_reference.find(output_name) != node_reference.end())
        {
            int refcount = node_reference[output_name];
            if (refcount > 1)
            {
                char splitname[256];
                sprintf(splitname, "splitncnn_%d", internal_split);
                fprintf(pp, "%-16s %-16s %d %d", "Split", splitname, 1, refcount);
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
