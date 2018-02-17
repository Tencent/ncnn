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

#include "onnx.pb.h"

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
    const char* onnxpb = argv[1];
    const char* ncnn_prototxt = argc >= 4 ? argv[2] : "ncnn.param";
    const char* ncnn_modelbin = argc >= 4 ? argv[3] : "ncnn.bin";

    onnx::ModelProto model;

    // load
    bool s1 = read_proto_from_binary(onnxpb, &model);
    if (!s1)
    {
        fprintf(stderr, "read_proto_from_binary failed\n");
        return -1;
    }

    // magic
    fprintf(stderr, "7767517\n");

    const onnx::GraphProto& graph = model.graph();

    int node_count = graph.node_size();

    // node reference
    std::map<std::string, int> node_reference;

    // weight node
    std::set<std::string> weight_nodes;

    for (int j=0; j<graph.initializer_size(); j++)
    {
        const onnx::TensorProto& initializer = graph.initializer(j);

        weight_nodes.insert(initializer.name());
    }

    // global definition line
    // [layer count] [blob count]
    std::set<std::string> blob_names;
    for (int i=0; i<node_count; i++)
    {
        const onnx::NodeProto& node = graph.node(i);

        for (int j=0; j<(int)node.input_size(); j++)
        {
            const std::string& input_name = node.input(j);

            // check weight
            if (weight_nodes.find(input_name) != weight_nodes.end())
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

        for (int j=0; j<(int)node.output_size(); j++)
        {
            const std::string& output_name = node.output(j);

            blob_names.insert(output_name);
        }
    }

    // include Input node
    for (int j=0; j<graph.input_size(); j++)
    {
        const std::string& input_name = graph.input(j).name();

        // check weight
        if (weight_nodes.find(input_name) != weight_nodes.end())
            continue;

        blob_names.insert(input_name);
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

    fprintf(stderr, "%lu %lu\n", node_count + node_reference.size(), blob_names.size() + splitncnn_blob_count);

    int internal_split = 0;

    // place Input at the beginning
    for (int j=0; j<graph.input_size(); j++)
    {
        const std::string& input_name = graph.input(j).name();

        // check weight
        if (weight_nodes.find(input_name) != weight_nodes.end())
            continue;

        fprintf(stderr, "%-16s %-24s 0 1 %s\n", "Input", input_name.c_str(), input_name.c_str());
    }

    for (int i=0; i<node_count; i++)
    {
        const onnx::NodeProto& node = graph.node(i);

        const std::string& op = node.op_type();

        std::string name = node.name();
        if (name.empty())
        {
            name = node.output(0);
        }

        int input_size = node.input_size();
        int output_size = node.output_size();

        for (int j=0; j<(int)node.input_size(); j++)
        {
            const std::string& input_name = node.input(j);

            // check weight
            if (weight_nodes.find(input_name) != weight_nodes.end())
            {
                input_size--;
            }

//             fprintf(stderr, "  input = %s\n", input_name.c_str());
        }

        for (int j=0; j<(int)node.output_size(); j++)
        {
            const std::string& output_name = node.output(j);

//             fprintf(stderr, "  output = %s\n", output_name.c_str());
        }

        if (op == "Conv")
        {
            fprintf(stderr, "%-16s", "Convolution");
        }
        else if (op == "BatchNormalization")
        {
            fprintf(stderr, "%-16s", "BatchNorm");
        }
        else
        {
            // TODO
            fprintf(stderr, "%-16s", op.c_str());
        }

        fprintf(stderr, " %-24s %d %d", name.c_str(), input_size, output_size);

        for (int j=0; j<node.input_size(); j++)
        {
            std::string input_name = node.input(j);

            // check weight
            if (weight_nodes.find(input_name) != weight_nodes.end())
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

            fprintf(stderr, " %s", input_name.c_str());
        }

        for (int j=0; j<node.output_size(); j++)
        {
            const std::string& output_name = node.output(j);

            fprintf(stderr, " %s", output_name.c_str());
        }

        // TODO op specific param
        for (int j=0; j<node.attribute_size(); j++)
        {
            onnx::AttributeProto attr = node.attribute(j);
//             fprintf(stderr, "  # %s %d\n", attr.name().c_str(), attr.type());
        }

        fprintf(stderr, "\n");

        for (int j=0; j<output_size; j++)
        {
            const std::string& output_name = node.output(j);
            if (node_reference.find(output_name) != node_reference.end())
            {
                int refcount = node_reference[output_name];
                if (refcount > 1)
                {
                    char splitname[256];
                    sprintf(splitname, "splitncnn_%d", internal_split);
                    fprintf(stderr, "%-16s %-24s %d %d", "Split", splitname, 1, refcount);

                    fprintf(stderr, " %s", output_name.c_str());

                    for (int k=0; k<refcount; k++)
                    {
                        fprintf(stderr, " %s_splitncnn_%d", output_name.c_str(), k);
                    }
                    fprintf(stderr, "\n");

                    internal_split++;
                }
            }
        }
    }

    return 0;
}
