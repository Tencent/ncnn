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
#include <stdint.h>
#include <string.h>

#include <map>
#include <set>
#include <string>
#include <vector>

class MXNetNode
{
public:
    std::string op;
    std::string name;
    std::map<std::string, std::string> attrs;
    std::vector<int> inputs;
};

class MXNetParam
{
public:
    std::string name;
    std::vector<float> data;
};

static void replace_backslash_doublequote_dollar(char* s)
{
    char* a = s;
    char* b = s+1;
    while (*a && *b)
    {
        if (*a == '\\' && *b == '\"')
        {
            *b = '$';
        }

        a++;
        b++;
    }
}

static std::vector<int> parse_input_list(const char* s)
{
    std::vector<int> inputs;

    if (memcmp(s, "[]", 2) == 0)
        return inputs;

    int nscan = 0;
    int nconsumed = 0;

    int id;

    int c = 1;// skip leading [
    nscan = sscanf(s + c, "[%d, %*d, %*d]%n", &id, &nconsumed);
    while (nscan == 1)
    {
        inputs.push_back(id);
//         fprintf(stderr, "%d\n", id);

        c += nconsumed;
        nscan = sscanf(s + c, "%*[^[][%d, %*d, %*d]%n", &id, &nconsumed);
    }

    return inputs;
}

static bool read_mxnet_json(const char* jsonpath, std::vector<MXNetNode>& nodes)
{
    FILE* fp = fopen(jsonpath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", jsonpath);
        return false;
    }

    int internal_unknown = 0;

    char line[1024];

    //{
    fgets(line, 1024, fp);

    MXNetNode n;

    bool in_nodes_list = false;
    bool in_node_block = false;
    bool in_attr_block = false;
    while (!feof(fp))
    {
        char* s = fgets(line, 1024, fp);
        if (!s)
            break;

        if (in_attr_block)
        {
            //      },
            if (memcmp(line, "      }", 7) == 0)
            {
                in_attr_block = false;
                continue;
            }

            // replace \" with \$
            replace_backslash_doublequote_dollar(line);

            //        "kernel": "(7,7)",
            char key[256] = {0};
            char value[256] = {0};
            int nscan = sscanf(line, "        \"%255[^\"]\": \"%255[^\"]\",", key, value);
            if (nscan == 2)
            {
                n.attrs[key] = value;
//                 fprintf(stderr, "# %s = %s\n", key, value);
                continue;
            }
        }

        if (in_node_block)
        {
            //    },
            if (memcmp(line, "    }", 5) == 0)
            {
                // new node
                if (n.name.empty())
                {
                    // assign default unknown name
                    char unknownname[256];
                    sprintf(unknownname, "unknownncnn_%d", internal_unknown);

                    n.name = unknownname;

                    internal_unknown++;
                }
                nodes.push_back(n);

                in_node_block = false;
                continue;
            }

            int nscan;

            //      "op": "Convolution",
            char op[256] = {0};
            nscan = sscanf(line, "      \"op\": \"%255[^\"]\",", op);
            if (nscan == 1)
            {
                n.op = op;
//                 fprintf(stderr, "op = %s\n", op);
                continue;
            }

            //      "name": "conv0",
            char name[256] = {0};
            nscan = sscanf(line, "      \"name\": \"%255[^\"]\",", name);
            if (nscan == 1)
            {
                n.name = name;
//                 fprintf(stderr, "name = %s\n", name);
                continue;
            }

            //      "inputs": []
            char inputs[256] = {0};
            nscan = sscanf(line, "      \"inputs\": %255[^\n]", inputs);
            if (nscan == 1)
            {
                n.inputs = parse_input_list(inputs);
//                 fprintf(stderr, "inputs = %s\n", inputs);
                continue;
            }

            // replace \" with \$
            replace_backslash_doublequote_dollar(line);

            //      "attr": {"__init__": "[\"zero\", {}]"},
            char key[256] = {0};
            char value[256] = {0};
            nscan = sscanf(line, "      \"attr\": {\"%255[^\"]\": \"%255[^\"]\"},", key, value);
            if (nscan == 2)
            {
                n.attrs[key] = value;
//                 fprintf(stderr, "# %s = %s\n", key, value);
                continue;
            }

            //      "attr": {
            if (memcmp(line, "      \"attr\": {", 15) == 0)
            {
                in_attr_block = true;
                continue;
            }

        }

        if (in_nodes_list)
        {
            //  ],
            if (memcmp(line, "  ],", 4) == 0)
            {
                in_nodes_list = false;
                // all nodes parsed
                break;
            }

            //    {
            if (memcmp(line, "    {", 5) == 0)
            {
                n = MXNetNode();

                in_node_block = true;
                continue;
            }

        }

        //  "nodes": [
        if (memcmp(line, "  \"nodes\": [", 12) == 0)
        {
            in_nodes_list = true;
            continue;
        }

    }

    fclose(fp);

    return true;
}

static bool read_mxnet_param(const char* parampath, std::vector<MXNetParam>& params)
{
    FILE* fp = fopen(parampath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", parampath);
        return false;
    }

    uint64_t header;
    uint64_t reserved;
    fread(&header, 1, sizeof(uint64_t), fp);
    fread(&reserved, 1, sizeof(uint64_t), fp);

    // NDArray vec

    // each data
    uint64_t data_count;
    fread(&data_count, 1, sizeof(uint64_t), fp);

//     fprintf(stderr, "data count = %d\n", (int)data_count);

    for (int i = 0; i < (int)data_count; i++)
    {
        uint32_t magic;// 0xF993FAC9
        fread(&magic, 1, sizeof(uint32_t), fp);

        int32_t stype;
        fread(&stype, 1, sizeof(int32_t), fp);

        // shape
        uint32_t ndim;
        fread(&ndim, 1, sizeof(uint32_t), fp);

        std::vector<int64_t> shape;
        shape.resize(ndim);
        fread(&shape[0], 1, ndim * sizeof(int64_t), fp);

        // context
        int32_t dev_type;
        int32_t dev_id;
        fread(&dev_type, 1, sizeof(int32_t), fp);
        fread(&dev_id, 1, sizeof(int32_t), fp);

        int32_t type_flag;
        fread(&type_flag, 1, sizeof(int32_t), fp);

        // data
        size_t len = 0;
        if (shape.size() == 1) len = shape[0];
        if (shape.size() == 2) len = shape[0] * shape[1];
        if (shape.size() == 3) len = shape[0] * shape[1] * shape[2];
        if (shape.size() == 4) len = shape[0] * shape[1] * shape[2] * shape[3];

        MXNetParam p;

        p.data.resize(len);
        fread(&p.data[0], 1, len * sizeof(float), fp);

        params.push_back(p);

//         fprintf(stderr, "%u read\n", len);
    }

    // each name
    uint64_t name_count;
    fread(&name_count, 1, sizeof(uint64_t), fp);

//     fprintf(stderr, "name count = %d\n", (int)name_count);

    for (int i = 0; i < (int)name_count; i++)
    {
        uint64_t len;
        fread(&len, 1, sizeof(uint64_t), fp);

        MXNetParam& p = params[i];

        p.name.resize(len);
        fread((char*)p.name.data(), 1, len, fp);

//         fprintf(stderr, "%s read\n", name.c_str());
    }

    fclose(fp);

    return true;
}

static bool find_param(const std::vector<MXNetParam>& params, const std::string& name, MXNetParam& p)
{
    for (int i=0; i<params.size(); i++)
    {
        if (params[i].name == name)
        {
            p = params[i];
            return true;
        }
    }

    return false;
}

static bool vector_has(const std::vector<int>& v, int id)
{
    for (int i=0; i<v.size(); i++)
    {
        if (v[i] == id)
            return true;
    }

    return false;
}

int main(int argc, char** argv)
{
    const char* jsonpath = argv[1];
    const char* parampath = argv[2];
    const char* ncnn_prototxt = argc >= 5 ? argv[3] : "ncnn.proto";
    const char* ncnn_modelbin = argc >= 5 ? argv[4] : "ncnn.bin";

    std::vector<MXNetNode> nodes;
    std::vector<MXNetParam> params;

    read_mxnet_json(jsonpath, nodes);
    read_mxnet_param(parampath, params);

    FILE* pp = fopen(ncnn_prototxt, "wb");
    FILE* bp = fopen(ncnn_modelbin, "wb");

    // magic
    fprintf(pp, "7767517\n");

    int node_count = nodes.size();

    // node reference
    std::map<int, int> node_reference;

    // input node
    std::vector<int> input_nodes;

    // weight node
    std::vector<int> weight_nodes;

    // weight init node
    std::vector<int> weight_init_nodes;

    // global definition line
    // [layer count] [blob count]
    std::set<std::string> blob_names;
    for (int i=0; i<node_count; i++)
    {
        const MXNetNode& n = nodes[i];

        const std::string& output_name = n.name;

        if (n.op == "null")
        {
            MXNetParam p;
            if (find_param(params, output_name, p))
            {
                weight_nodes.push_back(i);
            }
            else
            {
                if (n.attrs.find("__init__") != n.attrs.end())
                {
                    weight_init_nodes.push_back(i);
                }
                else
                {
                    // null node without data, treat it as network input
                    input_nodes.push_back(i);
                }
            }
            continue;
        }

        // input
        for (int j=0; j<n.inputs.size(); j++)
        {
            int input_index = n.inputs[j];
            if (vector_has(weight_nodes, input_index))
            {
                continue;
            }
            if (vector_has(weight_init_nodes, input_index))
            {
                continue;
            }

            const std::string& input_name = nodes[input_index].name;
//             fprintf(stderr, "input = %s\n", input_name.c_str());
            blob_names.insert(input_name);

            if (node_reference.find(input_index) == node_reference.end())
            {
                node_reference[input_index] = 1;
            }
            else
            {
                node_reference[input_index] = node_reference[input_index] + 1;
            }
        }

        // output
//         fprintf(stderr, "output = %s\n", output_name.c_str());
        blob_names.insert(output_name);
    }

    // remove node_reference entry with reference equals to one
    int splitncnn_blob_count = 0;
    std::map<int, int>::iterator it = node_reference.begin();
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

    fprintf(pp, "%lu %lu\n", node_count + node_reference.size() + input_nodes.size() - weight_nodes.size() - weight_init_nodes.size(), blob_names.size() + input_nodes.size() + splitncnn_blob_count);

    int internal_split = 0;

    for (int i=0; i<node_count; i++)
    {
        const MXNetNode& n = nodes[i];

        if (n.op == "null")
        {
            if (vector_has(weight_nodes, i))
            {
                continue;
            }
            if (vector_has(weight_init_nodes, i))
            {
                continue;
            }

            if (vector_has(input_nodes, i))
            {
                fprintf(pp, "%-16s", "Input");
            }
        }
        else
        {
            fprintf(pp, "%-16s", n.op.c_str());
        }

        int input_size = n.inputs.size();
        for (int j=0; j<n.inputs.size(); j++)
        {
            int input_index = n.inputs[j];
            if (vector_has(weight_nodes, input_index))
            {
                input_size--;
            }
            if (vector_has(weight_init_nodes, input_index))
            {
                input_size--;
            }
        }

        fprintf(pp, " %-32s %d 1", n.name.c_str(), input_size);

        for (int j=0; j<n.inputs.size(); j++)
        {
            int input_index = n.inputs[j];
            if (vector_has(weight_nodes, input_index))
            {
                continue;
            }
            if (vector_has(weight_init_nodes, input_index))
            {
                continue;
            }

            std::string input_name = nodes[input_index].name;

            if (node_reference.find(input_index) != node_reference.end())
            {
                int refidx = node_reference[input_index] - 1;
                node_reference[input_index] = refidx;

                char splitsuffix[256];
                sprintf(splitsuffix, "_splitncnn_%d", refidx);
                input_name = input_name + splitsuffix;
            }

            fprintf(pp, " %s", input_name.c_str());
        }

        fprintf(pp, " %s", n.name.c_str());

        // TODO op specific params

        fprintf(pp, "\n");

        if (node_reference.find(i) != node_reference.end())
        {
            int refcount = node_reference[i];
            if (refcount > 1)
            {
                std::string output_name = n.name;

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
