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

class MXNetParam;
class MXNetNode
{
public:
    bool has_attr(const char* key) const;

    class AttrProxy
    {
        MXNetNode const* _n;
        const char* const _key;
    public:
        AttrProxy( MXNetNode const* n, const char* key ) : _n(n), _key(key) {}
        operator int() const { return _n->attr_i(_key); }
        operator float() const { return _n->attr_f(_key); }
        operator std::string() const { return _n->attr_s(_key); }
        operator std::vector<int>() const { return _n->attr_ai(_key); }
    };

    AttrProxy attr(const char* key) const { return AttrProxy(this, key); }

    int attr_i(const char* key) const;
    float attr_f(const char* key) const;
    std::string attr_s(const char* key) const;
    std::vector<int> attr_ai(const char* key) const;

public:
    bool is_weight() const;
    bool has_weight(int i) const;
    std::vector<float> weight(int i, int init_len = 0) const;

    std::vector<MXNetNode>* nodes;// reference
    std::vector<MXNetParam>* params;// reference

public:
    std::string op;
    std::string name;
    std::map<std::string, std::string> attrs;
    std::vector<int> inputs;
    std::vector<int> subinputs;
    std::vector<int> weights;
};

class MXNetParam
{
public:
    std::string name;
    std::vector<float> data;
    std::string init;
};

bool MXNetNode::has_attr(const char* key) const
{
    const std::map<std::string, std::string>::const_iterator it = attrs.find(key);
    return it != attrs.end();
}

int MXNetNode::attr_i(const char* key) const
{
    const std::map<std::string, std::string>::const_iterator it = attrs.find(key);
    if (it == attrs.end())
        return 0;

    if (it->second == "False")
        return 0;

    if (it->second == "True")
        return 1;

    int i = 0;
    int nscan = sscanf(it->second.c_str(), "%d", &i);
    if (nscan != 1)
        return 0;

    return i;
}

float MXNetNode::attr_f(const char* key) const
{
    const std::map<std::string, std::string>::const_iterator it = attrs.find(key);
    if (it == attrs.end())
        return 0.f;

    float f = 0;
    int nscan = sscanf(it->second.c_str(), "%f", &f);
    if (nscan != 1)
        return 0.f;

    return f;
}

std::string MXNetNode::attr_s(const char* key) const
{
    const std::map<std::string, std::string>::const_iterator it = attrs.find(key);
    if (it == attrs.end())
        return std::string();

    return it->second;
}

std::vector<int> MXNetNode::attr_ai(const char* key) const
{
    const std::map<std::string, std::string>::const_iterator it = attrs.find(key);
    if (it == attrs.end())
        return std::vector<int>();

    // (1,2,3)
    std::vector<int> list;

    int i = 0;
    int c = 0;
    int nconsumed = 0;
    int nscan = sscanf(it->second.c_str() + c, "%*[(,]%d%n", &i, &nconsumed);
    while (nscan == 1)
    {
        list.push_back(i);
//         fprintf(stderr, "%d\n", i);

        i = 0;
        c += nconsumed;
        nscan = sscanf(it->second.c_str() + c, "%*[(,]%d%n", &i, &nconsumed);
    }

    return list;
}

bool MXNetNode::is_weight() const
{
    for (int i=0; i<(int)(*params).size(); i++)
    {
        const MXNetParam& p = (*params)[i];
        if (p.name == name)
            return true;
    }

    return false;
}

bool MXNetNode::has_weight(int i) const
{
    if (i < 0 || i >= (int)weights.size())
        return false;

    const std::string& name = (*nodes)[ weights[i] ].name;

    for (int i=0; i<(int)(*params).size(); i++)
    {
        const MXNetParam& p = (*params)[i];
        if (p.name == name)
            return true;
    }

    return false;
}

std::vector<float> MXNetNode::weight(int i, int init_len) const
{
    if (i < 0 || i >= (int)weights.size())
        return std::vector<float>();

    const std::string& name = (*nodes)[ weights[i] ].name;

    for (int i=0; i<(int)(*params).size(); i++)
    {
        const MXNetParam& p = (*params)[i];
        if (p.name != name)
            continue;

        if (!p.data.empty())
            return p.data;

        std::vector<float> data;

        if (!p.init.empty() && init_len != 0)
        {
            if (p.init == "[\\$zero\\$, {}]")
            {
                data.resize(init_len, 0.f);
            }
            else if (p.init == "[\\$one\\$, {}]")
            {
                data.resize(init_len, 1.f);
            }
        }

        return data;
    }

    return std::vector<float>();
}

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

static void parse_input_list(const char* s, std::vector<int>& inputs, std::vector<int>& subinputs)
{
    inputs.clear();
    subinputs.clear();

    if (memcmp(s, "[]", 2) == 0)
        return;

    int nscan = 0;
    int nconsumed = 0;

    int id;
    int subid;

    int c = 1;// skip leading [
    nscan = sscanf(s + c, "[%d, %d%n", &id, &subid, &nconsumed);
    while (nscan == 2)
    {
        inputs.push_back(id);
        subinputs.push_back(subid);
//         fprintf(stderr, "%d %d\n", id, subid);

        c += nconsumed;
        nscan = sscanf(s + c, "%*[^[][%d, %d%n", &id, &subid, &nconsumed);
    }
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
            int nscan = sscanf(line, "        \"%255[^\"]\": \"%255[^\"]\"", key, value);
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
                parse_input_list(inputs, n.inputs, n.subinputs);
//                 fprintf(stderr, "inputs = %s\n", inputs);
                continue;
            }

            //      "param": {},
            if (memcmp(line, "      \"param\": {}", 17) == 0)
            {
                continue;
            }

            // replace \" with \$
            replace_backslash_doublequote_dollar(line);

            //      "attr": {"__init__": "[\"zero\", {}]"},
            char key[256] = {0};
            char value[256] = {0};
            nscan = sscanf(line, "      \"attr\": {\"%255[^\"]\": \"%255[^\"]\"}", key, value);
            if (nscan == 2)
            {
                n.attrs[key] = value;
//                 fprintf(stderr, "# %s = %s\n", key, value);
                continue;
            }

            //      "attrs": {"__init__": "[\"zero\", {}]"},
            nscan = sscanf(line, "      \"attrs\": {\"%255[^\"]\": \"%255[^\"]\"}", key, value);
            if (nscan == 2)
            {
                n.attrs[key] = value;
//                 fprintf(stderr, "# %s = %s\n", key, value);
                continue;
            }

            //      "param": {"p": "0.5"},
            nscan = sscanf(line, "      \"param\": {\"%255[^\"]\": \"%255[^\"]\"}", key, value);
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

            //      "attrs": {
            if (memcmp(line, "      \"attrs\": {", 15) == 0)
            {
                in_attr_block = true;
                continue;
            }

            //      "param": {
            if (memcmp(line, "      \"param\": {", 16) == 0)
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

        // shape
        uint32_t ndim;
        std::vector<int64_t> shape;

        if (magic == 0xF993FAC9)
        {
            int32_t stype;
            fread(&stype, 1, sizeof(int32_t), fp);

            fread(&ndim, 1, sizeof(uint32_t), fp);

            shape.resize(ndim);
            fread(&shape[0], 1, ndim * sizeof(int64_t), fp);
        }
        else if (magic == 0xF993FAC8)
        {
            fread(&ndim, 1, sizeof(uint32_t), fp);

            shape.resize(ndim);
            fread(&shape[0], 1, ndim * sizeof(int64_t), fp);
        }
        else
        {
            ndim = magic;

            shape.resize(ndim);

            std::vector<uint32_t> shape32;
            shape32.resize(ndim);
            fread(&shape32[0], 1, ndim * sizeof(uint32_t), fp);

            for (int j=0; j<(int)ndim; j++)
            {
                shape[j] = shape32[j];
            }
        }

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

        // cut leading arg:
        if (memcmp(p.name.c_str(), "arg:", 4) == 0)
        {
            p.name = std::string(p.name.c_str() + 4);
        }
        if (memcmp(p.name.c_str(), "aux:", 4) == 0)
        {
            p.name = std::string(p.name.c_str() + 4);
        }

//         fprintf(stderr, "%s read\n", p.name.c_str());
    }

    fclose(fp);

    return true;
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

    // weight node
    std::vector<int> weight_nodes;

    // global definition line
    // [layer count] [blob count]
    std::set<std::string> blob_names;
    for (int i=0; i<node_count; i++)
    {
        MXNetNode& n = nodes[i];

        // assign global param reference
        n.nodes = &nodes;
        n.params = &params;

        const std::string& output_name = n.name;
        int output_size = 1;

        if (n.op == "null")
        {
            if (n.is_weight())
            {
                weight_nodes.push_back(i);
            }
            else
            {
                if (n.has_attr("__init__"))
                {
                    // init weight param
                    MXNetParam pi;
                    pi.name = n.name;
                    pi.init = (std::string)n.attr("__init__");
                    params.push_back(pi);

                    weight_nodes.push_back(i);
                }
                else
                {
                    // null node without data, treat it as network input
                }
            }
            continue;
        }
        else if (n.op == "SliceChannel")
        {
            output_size = n.attr("num_outputs");
        }

        // distinguish weights and inputs
        std::vector<int> weights;
        std::vector<int> inputs;
        for (int j=0; j<(int)n.inputs.size(); j++)
        {
            int input_index = n.inputs[j];
            if (nodes[input_index].is_weight())
            {
                weights.push_back(input_index);
                continue;
            }

            inputs.push_back(input_index);
        }
        n.inputs = inputs;
        n.weights = weights;

        // input
        for (int j=0; j<(int)n.inputs.size(); j++)
        {
            int input_index = n.inputs[j];
            int subinput_index = n.subinputs[j];

            std::string input_name = nodes[input_index].name;
//             fprintf(stderr, "input = %s\n", input_name.c_str());

            if (subinput_index != 0)
            {
                char subinputsuffix[256];
                sprintf(subinputsuffix, "_%d", subinput_index);
                input_name = input_name + subinputsuffix;
            }

            blob_names.insert(input_name);

            int input_uid = input_index | (subinput_index << 16);
            if (node_reference.find(input_uid) == node_reference.end())
            {
                node_reference[input_uid] = 1;
            }
            else
            {
                node_reference[input_uid] = node_reference[input_uid] + 1;
            }
        }

        // output
//         fprintf(stderr, "output = %s\n", output_name.c_str());
        blob_names.insert(output_name);

        for (int j=1; j<output_size; j++)
        {
            char subinputsuffix[256];
            sprintf(subinputsuffix, "_%d", j);
            std::string output_name_j = output_name + subinputsuffix;
            blob_names.insert(output_name_j);
        }
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

    fprintf(pp, "%lu %lu\n", node_count + node_reference.size() - weight_nodes.size(), blob_names.size() + splitncnn_blob_count);

    int internal_split = 0;

    for (int i=0; i<node_count; i++)
    {
        const MXNetNode& n = nodes[i];

        int output_size = 1;

        if (n.op == "null")
        {
            if (n.is_weight())
            {
                continue;
            }

            fprintf(pp, "%-16s", "Input");
        }
        else if (n.op == "Activation")
        {
            std::string type = n.attr("act_type");
            if (type == "relu")
            {
                fprintf(pp, "%-16s", "ReLU");
            }
            else if (type == "sigmoid")
            {
                fprintf(pp, "%-16s", "Sigmoid");
            }
            else if (type == "tanh")
            {
                fprintf(pp, "%-16s", "TanH");
            }
        }
        else if (n.op == "BatchNorm")
        {
            fprintf(pp, "%-16s", "BatchNorm");
        }
        else if (n.op == "Concat")
        {
            fprintf(pp, "%-16s", "Concat");
        }
        else if (n.op == "Convolution")
        {
            int num_group = n.attr("num_group");
            if (num_group > 0) {
                fprintf(pp, "%-16s", "ConvolutionDepthWise");
            } else {
                fprintf(pp, "%-16s", "Convolution");
            }
        }
        else if (n.op == "Dropout")
        {
            fprintf(pp, "%-16s", "Dropout");
        }
        else if (n.op == "elemwise_add")
        {
            fprintf(pp, "%-16s", "Eltwise");
        }
        else if (n.op == "elemwise_mul")
        {
            fprintf(pp, "%-16s", "Eltwise");
        }
        else if (n.op == "Flatten")
        {
            fprintf(pp, "%-16s", "Flatten");
        }
        else if (n.op == "FullyConnected")
        {
            fprintf(pp, "%-16s", "InnerProduct");
        }
        else if (n.op == "LeakyReLU")
        {
            std::string type = n.attr("act_type");
            if (type == "elu")
            {
                fprintf(pp, "%-16s", "ELU");
            }
            else if (type == "leaky")
            {
                fprintf(pp, "%-16s", "ReLU");
            }
            else if (type == "prelu")
            {
                fprintf(pp, "%-16s", "PReLU");
            }
        }
        else if (n.op == "Pooling")
        {
            fprintf(pp, "%-16s", "Pooling");
        }
        else if (n.op == "SliceChannel")
        {
            fprintf(pp, "%-16s", "Slice");
            output_size = n.attr("num_outputs");
        }
        else if (n.op == "SoftmaxOutput")
        {
            fprintf(pp, "%-16s", "Softmax");
        }
        else
        {
            fprintf(stderr, "%s not supported yet!\n", n.op.c_str());
            fprintf(pp, "%-16s", n.op.c_str());
        }

        int input_size = n.inputs.size();
        for (int j=0; j<(int)n.inputs.size(); j++)
        {
            int input_index = n.inputs[j];
            if (nodes[input_index].is_weight())
            {
                input_size--;
            }
        }

        if (n.op == "SoftmaxOutput")
        {
            // drop label
            input_size--;
        }

        fprintf(pp, " %-32s %d %d", n.name.c_str(), input_size, output_size);

        for (int j=0; j<(int)n.inputs.size(); j++)
        {
            int input_index = n.inputs[j];
            int subinput_index = n.subinputs[j];
            if (nodes[input_index].is_weight())
            {
                continue;
            }

            if (n.op == "SoftmaxOutput")
            {
                // drop label
                if (j == 1)
                    continue;
            }

            std::string input_name = nodes[input_index].name;

            int input_uid = input_index | (subinput_index << 16);
            if (node_reference.find(input_uid) != node_reference.end())
            {
                int refidx = node_reference[input_uid] - 1;
                node_reference[input_uid] = refidx;

                char splitsuffix[256];
                sprintf(splitsuffix, "_splitncnn_%d", refidx);
                input_name = input_name + splitsuffix;
            }

            if (subinput_index == 0)
            {
                fprintf(pp, " %s", input_name.c_str());
            }
            else
            {
                fprintf(pp, " %s_%d", input_name.c_str(), subinput_index);
            }
        }

        fprintf(pp, " %s", n.name.c_str());
        for (int j=1; j<output_size; j++)
        {
            fprintf(pp, " %s_%d", n.name.c_str(), j);
        }

        if (n.op == "null")
        {
            // dummy input shape
//             fprintf(pp, " 0 0 0");
        }
        else if (n.op == "Activation")
        {
            std::string type = n.attr("act_type");
            if (type == "relu")
            {
//                 fprintf(pp, " 0=%f", 0.f);
            }
        }
        else if (n.op == "BatchNorm")
        {
            float eps = 1e-3;
            if (n.has_attr("eps")) {
                eps = n.attr("eps");
            }

            std::vector<float> slope_data = n.weight(0);
            std::vector<float> bias_data = n.weight(1);

            int channels = slope_data.size();

            std::vector<float> mean_data = n.weight(2, channels);
            std::vector<float> var_data = n.weight(3, channels);

            for (int j=0; j<(int)var_data.size(); j++)
            {
                var_data[j] += eps;
            }

            fprintf(pp, " 0=%d", channels);

            fwrite(slope_data.data(), sizeof(float), slope_data.size(), bp);
            fwrite(mean_data.data(), sizeof(float), mean_data.size(), bp);
            fwrite(var_data.data(), sizeof(float), var_data.size(), bp);
            fwrite(bias_data.data(), sizeof(float), bias_data.size(), bp);
        }
        else if (n.op == "Concat")
        {
            int dim = n.attr("dim");
            fprintf(pp, " 0=%d", dim-1);
        }
        else if (n.op == "Convolution")
        {
            int num_filter = n.attr("num_filter");
            std::vector<int> kernel = n.attr("kernel");
            std::vector<int> dilate = n.attr("dilate");
            std::vector<int> stride = n.attr("stride");
            std::vector<int> pad = n.attr("pad");
            int no_bias = n.attr("no_bias");
            int num_group = n.attr("num_group");//TODO depthwise

            std::vector<float> weight_data = n.weight(0);
            std::vector<float> bias_data = n.weight(1);

            fprintf(pp, " 0=%d", num_filter);
            if (kernel.size() == 1) {
                fprintf(pp, " 1=%d", kernel[0]);
            } else if (kernel.size() == 2) {
                fprintf(pp, " 1=%d", kernel[1]);
                fprintf(pp, " 11=%d", kernel[0]);
            }

            if (dilate.size() == 1) {
                fprintf(pp, " 2=%d", dilate[0]);
            } else if (dilate.size() == 2) {
                fprintf(pp, " 2=%d", dilate[1]);
                fprintf(pp, " 12=%d", dilate[0]);
            }

            if (stride.size() == 1) {
                fprintf(pp, " 3=%d", stride[0]);
            } else if (stride.size() == 2) {
                fprintf(pp, " 3=%d", stride[1]);
                fprintf(pp, " 13=%d", stride[0]);
            }

            if (pad.size() == 1) {
                fprintf(pp, " 4=%d", pad[0]);
            } else if (pad.size() == 2) {
                fprintf(pp, " 4=%d", pad[1]);
                fprintf(pp, " 14=%d", pad[0]);
            }

            fprintf(pp, " 5=%d", no_bias == 1 ? 0 : 1);
            fprintf(pp, " 6=%d", (int)weight_data.size());
            if (num_group > 0) {
                fprintf(pp, " 7=%d", num_group);
            }

            int quantize_tag = 0;
            fwrite(&quantize_tag, sizeof(int), 1, bp);
            fwrite(weight_data.data(), sizeof(float), weight_data.size(), bp);
            fwrite(bias_data.data(), sizeof(float), bias_data.size(), bp);
        }
        else if (n.op == "Dropout")
        {
//             float p = n.attr("p");
//             fprintf(pp, " 0=%d", p);
        }
        else if (n.op == "elemwise_add")
        {
            int op_type = 1;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (n.op == "elemwise_mul")
        {
            int op_type = 0;
            fprintf(pp, " 0=%d", op_type);
        }
        else if (n.op == "Flatten")
        {
        }
        else if (n.op == "FullyConnected")
        {
            int num_hidden = n.attr("num_hidden");
            int no_bias = n.attr("no_bias");
//             int flatten = n.attr("flatten");

            // TODO flatten

            std::vector<float> weight_data = n.weight(0);
            std::vector<float> bias_data = n.weight(1);

            fprintf(pp, " 0=%d", num_hidden);
            fprintf(pp, " 1=%d", no_bias == 1 ? 0 : 1);
            fprintf(pp, " 2=%d", (int)weight_data.size());

            int quantize_tag = 0;
            fwrite(&quantize_tag, sizeof(int), 1, bp);
            fwrite(weight_data.data(), sizeof(float), weight_data.size(), bp);
            fwrite(bias_data.data(), sizeof(float), bias_data.size(), bp);
        }
        else if (n.op == "LeakyReLU")
        {
            std::string type = n.attr("act_type");
            if (type == "elu")
            {
            }
            else if (type == "leaky")
            {
            }
            else if (type == "prelu")
            {
                std::vector<float> weight_data = n.weight(0);

                fprintf(pp, " 0=%d", (int)weight_data.size());

                fwrite(weight_data.data(), sizeof(float), weight_data.size(), bp);
            }
        }
        else if (n.op == "Pooling")
        {
            std::string pool_type = n.attr("pool_type");
            std::vector<int> kernel = n.attr("kernel");
            std::vector<int> stride = n.attr("stride");
            std::vector<int> pad = n.attr("pad");
            std::string pooling_convention = n.attr("pooling_convention");
            int global_pool = n.attr("global_pool");

            int pool = 0;
            if (pool_type == "max")
            {
                pool = 0;
            }
            else if (pool_type == "avg")
            {
                pool = 1;
            }

            if (pooling_convention == "valid")
            {
                // TODO valid and full mode
            }

            fprintf(pp, " 0=%d", pool);
            if (!kernel.empty())
                fprintf(pp, " 1=%d", kernel[0]);
            if (!stride.empty())
                fprintf(pp, " 2=%d", stride[0]);
            if (!pad.empty())
                fprintf(pp, " 3=%d", pad[0]);
            fprintf(pp, " 4=%d", global_pool);
        }
        else if (n.op == "SliceChannel")
        {
            int num_outputs = n.attr("num_outputs");
            int squeeze_axis = n.attr("squeeze_axis");// TODO

            fprintf(pp, " -23300=%d", num_outputs);
            for (int j=0; j<num_outputs; j++)
            {
                fprintf(pp, ",-233");
            }
        }
        else if (n.op == "SoftmaxOutput")
        {
        }
        else
        {
            // TODO op specific params
            std::map<std::string, std::string>::const_iterator it = n.attrs.begin();
            for (; it != n.attrs.end(); it++)
            {
                fprintf(stderr, "# %s=%s\n", it->first.c_str(), it->second.c_str());
//                 fprintf(pp, " %s=%s", it->first.c_str(), it->second.c_str());
            }
        }

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
