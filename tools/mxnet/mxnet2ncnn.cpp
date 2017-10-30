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

#include <string>
#include <vector>

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

static void read_mxnet_json(const char* jsonpath)
{
    FILE* fp = fopen(jsonpath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", jsonpath);
        return;
    }

    char line[1024];

    //{
    fgets(line, 1024, fp);

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
            if (memcmp(line, "      },", 8) == 0)
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
                fprintf(stderr, "# %s = %s\n", key, value);
                continue;
            }
        }

        if (in_node_block)
        {
            //    },
            if (memcmp(line, "    },", 6) == 0)
            {
                // new node TODO

                in_node_block = false;
                continue;
            }

            int nscan;

            //      "op": "Convolution",
            char op[256] = {0};
            nscan = sscanf(line, "      \"op\": \"%255[^\"]\",", op);
            if (nscan == 1)
            {
                fprintf(stderr, "op = %s\n", op);
                continue;
            }

            //      "name": "conv0",
            char name[256] = {0};
            nscan = sscanf(line, "      \"name\": \"%255[^\"]\",", name);
            if (nscan == 1)
            {
                fprintf(stderr, "name = %s\n", name);
                continue;
            }

            //      "inputs": []
            char inputs[256] = {0};
            nscan = sscanf(line, "      \"inputs\": %255[^\n]", inputs);
            if (nscan == 1)
            {
                fprintf(stderr, "inputs = %s\n", inputs);
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
                fprintf(stderr, "# %s = %s\n", key, value);
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
                // all nodes parsed TODO
                break;
            }

            //    {
            if (memcmp(line, "    {", 5) == 0)
            {
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
}

static void read_mxnet_param(const char* parampath)
{
    FILE* fp = fopen(parampath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", parampath);
        return;
    }

    uint64_t header;
    uint64_t reserved;
    fread(&header, 1, sizeof(uint64_t), fp);
    fread(&reserved, 1, sizeof(uint64_t), fp);

    // NDArray vec

    // each data
    uint64_t data_count;
    fread(&data_count, 1, sizeof(uint64_t), fp);

    fprintf(stderr, "data count = %d\n", (int)data_count);

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

        std::vector<float> data;
        data.resize(len);
        fread(&data[0], 1, len * sizeof(float), fp);

        fprintf(stderr, "%u read\n", len);
    }

    // each name
    uint64_t name_count;
    fread(&name_count, 1, sizeof(uint64_t), fp);

    fprintf(stderr, "name count = %d\n", (int)name_count);

    for (int i = 0; i < (int)name_count; i++)
    {
        uint64_t len;
        fread(&len, 1, sizeof(uint64_t), fp);

        std::string name;
        name.resize(len);
        fread((char*)name.data(), 1, len, fp);

        fprintf(stderr, "%s read\n", name.c_str());
    }

    fclose(fp);
}

int main(int argc, char** argv)
{
    const char* jsonpath = argv[1];
    const char* parampath = argv[2];

    read_mxnet_json(jsonpath);

    read_mxnet_param(parampath);

    return 0;
}
