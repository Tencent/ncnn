// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "load_tnn.h"

#include "ir.h"

#include <stdio.h>
#include <string.h>

namespace pnnx {

static bool vstr_is_float(const char vstr[16])
{
    // look ahead for determine isfloat
    for (int j = 0; j < 16; j++)
    {
        if (vstr[j] == '\0')
            break;

        if (vstr[j] == '.' || tolower(vstr[j]) == 'e')
            return true;
    }

    return false;
}

static float vstr_to_float(const char vstr[16])
{
    double v = 0.0;

    const char* p = vstr;

    // sign
    bool sign = *p != '-';
    if (*p == '+' || *p == '-')
    {
        p++;
    }

    // digits before decimal point or exponent
    unsigned int v1 = 0;
    while (isdigit(*p))
    {
        v1 = v1 * 10 + (*p - '0');
        p++;
    }

    v = (double)v1;

    // digits after decimal point
    if (*p == '.')
    {
        p++;

        unsigned int pow10 = 1;
        unsigned int v2 = 0;

        while (isdigit(*p))
        {
            v2 = v2 * 10 + (*p - '0');
            pow10 *= 10;
            p++;
        }

        v += v2 / (double)pow10;
    }

    // exponent
    if (*p == 'e' || *p == 'E')
    {
        p++;

        // sign of exponent
        bool fact = *p != '-';
        if (*p == '+' || *p == '-')
        {
            p++;
        }

        // digits of exponent
        unsigned int expon = 0;
        while (isdigit(*p))
        {
            expon = expon * 10 + (*p - '0');
            p++;
        }

        double scale = 1.0;
        while (expon >= 8)
        {
            scale *= 1e8;
            expon -= 8;
        }
        while (expon > 0)
        {
            scale *= 10.0;
            expon -= 1;
        }

        v = fact ? v * scale : v / scale;
    }

    //     fprintf(stderr, "v = %f\n", v);
    return sign ? (float)v : (float)-v;
}

int load_tnn(const std::string& tnnpath, Graph& pnnx_graph)
{
    fprintf(stderr, "############# pass_level0 tnn\n");

    fprintf(stderr, "load_tnn %s\n", tnnpath.c_str());

    FILE* fp = fopen(tnnpath.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", tnnpath.c_str());
        return -1;
    }

    char line[4096];

    // "1 57 1 4206624772 ,"
    fgets(line, 4096, fp);
    int blob_count = 57;
    unsigned int magic = 4206624772;

    // "input 2 1 80000 0 ,"
    fgets(line, 4096, fp);
    if (magic == 4206624772)
    {
        // strip leading and tail double quote
        line[strlen(line) - 2] = '\0';
        line[0] = '\0';
        const char* pline = line + 1;

        // input operand name
        // rank 2
        // shape (1, 80000)
        // datatype 0=fp32

        int ncomsumed = 0;
        char blob_name[32];
        int rank = 0;
        sscanf(pline, "%s %d%n", blob_name, &rank, &ncomsumed);

        pline += ncomsumed;

        std::vector<int> shape(rank);
        for (int i = 0; i < rank; i++)
        {
            sscanf(pline, "%d%n", &shape[i], &ncomsumed);

            pline += ncomsumed;
        }

        int datatype = 0;
        sscanf(pline, "%d%n", &datatype, &ncomsumed);

        Operator* op = pnnx_graph.new_operator("pnnx.Input", "input0");

        Operand* r = pnnx_graph.new_operand(blob_name);

        r->producer = op;

        r->shape = shape;

        if (datatype == 0)
            r->type = 1;

        op->outputs.push_back(r);
    }

    // all operand names
    // " 108 109 110 111 112 113 114 116 118 119 120 125 126 128 130 131 132 133 135 136 138 139 142 144 145 147 148 151 153 154 156 157 160 162 163 165 166 169 171 172 174 175 178 180 181 183 184 188 189 190 191 192 194 85 clipwise_output embedding input ,"
    fgets(line, 4096, fp);
    {
        // strip leading and tail double quote
        line[strlen(line) - 2] = '\0';
        line[0] = '\0';
        const char* pline = line + 1;

        int ncomsumed = 0;

        for (int i = 0; i < blob_count; i++)
        {
            char blob_name[32];
            sscanf(pline, "%s%n", blob_name, &ncomsumed);

            pline += ncomsumed;

            // fprintf(stderr, "blob %s\n", blob_name);

            if (!pnnx_graph.get_operand(blob_name))
            {
                pnnx_graph.new_operand(blob_name);
            }
        }
    }

    // all output names
    // "clipwise_output embedding ,"
    fgets(line, 4096, fp);

    std::vector<std::string> output_names;
    {
        // strip leading and tail double quote
        line[strlen(line) - 2] = '\0';
        line[0] = '\0';
        const char* pline = line + 1;

        int ncomsumed = 0;

        while (1)
        {
            char blob_name[32];
            sscanf(pline, "%s%n", blob_name, &ncomsumed);

            pline += ncomsumed;

            if (strcmp(blob_name, ",") == 0)
                break;

            // fprintf(stderr, "blob %s\n", blob_name);

            output_names.push_back(blob_name);
        }
    }

    // layer count
    // " 56 ,"
    fgets(line, 4096, fp);
    int layer_count = 56;

    for (int i = 0; i < layer_count; i++)
    {
        // "Unsqueeze Unsqueeze_0 1 1 input 85 1 1 ,"
        fgets(line, 4096, fp);

        // strip leading and tail double quote
        line[strlen(line) - 2] = '\0';
        line[0] = '\0';
        const char* pline = line + 1;

        int ncomsumed = 0;

        char layer_type[32];
        char layer_name[32];
        int bottom_count;
        int top_count;
        sscanf(pline, "%s %s %d %d%n", layer_type, layer_name, &bottom_count, &top_count, &ncomsumed);

        pline += ncomsumed;

        // fprintf(stderr, "%s %s %d %d\n", layer_type, layer_name, bottom_count, top_count);

        Operator* op = pnnx_graph.new_operator(std::string("tnn.") + layer_type, layer_name);

        for (int j = 0; j < bottom_count; j++)
        {
            char blob_name[32];
            sscanf(pline, "%s%n", blob_name, &ncomsumed);

            pline += ncomsumed;

            // fprintf(stderr, "   bottom %s\n", blob_name);

            Operand* r = pnnx_graph.get_operand(blob_name);
            if (!r)
            {
                fprintf(stderr, "%s bottom %s not found\n", layer_name, blob_name);
            }
            r->consumers.push_back(op);
            op->inputs.push_back(r);
        }

        for (int j = 0; j < top_count; j++)
        {
            char blob_name[32];
            sscanf(pline, "%s%n", blob_name, &ncomsumed);

            pline += ncomsumed;

            // fprintf(stderr, "   top %s\n", blob_name);

            Operand* r = pnnx_graph.get_operand(blob_name);
            if (!r)
            {
                fprintf(stderr, "%s top %s not found\n", layer_name, blob_name);
            }
            r->producer = op;
            op->outputs.push_back(r);
        }

        // layer specific data
        // Unsqueeze            1 1 ,
        // Convolution1D        1 1 257 512 160 0 0 0 -1 1 0 ,

        int param_id = 0;
        while (1)
        {
            char vstr[16];
            sscanf(pline, "%s%n", vstr, &ncomsumed);

            pline += ncomsumed;

            if (strcmp(vstr, ",") == 0)
                break;

            // fprintf(stderr, "vstr %s\n", vstr);

            bool is_float = vstr_is_float(vstr);

            if (is_float)
            {
                float v = vstr_to_float(vstr);

                op->params[std::string("arg") + std::to_string(param_id)] = v;
            }
            else
            {
                int v = 0;
                int nscan = sscanf(vstr, "%d", &v);
                if (nscan == 1)
                {
                    op->params[std::string("arg") + std::to_string(param_id)] = v;
                }
                else
                {
                    // fallback to string type
                    op->params[std::string("arg") + std::to_string(param_id)] = vstr;
                }
            }

            param_id++;
        }
    }

    // append output nodes
    const int output_count = (int)output_names.size();
    for (int i = 0; i < output_count; i++)
    {
        Operator* op = pnnx_graph.new_operator("pnnx.Output", "output" + std::to_string(i));

        Operand* r = pnnx_graph.get_operand(output_names[i]);

        r->consumers.push_back(op);

        // fprintf(stderr, "r->name = %s\n", r->name.c_str());

        op->inputs.push_back(r);
    }


    fclose(fp);

    // replace simple operator
    for (Operator* op : pnnx_graph.ops)
    {
        // unary
        if (op->type == "tnn.Log") op->type = "aten::log";
        if (op->type == "tnn.ReLU") op->type = "aten::relu";
        if (op->type == "tnn.Sigmoid") op->type = "aten::sigmoid";

        // binary
        if (op->type == "tnn.Add") op->type = "aten::add";
    }

    return 0;
}

} // namespace pnnx
