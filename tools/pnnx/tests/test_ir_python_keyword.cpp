// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Graph::python must emit parseable python even when a module is named after a
// python keyword (from, class, ...) and must not merge it with a module that is
// really named like the escaped keyword (from_).

#include "ir.h"
#include "model_stat.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    const char* python = argc > 1 ? argv[1] : "python3";

    {
        FILE* fp = fopen("keyword.param", "wb");
        fprintf(fp, "7767517\n");
        fprintf(fp, "11 13\n");
        fprintf(fp, "pnnx.Input pnnx_input_0 0 1 0 #0=(2,3)f32\n");
        fprintf(fp, "nn.ReLU from 1 1 0 1 #0=(2,3)f32 #1=(2,3)f32\n");
        fprintf(fp, "nn.ReLU from_ 1 1 1 2 #1=(2,3)f32 #2=(2,3)f32\n");
        fprintf(fp, "nn.ReLU class 1 1 2 3 #2=(2,3)f32 #3=(2,3)f32\n");
        fprintf(fp, "nn.ReLU a.lambda 1 1 3 4 #3=(2,3)f32 #4=(2,3)f32\n");
        // modules that are called through their own forward emitters
        fprintf(fp, "nn.GRU if 1 2 4 5 6 input_size=3 hidden_size=3 num_layers=1 bias=False batch_first=True bidirectional=False #4=(2,3)f32 #5=(2,3)f32 #6=(1,3)f32\n");
        fprintf(fp, "nn.LSTM while 1 3 5 7 8 9 input_size=3 hidden_size=3 num_layers=1 bias=False batch_first=True bidirectional=False proj_size=0 #5=(2,3)f32 #7=(2,3)f32 #8=(1,3)f32 #9=(1,3)f32\n");
        fprintf(fp, "nn.MultiheadAttention with 1 1 7 10 embed_dim=3 num_heads=1 batch_first=True bias=False add_bias_kv=False add_zero_attn=False kdim=3 vdim=3 #7=(2,3)f32 #10=(2,3)f32\n");
        // module with a weight and a running statistics attribute
        fprintf(fp, "nn.BatchNorm1d def 1 1 10 11 num_features=3 eps=1e-05 affine=True @weight=(3)f32 @running_mean=(3)f32 #10=(2,3)f32 #11=(2,3)f32\n");
        // quantized module
        fprintf(fp, "nn.quantized.Linear del 1 1 11 12 in_features=3 out_features=3 scale=1.000000e+00 zero_point=0 @weight=(3,3)f32 @bias=(3)f32 #11=(2,3)f32 #12=(2,3)f32\n");
        fprintf(fp, "pnnx.Output pnnx_output_0 1 0 12\n");
        fclose(fp);

        // empty zip archive
        fp = fopen("keyword.bin", "wb");
        const unsigned char eocd[22] = {0x50, 0x4b, 0x05, 0x06};
        fwrite(eocd, 1, 22, fp);
        fclose(fp);
    }

    pnnx::Graph g;
    if (g.load("keyword.param", "keyword.bin") != 0)
    {
        fprintf(stderr, "load failed\n");
        return 1;
    }

    std::vector<std::vector<int64_t> > input_shapes;
    input_shapes.push_back(std::vector<int64_t>(2));
    input_shapes[0][0] = 2;
    input_shapes[0][1] = 3;
    pnnx::ModelStat model_stat;
    if (g.python("keyword_pnnx.py", "keyword.pnnx.bin", input_shapes, model_stat) != 0)
    {
        fprintf(stderr, "python failed\n");
        return 1;
    }

    std::string cmd = std::string(python) + " -m py_compile keyword_pnnx.py";
    if (system(cmd.c_str()) != 0)
    {
        fprintf(stderr, "generated python does not parse\n");
        return 1;
    }

    std::ifstream ifs("keyword_pnnx.py");
    std::stringstream ss;
    ss << ifs.rdbuf();
    const std::string src = ss.str();

    // the real from_ module and the escaped from module must stay distinct
    std::string needle = "        self.from_ = nn.ReLU()";
    size_t first = src.find(needle);
    if (first == std::string::npos || src.find(needle, first + 1) != std::string::npos)
    {
        fprintf(stderr, "expected exactly one self.from_ assignment\n");
        return 1;
    }

    // every keyword module must be addressed by its escaped attribute name
    const char* escaped[] = {"class_", "if_", "while_", "with_", "def_", "del_"};
    for (size_t i = 0; i < sizeof(escaped) / sizeof(escaped[0]); i++)
    {
        std::string attr = std::string("self.") + escaped[i];
        if (src.find("        " + attr + " = nn.") == std::string::npos)
        {
            fprintf(stderr, "missing %s definition\n", attr.c_str());
            return 1;
        }
        if (src.find(attr + "(") == std::string::npos && src.find(attr + ".set_weight_bias") == std::string::npos)
        {
            fprintf(stderr, "missing %s use\n", attr.c_str());
            return 1;
        }
    }

    return 0;
}
