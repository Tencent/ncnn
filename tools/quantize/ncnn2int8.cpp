// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
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

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <vector>

// ncnn public header
#include "datareader.h"

// ncnn private header
#include "net_quantize.h"

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* format, void* p) const
    {
        return 0;
    }
    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

int main(int argc, char** argv)
{
    if (argc != 6)
    {
        fprintf(stderr, "usage: %s [inparam] [inbin] [outparam] [outbin] [calibration table]\n", argv[0]);
        return -1;
    }

    const char* inparam = argv[1];
    const char* inbin = argv[2];
    const char* outparam = argv[3];
    const char* outbin = argv[4];
    const char* int8scale_table_path = argv[5];

    NetQuantize quantizer;

    // parse the calibration scale table
    bool success = false;
    if (std::string(int8scale_table_path).find(".ini") == std::string::npos)
    {
        quantizer.set_weight_suffix("_param_0");
        success = quantizer.read_txt_format(int8scale_table_path);
    }
    else
    {
        success = quantizer.read_ini_format(int8scale_table_path);
    }

    if (!success)
    {
        fprintf(stderr, "read_int8scale_table failed\n");
        return -1;
    }

    quantizer.load_param(inparam);
    if (strcmp(inbin, "null") == 0)
    {
        DataReaderFromEmpty dr;
        quantizer.load_model(dr);
        quantizer.gen_random_weight = true;
    }
    else
        quantizer.load_model(inbin);

    quantizer.quantize_mha();
    quantizer.quantize_convolution();
    quantizer.quantize_convolutiondepthwise();
    quantizer.quantize_innerproduct();
    quantizer.quantize_layernorm();
    quantizer.quantize_binaryop();

    quantizer.fuse_conv_requantize();
    quantizer.fuse_layernorm_requantize();
    quantizer.fuse_binaryop_requantize();

    quantizer.save(outparam, outbin);

    return 0;
}
