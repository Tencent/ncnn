// tpoisonooo is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2022 tpoisonooo. All rights reserved.
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
#include <memory>
#include <string>

// ncnn public header
#include "datareader.h"
#include "../helper/cmdline.h"

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
    fprintf(stdout, "ncnn2int8_toml will parse toml format ")
    cmd::line parser;
    parser.add<std::string>("inparam", "ip", "input param file path", true);
    parser.add<std::string>("inbin", "ib", "input weight file path, `null` means using random weight", true);
    parser.add<std::string>("outparam", "op", "output param file path", true);
    parser.add<std::string>("outbin", "ob", "output weight file path", true);
    parser.add<std::string>("table", "t", "the .toml file path of quantization table which contains opr scale", true);
    parser.parse_check(argc, argv);

    std::string inparam = parser.get<std::string>("inparam");
    std::string inbin = parser.get<std::string>("inbin");
    std::string outparam = parser.get<std::string>("outparam");
    std::string outbin = parser.get<std::string>("oubin");
    std::string table = parser.get<std::string>("table");

    NetQuantize quantizer;
    
    quantizer.read(table);
    quantizer.load_param(inparam.c_str());

    if (inbin == "null")
    {
        DataReaderFromEmpty dr;
        quantizer.load_model(dr);
        quantizer.gen_random_weight = true;
    }
    else
    {
        quantizer.load_model(inbin.c_str());
    }

    quantizer.quantize_convolution();
    quantizer.quantize_convolutiondepthwise();
    quantizer.quantize_innerproduct();

    quantizer.fuse_requantize();

    quantizer.save(outparam.c_str(), outbin.c_str());

    return 0;
}
