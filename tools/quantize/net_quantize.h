// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#pragma once
// ncnn private header
#include <memory>
#include "../modelwriter.h"
#include "ini_config.h"

class NetQuantize : public ModelWriter
{
public:
    NetQuantize()
    {
    }
    // conv and gemm quant param
    std::map<std::string, ncnn::Mat> blob_int8scale_table;
    std::map<std::string, ncnn::Mat> weight_int8scale_table;

    // MutiHeadAttention quant param
    std::map<std::string, std::shared_ptr<ini::Table> > mha_table;
    // LayerNorm quant param
    std::map<std::string, std::shared_ptr<ini::Table> > layernorm_table;

public:
    bool read_txt_format(const char* path);
    bool read_ini_format(const char* path);

    int quantize_convolution();
    int quantize_convolutiondepthwise();
    int quantize_innerproduct();
    int quantize_mha();
    int quantize_layernorm();
    int fuse_requantize();

    void set_weight_suffix(std::string s);

private:
    std::string suffix;
};
