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

#ifndef NCNN_NET_H
#define NCNN_NET_H

#include <stdio.h>
#include <vector>
#include "blob.h"
#include "layer.h"
#include "mat.h"
#include "platform.h"

namespace ncnn {

class Extractor;
class Net
{
public:
    // empty init
    Net();
    // clear and destroy
    ~Net();

#if NCNN_STRING
    // register custom layer by layer type name
    // return 0 if success
    int register_custom_layer(const char* type, layer_creator_func creator);
#endif // NCNN_STRING
    // register custom layer by layer type
    // return 0 if success
    int register_custom_layer(int index, layer_creator_func creator);

#if NCNN_STDIO
#if NCNN_STRING
    // load network structure from plain param file
    // return 0 if success
    int load_param(FILE* fp);
    int load_param(const char* protopath);
#endif // NCNN_STRING
    // load network structure from binary param file
    // return 0 if success
    int load_param_bin(FILE* fp);
    int load_param_bin(const char* protopath);

    // load network weight data from model file
    // return 0 if success
    int load_model(FILE* fp);
    int load_model(const char* modelpath);
#endif // NCNN_STDIO

    // load network structure from external memory
    // memory pointer must be 32-bit aligned
    // return bytes consumed
    int load_param(const unsigned char* mem);

    // reference network weight data from external memory
    // weight data is not copied but referenced
    // so external memory should be retained when used
    // memory pointer must be 32-bit aligned
    // return bytes consumed
    int load_model(const unsigned char* mem);

    // unload network structure and weight data
    void clear();

    // construct an Extractor from network
    Extractor create_extractor() const;

protected:
    friend class Extractor;
#if NCNN_STRING
    int find_blob_index_by_name(const char* name) const;
    int find_layer_index_by_name(const char* name) const;
    int custom_layer_to_index(const char* type);
#endif // NCNN_STRING
    Layer* create_custom_layer(int index);
    int forward_layer(int layer_index, std::vector<Mat>& blob_mats, bool lightmode) const;

protected:
    std::vector<Blob> blobs;
    std::vector<Layer*> layers;

    std::vector<layer_registry_entry> custom_layer_registry;
};

class Extractor
{
public:
    // enable light mode
    // intermediate blob will be recycled when enabled
    // disabled by default, but recommend to enable
    void set_light_mode(bool enable);

    // set thread count for this extractor
    // this will overwrite the global setting
    // default count is system depended
    void set_num_threads(int num_threads);

#if NCNN_STRING
    // set input by blob name
    // return 0 if success
    int input(const char* blob_name, const Mat& in);

    // get result by blob name
    // return 0 if success
    int extract(const char* blob_name, Mat& feat);
#endif // NCNN_STRING

    // set input by blob index
    // return 0 if success
    int input(int blob_index, const Mat& in);

    // get result by blob index
    // return 0 if success
    int extract(int blob_index, Mat& feat);

protected:
    friend Extractor Net::create_extractor() const;
    Extractor(const Net* net, int blob_count);

private:
    const Net* net;
    std::vector<Mat> blob_mats;
    bool lightmode;
    int num_threads;
};

} // namespace ncnn

#endif // NCNN_NET_H
