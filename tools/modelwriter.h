// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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
#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <stdint.h>
#include <algorithm>
#include <map>
#include <set>
#include <vector>

// ncnn public header
#include "datareader.h"
#include "layer.h"
#include "layer_type.h"
#include "net.h"

// ncnn private header
#include "layer/batchnorm.h"
#include "layer/bias.h"
#include "layer/binaryop.h"
#include "layer/clip.h"
#include "layer/concat.h"
#include "layer/convolution.h"
#include "layer/convolution1d.h"
#include "layer/convolution3d.h"
#include "layer/convolutiondepthwise.h"
#include "layer/convolutiondepthwise1d.h"
#include "layer/convolutiondepthwise3d.h"
#include "layer/crop.h"
#include "layer/deconvolution.h"
#include "layer/deconvolution1d.h"
#include "layer/deconvolution3d.h"
#include "layer/deconvolutiondepthwise.h"
#include "layer/deconvolutiondepthwise1d.h"
#include "layer/deconvolutiondepthwise3d.h"
#include "layer/detectionoutput.h"
#include "layer/dropout.h"
#include "layer/eltwise.h"
#include "layer/elu.h"
#include "layer/embed.h"
#include "layer/exp.h"
#include "layer/expanddims.h"
#include "layer/flatten.h"
#include "layer/gelu.h"
#include "layer/gemm.h"
#include "layer/groupnorm.h"
#include "layer/gru.h"
#include "layer/hardsigmoid.h"
#include "layer/hardswish.h"
#include "layer/innerproduct.h"
#include "layer/input.h"
#include "layer/instancenorm.h"
#include "layer/interp.h"
#include "layer/layernorm.h"
#include "layer/log.h"
#include "layer/lrn.h"
#include "layer/lstm.h"
#include "layer/matmul.h"
#include "layer/memorydata.h"
#include "layer/mvn.h"
#include "layer/multiheadattention.h"
#include "layer/normalize.h"
#include "layer/padding.h"
#include "layer/permute.h"
#include "layer/pixelshuffle.h"
#include "layer/pooling.h"
#include "layer/pooling1d.h"
#include "layer/pooling3d.h"
#include "layer/power.h"
#include "layer/prelu.h"
#include "layer/priorbox.h"
#include "layer/proposal.h"
#include "layer/psroipooling.h"
#include "layer/quantize.h"
#include "layer/reduction.h"
#include "layer/relu.h"
#include "layer/reorg.h"
#include "layer/requantize.h"
#include "layer/reshape.h"
#include "layer/rnn.h"
#include "layer/roialign.h"
#include "layer/roipooling.h"
#include "layer/scale.h"
#include "layer/shufflechannel.h"
#include "layer/slice.h"
#include "layer/softmax.h"
#include "layer/split.h"
#include "layer/squeeze.h"
#include "layer/threshold.h"
#include "layer/unaryop.h"
#include "layer/yolodetectionoutput.h"
#include "layer/yolov3detectionoutput.h"

// for gen_random_weight
#include "../tests/prng.h"

static struct prng_rand_t g_prng_rand_state;
#define SRAND(seed) prng_srand(seed, &g_prng_rand_state)
#define RAND()      prng_rand(&g_prng_rand_state)

class MemoryFootprintAllocator : public ncnn::Allocator
{
public:
    MemoryFootprintAllocator();

    virtual void* fastMalloc(size_t size);

    virtual void fastFree(void* ptr);

public:
    int current_memory_usage;
    int memory_footprint;
    ncnn::Mutex lock;
    std::map<void*, size_t> bookkeeper;
};

class CustomLayer : public ncnn::Layer
{
public:
    virtual int load_param(const ncnn::ParamDict& pd);

    void write_param(FILE* pp);

public:
    ncnn::ParamDict mpd;
};

class ModelWriter : public ncnn::Net
{
public:
    ModelWriter();

    virtual ncnn::Layer* create_custom_layer(const char* type);

    std::vector<ncnn::Blob>& blobs;
    std::vector<ncnn::Layer*>& layers;

    bool has_custom_layer;

public:
    // 0=fp32 1=fp16
    int storage_type;

    int gen_random_weight;

    // Cut param and bin -1=no cut
    int cutstart;
    int cutend;

public:
    int set_cutparam(const char* cutstartname, const char* cutendname);

    int shape_inference();
    int estimate_memory_footprint();

public:
    int fprintf_param_int_array(int id, const ncnn::Mat& m, FILE* pp);
    int fprintf_param_float_array(int id, const ncnn::Mat& m, FILE* pp);

    int fwrite_weight_tag_data(const ncnn::Mat& data, FILE* bp, float a = -1.2f, float b = 1.2f);
    int fwrite_weight_data(const ncnn::Mat& data, FILE* bp, float a = -1.2f, float b = 1.2f);

    int save(const char* parampath, const char* binpath);
};
