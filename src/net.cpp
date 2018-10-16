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

#include "net.h"
#include "layer_type.h"
#include "modelbin.h"
#include "paramdict.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

#if NCNN_BENCHMARK
#include "benchmark.h"
#endif // NCNN_BENCHMARK

namespace ncnn {

Net::Net()
{
    use_winograd_convolution = 1;
    use_sgemm_convolution = 1;
    use_int8_inference = 1;
}

Net::~Net()
{
    clear();
}

#if NCNN_STRING
int Net::register_custom_layer(const char* type, layer_creator_func creator)
{
    int typeindex = layer_to_index(type);
    if (typeindex != -1)
    {
        fprintf(stderr, "can not register build-in layer type %s\n", type);
        return -1;
    }

    int custom_index = custom_layer_to_index(type);
    if (custom_index == -1)
    {
        struct layer_registry_entry entry = { type, creator };
        custom_layer_registry.push_back(entry);
    }
    else
    {
        fprintf(stderr, "overwrite existing custom layer type %s\n", type);
        custom_layer_registry[custom_index].name = type;
        custom_layer_registry[custom_index].creator = creator;
    }

    return 0;
}
#endif // NCNN_STRING

int Net::register_custom_layer(int index, layer_creator_func creator)
{
    int custom_index = index & ~LayerType::CustomBit;
    if (index == custom_index)
    {
        fprintf(stderr, "can not register build-in layer index %d\n", custom_index);
        return -1;
    }

    if ((int)custom_layer_registry.size() <= custom_index)
    {
#if NCNN_STRING
        struct layer_registry_entry dummy = { "", 0 };
#else
        struct layer_registry_entry dummy = { 0 };
#endif // NCNN_STRING
        custom_layer_registry.resize(custom_index + 1, dummy);
    }

    if (custom_layer_registry[custom_index].creator)
    {
        fprintf(stderr, "overwrite existing custom layer index %d\n", custom_index);
    }

    custom_layer_registry[custom_index].creator = creator;
    return 0;
}

#if NCNN_STDIO
#if NCNN_STRING
int Net::load_param(FILE* fp)
{
    int magic = 0;
    fscanf(fp, "%d", &magic);
    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return -1;
    }

    // parse
    int layer_count = 0;
    int blob_count = 0;
    fscanf(fp, "%d %d", &layer_count, &blob_count);

    layers.resize(layer_count);
    blobs.resize(blob_count);

    ParamDict pd;
    pd.use_winograd_convolution = use_winograd_convolution;
    pd.use_sgemm_convolution = use_sgemm_convolution;
    pd.use_int8_inference = use_int8_inference;

    int blob_index = 0;
    for (int i=0; i<layer_count; i++)
    {
        int nscan = 0;

        char layer_type[257];
        char layer_name[257];
        int bottom_count = 0;
        int top_count = 0;
        nscan = fscanf(fp, "%256s %256s %d %d", layer_type, layer_name, &bottom_count, &top_count);
        if (nscan != 4)
        {
            continue;
        }

        Layer* layer = create_layer(layer_type);
        if (!layer)
        {
            layer = create_custom_layer(layer_type);
        }
        if (!layer)
        {
            fprintf(stderr, "layer %s not exists or registered\n", layer_type);
            clear();
            return -1;
        }

        layer->type = std::string(layer_type);
        layer->name = std::string(layer_name);
//         fprintf(stderr, "new layer %d %s\n", i, layer_name);

        layer->bottoms.resize(bottom_count);

        for (int j=0; j<bottom_count; j++)
        {
            char bottom_name[257];
            nscan = fscanf(fp, "%256s", bottom_name);
            if (nscan != 1)
            {
                continue;
            }

            int bottom_blob_index = find_blob_index_by_name(bottom_name);
            if (bottom_blob_index == -1)
            {
                Blob& blob = blobs[blob_index];

                bottom_blob_index = blob_index;

                blob.name = std::string(bottom_name);
//                 fprintf(stderr, "new blob %s\n", bottom_name);

                blob_index++;
            }

            Blob& blob = blobs[bottom_blob_index];

            blob.consumers.push_back(i);

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int j=0; j<top_count; j++)
        {
            Blob& blob = blobs[blob_index];

            char blob_name[257];
            nscan = fscanf(fp, "%256s", blob_name);
            if (nscan != 1)
            {
                continue;
            }

            blob.name = std::string(blob_name);
//             fprintf(stderr, "new blob %s\n", blob_name);

            blob.producer = i;

            layer->tops[j] = blob_index;

            blob_index++;
        }

        // layer specific params
        int pdlr = pd.load_param(fp);
        if (pdlr != 0)
        {
            fprintf(stderr, "ParamDict load_param failed\n");
            continue;
        }

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            fprintf(stderr, "layer load_param failed\n");
            continue;
        }

        layers[i] = layer;
    }

    return 0;
}

#if _MSC_VER
static inline int mem_sscanf_with_n(int* _internal_nconsumed_ptr, const char*& ptr, const char* format, ...)
{
    va_list args;
    va_start(args, format);

    int _n = vsscanf(ptr, format, args);

    va_end(args);

    ptr += *_internal_nconsumed_ptr;

    return *_internal_nconsumed_ptr > 0 ? _n : 0;
}
#define mem_sscanf(ptr, format, ...)  mem_sscanf_with_n(&_internal_nconsumed, ptr, format "%n", __VA_ARGS__, &_internal_nconsumed)
#else
// return value from macro requires gcc extension https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html
#define mem_sscanf(ptr, format, ...)  ({int _b=0; int _n = sscanf(ptr, format "%n", __VA_ARGS__, &_b); ptr+=_b;_b>0?_n:0;})
#endif // _MSC_VER

int Net::load_param_mem(const char* _mem)
{
#if _MSC_VER
    int _internal_nconsumed;
#endif

    int magic = 0;
    const char* mem = _mem;
    mem_sscanf(mem, "%d", &magic);
    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return -1;
    }

    // parse
    int layer_count = 0;
    int blob_count = 0;
    mem_sscanf(mem, "%d %d", &layer_count, &blob_count);

    layers.resize(layer_count);
    blobs.resize(blob_count);

    ParamDict pd;
    pd.use_winograd_convolution = use_winograd_convolution;
    pd.use_sgemm_convolution = use_sgemm_convolution;
    pd.use_int8_inference = use_int8_inference;

    int blob_index = 0;
    for (int i=0; i<layer_count; i++)
    {
        int nscan = 0;

        char layer_type[257];
        char layer_name[257];
        int bottom_count = 0;
        int top_count = 0;
        nscan = mem_sscanf(mem, "%256s %256s %d %d", layer_type, layer_name, &bottom_count, &top_count);
        if (nscan != 4)
        {
            continue;
        }

        Layer* layer = create_layer(layer_type);
        if (!layer)
        {
            layer = create_custom_layer(layer_type);
        }
        if (!layer)
        {
            fprintf(stderr, "layer %s not exists or registered\n", layer_type);
            clear();
            return -1;
        }

        layer->type = std::string(layer_type);
        layer->name = std::string(layer_name);
//         fprintf(stderr, "new layer %d %s\n", i, layer_name);

        layer->bottoms.resize(bottom_count);

        for (int j=0; j<bottom_count; j++)
        {
            char bottom_name[257];
            nscan = mem_sscanf(mem, "%256s", bottom_name);
            if (nscan != 1)
            {
                continue;
            }

            int bottom_blob_index = find_blob_index_by_name(bottom_name);
            if (bottom_blob_index == -1)
            {
                Blob& blob = blobs[blob_index];

                bottom_blob_index = blob_index;

                blob.name = std::string(bottom_name);
//                 fprintf(stderr, "new blob %s\n", bottom_name);

                blob_index++;
            }

            Blob& blob = blobs[bottom_blob_index];

            blob.consumers.push_back(i);

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int j=0; j<top_count; j++)
        {
            Blob& blob = blobs[blob_index];

            char blob_name[257];
            nscan = mem_sscanf(mem, "%256s", blob_name);
            if (nscan != 1)
            {
                continue;
            }

            blob.name = std::string(blob_name);
//             fprintf(stderr, "new blob %s\n", blob_name);

            blob.producer = i;

            layer->tops[j] = blob_index;

            blob_index++;
        }

        // layer specific params
        int pdlr = pd.load_param_mem(mem);
        if (pdlr != 0)
        {
            fprintf(stderr, "ParamDict load_param failed\n");
            continue;
        }

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            fprintf(stderr, "layer load_param failed\n");
            continue;
        }

        layers[i] = layer;
    }

    return 0;
}
int Net::load_param(const char* protopath)
{
    FILE* fp = fopen(protopath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", protopath);
        return -1;
    }

    int ret = load_param(fp);

    fclose(fp);

    return ret;
}
#endif // NCNN_STRING

int Net::load_param_bin(FILE* fp)
{
    int magic = 0;
    fread(&magic, sizeof(int), 1, fp);
    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return -1;
    }

    int layer_count = 0;
    fread(&layer_count, sizeof(int), 1, fp);

    int blob_count = 0;
    fread(&blob_count, sizeof(int), 1, fp);

    layers.resize(layer_count);
    blobs.resize(blob_count);

    ParamDict pd;
    pd.use_winograd_convolution = use_winograd_convolution;
    pd.use_sgemm_convolution = use_sgemm_convolution;
    pd.use_int8_inference = use_int8_inference;

    for (int i=0; i<layer_count; i++)
    {
        int typeindex;
        fread(&typeindex, sizeof(int), 1, fp);

        int bottom_count;
        fread(&bottom_count, sizeof(int), 1, fp);

        int top_count;
        fread(&top_count, sizeof(int), 1, fp);

        Layer* layer = create_layer(typeindex);
        if (!layer)
        {
            int custom_index = typeindex & ~LayerType::CustomBit;
            layer = create_custom_layer(custom_index);
        }
        if (!layer)
        {
            fprintf(stderr, "layer %d not exists or registered\n", typeindex);
            clear();
            return -1;
        }

//         layer->type = std::string(layer_type);
//         layer->name = std::string(layer_name);
//         fprintf(stderr, "new layer %d\n", typeindex);

        layer->bottoms.resize(bottom_count);
        for (int j=0; j<bottom_count; j++)
        {
            int bottom_blob_index;
            fread(&bottom_blob_index, sizeof(int), 1, fp);

            Blob& blob = blobs[bottom_blob_index];

            blob.consumers.push_back(i);

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int j=0; j<top_count; j++)
        {
            int top_blob_index;
            fread(&top_blob_index, sizeof(int), 1, fp);

            Blob& blob = blobs[top_blob_index];

//             blob.name = std::string(blob_name);
//             fprintf(stderr, "new blob %s\n", blob_name);

            blob.producer = i;

            layer->tops[j] = top_blob_index;
        }

        // layer specific params
        int pdlr = pd.load_param_bin(fp);
        if (pdlr != 0)
        {
            fprintf(stderr, "ParamDict load_param failed\n");
            continue;
        }

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            fprintf(stderr, "layer load_param failed\n");
            continue;
        }

        layers[i] = layer;
    }

    return 0;
}

int Net::load_param_bin(const char* protopath)
{
    FILE* fp = fopen(protopath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", protopath);
        return -1;
    }

    int ret = load_param_bin(fp);

    fclose(fp);

    return ret;
}

int Net::load_model(FILE* fp)
{
    if (layers.empty())
    {
        fprintf(stderr, "network graph not ready\n");
        return -1;
    }

    // load file
    int ret = 0;

    ModelBinFromStdio mb(fp);
    for (size_t i=0; i<layers.size(); i++)
    {
        Layer* layer = layers[i];

        int lret = layer->load_model(mb);
        if (lret != 0)
        {
            fprintf(stderr, "layer load_model %d failed\n", (int)i);
            ret = -1;
            break;
        }
    }

    return ret;
}

int Net::load_model(const char* modelpath)
{
    FILE* fp = fopen(modelpath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", modelpath);
        return -1;
    }

    int ret = load_model(fp);

    fclose(fp);

    return ret;
}
#endif // NCNN_STDIO

int Net::load_param(const unsigned char* _mem)
{
    if ((unsigned long)_mem & 0x3)
    {
        // reject unaligned memory
        fprintf(stderr, "memory not 32-bit aligned at %p\n", _mem);
        return 0;
    }

    const unsigned char* mem = _mem;

    int magic = *(int*)(mem);
    mem += 4;

    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return 0;
    }

    int layer_count = *(int*)(mem);
    mem += 4;

    int blob_count = *(int*)(mem);
    mem += 4;

    layers.resize(layer_count);
    blobs.resize(blob_count);

    ParamDict pd;
    pd.use_winograd_convolution = use_winograd_convolution;
    pd.use_sgemm_convolution = use_sgemm_convolution;
    pd.use_int8_inference = use_int8_inference;

    for (int i=0; i<layer_count; i++)
    {
        int typeindex = *(int*)mem;
        mem += 4;

        int bottom_count = *(int*)mem;
        mem += 4;

        int top_count = *(int*)mem;
        mem += 4;

        Layer* layer = create_layer(typeindex);
        if (!layer)
        {
            int custom_index = typeindex & ~LayerType::CustomBit;
            layer = create_custom_layer(custom_index);
        }
        if (!layer)
        {
            fprintf(stderr, "layer %d not exists or registered\n", typeindex);
            clear();
            return 0;
        }

//         layer->type = std::string(layer_type);
//         layer->name = std::string(layer_name);
//         fprintf(stderr, "new layer %d\n", typeindex);

        layer->bottoms.resize(bottom_count);
        for (int j=0; j<bottom_count; j++)
        {
            int bottom_blob_index = *(int*)mem;
            mem += 4;

            Blob& blob = blobs[bottom_blob_index];

            blob.consumers.push_back(i);

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int j=0; j<top_count; j++)
        {
            int top_blob_index = *(int*)mem;
            mem += 4;

            Blob& blob = blobs[top_blob_index];

//             blob.name = std::string(blob_name);
//             fprintf(stderr, "new blob %s\n", blob_name);

            blob.producer = i;

            layer->tops[j] = top_blob_index;
        }

        // layer specific params
        int pdlr = pd.load_param(mem);
        if (pdlr != 0)
        {
            fprintf(stderr, "ParamDict load_param failed\n");
            continue;
        }

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            fprintf(stderr, "layer load_param failed\n");
            continue;
        }

        layers[i] = layer;
    }

    return mem - _mem;
}

int Net::load_model(const unsigned char* _mem)
{
    if (layers.empty())
    {
        fprintf(stderr, "network graph not ready\n");
        return 0;
    }

    if ((unsigned long)_mem & 0x3)
    {
        // reject unaligned memory
        fprintf(stderr, "memory not 32-bit aligned at %p\n", _mem);
        return 0;
    }

    const unsigned char* mem = _mem;
    ModelBinFromMemory mb(mem);
    for (size_t i=0; i<layers.size(); i++)
    {
        Layer* layer = layers[i];

        int lret = layer->load_model(mb);
        if (lret != 0)
        {
            fprintf(stderr, "layer load_model failed\n");
            return -1;
        }
    }

    return mem - _mem;
}

void Net::clear()
{
    blobs.clear();
    for (size_t i=0; i<layers.size(); i++)
    {
        delete layers[i];
    }
    layers.clear();
}

Extractor Net::create_extractor() const
{
    return Extractor(this, blobs.size());
}

#if NCNN_STRING
int Net::find_blob_index_by_name(const char* name) const
{
    for (size_t i=0; i<blobs.size(); i++)
    {
        const Blob& blob = blobs[i];
        if (blob.name == name)
        {
            return i;
        }
    }

    fprintf(stderr, "find_blob_index_by_name %s failed\n", name);
    return -1;
}

int Net::find_layer_index_by_name(const char* name) const
{
    for (size_t i=0; i<layers.size(); i++)
    {
        const Layer* layer = layers[i];
        if (layer->name == name)
        {
            return i;
        }
    }

    fprintf(stderr, "find_layer_index_by_name %s failed\n", name);
    return -1;
}

int Net::custom_layer_to_index(const char* type)
{
    const int custom_layer_registry_entry_count = custom_layer_registry.size();
    for (int i=0; i<custom_layer_registry_entry_count; i++)
    {
        if (strcmp(type, custom_layer_registry[i].name) == 0)
            return i;
    }

    return -1;
}

Layer* Net::create_custom_layer(const char* type)
{
    int index = custom_layer_to_index(type);
    if (index == -1)
        return 0;

    return create_custom_layer(index);
}
#endif // NCNN_STRING

Layer* Net::create_custom_layer(int index)
{
    const int custom_layer_registry_entry_count = custom_layer_registry.size();
    if (index < 0 || index >= custom_layer_registry_entry_count)
        return 0;

    layer_creator_func layer_creator = custom_layer_registry[index].creator;
    if (!layer_creator)
        return 0;

    return layer_creator();
}

int Net::forward_layer(int layer_index, std::vector<Mat>& blob_mats, Option& opt) const
{
    const Layer* layer = layers[layer_index];

//     fprintf(stderr, "forward_layer %d %s\n", layer_index, layer->name.c_str());

    if (layer->one_blob_only)
    {
        // load bottom blob
        int bottom_blob_index = layer->bottoms[0];
        int top_blob_index = layer->tops[0];

        if (blob_mats[bottom_blob_index].dims == 0)
        {
            int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, opt);
            if (ret != 0)
                return ret;
        }

        Mat bottom_blob = blob_mats[bottom_blob_index];

        if (opt.lightmode)
        {
            // delete after taken in light mode
            blob_mats[bottom_blob_index].release();
            // deep copy for inplace forward if data is shared
            if (layer->support_inplace && *bottom_blob.refcount != 1)
            {
                bottom_blob = bottom_blob.clone();
            }
        }

        // forward
        if (opt.lightmode && layer->support_inplace)
        {
            Mat& bottom_top_blob = bottom_blob;
#if NCNN_BENCHMARK
            double start = get_current_time();
            int ret = layer->forward_inplace(bottom_top_blob, opt);
            double end = get_current_time();
            benchmark(layer, bottom_top_blob, bottom_top_blob, start, end);
#else
            int ret = layer->forward_inplace(bottom_top_blob, opt);
#endif // NCNN_BENCHMARK
            if (ret != 0)
                return ret;

            // store top blob
            blob_mats[top_blob_index] = bottom_top_blob;
        }
        else
        {
            Mat top_blob;
#if NCNN_BENCHMARK
            double start = get_current_time();
            int ret = layer->forward(bottom_blob, top_blob, opt);
            double end = get_current_time();
            benchmark(layer, bottom_blob, top_blob, start, end);
#else
            int ret = layer->forward(bottom_blob, top_blob, opt);
#endif // NCNN_BENCHMARK
            if (ret != 0)
                return ret;

            // store top blob
            blob_mats[top_blob_index] = top_blob;
        }

    }
    else
    {
        // load bottom blobs
        std::vector<Mat> bottom_blobs;
        bottom_blobs.resize(layer->bottoms.size());
        for (size_t i=0; i<layer->bottoms.size(); i++)
        {
            int bottom_blob_index = layer->bottoms[i];

            if (blob_mats[bottom_blob_index].dims == 0)
            {
                int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, opt);
                if (ret != 0)
                    return ret;
            }

            bottom_blobs[i] = blob_mats[bottom_blob_index];

            if (opt.lightmode)
            {
                // delete after taken in light mode
                blob_mats[bottom_blob_index].release();
                // deep copy for inplace forward if data is shared
                if (layer->support_inplace && *bottom_blobs[i].refcount != 1)
                {
                    bottom_blobs[i] = bottom_blobs[i].clone();
                }
            }
        }

        // forward
        if (opt.lightmode && layer->support_inplace)
        {
            std::vector<Mat>& bottom_top_blobs = bottom_blobs;
#if NCNN_BENCHMARK
            double start = get_current_time();
            int ret = layer->forward_inplace(bottom_top_blobs, opt);
            double end = get_current_time();
            benchmark(layer, start, end);
#else
            int ret = layer->forward_inplace(bottom_top_blobs, opt);
#endif // NCNN_BENCHMARK
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i=0; i<layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats[top_blob_index] = bottom_top_blobs[i];
            }
        }
        else
        {
            std::vector<Mat> top_blobs;
            top_blobs.resize(layer->tops.size());
#if NCNN_BENCHMARK
            double start = get_current_time();
            int ret = layer->forward(bottom_blobs, top_blobs, opt);
            double end = get_current_time();
            benchmark(layer, start, end);
#else
            int ret = layer->forward(bottom_blobs, top_blobs, opt);
#endif // NCNN_BENCHMARK
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i=0; i<layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats[top_blob_index] = top_blobs[i];
            }
        }
    }

//     fprintf(stderr, "forward_layer %d %s done\n", layer_index, layer->name.c_str());
//     const Mat& blob = blob_mats[layer->tops[0]];
//     fprintf(stderr, "[%-2d %-16s %-16s]  %d    blobs count = %-3d   size = %-3d x %-3d\n", layer_index, layer->type.c_str(), layer->name.c_str(), layer->tops[0], blob.c, blob.h, blob.w);

    return 0;
}

Extractor::Extractor(const Net* _net, int blob_count) : net(_net)
{
    blob_mats.resize(blob_count);
    opt = get_default_option();
}

void Extractor::set_light_mode(bool enable)
{
    opt.lightmode = enable;
}

void Extractor::set_num_threads(int num_threads)
{
    opt.num_threads = num_threads;
}

void Extractor::set_blob_allocator(Allocator* allocator)
{
    opt.blob_allocator = allocator;
}

void Extractor::set_workspace_allocator(Allocator* allocator)
{
    opt.workspace_allocator = allocator;
}

int Extractor::input(int blob_index, const Mat& in)
{
    if (blob_index < 0 || blob_index >= (int)blob_mats.size())
        return -1;

    blob_mats[blob_index] = in;

    return 0;
}

int Extractor::extract(int blob_index, Mat& feat)
{
    if (blob_index < 0 || blob_index >= (int)blob_mats.size())
        return -1;

    int ret = 0;

    if (blob_mats[blob_index].dims == 0)
    {
        int layer_index = net->blobs[blob_index].producer;
        ret = net->forward_layer(layer_index, blob_mats, opt);
    }

    feat = blob_mats[blob_index];

    return ret;
}

#if NCNN_STRING
int Extractor::input(const char* blob_name, const Mat& in)
{
    int blob_index = net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
        return -1;

    blob_mats[blob_index] = in;

    return 0;
}

int Extractor::extract(const char* blob_name, Mat& feat)
{
    int blob_index = net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
        return -1;

    int ret = 0;

    if (blob_mats[blob_index].dims == 0)
    {
        int layer_index = net->blobs[blob_index].producer;
        ret = net->forward_layer(layer_index, blob_mats, opt);
    }

    feat = blob_mats[blob_index];

    return ret;
}
#endif // NCNN_STRING

} // namespace ncnn
