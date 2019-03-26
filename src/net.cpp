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
#include "convolution.h"
#include "convolutiondepthwise.h"
#include "relu.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

#if NCNN_BENCHMARK
#include "benchmark.h"
#endif // NCNN_BENCHMARK

#if NCNN_VULKAN
#include "command.h"
#endif // NCNN_VULKAN

namespace ncnn {

Net::Net()
{
    use_winograd_convolution = 1;
    use_sgemm_convolution = 1;
    use_int8_inference = 1;
    use_vulkan_compute = 0;

#if NCNN_VULKAN
    vkdev = 0;
    vkdev_local = 0;
    weight_vkallocator = 0;
    weight_staging_vkallocator = 0;

    cast_float32_to_float16 = 0;
    cast_float16_to_float32 = 0;
    packing_pack1 = 0;
    packing_pack4 = 0;
#endif // NCNN_VULKAN
}

Net::~Net()
{
    clear();

#if NCNN_VULKAN
    delete vkdev_local;

    delete cast_float32_to_float16;
    delete cast_float16_to_float32;
    delete packing_pack1;
    delete packing_pack4;
#endif // NCNN_VULKAN
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
    int nbr = fscanf(fp, "%d", &magic);
    if (nbr != 1)
    {
        fprintf(stderr, "issue with param file\n");
        return -1;
    }
    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return -1;
    }

    // parse
    int layer_count = 0;
    int blob_count = 0;
    nbr = fscanf(fp, "%d %d", &layer_count, &blob_count);
    if (nbr != 2 || layer_count <= 0 || blob_count <= 0)
    {
        fprintf(stderr, "issue with param file\n");
        return -1;
    }

    layers.resize((size_t)layer_count);
    blobs.resize((size_t)blob_count);

#if NCNN_VULKAN
    if (use_vulkan_compute && !vkdev)
    {
        // use default vulkan device
        if (!vkdev_local)
            vkdev_local = new VulkanDevice;
        vkdev = vkdev_local;
    }
#endif // NCNN_VULKAN

    ParamDict pd;
    pd.use_winograd_convolution = use_winograd_convolution;
    pd.use_sgemm_convolution = use_sgemm_convolution;
    pd.use_int8_inference = use_int8_inference;
    pd.use_vulkan_compute = use_vulkan_compute;

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

#if NCNN_VULKAN
        if (use_vulkan_compute)
            layer->vkdev = vkdev;
#endif // NCNN_VULKAN

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
    *_internal_nconsumed_ptr = 0;

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

#if NCNN_VULKAN
    if (use_vulkan_compute && !vkdev)
    {
        // use default vulkan device
        if (!vkdev_local)
            vkdev_local = new VulkanDevice;
        vkdev = vkdev_local;
    }
#endif // NCNN_VULKAN

    ParamDict pd;
    pd.use_winograd_convolution = use_winograd_convolution;
    pd.use_sgemm_convolution = use_sgemm_convolution;
    pd.use_int8_inference = use_int8_inference;
    pd.use_vulkan_compute = use_vulkan_compute;

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

#if NCNN_VULKAN
        if (use_vulkan_compute)
            layer->vkdev = vkdev;
#endif // NCNN_VULKAN

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

template<typename T> bool readValue(T & val, FILE * fp)
{
    size_t res = fread(&val, sizeof(T), 1, fp);
    if (res != 1) {
        fprintf(stderr, "issue with param file reading\n");
        return false;
    }
    return true;
}

int Net::load_param_bin(FILE* fp)
{
    int magic = 0;
    if (!readValue(magic, fp))
        return -1;
    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return -1;
    }

    int layer_count = 0;
    if (!readValue(layer_count, fp))
        return -1;

    int blob_count = 0;
    if (!readValue(blob_count, fp))
        return -1;

    layers.resize(layer_count);
    blobs.resize(blob_count);

#if NCNN_VULKAN
    if (use_vulkan_compute && !vkdev)
    {
        // use default vulkan device
        if (!vkdev_local)
            vkdev_local = new VulkanDevice;
        vkdev = vkdev_local;
    }
#endif // NCNN_VULKAN

    ParamDict pd;
    pd.use_winograd_convolution = use_winograd_convolution;
    pd.use_sgemm_convolution = use_sgemm_convolution;
    pd.use_int8_inference = use_int8_inference;
    pd.use_vulkan_compute = use_vulkan_compute;

    for (int i=0; i<layer_count; i++)
    {
        int typeindex;
        if (!readValue(typeindex, fp))
            return -1;

        int bottom_count;
        if (!readValue(bottom_count, fp))
            return -1;

        int top_count;
        if (!readValue(top_count, fp))
            return -1;

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

#if NCNN_VULKAN
        if (use_vulkan_compute)
            layer->vkdev = vkdev;
#endif // NCNN_VULKAN

//         layer->type = std::string(layer_type);
//         layer->name = std::string(layer_name);
//         fprintf(stderr, "new layer %d\n", typeindex);

        layer->bottoms.resize(bottom_count);
        for (int j=0; j<bottom_count; j++)
        {
            int bottom_blob_index;
            if (!readValue(bottom_blob_index, fp))
                return -1;

            Blob& blob = blobs[bottom_blob_index];

            blob.consumers.push_back(i);

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int j=0; j<top_count; j++)
        {
            int top_blob_index;
            if (!readValue(top_blob_index, fp))
                return -1;

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
        
        //Here we found inconsistent content in the parameter file.
        if (!layer){
            fprintf(stderr, "load_model error at layer %d, parameter file has inconsistent content.\n", (int)i);
            ret = -1;
            break;
        }

        int lret = layer->load_model(mb);
        if (lret != 0)
        {
            fprintf(stderr, "layer load_model %d failed\n", (int)i);
            ret = -1;
            break;
        }
    }

#if NCNN_VULKAN
    if (use_vulkan_compute)
    {
        upload_model();

        create_pipeline();
    }
#endif // NCNN_VULKAN

    fuse_network();

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
        return -1;
    }

    const unsigned char* mem = _mem;

    int magic = *(int*)(mem);
    mem += 4;

    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return -1;
    }

    int layer_count = *(int*)(mem);
    mem += 4;

    int blob_count = *(int*)(mem);
    mem += 4;

    layers.resize(layer_count);
    blobs.resize(blob_count);

#if NCNN_VULKAN
    if (use_vulkan_compute && !vkdev)
    {
        // use default vulkan device
        if (!vkdev_local)
            vkdev_local = new VulkanDevice;
        vkdev = vkdev_local;
    }
#endif // NCNN_VULKAN

    ParamDict pd;
    pd.use_winograd_convolution = use_winograd_convolution;
    pd.use_sgemm_convolution = use_sgemm_convolution;
    pd.use_int8_inference = use_int8_inference;
    pd.use_vulkan_compute = use_vulkan_compute;

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
            return -1;
        }

#if NCNN_VULKAN
        if (use_vulkan_compute)
            layer->vkdev = vkdev;
#endif // NCNN_VULKAN

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
        return -1;
    }

    if ((unsigned long)_mem & 0x3)
    {
        // reject unaligned memory
        fprintf(stderr, "memory not 32-bit aligned at %p\n", _mem);
        return -1;
    }

    const unsigned char* mem = _mem;
    ModelBinFromMemory mb(mem);
    for (size_t i=0; i<layers.size(); i++)
    {
        Layer* layer = layers[i];

        //Here we found inconsistent content in the parameter file.
        if (!layer){
            fprintf(stderr, "load_model error at layer %d, parameter file has inconsistent content.\n", (int)i);
            return -1;
        }

        int lret = layer->load_model(mb);
        if (lret != 0)
        {
            fprintf(stderr, "layer load_model failed\n");
            return -1;
        }
    }

#if NCNN_VULKAN
    if (use_vulkan_compute)
    {
        upload_model();

        create_pipeline();
    }
#endif // NCNN_VULKAN

    return mem - _mem;
}

void Net::fuse_network()
{
    // set the int8 op fusion:requantize
#if NCNN_STRING && NCNN_REQUANT    
    // fprintf(stderr, "Test op fusion to int8 implement:\n");
    for (size_t i=0; i<layers.size(); i++)
    {
        Layer* layer = layers[i];

        if (layer->type == "Convolution" || layer->type == "ConvolutionDepthWise")
        {
            if (((Convolution*)layer)->use_int8_inference == false)
                continue;

            for (size_t n=0; n<blobs[layer->tops[0]].consumers.size(); n++)
            {
                int layer_next_index = blobs[layer->tops[0]].consumers[n];
                Layer* layer_next = layers[layer_next_index];

                if (layer_next->type == "ReLU")
                {
                    int layer_next_2_index = blobs[layer_next->tops[0]].consumers[0];
                    Layer* layer_next_2 = layers[layer_next_2_index];

                    if (layer_next_2->type == "Convolution" || layer_next_2->type == "ConvolutionDepthWise")
                    {
                        // fprintf(stderr, "%s, %s, %s\n", layer->name.c_str(), layer_next->name.c_str(), layer_next_2->name.c_str());
                        if (layer->type == "Convolution" && layer_next_2->type == "Convolution")
                        {
                            ((Convolution*)layer)->use_int8_requantize = true;
                            ((Convolution*)layer)->top_blob_int8_scale = ((Convolution*)layer_next_2)->bottom_blob_int8_scale;
                            ((Convolution*)layer)->create_requantize_op();
                        }
                        else if (layer->type == "ConvolutionDepthWise" && layer_next_2->type == "Convolution")
                        {
                            ((ConvolutionDepthWise*)layer)->use_int8_requantize = true;
                            ((ConvolutionDepthWise*)layer)->top_blob_int8_scale = ((Convolution*)layer_next_2)->bottom_blob_int8_scale;
                            ((ConvolutionDepthWise*)layer)->create_requantize_op();
                        }
                        else if (layer->type == "Convolution" && layer_next_2->type == "ConvolutionDepthWise")
                        {
                            ((Convolution*)layer)->use_int8_requantize = true;
                            ((Convolution*)layer)->top_blob_int8_scale = ((ConvolutionDepthWise*)layer_next_2)->bottom_blob_int8_scales[0];
                            ((Convolution*)layer)->create_requantize_op();
                        }
                        else
                        {
                            ((ConvolutionDepthWise*)layer)->use_int8_requantize = true;
                            ((ConvolutionDepthWise*)layer)->top_blob_int8_scale = ((ConvolutionDepthWise*)layer_next_2)->bottom_blob_int8_scales[0];
                            ((ConvolutionDepthWise*)layer)->create_requantize_op();
                        }
                    }
                    else if (layer_next_2->type == "Split")
                    {
                        bool all_conv = true;
                        for (size_t i=0; i<layer_next_2->tops.size(); i++)
                        {
                            int layer_next_3_index = blobs[layer_next_2->tops[i]].consumers[0];
                            if (layers[layer_next_3_index]->type != "Convolution" && layers[layer_next_3_index]->type != "ConvolutionDepthWise" && layers[layer_next_3_index]->type != "PriorBox" )
                            {
                                // fprintf(stderr, "%s, %s, %s, %s\n", layer->name.c_str(), layer_next->name.c_str(), layer_next_2->name.c_str(), layers[layer_next_3_index]->name.c_str());
                                all_conv = false;
                            }
                        }

                        if (all_conv == true && layer_next_2->tops.size() >= size_t(2))
                        {
                            // fprintf(stderr, "%s, %s, %s, ", layer->name.c_str(), layer_next->name.c_str(), layer_next_2->name.c_str());
                            for (size_t i=0; i<layer_next_2->tops.size(); i++)
                            {
                                int layer_next_3_index = blobs[layer_next_2->tops[i]].consumers[0];
                                Layer* layer_next_3 = layers[layer_next_3_index];

                                // fprintf(stderr, "%s, ", layer_next_3->name.c_str());
                                if (layer_next_3->type == "Convolution")
                                {
                                    ((Convolution*)layer)->top_blob_int8_scale = ((Convolution*)layer_next_3)->bottom_blob_int8_scale; 
                                }    
                            }

                            ((Convolution*)layer)->use_int8_requantize = true;
                            ((Convolution*)layer)->create_requantize_op();    
                            // fprintf(stderr, "\n");
                        }
                    }
                    else
                    {
                        // fprintf(stderr, "%s, %s\n", layer->name.c_str(), layer_next->name.c_str());
                    }
                }
                else if (layer_next->type == "Pooling")
                {
                    // ToDo
                }
                else
                {
                    // fprintf(stderr, "%s\n", layer->name.c_str());
                }                  
            }
        }
    }
#endif
}

void Net::clear()
{
#if NCNN_VULKAN
    destroy_pipeline();
#endif // NCNN_VULKAN

    blobs.clear();
    for (size_t i=0; i<layers.size(); i++)
    {
        delete layers[i];
    }
    layers.clear();

#if NCNN_VULKAN
    if (weight_vkallocator)
    {
        delete weight_vkallocator;
        weight_vkallocator = 0;
    }
    if (weight_staging_vkallocator)
    {
        delete weight_staging_vkallocator;
        weight_staging_vkallocator = 0;
    }
#endif // NCNN_VULKAN
}

Extractor Net::create_extractor() const
{
    return Extractor(this, blobs.size());
}

#if NCNN_VULKAN
void Net::set_vulkan_device(const VulkanDevice* _vkdev)
{
    vkdev = _vkdev;
}

int Net::upload_model()
{
    ncnn::VkTransfer cmd(vkdev);

    // create gpu device allocator if null
    if (!weight_vkallocator)
    {
        weight_vkallocator = new VkWeightBufferAllocator(vkdev);
    }
    if (!weight_staging_vkallocator)
    {
        weight_staging_vkallocator = new VkWeightStagingBufferAllocator(vkdev);
    }

    cmd.weight_vkallocator = weight_vkallocator;
    cmd.staging_vkallocator = weight_staging_vkallocator;

    for (size_t i=0; i<layers.size(); i++)
    {
        if (layers[i]->support_vulkan)
        {
            int uret = layers[i]->upload_model(cmd);
            if (uret != 0)
            {
                fprintf(stderr, "layer upload_model %d failed\n", (int)i);
                return -1;
            }
        }
    }

    cmd.submit();

    cmd.wait();

    return 0;
}

int Net::create_pipeline()
{
    //#pragma omp parallel for
    for (int i=0; i<(int)layers.size(); i++)
    {
        if (layers[i]->support_vulkan)
        {
            int cret = layers[i]->create_pipeline();
            if (cret != 0)
            {
                fprintf(stderr, "layer create_pipeline %d failed\n", (int)i);
            }
        }
    }

    if (vkdev->info.support_fp16_storage)
    {
        {
        cast_float32_to_float16 = ncnn::create_layer(ncnn::LayerType::Cast);
        cast_float32_to_float16->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 1);
        pd.set(1, 2);
        pd.use_vulkan_compute = 1;

        cast_float32_to_float16->load_param(pd);
        }

        {
        cast_float16_to_float32 = ncnn::create_layer(ncnn::LayerType::Cast);
        cast_float16_to_float32->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 2);
        pd.set(1, 1);
        pd.use_vulkan_compute = 1;

        cast_float16_to_float32->load_param(pd);
        }

        cast_float32_to_float16->create_pipeline();

        cast_float16_to_float32->create_pipeline();
    }

    {
    packing_pack1 = ncnn::create_layer(ncnn::LayerType::Packing);
    packing_pack1->vkdev = vkdev;

    ncnn::ParamDict pd;
    pd.set(0, 1);
    pd.use_vulkan_compute = 1;

    packing_pack1->load_param(pd);
    }

    {
    packing_pack4 = ncnn::create_layer(ncnn::LayerType::Packing);
    packing_pack4->vkdev = vkdev;

    ncnn::ParamDict pd;
    pd.set(0, 4);
    pd.use_vulkan_compute = 1;

    packing_pack4->load_param(pd);
    }

    packing_pack1->create_pipeline();

    packing_pack4->create_pipeline();

    return 0;
}

int Net::destroy_pipeline()
{
    //#pragma omp parallel for
    for (int i=0; i<(int)layers.size(); i++)
    {
        if (layers[i]->support_vulkan)
        {
            layers[i]->destroy_pipeline();
        }
    }

    if (cast_float32_to_float16)
        cast_float32_to_float16->destroy_pipeline();

    if (cast_float16_to_float32)
        cast_float16_to_float32->destroy_pipeline();

    if (packing_pack1)
        packing_pack1->destroy_pipeline();

    if (packing_pack4)
        packing_pack4->destroy_pipeline();

    return 0;
}
#endif // NCNN_VULKAN

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

#if NCNN_VULKAN
int Net::forward_layer(int layer_index, std::vector<Mat>& blob_mats, std::vector<VkMat>& blob_mats_gpu, std::vector<int>& wait_barrier_counts, VkCompute& cmd, Option& opt) const
{
    const Layer* layer = layers[layer_index];

//     fprintf(stderr, "forward_layer %d %d %s\n", layer->support_vulkan, layer_index, layer->name.c_str());

    if (layer->support_vulkan)
    {
        if (layer->one_blob_only)
        {
            // load bottom blob
            int bottom_blob_index = layer->bottoms[0];
            int top_blob_index = layer->tops[0];

            if (blob_mats_gpu[bottom_blob_index].dims == 0)
            {
                if (blob_mats[bottom_blob_index].dims == 0)
                {
                    int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, blob_mats_gpu, wait_barrier_counts, cmd, opt);
                    if (ret != 0)
                        return ret;
                }

                if (blob_mats_gpu[bottom_blob_index].dims == 0)
                {
                    const Mat& bottom_blob_cpu = blob_mats[bottom_blob_index];

                    // upload
                    VkMat bottom_blob_unpacked;
                    bottom_blob_unpacked.create_like(bottom_blob_cpu, opt.blob_vkallocator, opt.staging_vkallocator);

                    bottom_blob_unpacked.prepare_staging_buffer();
                    bottom_blob_unpacked.upload(bottom_blob_cpu);

                    cmd.record_upload(bottom_blob_unpacked);

                    // cast to fp16
                    VkMat bottom_blob_unpacked_fp16;
                    if (vkdev->info.support_fp16_storage)
                    {
                        cast_float32_to_float16->forward(bottom_blob_unpacked, bottom_blob_unpacked_fp16, cmd, opt);
                    }
                    else
                    {
                        bottom_blob_unpacked_fp16 = bottom_blob_unpacked;
                    }

                    // packing
                    VkMat& bottom_blob = blob_mats_gpu[bottom_blob_index];
                    packing_pack4->forward(bottom_blob_unpacked_fp16, bottom_blob, cmd, opt);

//                     fprintf(stderr, "upload %d %d %d %d  %lu %d\n", bottom_blob.total() * bottom_blob.elemsize, bottom_blob.w, bottom_blob.h, bottom_blob.c, bottom_blob.elemsize, bottom_blob.packing);
                }
            }

            VkMat bottom_blob = blob_mats_gpu[bottom_blob_index];

            if (opt.lightmode)
            {

                wait_barrier_counts[bottom_blob_index] += layer->tops.size();

                // deep copy for inplace forward if data is shared
                if (layer->support_inplace && wait_barrier_counts[bottom_blob_index] != 1)
                {
                    VkMat bottom_blob_copy;
                    bottom_blob_copy.create_like(bottom_blob, bottom_blob.allocator, bottom_blob.staging_allocator);

//                     fprintf(stderr, "clone %p %p\n", bottom_blob.buffer(), bottom_blob_copy.buffer());

                    cmd.record_clone(bottom_blob, bottom_blob_copy);
                    bottom_blob = bottom_blob_copy;

                    wait_barrier_counts[bottom_blob_index]--;
                }
            }

            // forward
            if (opt.lightmode && layer->support_inplace)
            {
                VkMat& bottom_top_blob = bottom_blob;
                int ret = layer->forward_inplace(bottom_top_blob, cmd, opt);
                if (ret != 0)
                    return ret;

                // store top blob
                blob_mats_gpu[top_blob_index] = bottom_top_blob;
            }
            else
            {
                VkMat top_blob;
                int ret = layer->forward(bottom_blob, top_blob, cmd, opt);
                if (ret != 0)
                    return ret;

                // store top blob
                blob_mats_gpu[top_blob_index] = top_blob;
            }

            if (opt.lightmode)
            {
                // reclaim producer bottom_blob as free when consuming bottom_blob
                const Layer* producer = layers[ blobs[bottom_blob_index].producer ];
                for (size_t i=0; i<producer->bottoms.size(); i++)
                {
                    int producer_bottom_blob_index = producer->bottoms[i];

                    wait_barrier_counts[producer_bottom_blob_index]--;
                    if (wait_barrier_counts[producer_bottom_blob_index] == 0)
                    {
//                         fprintf(stderr, "reclaim free %p\n", blob_mats_gpu[producer_bottom_blob_index].buffer());

                        blob_mats_gpu[producer_bottom_blob_index].release();
                    }
                }
            }
        }
        else
        {
            // load bottom blobs
            std::vector<VkMat> bottom_blobs;
            bottom_blobs.resize(layer->bottoms.size());
            for (size_t i=0; i<layer->bottoms.size(); i++)
            {
                int bottom_blob_index = layer->bottoms[i];

                if (blob_mats_gpu[bottom_blob_index].dims == 0)
                {
                    if (blob_mats[bottom_blob_index].dims == 0)
                    {
                        int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, blob_mats_gpu, wait_barrier_counts, cmd, opt);
                        if (ret != 0)
                            return ret;
                    }

                    if (blob_mats_gpu[bottom_blob_index].dims == 0)
                    {
                        const Mat& bottom_blob_cpu = blob_mats[bottom_blob_index];

                        // upload
                        VkMat bottom_blob_unpacked;
                        bottom_blob_unpacked.create_like(bottom_blob_cpu, opt.blob_vkallocator, opt.staging_vkallocator);

                        bottom_blob_unpacked.prepare_staging_buffer();
                        bottom_blob_unpacked.upload(bottom_blob_cpu);

                        cmd.record_upload(bottom_blob_unpacked);

                        // cast to fp16
                        VkMat bottom_blob_unpacked_fp16;
                        if (vkdev->info.support_fp16_storage)
                        {
                            cast_float32_to_float16->forward(bottom_blob_unpacked, bottom_blob_unpacked_fp16, cmd, opt);
                        }
                        else
                        {
                            bottom_blob_unpacked_fp16 = bottom_blob_unpacked;
                        }

                        // packing
                        VkMat& bottom_blob = blob_mats_gpu[bottom_blob_index];
                        packing_pack4->forward(bottom_blob_unpacked_fp16, bottom_blob, cmd, opt);

//                         fprintf(stderr, "upload %d %d %d %d  %lu %d\n", bottom_blob.total() * bottom_blob.elemsize, bottom_blob.w, bottom_blob.h, bottom_blob.c, bottom_blob.elemsize, bottom_blob.packing);
                    }
                }

                bottom_blobs[i] = blob_mats_gpu[bottom_blob_index];

                if (opt.lightmode)
                {

                    wait_barrier_counts[bottom_blob_index] = layer->tops.size();

                    // deep copy for inplace forward if data is shared
                    if (layer->support_inplace && wait_barrier_counts[bottom_blob_index] != 1)
                    {
                        VkMat bottom_blob_copy;
                        bottom_blob_copy.create_like(bottom_blobs[i], bottom_blobs[i].allocator, bottom_blobs[i].staging_allocator);

//                         fprintf(stderr, "clone %p %p\n", bottom_blobs[i].buffer(), bottom_blob_copy.buffer());

                        cmd.record_clone(bottom_blobs[i], bottom_blob_copy);
                        bottom_blobs[i] = bottom_blob_copy;

                        wait_barrier_counts[bottom_blob_index]--;
                    }
                }
            }

            // forward
            if (opt.lightmode && layer->support_inplace)
            {
                std::vector<VkMat>& bottom_top_blobs = bottom_blobs;
                int ret = layer->forward_inplace(bottom_top_blobs, cmd, opt);
                if (ret != 0)
                    return ret;

                // store top blobs
                for (size_t i=0; i<layer->tops.size(); i++)
                {
                    int top_blob_index = layer->tops[i];

                    blob_mats_gpu[top_blob_index] = bottom_top_blobs[i];
                }
            }
            else
            {
                std::vector<VkMat> top_blobs;
                top_blobs.resize(layer->tops.size());
                int ret = layer->forward(bottom_blobs, top_blobs, cmd, opt);
                if (ret != 0)
                    return ret;

                // store top blobs
                for (size_t i=0; i<layer->tops.size(); i++)
                {
                    int top_blob_index = layer->tops[i];

                    blob_mats_gpu[top_blob_index] = top_blobs[i];
                }
            }

            if (opt.lightmode)
            {
                for (size_t i=0; i<layer->bottoms.size(); i++)
                {
                    int bottom_blob_index = layer->bottoms[i];

                    // reclaim producer bottom_blob as free when consuming bottom_blob
                    const Layer* producer = layers[ blobs[bottom_blob_index].producer ];
                    for (size_t i=0; i<producer->bottoms.size(); i++)
                    {
                        int producer_bottom_blob_index = producer->bottoms[i];

                        wait_barrier_counts[producer_bottom_blob_index]--;
                        if (wait_barrier_counts[producer_bottom_blob_index] == 0)
                        {
//                             fprintf(stderr, "reclaim free %p\n", blob_mats_gpu[producer_bottom_blob_index].buffer());

                            blob_mats_gpu[producer_bottom_blob_index].release();
                        }
                    }
                }
            }

        }

    }
    else
    {
        if (layer->one_blob_only)
        {
            // load bottom blob
            int bottom_blob_index = layer->bottoms[0];
            int top_blob_index = layer->tops[0];

            if (blob_mats[bottom_blob_index].dims == 0)
            {
                if (blob_mats_gpu[bottom_blob_index].dims == 0)
                {
                    int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, blob_mats_gpu, wait_barrier_counts, cmd, opt);
                    if (ret != 0)
                        return ret;
                }

                if (blob_mats[bottom_blob_index].dims == 0)
                {
                    const VkMat& bottom_blob = blob_mats_gpu[bottom_blob_index];

                    // unpacking
                    VkMat bottom_blob_unpacked;
                    packing_pack1->forward(bottom_blob, bottom_blob_unpacked, cmd, opt);

                    // cast to fp32
                    VkMat bottom_blob_unpacked_fp32;
                    if (vkdev->info.support_fp16_storage)
                    {
                        cast_float16_to_float32->forward(bottom_blob_unpacked, bottom_blob_unpacked_fp32, cmd, opt);
                    }
                    else
                    {
                        bottom_blob_unpacked_fp32 = bottom_blob_unpacked;
                    }

                    // download
                    bottom_blob_unpacked_fp32.prepare_staging_buffer();
                    cmd.record_download(bottom_blob_unpacked_fp32);

                    cmd.submit();

                    cmd.wait();

                    cmd.reset();

                    Mat& bottom_blob_cpu = blob_mats[bottom_blob_index];
                    bottom_blob_cpu.create_like(bottom_blob_unpacked_fp32, opt.blob_allocator);
                    bottom_blob_unpacked_fp32.download(bottom_blob_cpu);

                    bottom_blob_unpacked_fp32.discard_staging_buffer();

//                     fprintf(stderr, "download %d %d %d %d  %lu %d\n", bottom_blob.total() * bottom_blob.elemsize, bottom_blob.w, bottom_blob.h, bottom_blob.c, bottom_blob.elemsize, bottom_blob.packing);
                }
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

                int ret = layer->forward_inplace(bottom_top_blob, opt);
                if (ret != 0)
                    return ret;

                // store top blob
                blob_mats[top_blob_index] = bottom_top_blob;
            }
            else
            {
                Mat top_blob;

                int ret = layer->forward(bottom_blob, top_blob, opt);
                if (ret != 0)
                    return ret;

                // store top blob
                blob_mats[top_blob_index] = top_blob;
            }

        }
        else
        {
            // load bottom blobs
            std::vector<VkMat> bottom_blobs_unpacked_fp32;
            bottom_blobs_unpacked_fp32.resize(layer->bottoms.size());
            for (size_t i=0; i<layer->bottoms.size(); i++)
            {
                int bottom_blob_index = layer->bottoms[i];

                if (blob_mats[bottom_blob_index].dims == 0)
                {
                    if (blob_mats_gpu[bottom_blob_index].dims == 0)
                    {
                        int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, blob_mats_gpu, wait_barrier_counts, cmd, opt);
                        if (ret != 0)
                            return ret;
                    }

                    if (blob_mats[bottom_blob_index].dims == 0)
                    {
                        const VkMat& bottom_blob = blob_mats_gpu[bottom_blob_index];

                        // unpacking
                        VkMat bottom_blob_unpacked;
                        packing_pack1->forward(bottom_blob, bottom_blob_unpacked, cmd, opt);

                        // cast to fp32
                        if (vkdev->info.support_fp16_storage)
                        {
                            cast_float16_to_float32->forward(bottom_blob_unpacked, bottom_blobs_unpacked_fp32[i], cmd, opt);
                        }
                        else
                        {
                            bottom_blobs_unpacked_fp32[i] = bottom_blob_unpacked;
                        }

                        // download
                        bottom_blobs_unpacked_fp32[i].prepare_staging_buffer();
                        cmd.record_download(bottom_blobs_unpacked_fp32[i]);
                    }
                }
            }

            {
                cmd.submit();

                cmd.wait();

                cmd.reset();
            }

            std::vector<Mat> bottom_blobs;
            bottom_blobs.resize(layer->bottoms.size());
            for (size_t i=0; i<layer->bottoms.size(); i++)
            {
                int bottom_blob_index = layer->bottoms[i];

                if (blob_mats[bottom_blob_index].dims == 0)
                {
                    Mat& bottom_blob_cpu = blob_mats[bottom_blob_index];
                    bottom_blob_cpu.create_like(bottom_blobs_unpacked_fp32[i], opt.blob_allocator);
                    bottom_blobs_unpacked_fp32[i].download(bottom_blob_cpu);

                    bottom_blobs_unpacked_fp32[i].discard_staging_buffer();

//                     fprintf(stderr, "download %d %d %d %d  %lu %d\n", bottom_blob.total() * bottom_blob.elemsize, bottom_blob.w, bottom_blob.h, bottom_blob.c, bottom_blob.elemsize, bottom_blob.packing);
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

                int ret = layer->forward_inplace(bottom_top_blobs, opt);
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

                int ret = layer->forward(bottom_blobs, top_blobs, opt);
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
    }

//     fprintf(stderr, "forward_layer %d %d %s done\n", layer->support_vulkan, layer_index, layer->name.c_str());

    return 0;
}
#endif // NCNN_VULKAN

Extractor::Extractor(const Net* _net, int blob_count) : net(_net)
{
    blob_mats.resize(blob_count);
    opt = get_default_option();

#if NCNN_VULKAN
    opt.vulkan_compute = net->use_vulkan_compute;

    if (net->use_vulkan_compute)
    {
        blob_mats_gpu.resize(blob_count);
        wait_barrier_counts.resize(blob_count, 0);

        // set default vulkan blob/workspace/staging allocator
        opt.blob_vkallocator = net->vkdev->allocator();
        opt.workspace_vkallocator = net->vkdev->allocator();
        opt.staging_vkallocator = net->vkdev->staging_allocator();
    }
#endif // NCNN_VULKAN
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

#if NCNN_VULKAN
void Extractor::set_vulkan_compute(bool enable)
{
    if (net->use_vulkan_compute)
    {
        opt.vulkan_compute = enable;
    }
    else
    {
        fprintf(stderr, "set_vulkan_compute failed, network use_vulkan_compute disabled\n");
    }
}

void Extractor::set_blob_vkallocator(VkAllocator* allocator)
{
    opt.blob_vkallocator = allocator;
}

void Extractor::set_workspace_vkallocator(VkAllocator* allocator)
{
    opt.workspace_vkallocator = allocator;
}

void Extractor::set_staging_vkallocator(VkAllocator* allocator)
{
    opt.staging_vkallocator = allocator;
}
#endif // NCNN_VULKAN

#if NCNN_STRING
int Extractor::input(const char* blob_name, const Mat& in)
{
    int blob_index = net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
        return -1;

    return input(blob_index, in);
}

int Extractor::extract(const char* blob_name, Mat& feat)
{
    int blob_index = net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
        return -1;

    return extract(blob_index, feat);
}
#endif // NCNN_STRING

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

#if NCNN_VULKAN
        if (opt.vulkan_compute)
        {
            ncnn::VkCompute cmd(net->vkdev);

            VkMat feat_gpu;
            ret = extract(blob_index, feat_gpu, cmd);

            if (blob_mats[blob_index].dims == 0 && feat_gpu.dims != 0)
            {
                // unpacking
                VkMat feat_gpu_unpacked;
                net->packing_pack1->forward(feat_gpu, feat_gpu_unpacked, cmd, opt);

                // cast to fp32
                VkMat feat_gpu_unpacked_fp32;
                if (net->vkdev->info.support_fp16_storage)
                {
                    net->cast_float16_to_float32->forward(feat_gpu_unpacked, feat_gpu_unpacked_fp32, cmd, opt);
                }
                else
                {
                    feat_gpu_unpacked_fp32 = feat_gpu_unpacked;
                }

                // download
                feat_gpu_unpacked_fp32.prepare_staging_buffer();
                cmd.record_download(feat_gpu_unpacked_fp32);

                cmd.submit();

                cmd.wait();

                Mat& feat_cpu = blob_mats[blob_index];
                feat_cpu.create_like(feat_gpu_unpacked_fp32, opt.blob_allocator);
                feat_gpu_unpacked_fp32.download(feat_cpu);

                feat_gpu_unpacked_fp32.discard_staging_buffer();
            }
        }
        else
        {
            ret = net->forward_layer(layer_index, blob_mats, opt);
        }
#else
        ret = net->forward_layer(layer_index, blob_mats, opt);
#endif // NCNN_VULKAN

    }

    feat = blob_mats[blob_index];

    return ret;
}

#if NCNN_VULKAN
#if NCNN_STRING
int Extractor::input(const char* blob_name, const VkMat& in)
{
    int blob_index = net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
        return -1;

    return input(blob_index, in);
}

int Extractor::extract(const char* blob_name, VkMat& feat, VkCompute& cmd)
{
    int blob_index = net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
        return -1;

    return extract(blob_index, feat, cmd);
}
#endif // NCNN_STRING

int Extractor::input(int blob_index, const VkMat& in)
{
    if (blob_index < 0 || blob_index >= (int)blob_mats.size())
        return -1;

    blob_mats_gpu[blob_index] = in;

    return 0;
}

int Extractor::extract(int blob_index, VkMat& feat, VkCompute& cmd)
{
    if (blob_index < 0 || blob_index >= (int)blob_mats.size())
        return -1;

    int ret = 0;

    if (blob_mats_gpu[blob_index].dims == 0)
    {
        int layer_index = net->blobs[blob_index].producer;
        ret = net->forward_layer(layer_index, blob_mats, blob_mats_gpu, wait_barrier_counts, cmd, opt);
    }

    feat = blob_mats_gpu[blob_index];

    return ret;
}
#endif // NCNN_VULKAN

} // namespace ncnn
