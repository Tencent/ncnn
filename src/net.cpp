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
#include "datareader.h"
#include "modelbin.h"
#include "paramdict.h"
#include "convolution.h"
#include "convolutiondepthwise.h"
#include "relu.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#if NCNN_BENCHMARK
#include "benchmark.h"
#endif // NCNN_BENCHMARK

#if NCNN_VULKAN
#include "command.h"
#endif // NCNN_VULKAN

namespace ncnn {

Net::Net()
{
#if NCNN_VULKAN
    vkdev = 0;
    weight_vkallocator = 0;
    weight_staging_vkallocator = 0;

    cast_float32_to_float16 = 0;
    cast_float16_to_float32 = 0;
    packing_pack1 = 0;
    packing_pack4 = 0;
    packing_pack8 = 0;
#endif // NCNN_VULKAN
}

Net::~Net()
{
    clear();

#if NCNN_VULKAN
    delete cast_float32_to_float16;
    delete cast_float16_to_float32;
    delete packing_pack1;
    delete packing_pack4;
    delete packing_pack8;
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

#if NCNN_STRING
int Net::load_param(const DataReader& dr)
{
#define SCAN_VALUE(fmt, v) \
    if (dr.scan(fmt, &v) != 1) \
    { \
        fprintf(stderr, "parse " #v " failed\n"); \
        return -1; \
    }

    int magic = 0;
    SCAN_VALUE("%d", magic)
    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return -1;
    }

    // parse
    int layer_count = 0;
    int blob_count = 0;
    SCAN_VALUE("%d", layer_count)
    SCAN_VALUE("%d", blob_count)
    if (layer_count <= 0 || blob_count <= 0)
    {
        fprintf(stderr, "invalid layer_count or blob_count\n");
        return -1;
    }

    layers.resize((size_t)layer_count);
    blobs.resize((size_t)blob_count);

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        if (!vkdev) vkdev = get_gpu_device();
        if (!vkdev) opt.use_vulkan_compute = false;// no vulkan device, fallback to cpu
    }
    if (opt.use_vulkan_compute)
    {
        // sanitize use options
        if (!vkdev->info.support_fp16_packed) opt.use_fp16_packed = false;
        if (!vkdev->info.support_fp16_storage) opt.use_fp16_storage = false;
        if (!vkdev->info.support_fp16_arithmetic) opt.use_fp16_arithmetic = false;
        if (!vkdev->info.support_int8_storage) opt.use_int8_storage = false;
        if (!vkdev->info.support_int8_arithmetic) opt.use_int8_arithmetic = false;
    }
#endif // NCNN_VULKAN

    ParamDict pd;

    int blob_index = 0;
    for (int i=0; i<layer_count; i++)
    {
        char layer_type[256];
        char layer_name[256];
        int bottom_count = 0;
        int top_count = 0;
        SCAN_VALUE("%255s", layer_type)
        SCAN_VALUE("%255s", layer_name)
        SCAN_VALUE("%d", bottom_count)
        SCAN_VALUE("%d", top_count)

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
        if (opt.use_vulkan_compute)
            layer->vkdev = vkdev;
#endif // NCNN_VULKAN

        layer->type = std::string(layer_type);
        layer->name = std::string(layer_name);
//         fprintf(stderr, "new layer %d %s\n", i, layer_name);

        layer->bottoms.resize(bottom_count);

        for (int j=0; j<bottom_count; j++)
        {
            char bottom_name[256];
            SCAN_VALUE("%255s", bottom_name)

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

            char blob_name[256];
            SCAN_VALUE("%255s", blob_name)

            blob.name = std::string(blob_name);
//             fprintf(stderr, "new blob %s\n", blob_name);

            blob.producer = i;

            layer->tops[j] = blob_index;

            blob_index++;
        }

        // layer specific params
        int pdlr = pd.load_param(dr);
        if (pdlr != 0)
        {
            fprintf(stderr, "ParamDict load_param failed\n");
            continue;
        }

        // pull out top shape hints
        Mat shape_hints = pd.get(30, Mat());
        if (!shape_hints.empty())
        {
            const int* psh = shape_hints;
            for (int j=0; j<top_count; j++)
            {
                Blob& blob = blobs[layer->tops[j]];

                int dims = psh[0];
                if (dims == 1)
                {
                    blob.shape = Mat(psh[1], (void*)0, 4u, 1);
                }
                if (dims == 2)
                {
                    blob.shape = Mat(psh[1], psh[2], (void*)0, 4u, 1);
                }
                if (dims == 3)
                {
                    blob.shape = Mat(psh[1], psh[2], psh[3], (void*)0, 4u, 1);
                }

                psh += 4;
            }
        }

        // set bottom and top shape hints
        layer->bottom_shapes.resize(bottom_count);
        for (int j=0; j<bottom_count; j++)
        {
            layer->bottom_shapes[j] = blobs[layer->bottoms[j]].shape;
        }

        layer->top_shapes.resize(top_count);
        for (int j=0; j<top_count; j++)
        {
            layer->top_shapes[j] = blobs[layer->tops[j]].shape;
        }

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            fprintf(stderr, "layer load_param failed\n");
            continue;
        }

        layers[i] = layer;
    }

#undef SCAN_VALUE
    return 0;
}
#endif // NCNN_STRING

int Net::load_param_bin(const DataReader& dr)
{
#define READ_VALUE(buf) \
    if (dr.read(&buf, sizeof(buf)) != sizeof(buf)) \
    { \
        fprintf(stderr, "read " #buf " failed\n"); \
        return -1; \
    }

    int magic = 0;
    READ_VALUE(magic)
    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return -1;
    }

    int layer_count = 0;
    int blob_count = 0;
    READ_VALUE(layer_count)
    READ_VALUE(blob_count)
    if (layer_count <= 0 || blob_count <= 0)
    {
        fprintf(stderr, "invalid layer_count or blob_count\n");
        return -1;
    }

    layers.resize(layer_count);
    blobs.resize(blob_count);

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        if (!vkdev) vkdev = get_gpu_device();
        if (!vkdev) opt.use_vulkan_compute = false;// no vulkan device, fallback to cpu
    }
    if (opt.use_vulkan_compute)
    {
        // sanitize use options
        if (!vkdev->info.support_fp16_packed) opt.use_fp16_packed = false;
        if (!vkdev->info.support_fp16_storage) opt.use_fp16_storage = false;
        if (!vkdev->info.support_fp16_arithmetic) opt.use_fp16_arithmetic = false;
        if (!vkdev->info.support_int8_storage) opt.use_int8_storage = false;
        if (!vkdev->info.support_int8_arithmetic) opt.use_int8_arithmetic = false;
    }
#endif // NCNN_VULKAN

    ParamDict pd;

    for (int i=0; i<layer_count; i++)
    {
        int typeindex;
        int bottom_count;
        int top_count;
        READ_VALUE(typeindex)
        READ_VALUE(bottom_count)
        READ_VALUE(top_count)

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
        if (opt.use_vulkan_compute)
            layer->vkdev = vkdev;
#endif // NCNN_VULKAN

//         layer->type = std::string(layer_type);
//         layer->name = std::string(layer_name);
//         fprintf(stderr, "new layer %d\n", typeindex);

        layer->bottoms.resize(bottom_count);
        for (int j=0; j<bottom_count; j++)
        {
            int bottom_blob_index;
            READ_VALUE(bottom_blob_index)

            Blob& blob = blobs[bottom_blob_index];

            blob.consumers.push_back(i);

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int j=0; j<top_count; j++)
        {
            int top_blob_index;
            READ_VALUE(top_blob_index)

            Blob& blob = blobs[top_blob_index];

//             blob.name = std::string(blob_name);
//             fprintf(stderr, "new blob %s\n", blob_name);

            blob.producer = i;

            layer->tops[j] = top_blob_index;
        }

        // layer specific params
        int pdlr = pd.load_param_bin(dr);
        if (pdlr != 0)
        {
            fprintf(stderr, "ParamDict load_param failed\n");
            continue;
        }

        // pull out top blob shape hints
        Mat shape_hints = pd.get(30, Mat());
        if (!shape_hints.empty())
        {
            const int* psh = shape_hints;
            for (int j=0; j<top_count; j++)
            {
                Blob& blob = blobs[layer->tops[j]];

                int dims = psh[0];
                if (dims == 1)
                {
                    blob.shape = Mat(psh[1], (void*)0, 4u, 1);
                }
                if (dims == 2)
                {
                    blob.shape = Mat(psh[1], psh[2], (void*)0, 4u, 1);
                }
                if (dims == 3)
                {
                    blob.shape = Mat(psh[1], psh[2], psh[3], (void*)0, 4u, 1);
                }

                psh += 4;
            }
        }

        // set bottom and top shape hints
        layer->bottom_shapes.resize(bottom_count);
        for (int j=0; j<bottom_count; j++)
        {
            layer->bottom_shapes[j] = blobs[layer->bottoms[j]].shape;
        }

        layer->top_shapes.resize(top_count);
        for (int j=0; j<top_count; j++)
        {
            layer->top_shapes[j] = blobs[layer->tops[j]].shape;
        }

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            fprintf(stderr, "layer load_param failed\n");
            continue;
        }

        layers[i] = layer;
    }

#undef READ_VALUE
    return 0;
}

int Net::load_model(const DataReader& dr)
{
    if (layers.empty())
    {
        fprintf(stderr, "network graph not ready\n");
        return -1;
    }

    // load file
    int ret = 0;

    ModelBinFromDataReader mb(dr);
    for (size_t i=0; i<layers.size(); i++)
    {
        Layer* layer = layers[i];

        //Here we found inconsistent content in the parameter file.
        if (!layer)
        {
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

    fuse_network();

    for (size_t i=0; i<layers.size(); i++)
    {
        Layer* layer = layers[i];

        //Here we found inconsistent content in the parameter file.
        if (!layer)
        {
            fprintf(stderr, "load_model error at layer %d, parameter file has inconsistent content.\n", (int)i);
            ret = -1;
            break;
        }

        int cret = layer->create_pipeline(opt);
        if (cret != 0)
        {
            fprintf(stderr, "layer create_pipeline %d failed\n", (int)i);
            ret = -1;
            break;
        }
    }

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        create_pipeline();

        upload_model();
    }
#endif // NCNN_VULKAN

    return ret;
}

#if NCNN_STDIO
#if NCNN_STRING
int Net::load_param(FILE* fp)
{
    DataReaderFromStdio dr(fp);
    return load_param(dr);
}

int Net::load_param_mem(const char* _mem)
{
    const unsigned char* mem = (const unsigned char*)_mem;
    DataReaderFromMemory dr(mem);
    return load_param(dr);
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
    DataReaderFromStdio dr(fp);
    return load_param_bin(dr);
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
    DataReaderFromStdio dr(fp);
    return load_model(dr);
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
    const unsigned char* mem = _mem;
    DataReaderFromMemory dr(mem);
    load_param_bin(dr);
    return static_cast<int>(mem - _mem);
}

int Net::load_model(const unsigned char* _mem)
{
    const unsigned char* mem = _mem;
    DataReaderFromMemory dr(mem);
    load_model(dr);
    return static_cast<int>(mem - _mem);
}

#if __ANDROID_API__ >= 9
#if NCNN_STRING
int Net::load_param(AAsset* asset)
{
    DataReaderFromAndroidAsset dr(asset);
    return load_param(dr);
}

int Net::load_param(AAssetManager* mgr, const char* assetpath)
{
    AAsset* asset = AAssetManager_open(mgr, assetpath, AASSET_MODE_BUFFER);
    if (!asset)
    {
        fprintf(stderr, "AAssetManager_open %s failed\n", assetpath);
        return -1;
    }

    int ret = load_param(asset);
    AAsset_close(asset);
    return ret;
}
#endif // NCNN_STRING

int Net::load_param_bin(AAsset* asset)
{
    DataReaderFromAndroidAsset dr(asset);
    return load_param_bin(dr);
}

int Net::load_param_bin(AAssetManager* mgr, const char* assetpath)
{
    AAsset* asset = AAssetManager_open(mgr, assetpath, AASSET_MODE_BUFFER);
    if (!asset)
    {
        fprintf(stderr, "AAssetManager_open %s failed\n", assetpath);
        return -1;
    }

    int ret = load_param_bin(asset);
    AAsset_close(asset);
    return ret;
}

int Net::load_model(AAsset* asset)
{
    DataReaderFromAndroidAsset dr(asset);
    return load_model(dr);
}

int Net::load_model(AAssetManager* mgr, const char* assetpath)
{
    AAsset* asset = AAssetManager_open(mgr, assetpath, AASSET_MODE_STREAMING);
    if (!asset)
    {
        fprintf(stderr, "AAssetManager_open %s failed\n", assetpath);
        return -1;
    }

    int ret = load_model(asset);
    AAsset_close(asset);
    return ret;
}
#endif // __ANDROID_API__ >= 9

int Net::fuse_network()
{
    // set the int8 op fusion:requantize
#if NCNN_STRING && NCNN_REQUANT    
    // fprintf(stderr, "Test op fusion to int8 implement:\n");
    // parse the network whether is a quantization model
    bool net_quantized = false;
    for (size_t i=0; i<layers.size(); i++)
    {
        Layer* layer = layers[i];
        if (layer->type == "Convolution" || layer->type == "ConvolutionDepthWise")
        {
            if (layer->type == "Convolution" && (((Convolution*)layer)->weight_data.elemsize != 1u))
                continue;
            if (layer->type == "ConvolutionDepthWise" && (((ConvolutionDepthWise*)layer)->weight_data.elemsize != 1u))
                continue;    
            net_quantized = true;
        }
    }

    if (net_quantized == false)
        return 0;

    for (size_t i=0; i<layers.size(); i++)
    {
        Layer* layer = layers[i];

        if (layer->type == "Convolution" || layer->type == "ConvolutionDepthWise")
        {
            if (layer->type == "Convolution" && (((Convolution*)layer)->weight_data.elemsize != 1u))
                continue;
            if (layer->type == "ConvolutionDepthWise" && (((ConvolutionDepthWise*)layer)->weight_data.elemsize != 1u))
                continue;

            for (size_t n=0; n<blobs[layer->tops[0]].consumers.size(); n++)
            {
                int layer_next_index = blobs[layer->tops[0]].consumers[n];
                Layer* layer_next = layers[layer_next_index];

                if (layer_next->type == "Convolution" || layer_next->type == "ConvolutionDepthWise")
                {
                    if (layer_next->type == "Convolution" && ((Convolution*)layer_next)->weight_data.elemsize != 1u)
                        continue;
                    if (layer_next->type == "ConvolutionDepthWise" && ((ConvolutionDepthWise*)layer_next)->weight_data.elemsize != 1u)
                        continue;    

                    // fprintf(stderr, "%s, %s\n", layer->name.c_str(), layer_next->name.c_str());
                    if (layer->type == "Convolution" && layer_next->type == "Convolution")
                    {
                        ((Convolution*)layer)->use_int8_requantize = true;
                        ((Convolution*)layer)->top_blob_int8_scale = ((Convolution*)layer_next)->bottom_blob_int8_scale;
                    }
                    else if (layer->type == "ConvolutionDepthWise" && layer_next->type == "Convolution")
                    {
                        ((ConvolutionDepthWise*)layer)->use_int8_requantize = true;
                        ((ConvolutionDepthWise*)layer)->top_blob_int8_scale = ((Convolution*)layer_next)->bottom_blob_int8_scale;
                    }
                    else if (layer->type == "Convolution" && layer_next->type == "ConvolutionDepthWise")
                    {
                        ((Convolution*)layer)->use_int8_requantize = true;
                        ((Convolution*)layer)->top_blob_int8_scale = ((ConvolutionDepthWise*)layer_next)->bottom_blob_int8_scales[0];
                    }
                    else
                    {
                        ((ConvolutionDepthWise*)layer)->use_int8_requantize = true;
                        ((ConvolutionDepthWise*)layer)->top_blob_int8_scale = ((ConvolutionDepthWise*)layer_next)->bottom_blob_int8_scales[0];
                    }
                }                  
                else if (layer_next->type == "ReLU")
                {
                    int layer_next_2_index = blobs[layer_next->tops[0]].consumers[0];
                    Layer* layer_next_2 = layers[layer_next_2_index];

                    if (layer_next_2->type == "Convolution" || layer_next_2->type == "ConvolutionDepthWise")
                    {
                        if (layer_next_2->type == "Convolution" && ((Convolution*)layer_next_2)->weight_data.elemsize != 1u)
                            continue;
                        if (layer_next_2->type == "ConvolutionDepthWise" && ((ConvolutionDepthWise*)layer_next_2)->weight_data.elemsize != 1u)
                            continue;    

//                         fprintf(stderr, "%s, %s, %s\n", layer->name.c_str(), layer_next->name.c_str(), layer_next_2->name.c_str());
                        if (layer->type == "Convolution" && layer_next_2->type == "Convolution")
                        {
                            ((Convolution*)layer)->use_int8_requantize = true;
                            ((Convolution*)layer)->top_blob_int8_scale = ((Convolution*)layer_next_2)->bottom_blob_int8_scale;
                        }
                        else if (layer->type == "ConvolutionDepthWise" && layer_next_2->type == "Convolution")
                        {
                            ((ConvolutionDepthWise*)layer)->use_int8_requantize = true;
                            ((ConvolutionDepthWise*)layer)->top_blob_int8_scale = ((Convolution*)layer_next_2)->bottom_blob_int8_scale;
                        }
                        else if (layer->type == "Convolution" && layer_next_2->type == "ConvolutionDepthWise")
                        {
                            ((Convolution*)layer)->use_int8_requantize = true;
                            ((Convolution*)layer)->top_blob_int8_scale = ((ConvolutionDepthWise*)layer_next_2)->bottom_blob_int8_scales[0];
                        }
                        else
                        {
                            ((ConvolutionDepthWise*)layer)->use_int8_requantize = true;
                            ((ConvolutionDepthWise*)layer)->top_blob_int8_scale = ((ConvolutionDepthWise*)layer_next_2)->bottom_blob_int8_scales[0];
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
    return 0;
}

void Net::clear()
{
#if NCNN_VULKAN
    destroy_pipeline();
#endif // NCNN_VULKAN

    blobs.clear();
    for (size_t i=0; i<layers.size(); i++)
    {
        int dret = layers[i]->destroy_pipeline(opt);
        if (dret != 0)
        {
            fprintf(stderr, "layer destroy_pipeline failed\n");
            // ignore anyway
        }

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
void Net::set_vulkan_device(int device_index)
{
    vkdev = get_gpu_device(device_index);
}

void Net::set_vulkan_device(const VulkanDevice* _vkdev)
{
    vkdev = _vkdev;
}

const VulkanDevice* Net::vulkan_device() const
{
    return vkdev;
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
            int uret = layers[i]->upload_model(cmd, opt);
            if (uret != 0)
            {
                fprintf(stderr, "layer upload_model %d failed\n", (int)i);
                return -1;
            }
        }
    }

    cmd.submit_and_wait();

    return 0;
}

int Net::create_pipeline()
{
    if (opt.use_fp16_storage && vkdev->info.type != 0)
    {
        {
        cast_float32_to_float16 = ncnn::create_layer(ncnn::LayerType::Cast);
        cast_float32_to_float16->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 1);
        pd.set(1, 2);

        cast_float32_to_float16->load_param(pd);
        }

        {
        cast_float16_to_float32 = ncnn::create_layer(ncnn::LayerType::Cast);
        cast_float16_to_float32->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, 2);
        pd.set(1, 1);

        cast_float16_to_float32->load_param(pd);
        }

        cast_float32_to_float16->create_pipeline(opt);

        cast_float16_to_float32->create_pipeline(opt);
    }

    {
    packing_pack1 = ncnn::create_layer(ncnn::LayerType::Packing);
    packing_pack1->vkdev = vkdev;

    ncnn::ParamDict pd;
    pd.set(0, 1);

    packing_pack1->load_param(pd);
    }

    {
    packing_pack4 = ncnn::create_layer(ncnn::LayerType::Packing);
    packing_pack4->vkdev = vkdev;

    ncnn::ParamDict pd;
    pd.set(0, 4);

    packing_pack4->load_param(pd);
    }

    {
    packing_pack8 = ncnn::create_layer(ncnn::LayerType::Packing);
    packing_pack8->vkdev = vkdev;

    ncnn::ParamDict pd;
    pd.set(0, 8);

    packing_pack8->load_param(pd);
    }

    packing_pack1->create_pipeline(opt);
    packing_pack4->create_pipeline(opt);
    packing_pack8->create_pipeline(opt);

    return 0;
}

int Net::destroy_pipeline()
{
    if (cast_float32_to_float16)
        cast_float32_to_float16->destroy_pipeline(opt);

    if (cast_float16_to_float32)
        cast_float16_to_float32->destroy_pipeline(opt);

    if (packing_pack1)
        packing_pack1->destroy_pipeline(opt);

    if (packing_pack4)
        packing_pack4->destroy_pipeline(opt);

    if (packing_pack8)
        packing_pack8->destroy_pipeline(opt);

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
            return static_cast<int>(i);
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
            return static_cast<int>(i);
        }
    }

    fprintf(stderr, "find_layer_index_by_name %s failed\n", name);
    return -1;
}

int Net::custom_layer_to_index(const char* type)
{
    const size_t custom_layer_registry_entry_count = custom_layer_registry.size();
    for (size_t i=0; i<custom_layer_registry_entry_count; i++)
    {
        if (strcmp(type, custom_layer_registry[i].name) == 0)
            return static_cast<int>(i);
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
    const size_t custom_layer_registry_entry_count = custom_layer_registry.size();
    if (index < 0 || static_cast<unsigned int>(index) >= custom_layer_registry_entry_count)
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

        if (opt.use_bf16_storage)
        {
            if (bottom_blob.elemsize / bottom_blob.elempack == 4u && layer->support_bf16_storage)
            {
                Mat bottom_blob_bf16;
                cast_float32_to_bfloat16(bottom_blob, bottom_blob_bf16, opt);
                bottom_blob = bottom_blob_bf16;
            }
            if (bottom_blob.elemsize / bottom_blob.elempack == 2u && !layer->support_bf16_storage)
            {
                Mat bottom_blob_fp32;
                cast_bfloat16_to_float32(bottom_blob, bottom_blob_fp32, opt);
                bottom_blob = bottom_blob_fp32;
            }
        }

        if (opt.use_packing_layout)
        {
            int elempack = layer->support_packing ? 4 : 1;

            Mat bottom_blob_packed;
            convert_packing(bottom_blob, bottom_blob_packed, elempack, opt);
            bottom_blob = bottom_blob_packed;
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
        std::vector<Mat> bottom_blobs(layer->bottoms.size());
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

            if (opt.use_bf16_storage)
            {
                if (bottom_blobs[i].elemsize / bottom_blobs[i].elempack == 4u && layer->support_bf16_storage)
                {
                    Mat bottom_blob_bf16;
                    cast_float32_to_bfloat16(bottom_blobs[i], bottom_blob_bf16, opt);
                    bottom_blobs[i] = bottom_blob_bf16;
                }
                if (bottom_blobs[i].elemsize / bottom_blobs[i].elempack == 2u && !layer->support_bf16_storage)
                {
                    Mat bottom_blob_fp32;
                    cast_bfloat16_to_float32(bottom_blobs[i], bottom_blob_fp32, opt);
                    bottom_blobs[i] = bottom_blob_fp32;
                }
            }

            if (opt.use_packing_layout)
            {
                int elempack = layer->support_packing ? 4 : 1;

                Mat bottom_blob_packed;
                convert_packing(bottom_blobs[i], bottom_blob_packed, elempack, opt);
                bottom_blobs[i] = bottom_blob_packed;
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
            std::vector<Mat> top_blobs(layer->tops.size());
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
int Net::forward_layer(int layer_index, std::vector<Mat>& blob_mats, std::vector<VkMat>& blob_mats_gpu, VkCompute& cmd, Option& opt) const
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
                    int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, blob_mats_gpu, cmd, opt);
                    if (ret != 0)
                        return ret;
                }

                if (blob_mats_gpu[bottom_blob_index].dims == 0)
                {
                    const Mat& bottom_blob_cpu = blob_mats[bottom_blob_index];

                    // cpu cast to fp16 (discrete gpu)
                    Mat bottom_blob_cpu_fp16;
                    if (opt.use_fp16_storage && vkdev->info.type == 0)
                    {
                        ncnn::cast_float32_to_float16(bottom_blob_cpu, bottom_blob_cpu_fp16, opt);
                    }
                    else if (bottom_blob_cpu.elempack == 4 && opt.use_fp16_packed && !opt.use_fp16_storage && vkdev->info.type == 0)
                    {
                        ncnn::cast_float32_to_float16(bottom_blob_cpu, bottom_blob_cpu_fp16, opt);
                    }
                    else
                    {
                        bottom_blob_cpu_fp16 = bottom_blob_cpu;
                    }

                    // upload
                    VkMat bottom_blob_unpacked;
                    bottom_blob_unpacked.create_like(bottom_blob_cpu_fp16, opt.blob_vkallocator, opt.staging_vkallocator);

                    bottom_blob_unpacked.prepare_staging_buffer();
                    bottom_blob_unpacked.upload(bottom_blob_cpu_fp16);

                    cmd.record_upload(bottom_blob_unpacked);

                    // cast to fp16 (integrated gpu)
                    VkMat bottom_blob_unpacked_fp16;
                    if (opt.use_fp16_storage && vkdev->info.type != 0)
                    {
                        cast_float32_to_float16->forward(bottom_blob_unpacked, bottom_blob_unpacked_fp16, cmd, opt);
                    }
                    else
                    {
                        bottom_blob_unpacked_fp16 = bottom_blob_unpacked;
                    }

                    // packing
                    VkMat& bottom_blob = blob_mats_gpu[bottom_blob_index];
                    if (opt.use_shader_pack8)
                    {
                        packing_pack8->forward(bottom_blob_unpacked_fp16, bottom_blob, cmd, opt);
                        if (bottom_blob.elempack != 8)
                            packing_pack4->forward(bottom_blob_unpacked_fp16, bottom_blob, cmd, opt);
                    }
                    else
                        packing_pack4->forward(bottom_blob_unpacked_fp16, bottom_blob, cmd, opt);

//                     fprintf(stderr, "upload %p[+%lu]\n", bottom_blob.buffer(), bottom_blob.buffer_offset());
                }
            }

            VkMat bottom_blob = blob_mats_gpu[bottom_blob_index];

            if (opt.lightmode)
            {
                // delete after taken in light mode
                blob_mats_gpu[bottom_blob_index].release();
                // deep copy for inplace forward if data is shared
                if (layer->support_inplace && *bottom_blob.refcount != 1)
                {
                    VkMat bottom_blob_copy;
                    bottom_blob_copy.create_like(bottom_blob, bottom_blob.allocator, bottom_blob.staging_allocator);

//                     fprintf(stderr, "clone %p[+%lu] %p[+%lu]\n", bottom_blob.buffer(), bottom_blob.buffer_offset(), bottom_blob_copy.buffer(), bottom_blob_copy.buffer_offset());

                    cmd.record_clone(bottom_blob, bottom_blob_copy);
                    bottom_blob = bottom_blob_copy;
                }
            }

            // forward
            if (opt.lightmode && layer->support_inplace)
            {
                VkMat& bottom_top_blob = bottom_blob;
#if NCNN_BENCHMARK
                cmd.record_write_timestamp(layer_index * 2);
                int ret = layer->forward_inplace(bottom_top_blob, cmd, opt);
                cmd.record_write_timestamp(layer_index * 2 + 1);
#else
                int ret = layer->forward_inplace(bottom_top_blob, cmd, opt);
#endif // NCNN_BENCHMARK
                if (ret != 0)
                    return ret;

                // store top blob
                blob_mats_gpu[top_blob_index] = bottom_top_blob;
            }
            else
            {
                VkMat top_blob;
#if NCNN_BENCHMARK
                cmd.record_write_timestamp(layer_index * 2);
                int ret = layer->forward(bottom_blob, top_blob, cmd, opt);
                cmd.record_write_timestamp(layer_index * 2 + 1);
#else
                int ret = layer->forward(bottom_blob, top_blob, cmd, opt);
#endif // NCNN_BENCHMARK
                if (ret != 0)
                    return ret;

                // store top blob
                blob_mats_gpu[top_blob_index] = top_blob;
            }
        }
        else
        {
            // load bottom blobs
            std::vector<VkMat> bottom_blobs(layer->bottoms.size());
            std::vector<VkMat> bottom_blobs_unpacked(layer->bottoms.size());
            for (size_t i=0; i<layer->bottoms.size(); i++)
            {
                int bottom_blob_index = layer->bottoms[i];

                if (blob_mats_gpu[bottom_blob_index].dims == 0)
                {
                    if (blob_mats[bottom_blob_index].dims == 0)
                    {
                        int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, blob_mats_gpu, cmd, opt);
                        if (ret != 0)
                            return ret;
                    }

                    if (blob_mats_gpu[bottom_blob_index].dims == 0)
                    {
                        const Mat& bottom_blob_cpu = blob_mats[bottom_blob_index];

                        // cast to fp16 (discrete gpu)
                        Mat bottom_blob_cpu_fp16;
                        if (opt.use_fp16_storage && vkdev->info.type == 0)
                        {
                            ncnn::cast_float32_to_float16(bottom_blob_cpu, bottom_blob_cpu_fp16, opt);
                        }
                        else if (bottom_blob_cpu.elempack == 4 && opt.use_fp16_packed && !opt.use_fp16_storage && vkdev->info.type == 0)
                        {
                            ncnn::cast_float32_to_float16(bottom_blob_cpu, bottom_blob_cpu_fp16, opt);
                        }
                        else
                        {
                            bottom_blob_cpu_fp16 = bottom_blob_cpu;
                        }

                        // upload
                        VkMat& bottom_blob_unpacked = bottom_blobs_unpacked[i];
                        bottom_blob_unpacked.create_like(bottom_blob_cpu_fp16, opt.blob_vkallocator, opt.staging_vkallocator);

                        bottom_blob_unpacked.prepare_staging_buffer();
                        bottom_blob_unpacked.upload(bottom_blob_cpu_fp16);

                        cmd.record_upload(bottom_blob_unpacked);

                        // cast to fp16 (integrated gpu)
                        VkMat bottom_blob_unpacked_fp16;
                        if (opt.use_fp16_storage && vkdev->info.type != 0)
                        {
                            cast_float32_to_float16->forward(bottom_blob_unpacked, bottom_blob_unpacked_fp16, cmd, opt);
                        }
                        else
                        {
                            bottom_blob_unpacked_fp16 = bottom_blob_unpacked;
                        }

                        // packing
                        VkMat& bottom_blob = blob_mats_gpu[bottom_blob_index];
                        if (opt.use_shader_pack8)
                        {
                            packing_pack8->forward(bottom_blob_unpacked, bottom_blob, cmd, opt);
                            if (bottom_blob.elempack != 8)
                                packing_pack4->forward(bottom_blob_unpacked, bottom_blob, cmd, opt);
                        }
                        else
                            packing_pack4->forward(bottom_blob_unpacked, bottom_blob, cmd, opt);

//                         fprintf(stderr, "upload %p[+%lu]\n", bottom_blob.buffer(), bottom_blob.buffer_offset());
                    }
                }

                bottom_blobs[i] = blob_mats_gpu[bottom_blob_index];

                if (opt.lightmode)
                {
                    // delete after taken in light mode
                    blob_mats_gpu[bottom_blob_index].release();
                    // deep copy for inplace forward if data is shared
                    if (layer->support_inplace && *bottom_blobs[i].refcount != 1)
                    {
                        VkMat bottom_blob_copy;
                        bottom_blob_copy.create_like(bottom_blobs[i], bottom_blobs[i].allocator, bottom_blobs[i].staging_allocator);

//                         fprintf(stderr, "clone %p[+%lu] %p[+%lu]\n", bottom_blobs[i].buffer(), bottom_blobs[i].buffer_offset(), bottom_blob_copy.buffer(), bottom_blob_copy.buffer_offset());

                        cmd.record_clone(bottom_blobs[i], bottom_blob_copy);
                        bottom_blobs[i] = bottom_blob_copy;
                    }
                }
            }

            // forward
            if (opt.lightmode && layer->support_inplace)
            {
                std::vector<VkMat>& bottom_top_blobs = bottom_blobs;
#if NCNN_BENCHMARK
                cmd.record_write_timestamp(layer_index * 2);
                int ret = layer->forward_inplace(bottom_top_blobs, cmd, opt);
                cmd.record_write_timestamp(layer_index * 2 + 1);
#else
                int ret = layer->forward_inplace(bottom_top_blobs, cmd, opt);
#endif // NCNN_BENCHMARK
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
                std::vector<VkMat> top_blobs(layer->tops.size());
#if NCNN_BENCHMARK
                cmd.record_write_timestamp(layer_index * 2);
                int ret = layer->forward(bottom_blobs, top_blobs, cmd, opt);
                cmd.record_write_timestamp(layer_index * 2 + 1);
#else
                int ret = layer->forward(bottom_blobs, top_blobs, cmd, opt);
#endif // NCNN_BENCHMARK
                if (ret != 0)
                    return ret;

                // store top blobs
                for (size_t i=0; i<layer->tops.size(); i++)
                {
                    int top_blob_index = layer->tops[i];

                    blob_mats_gpu[top_blob_index] = top_blobs[i];
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
                    int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, blob_mats_gpu, cmd, opt);
                    if (ret != 0)
                        return ret;
                }

                if (blob_mats[bottom_blob_index].dims == 0)
                {
                    VkMat bottom_blob = blob_mats_gpu[bottom_blob_index];

//                     fprintf(stderr, "download %p[+%lu]\n", bottom_blob.buffer(), bottom_blob.buffer_offset());

                    if (opt.lightmode)
                    {
                        // delete after taken in light mode
                        blob_mats_gpu[bottom_blob_index].release();
                        // deep copy for inplace forward if data is shared
                        if (layer->support_inplace && *bottom_blob.refcount != 1)
                        {
                            VkMat bottom_blob_copy;
                            bottom_blob_copy.create_like(bottom_blob, bottom_blob.allocator, bottom_blob.staging_allocator);

//                             fprintf(stderr, "clone %p[+%lu] %p[+%lu]\n", bottom_blob.buffer(), bottom_blob.buffer_offset(), bottom_blob_copy.buffer(), bottom_blob_copy.buffer_offset());

                            cmd.record_clone(bottom_blob, bottom_blob_copy);
                            bottom_blob = bottom_blob_copy;
                        }
                    }

                    VkMat bottom_blob_unpacked_fp16;
                    if (opt.use_packing_layout && layer->support_packing)
                    {
//                         bottom_blob_unpacked_fp16 = bottom_blob;
                        packing_pack4->forward(bottom_blob, bottom_blob_unpacked_fp16, cmd, opt);
                    }
                    else
                    {
                        // unpacking
                        packing_pack1->forward(bottom_blob, bottom_blob_unpacked_fp16, cmd, opt);
                    }

                    // cast to fp32 (integrated gpu)
                    VkMat bottom_blob_unpacked;
                    if (opt.use_fp16_storage && vkdev->info.type != 0)
                    {
                        cast_float16_to_float32->forward(bottom_blob_unpacked_fp16, bottom_blob_unpacked, cmd, opt);
                    }
                    else
                    {
                        bottom_blob_unpacked = bottom_blob_unpacked_fp16;
                    }

                    // download
                    bottom_blob_unpacked.prepare_staging_buffer();
                    cmd.record_download(bottom_blob_unpacked);

                    cmd.submit_and_wait();

#if NCNN_BENCHMARK
                    std::vector<uint64_t> results(layer_index * 2);
                    cmd.get_query_pool_results(0, layer_index * 2, results);
                    for (int i=0; i<layer_index; i++)
                    {
                        uint64_t start = results[i*2];
                        uint64_t end = results[i*2+1];
                        if (start == 0 || end == 0)
                            continue;

                        double duration_us = (end - start) * vkdev->info.timestamp_period / 1000;
                        fprintf(stderr, "%-24s %-30s %8.2lfus    |\n", layers[i]->type.c_str(), layers[i]->name.c_str(), duration_us);
                    }
#endif // NCNN_BENCHMARK

                    cmd.reset();

                    Mat bottom_blob_cpu_fp16;
                    bottom_blob_cpu_fp16.create_like(bottom_blob_unpacked, opt.blob_allocator);
                    bottom_blob_unpacked.download(bottom_blob_cpu_fp16);

                    bottom_blob_unpacked.discard_staging_buffer();

                    // cast to fp32 (discrete gpu)
                    Mat& bottom_blob_cpu = blob_mats[bottom_blob_index];
                    if (opt.use_fp16_storage && vkdev->info.type == 0)
                    {
                        ncnn::cast_float16_to_float32(bottom_blob_cpu_fp16, bottom_blob_cpu, opt);
                    }
                    else if (bottom_blob_cpu_fp16.elempack == 4 && opt.use_fp16_packed && !opt.use_fp16_storage && vkdev->info.type == 0)
                    {
                        ncnn::cast_float16_to_float32(bottom_blob_cpu_fp16, bottom_blob_cpu, opt);
                    }
                    else
                    {
                        bottom_blob_cpu = bottom_blob_cpu_fp16;
                    }
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

            if (opt.use_packing_layout)
            {
                int elempack = layer->support_packing ? 4 : 1;

                Mat bottom_blob_packed;
                convert_packing(bottom_blob, bottom_blob_packed, elempack, opt);
                bottom_blob = bottom_blob_packed;
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
            std::vector<VkMat> bottom_blobs_unpacked(layer->bottoms.size());
            for (size_t i=0; i<layer->bottoms.size(); i++)
            {
                int bottom_blob_index = layer->bottoms[i];

                if (blob_mats[bottom_blob_index].dims == 0)
                {
                    if (blob_mats_gpu[bottom_blob_index].dims == 0)
                    {
                        int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, blob_mats_gpu, cmd, opt);
                        if (ret != 0)
                            return ret;
                    }

                    if (blob_mats[bottom_blob_index].dims == 0)
                    {
                        VkMat bottom_blob = blob_mats_gpu[bottom_blob_index];

//                         fprintf(stderr, "download %p[+%lu]\n", bottom_blob.buffer(), bottom_blob.buffer_offset());

                        if (opt.lightmode)
                        {
                            // delete after taken in light mode
                            blob_mats_gpu[bottom_blob_index].release();
                            // deep copy for inplace forward if data is shared
                            if (layer->support_inplace && *bottom_blob.refcount != 1)
                            {
                                VkMat bottom_blob_copy;
                                bottom_blob_copy.create_like(bottom_blob, bottom_blob.allocator, bottom_blob.staging_allocator);

//                                 fprintf(stderr, "clone %p[+%lu] %p[+%lu]\n", bottom_blob.buffer(), bottom_blob.buffer_offset(), bottom_blob_copy.buffer(), bottom_blob_copy.buffer_offset());

                                cmd.record_clone(bottom_blob, bottom_blob_copy);
                                bottom_blob = bottom_blob_copy;
                            }
                        }

                        VkMat bottom_blob_unpacked_fp16;
                        if (opt.use_packing_layout && layer->support_packing)
                        {
//                             bottom_blob_unpacked_fp16 = bottom_blob;
                            packing_pack4->forward(bottom_blob, bottom_blob_unpacked_fp16, cmd, opt);
                        }
                        else
                        {
                            // unpacking
                            packing_pack1->forward(bottom_blob, bottom_blob_unpacked_fp16, cmd, opt);
                        }

                        // cast to fp32 (integrated gpu)
                        VkMat& bottom_blob_unpacked = bottom_blobs_unpacked[i];
                        if (opt.use_fp16_storage && vkdev->info.type != 0)
                        {
                            cast_float16_to_float32->forward(bottom_blob_unpacked_fp16, bottom_blob_unpacked, cmd, opt);
                        }
                        else
                        {
                            bottom_blob_unpacked = bottom_blob_unpacked_fp16;
                        }

                        // download
                        bottom_blob_unpacked.prepare_staging_buffer();
                        cmd.record_download(bottom_blob_unpacked);
                    }
                }
            }

            {
                cmd.submit_and_wait();

#if NCNN_BENCHMARK
                std::vector<uint64_t> results(layer_index * 2);
                cmd.get_query_pool_results(0, layer_index * 2, results);
                for (int i=0; i<layer_index; i++)
                {
                    uint64_t start = results[i*2];
                    uint64_t end = results[i*2+1];
                    if (start == 0 || end == 0)
                        continue;

                    double duration_us = (end - start) * vkdev->info.timestamp_period / 1000;
                    fprintf(stderr, "%-24s %-30s %8.2lfus    |\n", layers[i]->type.c_str(), layers[i]->name.c_str(), duration_us);
                }
#endif // NCNN_BENCHMARK

                cmd.reset();
            }

            std::vector<Mat> bottom_blobs(layer->bottoms.size());
            for (size_t i=0; i<layer->bottoms.size(); i++)
            {
                int bottom_blob_index = layer->bottoms[i];

                if (blob_mats[bottom_blob_index].dims == 0)
                {
                    VkMat& bottom_blob_unpacked = bottom_blobs_unpacked[i];

                    Mat bottom_blob_cpu_fp16;
                    bottom_blob_cpu_fp16.create_like(bottom_blob_unpacked, opt.blob_allocator);
                    bottom_blob_unpacked.download(bottom_blob_cpu_fp16);

                    bottom_blob_unpacked.discard_staging_buffer();

                    // cast to fp32 (discrete gpu)
                    Mat& bottom_blob_cpu = blob_mats[bottom_blob_index];
                    if (opt.use_fp16_storage && vkdev->info.type == 0)
                    {
                        ncnn::cast_float16_to_float32(bottom_blob_cpu_fp16, bottom_blob_cpu, opt);
                    }
                    else if (bottom_blob_cpu_fp16.elempack == 4 && opt.use_fp16_packed && !opt.use_fp16_storage && vkdev->info.type == 0)
                    {
                        ncnn::cast_float16_to_float32(bottom_blob_cpu_fp16, bottom_blob_cpu, opt);
                    }
                    else
                    {
                        bottom_blob_cpu = bottom_blob_cpu_fp16;
                    }
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

                if (opt.use_packing_layout)
                {
                    int elempack = layer->support_packing ? 4 : 1;

                    Mat bottom_blob_packed;
                    convert_packing(bottom_blobs[i], bottom_blob_packed, elempack, opt);
                    bottom_blobs[i] = bottom_blob_packed;
                }
            }

            bottom_blobs_unpacked.clear();

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
                std::vector<Mat> top_blobs(layer->tops.size());
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
    }

//     fprintf(stderr, "forward_layer %d %d %s done\n", layer->support_vulkan, layer_index, layer->name.c_str());

    return 0;
}
#endif // NCNN_VULKAN

Extractor::Extractor(const Net* _net, size_t blob_count) : net(_net)
{
    blob_mats.resize(blob_count);
    opt = net->opt;

#if NCNN_VULKAN
    if (net->opt.use_vulkan_compute)
    {
        local_blob_vkallocator = 0;
        local_staging_vkallocator = 0;

        blob_mats_gpu.resize(blob_count);
    }
#endif // NCNN_VULKAN
}

Extractor::~Extractor()
{
    blob_mats.clear();

#if NCNN_VULKAN
    if (net->opt.use_vulkan_compute)
    {
        blob_mats_gpu.clear();

        if (local_blob_vkallocator)
        {
            net->vkdev->reclaim_blob_allocator(local_blob_vkallocator);
        }
        if (local_staging_vkallocator)
        {
            net->vkdev->reclaim_staging_allocator(local_staging_vkallocator);
        }
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
    if (net->opt.use_vulkan_compute)
    {
        opt.use_vulkan_compute = enable;
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
        if (opt.use_vulkan_compute)
        {
            // use local allocator
            if (!opt.blob_vkallocator)
            {
                local_blob_vkallocator = net->vkdev->acquire_blob_allocator();
                opt.blob_vkallocator = local_blob_vkallocator;
            }
            if (!opt.workspace_vkallocator)
            {
                opt.workspace_vkallocator = opt.blob_vkallocator;
            }
            if (!opt.staging_vkallocator)
            {
                local_staging_vkallocator = net->vkdev->acquire_staging_allocator();
                opt.staging_vkallocator = local_staging_vkallocator;
            }

            ncnn::VkCompute cmd(net->vkdev);
#if NCNN_BENCHMARK
            cmd.create_query_pool(net->layers.size() * 2);
#endif // NCNN_BENCHMARK

            VkMat feat_gpu;
            ret = extract(blob_index, feat_gpu, cmd);

            if (blob_mats[blob_index].dims == 0 && feat_gpu.dims != 0)
            {
                // unpacking
                VkMat feat_gpu_unpacked_fp16;
                net->packing_pack1->forward(feat_gpu, feat_gpu_unpacked_fp16, cmd, opt);

                // cast to fp32 (integrated gpu)
                VkMat feat_gpu_unpacked;
                if (opt.use_fp16_storage && net->vkdev->info.type != 0)
                {
                    net->cast_float16_to_float32->forward(feat_gpu_unpacked_fp16, feat_gpu_unpacked, cmd, opt);
                }
                else
                {
                    feat_gpu_unpacked = feat_gpu_unpacked_fp16;
                }

                // download
                feat_gpu_unpacked.prepare_staging_buffer();
                cmd.record_download(feat_gpu_unpacked);

                cmd.submit_and_wait();

#if NCNN_BENCHMARK
                std::vector<uint64_t> results(net->layers.size() * 2);
                cmd.get_query_pool_results(0, net->layers.size() * 2, results);
                for (int i=0; i<net->layers.size(); i++)
                {
                    uint64_t start = results[i*2];
                    uint64_t end = results[i*2+1];
                    if (start == 0 || end == 0)
                        continue;

                    double duration_us = (end - start) * net->vkdev->info.timestamp_period / 1000;
                    fprintf(stderr, "%-24s %-30s %8.2lfus    |\n", net->layers[i]->type.c_str(), net->layers[i]->name.c_str(), duration_us);
                }
#endif // NCNN_BENCHMARK

                Mat feat_cpu_fp16;
                feat_cpu_fp16.create_like(feat_gpu_unpacked, opt.blob_allocator);
                feat_gpu_unpacked.download(feat_cpu_fp16);

                feat_gpu_unpacked.discard_staging_buffer();

                // cast to fp32 (discrete gpu)
                Mat& feat_cpu = blob_mats[blob_index];
                if (opt.use_fp16_storage && net->vkdev->info.type == 0)
                {
                    ncnn::cast_float16_to_float32(feat_cpu_fp16, feat_cpu, opt);
                }
                else if (feat_cpu_fp16.elempack == 4 && opt.use_fp16_packed && !opt.use_fp16_storage && net->vkdev->info.type == 0)
                {
                    ncnn::cast_float16_to_float32(feat_cpu_fp16, feat_cpu, opt);
                }
                else
                {
                    feat_cpu = feat_cpu_fp16;
                }
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

    if (opt.use_packing_layout)
    {
        Mat bottom_blob_unpacked;
        convert_packing(feat, bottom_blob_unpacked, 1, opt);
        feat = bottom_blob_unpacked;
    }

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
        ret = net->forward_layer(layer_index, blob_mats, blob_mats_gpu, cmd, opt);
    }

    feat = blob_mats_gpu[blob_index];

    return ret;
}
#endif // NCNN_VULKAN

} // namespace ncnn
