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

#include "cpu.h"
#include "datareader.h"
#include "layer_type.h"
#include "modelbin.h"
#include "paramdict.h"

#include <stdarg.h>
#include <stdint.h>
#include <string.h>

#if NCNN_BENCHMARK
#include "benchmark.h"
#endif // NCNN_BENCHMARK

#if NCNN_VULKAN
#include "command.h"
#include "pipelinecache.h"
#endif // NCNN_VULKAN

namespace ncnn {

class NetPrivate
{
public:
    NetPrivate(Option& _opt);

    Option& opt;

#if NCNN_VULKAN

    int upload_model();

#endif // NCNN_VULKAN

    friend class Extractor;
    int forward_layer(int layer_index, std::vector<Mat>& blob_mats, const Option& opt) const;

#if NCNN_VULKAN
    int forward_layer(int layer_index, std::vector<Mat>& blob_mats, std::vector<VkMat>& blob_mats_gpu, VkCompute& cmd, const Option& opt) const;
    int forward_layer(int layer_index, std::vector<Mat>& blob_mats, std::vector<VkMat>& blob_mats_gpu, std::vector<VkImageMat>& blob_mats_gpu_image, VkCompute& cmd, const Option& opt) const;
#endif // NCNN_VULKAN

    int convert_layout(Mat& bottom_blob, const Layer* layer, const Option& opt) const;

    int do_forward_layer(const Layer* layer, std::vector<Mat>& blob_mats, const Option& opt) const;
#if NCNN_VULKAN
    int do_forward_layer(const Layer* layer, std::vector<VkMat>& blob_mats_gpu, VkCompute& cmd, const Option& opt) const;
    int do_forward_layer(const Layer* layer, std::vector<VkImageMat>& blob_mats_gpu_image, VkCompute& cmd, const Option& opt) const;
#endif // NCNN_VULKAN

    void update_input_output_indexes();
#if NCNN_STRING
    void update_input_output_names();
#endif // NCNN_STRING

    std::vector<Blob> blobs;
    std::vector<Layer*> layers;

    std::vector<int> input_blob_indexes;
    std::vector<int> output_blob_indexes;
#if NCNN_STRING
    std::vector<const char*> input_blob_names;
    std::vector<const char*> output_blob_names;
#endif // NCNN_STRING

    std::vector<custom_layer_registry_entry> custom_layer_registry;
    std::vector<overwrite_builtin_layer_registry_entry> overwrite_builtin_layer_registry;

    PoolAllocator* local_blob_allocator;
    PoolAllocator* local_workspace_allocator;

#if NCNN_VULKAN
    const VulkanDevice* vkdev;

    VkAllocator* weight_vkallocator;
    VkAllocator* weight_staging_vkallocator;

    PipelineCache* pipeline_cache;
#endif // NCNN_VULKAN
};

NetPrivate::NetPrivate(Option& _opt)
    : opt(_opt)
{
    local_blob_allocator = 0;
    local_workspace_allocator = 0;

#if NCNN_VULKAN
    vkdev = 0;
    weight_vkallocator = 0;
    weight_staging_vkallocator = 0;
    pipeline_cache = 0;
#endif // NCNN_VULKAN
}

static Option get_masked_option(const Option& opt, int featmask)
{
    // mask option usage as layer specific featmask
    Option opt1 = opt;
    opt1.use_fp16_arithmetic = opt1.use_fp16_arithmetic && !(featmask & (1 << 0));
    opt1.use_fp16_storage = opt1.use_fp16_storage && !(featmask & (1 << 1));
    opt1.use_fp16_packed = opt1.use_fp16_packed && !(featmask & (1 << 1));
    opt1.use_bf16_storage = opt1.use_bf16_storage && !(featmask & (1 << 2));
    opt1.use_int8_packed = opt1.use_int8_packed && !(featmask & (1 << 3));
    opt1.use_int8_storage = opt1.use_int8_storage && !(featmask & (1 << 3));
    opt1.use_int8_arithmetic = opt1.use_int8_arithmetic && !(featmask & (1 << 3));
    opt1.use_vulkan_compute = opt1.use_vulkan_compute && !(featmask & (1 << 4));
    opt1.use_image_storage = opt1.use_image_storage && !(featmask & (1 << 4));
    opt1.use_tensor_storage = opt1.use_tensor_storage && !(featmask & (1 << 4));
    opt1.use_sgemm_convolution = opt1.use_sgemm_convolution && !(featmask & (1 << 5));
    opt1.use_winograd_convolution = opt1.use_winograd_convolution && !(featmask & (1 << 6));

    return opt1;
}

#if NCNN_VULKAN
int NetPrivate::upload_model()
{
    ncnn::VkTransfer cmd(vkdev);

    // create gpu device allocator if null
    if (!weight_vkallocator)
    {
        weight_vkallocator = new VkWeightAllocator(vkdev);
    }
    if (!weight_staging_vkallocator)
    {
        weight_staging_vkallocator = new VkWeightStagingAllocator(vkdev);
    }

    Option opt_upload = opt;
    opt_upload.blob_vkallocator = weight_vkallocator;
    opt_upload.workspace_vkallocator = weight_vkallocator;
    opt_upload.staging_vkallocator = weight_staging_vkallocator;

    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i]->support_vulkan)
        {
            int uret = layers[i]->upload_model(cmd, get_masked_option(opt_upload, layers[i]->featmask));
            if (uret != 0)
            {
                NCNN_LOGE("layer upload_model %d failed", (int)i);
                return -1;
            }
        }
    }

    return cmd.submit_and_wait();
}
#endif // NCNN_VULKAN

int NetPrivate::forward_layer(int layer_index, std::vector<Mat>& blob_mats, const Option& opt) const
{
    const Layer* layer = layers[layer_index];

    //     NCNN_LOGE("forward_layer %d %s", layer_index, layer->name.c_str());

    // load bottom blobs
    for (size_t i = 0; i < layer->bottoms.size(); i++)
    {
        int bottom_blob_index = layer->bottoms[i];

        if (blob_mats[bottom_blob_index].dims == 0)
        {
            int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, opt);
            if (ret != 0)
                return ret;
        }
    }

#if NCNN_BENCHMARK
    double start = get_current_time();
    Mat bottom_blob;
    if (layer->one_blob_only)
    {
        int bottom_blob_index = layer->bottoms[0];
        bottom_blob.dims = blob_mats[bottom_blob_index].dims;
        bottom_blob.w = blob_mats[bottom_blob_index].w;
        bottom_blob.h = blob_mats[bottom_blob_index].h;
        bottom_blob.c = blob_mats[bottom_blob_index].c;
        bottom_blob.elempack = blob_mats[bottom_blob_index].elempack;
        bottom_blob.elemsize = blob_mats[bottom_blob_index].elemsize;
    }
#endif
    int ret = 0;
    if (layer->featmask)
    {
        ret = do_forward_layer(layer, blob_mats, get_masked_option(opt, layer->featmask));
    }
    else
    {
        ret = do_forward_layer(layer, blob_mats, opt);
    }
#if NCNN_BENCHMARK
    double end = get_current_time();
    if (layer->one_blob_only)
    {
        int top_blob_index = layer->tops[0];
        benchmark(layer, bottom_blob, blob_mats[top_blob_index], start, end);
    }
    else
    {
        benchmark(layer, start, end);
    }
#endif
    if (ret != 0)
        return ret;

    //     NCNN_LOGE("forward_layer %d %s done", layer_index, layer->name.c_str());
    //     const Mat& blob = blob_mats[layer->tops[0]];
    //     NCNN_LOGE("[%-2d %-16s %-16s]  %d    blobs count = %-3d   size = %-3d x %-3d", layer_index, layer->type.c_str(), layer->name.c_str(), layer->tops[0], blob.c, blob.h, blob.w);

    return 0;
}

#if NCNN_VULKAN
int NetPrivate::forward_layer(int layer_index, std::vector<Mat>& blob_mats, std::vector<VkMat>& blob_mats_gpu, VkCompute& cmd, const Option& opt) const
{
    const Layer* layer = layers[layer_index];

    //     NCNN_LOGE("forward_layer %d %d %s", layer->support_vulkan, layer_index, layer->name.c_str());

    bool cmd_submit_and_wait = false;

    // load bottom blobs
    for (size_t i = 0; i < layer->bottoms.size(); i++)
    {
        int bottom_blob_index = layer->bottoms[i];

        if (blob_mats_gpu[bottom_blob_index].dims == 0 && blob_mats[bottom_blob_index].dims == 0)
        {
            int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, blob_mats_gpu, cmd, opt);
            if (ret != 0)
                return ret;
        }

        if (layer->support_vulkan)
        {
            if (blob_mats_gpu[bottom_blob_index].dims == 0)
            {
                // host to buffer
                cmd.record_upload(blob_mats[bottom_blob_index], blob_mats_gpu[bottom_blob_index], opt);

                if (opt.lightmode)
                {
                    // delete after taken in light mode
                    blob_mats[bottom_blob_index].release();
                }
            }
        }
        else
        {
            if (blob_mats[bottom_blob_index].dims == 0)
            {
                Option opt_download = opt;
                opt_download.use_packing_layout = layer->support_packing;

                // buffer to host
                cmd.record_download(blob_mats_gpu[bottom_blob_index], blob_mats[bottom_blob_index], opt_download);

                if (opt.lightmode)
                {
                    // delete after taken in light mode
                    blob_mats_gpu[bottom_blob_index].release();
                }

                cmd_submit_and_wait = true;
            }
        }
    }

    int ret;
    if (cmd_submit_and_wait)
    {
        ret = cmd.submit_and_wait();

#if NCNN_BENCHMARK
        std::vector<uint64_t> results(layer_index * 2);
        cmd.get_query_pool_results(0, layer_index * 2, results);
        for (int i = 0; i < layer_index; i++)
        {
            uint64_t start = results[i * 2];
            uint64_t end = results[i * 2 + 1];
            if (start == 0 || end == 0)
                continue;

            double duration_us = (end - start) * vkdev->info.timestamp_period() / 1000;
            NCNN_LOGE("%-24s %-30s %8.2lfus    |", layers[i]->type.c_str(), layers[i]->name.c_str(), duration_us);
        }
#endif // NCNN_BENCHMARK

        cmd.reset();
        if (ret != 0)
            return ret;
    }

    if (layer->support_vulkan)
    {
#if NCNN_BENCHMARK
        cmd.record_write_timestamp(layer_index * 2);
#endif
        if (layer->featmask)
        {
            ret = do_forward_layer(layer, blob_mats_gpu, cmd, get_masked_option(opt, layer->featmask));
        }
        else
        {
            ret = do_forward_layer(layer, blob_mats_gpu, cmd, opt);
        }
#if NCNN_BENCHMARK
        cmd.record_write_timestamp(layer_index * 2 + 1);
#endif
    }
    else
    {
#if NCNN_BENCHMARK
        double start = get_current_time();
        Mat bottom_blob;
        if (layer->one_blob_only)
        {
            int bottom_blob_index = layer->bottoms[0];
            bottom_blob = blob_mats[bottom_blob_index].shape();
        }
#endif
        if (layer->featmask)
        {
            ret = do_forward_layer(layer, blob_mats, get_masked_option(opt, layer->featmask));
        }
        else
        {
            ret = do_forward_layer(layer, blob_mats, opt);
        }
#if NCNN_BENCHMARK
        double end = get_current_time();
        if (layer->one_blob_only)
        {
            int top_blob_index = layer->tops[0];
            benchmark(layer, bottom_blob, blob_mats[top_blob_index], start, end);
        }
        else
        {
            benchmark(layer, start, end);
        }
#endif
    }
    if (ret != 0)
        return ret;

    //     NCNN_LOGE("forward_layer %d %d %s done", layer->support_vulkan, layer_index, layer->name.c_str());

    return 0;
}

int NetPrivate::forward_layer(int layer_index, std::vector<Mat>& blob_mats, std::vector<VkMat>& blob_mats_gpu, std::vector<VkImageMat>& blob_mats_gpu_image, VkCompute& cmd, const Option& opt) const
{
    const Layer* layer = layers[layer_index];

    //     NCNN_LOGE("forward_layer %d %d %s", layer->support_vulkan, layer_index, layer->name.c_str());

    bool cmd_submit_and_wait = false;
    bool image_allocation_failed = false;

IMAGE_ALLOCATION_FAILED:

    if (image_allocation_failed)
    {
#if NCNN_STRING
        NCNN_LOGE("forward_layer %d %s image allocation failed, fallback to cpu", layer_index, layer->name.c_str());
#else
        NCNN_LOGE("forward_layer %d image allocation failed, fallback to cpu", layer_index);
#endif
    }

    // load bottom blobs
    for (size_t i = 0; i < layer->bottoms.size(); i++)
    {
        int bottom_blob_index = layer->bottoms[i];

        if (blob_mats_gpu_image[bottom_blob_index].dims == 0 && blob_mats_gpu[bottom_blob_index].dims == 0 && blob_mats[bottom_blob_index].dims == 0)
        {
            int ret = forward_layer(blobs[bottom_blob_index].producer, blob_mats, blob_mats_gpu, blob_mats_gpu_image, cmd, opt);
            if (ret != 0)
                return ret;
        }

        if (layer->support_vulkan && !image_allocation_failed)
        {
            if (layer->support_image_storage)
            {
                if (blob_mats_gpu_image[bottom_blob_index].dims == 0)
                {
                    if (blob_mats_gpu[bottom_blob_index].dims == 0)
                    {
                        // host to image
                        cmd.record_upload(blob_mats[bottom_blob_index], blob_mats_gpu_image[bottom_blob_index], opt);

                        if (blob_mats_gpu_image[bottom_blob_index].empty())
                        {
                            image_allocation_failed = true;
                            goto IMAGE_ALLOCATION_FAILED;
                        }

                        if (opt.lightmode)
                        {
                            // delete after taken in light mode
                            blob_mats[bottom_blob_index].release();
                        }
                    }
                    else
                    {
                        // buffer to image
                        cmd.record_buffer_to_image(blob_mats_gpu[bottom_blob_index], blob_mats_gpu_image[bottom_blob_index], opt);

                        if (blob_mats_gpu_image[bottom_blob_index].empty())
                        {
                            image_allocation_failed = true;
                            goto IMAGE_ALLOCATION_FAILED;
                        }

                        if (opt.lightmode)
                        {
                            // delete after taken in light mode
                            blob_mats_gpu[bottom_blob_index].release();
                        }
                    }
                }
            }
            else
            {
                if (blob_mats_gpu[bottom_blob_index].dims == 0)
                {
                    if (blob_mats_gpu_image[bottom_blob_index].dims == 0)
                    {
                        // host to buffer
                        cmd.record_upload(blob_mats[bottom_blob_index], blob_mats_gpu[bottom_blob_index], opt);

                        if (opt.lightmode)
                        {
                            // delete after taken in light mode
                            blob_mats[bottom_blob_index].release();
                        }
                    }
                    else
                    {
                        // image to buffer
                        cmd.record_image_to_buffer(blob_mats_gpu_image[bottom_blob_index], blob_mats_gpu[bottom_blob_index], opt);

                        if (opt.lightmode)
                        {
                            // delete after taken in light mode
                            blob_mats_gpu_image[bottom_blob_index].release();
                        }
                    }
                }
            }
        }
        else
        {
            if (blob_mats[bottom_blob_index].dims == 0)
            {
                if (blob_mats_gpu_image[bottom_blob_index].dims == 0)
                {
                    // buffer to host
                    cmd.record_download(blob_mats_gpu[bottom_blob_index], blob_mats[bottom_blob_index], opt);

                    if (opt.lightmode)
                    {
                        // delete after taken in light mode
                        blob_mats_gpu[bottom_blob_index].release();
                    }

                    cmd_submit_and_wait = true;
                }
                else
                {
                    // image to host
                    cmd.record_download(blob_mats_gpu_image[bottom_blob_index], blob_mats[bottom_blob_index], opt);

                    if (opt.lightmode)
                    {
                        // delete after taken in light mode
                        blob_mats_gpu_image[bottom_blob_index].release();
                    }

                    cmd_submit_and_wait = true;
                }
            }
        }
    }

    int ret;
    if (cmd_submit_and_wait)
    {
        ret = cmd.submit_and_wait();

#if NCNN_BENCHMARK
        std::vector<uint64_t> results(layer_index * 2);
        cmd.get_query_pool_results(0, layer_index * 2, results);
        for (int i = 0; i < layer_index; i++)
        {
            uint64_t start = results[i * 2];
            uint64_t end = results[i * 2 + 1];
            if (start == 0 || end == 0)
                continue;

            double duration_us = (end - start) * vkdev->info.timestamp_period() / 1000;
            NCNN_LOGE("%-24s %-30s %8.2lfus    |", layers[i]->type.c_str(), layers[i]->name.c_str(), duration_us);
        }
#endif // NCNN_BENCHMARK

        cmd.reset();

        if (ret != 0)
            return ret;
    }

    if (layer->support_vulkan && !image_allocation_failed)
    {
#if NCNN_BENCHMARK
        cmd.record_write_timestamp(layer_index * 2);
#endif
        if (layer->support_image_storage)
        {
            if (layer->featmask)
            {
                ret = do_forward_layer(layer, blob_mats_gpu_image, cmd, get_masked_option(opt, layer->featmask));
            }
            else
            {
                ret = do_forward_layer(layer, blob_mats_gpu_image, cmd, opt);
            }
            if (ret == -100)
            {
                image_allocation_failed = true;
                goto IMAGE_ALLOCATION_FAILED;
            }
        }
        else
        {
            if (layer->featmask)
            {
                ret = do_forward_layer(layer, blob_mats_gpu, cmd, get_masked_option(opt, layer->featmask));
            }
            else
            {
                ret = do_forward_layer(layer, blob_mats_gpu, cmd, opt);
            }
        }
#if NCNN_BENCHMARK
        cmd.record_write_timestamp(layer_index * 2 + 1);
#endif
    }
    else
    {
#if NCNN_BENCHMARK
        double start = get_current_time();
        Mat bottom_blob;
        if (layer->one_blob_only)
        {
            int bottom_blob_index = layer->bottoms[0];
            bottom_blob = blob_mats[bottom_blob_index].shape();
        }
#endif
        if (layer->featmask)
        {
            ret = do_forward_layer(layer, blob_mats, get_masked_option(opt, layer->featmask));
        }
        else
        {
            ret = do_forward_layer(layer, blob_mats, opt);
        }
#if NCNN_BENCHMARK
        double end = get_current_time();
        if (layer->one_blob_only)
        {
            int top_blob_index = layer->tops[0];
            benchmark(layer, bottom_blob, blob_mats[top_blob_index], start, end);
        }
        else
        {
            benchmark(layer, start, end);
        }
#endif
    }
    if (ret != 0)
        return ret;

    //     NCNN_LOGE("forward_layer %d %d %s done", layer->support_vulkan, layer_index, layer->name.c_str());

    return 0;
}
#endif // NCNN_VULKAN

int NetPrivate::convert_layout(Mat& bottom_blob, const Layer* layer, const Option& opt) const
{
    // clang-format off
    // *INDENT-OFF*
#if NCNN_ARM82
    if (opt.use_fp16_storage && cpu_support_arm_asimdhp())
    {
        if (bottom_blob.elembits() == 32 && layer->support_fp16_storage)
        {
            Mat bottom_blob_fp16;
            cast_float32_to_float16(bottom_blob, bottom_blob_fp16, opt);
            bottom_blob = bottom_blob_fp16;
        }
        if (bottom_blob.elembits() == 16 && !layer->support_fp16_storage)
        {
            Mat bottom_blob_fp32;
            cast_float16_to_float32(bottom_blob, bottom_blob_fp32, opt);
            bottom_blob = bottom_blob_fp32;
        }
    }
    else
#endif // NCNN_ARM82
#if NCNN_RVV
    if (opt.use_fp16_storage && cpu_support_riscv_v() && cpu_support_riscv_zfh())
    {
        if (bottom_blob.elembits() == 32 && layer->support_fp16_storage)
        {
            Mat bottom_blob_fp16;
            cast_float32_to_float16(bottom_blob, bottom_blob_fp16, opt);
            bottom_blob = bottom_blob_fp16;
        }
        if (bottom_blob.elembits() == 16 && !layer->support_fp16_storage)
        {
            Mat bottom_blob_fp32;
            cast_float16_to_float32(bottom_blob, bottom_blob_fp32, opt);
            bottom_blob = bottom_blob_fp32;
        }
    }
    else
#endif // NCNN_RVV
#if NCNN_BF16
    if (opt.use_bf16_storage)
    {
        if (bottom_blob.elembits() == 32 && layer->support_bf16_storage)
        {
            Mat bottom_blob_bf16;
            cast_float32_to_bfloat16(bottom_blob, bottom_blob_bf16, opt);
            bottom_blob = bottom_blob_bf16;
        }
        if (bottom_blob.elembits() == 16 && !layer->support_bf16_storage)
        {
            Mat bottom_blob_fp32;
            cast_bfloat16_to_float32(bottom_blob, bottom_blob_fp32, opt);
            bottom_blob = bottom_blob_fp32;
        }
    }
    else
#endif // NCNN_BF16
    {
        // no type conversion
    }
    // *INDENT-ON*
    // clang-format on

    int dst_elempack = 1;
    if (opt.use_packing_layout)
    {
        // resolve dst_elempack
        int dims = bottom_blob.dims;
        int elemcount = 0;
        if (dims == 1) elemcount = bottom_blob.elempack * bottom_blob.w;
        if (dims == 2) elemcount = bottom_blob.elempack * bottom_blob.h;
        if (dims == 3 || dims == 4) elemcount = bottom_blob.elempack * bottom_blob.c;

        int elembits = bottom_blob.elembits();

        if (layer->support_packing)
        {
            if (elembits == 32)
            {
#if NCNN_AVX512
                if (elemcount % 16 == 0 && ncnn::cpu_support_x86_avx512())
                    dst_elempack = 16;
                else if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx())
                    dst_elempack = 8;
                else if (elemcount % 4 == 0)
                    dst_elempack = 4;
#elif NCNN_AVX
                if (elemcount % 8 == 0 && ncnn::cpu_support_x86_avx())
                    dst_elempack = 8;
                else if (elemcount % 4 == 0)
                    dst_elempack = 4;
#elif NCNN_RVV
                const int packn = ncnn::cpu_riscv_vlenb() / 4;
                if (elemcount % packn == 0)
                    dst_elempack = packn;
#else
                if (elemcount % 4 == 0)
                    dst_elempack = 4;
#endif
            }
            if (elembits == 16)
            {
#if NCNN_ARM82
                if (elemcount % 8 == 0 && ncnn::cpu_support_arm_asimdhp() && opt.use_fp16_arithmetic)
                    dst_elempack = 8;
                else if (elemcount % 4 == 0)
                    dst_elempack = 4;
#elif NCNN_RVV
                const int packn = ncnn::cpu_riscv_vlenb() / 2;
                if (elemcount % packn == 0)
                    dst_elempack = packn;
#else
                if (elemcount % 4 == 0)
                    dst_elempack = 4;
#endif
            }
            if (elembits == 8)
            {
#if NCNN_RVV
                const int packn = ncnn::cpu_riscv_vlenb() / 1;
                if (elemcount % packn == 0)
                    dst_elempack = packn;
#else
                if (elemcount % 8 == 0)
                    dst_elempack = 8;
#endif
            }
        }
    }

    if (bottom_blob.elempack != dst_elempack)
    {
        Mat bottom_blob_packed;
        convert_packing(bottom_blob, bottom_blob_packed, dst_elempack, opt);
        bottom_blob = bottom_blob_packed;
    }

    return 0;
}

int NetPrivate::do_forward_layer(const Layer* layer, std::vector<Mat>& blob_mats, const Option& opt) const
{
    if (layer->one_blob_only)
    {
        int bottom_blob_index = layer->bottoms[0];
        int top_blob_index = layer->tops[0];

        Mat& bottom_blob_ref = blob_mats[bottom_blob_index];
        Mat bottom_blob;

        if (opt.lightmode)
        {
            // deep copy for inplace forward if data is shared
            if (layer->support_inplace && *bottom_blob_ref.refcount != 1)
            {
                bottom_blob = bottom_blob_ref.clone(opt.blob_allocator);
            }
        }
        if (bottom_blob.dims == 0)
        {
            bottom_blob = bottom_blob_ref;
        }

        convert_layout(bottom_blob, layer, opt);

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

        if (opt.lightmode)
        {
            // delete after taken in light mode
            blob_mats[bottom_blob_index].release();
        }
    }
    else
    {
        std::vector<Mat> bottom_blobs(layer->bottoms.size());
        for (size_t i = 0; i < layer->bottoms.size(); i++)
        {
            int bottom_blob_index = layer->bottoms[i];

            Mat& bottom_blob_ref = blob_mats[bottom_blob_index];
            bottom_blobs[i].release();

            if (opt.lightmode)
            {
                // deep copy for inplace forward if data is shared
                if (layer->support_inplace && *bottom_blob_ref.refcount != 1)
                {
                    bottom_blobs[i] = bottom_blob_ref.clone(opt.blob_allocator);
                }
            }
            if (bottom_blobs[i].dims == 0)
            {
                bottom_blobs[i] = bottom_blob_ref;
            }

            convert_layout(bottom_blobs[i], layer, opt);
        }

        // forward
        if (opt.lightmode && layer->support_inplace)
        {
            std::vector<Mat>& bottom_top_blobs = bottom_blobs;
            int ret = layer->forward_inplace(bottom_top_blobs, opt);
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i = 0; i < layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats[top_blob_index] = bottom_top_blobs[i];
            }
        }
        else
        {
            std::vector<Mat> top_blobs(layer->tops.size());
            int ret = layer->forward(bottom_blobs, top_blobs, opt);
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i = 0; i < layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats[top_blob_index] = top_blobs[i];
            }
        }

        if (opt.lightmode)
        {
            for (size_t i = 0; i < layer->bottoms.size(); i++)
            {
                int bottom_blob_index = layer->bottoms[i];

                // delete after taken in light mode
                blob_mats[bottom_blob_index].release();
            }
        }
    }

    return 0;
}

#if NCNN_VULKAN
int NetPrivate::do_forward_layer(const Layer* layer, std::vector<VkMat>& blob_mats_gpu, VkCompute& cmd, const Option& opt) const
{
    if (layer->one_blob_only)
    {
        // load bottom blob
        int bottom_blob_index = layer->bottoms[0];
        int top_blob_index = layer->tops[0];

        VkMat& bottom_blob_ref = blob_mats_gpu[bottom_blob_index];
        VkMat bottom_blob;

        if (opt.lightmode)
        {
            // deep copy for inplace forward if data is shared
            if (layer->support_inplace && *bottom_blob_ref.refcount != 1)
            {
                cmd.record_clone(bottom_blob_ref, bottom_blob, opt);
                //                     NCNN_LOGE("clone %p[+%lu] %p[+%lu]", bottom_blob_ref.buffer(), bottom_blob_ref.buffer_offset(), bottom_blob.buffer(), bottom_blob.buffer_offset());
            }
        }
        if (bottom_blob.dims == 0)
        {
            bottom_blob = bottom_blob_ref;
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
            // delete after taken in light mode
            blob_mats_gpu[bottom_blob_index].release();
        }
    }
    else
    {
        // load bottom blobs
        std::vector<VkMat> bottom_blobs(layer->bottoms.size());
        for (size_t i = 0; i < layer->bottoms.size(); i++)
        {
            int bottom_blob_index = layer->bottoms[i];

            VkMat& bottom_blob_ref = blob_mats_gpu[bottom_blob_index];
            bottom_blobs[i].release();

            if (opt.lightmode)
            {
                // deep copy for inplace forward if data is shared
                if (layer->support_inplace && *bottom_blob_ref.refcount != 1)
                {
                    cmd.record_clone(bottom_blob_ref, bottom_blobs[i], opt);
                    //                         NCNN_LOGE("clone %p[+%lu] %p[+%lu]", bottom_blob_ref.buffer(), bottom_blob_ref.buffer_offset(), bottom_blobs[i].buffer(), bottom_blobs[i].buffer_offset());
                }
            }
            if (bottom_blobs[i].dims == 0)
            {
                bottom_blobs[i] = bottom_blob_ref;
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
            for (size_t i = 0; i < layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats_gpu[top_blob_index] = bottom_top_blobs[i];
            }
        }
        else
        {
            std::vector<VkMat> top_blobs(layer->tops.size());
            int ret = layer->forward(bottom_blobs, top_blobs, cmd, opt);
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i = 0; i < layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats_gpu[top_blob_index] = top_blobs[i];
            }
        }

        if (opt.lightmode)
        {
            for (size_t i = 0; i < layer->bottoms.size(); i++)
            {
                int bottom_blob_index = layer->bottoms[i];

                // delete after taken in light mode
                blob_mats_gpu[bottom_blob_index].release();
            }
        }
    }

    return 0;
}

int NetPrivate::do_forward_layer(const Layer* layer, std::vector<VkImageMat>& blob_mats_gpu_image, VkCompute& cmd, const Option& opt) const
{
    if (layer->one_blob_only)
    {
        // load bottom blob
        int bottom_blob_index = layer->bottoms[0];
        int top_blob_index = layer->tops[0];

        VkImageMat& bottom_blob_ref = blob_mats_gpu_image[bottom_blob_index];
        VkImageMat bottom_blob;

        if (opt.lightmode)
        {
            // deep copy for inplace forward if data is shared
            if (layer->support_inplace && *bottom_blob_ref.refcount != 1)
            {
                cmd.record_clone(bottom_blob_ref, bottom_blob, opt);
                //                         NCNN_LOGE("clone %p[+%lu] %p[+%lu]", bottom_blob_ref.buffer(), bottom_blob_ref.buffer_offset(), bottom_blob.buffer(), bottom_blob.buffer_offset());
            }
        }
        if (bottom_blob.dims == 0)
        {
            bottom_blob = bottom_blob_ref;
        }

        // forward
        if (opt.lightmode && layer->support_inplace)
        {
            VkImageMat& bottom_top_blob = bottom_blob;
            int ret = layer->forward_inplace(bottom_top_blob, cmd, opt);
            if (ret != 0)
                return ret;

            // store top blob
            blob_mats_gpu_image[top_blob_index] = bottom_top_blob;
        }
        else
        {
            VkImageMat top_blob;
            int ret = layer->forward(bottom_blob, top_blob, cmd, opt);
            if (ret != 0)
                return ret;

            // store top blob
            blob_mats_gpu_image[top_blob_index] = top_blob;
        }

        if (opt.lightmode)
        {
            // delete after taken in light mode
            blob_mats_gpu_image[bottom_blob_index].release();
        }
    }
    else
    {
        // load bottom blobs
        std::vector<VkImageMat> bottom_blobs(layer->bottoms.size());
        for (size_t i = 0; i < layer->bottoms.size(); i++)
        {
            int bottom_blob_index = layer->bottoms[i];

            VkImageMat& bottom_blob_ref = blob_mats_gpu_image[bottom_blob_index];

            if (opt.lightmode)
            {
                // deep copy for inplace forward if data is shared
                if (layer->support_inplace && *bottom_blob_ref.refcount != 1)
                {
                    cmd.record_clone(bottom_blob_ref, bottom_blobs[i], opt);
                    //                             NCNN_LOGE("clone %p[+%lu] %p[+%lu]", bottom_blob_ref.buffer(), bottom_blob_ref.buffer_offset(), bottom_blobs[i].buffer(), bottom_blobs[i].buffer_offset());
                }
            }
            if (bottom_blobs[i].dims == 0)
            {
                bottom_blobs[i] = bottom_blob_ref;
            }
        }

        // forward
        if (opt.lightmode && layer->support_inplace)
        {
            std::vector<VkImageMat>& bottom_top_blobs = bottom_blobs;
            int ret = layer->forward_inplace(bottom_top_blobs, cmd, opt);
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i = 0; i < layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats_gpu_image[top_blob_index] = bottom_top_blobs[i];
            }
        }
        else
        {
            std::vector<VkImageMat> top_blobs(layer->tops.size());
            int ret = layer->forward(bottom_blobs, top_blobs, cmd, opt);
            if (ret != 0)
                return ret;

            // store top blobs
            for (size_t i = 0; i < layer->tops.size(); i++)
            {
                int top_blob_index = layer->tops[i];

                blob_mats_gpu_image[top_blob_index] = top_blobs[i];
            }
        }

        if (opt.lightmode)
        {
            for (size_t i = 0; i < layer->bottoms.size(); i++)
            {
                int bottom_blob_index = layer->bottoms[i];

                // delete after taken in light mode
                blob_mats_gpu_image[bottom_blob_index].release();
            }
        }
    }

    return 0;
}
#endif // NCNN_VULKAN

void NetPrivate::update_input_output_indexes()
{
    input_blob_indexes.clear();
    output_blob_indexes.clear();

    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i]->typeindex == LayerType::Input)
        {
            int blob_index = layers[i]->tops[0];
            input_blob_indexes.push_back(blob_index);
        }
    }

    for (size_t i = 0; i < blobs.size(); i++)
    {
        if (blobs[i].producer != -1 && blobs[i].consumer == -1)
        {
            output_blob_indexes.push_back(i);
        }
    }
}

#if NCNN_STRING
void NetPrivate::update_input_output_names()
{
    input_blob_names.clear();
    output_blob_names.clear();

    for (size_t i = 0; i < input_blob_indexes.size(); i++)
    {
        int blob_index = input_blob_indexes[i];
        input_blob_names.push_back(blobs[blob_index].name.c_str());
    }

    for (size_t i = 0; i < output_blob_indexes.size(); i++)
    {
        int blob_index = output_blob_indexes[i];
        output_blob_names.push_back(blobs[blob_index].name.c_str());
    }
}
#endif // NCNN_STRING

Net::Net()
    : d(new NetPrivate(opt))
{
}

Net::~Net()
{
    clear();

    delete d;
}

Net::Net(const Net&)
    : d(0)
{
}

Net& Net::operator=(const Net&)
{
    return *this;
}

#if NCNN_STRING
int Net::register_custom_layer(const char* type, layer_creator_func creator, layer_destroyer_func destroyer, void* userdata)
{
    int typeindex = layer_to_index(type);
    if (typeindex != -1)
    {
        NCNN_LOGE("overwrite built-in layer type %s", type);

        for (size_t i = 0; i < d->overwrite_builtin_layer_registry.size(); i++)
        {
            if (d->overwrite_builtin_layer_registry[i].typeindex == typeindex)
            {
                NCNN_LOGE("overwrite existing overwritten built-in layer index %d", typeindex);

                d->overwrite_builtin_layer_registry[i].creator = creator;
                d->overwrite_builtin_layer_registry[i].destroyer = destroyer;
                d->overwrite_builtin_layer_registry[i].userdata = userdata;
                return 0;
            }
        }

        struct overwrite_builtin_layer_registry_entry entry = {typeindex, creator, destroyer, userdata};
        d->overwrite_builtin_layer_registry.push_back(entry);
        return 0;
    }

    int custom_index = custom_layer_to_index(type);
    if (custom_index == -1)
    {
        struct custom_layer_registry_entry entry = {type, creator, destroyer, userdata};
        d->custom_layer_registry.push_back(entry);
    }
    else
    {
        NCNN_LOGE("overwrite existing custom layer type %s", type);
        d->custom_layer_registry[custom_index].name = type;
        d->custom_layer_registry[custom_index].creator = creator;
        d->custom_layer_registry[custom_index].destroyer = destroyer;
        d->custom_layer_registry[custom_index].userdata = userdata;
    }

    return 0;
}
#endif // NCNN_STRING

int Net::register_custom_layer(int index, layer_creator_func creator, layer_destroyer_func destroyer, void* userdata)
{
    int custom_index = index & ~LayerType::CustomBit;
    if (index == custom_index)
    {
        NCNN_LOGE("overwrite built-in layer type %d", index);

        for (size_t i = 0; i < d->overwrite_builtin_layer_registry.size(); i++)
        {
            if (d->overwrite_builtin_layer_registry[i].typeindex == index)
            {
                NCNN_LOGE("overwrite existing overwritten built-in layer index %d", index);

                d->overwrite_builtin_layer_registry[i].creator = creator;
                d->overwrite_builtin_layer_registry[i].destroyer = destroyer;
                d->overwrite_builtin_layer_registry[i].userdata = userdata;
                return 0;
            }
        }

        struct overwrite_builtin_layer_registry_entry entry = {index, creator, destroyer, userdata};
        d->overwrite_builtin_layer_registry.push_back(entry);
        return 0;
    }

    if ((int)d->custom_layer_registry.size() <= custom_index)
    {
#if NCNN_STRING
        struct custom_layer_registry_entry dummy = {"", 0, 0, 0};
#else
        struct custom_layer_registry_entry dummy = {0, 0, 0};
#endif // NCNN_STRING
        d->custom_layer_registry.resize(custom_index + 1, dummy);
    }

    if (d->custom_layer_registry[custom_index].creator)
    {
        NCNN_LOGE("overwrite existing custom layer index %d", custom_index);
    }

    d->custom_layer_registry[custom_index].creator = creator;
    d->custom_layer_registry[custom_index].destroyer = destroyer;
    d->custom_layer_registry[custom_index].userdata = userdata;
    return 0;
}

#if NCNN_STRING
int Net::load_param(const DataReader& dr)
{
#define SCAN_VALUE(fmt, v)                \
    if (dr.scan(fmt, &v) != 1)            \
    {                                     \
        NCNN_LOGE("parse " #v " failed"); \
        return -1;                        \
    }

    int magic = 0;
    SCAN_VALUE("%d", magic)
    if (magic != 7767517)
    {
        NCNN_LOGE("param is too old, please regenerate");
        return -1;
    }

    // parse
    int layer_count = 0;
    int blob_count = 0;
    SCAN_VALUE("%d", layer_count)
    SCAN_VALUE("%d", blob_count)
    if (layer_count <= 0 || blob_count <= 0)
    {
        NCNN_LOGE("invalid layer_count or blob_count");
        return -1;
    }

    d->layers.resize((size_t)layer_count);
    d->blobs.resize((size_t)blob_count);

#if NCNN_VULKAN
    // TODO enable gpu when bf16 conversion implemented
    if (opt.use_bf16_storage)
        opt.use_vulkan_compute = false;

    if (opt.use_vulkan_compute)
    {
        if (!d->vkdev) d->vkdev = get_gpu_device();
        if (!d->vkdev) opt.use_vulkan_compute = false; // no vulkan device, fallback to cpu
    }
    if (opt.use_vulkan_compute)
    {
        // sanitize use options
        if (!d->vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
        if (!d->vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;
        if (!d->vkdev->info.support_fp16_arithmetic()) opt.use_fp16_arithmetic = false;
        if (!d->vkdev->info.support_int8_storage()) opt.use_int8_storage = false;
        if (!d->vkdev->info.support_int8_arithmetic()) opt.use_int8_arithmetic = false;
        if (!d->vkdev->info.support_cooperative_matrix()) opt.use_cooperative_matrix = false;

        if (d->vkdev->info.bug_buffer_image_load_zero()) opt.use_image_storage = false;

        // enable local memory optimization on discrete gpu only
        if (d->vkdev->info.type() != 0) opt.use_shader_local_memory = false;

        // fp16a makes no sense when fp16 storage disabled
        if (!opt.use_fp16_packed && !opt.use_fp16_storage) opt.use_fp16_arithmetic = false;
    }
    else
    {
        // fp16a makes no sense when fp16 storage disabled
        if (!opt.use_fp16_storage) opt.use_fp16_arithmetic = false;
    }
#endif // NCNN_VULKAN

    ParamDict pd;

    int blob_index = 0;
    for (int i = 0; i < layer_count; i++)
    {
        char layer_type[256];
        char layer_name[256];
        int bottom_count = 0;
        int top_count = 0;
        SCAN_VALUE("%255s", layer_type)
        SCAN_VALUE("%255s", layer_name)
        SCAN_VALUE("%d", bottom_count)
        SCAN_VALUE("%d", top_count)

        Layer* layer = create_overwrite_builtin_layer(layer_type);
        if (!layer)
        {
            layer = create_layer(layer_type);
        }
        if (!layer)
        {
            layer = create_custom_layer(layer_type);
        }
        if (!layer)
        {
            NCNN_LOGE("layer %s not exists or registered", layer_type);
            clear();
            return -1;
        }

#if NCNN_VULKAN
        if (opt.use_vulkan_compute)
            layer->vkdev = d->vkdev;
#endif // NCNN_VULKAN

        layer->type = std::string(layer_type);
        layer->name = std::string(layer_name);
        //         NCNN_LOGE("new layer %d %s", i, layer_name);

        layer->bottoms.resize(bottom_count);

        for (int j = 0; j < bottom_count; j++)
        {
            char bottom_name[256];
            SCAN_VALUE("%255s", bottom_name)

            int bottom_blob_index = find_blob_index_by_name(bottom_name);
            if (bottom_blob_index == -1)
            {
                Blob& blob = d->blobs[blob_index];

                bottom_blob_index = blob_index;

                blob.name = std::string(bottom_name);
                //                 NCNN_LOGE("new blob %s", bottom_name);

                blob_index++;
            }

            Blob& blob = d->blobs[bottom_blob_index];

            blob.consumer = i;

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int j = 0; j < top_count; j++)
        {
            Blob& blob = d->blobs[blob_index];

            char blob_name[256];
            SCAN_VALUE("%255s", blob_name)

            blob.name = std::string(blob_name);
            //             NCNN_LOGE("new blob %s", blob_name);

            blob.producer = i;

            layer->tops[j] = blob_index;

            blob_index++;
        }

        // layer specific params
        int pdlr = pd.load_param(dr);
        if (pdlr != 0)
        {
            NCNN_LOGE("ParamDict load_param %d %s failed", i, layer->name.c_str());
            continue;
        }

        if (layer->support_int8_storage)
        {
            // no int8 gpu support yet
            opt.use_vulkan_compute = false;
        }

        // pull out top shape hints
        Mat shape_hints = pd.get(30, Mat());
        if (!shape_hints.empty())
        {
            const int* psh = shape_hints;
            for (int j = 0; j < top_count; j++)
            {
                Blob& blob = d->blobs[layer->tops[j]];

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
        for (int j = 0; j < bottom_count; j++)
        {
            layer->bottom_shapes[j] = d->blobs[layer->bottoms[j]].shape;
        }

        layer->top_shapes.resize(top_count);
        for (int j = 0; j < top_count; j++)
        {
            layer->top_shapes[j] = d->blobs[layer->tops[j]].shape;
        }

        // pull out layer specific feature disabled set
        layer->featmask = pd.get(31, 0);

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            NCNN_LOGE("layer load_param %d %s failed", i, layer->name.c_str());
            continue;
        }

        d->layers[i] = layer;
    }

    d->update_input_output_indexes();
    d->update_input_output_names();

#undef SCAN_VALUE
    return 0;
}
#endif // NCNN_STRING

int Net::load_param_bin(const DataReader& dr)
{
#define READ_VALUE(buf)                            \
    if (dr.read(&buf, sizeof(buf)) != sizeof(buf)) \
    {                                              \
        NCNN_LOGE("read " #buf " failed");         \
        return -1;                                 \
    }

    int magic = 0;
    READ_VALUE(magic)
    if (magic != 7767517)
    {
        NCNN_LOGE("param is too old, please regenerate");
        return -1;
    }

    int layer_count = 0;
    int blob_count = 0;
    READ_VALUE(layer_count)
    READ_VALUE(blob_count)
    if (layer_count <= 0 || blob_count <= 0)
    {
        NCNN_LOGE("invalid layer_count or blob_count");
        return -1;
    }

    d->layers.resize(layer_count);
    d->blobs.resize(blob_count);

#if NCNN_VULKAN
    // TODO enable gpu when bf16 conversion implemented
    if (opt.use_bf16_storage)
        opt.use_vulkan_compute = false;

    if (opt.use_vulkan_compute)
    {
        if (!d->vkdev) d->vkdev = get_gpu_device();
        if (!d->vkdev) opt.use_vulkan_compute = false; // no vulkan device, fallback to cpu
    }
    if (opt.use_vulkan_compute)
    {
        // sanitize use options
        if (!d->vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
        if (!d->vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;
        if (!d->vkdev->info.support_fp16_arithmetic()) opt.use_fp16_arithmetic = false;
        if (!d->vkdev->info.support_int8_storage()) opt.use_int8_storage = false;
        if (!d->vkdev->info.support_int8_arithmetic()) opt.use_int8_arithmetic = false;
        if (!d->vkdev->info.support_cooperative_matrix()) opt.use_cooperative_matrix = false;

        if (d->vkdev->info.bug_buffer_image_load_zero()) opt.use_image_storage = false;

        // enable local memory optimization on discrete gpu only
        if (d->vkdev->info.type() != 0) opt.use_shader_local_memory = false;

        // fp16a makes no sense when fp16 storage disabled
        if (!opt.use_fp16_packed && !opt.use_fp16_storage) opt.use_fp16_arithmetic = false;
    }
    else
    {
        // fp16a makes no sense when fp16 storage disabled
        if (!opt.use_fp16_storage) opt.use_fp16_arithmetic = false;
    }
#endif // NCNN_VULKAN

    ParamDict pd;

    for (int i = 0; i < layer_count; i++)
    {
        int typeindex;
        int bottom_count;
        int top_count;
        READ_VALUE(typeindex)
        READ_VALUE(bottom_count)
        READ_VALUE(top_count)

        Layer* layer = create_overwrite_builtin_layer(typeindex);
        if (!layer)
        {
            layer = create_layer(typeindex);
        }
        if (!layer)
        {
            int custom_index = typeindex & ~LayerType::CustomBit;
            layer = create_custom_layer(custom_index);
        }
        if (!layer)
        {
            NCNN_LOGE("layer %d not exists or registered", typeindex);
            clear();
            return -1;
        }

#if NCNN_VULKAN
        if (opt.use_vulkan_compute)
            layer->vkdev = d->vkdev;
#endif // NCNN_VULKAN

        //         layer->type = std::string(layer_type);
        //         layer->name = std::string(layer_name);
        //         NCNN_LOGE("new layer %d", typeindex);

        layer->bottoms.resize(bottom_count);
        for (int j = 0; j < bottom_count; j++)
        {
            int bottom_blob_index;
            READ_VALUE(bottom_blob_index)

            Blob& blob = d->blobs[bottom_blob_index];

            blob.consumer = i;

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int j = 0; j < top_count; j++)
        {
            int top_blob_index;
            READ_VALUE(top_blob_index)

            Blob& blob = d->blobs[top_blob_index];

            //             blob.name = std::string(blob_name);
            //             NCNN_LOGE("new blob %s", blob_name);

            blob.producer = i;

            layer->tops[j] = top_blob_index;
        }

        // layer specific params
        int pdlr = pd.load_param_bin(dr);
        if (pdlr != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("ParamDict load_param %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("ParamDict load_param %d failed", i);
#endif
            continue;
        }

        if (layer->support_int8_storage)
        {
            // no int8 gpu support yet
            opt.use_vulkan_compute = false;
        }

        // pull out top blob shape hints
        Mat shape_hints = pd.get(30, Mat());
        if (!shape_hints.empty())
        {
            const int* psh = shape_hints;
            for (int j = 0; j < top_count; j++)
            {
                Blob& blob = d->blobs[layer->tops[j]];

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
        for (int j = 0; j < bottom_count; j++)
        {
            layer->bottom_shapes[j] = d->blobs[layer->bottoms[j]].shape;
        }

        layer->top_shapes.resize(top_count);
        for (int j = 0; j < top_count; j++)
        {
            layer->top_shapes[j] = d->blobs[layer->tops[j]].shape;
        }

        // pull out layer specific feature disabled set
        layer->featmask = pd.get(31, 0);

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("layer load_param %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("layer load_param %d failed", i);
#endif
            continue;
        }

        d->layers[i] = layer;
    }

    d->update_input_output_indexes();

#undef READ_VALUE
    return 0;
}

int Net::load_model(const DataReader& dr)
{
    if (d->layers.empty())
    {
        NCNN_LOGE("network graph not ready");
        return -1;
    }

    int layer_count = (int)d->layers.size();

    // load file
    int ret = 0;

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        if (!opt.pipeline_cache)
        {
            if (!d->pipeline_cache)
                d->pipeline_cache = new PipelineCache(d->vkdev);
            opt.pipeline_cache = d->pipeline_cache;
        }
    }
#endif // NCNN_VULKAN

    ModelBinFromDataReader mb(dr);
    for (int i = 0; i < layer_count; i++)
    {
        Layer* layer = d->layers[i];

        //Here we found inconsistent content in the parameter file.
        if (!layer)
        {
            NCNN_LOGE("load_model error at layer %d, parameter file has inconsistent content.", i);
            ret = -1;
            break;
        }

        int lret = layer->load_model(mb);
        if (lret != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("layer load_model %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("layer load_model %d failed", i);
#endif
            ret = -1;
            break;
        }

        if (layer->support_int8_storage)
        {
            // no int8 gpu support yet
            opt.use_vulkan_compute = false;
        }

        Option opt1 = get_masked_option(opt, layer->featmask);
#if NCNN_VULKAN
        if (opt1.use_vulkan_compute)
        {
            if (!layer->support_image_storage) opt1.use_image_storage = false;
        }
        else
        {
            layer->vkdev = 0;
            layer->support_vulkan = false;
        }
#endif // NCNN_VULKAN

        int cret = layer->create_pipeline(opt1);
        if (cret != 0)
        {
#if NCNN_STRING
            NCNN_LOGE("layer create_pipeline %d %s failed", i, layer->name.c_str());
#else
            NCNN_LOGE("layer create_pipeline %d failed", i);
#endif
            ret = -1;
            break;
        }
    }

    if (opt.use_local_pool_allocator)
    {
        if (opt.blob_allocator == 0)
        {
            if (!d->local_blob_allocator)
            {
                d->local_blob_allocator = new PoolAllocator;
                d->local_blob_allocator->set_size_compare_ratio(0.f);
            }
        }
        if (opt.workspace_allocator == 0)
        {
            if (!d->local_workspace_allocator)
            {
                d->local_workspace_allocator = new PoolAllocator;
                d->local_workspace_allocator->set_size_compare_ratio(0.f);
            }
        }
    }

#if NCNN_VULKAN
    if (ret == 0 && opt.use_vulkan_compute)
    {
        ret = d->upload_model();
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
        NCNN_LOGE("fopen %s failed", protopath);
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
        NCNN_LOGE("fopen %s failed", protopath);
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
        NCNN_LOGE("fopen %s failed", modelpath);
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

#if NCNN_PLATFORM_API
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
        NCNN_LOGE("AAssetManager_open %s failed", assetpath);
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
        NCNN_LOGE("AAssetManager_open %s failed", assetpath);
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
        NCNN_LOGE("AAssetManager_open %s failed", assetpath);
        return -1;
    }

    int ret = load_model(asset);
    AAsset_close(asset);
    return ret;
}
#endif // __ANDROID_API__ >= 9
#endif // NCNN_PLATFORM_API

void Net::clear()
{
    d->blobs.clear();
    for (size_t i = 0; i < d->layers.size(); i++)
    {
        Layer* layer = d->layers[i];

        Option opt1 = get_masked_option(opt, layer->featmask);
#if NCNN_VULKAN
        if (!layer->support_image_storage)
        {
            opt1.use_image_storage = false;
        }
#endif // NCNN_VULKAN

        int dret = layer->destroy_pipeline(opt1);
        if (dret != 0)
        {
            NCNN_LOGE("layer destroy_pipeline failed");
            // ignore anyway
        }

        if (layer->typeindex & ncnn::LayerType::CustomBit)
        {
            int custom_index = layer->typeindex & ~ncnn::LayerType::CustomBit;
            if (d->custom_layer_registry[custom_index].destroyer)
            {
                d->custom_layer_registry[custom_index].destroyer(layer, d->custom_layer_registry[custom_index].userdata);
            }
            else
            {
                delete layer;
            }
        }
        else
        {
            // check overwrite builtin layer destroyer
            int index = -1;
            const size_t overwrite_builtin_layer_registry_entry_count = d->overwrite_builtin_layer_registry.size();
            for (size_t i = 0; i < overwrite_builtin_layer_registry_entry_count; i++)
            {
                if (d->overwrite_builtin_layer_registry[i].typeindex == layer->typeindex)
                {
                    index = i;
                    break;
                }
            }

            if (index != -1 && d->overwrite_builtin_layer_registry[index].destroyer)
            {
                d->overwrite_builtin_layer_registry[index].destroyer(layer, d->overwrite_builtin_layer_registry[index].userdata);
            }
            else
            {
                delete layer;
            }
        }
    }
    d->layers.clear();

    if (d->local_blob_allocator)
    {
        delete d->local_blob_allocator;
        d->local_blob_allocator = 0;
    }
    if (d->local_workspace_allocator)
    {
        delete d->local_workspace_allocator;
        d->local_workspace_allocator = 0;
    }

#if NCNN_VULKAN
    if (d->weight_vkallocator)
    {
        delete d->weight_vkallocator;
        d->weight_vkallocator = 0;
    }
    if (d->weight_staging_vkallocator)
    {
        delete d->weight_staging_vkallocator;
        d->weight_staging_vkallocator = 0;
    }
    if (d->pipeline_cache)
    {
        delete d->pipeline_cache;
        d->pipeline_cache = 0;
        opt.pipeline_cache = 0;
    }
#endif // NCNN_VULKAN
}

Extractor Net::create_extractor() const
{
    return Extractor(this, d->blobs.size());
}

const std::vector<int>& Net::input_indexes() const
{
    return d->input_blob_indexes;
}

const std::vector<int>& Net::output_indexes() const
{
    return d->output_blob_indexes;
}

#if NCNN_STRING
const std::vector<const char*>& Net::input_names() const
{
    return d->input_blob_names;
}

const std::vector<const char*>& Net::output_names() const
{
    return d->output_blob_names;
}
#endif

const std::vector<Blob>& Net::blobs() const
{
    return d->blobs;
}

const std::vector<Layer*>& Net::layers() const
{
    return d->layers;
}

std::vector<Blob>& Net::mutable_blobs()
{
    return d->blobs;
}

std::vector<Layer*>& Net::mutable_layers()
{
    return d->layers;
}

#if NCNN_VULKAN
void Net::set_vulkan_device(int device_index)
{
    d->vkdev = get_gpu_device(device_index);
}

void Net::set_vulkan_device(const VulkanDevice* _vkdev)
{
    d->vkdev = _vkdev;
}

const VulkanDevice* Net::vulkan_device() const
{
    return d->vkdev;
}
#endif // NCNN_VULKAN

#if NCNN_STRING
int Net::find_blob_index_by_name(const char* name) const
{
    for (size_t i = 0; i < d->blobs.size(); i++)
    {
        const Blob& blob = d->blobs[i];
        if (blob.name == name)
        {
            return static_cast<int>(i);
        }
    }

    NCNN_LOGE("find_blob_index_by_name %s failed", name);
    return -1;
}

int Net::find_layer_index_by_name(const char* name) const
{
    for (size_t i = 0; i < d->layers.size(); i++)
    {
        const Layer* layer = d->layers[i];
        if (layer->name == name)
        {
            return static_cast<int>(i);
        }
    }

    NCNN_LOGE("find_layer_index_by_name %s failed", name);
    return -1;
}

int Net::custom_layer_to_index(const char* type)
{
    const size_t custom_layer_registry_entry_count = d->custom_layer_registry.size();
    for (size_t i = 0; i < custom_layer_registry_entry_count; i++)
    {
        if (strcmp(type, d->custom_layer_registry[i].name) == 0)
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

Layer* Net::create_overwrite_builtin_layer(const char* type)
{
    int typeindex = layer_to_index(type);
    if (typeindex == -1)
        return 0;

    return create_overwrite_builtin_layer(typeindex);
}
#endif // NCNN_STRING

Layer* Net::create_custom_layer(int index)
{
    const size_t custom_layer_registry_entry_count = d->custom_layer_registry.size();
    if (index < 0 || static_cast<unsigned int>(index) >= custom_layer_registry_entry_count)
        return 0;

    layer_creator_func layer_creator = d->custom_layer_registry[index].creator;
    if (!layer_creator)
        return 0;

    Layer* layer = layer_creator(d->custom_layer_registry[index].userdata);
    layer->typeindex = ncnn::LayerType::CustomBit | index;
    return layer;
}

Layer* Net::create_overwrite_builtin_layer(int typeindex)
{
    int index = -1;
    const size_t overwrite_builtin_layer_registry_entry_count = d->overwrite_builtin_layer_registry.size();
    for (size_t i = 0; i < overwrite_builtin_layer_registry_entry_count; i++)
    {
        if (d->overwrite_builtin_layer_registry[i].typeindex == typeindex)
        {
            index = i;
            break;
        }
    }

    if (index == -1)
        return 0;

    layer_creator_func layer_creator = d->overwrite_builtin_layer_registry[index].creator;
    if (!layer_creator)
        return 0;

    Layer* layer = layer_creator(d->overwrite_builtin_layer_registry[index].userdata);
    layer->typeindex = typeindex;
    return layer;
}

class ExtractorPrivate
{
public:
    ExtractorPrivate(const Net* _net)
        : net(_net)
    {
    }
    const Net* net;
    std::vector<Mat> blob_mats;
    Option opt;

#if NCNN_VULKAN
    VkAllocator* local_blob_vkallocator;
    VkAllocator* local_staging_vkallocator;

    std::vector<VkMat> blob_mats_gpu;
    std::vector<VkImageMat> blob_mats_gpu_image;
#endif // NCNN_VULKAN
};

Extractor::Extractor(const Net* _net, size_t blob_count)
    : d(new ExtractorPrivate(_net))
{
    d->blob_mats.resize(blob_count);
    d->opt = d->net->opt;

#if NCNN_VULKAN
    if (d->net->opt.use_vulkan_compute)
    {
        d->local_blob_vkallocator = 0;
        d->local_staging_vkallocator = 0;

        d->blob_mats_gpu.resize(blob_count);
        d->blob_mats_gpu_image.resize(blob_count);
    }
#endif // NCNN_VULKAN
}

Extractor::~Extractor()
{
    clear();

    delete d;
}

Extractor::Extractor(const Extractor& rhs)
    : d(new ExtractorPrivate(0))
{
    d->net = rhs.d->net;
    d->blob_mats = rhs.d->blob_mats;
    d->opt = rhs.d->opt;

#if NCNN_VULKAN
    d->local_blob_vkallocator = 0;
    d->local_staging_vkallocator = 0;

    d->blob_mats_gpu = rhs.d->blob_mats_gpu;
    d->blob_mats_gpu_image = rhs.d->blob_mats_gpu_image;
#endif // NCNN_VULKAN
}

Extractor& Extractor::operator=(const Extractor& rhs)
{
    if (this == &rhs)
        return *this;

    d->net = rhs.d->net;
    d->blob_mats = rhs.d->blob_mats;
    d->opt = rhs.d->opt;

#if NCNN_VULKAN
    d->local_blob_vkallocator = 0;
    d->local_staging_vkallocator = 0;

    d->blob_mats_gpu = rhs.d->blob_mats_gpu;
    d->blob_mats_gpu_image = rhs.d->blob_mats_gpu_image;
#endif // NCNN_VULKAN

    return *this;
}

void Extractor::clear()
{
    d->blob_mats.clear();

#if NCNN_VULKAN
    if (d->opt.use_vulkan_compute)
    {
        d->blob_mats_gpu.clear();
        d->blob_mats_gpu_image.clear();

        if (d->local_blob_vkallocator)
        {
            d->net->vulkan_device()->reclaim_blob_allocator(d->local_blob_vkallocator);
        }
        if (d->local_staging_vkallocator)
        {
            d->net->vulkan_device()->reclaim_staging_allocator(d->local_staging_vkallocator);
        }
    }
#endif // NCNN_VULKAN
}

void Extractor::set_light_mode(bool enable)
{
    d->opt.lightmode = enable;
}

void Extractor::set_num_threads(int num_threads)
{
    d->opt.num_threads = num_threads;
}

void Extractor::set_blob_allocator(Allocator* allocator)
{
    d->opt.blob_allocator = allocator;
}

void Extractor::set_workspace_allocator(Allocator* allocator)
{
    d->opt.workspace_allocator = allocator;
}

#if NCNN_VULKAN
void Extractor::set_vulkan_compute(bool enable)
{
    if (d->net->d->opt.use_vulkan_compute)
    {
        d->opt.use_vulkan_compute = enable;
    }
    else
    {
        NCNN_LOGE("set_vulkan_compute failed, network use_vulkan_compute disabled");
    }
}

void Extractor::set_blob_vkallocator(VkAllocator* allocator)
{
    d->opt.blob_vkallocator = allocator;
}

void Extractor::set_workspace_vkallocator(VkAllocator* allocator)
{
    d->opt.workspace_vkallocator = allocator;
}

void Extractor::set_staging_vkallocator(VkAllocator* allocator)
{
    d->opt.staging_vkallocator = allocator;
}
#endif // NCNN_VULKAN

#if NCNN_STRING
int Extractor::input(const char* blob_name, const Mat& in)
{
    int blob_index = d->net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
    {
        NCNN_LOGE("Try");
        const std::vector<const char*>& input_names = d->net->input_names();
        for (size_t i = 0; i < input_names.size(); i++)
        {
            NCNN_LOGE("    ex.input(\"%s\", in%d);", input_names[i], (int)i);
        }

        return -1;
    }

    return input(blob_index, in);
}

int Extractor::extract(const char* blob_name, Mat& feat, int type)
{
    int blob_index = d->net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
    {
        NCNN_LOGE("Try");
        const std::vector<const char*>& output_names = d->net->output_names();
        for (size_t i = 0; i < output_names.size(); i++)
        {
            NCNN_LOGE("    ex.extract(\"%s\", out%d);", output_names[i], (int)i);
        }

        return -1;
    }

    return extract(blob_index, feat, type);
}
#endif // NCNN_STRING

int Extractor::input(int blob_index, const Mat& in)
{
    if (blob_index < 0 || blob_index >= (int)d->blob_mats.size())
        return -1;

    d->blob_mats[blob_index] = in;

    return 0;
}

int Extractor::extract(int blob_index, Mat& feat, int type)
{
    if (blob_index < 0 || blob_index >= (int)d->blob_mats.size())
        return -1;

    int old_blocktime = get_kmp_blocktime();
    set_kmp_blocktime(d->opt.openmp_blocktime);

    int old_flush_denormals = get_flush_denormals();
    set_flush_denormals(d->opt.flush_denormals);

    int ret = 0;

    if (d->blob_mats[blob_index].dims == 0)
    {
        int layer_index = d->net->blobs()[blob_index].producer;

        // use local allocator
        if (d->opt.use_local_pool_allocator)
        {
            if (!d->opt.blob_allocator)
            {
                d->opt.blob_allocator = d->net->d->local_blob_allocator;
            }
            if (!d->opt.workspace_allocator)
            {
                d->opt.workspace_allocator = d->net->d->local_workspace_allocator;
            }
        }

#if NCNN_VULKAN
        if (d->opt.use_vulkan_compute)
        {
            // use local allocator
            if (!d->opt.blob_vkallocator)
            {
                d->local_blob_vkallocator = d->net->vulkan_device()->acquire_blob_allocator();
                d->opt.blob_vkallocator = d->local_blob_vkallocator;
            }
            if (!d->opt.workspace_vkallocator)
            {
                d->opt.workspace_vkallocator = d->opt.blob_vkallocator;
            }
            if (!d->opt.staging_vkallocator)
            {
                d->local_staging_vkallocator = d->net->vulkan_device()->acquire_staging_allocator();
                d->opt.staging_vkallocator = d->local_staging_vkallocator;
            }

            ncnn::VkCompute cmd(d->net->vulkan_device());
#if NCNN_BENCHMARK
            cmd.create_query_pool(d->net->layers().size() * 2);
#endif // NCNN_BENCHMARK

            // TODO vkimagemat for adreno
            if (d->opt.use_image_storage)
            {
                VkImageMat feat_gpu;
                ret = extract(blob_index, feat_gpu, cmd);

                if (ret == 0 && d->blob_mats[blob_index].dims == 0 && feat_gpu.dims != 0)
                {
                    cmd.record_download(feat_gpu, d->blob_mats[blob_index], d->opt);

                    ret = cmd.submit_and_wait();

#if NCNN_BENCHMARK
                    std::vector<uint64_t> results(d->net->layers().size() * 2);
                    cmd.get_query_pool_results(0, d->net->layers().size() * 2, results);
                    for (size_t i = 0; i < d->net->layers().size(); i++)
                    {
                        uint64_t start = results[i * 2];
                        uint64_t end = results[i * 2 + 1];
                        if (start == 0 || end == 0)
                            continue;

                        double duration_us = (end - start) * d->net->vulkan_device()->info.timestamp_period() / 1000;
                        NCNN_LOGE("%-24s %-30s %8.2lfus    |", d->net->layers()[i]->type.c_str(), d->net->layers()[i]->name.c_str(), duration_us);
                    }
#endif // NCNN_BENCHMARK
                }
            }
            else
            {
                VkMat feat_gpu;
                ret = extract(blob_index, feat_gpu, cmd);

                if (ret == 0 && d->blob_mats[blob_index].dims == 0 && feat_gpu.dims != 0)
                {
                    cmd.record_download(feat_gpu, d->blob_mats[blob_index], d->opt);

                    ret = cmd.submit_and_wait();

#if NCNN_BENCHMARK
                    std::vector<uint64_t> results(d->net->layers().size() * 2);
                    cmd.get_query_pool_results(0, d->net->layers().size() * 2, results);
                    for (size_t i = 0; i < d->net->layers().size(); i++)
                    {
                        uint64_t start = results[i * 2];
                        uint64_t end = results[i * 2 + 1];
                        if (start == 0 || end == 0)
                            continue;

                        double duration_us = (end - start) * d->net->vulkan_device()->info.timestamp_period() / 1000;
                        NCNN_LOGE("%-24s %-30s %8.2lfus    |", d->net->layers()[i]->type.c_str(), d->net->layers()[i]->name.c_str(), duration_us);
                    }
#endif // NCNN_BENCHMARK
                }
            }
        }
        else
        {
            ret = d->net->d->forward_layer(layer_index, d->blob_mats, d->opt);
        }
#else
        ret = d->net->d->forward_layer(layer_index, d->blob_mats, d->opt);
#endif // NCNN_VULKAN
    }

    feat = d->blob_mats[blob_index];

    if (d->opt.use_packing_layout && (type == 0) && feat.elempack != 1)
    {
        Mat bottom_blob_unpacked;
        convert_packing(feat, bottom_blob_unpacked, 1, d->opt);
        feat = bottom_blob_unpacked;
    }

    // clang-format off
    // *INDENT-OFF*
#if NCNN_ARM82
    if (d->opt.use_fp16_storage && cpu_support_arm_asimdhp() && (type == 0))
    {
        if (feat.elembits() == 16)
        {
            Mat feat_fp32;
            cast_float16_to_float32(feat, feat_fp32, d->opt);
            feat = feat_fp32;
        }
    }
    else
#endif // NCNN_ARM82
#if NCNN_BF16
    if (d->opt.use_bf16_storage && (type == 0))
    {
        if (feat.elembits() == 16)
        {
            Mat feat_fp32;
            cast_bfloat16_to_float32(feat, feat_fp32, d->opt);
            feat = feat_fp32;
        }
    }
    else
#endif // NCNN_BF16
    if (feat.elembits() == 8 && (type == 0))
    {
        Mat feat_fp32;
        cast_int8_to_float32(feat, feat_fp32, d->opt);
        feat = feat_fp32;
    }
    // *INDENT-ON*
    // clang-format on

    if (d->opt.use_local_pool_allocator && feat.allocator == d->net->d->local_blob_allocator)
    {
        // detach the returned mat from local pool allocator
        // so we could destroy net instance much earlier
        feat = feat.clone();
    }

    set_kmp_blocktime(old_blocktime);
    set_flush_denormals(old_flush_denormals);

    return ret;
}

#if NCNN_VULKAN
#if NCNN_STRING
int Extractor::input(const char* blob_name, const VkMat& in)
{
    int blob_index = d->net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
    {
        NCNN_LOGE("Try");
        const std::vector<const char*>& input_names = d->net->input_names();
        for (size_t i = 0; i < input_names.size(); i++)
        {
            NCNN_LOGE("    ex.input(\"%s\", in%d);", input_names[i], (int)i);
        }

        return -1;
    }

    return input(blob_index, in);
}

int Extractor::extract(const char* blob_name, VkMat& feat, VkCompute& cmd)
{
    int blob_index = d->net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
    {
        NCNN_LOGE("Try");
        const std::vector<const char*>& output_names = d->net->output_names();
        for (size_t i = 0; i < output_names.size(); i++)
        {
            NCNN_LOGE("    ex.extract(\"%s\", out%d);", output_names[i], (int)i);
        }

        return -1;
    }

    return extract(blob_index, feat, cmd);
}

int Extractor::input(const char* blob_name, const VkImageMat& in)
{
    int blob_index = d->net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
    {
        NCNN_LOGE("Try");
        const std::vector<const char*>& input_names = d->net->input_names();
        for (size_t i = 0; i < input_names.size(); i++)
        {
            NCNN_LOGE("    ex.input(\"%s\", in%d);", input_names[i], (int)i);
        }

        return -1;
    }

    return input(blob_index, in);
}

int Extractor::extract(const char* blob_name, VkImageMat& feat, VkCompute& cmd)
{
    int blob_index = d->net->find_blob_index_by_name(blob_name);
    if (blob_index == -1)
    {
        NCNN_LOGE("Try");
        const std::vector<const char*>& output_names = d->net->output_names();
        for (size_t i = 0; i < output_names.size(); i++)
        {
            NCNN_LOGE("    ex.extract(\"%s\", out%d);", output_names[i], (int)i);
        }

        return -1;
    }

    return extract(blob_index, feat, cmd);
}
#endif // NCNN_STRING

int Extractor::input(int blob_index, const VkMat& in)
{
    if (blob_index < 0 || blob_index >= (int)d->blob_mats.size())
        return -1;

    d->blob_mats_gpu[blob_index] = in;

    return 0;
}

int Extractor::extract(int blob_index, VkMat& feat, VkCompute& cmd)
{
    if (blob_index < 0 || blob_index >= (int)d->blob_mats.size())
        return -1;

    int old_blocktime = get_kmp_blocktime();
    set_kmp_blocktime(d->opt.openmp_blocktime);

    int old_flush_denormals = get_flush_denormals();
    set_flush_denormals(d->opt.flush_denormals);

    int ret = 0;

    if (d->blob_mats_gpu[blob_index].dims == 0)
    {
        if (d->blob_mats_gpu_image[blob_index].dims != 0)
        {
            // image to buffer
            cmd.record_image_to_buffer(d->blob_mats_gpu_image[blob_index], d->blob_mats_gpu[blob_index], d->opt);
        }
        else if (d->blob_mats[blob_index].dims != 0)
        {
            // host to buffer
            cmd.record_upload(d->blob_mats[blob_index], d->blob_mats_gpu[blob_index], d->opt);
        }
        else
        {
            int layer_index = d->net->blobs()[blob_index].producer;
            ret = d->net->d->forward_layer(layer_index, d->blob_mats, d->blob_mats_gpu, cmd, d->opt);
        }
    }

    feat = d->blob_mats_gpu[blob_index];

    set_kmp_blocktime(old_blocktime);
    set_flush_denormals(old_flush_denormals);

    return ret;
}

int Extractor::input(int blob_index, const VkImageMat& in)
{
    if (blob_index < 0 || blob_index >= (int)d->blob_mats.size())
        return -1;

    d->blob_mats_gpu_image[blob_index] = in;

    return 0;
}

int Extractor::extract(int blob_index, VkImageMat& feat, VkCompute& cmd)
{
    if (blob_index < 0 || blob_index >= (int)d->blob_mats.size())
        return -1;

    int old_blocktime = get_kmp_blocktime();
    set_kmp_blocktime(d->opt.openmp_blocktime);

    int old_flush_denormals = get_flush_denormals();
    set_flush_denormals(d->opt.flush_denormals);

    int ret = 0;

    if (d->blob_mats_gpu_image[blob_index].dims == 0)
    {
        if (d->blob_mats_gpu[blob_index].dims != 0)
        {
            // buffer to image
            cmd.record_buffer_to_image(d->blob_mats_gpu[blob_index], d->blob_mats_gpu_image[blob_index], d->opt);
        }
        else if (d->blob_mats[blob_index].dims != 0)
        {
            // host to image
            cmd.record_upload(d->blob_mats[blob_index], d->blob_mats_gpu_image[blob_index], d->opt);
        }
        else
        {
            int layer_index = d->net->blobs()[blob_index].producer;
            ret = d->net->d->forward_layer(layer_index, d->blob_mats, d->blob_mats_gpu, d->blob_mats_gpu_image, cmd, d->opt);
        }
    }

    feat = d->blob_mats_gpu_image[blob_index];

    if (feat.empty())
    {
        NCNN_LOGE("extract %d image allocation failed", blob_index);
        ret = -100;
    }

    set_kmp_blocktime(old_blocktime);
    set_flush_denormals(old_flush_denormals);

    return ret;
}
#endif // NCNN_VULKAN

} // namespace ncnn
