// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "command.h"

#if NCNN_VULKAN

#include "option.h"
#include "pipeline.h"
#include <limits.h>

#if !defined _WIN32
#include <time.h>
#endif

#if NCNN_BENCHMARK
#include "benchmark.h"
#endif // NCNN_BENCHMARK

namespace ncnn {

int fit_weight_buffer_size(const VulkanDevice* vkdev, size_t memory_limit, bool host, size_t& capacity);
VkBufferMemory* create_weight_buffer(const VulkanDevice* vkdev, size_t size, int storage_type, bool unified, size_t memory_limit, size_t& allocation_size, bool concurrent);
void destroy_weight_buffer(const VulkanDevice* vkdev, VkBufferMemory* buffer);
int flush_weight_buffer(const VulkanDevice* vkdev, const VkBufferMemory* buffer, size_t offset, size_t size);

static void wait_for_pending_commands()
{
#if defined _WIN32
    Sleep(1);
#else
    struct timespec delay = {0, 1000000};
    nanosleep(&delay, 0);
#endif
}

static bool is_deferred_buffer(const VkBufferMemory* memory)
{
    return memory && memory->storage_type != 0;
}

static bool needs_host_copy(const VkMat& src)
{
    return src.data->storage_type == 2 || !src.buffer();
}

class VkComputePrivate
{
public:
    VkComputePrivate(const VulkanDevice* _vkdev);
    ~VkComputePrivate();

    int init();
    int begin_command_buffer();
    int end_command_buffer();

    int record_submit_compute();
    int record_resolve(const VkMat& src, VkMat& dst);

    void flush_records();
    void clear_records();

    int submit_records();
    int wait_fence(VkFence fence);
    int finish_execution(int result = 0);
    void clear_execution();

    const VulkanDevice* vkdev;

    VkCommandPool compute_command_pool;

    // command buffers are appended in recording order
    std::vector<VkCommandBuffer> compute_command_buffers;
    std::vector<VkFence> compute_command_fences;
    std::vector<bool> compute_command_pending;
    int compute_command_buffer_index;

    std::vector<VkSemaphore> compute_upload_semaphores;
    std::vector<VkSemaphore> compute_wait_semaphores;

    VkCommandPool transfer_command_pool;
    uint32_t transfer_queue_family_index;

    std::vector<VkCommandBuffer> upload_command_buffers;
    std::vector<VkFence> upload_command_fences;
    std::vector<bool> upload_command_pending;
    std::vector<VkSemaphore> upload_compute_semaphores;

    // owns all compute and upload synchronization semaphores
    std::vector<VkSemaphore> execution_semaphores;
    bool executing;

    VkAllocator* local_weight_vkallocator;
    VkAllocator* local_staging_vkallocator;
    VkMat weight_staging_buffer;
    std::vector<VkMat> weight_buffers;
    int weight_wait_compute_index;
    int weight_last_compute_index;
    VkSemaphore weight_upload_wait_semaphore;
    bool use_weight_upload_online;

    std::vector<VkMat> upload_src_buffers;
    // external views with command-owned metadata copies
    std::vector<VkMat> upload_dst_buffers;

    std::vector<VkMat> upload_staging_buffers;
    std::vector<VkMat> download_post_buffers;
    std::vector<Mat> download_post_mats_fp16;
    std::vector<Mat> download_post_mats;

    std::vector<VkImageMemory*> image_blocks_to_destroy;

    // the good-old path for device without VK_KHR_push_descriptor
    std::vector<VkDescriptorPool> descriptor_pools;
    std::vector<VkDescriptorSet> descriptorsets;

    struct record
    {
        enum
        {
            TYPE_copy_buffer,
            TYPE_copy_image,
            TYPE_copy_buffer_to_image,
            TYPE_copy_image_to_buffer,
            TYPE_bind_pipeline,
            TYPE_bind_descriptorsets,
            TYPE_push_constants,
            TYPE_dispatch,
            TYPE_memory_barrers,
            TYPE_buffer_barrers,
            TYPE_image_barrers,

#if NCNN_BENCHMARK
            TYPE_write_timestamp,
#endif // NCNN_BENCHMARK

            TYPE_host_copy,
            TYPE_wait_compute,
            TYPE_wait_upload,
            TYPE_submit_compute,
            TYPE_submit_upload,

            TYPE_post_download,
            TYPE_post_cast_float16_to_float32,
            TYPE_post_cast_bfloat16_to_float32,
        };

        int type;
        VkCommandBuffer command_buffer;

        union
        {
            struct
            {
                VkBuffer src;
                VkBuffer dst;
                uint32_t region_count;
                const VkBufferCopy* regions;
            } copy_buffer;
            struct
            {
                VkImage src;
                VkImageLayout src_layout;
                VkImage dst;
                VkImageLayout dst_layout;
                uint32_t region_count;
                const VkImageCopy* regions;
            } copy_image;
            struct
            {
                VkBuffer src;
                VkImage dst;
                VkImageLayout layout;
                uint32_t region_count;
                const VkBufferImageCopy* regions;
            } copy_buffer_to_image;
            struct
            {
                VkImage src;
                VkImageLayout layout;
                VkBuffer dst;
                uint32_t region_count;
                const VkBufferImageCopy* regions;
            } copy_image_to_buffer;

            struct
            {
                VkPipelineBindPoint bind_point;
                VkPipeline pipeline;
#if NCNN_COVERAGE
                VkPipelineLayout pipeline_layout;
#endif
            } bind_pipeline;
            struct
            {
                VkPipelineBindPoint bind_point;
                VkPipelineLayout pipeline_layout;
                uint32_t descriptorset_count;
                uint32_t descriptorset_offset;
            } bind_descriptorsets;
            struct
            {
                VkPipelineLayout pipeline_layout;
                VkShaderStageFlags stage_flags;
                uint32_t size;
                const void* values;
            } push_constants;

            struct
            {
                uint32_t group_count_x;
                uint32_t group_count_y;
                uint32_t group_count_z;
            } dispatch;

            struct
            {
                VkPipelineStageFlags src_stage;
                VkPipelineStageFlags dst_stage;
                uint32_t barrier_count;
                const VkMemoryBarrier* barriers;
            } memory_barrers;
            struct
            {
                VkPipelineStageFlags src_stage;
                VkPipelineStageFlags dst_stage;
                uint32_t barrier_count;
                const VkBufferMemoryBarrier* barriers;
            } buffer_barrers;
            struct
            {
                VkPipelineStageFlags src_stage;
                VkPipelineStageFlags dst_stage;
                uint32_t barrier_count;
                const VkImageMemoryBarrier* barriers;
            } image_barrers;

#if NCNN_BENCHMARK
            struct
            {
                uint32_t query;
            } write_timestamp;
#endif // NCNN_BENCHMARK

            struct
            {
                size_t src_index;
                size_t dst_index;
                size_t src_offset;
                size_t size;
            } host_copy;
            struct
            {
                int command_buffer_index;
            } wait;
            struct
            {
                int command_buffer_index;
                uint32_t wait_semaphore_count;
                const VkSemaphore* wait_semaphores;
                VkSemaphore signal_semaphore;
            } submit;

            struct
            {
                uint32_t download_post_buffer_mat_offset;
                uint32_t download_post_mat_fp16_offset;
            } post_download;
            struct
            {
                uint32_t download_post_mat_fp16_offset;
                uint32_t download_post_mat_offset;
                int num_threads;
            } post_cast_float16_to_float32;
            struct
            {
                uint32_t download_post_mat_bf16_offset;
                uint32_t download_post_mat_offset;
                int num_threads;
            } post_cast_bfloat16_to_float32;
        };
    };

    std::vector<record> delayed_records;
    size_t recorded_count;

    uint64_t pending_dispatch_total;

#if NCNN_BENCHMARK
    uint32_t query_count;
    VkQueryPool query_pool;
    bool query_results_ready;

    uint64_t bytes_read;
    uint64_t bytes_uploaded;
    double weight_start_time;
#endif // NCNN_BENCHMARK
};

VkComputePrivate::VkComputePrivate(const VulkanDevice* _vkdev)
    : vkdev(_vkdev)
{
    compute_command_pool = 0;
    compute_command_buffers.resize(1, 0);
    compute_command_fences.resize(1, 0);
    compute_command_pending.resize(1, false);
    compute_command_buffer_index = -1;

    compute_upload_semaphores.resize(1, 0);

    transfer_command_pool = 0;
    transfer_queue_family_index = vkdev->info.transfer_queue_family_index();

    executing = false;

    local_weight_vkallocator = 0;
    local_staging_vkallocator = 0;
    weight_wait_compute_index = -1;
    weight_last_compute_index = -1;
    weight_upload_wait_semaphore = 0;
    use_weight_upload_online = false;

    recorded_count = 0;

    pending_dispatch_total = 0;

#if NCNN_BENCHMARK
    query_count = 0;
    query_pool = 0;
    query_results_ready = false;

    bytes_read = 0;
    bytes_uploaded = 0;
    weight_start_time = 0;
#endif // NCNN_BENCHMARK

    init();
}

VkComputePrivate::~VkComputePrivate()
{
    while (executing)
    {
        finish_execution();
        if (executing)
            wait_for_pending_commands();
    }
    if (compute_command_buffers[0])
        vkResetCommandBuffer(compute_command_buffers[0], 0);
    clear_execution();
    vkDestroyCommandPool(vkdev->vkdevice(), transfer_command_pool, 0);
    clear_records();
    delete local_weight_vkallocator;
    delete local_staging_vkallocator;

    for (size_t i = 0; i < image_blocks_to_destroy.size(); i++)
    {
        VkImageMemory* ptr = image_blocks_to_destroy[i];

        int old_command_refcount = NCNN_XADD(&ptr->command_refcount, -1);
        if (ptr->refcount == 0 && old_command_refcount == 1)
        {
            // no userspace reference and we are the last command reference
            vkDestroyImageView(vkdev->vkdevice(), ptr->imageview, 0);
            vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);

            delete ptr;
        }
        else
        {
            // reference exists in user code or other command
        }
    }
    image_blocks_to_destroy.clear();

    if (!vkdev->info.support_VK_KHR_push_descriptor())
    {
        for (size_t i = 0; i < descriptorsets.size(); i++)
        {
            vkFreeDescriptorSets(vkdev->vkdevice(), descriptor_pools[i], 1, &descriptorsets[i]);
            vkDestroyDescriptorPool(vkdev->vkdevice(), descriptor_pools[i], 0);
        }
    }

#if NCNN_BENCHMARK
    if (query_pool)
    {
        // all submitted commands that refer to queryPool must have completed execution
        vkResetCommandBuffer(compute_command_buffers[0], 0);

        vkDestroyQueryPool(vkdev->vkdevice(), query_pool, 0);
    }
#endif // NCNN_BENCHMARK

    vkDestroyFence(vkdev->vkdevice(), compute_command_fences[0], 0);

    if (compute_command_buffers[0])
        vkFreeCommandBuffers(vkdev->vkdevice(), compute_command_pool, 1, &compute_command_buffers[0]);
    vkDestroyCommandPool(vkdev->vkdevice(), compute_command_pool, 0);
}

int VkComputePrivate::init()
{
    // compute_command_pool
    {
        VkCommandPoolCreateInfo commandPoolCreateInfo;
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.pNext = 0;
        commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandPoolCreateInfo.queueFamilyIndex = vkdev->info.compute_queue_family_index();

        VkResult ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &compute_command_pool);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateCommandPool failed %d", ret);
            return -1;
        }
    }

    // compute_command_buffers
    {
        VkCommandBufferAllocateInfo commandBufferAllocateInfo;
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.pNext = 0;
        commandBufferAllocateInfo.commandPool = compute_command_pool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;

        VkResult ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &commandBufferAllocateInfo, &compute_command_buffers[0]);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkAllocateCommandBuffers failed %d", ret);
            return -1;
        }
    }

    // compute_command_fences
    {
        VkFenceCreateInfo fenceCreateInfo;
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.pNext = 0;
        fenceCreateInfo.flags = 0;

        VkResult ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &compute_command_fences[0]);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateFence failed %d", ret);
            return -1;
        }
    }

    // transfer_command_pool
    {
        VkCommandPoolCreateInfo commandPoolCreateInfo;
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.pNext = 0;
        commandPoolCreateInfo.flags = 0;
        commandPoolCreateInfo.queueFamilyIndex = transfer_queue_family_index;

        VkResult ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &transfer_command_pool);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateCommandPool failed %d", ret);
            return ret;
        }
    }

    return begin_command_buffer();
}

int VkComputePrivate::begin_command_buffer()
{
    VkCommandBufferBeginInfo commandBufferBeginInfo;
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferBeginInfo.pNext = 0;
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    commandBufferBeginInfo.pInheritanceInfo = 0;

    VkResult ret = vkBeginCommandBuffer(compute_command_buffers[0], &commandBufferBeginInfo);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkBeginCommandBuffer failed %d", ret);
        return -1;
    }

    compute_command_buffer_index = 0;

#if NCNN_BENCHMARK
    if (query_pool)
    {
        vkCmdResetQueryPool(compute_command_buffers[0], query_pool, 0, query_count);
    }

    bytes_read = 0;
    bytes_uploaded = 0;
    weight_start_time = get_current_time();
#endif // NCNN_BENCHMARK

    return 0;
}

int VkComputePrivate::end_command_buffer()
{
#if NCNN_COVERAGE
    vkdev->record_shader_coverage_barrier(compute_command_buffers[0]);
#endif
    VkResult ret = vkEndCommandBuffer(compute_command_buffers[0]);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEndCommandBuffer failed %d", ret);
        return ret;
    }

    return 0;
}

VkCompute::VkCompute(const VulkanDevice* _vkdev)
    : vkdev(_vkdev), d(new VkComputePrivate(_vkdev))
{
}

VkCompute::~VkCompute()
{
    delete d;
}

void VkCompute::set_use_weight_upload_online(bool enable)
{
    d->use_weight_upload_online = enable;
}

void VkCompute::record_upload(const Mat& src, VkMat& dst, const Option& opt)
{
    // NCNN_LOGE("record_upload buffer");

    Mat src_fp16;
    if (src.elemsize == src.elempack * 4u)
    {
        // cpu cast to fp16 (discrete gpu)
        if (vkdev->info.type() == 0 && (opt.use_bf16_storage || opt.use_bf16_packed))
        {
            ncnn::cast_float32_to_bfloat16(src, src_fp16, opt);
        }
        else if (vkdev->info.type() == 0 && (opt.use_fp16_storage || opt.use_fp16_packed))
        {
            ncnn::cast_float32_to_float16(src, src_fp16, opt);
        }
        else
        {
            src_fp16 = src;
        }
    }
    else
    {
        src_fp16 = src;
    }

    if (src_fp16.empty())
        return;

    // vkdev->convert_packing only handles elempack=1/4
    if (src_fp16.elempack > 4)
    {
        Mat src_fp16_pack4;
        ncnn::convert_packing(src_fp16, src_fp16_pack4, 4, opt);
        src_fp16 = src_fp16_pack4;
    }

    // upload
    VkMat dst_staging;
    dst_staging.create_like(src_fp16, opt.staging_vkallocator);
    if (dst_staging.empty() || !dst_staging.mapped_ptr())
        return;

    // stash staging
    d->upload_staging_buffers.push_back(dst_staging);

    //     NCNN_LOGE("upload_staging_buffer %p  ->   %p +%d ~%d", src_fp16.data, dst_staging.buffer(), dst_staging.buffer_offset(), dst_staging.buffer_capacity());

    // memcpy src to device
    for (int b = 0; b < src_fp16.n; b++)
    {
        const Mat src_b = src_fp16.batch(b);
        VkMat staging_b = dst_staging.batch(b);
        memcpy(staging_b.mapped_ptr(), src_b.data, src_b.total() * src_b.elemsize);
    }

    // mark device host-write @ null
    dst_staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    dst_staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

    // resolve dst_elempack
    int dims = src_fp16.dims;
    int elemcount = 0;
    if (dims == 1) elemcount = src_fp16.elempack * src_fp16.w;
    if (dims == 2) elemcount = src_fp16.elempack * src_fp16.h;
    if (dims == 3 || dims == 4) elemcount = src_fp16.elempack * src_fp16.c;

    int dst_elempack = elemcount % 4 == 0 ? 4 : 1;

    // gpu cast to fp16 on the fly (integrated gpu)
    int cast_type_to = 0;
    if (vkdev->info.type() != 0)
    {
        if (opt.use_bf16_storage || opt.use_bf16_packed)
            cast_type_to = 5;
        else if (opt.use_fp16_storage || opt.use_fp16_packed)
            cast_type_to = 2;
        else
            cast_type_to = 1;
    }
    vkdev->convert_packing(dst_staging, dst, dst_elempack, cast_type_to, *this, opt);
}

void VkCompute::record_download(const VkMat& src, Mat& dst, const Option& opt)
{
    // NCNN_LOGE("record_download buffer");

    // resolve dst_elempack
    int dims = src.dims;
    int elemcount = 0;
    if (dims == 1) elemcount = src.elempack * src.w;
    if (dims == 2) elemcount = src.elempack * src.h;
    if (dims == 3 || dims == 4) elemcount = src.elempack * src.c;

    int dst_elempack = 1;
    if (opt.use_packing_layout)
        dst_elempack = elemcount % 4 == 0 ? 4 : 1;
    else
        dst_elempack = 1;

    // gpu cast to fp32 on the fly (integrated gpu)
    Option opt_staging = opt;
    if (!opt_staging.blob_vkallocator->mappable || !opt_staging.blob_vkallocator->cached)
    {
        // On an integrated gpu the blob allocator prefers DEVICE_LOCAL|HOST_VISIBLE
        // "unified" memory, which is mappable but WRITE-COMBINED, not HOST_CACHED.
        // Uploading to it is fast, but memcpy'ing OUT of it is uncached-read slow --
        // measured 0.33 GB/s against 10.3 GB/s for upload on Radeon 8060S. Route the
        // download through the staging allocator, which requests HOST_CACHED.
        opt_staging.blob_vkallocator = opt.staging_vkallocator;
    }
    int cast_type_to = 0;
    if (vkdev->info.type() != 0)
    {
        cast_type_to = 1;
    }

    if (src.elemsize == src.elempack * 1u)
    {
        cast_type_to = 4;
    }

    VkMat dst_staging;
    vkdev->convert_packing(src, dst_staging, dst_elempack, cast_type_to, *this, opt_staging);
    if (dst_staging.empty())
        return;

    Mat dst_fp16;
    record_clone(dst_staging, dst_fp16, opt);
    if (dst_fp16.empty())
        return;

    // cast to fp32 (discrete gpu)
    if (dst_fp16.elemsize == dst_fp16.elempack * 2u)
    {
        if (vkdev->info.type() == 0 && (opt.use_bf16_storage || opt.use_bf16_packed))
        {
            int dims = dst_fp16.dims;
            if (dims == 1)
                dst.create(dst_fp16.w, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, dst_fp16.n, opt.blob_allocator);
            if (dims == 2)
                dst.create(dst_fp16.w, dst_fp16.h, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, dst_fp16.n, opt.blob_allocator);
            if (dims == 3)
                dst.create(dst_fp16.w, dst_fp16.h, dst_fp16.c, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, dst_fp16.n, opt.blob_allocator);
            if (dims == 4)
                dst.create(dst_fp16.w, dst_fp16.h, dst_fp16.d, dst_fp16.c, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, dst_fp16.n, opt.blob_allocator);

            if (dst.empty())
                return;

            d->download_post_mats.push_back(dst);

            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_post_cast_bfloat16_to_float32;
            r.command_buffer = 0;
            r.post_cast_bfloat16_to_float32.download_post_mat_bf16_offset = d->download_post_mats_fp16.size() - 1;
            r.post_cast_bfloat16_to_float32.download_post_mat_offset = d->download_post_mats.size() - 1;
            r.post_cast_bfloat16_to_float32.num_threads = opt.num_threads;
            d->delayed_records.push_back(r);
        }
        else if (vkdev->info.type() == 0 && (opt.use_fp16_storage || opt.use_fp16_packed))
        {
            int dims = dst_fp16.dims;
            if (dims == 1)
                dst.create(dst_fp16.w, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, dst_fp16.n, opt.blob_allocator);
            if (dims == 2)
                dst.create(dst_fp16.w, dst_fp16.h, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, dst_fp16.n, opt.blob_allocator);
            if (dims == 3)
                dst.create(dst_fp16.w, dst_fp16.h, dst_fp16.c, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, dst_fp16.n, opt.blob_allocator);
            if (dims == 4)
                dst.create(dst_fp16.w, dst_fp16.h, dst_fp16.d, dst_fp16.c, (size_t)(dst_fp16.elempack * 4u), dst_fp16.elempack, dst_fp16.n, opt.blob_allocator);

            if (dst.empty())
                return;

            d->download_post_mats.push_back(dst);

            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_post_cast_float16_to_float32;
            r.command_buffer = 0;
            r.post_cast_float16_to_float32.download_post_mat_fp16_offset = d->download_post_mats_fp16.size() - 1;
            r.post_cast_float16_to_float32.download_post_mat_offset = d->download_post_mats.size() - 1;
            r.post_cast_float16_to_float32.num_threads = opt.num_threads;
            d->delayed_records.push_back(r);
        }
        else
        {
            dst = dst_fp16;
        }
    }
    else
    {
        dst = dst_fp16;
    }
}

void VkCompute::record_clone(const Mat& src, VkMat& dst, const Option& opt)
{
    //     NCNN_LOGE("record_clone host to buffer");

    // host to staging
    VkMat dst_staging;
    dst_staging.create_like(src, opt.staging_vkallocator);
    if (dst_staging.empty() || !dst_staging.mapped_ptr())
        return;

    // memcpy src to device
    for (int b = 0; b < src.n; b++)
    {
        const Mat src_b = src.batch(b);
        VkMat staging_b = dst_staging.batch(b);
        memcpy(staging_b.mapped_ptr(), src_b.data, src_b.total() * src_b.elemsize);
    }

    // mark device host-write @ null
    dst_staging.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
    dst_staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

    // staging to device
    record_clone(dst_staging, dst, opt);

    // stash staging
    d->upload_staging_buffers.push_back(dst_staging);
}

void VkCompute::record_clone(const Mat& src, VkImageMat& dst, const Option& opt)
{
    //     NCNN_LOGE("record_clone host to image");

    // host to staging
    VkMat dst_staging;
    Option opt_staging = opt;
    opt_staging.blob_vkallocator = opt.staging_vkallocator;
    record_clone(src, dst_staging, opt_staging);
    if (dst_staging.empty())
        return;

    // staging to image
    record_clone(dst_staging, dst, opt);

    // stash staging
    d->upload_staging_buffers.push_back(dst_staging);
}

void VkCompute::record_clone(const VkMat& src, Mat& dst, const Option& opt)
{
    //     NCNN_LOGE("record_clone buffer to host");

    VkMat src_staging;
    if (!src.mapped_ptr() || is_deferred_buffer(src.data))
    {
        Option opt_staging = opt;
        opt_staging.blob_vkallocator = opt.staging_vkallocator;
        record_clone(src, src_staging, opt_staging);
        if (src_staging.empty())
            return;
    }
    else
    {
        src_staging = src;
    }
    if (!src_staging.mapped_ptr())
        return;

    // create dst
    dst.create_like(src_staging, opt.blob_allocator);
    if (dst.empty())
        return;

    // barrier device any @ compute to host-read @ compute
    if (src_staging.data->access_flags & VK_ACCESS_HOST_WRITE_BIT || src_staging.data->stage_flags != VK_PIPELINE_STAGE_HOST_BIT)
    {
        VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = src_staging.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = src_staging.buffer();
        barriers[0].offset = src_staging.data->offset;
        barriers[0].size = src_staging.data->capacity;

        VkPipelineStageFlags src_stage = src_staging.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_HOST_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdPipelineBarrier(d->compute_command_buffers.back(), src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_buffer_barrers;
            r.command_buffer = d->compute_command_buffers.back();
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark device host-read @ any
        src_staging.data->access_flags = VK_ACCESS_HOST_READ_BIT;
        src_staging.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    }

    // stash download post buffer and mat
    d->download_post_buffers.push_back(src_staging);
    d->download_post_mats_fp16.push_back(dst);

    // post memcpy device to dst
    {
        VkComputePrivate::record r;
        r.type = VkComputePrivate::record::TYPE_post_download;
        r.command_buffer = 0;
        r.post_download.download_post_buffer_mat_offset = d->download_post_buffers.size() - 1;
        r.post_download.download_post_mat_fp16_offset = d->download_post_mats_fp16.size() - 1;
        d->delayed_records.push_back(r);
    }
}

void VkCompute::record_clone(const VkImageMat& src, Mat& dst, const Option& opt)
{
    //     NCNN_LOGE("record_clone image to host");

    // image to staging
    VkMat src_staging;
    Option opt_staging = opt;
    opt_staging.blob_vkallocator = opt.staging_vkallocator;
    record_clone(src, src_staging, opt_staging);
    if (src_staging.empty())
        return;

    // staging to host
    record_clone(src_staging, dst, opt);
}

void VkCompute::record_clone(const VkMat& _src, VkMat& dst, const Option& opt)
{
    // keep the preceding weights live while resolving this operation
    std::vector<VkMat> previous_weights;
    if (d->use_weight_upload_online && is_deferred_buffer(_src.data))
    {
        const int previous_compute_index = d->compute_command_buffer_index;
        int ret = d->record_submit_compute();
        if (ret != 0)
            return;

        d->weight_wait_compute_index = d->weight_last_compute_index;
        d->weight_last_compute_index = d->weight_buffers.empty() ? -1 : previous_compute_index;
        std::swap(previous_weights, d->weight_buffers);
        d->weight_upload_wait_semaphore = d->weight_wait_compute_index < 0 ? 0 : d->compute_upload_semaphores[d->weight_wait_compute_index];
    }

    VkMat src;
    int ret = d->record_resolve(_src, src);
    if (ret != 0)
        return;

    //     NCNN_LOGE("record_clone buffer to buffer");

    // create dst
    dst.create_like(src, opt.blob_vkallocator);
    if (dst.empty())
        return;

    // barrier device any @ compute to transfer-read @ compute
    if (src.data->access_flags & VK_ACCESS_TRANSFER_WRITE_BIT || src.data->stage_flags != VK_PIPELINE_STAGE_TRANSFER_BIT)
    {
        VkBufferMemoryBarrier barrier;
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = 0;
        barrier.srcAccessMask = src.data->access_flags;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = src.buffer();
        barrier.offset = src.data->offset;
        barrier.size = src.data->capacity;

        VkPipelineStageFlags src_stage = src.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdPipelineBarrier(d->compute_command_buffers.back(), src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }
        else
        {
            VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
            barriers[0] = barrier;

            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_buffer_barrers;
            r.command_buffer = d->compute_command_buffers.back();
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark device transfer-read @ transfer
        src.data->access_flags = VK_ACCESS_TRANSFER_READ_BIT;
        src.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    // mark device transfer-write @ transfer
    dst.data->access_flags = VK_ACCESS_TRANSFER_WRITE_BIT;
    dst.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;

    // record device to device
    {
        VkBufferCopy region;
        region.srcOffset = src.buffer_offset();
        region.dstOffset = dst.buffer_offset();
        region.size = std::min(src.buffer_capacity(), dst.buffer_capacity());

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdCopyBuffer(d->compute_command_buffers.back(), src.buffer(), dst.buffer(), 1, &region);
        }
        else
        {
            VkBufferCopy* regions = new VkBufferCopy[1];
            regions[0] = region;

            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_copy_buffer;
            r.command_buffer = d->compute_command_buffers.back();
            r.copy_buffer.src = src.buffer();
            r.copy_buffer.dst = dst.buffer();
            r.copy_buffer.region_count = 1;
            r.copy_buffer.regions = regions;
            d->delayed_records.push_back(r);
        }
    }
}

void VkCompute::record_clone(const VkImageMat& src, VkImageMat& dst, const Option& opt)
{
    //     NCNN_LOGE("record_clone image to image");

    // create dst
    dst.create_like(src, opt.blob_vkallocator);
    if (dst.empty())
        return;

    // image layout transform any @ any to transfer-src-optimal @ compute
    if (src.data->access_flags & VK_ACCESS_TRANSFER_WRITE_BIT || src.data->image_layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL || src.data->stage_flags != VK_PIPELINE_STAGE_TRANSFER_BIT)
    {
        VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = src.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[0].oldLayout = src.data->image_layout;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = src.image();
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = src.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdPipelineBarrier(d->compute_command_buffers.back(), src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_image_barrers;
            r.command_buffer = d->compute_command_buffers.back();
            r.image_barrers.src_stage = src_stage;
            r.image_barrers.dst_stage = dst_stage;
            r.image_barrers.barrier_count = 1;
            r.image_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark image transfer-src-optimal @ compute
        src.data->access_flags = VK_ACCESS_TRANSFER_READ_BIT;
        src.data->image_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        src.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    // image layout transform undefined @ null to transfer-dst-optimal @ compute
    {
        VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = 0;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = dst.image();
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdPipelineBarrier(d->compute_command_buffers.back(), src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_image_barrers;
            r.command_buffer = d->compute_command_buffers.back();
            r.image_barrers.src_stage = src_stage;
            r.image_barrers.dst_stage = dst_stage;
            r.image_barrers.barrier_count = 1;
            r.image_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark image transfer-dst-optimal @ compute
        dst.data->access_flags = VK_ACCESS_TRANSFER_WRITE_BIT;
        dst.data->image_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        dst.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    // record device to staging
    {
        VkImageCopy* regions = new VkImageCopy[1];
        regions[0].srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        regions[0].srcSubresource.mipLevel = 0;
        regions[0].srcSubresource.baseArrayLayer = 0;
        regions[0].srcSubresource.layerCount = 1;
        regions[0].srcOffset.x = 0;
        regions[0].srcOffset.y = 0;
        regions[0].srcOffset.z = 0;
        regions[0].dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        regions[0].dstSubresource.mipLevel = 0;
        regions[0].dstSubresource.baseArrayLayer = 0;
        regions[0].dstSubresource.layerCount = 1;
        regions[0].dstOffset.x = 0;
        regions[0].dstOffset.y = 0;
        regions[0].dstOffset.z = 0;
        regions[0].extent.width = src.data->width;
        regions[0].extent.height = src.data->height;
        regions[0].extent.depth = src.data->depth;

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdCopyImage(d->compute_command_buffers.back(), src.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, regions);
            delete[] regions;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_copy_image;
            r.command_buffer = d->compute_command_buffers.back();
            r.copy_image.src = src.image();
            r.copy_image.src_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            r.copy_image.dst = dst.image();
            r.copy_image.dst_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            r.copy_image.region_count = 1;
            r.copy_image.regions = regions;
            d->delayed_records.push_back(r);
        }
    }

    // image and imageview can not be destroyed until command execution ends
    NCNN_XADD(&src.data->command_refcount, 1);
    NCNN_XADD(&dst.data->command_refcount, 1);
    d->image_blocks_to_destroy.push_back(src.data);
    d->image_blocks_to_destroy.push_back(dst.data);
}

void VkCompute::record_clone(const VkMat& _src, VkImageMat& dst, const Option& opt)
{
    // keep the preceding weights live while resolving this operation
    std::vector<VkMat> previous_weights;
    if (d->use_weight_upload_online && is_deferred_buffer(_src.data))
    {
        const int previous_compute_index = d->compute_command_buffer_index;
        int ret = d->record_submit_compute();
        if (ret != 0)
            return;

        d->weight_wait_compute_index = d->weight_last_compute_index;
        d->weight_last_compute_index = d->weight_buffers.empty() ? -1 : previous_compute_index;
        std::swap(previous_weights, d->weight_buffers);
        d->weight_upload_wait_semaphore = d->weight_wait_compute_index < 0 ? 0 : d->compute_upload_semaphores[d->weight_wait_compute_index];
    }

    VkMat src;
    int ret = d->record_resolve(_src, src);
    if (ret != 0)
        return;

    //     NCNN_LOGE("record_clone buffer to image");

    // create dst
    dst.create_like(src, opt.blob_vkallocator);
    if (dst.empty())
        return;

    // barrier device any @ any to transfer-read @ compute
    if (src.data->access_flags & VK_ACCESS_SHADER_WRITE_BIT || src.data->stage_flags != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    {
        VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = src.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = src.buffer();
        barriers[0].offset = src.data->offset;
        barriers[0].size = src.data->capacity;

        VkPipelineStageFlags src_stage = src.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdPipelineBarrier(d->compute_command_buffers.back(), src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_buffer_barrers;
            r.command_buffer = d->compute_command_buffers.back();
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark device transfer-read @ compute
        src.data->access_flags = VK_ACCESS_TRANSFER_READ_BIT;
        src.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    // image layout transform undefined @ null to transfer-dst-optimal @ compute
    {
        VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = 0;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = dst.image();
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdPipelineBarrier(d->compute_command_buffers.back(), src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_image_barrers;
            r.command_buffer = d->compute_command_buffers.back();
            r.image_barrers.src_stage = src_stage;
            r.image_barrers.dst_stage = dst_stage;
            r.image_barrers.barrier_count = 1;
            r.image_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark image transfer-dst-optimal @ compute
        dst.data->access_flags = VK_ACCESS_TRANSFER_WRITE_BIT;
        dst.data->image_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        dst.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    // record device to image
    {
        int region_count;
        VkBufferImageCopy* regions;
        if (dst.elemsize * dst.w * dst.h % 16 == 0)
        {
            region_count = 1;
            regions = new VkBufferImageCopy[1];
            regions[0].bufferOffset = src.buffer_offset();
            regions[0].bufferRowLength = 0;
            regions[0].bufferImageHeight = 0;
            regions[0].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            regions[0].imageSubresource.mipLevel = 0;
            regions[0].imageSubresource.baseArrayLayer = 0;
            regions[0].imageSubresource.layerCount = 1;
            regions[0].imageOffset.x = 0;
            regions[0].imageOffset.y = 0;
            regions[0].imageOffset.z = 0;
            regions[0].imageExtent.width = dst.data->width;
            regions[0].imageExtent.height = dst.data->height;
            regions[0].imageExtent.depth = dst.data->depth;
        }
        else
        {
            region_count = dst.c;
            regions = new VkBufferImageCopy[region_count];
            for (int i = 0; i < region_count; i++)
            {
                regions[i].bufferOffset = src.buffer_offset() + src.cstep * src.elemsize * i;
                regions[i].bufferRowLength = 0;
                regions[i].bufferImageHeight = 0;
                regions[i].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                regions[i].imageSubresource.mipLevel = 0;
                regions[i].imageSubresource.baseArrayLayer = 0;
                regions[i].imageSubresource.layerCount = 1;
                regions[i].imageOffset.x = 0;
                regions[i].imageOffset.y = 0;
                regions[i].imageOffset.z = i;
                regions[i].imageExtent.width = dst.data->width;
                regions[i].imageExtent.height = dst.data->height;
                regions[i].imageExtent.depth = 1;
            }
        }

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdCopyBufferToImage(d->compute_command_buffers.back(), src.buffer(), dst.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region_count, regions);
            delete[] regions;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_copy_buffer_to_image;
            r.command_buffer = d->compute_command_buffers.back();
            r.copy_buffer_to_image.src = src.buffer();
            r.copy_buffer_to_image.dst = dst.image();
            r.copy_buffer_to_image.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            r.copy_buffer_to_image.region_count = region_count;
            r.copy_buffer_to_image.regions = regions;
            d->delayed_records.push_back(r);
        }
    }

    // image and imageview can not be destroyed until command execution ends
    NCNN_XADD(&dst.data->command_refcount, 1);
    d->image_blocks_to_destroy.push_back(dst.data);
}

void VkCompute::record_clone(const VkImageMat& src, VkMat& dst, const Option& opt)
{
    //     NCNN_LOGE("record_clone image to buffer");

    // create dst
    dst.create_like(src, opt.blob_vkallocator);
    if (dst.empty())
        return;

    // image layout transform any @ any to transfer-src-optimal @ compute
    if (src.data->access_flags & VK_ACCESS_TRANSFER_WRITE_BIT || src.data->image_layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL || src.data->stage_flags != VK_PIPELINE_STAGE_TRANSFER_BIT)
    {
        VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = src.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[0].oldLayout = src.data->image_layout;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = src.image();
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = src.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdPipelineBarrier(d->compute_command_buffers.back(), src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_image_barrers;
            r.command_buffer = d->compute_command_buffers.back();
            r.image_barrers.src_stage = src_stage;
            r.image_barrers.dst_stage = dst_stage;
            r.image_barrers.barrier_count = 1;
            r.image_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark image transfer-src-optimal @ compute
        src.data->access_flags = VK_ACCESS_TRANSFER_READ_BIT;
        src.data->image_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        src.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    {
        // barrier device any @ null to transfer-write @ compute

        // mark device transfer-write @ transfer
        dst.data->access_flags = VK_ACCESS_TRANSFER_WRITE_BIT;
        dst.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    // record image to device
    {
        int region_count;
        VkBufferImageCopy* regions;
        if (src.elemsize * src.w * src.h % 16 == 0)
        {
            region_count = 1;
            regions = new VkBufferImageCopy[1];
            regions[0].bufferOffset = dst.buffer_offset();
            regions[0].bufferRowLength = 0;
            regions[0].bufferImageHeight = 0;
            regions[0].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            regions[0].imageSubresource.mipLevel = 0;
            regions[0].imageSubresource.baseArrayLayer = 0;
            regions[0].imageSubresource.layerCount = 1;
            regions[0].imageOffset.x = 0;
            regions[0].imageOffset.y = 0;
            regions[0].imageOffset.z = 0;
            regions[0].imageExtent.width = src.data->width;
            regions[0].imageExtent.height = src.data->height;
            regions[0].imageExtent.depth = src.data->depth;
        }
        else
        {
            region_count = src.c;
            regions = new VkBufferImageCopy[region_count];
            for (int i = 0; i < region_count; i++)
            {
                regions[i].bufferOffset = dst.buffer_offset() + dst.cstep * dst.elemsize * i;
                regions[i].bufferRowLength = 0;
                regions[i].bufferImageHeight = 0;
                regions[i].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                regions[i].imageSubresource.mipLevel = 0;
                regions[i].imageSubresource.baseArrayLayer = 0;
                regions[i].imageSubresource.layerCount = 1;
                regions[i].imageOffset.x = 0;
                regions[i].imageOffset.y = 0;
                regions[i].imageOffset.z = i;
                regions[i].imageExtent.width = src.data->width;
                regions[i].imageExtent.height = src.data->height;
                regions[i].imageExtent.depth = 1;
            }
        }

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdCopyImageToBuffer(d->compute_command_buffers.back(), src.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.buffer(), region_count, regions);
            delete[] regions;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_copy_image_to_buffer;
            r.command_buffer = d->compute_command_buffers.back();
            r.copy_image_to_buffer.src = src.image();
            r.copy_image_to_buffer.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            r.copy_image_to_buffer.dst = dst.buffer();
            r.copy_image_to_buffer.region_count = region_count;
            r.copy_image_to_buffer.regions = regions;
            d->delayed_records.push_back(r);
        }
    }

    // image and imageview can not be destroyed until command execution ends
    NCNN_XADD(&src.data->command_refcount, 1);
    d->image_blocks_to_destroy.push_back(src.data);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& bindings, const std::vector<vk_constant_type>& constants, const VkMat& dispatcher)
{
    record_pipeline(pipeline, bindings, std::vector<VkImageMat>(), constants, dispatcher);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkImageMat>& bindings, const std::vector<vk_constant_type>& constants, const VkImageMat& dispatcher)
{
    record_pipeline(pipeline, std::vector<VkMat>(), bindings, constants, dispatcher);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const VkMat& dispatcher)
{
    Mat dispatcher_mat(dispatcher.w, dispatcher.h, dispatcher.d, dispatcher.c, (void*)0);

    record_pipeline(pipeline, buffer_bindings, image_bindings, constants, dispatcher_mat);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const VkImageMat& dispatcher)
{
    Mat dispatcher_mat(dispatcher.w, dispatcher.h, dispatcher.d, dispatcher.c, (void*)0);

    record_pipeline(pipeline, buffer_bindings, image_bindings, constants, dispatcher_mat);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& _buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const Mat& dispatcher)
{
    //     NCNN_LOGE("record_pipeline %p", pipeline);

    std::vector<VkMat> buffer_bindings = _buffer_bindings;

    // keep the preceding weights live while resolving this operation
    std::vector<VkMat> previous_weights;
    bool first_weight = true;

    for (size_t i = 0; i < buffer_bindings.size(); i++)
    {
        if (!is_deferred_buffer(buffer_bindings[i].data))
            continue;

        if (first_weight && d->use_weight_upload_online)
        {
            const int previous_compute_index = d->compute_command_buffer_index;
            int ret = d->record_submit_compute();
            if (ret != 0)
                return;

            d->weight_wait_compute_index = d->weight_last_compute_index;
            d->weight_last_compute_index = d->weight_buffers.empty() ? -1 : previous_compute_index;
            std::swap(previous_weights, d->weight_buffers);
            d->weight_upload_wait_semaphore = d->weight_wait_compute_index < 0 ? 0 : d->compute_upload_semaphores[d->weight_wait_compute_index];
            first_weight = false;
        }

        int ret = d->record_resolve(_buffer_bindings[i], buffer_bindings[i]);
        if (ret != 0)
            return;

        // resolve each allocation once, preserving each binding view
        for (size_t j = i + 1; j < buffer_bindings.size(); j++)
        {
            if (_buffer_bindings[j].data != _buffer_bindings[i].data)
                continue;

            // share storage without changing the binding shape or offset
            VkMat resolved = buffer_bindings[i];
            std::swap(buffer_bindings[j].data, resolved.data);
            std::swap(buffer_bindings[j].refcount, resolved.refcount);
            std::swap(buffer_bindings[j].allocator, resolved.allocator);
        }
    }

    const int buffer_binding_count = (int)buffer_bindings.size();
    const int image_binding_count = (int)image_bindings.size();
    const int constant_count = (int)constants.size();

    const int binding_count = buffer_binding_count + image_binding_count;
    const ShaderInfo& shader_info = pipeline->shader_info();

    if (binding_count != shader_info.binding_count)
    {
        NCNN_LOGE("binding_count not match, expect %d but got %d + %d", shader_info.binding_count, buffer_binding_count, image_binding_count);
    }

    if (constant_count != shader_info.push_constant_count)
    {
        NCNN_LOGE("push_constant_count not match, expect %d but got %d", shader_info.push_constant_count, constant_count);
    }

    int buffer_index = 0;
    int image_index = 0;
    for (int i = 0; i < binding_count; i++)
    {
        int binding_type = shader_info.binding_types[i];

        if (binding_type == 1)
        {
            const VkMat& binding = buffer_bindings[buffer_index];
            buffer_index++;
            // Empty descriptors use device-owned placeholders below. They are
            // not accessed by this dispatch and have no mutable command state.
            if (binding.empty())
                continue;

            //             NCNN_LOGE("binding #%d buffer = %d %d %d %d @ %lu %d = %p +%ld ~%ld", i, binding.dims, binding.w, binding.h, binding.c, binding.elemsize, binding.elempack, binding.buffer(), binding.buffer_offset(), binding.buffer_capacity());

            barrier_readwrite(binding);
        }
        else if (binding_type == 2)
        {
            const VkImageMat& binding = image_bindings[image_index];
            image_index++;
            if (binding.empty())
                continue;

            //             NCNN_LOGE("binding #%d image = %d %d %d %d @ %lu %d = %p +%ld ~%ld %p", i, binding.dims, binding.w, binding.h, binding.c, binding.elemsize, binding.elempack, binding.image(), binding.data->bind_offset, binding.data->bind_capacity, binding.imageview());

            barrier_readwrite(binding);

            // image and imageview can not be destroyed until command execution ends
            NCNN_XADD(&binding.data->command_refcount, 1);
            d->image_blocks_to_destroy.push_back(binding.data);
        }
        else // if (binding_type == 3)
        {
            const VkImageMat& binding = image_bindings[image_index];
            image_index++;
            if (binding.empty())
                continue;

            //             NCNN_LOGE("binding #%d sampler = %d %d %d %d @ %lu %d = %p +%ld ~%ld %p", i, binding.dims, binding.w, binding.h, binding.c, binding.elemsize, binding.elempack, binding.image(), binding.data->bind_offset, binding.data->bind_capacity, binding.imageview());

            // if the same image used for both storage image and combined image sampler
            // only apply image layout transition to general
            bool image_read_write = false;
            for (int j = 0; j < image_binding_count; j++)
            {
                if (shader_info.binding_types[j] == 2 && binding.data == image_bindings[j].data)
                {
                    // the same image is used as storage image, skip it
                    image_read_write = true;
                    break;
                }
            }
            if (image_read_write)
                continue;

            barrier_readonly(binding);

            // image and imageview can not be destroyed until command execution ends
            NCNN_XADD(&binding.data->command_refcount, 1);
            d->image_blocks_to_destroy.push_back(binding.data);
        }
    }

    // record bind pipeline
    {
        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdBindPipeline(d->compute_command_buffers.back(), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
#if NCNN_COVERAGE
            vkdev->bind_shader_coverage(d->compute_command_buffers.back(), pipeline->pipeline_layout());
#endif
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_bind_pipeline;
            r.command_buffer = d->compute_command_buffers.back();
            r.bind_pipeline.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
            r.bind_pipeline.pipeline = pipeline->pipeline();
#if NCNN_COVERAGE
            r.bind_pipeline.pipeline_layout = pipeline->pipeline_layout();
#endif
            d->delayed_records.push_back(r);
        }
    }

    // record update bindings
    if (binding_count > 0)
    {
        std::vector<unsigned char> descriptorInfos;
        {
            descriptorInfos.resize(sizeof(VkDescriptorBufferInfo) * buffer_binding_count + sizeof(VkDescriptorImageInfo) * image_binding_count);

            unsigned char* p_descriptorInfos = descriptorInfos.data();
            int descriptorBufferInfo_index = 0;
            int descriptorImageInfo_index = 0;
            for (int i = 0; i < binding_count; i++)
            {
                int binding_type = shader_info.binding_types[i];

                if (binding_type == 1)
                {
                    const VkMat& binding = buffer_bindings[descriptorBufferInfo_index].empty() ? vkdev->get_dummy_buffer() : buffer_bindings[descriptorBufferInfo_index];
                    descriptorBufferInfo_index++;

                    VkDescriptorBufferInfo descriptorBufferInfo;
                    descriptorBufferInfo.buffer = binding.buffer();
                    descriptorBufferInfo.offset = binding.buffer_offset();
                    descriptorBufferInfo.range = binding.buffer_capacity();

                    memcpy(p_descriptorInfos, &descriptorBufferInfo, sizeof(VkDescriptorBufferInfo));
                    p_descriptorInfos += sizeof(VkDescriptorBufferInfo);
                }
                else //if (binding_type == 2 || binding_type == 3)
                {
                    const VkImageMat& binding = image_bindings[descriptorImageInfo_index].empty() ? vkdev->get_dummy_image() : image_bindings[descriptorImageInfo_index];
                    descriptorImageInfo_index++;

                    // we always use immutable nearest sampler set in descroptor layout during pipeline creation
                    VkDescriptorImageInfo descriptorImageInfo;
                    descriptorImageInfo.sampler = 0;
                    descriptorImageInfo.imageView = binding.imageview();
                    descriptorImageInfo.imageLayout = binding.data->image_layout;

                    memcpy(p_descriptorInfos, &descriptorImageInfo, sizeof(VkDescriptorImageInfo));
                    p_descriptorInfos += sizeof(VkDescriptorImageInfo);
                }
            }
        }

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkdev->vkCmdPushDescriptorSetWithTemplateKHR(d->compute_command_buffers.back(), pipeline->descriptor_update_template(), pipeline->pipeline_layout(), 0, descriptorInfos.data());
        }
        else
        {
            // create new descriptor_pool and descriptorset
            VkDescriptorPool descriptor_pool;
            {
                int buffer_binding_count = 0;
                int image_binding_count = 0;
                int sampler_binding_count = 0;
                for (int i = 0; i < binding_count; i++)
                {
                    if (shader_info.binding_types[i] == 1)
                        buffer_binding_count++;
                    else if (shader_info.binding_types[i] == 2)
                        image_binding_count++;
                    else if (shader_info.binding_types[i] == 3)
                        sampler_binding_count++;
                }

                // VUID-VkDescriptorPoolSize-descriptorCount-00302: each
                // descriptorCount must be > 0. Skip unused descriptor types
                // (common on buffer-only compute shaders; Mesa v3dv asserts).
                // See https://github.com/Tencent/ncnn/issues/6951
                VkDescriptorPoolSize poolSizes[3];
                uint32_t pool_size_count = 0;
                if (buffer_binding_count > 0)
                {
                    poolSizes[pool_size_count].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    poolSizes[pool_size_count].descriptorCount = buffer_binding_count;
                    pool_size_count++;
                }
                if (image_binding_count > 0)
                {
                    poolSizes[pool_size_count].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    poolSizes[pool_size_count].descriptorCount = image_binding_count;
                    pool_size_count++;
                }
                if (sampler_binding_count > 0)
                {
                    poolSizes[pool_size_count].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    poolSizes[pool_size_count].descriptorCount = sampler_binding_count;
                    pool_size_count++;
                }

                VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
                descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                descriptorPoolCreateInfo.pNext = 0;
                descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
                descriptorPoolCreateInfo.maxSets = 1;
                descriptorPoolCreateInfo.poolSizeCount = pool_size_count;
                descriptorPoolCreateInfo.pPoolSizes = poolSizes;

                VkResult ret = vkCreateDescriptorPool(vkdev->vkdevice(), &descriptorPoolCreateInfo, 0, &descriptor_pool);
                if (ret != VK_SUCCESS)
                {
                    NCNN_LOGE("vkCreateDescriptorPool failed %d", ret);
                    return;
                }
            }

            VkDescriptorSet descriptorset;
            {
                VkDescriptorSetLayout descriptorset_layout = pipeline->descriptorset_layout();

                VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
                descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                descriptorSetAllocateInfo.pNext = 0;
                descriptorSetAllocateInfo.descriptorPool = descriptor_pool;
                descriptorSetAllocateInfo.descriptorSetCount = 1;
                descriptorSetAllocateInfo.pSetLayouts = &descriptorset_layout;

                VkResult ret = vkAllocateDescriptorSets(vkdev->vkdevice(), &descriptorSetAllocateInfo, &descriptorset);
                if (ret != VK_SUCCESS)
                {
                    NCNN_LOGE("vkAllocateDescriptorSets failed %d", ret);
                    vkDestroyDescriptorPool(vkdev->vkdevice(), descriptor_pool, 0);
                    return;
                }
            }
            d->descriptor_pools.push_back(descriptor_pool);
            d->descriptorsets.push_back(descriptorset);

            if (vkdev->info.support_VK_KHR_descriptor_update_template())
            {
                vkdev->vkUpdateDescriptorSetWithTemplateKHR(vkdev->vkdevice(), descriptorset, pipeline->descriptor_update_template(), descriptorInfos.data());
            }
            else
            {
                std::vector<VkWriteDescriptorSet> writeDescriptorSets(binding_count);
                const unsigned char* p_descriptorInfos = descriptorInfos.data();
                for (int i = 0; i < binding_count; i++)
                {
                    int binding_type = shader_info.binding_types[i];

                    writeDescriptorSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    writeDescriptorSets[i].pNext = 0;
                    writeDescriptorSets[i].dstSet = descriptorset;
                    writeDescriptorSets[i].dstBinding = i;
                    writeDescriptorSets[i].dstArrayElement = 0;
                    writeDescriptorSets[i].descriptorCount = 1;
                    writeDescriptorSets[i].pTexelBufferView = 0;

                    if (binding_type == 1)
                    {
                        writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        writeDescriptorSets[i].pImageInfo = 0;
                        writeDescriptorSets[i].pBufferInfo = (const VkDescriptorBufferInfo*)p_descriptorInfos;

                        p_descriptorInfos += sizeof(VkDescriptorBufferInfo);
                    }
                    else // if (binding_type == 2 || binding_type == 3)
                    {
                        writeDescriptorSets[i].descriptorType = binding_type == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                        writeDescriptorSets[i].pImageInfo = (const VkDescriptorImageInfo*)p_descriptorInfos;
                        writeDescriptorSets[i].pBufferInfo = 0;

                        p_descriptorInfos += sizeof(VkDescriptorImageInfo);
                    }
                }

                vkUpdateDescriptorSets(vkdev->vkdevice(), binding_count, writeDescriptorSets.data(), 0, 0);
            }

            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_bind_descriptorsets;
            r.command_buffer = d->compute_command_buffers.back();
            r.bind_descriptorsets.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
            r.bind_descriptorsets.pipeline_layout = pipeline->pipeline_layout();
            r.bind_descriptorsets.descriptorset_count = 1;
            r.bind_descriptorsets.descriptorset_offset = d->descriptorsets.size() - 1;
            d->delayed_records.push_back(r);
        }
    }

    // record push constants
    if (constant_count > 0)
    {
        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdPushConstants(d->compute_command_buffers.back(), pipeline->pipeline_layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, constant_count * sizeof(vk_constant_type), constants.data());
        }
        else
        {
            uint32_t size = constant_count * sizeof(vk_constant_type);
            unsigned char* constant_values = new unsigned char[size];
            memcpy(constant_values, constants.data(), size);

            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_push_constants;
            r.command_buffer = d->compute_command_buffers.back();
            r.push_constants.pipeline_layout = pipeline->pipeline_layout();
            r.push_constants.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
            r.push_constants.size = size;
            r.push_constants.values = constant_values;
            d->delayed_records.push_back(r);
        }
    }

    // record dispatch
    {
        uint32_t group_count_x = (dispatcher.w + pipeline->local_size_x() - 1) / pipeline->local_size_x();
        uint32_t group_count_y = (dispatcher.h * (dispatcher.d ? dispatcher.d : 1) + pipeline->local_size_y() - 1) / pipeline->local_size_y();
        uint32_t group_count_z = (dispatcher.c + pipeline->local_size_z() - 1) / pipeline->local_size_z();

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdDispatch(d->compute_command_buffers.back(), group_count_x, group_count_y, group_count_z);
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_dispatch;
            r.command_buffer = d->compute_command_buffers.back();
            r.dispatch.group_count_x = group_count_x;
            r.dispatch.group_count_y = group_count_y;
            r.dispatch.group_count_z = group_count_z;
            d->delayed_records.push_back(r);
        }

        d->pending_dispatch_total += (uint64_t)group_count_x * group_count_y * group_count_z;
    }
}

#if NCNN_BENCHMARK
void VkCompute::record_write_timestamp(uint32_t query)
{
    if (vkdev->info.support_VK_KHR_push_descriptor())
    {
        if (d->query_pool)
        {
            vkCmdWriteTimestamp(d->compute_command_buffers.back(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, d->query_pool, query);
        }
    }
    else
    {
        VkComputePrivate::record r;
        r.type = VkComputePrivate::record::TYPE_write_timestamp;
        r.command_buffer = d->compute_command_buffers.back();
        r.write_timestamp.query = query;
        d->delayed_records.push_back(r);
    }
}
#endif // NCNN_BENCHMARK

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 26
void VkCompute::record_import_android_hardware_buffer(const ImportAndroidHardwareBufferPipeline* pipeline, const VkImageMat& src, const VkMat& dst)
{
    // image layout transform undefined @ null to general @ compute
    {
        VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = 0;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = src.image();
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdPipelineBarrier(d->compute_command_buffers.back(), src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_image_barrers;
            r.command_buffer = d->compute_command_buffers.back();
            r.image_barrers.src_stage = src_stage;
            r.image_barrers.dst_stage = dst_stage;
            r.image_barrers.barrier_count = 1;
            r.image_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }
    }

    VkDescriptorImageInfo descriptorImageInfo;
    descriptorImageInfo.sampler = pipeline->sampler;
    descriptorImageInfo.imageView = src.imageview();
    descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo descriptorBufferInfo;
    descriptorBufferInfo.buffer = dst.buffer();
    descriptorBufferInfo.offset = dst.buffer_offset();
    descriptorBufferInfo.range = dst.total() * dst.elemsize;

    struct ImportAndroidHardwareBufferDescriptorInfo
    {
        VkDescriptorImageInfo imageInfo;
        VkDescriptorBufferInfo bufferInfo;
        VkDescriptorBufferInfo buffer4Info;
    };
    ImportAndroidHardwareBufferDescriptorInfo info;
    info.imageInfo = descriptorImageInfo;
    info.bufferInfo = descriptorBufferInfo;
    info.buffer4Info = descriptorBufferInfo;

    const uint32_t group_count_x = (dst.w + pipeline->local_size_x() - 1) / pipeline->local_size_x();
    const uint32_t group_count_y = (dst.h + pipeline->local_size_y() - 1) / pipeline->local_size_y();
    const uint32_t group_count_z = (dst.c + pipeline->local_size_z() - 1) / pipeline->local_size_z();
    if (vkdev->info.support_VK_KHR_push_descriptor())
    {
        vkCmdBindPipeline(d->compute_command_buffers.back(), VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
#if NCNN_COVERAGE
        vkdev->bind_shader_coverage(d->compute_command_buffers.back(), pipeline->pipeline_layout());
#endif
        vkdev->vkCmdPushDescriptorSetWithTemplateKHR(d->compute_command_buffers.back(), pipeline->descriptor_update_template(), pipeline->pipeline_layout(), 0, &info);
        vkCmdDispatch(d->compute_command_buffers.back(), group_count_x, group_count_y, group_count_z);
    }
    else
    {
        VkComputePrivate::record r;
        r.type = VkComputePrivate::record::TYPE_bind_pipeline;
        r.command_buffer = d->compute_command_buffers.back();
        r.bind_pipeline.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
        r.bind_pipeline.pipeline = pipeline->pipeline();
#if NCNN_COVERAGE
        r.bind_pipeline.pipeline_layout = pipeline->pipeline_layout();
#endif
        d->delayed_records.push_back(r);

        const int binding_types[3] = {3, 1, 1};
        // create new descriptor_pool and descriptorset
        VkDescriptorPool descriptor_pool;
        {
            int buffer_binding_count = 0;
            int image_binding_count = 0;
            int sampler_binding_count = 0;
            for (int i = 0; i < 3; i++)
            {
                if (binding_types[i] == 1)
                    buffer_binding_count++;
                else if (binding_types[i] == 2)
                    image_binding_count++;
                else if (binding_types[i] == 3)
                    sampler_binding_count++;
            }

            // VUID-VkDescriptorPoolSize-descriptorCount-00302: each
            // descriptorCount must be > 0. Skip unused descriptor types
            // (common on buffer-only compute shaders; Mesa v3dv asserts).
            // See https://github.com/Tencent/ncnn/issues/6951
            VkDescriptorPoolSize poolSizes[3];
            uint32_t pool_size_count = 0;
            if (buffer_binding_count > 0)
            {
                poolSizes[pool_size_count].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                poolSizes[pool_size_count].descriptorCount = buffer_binding_count;
                pool_size_count++;
            }
            if (image_binding_count > 0)
            {
                poolSizes[pool_size_count].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                poolSizes[pool_size_count].descriptorCount = image_binding_count;
                pool_size_count++;
            }
            if (sampler_binding_count > 0)
            {
                poolSizes[pool_size_count].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                poolSizes[pool_size_count].descriptorCount = sampler_binding_count;
                pool_size_count++;
            }

            VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
            descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            descriptorPoolCreateInfo.pNext = 0;
            descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
            descriptorPoolCreateInfo.maxSets = 1;
            descriptorPoolCreateInfo.poolSizeCount = pool_size_count;
            descriptorPoolCreateInfo.pPoolSizes = poolSizes;

            VkResult ret = vkCreateDescriptorPool(vkdev->vkdevice(), &descriptorPoolCreateInfo, 0, &descriptor_pool);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkCreateDescriptorPool failed %d", ret);
                return;
            }
        }

        VkDescriptorSet descriptorset;
        {
            VkDescriptorSetLayout descriptorset_layout = pipeline->descriptorset_layout();

            VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
            descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            descriptorSetAllocateInfo.pNext = 0;
            descriptorSetAllocateInfo.descriptorPool = descriptor_pool;
            descriptorSetAllocateInfo.descriptorSetCount = 1;
            descriptorSetAllocateInfo.pSetLayouts = &descriptorset_layout;

            VkResult ret = vkAllocateDescriptorSets(vkdev->vkdevice(), &descriptorSetAllocateInfo, &descriptorset);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkAllocateDescriptorSets failed %d", ret);
                vkDestroyDescriptorPool(vkdev->vkdevice(), descriptor_pool, 0);
                return;
            }
        }
        d->descriptor_pools.push_back(descriptor_pool);
        d->descriptorsets.push_back(descriptorset);

        if (vkdev->info.support_VK_KHR_descriptor_update_template())
        {
            vkdev->vkUpdateDescriptorSetWithTemplateKHR(vkdev->vkdevice(), descriptorset, pipeline->descriptor_update_template(), (const unsigned char*)&info);
        }
        else
        {
            std::vector<VkWriteDescriptorSet> writeDescriptorSets(3);
            const unsigned char* p_descriptorInfos = (const unsigned char*)&info;
            for (int i = 0; i < 3; i++)
            {
                int binding_type = binding_types[i];

                writeDescriptorSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeDescriptorSets[i].pNext = 0;
                writeDescriptorSets[i].dstSet = descriptorset;
                writeDescriptorSets[i].dstBinding = i;
                writeDescriptorSets[i].dstArrayElement = 0;
                writeDescriptorSets[i].descriptorCount = 1;
                writeDescriptorSets[i].pTexelBufferView = 0;

                if (binding_type == 1)
                {
                    writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    writeDescriptorSets[i].pImageInfo = 0;
                    writeDescriptorSets[i].pBufferInfo = (const VkDescriptorBufferInfo*)p_descriptorInfos;

                    p_descriptorInfos += sizeof(VkDescriptorBufferInfo);
                }
                else // if (binding_type == 2 || binding_type == 3)
                {
                    writeDescriptorSets[i].descriptorType = binding_type == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    writeDescriptorSets[i].pImageInfo = (const VkDescriptorImageInfo*)p_descriptorInfos;
                    writeDescriptorSets[i].pBufferInfo = 0;

                    p_descriptorInfos += sizeof(VkDescriptorImageInfo);
                }
            }

            vkUpdateDescriptorSets(vkdev->vkdevice(), 3, writeDescriptorSets.data(), 0, 0);
        }

        r.type = VkComputePrivate::record::TYPE_bind_descriptorsets;
        r.bind_descriptorsets.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
        r.bind_descriptorsets.pipeline_layout = pipeline->pipeline_layout();
        r.bind_descriptorsets.descriptorset_count = 1;
        r.bind_descriptorsets.descriptorset_offset = d->descriptorsets.size() - 1;
        d->delayed_records.push_back(r);

        r.type = VkComputePrivate::record::TYPE_dispatch;
        r.dispatch.group_count_x = group_count_x;
        r.dispatch.group_count_y = group_count_y;
        r.dispatch.group_count_z = group_count_z;
        d->delayed_records.push_back(r);
    }
}
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API

int VkCompute::submit_and_wait()
{
    //     NCNN_LOGE("submit_and_wait");

    if (d->executing)
        return -1;
    for (size_t i = 0; i < d->upload_staging_buffers.size(); i++)
    {
        const VkMat& staging = d->upload_staging_buffers[i];
        int ret = staging.allocator->flush(staging.data);
        if (ret != 0)
            return ret;
    }
    int ret = 0;
    if (d->use_weight_upload_online)
    {
        ret = d->record_submit_compute();
        if (ret != 0)
            return ret;

        d->executing = true;
        ret = d->submit_records();
    }
    else
    {
        if (!vkdev->info.support_VK_KHR_push_descriptor())
        {
            d->flush_records();
        }

        // end command buffer
        {
            int ret = d->end_command_buffer();
            if (ret != 0)
                return ret;
        }

        // acquire queue and reclaim on return
        VkQueue compute_queue = vkdev->acquire_queue(vkdev->info.compute_queue_family_index());
        if (compute_queue == 0)
        {
            NCNN_LOGE("out of compute queue");
            return -1;
        }

        // submit compute
        {
            VkSubmitInfo submitInfo;
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = 0;
            submitInfo.waitSemaphoreCount = 0;
            submitInfo.pWaitSemaphores = 0;
            submitInfo.pWaitDstStageMask = 0;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &d->compute_command_buffers[0];
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = 0;

            VkResult ret = vkQueueSubmit(compute_queue, 1, &submitInfo, d->compute_command_fences[0]);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkQueueSubmit failed %d", ret);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
                return ret;
            }
            d->compute_command_pending[0] = true;
            d->executing = true;
        }

        vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
    }

    // wait
    {
        ret = d->finish_execution(ret);
        if (ret != 0)
            return ret;
    }

    // handle deferred post records
    for (size_t i = 0; i < d->delayed_records.size(); i++)
    {
        const VkComputePrivate::record& r = d->delayed_records[i];

        switch (r.type)
        {
        case VkComputePrivate::record::TYPE_post_download:
        {
            const VkMat& src = d->download_post_buffers[r.post_download.download_post_buffer_mat_offset];
            Mat& dst = d->download_post_mats_fp16[r.post_download.download_post_mat_fp16_offset];

            // NCNN_LOGE("post_download  %p +%d ~%d  -> %p", src.buffer(), src.buffer_offset(), src.buffer_capacity(), dst.data);

            if (src.allocator->invalidate(src.data))
                return -1;
            for (int b = 0; b < dst.n; b++)
            {
                const VkMat src_b = src.batch(b);
                Mat dst_b = dst.batch(b);
                memcpy(dst_b.data, src_b.mapped_ptr(), dst_b.total() * dst_b.elemsize);
            }
            break;
        }
        case VkComputePrivate::record::TYPE_post_cast_float16_to_float32:
        {
            // NCNN_LOGE("post_cast_float16_to_float32");

            const Mat& src = d->download_post_mats_fp16[r.post_cast_float16_to_float32.download_post_mat_fp16_offset];
            Mat& dst = d->download_post_mats[r.post_cast_float16_to_float32.download_post_mat_offset];

            Option opt;
            opt.num_threads = r.post_cast_float16_to_float32.num_threads;
            opt.blob_allocator = dst.allocator;
            ncnn::cast_float16_to_float32(src, dst, opt);
            break;
        }
        case VkComputePrivate::record::TYPE_post_cast_bfloat16_to_float32:
        {
            // NCNN_LOGE("post_cast_bfloat16_to_float32");

            const Mat& src = d->download_post_mats_fp16[r.post_cast_bfloat16_to_float32.download_post_mat_bf16_offset];
            Mat& dst = d->download_post_mats[r.post_cast_bfloat16_to_float32.download_post_mat_offset];

            Option opt;
            opt.num_threads = r.post_cast_bfloat16_to_float32.num_threads;
            opt.blob_allocator = dst.allocator;
            ncnn::cast_bfloat16_to_float32(src, dst, opt);
            break;
        }
        default:
            break;
        }
    }

#if NCNN_BENCHMARK
    d->query_results_ready = true;
#endif // NCNN_BENCHMARK
    d->clear_records();

    d->pending_dispatch_total = 0;

    return 0;
}

int VkCompute::reset()
{
    if (d->executing)
        return -1;
#if NCNN_BENCHMARK
    d->query_results_ready = false;
#endif // NCNN_BENCHMARK
    // reset command buffers before releasing upload and weight views
    {
        VkResult ret = vkResetCommandBuffer(d->compute_command_buffers[0], 0);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkResetCommandBuffer failed %d", ret);
            return -1;
        }
    }

    d->upload_staging_buffers.clear();
    d->download_post_buffers.clear();
    d->download_post_mats_fp16.clear();
    d->download_post_mats.clear();

    for (size_t i = 0; i < d->image_blocks_to_destroy.size(); i++)
    {
        VkImageMemory* ptr = d->image_blocks_to_destroy[i];

        int old_command_refcount = NCNN_XADD(&ptr->command_refcount, -1);
        if (ptr->refcount == 0 && old_command_refcount == 1)
        {
            // no userspace reference and we are the last command reference
            vkDestroyImageView(vkdev->vkdevice(), ptr->imageview, 0);
            vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);

            delete ptr;
        }
        else
        {
            // reference exists in user code or other command
        }
    }
    d->image_blocks_to_destroy.clear();

    if (!vkdev->info.support_VK_KHR_push_descriptor())
    {
        for (size_t i = 0; i < d->descriptorsets.size(); i++)
        {
            vkFreeDescriptorSets(vkdev->vkdevice(), d->descriptor_pools[i], 1, &d->descriptorsets[i]);
            vkDestroyDescriptorPool(vkdev->vkdevice(), d->descriptor_pools[i], 0);
        }
        d->descriptor_pools.clear();
        d->descriptorsets.clear();
    }

    d->clear_execution();
    d->clear_records();

    d->pending_dispatch_total = 0;

    // reset command buffer and fence
    {
        VkResult ret = vkResetFences(vkdev->vkdevice(), 1, &d->compute_command_fences[0]);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkResetFences failed %d", ret);
            return -1;
        }
    }

    return d->begin_command_buffer();
}

uint64_t VkCompute::pending_dispatch_total() const
{
    return d->pending_dispatch_total;
}

#if NCNN_BENCHMARK
int VkCompute::create_query_pool(uint32_t _query_count)
{
    d->query_count = _query_count;

    VkQueryPoolCreateInfo queryPoolCreateInfo;
    queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolCreateInfo.pNext = 0;
    queryPoolCreateInfo.flags = 0;
    queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolCreateInfo.queryCount = d->query_count;
    queryPoolCreateInfo.pipelineStatistics = 0;

    VkResult ret = vkCreateQueryPool(vkdev->vkdevice(), &queryPoolCreateInfo, 0, &d->query_pool);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateQueryPool failed %d", ret);
        return -1;
    }

    vkCmdResetQueryPool(d->compute_command_buffers.back(), d->query_pool, 0, d->query_count);

    return 0;
}

int VkCompute::get_query_pool_results(uint32_t first_query, uint32_t query_count, std::vector<uint64_t>& results)
{
    if (!d->query_results_ready || !d->query_pool || first_query > d->query_count || query_count > d->query_count - first_query)
        return -1;
    if (results.size() < first_query + query_count)
    {
        NCNN_LOGE("results not large enough");
        return -1;
    }

    VkResult ret = vkGetQueryPoolResults(vkdev->vkdevice(), d->query_pool, first_query, query_count,
                                         query_count * sizeof(uint64_t), results.data() + first_query, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    if (ret != VK_SUCCESS && ret != VK_NOT_READY)
    {
        NCNN_LOGE("vkGetQueryPoolResults failed %d", ret);
        return -1;
    }

    return 0;
}
#endif // NCNN_BENCHMARK

void VkCompute::barrier_readwrite(const VkMat& binding)
{
    if (is_deferred_buffer(binding.data))
        return;

    if (binding.data->access_flags & VK_ACCESS_SHADER_WRITE_BIT || binding.data->stage_flags != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    {
        // barrier device any @ compute/null to shader-readwrite @ compute
        VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = binding.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = binding.buffer();
        barriers[0].offset = binding.data->offset;
        barriers[0].size = binding.data->capacity;

        VkPipelineStageFlags src_stage = binding.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdPipelineBarrier(d->compute_command_buffers.back(), src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_buffer_barrers;
            r.command_buffer = d->compute_command_buffers.back();
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark device shader-readwrite @ compute
        binding.data->access_flags = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        binding.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
}

void VkCompute::barrier_readwrite(const VkImageMat& binding)
{
    if (binding.data->access_flags & VK_ACCESS_SHADER_WRITE_BIT || binding.data->image_layout != VK_IMAGE_LAYOUT_GENERAL || binding.data->stage_flags != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    {
        // image layout transform any @ any to shader-write @ compute
        VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = binding.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barriers[0].oldLayout = binding.data->image_layout;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = binding.image();
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = binding.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdPipelineBarrier(d->compute_command_buffers.back(), src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_image_barrers;
            r.command_buffer = d->compute_command_buffers.back();
            r.image_barrers.src_stage = src_stage;
            r.image_barrers.dst_stage = dst_stage;
            r.image_barrers.barrier_count = 1;
            r.image_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark image shader-write @ compute
        binding.data->access_flags = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        binding.data->image_layout = VK_IMAGE_LAYOUT_GENERAL;
        binding.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
}

void VkCompute::barrier_readonly(const VkImageMat& binding)
{
    if (binding.data->access_flags & VK_ACCESS_SHADER_WRITE_BIT || binding.data->image_layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL || binding.data->stage_flags != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    {
        // image layout transform any @ any to shader-readonly-optimal @ compute
        VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = binding.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[0].oldLayout = binding.data->image_layout;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = binding.image();
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = binding.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdPipelineBarrier(d->compute_command_buffers.back(), src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
            delete[] barriers;
        }
        else
        {
            VkComputePrivate::record r;
            r.type = VkComputePrivate::record::TYPE_image_barrers;
            r.command_buffer = d->compute_command_buffers.back();
            r.image_barrers.src_stage = src_stage;
            r.image_barrers.dst_stage = dst_stage;
            r.image_barrers.barrier_count = 1;
            r.image_barrers.barriers = barriers;
            d->delayed_records.push_back(r);
        }

        // mark image shader-readonly-optimal @ compute
        binding.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
        binding.data->image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        binding.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
}

int VkComputePrivate::wait_fence(VkFence fence)
{
    VkResult ret = vkGetFenceStatus(vkdev->vkdevice(), fence);
    if (ret == VK_SUCCESS)
        return 0;
    if (ret != VK_NOT_READY)
    {
        NCNN_LOGE("vkGetFenceStatus failed %d", ret);
        return ret;
    }
    ret = vkWaitForFences(vkdev->vkdevice(), 1, &fence, VK_TRUE, (uint64_t)-1);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkWaitForFences failed %d", ret);
        return ret;
    }
    return 0;
}

int VkComputePrivate::finish_execution(int result)
{
    if (!executing)
        return result;

    bool incomplete = false;

    for (size_t i = 0; i < compute_command_fences.size(); i++)
    {
        if (!compute_command_pending[i])
            continue;

        int ret = wait_fence(compute_command_fences[i]);
        if (ret != 0 && result != VK_ERROR_DEVICE_LOST)
            result = ret;
        if (ret == 0 || ret == VK_ERROR_DEVICE_LOST)
            compute_command_pending[i] = false;
        else
            incomplete = true;
    }

    for (size_t i = 0; i < upload_command_fences.size(); i++)
    {
        if (!upload_command_pending[i])
            continue;

        int ret = wait_fence(upload_command_fences[i]);
        if (ret != 0 && result != VK_ERROR_DEVICE_LOST)
            result = ret;
        if (ret == 0 || ret == VK_ERROR_DEVICE_LOST)
            upload_command_pending[i] = false;
        else
            incomplete = true;
    }

    if (incomplete)
        return result;

    executing = false;
    if (use_weight_upload_online)
        clear_execution();

    return result;
}

void VkComputePrivate::clear_execution()
{
    // release command resources after all compute and upload submissions complete
    for (size_t i = 1; i < compute_command_buffers.size(); i++)
    {
        vkDestroyFence(vkdev->vkdevice(), compute_command_fences[i], 0);
        if (compute_command_buffers[i])
            vkFreeCommandBuffers(vkdev->vkdevice(), compute_command_pool, 1, &compute_command_buffers[i]);
    }
    compute_command_buffers.resize(1);
    compute_command_fences.resize(1);
    compute_command_pending.resize(1);
    compute_command_buffer_index = -1;

    compute_upload_semaphores.assign(1, 0);
    compute_wait_semaphores.clear();

    for (size_t i = 0; i < upload_command_fences.size(); i++)
    {
        vkDestroyFence(vkdev->vkdevice(), upload_command_fences[i], 0);
        if (upload_command_buffers[i])
            vkFreeCommandBuffers(vkdev->vkdevice(), transfer_command_pool, 1, &upload_command_buffers[i]);
    }
    upload_command_buffers.clear();
    upload_command_fences.clear();
    upload_command_pending.clear();
    upload_compute_semaphores.clear();

    for (size_t i = 0; i < execution_semaphores.size(); i++)
    {
        vkDestroySemaphore(vkdev->vkdevice(), execution_semaphores[i], 0);
    }
    execution_semaphores.clear();

    weight_wait_compute_index = -1;
    weight_last_compute_index = -1;
    weight_upload_wait_semaphore = 0;

#if NCNN_BENCHMARK
    if (use_weight_upload_online && weight_start_time)
        NCNN_LOGE("weight upload read=%llu upload=%llu time=%.2fms", (unsigned long long)bytes_read, (unsigned long long)bytes_uploaded, get_current_time() - weight_start_time);
    weight_start_time = 0;
#endif // NCNN_BENCHMARK
}

void VkComputePrivate::clear_records()
{
    for (size_t i = 0; i < delayed_records.size(); i++)
    {
        const record& r = delayed_records[i];
        switch (r.type)
        {
        case record::TYPE_copy_buffer:
            delete[] r.copy_buffer.regions;
            break;
        case record::TYPE_copy_image:
            delete[] r.copy_image.regions;
            break;
        case record::TYPE_copy_buffer_to_image:
            delete[] r.copy_buffer_to_image.regions;
            break;
        case record::TYPE_copy_image_to_buffer:
            delete[] r.copy_image_to_buffer.regions;
            break;
        case record::TYPE_push_constants:
            delete[](const unsigned char*) r.push_constants.values;
            break;
        case record::TYPE_memory_barrers:
            delete[] r.memory_barrers.barriers;
            break;
        case record::TYPE_buffer_barrers:
            delete[] r.buffer_barrers.barriers;
            break;
        case record::TYPE_image_barrers:
            delete[] r.image_barrers.barriers;
            break;
        case record::TYPE_submit_compute:
        case record::TYPE_submit_upload:
            delete[] r.submit.wait_semaphores;
            break;
        default:
            break;
        }
    }
    delayed_records.clear();
    recorded_count = 0;

    weight_buffers.clear();

    upload_src_buffers.clear();
    for (size_t i = 0; i < upload_dst_buffers.size(); i++)
    {
        VkBufferMemory* ptr = upload_dst_buffers[i].data;
        upload_dst_buffers[i].release();
        delete ptr;
    }
    upload_dst_buffers.clear();
    weight_staging_buffer.release();
}

void VkComputePrivate::flush_records()
{
    for (; recorded_count < delayed_records.size(); recorded_count++)
    {
        const record& r = delayed_records[recorded_count];

        switch (r.type)
        {
        case VkComputePrivate::record::TYPE_copy_buffer:
        {
            vkCmdCopyBuffer(r.command_buffer, r.copy_buffer.src, r.copy_buffer.dst, r.copy_buffer.region_count, r.copy_buffer.regions);
            break;
        }
        case VkComputePrivate::record::TYPE_copy_image:
        {
            vkCmdCopyImage(r.command_buffer, r.copy_image.src, r.copy_image.src_layout, r.copy_image.dst, r.copy_image.dst_layout, r.copy_image.region_count, r.copy_image.regions);
            break;
        }
        case VkComputePrivate::record::TYPE_copy_buffer_to_image:
        {
            vkCmdCopyBufferToImage(r.command_buffer, r.copy_buffer_to_image.src, r.copy_buffer_to_image.dst, r.copy_buffer_to_image.layout, r.copy_buffer_to_image.region_count, r.copy_buffer_to_image.regions);
            break;
        }
        case VkComputePrivate::record::TYPE_copy_image_to_buffer:
        {
            vkCmdCopyImageToBuffer(r.command_buffer, r.copy_image_to_buffer.src, r.copy_image_to_buffer.layout, r.copy_image_to_buffer.dst, r.copy_image_to_buffer.region_count, r.copy_image_to_buffer.regions);
            break;
        }
        case VkComputePrivate::record::TYPE_bind_pipeline:
        {
            vkCmdBindPipeline(r.command_buffer, r.bind_pipeline.bind_point, r.bind_pipeline.pipeline);
#if NCNN_COVERAGE
            vkdev->bind_shader_coverage(r.command_buffer, r.bind_pipeline.pipeline_layout);
#endif
            break;
        }
        case VkComputePrivate::record::TYPE_bind_descriptorsets:
        {
            vkCmdBindDescriptorSets(r.command_buffer, r.bind_descriptorsets.bind_point, r.bind_descriptorsets.pipeline_layout, 0, r.bind_descriptorsets.descriptorset_count, &descriptorsets[r.bind_descriptorsets.descriptorset_offset], 0, 0);
            break;
        }
        case VkComputePrivate::record::TYPE_push_constants:
        {
            vkCmdPushConstants(r.command_buffer, r.push_constants.pipeline_layout, r.push_constants.stage_flags, 0, r.push_constants.size, r.push_constants.values);
            break;
        }
        case VkComputePrivate::record::TYPE_dispatch:
        {
            vkCmdDispatch(r.command_buffer, r.dispatch.group_count_x, r.dispatch.group_count_y, r.dispatch.group_count_z);
            break;
        }
        case VkComputePrivate::record::TYPE_memory_barrers:
        {
            vkCmdPipelineBarrier(r.command_buffer, r.memory_barrers.src_stage, r.memory_barrers.dst_stage, 0, r.memory_barrers.barrier_count, r.memory_barrers.barriers, 0, 0, 0, 0);
            break;
        }
        case VkComputePrivate::record::TYPE_buffer_barrers:
        {
            vkCmdPipelineBarrier(r.command_buffer, r.buffer_barrers.src_stage, r.buffer_barrers.dst_stage, 0, 0, 0, r.buffer_barrers.barrier_count, r.buffer_barrers.barriers, 0, 0);
            break;
        }
        case VkComputePrivate::record::TYPE_image_barrers:
        {
            vkCmdPipelineBarrier(r.command_buffer, r.image_barrers.src_stage, r.image_barrers.dst_stage, 0, 0, 0, 0, 0, r.image_barrers.barrier_count, r.image_barrers.barriers);
            break;
        }
#if NCNN_BENCHMARK
        case VkComputePrivate::record::TYPE_write_timestamp:
        {
            if (query_pool)
            {
                vkCmdWriteTimestamp(r.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, r.write_timestamp.query);
            }
            break;
        }
#endif // NCNN_BENCHMARK
        case VkComputePrivate::record::TYPE_host_copy:
        case VkComputePrivate::record::TYPE_wait_compute:
        case VkComputePrivate::record::TYPE_wait_upload:
        case VkComputePrivate::record::TYPE_submit_compute:
        case VkComputePrivate::record::TYPE_submit_upload:
        case VkComputePrivate::record::TYPE_post_download:
        case VkComputePrivate::record::TYPE_post_cast_float16_to_float32:
        case VkComputePrivate::record::TYPE_post_cast_bfloat16_to_float32:
        default:
            break;
        }
    }
}

int VkComputePrivate::record_submit_compute()
{
    if (compute_command_buffer_index < 0)
        return 0;

    // end command buffer
    {
        flush_records();
#if NCNN_COVERAGE
        vkdev->record_shader_coverage_barrier(compute_command_buffers[compute_command_buffer_index]);
#endif
        VkResult ret = vkEndCommandBuffer(compute_command_buffers[compute_command_buffer_index]);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkEndCommandBuffer failed %d", ret);
            return ret;
        }
    }

    // compute_upload_semaphores
    {
        VkSemaphoreCreateInfo semaphoreCreateInfo;
        semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        semaphoreCreateInfo.pNext = 0;
        semaphoreCreateInfo.flags = 0;

        VkResult ret = vkCreateSemaphore(vkdev->vkdevice(), &semaphoreCreateInfo, 0, &compute_upload_semaphores[compute_command_buffer_index]);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateSemaphore failed %d", ret);
            return ret;
        }
        execution_semaphores.push_back(compute_upload_semaphores[compute_command_buffer_index]);
    }

    // record submit compute
    {
        VkSemaphore* wait_semaphores = 0;
        if (!compute_wait_semaphores.empty())
        {
            wait_semaphores = new VkSemaphore[compute_wait_semaphores.size()];
            memcpy(wait_semaphores, compute_wait_semaphores.data(), compute_wait_semaphores.size() * sizeof(VkSemaphore));
        }

        record r;
        r.type = record::TYPE_submit_compute;
        r.command_buffer = compute_command_buffers[compute_command_buffer_index];
        r.submit.command_buffer_index = compute_command_buffer_index;
        r.submit.wait_semaphore_count = (uint32_t)compute_wait_semaphores.size();
        r.submit.wait_semaphores = wait_semaphores;
        r.submit.signal_semaphore = compute_upload_semaphores[compute_command_buffer_index];
        delayed_records.push_back(r);
    }
    compute_command_buffer_index = -1;
    compute_wait_semaphores.clear();

    return 0;
}

int VkComputePrivate::record_resolve(const VkMat& src, VkMat& dst)
{
    if (!is_deferred_buffer(src.data))
    {
        dst = src;
        return 0;
    }

    if (!use_weight_upload_online)
    {
        NCNN_LOGE("deferred weight requires set_use_weight_upload_online(true) before recording");
        return -1;
    }

    // private concurrent buffers allow reuse across compute and transfer queues
    if (!local_weight_vkallocator)
        local_weight_vkallocator = new VkBlobAllocator(vkdev, 16 * 1024 * 1024, true);

    // create dst for the complete allocation, including bytes outside the view
    const size_t size = src.data->capacity;
    {
        dst.release();
        dst.create(1, size, 1, local_weight_vkallocator);
        if (dst.empty() || dst.data->capacity < size)
            return -100;
        dst.dims = src.dims;
        dst.w = src.w;
        dst.h = src.h;
        dst.d = src.d;
        dst.c = src.c;
        dst.cstep = src.cstep;
        dst.elemsize = src.elemsize;
        dst.elempack = src.elempack;
#if NCNN_BATCH
        dst.n = src.n;
        dst.nstep = src.nstep;
        dst.offset = src.offset;
#endif
    }

    // full allocation views for upload; dst retains the consumer shape and offset
    VkMat dst_upload(1, dst.data, size, 1, local_weight_vkallocator);
    VkMat src_weight = src;
#if NCNN_BATCH
    src_weight.offset = 0;
#endif

    if (!vkdev->info.support_VK_KHR_push_descriptor())
    {
        flush_records();
    }

    VkAccessFlags upload_access = 0;
    VkPipelineStageFlags upload_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    // host writes wait for older compute; the preceding weight group is still live
    if ((vkdev->info.type() == 1 || vkdev->info.type() == 3) && dst_upload.mapped_ptr() && needs_host_copy(src_weight))
    {
        record r;
        r.type = record::TYPE_wait_compute;
        r.command_buffer = 0;
        r.wait.command_buffer_index = weight_wait_compute_index;
        delayed_records.push_back(r);

        upload_src_buffers.push_back(src_weight);
        VkBufferMemory* ptr = new VkBufferMemory(*dst_upload.data);
        upload_dst_buffers.push_back(VkMat(1, ptr, size, 1, dst_upload.allocator));

        r.type = record::TYPE_host_copy;
        r.host_copy.src_index = upload_src_buffers.size() - 1;
        r.host_copy.dst_index = upload_dst_buffers.size() - 1;
        r.host_copy.src_offset = 0;
        r.host_copy.size = size;
        delayed_records.push_back(r);

        upload_access = VK_ACCESS_HOST_WRITE_BIT;
        upload_stage = VK_PIPELINE_STAGE_HOST_BIT;
    }
    else
    {
        VkMat src_upload;
        size_t chunk_capacity = size;
        if (!needs_host_copy(src_weight))
        {
            // upload directly from host buffer
            upload_src_buffers.push_back(src_weight);
            src_upload = src_weight;
        }
        else
        {
            // stage file or host bytes in bounded chunks
            if (weight_staging_buffer.empty())
            {
                if (!local_staging_vkallocator)
                    local_staging_vkallocator = new VkStagingAllocator(vkdev);
                if (!local_staging_vkallocator)
                    return -1;
                weight_staging_buffer.create(1, (size_t)16 * 1024 * 1024, 1, local_staging_vkallocator);
                if (weight_staging_buffer.empty() || !weight_staging_buffer.mapped_ptr())
                    return -100;
            }
            src_upload = weight_staging_buffer;
            chunk_capacity = src_upload.buffer_capacity();
        }

        // record upload chunks
        for (size_t offset = 0; offset < size;)
        {
            const size_t chunk_size = std::min(chunk_capacity, size - offset);
            const size_t upload_size = alignSize(chunk_size, 4);
            const bool last = chunk_size == size - offset;
            if (needs_host_copy(src_weight))
            {
                if (!upload_command_buffers.empty())
                {
                    record r;
                    r.type = record::TYPE_wait_upload;
                    r.command_buffer = 0;
                    r.wait.command_buffer_index = (int)upload_command_buffers.size() - 1;
                    delayed_records.push_back(r);
                }

                upload_src_buffers.push_back(src_weight);
                VkBufferMemory* ptr = new VkBufferMemory(*src_upload.data);
                upload_dst_buffers.push_back(VkMat(1, ptr, src_upload.elemsize, 1, src_upload.allocator));

                record r;
                r.type = record::TYPE_host_copy;
                r.command_buffer = 0;
                r.host_copy.src_index = upload_src_buffers.size() - 1;
                r.host_copy.dst_index = upload_dst_buffers.size() - 1;
                r.host_copy.src_offset = offset;
                r.host_copy.size = chunk_size;
                delayed_records.push_back(r);
            }

            if (upload_command_buffers.size() >= (size_t)INT_MAX)
                return -1;

            const int command_buffer_index = (int)upload_command_buffers.size();
            upload_command_buffers.push_back(0);
            upload_command_fences.push_back(0);
            upload_compute_semaphores.push_back(0);
            upload_command_pending.push_back(false);

            // upload_command_buffers
            {
                VkCommandBufferAllocateInfo commandBufferAllocateInfo;
                commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                commandBufferAllocateInfo.pNext = 0;
                commandBufferAllocateInfo.commandPool = transfer_command_pool;
                commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                commandBufferAllocateInfo.commandBufferCount = 1;

                VkResult ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &commandBufferAllocateInfo, &upload_command_buffers[command_buffer_index]);
                if (ret != VK_SUCCESS)
                {
                    NCNN_LOGE("vkAllocateCommandBuffers failed %d", ret);
                    return -1;
                }
            }

            // upload_command_fences
            {
                VkFenceCreateInfo fenceCreateInfo;
                fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                fenceCreateInfo.pNext = 0;
                fenceCreateInfo.flags = 0;

                VkResult ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &upload_command_fences[command_buffer_index]);
                if (ret != VK_SUCCESS)
                {
                    NCNN_LOGE("vkCreateFence failed %d", ret);
                    return -1;
                }
            }

            // upload_compute_semaphores
            if (last)
            {
                VkSemaphoreCreateInfo semaphoreCreateInfo;
                semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
                semaphoreCreateInfo.pNext = 0;
                semaphoreCreateInfo.flags = 0;

                VkResult ret = vkCreateSemaphore(vkdev->vkdevice(), &semaphoreCreateInfo, 0, &upload_compute_semaphores[command_buffer_index]);
                if (ret != VK_SUCCESS)
                {
                    NCNN_LOGE("vkCreateSemaphore failed %d", ret);
                    return -1;
                }
                execution_semaphores.push_back(upload_compute_semaphores[command_buffer_index]);
            }

            // begin command buffer
            {
                VkCommandBufferBeginInfo commandBufferBeginInfo;
                commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                commandBufferBeginInfo.pNext = 0;
                commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                commandBufferBeginInfo.pInheritanceInfo = 0;

                VkResult ret = vkBeginCommandBuffer(upload_command_buffers[command_buffer_index], &commandBufferBeginInfo);
                if (ret != VK_SUCCESS)
                {
                    NCNN_LOGE("vkBeginCommandBuffer failed %d", ret);
                    return -1;
                }
            }

            // barrier upload host-write to transfer-read
            {
                VkBufferMemoryBarrier barrier;
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.pNext = 0;
                barrier.srcAccessMask = src_upload.mapped_ptr() ? VK_ACCESS_HOST_WRITE_BIT : VK_ACCESS_MEMORY_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.buffer = src_upload.buffer();
                barrier.offset = src_upload.buffer_offset();
                barrier.size = upload_size;

                VkPipelineStageFlags src_stage = src_upload.mapped_ptr() ? VK_PIPELINE_STAGE_HOST_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

                if (vkdev->info.support_VK_KHR_push_descriptor())
                {
                    vkCmdPipelineBarrier(upload_command_buffers[command_buffer_index], src_stage, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, 0, 1, &barrier, 0, 0);
                }
                else
                {
                    VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
                    barriers[0] = barrier;

                    record r;
                    r.type = record::TYPE_buffer_barrers;
                    r.command_buffer = upload_command_buffers[command_buffer_index];
                    r.buffer_barrers.src_stage = src_stage;
                    r.buffer_barrers.dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                    r.buffer_barrers.barrier_count = 1;
                    r.buffer_barrers.barriers = barriers;
                    delayed_records.push_back(r);
                }
            }

            // record upload to device
            {
                VkBufferCopy region;
                region.srcOffset = src_upload.buffer_offset();
                region.dstOffset = dst_upload.buffer_offset() + offset;
                region.size = upload_size;

                if (vkdev->info.support_VK_KHR_push_descriptor())
                {
                    vkCmdCopyBuffer(upload_command_buffers[command_buffer_index], src_upload.buffer(), dst_upload.buffer(), 1, &region);
                }
                else
                {
                    VkBufferCopy* regions = new VkBufferCopy[1];
                    regions[0] = region;

                    record r;
                    r.type = record::TYPE_copy_buffer;
                    r.command_buffer = upload_command_buffers[command_buffer_index];
                    r.copy_buffer.src = src_upload.buffer();
                    r.copy_buffer.dst = dst_upload.buffer();
                    r.copy_buffer.region_count = 1;
                    r.copy_buffer.regions = regions;
                    delayed_records.push_back(r);
                }
            }

            // barrier upload transfer-write before signaling compute
            if (last)
            {
                VkBufferMemoryBarrier barrier;
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.pNext = 0;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = 0;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.buffer = dst_upload.buffer();
                barrier.offset = dst_upload.buffer_offset();
                barrier.size = dst_upload.buffer_capacity();

                if (vkdev->info.support_VK_KHR_push_descriptor())
                {
                    vkCmdPipelineBarrier(upload_command_buffers[command_buffer_index], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, 0, 1, &barrier, 0, 0);
                }
                else
                {
                    VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
                    barriers[0] = barrier;

                    record r;
                    r.type = record::TYPE_buffer_barrers;
                    r.command_buffer = upload_command_buffers[command_buffer_index];
                    r.buffer_barrers.src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                    r.buffer_barrers.dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
                    r.buffer_barrers.barrier_count = 1;
                    r.buffer_barrers.barriers = barriers;
                    delayed_records.push_back(r);
                }
            }

            // end command buffer
            {
                flush_records();
                VkResult ret = vkEndCommandBuffer(upload_command_buffers[command_buffer_index]);
                if (ret != VK_SUCCESS)
                {
                    NCNN_LOGE("vkEndCommandBuffer failed %d", ret);
                    return -1;
                }
            }

            // record submit upload
            {
                VkSemaphore* wait_semaphores = 0;
                if (weight_upload_wait_semaphore)
                {
                    wait_semaphores = new VkSemaphore[1];
                    wait_semaphores[0] = weight_upload_wait_semaphore;
                }

                record r;
                r.type = record::TYPE_submit_upload;
                r.command_buffer = upload_command_buffers[command_buffer_index];
                r.submit.command_buffer_index = command_buffer_index;
                r.submit.wait_semaphore_count = weight_upload_wait_semaphore ? 1 : 0;
                r.submit.wait_semaphores = wait_semaphores;
                r.submit.signal_semaphore = last ? upload_compute_semaphores[command_buffer_index] : 0;
                delayed_records.push_back(r);
                weight_upload_wait_semaphore = 0;
                if (last)
                    compute_wait_semaphores.push_back(upload_compute_semaphores[command_buffer_index]);
            }
#if NCNN_BENCHMARK
            bytes_uploaded += upload_size;
#endif
            offset += chunk_size;
        }
    }

    if (compute_command_buffer_index < 0)
    {
        if (compute_command_buffers.size() >= (size_t)INT_MAX)
            return -1;

        const int command_buffer_index = (int)compute_command_buffers.size();
        compute_command_buffers.push_back(0);
        compute_command_fences.push_back(0);
        compute_upload_semaphores.push_back(0);
        compute_command_pending.push_back(false);

        // compute_command_buffers
        {
            VkCommandBufferAllocateInfo commandBufferAllocateInfo;
            commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            commandBufferAllocateInfo.pNext = 0;
            commandBufferAllocateInfo.commandPool = compute_command_pool;
            commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            commandBufferAllocateInfo.commandBufferCount = 1;

            VkResult ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &commandBufferAllocateInfo, &compute_command_buffers[command_buffer_index]);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkAllocateCommandBuffers failed %d", ret);
                return -1;
            }
        }

        // compute_command_fences
        {
            VkFenceCreateInfo fenceCreateInfo;
            fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceCreateInfo.pNext = 0;
            fenceCreateInfo.flags = 0;

            VkResult ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &compute_command_fences[command_buffer_index]);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkCreateFence failed %d", ret);
                return -1;
            }
        }

        // begin command buffer
        {
            VkCommandBufferBeginInfo commandBufferBeginInfo;
            commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            commandBufferBeginInfo.pNext = 0;
            commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            commandBufferBeginInfo.pInheritanceInfo = 0;

            VkResult ret = vkBeginCommandBuffer(compute_command_buffers[command_buffer_index], &commandBufferBeginInfo);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkBeginCommandBuffer failed %d", ret);
                return -1;
            }
        }

        compute_command_buffer_index = command_buffer_index;
    }

    // barrier uploaded weights to shader-read and transfer-read
    {
        VkBufferMemoryBarrier barrier;
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = 0;
        barrier.srcAccessMask = upload_access;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = dst_upload.buffer();
        barrier.offset = dst_upload.buffer_offset();
        barrier.size = dst_upload.buffer_capacity();

        VkPipelineStageFlags src_stage = upload_stage;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor())
        {
            vkCmdPipelineBarrier(compute_command_buffers[compute_command_buffer_index], src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }
        else
        {
            VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
            barriers[0] = barrier;

            record r;
            r.type = record::TYPE_buffer_barrers;
            r.command_buffer = compute_command_buffers[compute_command_buffer_index];
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            delayed_records.push_back(r);
        }

        // mark resolved weight shader-readonly @ compute
        dst.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
        dst.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }

    // keep these weights live while the next operation allocates its weights
    weight_buffers.push_back(dst);

    return 0;
}

int VkComputePrivate::submit_records()
{
    VkSemaphore compute_compute_semaphore = 0;
    VkSemaphore upload_upload_semaphore = 0;

    for (size_t i = 0; i < delayed_records.size(); i++)
    {
        const record& r = delayed_records[i];
        int ret = 0;

        switch (r.type)
        {
        case record::TYPE_host_copy:
        {
            const VkMat& src = upload_src_buffers[r.host_copy.src_index];
            const VkMat& dst = upload_dst_buffers[r.host_copy.dst_index];
            const size_t size = alignSize(r.host_copy.size, 4);
            unsigned char* ptr = (unsigned char*)dst.mapped_ptr();
            memcpy(ptr, (const unsigned char*)src.mapped_ptr() + r.host_copy.src_offset, r.host_copy.size);
            memset(ptr + r.host_copy.size, 0, size - r.host_copy.size);
            ret = flush_weight_buffer(vkdev, dst.data, dst.offset, size);
#if NCNN_BENCHMARK
            if (src.data->storage_type == 2)
                bytes_read += r.host_copy.size;
#endif
            break;
        }
        case record::TYPE_wait_compute:
        case record::TYPE_wait_upload:
        {
            const int command_buffer_index = r.wait.command_buffer_index;
            const bool compute = r.type == record::TYPE_wait_compute;
            const std::vector<VkFence>& fences = compute ? compute_command_fences : upload_command_fences;
            std::vector<bool>& pending = compute ? compute_command_pending : upload_command_pending;
            if (command_buffer_index >= 0 && (size_t)command_buffer_index < fences.size() && pending[command_buffer_index])
            {
                ret = wait_fence(fences[command_buffer_index]);
                if (ret == 0 || ret == VK_ERROR_DEVICE_LOST)
                    pending[command_buffer_index] = false;
            }
            break;
        }
        case record::TYPE_submit_compute:
        case record::TYPE_submit_upload:
        {
            const int command_buffer_index = r.submit.command_buffer_index;
            const bool compute = r.type == record::TYPE_submit_compute;
            VkSemaphore& previous_semaphore = compute ? compute_compute_semaphore : upload_upload_semaphore;
            VkSemaphore signal_semaphore = 0;

            // preserve submission order even when the next queue lease differs
            {
                VkSemaphoreCreateInfo semaphoreCreateInfo;
                semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
                semaphoreCreateInfo.pNext = 0;
                semaphoreCreateInfo.flags = 0;

                VkResult ret = vkCreateSemaphore(vkdev->vkdevice(), &semaphoreCreateInfo, 0, &signal_semaphore);
                if (ret != VK_SUCCESS)
                {
                    NCNN_LOGE("vkCreateSemaphore failed %d", ret);
                    return -1;
                }
                // retain even an unsubmitted signal until execution cleanup
                execution_semaphores.push_back(signal_semaphore);
            }

            std::vector<VkSemaphore> wait_semaphores(r.submit.wait_semaphore_count);
            for (uint32_t j = 0; j < r.submit.wait_semaphore_count; j++)
                wait_semaphores[j] = r.submit.wait_semaphores[j];
            if (previous_semaphore)
                wait_semaphores.push_back(previous_semaphore);
            std::vector<VkPipelineStageFlags> wait_stages(wait_semaphores.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            std::vector<VkSemaphore> signal_semaphores;
            if (r.submit.signal_semaphore)
                signal_semaphores.push_back(r.submit.signal_semaphore);
            signal_semaphores.push_back(signal_semaphore);

            // acquire queue and reclaim on return
            const uint32_t queue_family_index = compute ? vkdev->info.compute_queue_family_index() : transfer_queue_family_index;
            VkQueue queue = vkdev->acquire_queue(queue_family_index);
            if (queue == 0)
            {
                NCNN_LOGE("out of weight %s queue", compute ? "compute" : "transfer");
                return -1;
            }

            // submit recorded commands
            {
                VkSubmitInfo submitInfo;
                submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submitInfo.pNext = 0;
                submitInfo.waitSemaphoreCount = (uint32_t)wait_semaphores.size();
                submitInfo.pWaitSemaphores = wait_semaphores.data();
                submitInfo.pWaitDstStageMask = wait_stages.data();
                submitInfo.commandBufferCount = 1;
                submitInfo.pCommandBuffers = &r.command_buffer;
                submitInfo.signalSemaphoreCount = (uint32_t)signal_semaphores.size();
                submitInfo.pSignalSemaphores = signal_semaphores.data();

                VkResult ret = vkQueueSubmit(queue, 1, &submitInfo, compute ? compute_command_fences[command_buffer_index] : upload_command_fences[command_buffer_index]);
                if (ret != VK_SUCCESS)
                {
                    NCNN_LOGE("vkQueueSubmit failed %d", ret);
                    vkdev->reclaim_queue(queue_family_index, queue);
                    return ret;
                }
            }
            vkdev->reclaim_queue(queue_family_index, queue);

            previous_semaphore = signal_semaphore;
            if (compute)
                compute_command_pending[command_buffer_index] = true;
            else
                upload_command_pending[command_buffer_index] = true;
            break;
        }
        default:
            break;
        }
        if (ret != 0)
            return ret;
    }

    return 0;
}

class VkTransferPrivate
{
public:
    VkTransferPrivate(const VulkanDevice* _vkdev);
    ~VkTransferPrivate();

    int init();
    int begin_command_buffer();
    int end_command_buffer();
    int finish_upload(int error = 0);
    int submit_and_wait();
    int reset();
    int upload_buffers();
    void record_upload(const Mat& src, const VkMat& dst, const Option& opt);
    void record_copy(const VkBufferMemory& src, const VkBufferMemory& dst, size_t size);
    void record_host_upload(const VkBufferMemory& dst);

    const VulkanDevice* vkdev;

    uint64_t pending_upload_total;
    uint64_t queued_upload_total;
    int upload_error;
    bool compute_command_pending;
    bool upload_command_pending;

    VkCommandPool compute_command_pool;
    VkCommandPool transfer_command_pool;

    VkCommandBuffer upload_command_buffer;
    VkCommandBuffer compute_command_buffer;

    VkSemaphore upload_compute_semaphore;

    VkFence upload_command_fence;
    VkFence compute_command_fence;

    struct upload_record
    {
        Mat src;
        VkMat dst;
        size_t host_memory_limit;
        VkAllocator* staging_vkallocator;
    };
    std::vector<upload_record> uploads;
    VkMat pending_staging_mat;
    VkBufferMemory* pending_staging_buffer;
};

VkTransferPrivate::VkTransferPrivate(const VulkanDevice* _vkdev)
    : vkdev(_vkdev)
{
    pending_upload_total = 0;
    queued_upload_total = 0;
    upload_error = 0;
    compute_command_pending = false;
    upload_command_pending = false;

    compute_command_pool = 0;
    transfer_command_pool = 0;

    upload_command_buffer = 0;
    compute_command_buffer = 0;

    upload_compute_semaphore = 0;

    upload_command_fence = 0;
    compute_command_fence = 0;

    pending_staging_buffer = 0;

    upload_error = init();
}

VkTransferPrivate::~VkTransferPrivate()
{
    do
    {
        finish_upload(upload_error);
        if (compute_command_pending || upload_command_pending)
            wait_for_pending_commands();
    } while (compute_command_pending || upload_command_pending);
    destroy_weight_buffer(vkdev, pending_staging_buffer);

    vkDestroyFence(vkdev->vkdevice(), compute_command_fence, 0);

    if (compute_command_buffer)
        vkFreeCommandBuffers(vkdev->vkdevice(), compute_command_pool, 1, &compute_command_buffer);
    vkDestroyCommandPool(vkdev->vkdevice(), compute_command_pool, 0);

    if (!vkdev->info.unified_compute_transfer_queue())
    {
        vkDestroyFence(vkdev->vkdevice(), upload_command_fence, 0);

        vkDestroySemaphore(vkdev->vkdevice(), upload_compute_semaphore, 0);

        if (upload_command_buffer)
            vkFreeCommandBuffers(vkdev->vkdevice(), transfer_command_pool, 1, &upload_command_buffer);
        vkDestroyCommandPool(vkdev->vkdevice(), transfer_command_pool, 0);
    }
}

int VkTransferPrivate::init()
{
    // compute_command_pool
    {
        VkCommandPoolCreateInfo commandPoolCreateInfo;
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.pNext = 0;
        commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandPoolCreateInfo.queueFamilyIndex = vkdev->info.compute_queue_family_index();

        VkResult ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &compute_command_pool);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateCommandPool failed %d", ret);
            return -1;
        }
    }

    // compute_command_buffer
    {
        VkCommandBufferAllocateInfo commandBufferAllocateInfo;
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.pNext = 0;
        commandBufferAllocateInfo.commandPool = compute_command_pool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;

        VkResult ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &commandBufferAllocateInfo, &compute_command_buffer);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkAllocateCommandBuffers failed %d", ret);
            return -1;
        }
    }

    // compute_command_fence
    {
        VkFenceCreateInfo fenceCreateInfo;
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.pNext = 0;
        fenceCreateInfo.flags = 0;

        VkResult ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &compute_command_fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateFence failed %d", ret);
            return -1;
        }
    }

    if (!vkdev->info.unified_compute_transfer_queue())
    {
        // transfer_command_pool
        {
            VkCommandPoolCreateInfo commandPoolCreateInfo;
            commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            commandPoolCreateInfo.pNext = 0;
            commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            commandPoolCreateInfo.queueFamilyIndex = vkdev->info.transfer_queue_family_index();

            VkResult ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &transfer_command_pool);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkCreateCommandPool failed %d", ret);
                return -1;
            }
        }

        // upload_command_buffer
        {
            VkCommandBufferAllocateInfo commandBufferAllocateInfo;
            commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            commandBufferAllocateInfo.pNext = 0;
            commandBufferAllocateInfo.commandPool = transfer_command_pool;
            commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            commandBufferAllocateInfo.commandBufferCount = 1;

            VkResult ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &commandBufferAllocateInfo, &upload_command_buffer);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkAllocateCommandBuffers failed %d", ret);
                return -1;
            }
        }

        // upload_compute_semaphore
        {
            VkSemaphoreCreateInfo semaphoreCreateInfo;
            semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            semaphoreCreateInfo.pNext = 0;
            semaphoreCreateInfo.flags = 0;

            VkResult ret = vkCreateSemaphore(vkdev->vkdevice(), &semaphoreCreateInfo, 0, &upload_compute_semaphore);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkCreateSemaphore failed %d", ret);
                return -1;
            }
        }

        // upload_command_fence
        {
            VkFenceCreateInfo fenceCreateInfo;
            fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceCreateInfo.pNext = 0;
            fenceCreateInfo.flags = 0;

            VkResult ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &upload_command_fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkCreateFence failed %d", ret);
                return -1;
            }
        }
    }

    return begin_command_buffer();
}

int VkTransferPrivate::begin_command_buffer()
{
    {
        VkCommandBufferBeginInfo commandBufferBeginInfo;
        commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        commandBufferBeginInfo.pNext = 0;
        commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        commandBufferBeginInfo.pInheritanceInfo = 0;

        VkResult ret = vkBeginCommandBuffer(compute_command_buffer, &commandBufferBeginInfo);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkBeginCommandBuffer failed %d", ret);
            return -1;
        }
    }

    if (!vkdev->info.unified_compute_transfer_queue())
    {
        {
            VkCommandBufferBeginInfo commandBufferBeginInfo;
            commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            commandBufferBeginInfo.pNext = 0;
            commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            commandBufferBeginInfo.pInheritanceInfo = 0;

            VkResult ret = vkBeginCommandBuffer(upload_command_buffer, &commandBufferBeginInfo);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkBeginCommandBuffer failed %d", ret);
                return -1;
            }
        }
    }

    return 0;
}

int VkTransferPrivate::end_command_buffer()
{
    {
        VkResult ret = vkEndCommandBuffer(compute_command_buffer);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkEndCommandBuffer failed %d", ret);
            return -1;
        }
    }

    if (!vkdev->info.unified_compute_transfer_queue())
    {
        {
            VkResult ret = vkEndCommandBuffer(upload_command_buffer);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkEndCommandBuffer failed %d", ret);
                return -1;
            }
        }
    }

    return 0;
}

VkTransfer::VkTransfer(const VulkanDevice* _vkdev)
    : vkdev(_vkdev), d(new VkTransferPrivate(_vkdev))
{
}

VkTransfer::~VkTransfer()
{
    delete d;
}

void VkTransfer::record_upload(const Mat& src, VkMat& dst, const Option& opt, bool flatten)
{
    if (d->upload_error)
        return;

    if (src.empty())
        return;

    //     NCNN_LOGE("record_upload src = %d | %d %d %d @ %d", src.dims, src.w, src.h, src.c, src.elempack);

    // NOTE keep the hack here ?
    if (src.elembits() == 32)
    {
        if (opt.use_bf16_storage || opt.use_bf16_packed)
        {
            Mat src_bf16;
            cast_float32_to_bfloat16(src, src_bf16, opt);

            record_upload(src_bf16, dst, opt, flatten);

            return;
        }
        else if (opt.use_fp16_storage || opt.use_fp16_packed)
        {
            Mat src_fp16;
            cast_float32_to_float16(src, src_fp16, opt);

            record_upload(src_fp16, dst, opt, flatten);

            return;
        }
    }

    const size_t flattened_size = (size_t)src.w * src.h * src.d * src.c;
    if (flatten && flattened_size > 2147483647)
    {
        NCNN_LOGE("weight exceeds maximum flattened dimension");
        d->upload_error = -1;
        return;
    }
    Mat src_flattened = flatten ? src.reshape((int)flattened_size) : src;

    if (src_flattened.empty())
    {
        d->upload_error = -100;
        return;
    }

    // create dst
    dst.create_like(src_flattened, opt.blob_vkallocator);

    if (dst.empty())
    {
        d->upload_error = -100;
        return;
    }

    // preserve the source contents while an unmapped upload waits for staging
    if (dst.data->storage_type == 2 || dst.mapped_ptr())
        d->record_upload(src_flattened, dst, opt);
    else
    {
        Mat snapshot = src_flattened.clone();
        if (snapshot.empty())
        {
            d->upload_error = -100;
            return;
        }
        d->record_upload(snapshot, dst, opt);
    }
}

void VkTransferPrivate::record_upload(const Mat& src, const VkMat& dst, const Option& opt)
{
    if (upload_error || src.empty())
        return;
    if (dst.empty() || !dst.allocator || dst.allocator->vkdev != vkdev || src.n != dst.n
            || dst.offset > dst.data->capacity || dst.buffer_capacity() > dst.data->capacity - dst.offset)
    {
        upload_error = -1;
        return;
    }
    for (int b = 0; b < src.n; b++)
    {
        const size_t offset = dst.batch(b).offset;
        const size_t end = b + 1 == src.n ? dst.offset + dst.buffer_capacity() : dst.batch(b + 1).offset;
        if (offset > end || src.batch(b).total() * src.elemsize > end - offset)
        {
            upload_error = -1;
            return;
        }
    }
    if (dst.data->storage_type == 2)
    {
        if (!(dst.data->access_flags & VK_ACCESS_HOST_WRITE_BIT) || !dst.mapped_ptr())
        {
            NCNN_LOGE("disk weight requires a writable source range during upload");
            upload_error = -1;
            return;
        }
        for (int b = 0; b < src.n; b++)
        {
            const Mat src_b = src.batch(b);
            const VkMat dst_b = dst.batch(b);
            const size_t end = b + 1 == src.n ? dst.offset + dst.buffer_capacity() : dst.batch(b + 1).offset;
            const size_t size = src_b.total() * src_b.elemsize;
            unsigned char* ptr = (unsigned char*)dst_b.mapped_ptr();
            memcpy(ptr, src_b.data, size);
            memset(ptr + size, 0, end - dst_b.offset - size);
        }
        return;
    }
    if (dst.mapped_ptr())
    {
        for (int b = 0; b < src.n; b++)
        {
            const Mat src_b = src.batch(b);
            const VkMat dst_b = dst.batch(b);
            const size_t end = b + 1 == src.n ? dst.offset + dst.buffer_capacity() : dst.batch(b + 1).offset;
            const size_t size = src_b.total() * src_b.elemsize;
            unsigned char* ptr = (unsigned char*)dst_b.mapped_ptr();
            memcpy(ptr, src_b.data, size);
            memset(ptr + size, 0, end - dst_b.offset - size);
        }
        upload_error = dst.allocator->flush(dst.data);
        if (upload_error)
            return;
        if (dst.data->storage_type != 1)
        {
            record_host_upload(*dst.data);
            pending_upload_total += dst.buffer_capacity();
        }
        dst.data->access_flags = dst.data->storage_type == 1 ? VK_ACCESS_HOST_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
        dst.data->stage_flags = dst.data->storage_type == 1 ? VK_PIPELINE_STAGE_HOST_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        return;
    }
    if (dst.buffer_capacity() > (uint64_t)-1 - queued_upload_total)
    {
        upload_error = -1;
        return;
    }
    VkTransferPrivate::upload_record r;
    r.src = src;
    r.dst = dst;
    r.host_memory_limit = opt.weight_host_memory_limit;
    r.staging_vkallocator = opt.staging_vkallocator;
    uploads.push_back(r);
    queued_upload_total += dst.buffer_capacity();
}

void VkTransferPrivate::record_copy(const VkBufferMemory& src, const VkBufferMemory& dst, size_t size)
{
    VkCommandBuffer command_buffer;
    if (vkdev->info.unified_compute_transfer_queue())
    {
        command_buffer = compute_command_buffer;
    }
    else
    {
        command_buffer = upload_command_buffer;
    }

    // barrier staging host-write @ null to transfer-read @ queue
    {
        VkBufferMemoryBarrier barrier;
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = 0;
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = src.buffer;
        barrier.offset = src.offset;
        barrier.size = size;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_HOST_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
    }

    // record staging to device
    {
        VkBufferCopy region;
        region.srcOffset = src.offset;
        region.dstOffset = dst.offset;
        region.size = size;

        vkCmdCopyBuffer(command_buffer, src.buffer, dst.buffer, 1, &region);
    }

    if (vkdev->info.unified_compute_transfer_queue())
    {
        // barrier device transfer-write @ compute to shader-read @ compute
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = dst.buffer;
            barrier.offset = dst.offset;
            barrier.size = size;

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }
    }
    else
    {
        // queue ownership transfer transfer-write @ transfer to shader-read @ compute

        // release
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = 0;
            barrier.srcQueueFamilyIndex = vkdev->info.transfer_queue_family_index();
            barrier.dstQueueFamilyIndex = vkdev->info.compute_queue_family_index();
            barrier.buffer = dst.buffer;
            barrier.offset = dst.offset;
            barrier.size = size;

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

            vkCmdPipelineBarrier(upload_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }

        // acquire
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = vkdev->info.transfer_queue_family_index();
            barrier.dstQueueFamilyIndex = vkdev->info.compute_queue_family_index();
            barrier.buffer = dst.buffer;
            barrier.offset = dst.offset;
            barrier.size = size;

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }
    }
}

void VkTransferPrivate::record_host_upload(const VkBufferMemory& dst)
{
    // barrier device host-write @ null to shader-read @ compute
    {
        VkBufferMemoryBarrier barrier;
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = 0;
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = dst.buffer;
        barrier.offset = dst.offset;
        barrier.size = dst.capacity;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_HOST_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
    }
}

int VkTransferPrivate::upload_buffers()
{
    if (upload_error)
        return upload_error;
    size_t capacity = 0;
    size_t host_memory_limit = 0;
    const size_t staging_buffer_size = 8 * 1024 * 1024;
    for (size_t i = 0; i < uploads.size(); i++)
    {
        const upload_record& r = uploads[i];
        capacity += std::min(alignSize(r.dst.buffer_capacity(), 4), staging_buffer_size - capacity);
        if (r.host_memory_limit)
            host_memory_limit = host_memory_limit ? std::min(host_memory_limit, r.host_memory_limit) : r.host_memory_limit;
    }
    VkBufferMemory* staging = 0;
    VkMat staging_mat;
    VkAllocator* staging_vkallocator = 0;
    int ret = 0;
    if (capacity)
        ret = fit_weight_buffer_size(vkdev, host_memory_limit, true, capacity);
    size_t used = 0;
    for (size_t i = 0; ret == 0 && i < uploads.size(); i++)
    {
        const upload_record& r = uploads[i];
        const VkMat& dst = r.dst;
        if (!staging || staging_vkallocator != r.staging_vkallocator)
        {
            if (used)
            {
                ret = staging_vkallocator ? staging_vkallocator->flush(staging) : flush_weight_buffer(vkdev, staging, 0, used);
                if (ret == 0)
                    ret = submit_and_wait();
                if (ret == 0)
                    ret = reset();
                used = 0;
            }
            if (ret != 0)
                break;
            if (staging && !staging_vkallocator)
                destroy_weight_buffer(vkdev, staging);
            staging_mat.release();
            staging = 0;
            staging_vkallocator = r.staging_vkallocator;
            if (staging_vkallocator)
            {
                if (staging_vkallocator->vkdev != vkdev)
                {
                    ret = -1;
                    break;
                }
                staging_mat.create((int)capacity, (size_t)1u, staging_vkallocator);
                staging = staging_mat.data;
            }
            else
            {
                size_t allocation_size;
                const bool unified = vkdev->info.type() == 1 || vkdev->info.type() == 3;
                staging = create_weight_buffer(vkdev, capacity, 1, unified, host_memory_limit, allocation_size, true);
            }
            if (!staging || !staging->mapped_ptr)
            {
                ret = -100;
                break;
            }
            if (host_memory_limit)
            {
                VkMemoryRequirements requirements;
                vkGetBufferMemoryRequirements(vkdev->vkdevice(), staging->buffer, &requirements);
                if (requirements.size > host_memory_limit)
                {
                    ret = -1;
                    break;
                }
            }
        }
        for (int b = 0; ret == 0 && b < r.src.n; b++)
        {
            const Mat src_b = r.src.batch(b);
            const VkMat dst_b = dst.batch(b);
            const size_t src_size = src_b.total() * src_b.elemsize;
            const size_t end = b + 1 == r.src.n ? dst.offset + dst.buffer_capacity() : dst.batch(b + 1).offset;
            const size_t size = end - dst_b.offset;
            for (size_t offset = 0; ret == 0 && offset < size;)
            {
                if (used == capacity)
                {
                    ret = staging_vkallocator ? staging_vkallocator->flush(staging) : flush_weight_buffer(vkdev, staging, 0, used);
                    if (ret == 0)
                        ret = submit_and_wait();
                    if (ret == 0)
                        ret = reset();
                    used = 0;
                    if (ret != 0)
                        break;
                }
                const size_t n = std::min(capacity - used, size - offset);
                const size_t copy_size = alignSize(n, 4);
                if (offset + copy_size > dst.data->capacity - dst_b.offset)
                {
                    ret = -1;
                    break;
                }
                const size_t src_bytes = offset < src_size ? std::min(n, src_size - offset) : 0;
                unsigned char* ptr = (unsigned char*)staging->mapped_ptr + staging->offset + used;
                if (src_bytes)
                    memcpy(ptr, (const unsigned char*)src_b.data + offset, src_bytes);
                memset(ptr + src_bytes, 0, copy_size - src_bytes);
                VkBufferMemory src_buffer = *staging;
                src_buffer.offset += used;
                VkBufferMemory dst_buffer = *dst.data;
                dst_buffer.offset = dst_b.buffer_offset() + offset;
                record_copy(src_buffer, dst_buffer, copy_size);
                pending_upload_total += copy_size;
                used += copy_size;
                offset += n;
            }
        }
        if (ret == 0)
        {
            dst.data->access_flags = dst.data->storage_type == 1 ? VK_ACCESS_HOST_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
            dst.data->stage_flags = dst.data->storage_type == 1 ? VK_PIPELINE_STAGE_HOST_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        }
    }
    if (ret == 0 && used)
        ret = staging_vkallocator ? staging_vkallocator->flush(staging) : flush_weight_buffer(vkdev, staging, 0, used);
    if (ret == 0)
        ret = submit_and_wait();
    // A failed fence wait does not prove completion. Keep staging alive for
    // finish_upload() to drain again during command destruction.
    if (compute_command_pending || upload_command_pending)
    {
        if (staging_vkallocator)
            pending_staging_mat = staging_mat;
        else
            pending_staging_buffer = staging;
    }
    else if (staging && !staging_vkallocator)
    {
        destroy_weight_buffer(vkdev, staging);
    }
    if (ret != 0)
        upload_error = ret;
    if (ret == 0)
    {
        uploads.clear();
        queued_upload_total = 0;
    }
    return ret;
}

int VkTransferPrivate::finish_upload(int error)
{
    VkFence fences[2];
    uint32_t count = 0;
    if (upload_command_pending)
        fences[count++] = upload_command_fence;
    if (compute_command_pending)
        fences[count++] = compute_command_fence;
    if (count)
    {
        const VkResult ret = vkWaitForFences(vkdev->vkdevice(), count, fences, VK_TRUE, (uint64_t)-1);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkWaitForFences failed %d", ret);
            error = -1;
        }
        if (ret == VK_SUCCESS || ret == VK_ERROR_DEVICE_LOST)
        {
            compute_command_pending = false;
            upload_command_pending = false;
        }
    }
    if (error)
        upload_error = error;
    else
        pending_upload_total = 0;
    return error;
}

int VkTransfer::submit_and_wait()
{
    return d->upload_buffers();
}

int VkTransfer::reset()
{
    const int ret = d->reset();
    if (ret == 0)
    {
        d->uploads.clear();
        d->queued_upload_total = 0;
    }
    return ret;
}

int VkTransferPrivate::submit_and_wait()
{
    if (upload_error)
        return upload_error;
    if (pending_upload_total == 0)
        return 0;

    //     NCNN_LOGE("submit_and_wait");

    // end command buffer
    {
        int ret = end_command_buffer();
        if (ret != 0)
            return finish_upload(ret);
    }

    VkQueue compute_queue = vkdev->acquire_queue(vkdev->info.compute_queue_family_index());
    if (compute_queue == 0)
    {
        NCNN_LOGE("out of compute queue");
        return finish_upload(-1);
    }

    if (vkdev->info.unified_compute_transfer_queue())
    {
        // submit compute
        {
            VkSubmitInfo submitInfo;
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = 0;
            submitInfo.waitSemaphoreCount = 0;
            submitInfo.pWaitSemaphores = 0;
            submitInfo.pWaitDstStageMask = 0;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &compute_command_buffer;
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = 0;

            VkResult ret = vkQueueSubmit(compute_queue, 1, &submitInfo, compute_command_fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkQueueSubmit failed %d", ret);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
                return finish_upload(-1);
            }
            compute_command_pending = true;
        }
    }
    else
    {
        VkQueue transfer_queue = vkdev->acquire_queue(vkdev->info.transfer_queue_family_index());
        if (transfer_queue == 0)
        {
            NCNN_LOGE("out of transfer queue");
            vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
            return finish_upload(-1);
        }

        // submit upload compute
        {
            VkSubmitInfo submitInfo;
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = 0;
            submitInfo.waitSemaphoreCount = 0;
            submitInfo.pWaitSemaphores = 0;
            submitInfo.pWaitDstStageMask = 0;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &upload_command_buffer;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &upload_compute_semaphore;

            VkResult ret = vkQueueSubmit(transfer_queue, 1, &submitInfo, upload_command_fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkQueueSubmit failed %d", ret);
                vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index(), transfer_queue);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
                return finish_upload(-1);
            }
            upload_command_pending = true;
        }
        {
            VkPipelineStageFlags wait_dst_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; // FIXME

            VkSubmitInfo submitInfo;
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = 0;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &upload_compute_semaphore;
            submitInfo.pWaitDstStageMask = &wait_dst_stage;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &compute_command_buffer;
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = 0;

            VkResult ret = vkQueueSubmit(compute_queue, 1, &submitInfo, compute_command_fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkQueueSubmit failed %d", ret);
                vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index(), transfer_queue);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);
                return finish_upload(-1);
            }
            compute_command_pending = true;
        }

        vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index(), transfer_queue);
    }

    vkdev->reclaim_queue(vkdev->info.compute_queue_family_index(), compute_queue);

    return finish_upload();
}

int VkTransferPrivate::reset()
{
    if (upload_error)
        return upload_error;

    pending_upload_total = 0;

    // reset command buffer and fence
    {
        VkResult ret = vkResetCommandBuffer(compute_command_buffer, 0);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkResetCommandBuffer failed %d", ret);
            upload_error = -1;
            return upload_error;
        }
    }
    {
        VkResult ret = vkResetFences(vkdev->vkdevice(), 1, &compute_command_fence);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkResetFences failed %d", ret);
            upload_error = -1;
            return upload_error;
        }
    }

    if (!vkdev->info.unified_compute_transfer_queue())
    {
        {
            VkResult ret = vkResetCommandBuffer(upload_command_buffer, 0);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkResetCommandBuffer failed %d", ret);
                upload_error = -1;
                return upload_error;
            }
        }
        {
            VkResult ret = vkResetFences(vkdev->vkdevice(), 1, &upload_command_fence);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("vkResetFences failed %d", ret);
                upload_error = -1;
                return upload_error;
            }
        }
    }

    upload_error = begin_command_buffer();

    return upload_error;
}

uint64_t VkTransfer::pending_upload_total() const
{
    return d->pending_upload_total + d->queued_upload_total;
}

} // namespace ncnn

#endif // NCNN_VULKAN
