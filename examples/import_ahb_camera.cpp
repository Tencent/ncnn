// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause
//
// Reference example for ncnn's Android-hardware-buffer zero-copy input
// path. Companion to:
//
//   docs/how-to-use-and-FAQ/use-ncnn-with-android-hardware-buffer.md
//
// This file is NOT compiled by the desktop `examples/CMakeLists.txt`
// because every type it touches (`AHardwareBuffer`, `AImageReader`,
// `VkAndroidHardwareBufferImageAllocator`, ...) is only declared when
// ncnn is configured with `-DANDROID_PLATFORM=android-26` (or higher)
// AND `-DNCNN_VULKAN=ON`. Treat it as a copy-into-your-NDK-app
// template: lift the helpers into your AImageReader callback, your
// camera-open path, and your inference loop respectively.
//
// What this example demonstrates end-to-end:
//
//   1. Runtime capability probe       --  ahb_runtime_supported()
//   2. AImageReader setup with the
//      required GPU-sampled usage     --  open_camera_reader()
//   3. AHardwareBuffer -> VkImageMat
//      import + per-AHB pipeline      --  import_ahb_or_get_cached()
//      cache (keyed by AHB pointer)
//   4. Pre-cast to the network's
//      (elempack, dtype) before        --  cast_to_extractor_format()
//      ex.input(VkMat)
//   5. End-to-end image-available
//      callback wiring through        --  on_image_available_example()
//      ncnn::Extractor
//
// Requirements (mirrored from the doc):
//   * Android API level 26 or higher.
//   * ncnn built with -DANDROID_PLATFORM=android-26 -DNCNN_VULKAN=ON.
//     The official prebuilt archives published on the releases page are
//     stripped of every AHB symbol and the linker will refuse them.
//   * A Vulkan-capable GPU advertising
//     VK_ANDROID_external_memory_android_hardware_buffer. Verified on
//     Adreno 830 (Vulkan 1.3.284, spec version 5) and Mali-G925
//     Immortalis MC12 (Vulkan 1.3.278, spec version 5).
//
// Caveats lifted from the doc -- read these BEFORE adapting:
//
//   * VkAndroidHardwareBufferImageAllocator does not take a ref on its
//     AHB. If the allocator (or its VkImageMat) outlives the AImage
//     from which the AHB came, call AHardwareBuffer_acquire(ahb) before
//     AImage_delete(image) and AHardwareBuffer_release(ahb) at cache
//     eviction. The cache helper below does this for you.
//   * `record_import_android_hardware_buffer` produces RGB float in
//     the [0, 255] range -- same convention as `from_pixels`, NOT
//     [0, 1]. If your model expects normalised input, fold the 1/255
//     into a post-import Scale layer or a custom compute pass.
//   * `ex.input(VkMat)` does NOT auto-convert (elempack, dtype). If the
//     network was loaded with `use_fp16_storage` or `use_fp16_packed`
//     and you feed the raw `fp32, elempack=1` output of the import
//     pipeline, the first conv layer's bottom-blob lookup will trigger
//     an infinite recursion in NetPrivate::forward_layer trying to
//     resolve an implicit converter that does not exist. Always run
//     `vkdev->convert_packing(dst, dst_casted, target_elempack,
//     cast_type_to, cmd, opt)` BEFORE `ex.input` -- see
//     cast_to_extractor_format() below.
//   * The imported VkImage is created with VK_IMAGE_USAGE_SAMPLED_BIT
//     only. Do not write to it through this path; for an output AHB,
//     allocate a separate exportable buffer and use
//     vkGetMemoryAndroidHardwareBufferANDROID.

#if !defined(__ANDROID__) || (defined(__ANDROID_API__) && (__ANDROID_API__ < 26))
#error "This example targets Android API 26 or higher. Build with -DANDROID_PLATFORM=android-26."
#endif

#include "gpu.h"
#include "mat.h"
#include "net.h"

#include <android/hardware_buffer.h>
#include <media/NdkImage.h>
#include <media/NdkImageReader.h>

#include <unordered_map>

// ─── 1. capability probe ───────────────────────────────────────────────

// Returns the AHB external-memory extension spec version (> 0 on
// supported devices), or 0 when the device does not advertise the
// extension. Call once at startup, before opening the camera, and fall
// back to the legacy `from_pixels` path when the result is 0.
static int ahb_runtime_supported(ncnn::VulkanDevice* vkdev)
{
    const ncnn::GpuInfo& gi = vkdev->info;
    return gi.support_VK_ANDROID_external_memory_android_hardware_buffer();
}

// ─── 2. AImageReader with the GPU-sampled usage bit ────────────────────

// `AImageReader_new` does NOT request GPU_SAMPLED_IMAGE, which leaves
// the produced AHardwareBuffer unimportable into Vulkan. Use the
// _newWithUsage variant and include the bit explicitly. Some HALs
// silently drop usage bits they don't support, so verify the first
// frame's AHB via AHardwareBuffer_describe before assuming the path is
// live (see ahb_has_gpu_sampled_flag below).
static AImageReader* open_camera_reader(int width, int height)
{
    AImageReader* reader = nullptr;
    media_status_t st = AImageReader_newWithUsage(
                            width, height, AIMAGE_FORMAT_YUV_420_888,
                            AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN
                            | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE,
                            /*maxImages=*/3, &reader);
    if (st != AMEDIA_OK) return nullptr;
    return reader;
}

static bool ahb_has_gpu_sampled_flag(AHardwareBuffer* ahb)
{
    AHardwareBuffer_Desc desc{};
    AHardwareBuffer_describe(ahb, &desc);
    return (desc.usage & AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE) != 0;
}

// ─── 3. per-AHB cache (pipeline + allocator + VkImageMat) ──────────────
//
// Per-frame `pipe.create` is the expensive step (~24 ms median on
// Adreno 830). It depends only on the immutable sampler derived from
// the AHB's externalFormat plus the (type_to, rotate_from, target_w,
// target_h) specialisation constants -- all stable for a given camera
// session. AImageReader cycles through a small AHB pool (typically
// near `maxImages` under steady state), so a cache keyed by the AHB
// pointer stays bounded and brings the steady-state setup cost from
// 24 ms to a few microseconds.

struct AhbCacheEntry
{
    AHardwareBuffer* ahb;
    ncnn::VkAndroidHardwareBufferImageAllocator* alloc;
    ncnn::ImportAndroidHardwareBufferPipeline* pipe;
    ncnn::VkImageMat src;
};

// Look up or create a cache entry for `ahb`. Camera-thread only -- the
// map is not synchronised.
static AhbCacheEntry& import_ahb_or_get_cached(
    std::unordered_map<AHardwareBuffer*, AhbCacheEntry>& cache,
    ncnn::VulkanDevice* vkdev,
    AHardwareBuffer* ahb,
    int target_w,
    int target_h,
    int type_to,     // 1=RGB, 2=BGR, 3=GRAY, 4=RGBA, 5=BGRA
    int rotate_from, // EXIF orientation code: 1 = identity, 5..8 = rotations
    const ncnn::Option& opt)
{
    auto it = cache.find(ahb);
    if (it != cache.end()) return it->second;

    // The allocator does NOT take a ref on its AHB. Acquire one here so
    // the cached entry can outlive the originating AImage.
    AHardwareBuffer_acquire(ahb);

    AhbCacheEntry e{};
    e.ahb = ahb;
    e.alloc = new ncnn::VkAndroidHardwareBufferImageAllocator(vkdev, ahb);
    e.src = ncnn::VkImageMat::from_android_hardware_buffer(e.alloc);
    e.pipe = new ncnn::ImportAndroidHardwareBufferPipeline(vkdev);
    e.pipe->create(e.alloc, type_to, rotate_from, target_w, target_h, opt);

    auto ins = cache.emplace(ahb, std::move(e));
    return ins.first->second;
}

// Call at camera-close time (or on init failure). Each entry holds one
// Vulkan pipeline, one allocator, and one AHardwareBuffer ref.
static void evict_ahb_cache(
    std::unordered_map<AHardwareBuffer*, AhbCacheEntry>& cache)
{
    for (auto& kv : cache)
    {
        delete kv.second.pipe;
        delete kv.second.alloc;
        AHardwareBuffer_release(kv.second.ahb);
    }
    cache.clear();
}

// ─── 4. pre-cast to the network's (elempack, dtype) ────────────────────
//
// The import pipeline outputs `fp32, elempack=1`. Networks loaded with
// use_fp16_storage or use_fp16_packed expect a different packing
// (typically fp16, elempack=4 or elempack=8 on Adreno). `ex.input` will
// NOT convert for you on the GPU path -- pre-cast here, on the same
// VkCompute, so the conversion happens in-pipeline.
//
// `cast_type_to` is a raw int code (see VulkanDevice::convert_packing in
// src/gpu.cpp):
//
//     0 = auto (keep source dtype)
//     1 = fp32
//     2 = fp16
//     3 = int32
//     4 = int8
//     5 = bf16
static ncnn::VkMat cast_to_extractor_format(
    ncnn::VulkanDevice* vkdev,
    const ncnn::VkMat& src,
    int target_elempack,
    int cast_type_to,
    ncnn::VkCompute& cmd,
    const ncnn::Option& opt)
{
    ncnn::VkMat dst;
    vkdev->convert_packing(src, dst, target_elempack, cast_type_to, cmd, opt);
    return dst;
}

// ─── 5. end-to-end image-available callback ────────────────────────────
//
// This is the shape of the callback you should install with
// AImageReader_setImageListener. The `Ctx` struct stores the state
// that survives across frames; allocate one alongside the AImageReader.
//
// In your real app, replace the placeholders marked TODO with the
// values your network expects (input blob name, target_size, target
// elempack/dtype, output blob name(s)).

struct Ctx
{
    ncnn::Net* net;
    ncnn::VulkanDevice* vkdev;
    std::unordered_map<AHardwareBuffer*, AhbCacheEntry> ahb_cache;
};

static void on_image_available_example(void* userdata, AImageReader* reader)
{
    Ctx* ctx = static_cast<Ctx*>(userdata);

    AImage* image = nullptr;
    if (AImageReader_acquireLatestImage(reader, &image) != AMEDIA_OK || !image)
        return;

    AHardwareBuffer* ahb = nullptr;
    if (AImage_getHardwareBuffer(image, &ahb) != AMEDIA_OK || !ahb)
    {
        AImage_delete(image);
        return;
    }

    // First-frame sanity check: the camera HAL may have silently
    // dropped GPU_SAMPLED_IMAGE. If so this AHB cannot be imported into
    // Vulkan; abort this frame. In production code you would flip a
    // flag on `ctx` here and route subsequent frames through the legacy
    // CPU `from_pixels` path for the rest of the session.
    if (!ahb_has_gpu_sampled_flag(ahb))
    {
        AImage_delete(image);
        return;
    }

    // Network input dimensions. The import pipeline does the YCbCr ->
    // RGB conversion AND the resize in a single compute pass, so you
    // can target the network's input size directly without a separate
    // resize step.
    const int target_w = 1280; // TODO: replace with your network's expected width
    const int target_h = 720;  // TODO: replace with your network's expected height

    ncnn::Option opt = ctx->net->opt;
    opt.use_vulkan_compute = true;
    opt.blob_vkallocator = ctx->vkdev->acquire_blob_allocator();
    opt.staging_vkallocator = ctx->vkdev->acquire_staging_allocator();

    AhbCacheEntry& e = import_ahb_or_get_cached(
                           ctx->ahb_cache, ctx->vkdev, ahb,
                           target_w, target_h,
                           /*type_to=*/1,     // RGB
                           /*rotate_from=*/1, // identity (set this per your camera orientation)
                           opt);

    ncnn::VkCompute cmd(ctx->vkdev);

    // Step 1: import. `dst` holds the YCbCr->RGB resolved image in
    // fp32, elempack=1, range [0, 255].
    ncnn::VkMat dst;
    dst.create(target_w, target_h, /*c=*/3, /*elemsize=*/4u, /*elempack=*/1,
               opt.blob_vkallocator);
    cmd.record_import_android_hardware_buffer(e.pipe, e.src, dst);

    // Step 2: pre-cast to the format the network actually expects.
    // For use_fp16_storage + elempack=4 (common Adreno path) use
    // target_elempack=4, cast_type_to=2 (fp16). When your network was
    // loaded with the default fp32 settings you can skip this cast and
    // pass `dst` straight into ex.input. When in doubt -- ALWAYS cast.
    // See cast_to_extractor_format() above for the cast_type_to codes.
    ncnn::VkMat in_blob = cast_to_extractor_format(
                              ctx->vkdev, dst,
                              /*target_elempack=*/4, // TODO: read from your model
                              /*cast_type_to=*/2,    // fp16; TODO: match your model
                              cmd, opt);

    // Step 3: run the network. ex.input takes the casted VkMat; the
    // rest of the extractor pipeline is identical to the CPU path.
    ncnn::Extractor ex = ctx->net->create_extractor();
    ex.input("in0", in_blob); // TODO: your network's input blob name

    ncnn::VkMat out_vk;
    ex.extract("out0", out_vk, cmd); // TODO: your network's output blob name

    cmd.submit_and_wait();

    // The output is still on the GPU. Download or chain into a follow-
    // up compute pass per your app's needs. The doc has a worked
    // example of feeding the output back into a composite pass without
    // a round-trip through the host.

    ctx->vkdev->reclaim_blob_allocator(opt.blob_vkallocator);
    ctx->vkdev->reclaim_staging_allocator(opt.staging_vkallocator);

    AImage_delete(image);
}

// ─── 6. minimal init / teardown sketch ─────────────────────────────────
//
// These are not real `main()` entry points -- they show the order in
// which to call the helpers above from your app's onCreate / onPause /
// onResume.

static bool init_example(Ctx* ctx, const char* param_path, const char* model_path)
{
    ctx->vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    if (!ctx->vkdev) return false;
    if (ahb_runtime_supported(ctx->vkdev) <= 0) return false;

    ctx->net = new ncnn::Net();
    ctx->net->opt.use_vulkan_compute = true;
    if (ctx->net->load_param(param_path)) return false;
    if (ctx->net->load_model(model_path)) return false;
    return true;
}

static void destroy_example(Ctx* ctx)
{
    evict_ahb_cache(ctx->ahb_cache);
    delete ctx->net;
    ctx->net = nullptr;
    ctx->vkdev = nullptr;
}
