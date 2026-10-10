# use ncnn with android hardware buffer (zero-copy input)

When the input to inference is a camera frame or rendered surface on Android, the `AHardwareBuffer` it produces can be imported directly into Vulkan and fed to ncnn as a `VkImageMat`, avoiding the host→device copy that `ncnn::Mat::from_pixels` would incur.

This page documents the existing AHB import path (`VkAndroidHardwareBufferImageAllocator`, `VkImageMat::from_android_hardware_buffer`, `ImportAndroidHardwareBufferPipeline`) — what builds it, what calls it, and the caveats. See [#5531](https://github.com/Tencent/ncnn/issues/5531) for the diagnostic backstory.

### requirements

- Android API level **26 or higher**.
- ncnn built with `-DANDROID_PLATFORM=android-26` (or higher) and `-DNCNN_VULKAN=ON`. The official prebuilt archives published on the releases page are built with `-DANDROID_PLATFORM=android-21` (see [`build-android.cmd`](../../build-android.cmd)) and have every AHB symbol stripped from `libncnn.a`. If you link the official prebuilt and try to use any of `VkAndroidHardwareBufferImageAllocator`, `VkImageMat::from_android_hardware_buffer`, or `ImportAndroidHardwareBufferPipeline`, the linker will fail with `undefined symbol: ...`.
- A Vulkan-capable GPU that supports `VK_ANDROID_external_memory_android_hardware_buffer`. Verified on Adreno 830 (Vulkan 1.3.284) and Mali-G925 Immortalis MC12 (Vulkan 1.3.278) — both report `spec version 5`. The camera HAL on each surfaces the same YCbCr externalFormat (`AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420`); only the auxiliary usage bits differ between vendors and are not consumed by this path.

Quick runtime capability check:

```cpp
ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
const ncnn::GpuInfo& gi   = vkdev->info;
const int ahb_ext = gi.support_VK_ANDROID_external_memory_android_hardware_buffer();
// > 0 -> supported (the integer is the extension's spec version).
```

### setup the AImageReader

The HAL-provided `AHardwareBuffer` is only Vulkan-importable if the `AImageReader` was created with `AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE` in its usage mask. `AImageReader_new` does NOT include that flag — use `AImageReader_newWithUsage`:

```cpp
AImageReader* reader = nullptr;
AImageReader_newWithUsage(
    width, height, AIMAGE_FORMAT_YUV_420_888,
    AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN
        | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE,
    /*maxImages=*/3, &reader);
```

Verify the first frame's AHB actually carries the flag via `AHardwareBuffer_describe` — some HALs silently drop usage bits they don't support.

### feed an AHB into ncnn

Inside `AImageReader`'s `onImageAvailable` callback, after `AImage_getHardwareBuffer(image, &ahb)`:

```cpp
ncnn::Option opt;
opt.use_vulkan_compute  = true;
opt.use_fp16_storage    = false;  // keep float32 unless you matched it elsewhere
opt.blob_vkallocator    = vkdev->acquire_blob_allocator();
opt.staging_vkallocator = vkdev->acquire_staging_allocator();

ncnn::VkAndroidHardwareBufferImageAllocator alloc(vkdev, ahb);
ncnn::VkImageMat src = ncnn::VkImageMat::from_android_hardware_buffer(&alloc);

ncnn::ImportAndroidHardwareBufferPipeline pipe(vkdev);
pipe.create(&alloc,
            /*type_to=*/1 /*PIXEL_RGB*/,
            /*rotate_from=*/1 /*identity*/,
            /*target_w=*/width, /*target_h=*/height,
            opt);

ncnn::VkMat dst;
dst.create(width, height, /*c=*/3, /*elemsize=*/4u, /*elempack=*/1, opt.blob_vkallocator);

ncnn::VkCompute cmd(vkdev);
cmd.record_import_android_hardware_buffer(&pipe, src, dst);

// `dst` is now a regular ncnn::VkMat — feed it into your extractor as the
// first-blob input. IMPORTANT: see the "ex.input(VkMat) does not auto-convert"
// caveat below — if your network was loaded with use_fp16_storage or
// use_fp16_packed you must pre-cast `dst` to the matching (elempack, dtype)
// via vkdev->convert_packing BEFORE ex.input, or extractor will recurse on
// itself trying to resolve an implicit converter.

cmd.submit_and_wait();

vkdev->reclaim_blob_allocator(opt.blob_vkallocator);
vkdev->reclaim_staging_allocator(opt.staging_vkallocator);
```

`type_to` chooses the output channel order: `1 = RGB, 2 = BGR, 3 = GRAY, 4 = RGBA, 5 = BGRA` (see [`convert_ycbcr.comp`](../../src/convert_ycbcr.comp)). `rotate_from` is the EXIF orientation code (`1` identity, `5`–`8` for 90° rotations). When `target_w/h` differ from the AHB's dimensions the same compute shader does the resize during YCbCr → RGB conversion.

### caveats

**`AHardwareBuffer_acquire` for any cache.** `VkAndroidHardwareBufferImageAllocator` does not take a ref on its AHB. If the allocator (or its `VkImageMat`) is going to outlive the `AImage` the AHB came from, the caller must `AHardwareBuffer_acquire(ahb)` before `AImage_delete(image)` and `AHardwareBuffer_release(ahb)` when evicting the cache entry.

**Per-frame `pipe.create` is expensive (~24 ms median on Adreno 830).** The pipeline depends only on the immutable sampler (= the `samplerYcbcrConversion` derived from the AHB's `externalFormat`) and the `(type_to, rotate_from, target_w, target_h)` specialisation constants, all of which are stable for a given camera session. `AImageReader` cycles through a small AHB pool — typically near `maxImages` under steady state, though the camera HAL may transiently allocate extras across surface/focus events — so caching the allocator + pipeline by AHB pointer keeps the cache bounded and brings the steady-state setup cost from 24 ms to a few microseconds:

```cpp
struct CacheEntry {
    ncnn::VkAndroidHardwareBufferImageAllocator* alloc;
    ncnn::ImportAndroidHardwareBufferPipeline*   pipe;
    ncnn::VkImageMat                             src;
};
static std::unordered_map<AHardwareBuffer*, CacheEntry> cache;  // camera-thread only

auto it = cache.find(ahb);
if (it == cache.end()) {
    AHardwareBuffer_acquire(ahb);
    auto* a = new ncnn::VkAndroidHardwareBufferImageAllocator(vkdev, ahb);
    auto  s = ncnn::VkImageMat::from_android_hardware_buffer(a);
    auto* p = new ncnn::ImportAndroidHardwareBufferPipeline(vkdev);
    p->create(a, 1, 1, width, height, opt);
    cache[ahb] = { a, p, std::move(s) };
    it = cache.find(ahb);
}
const CacheEntry& e = it->second;
cmd.record_import_android_hardware_buffer(e.pipe, e.src, dst);
```

**Read-only input.** The imported `VkImage` is created with `VK_IMAGE_USAGE_SAMPLED_BIT` only. Don't try to write to the AHB through this path; for an output AHB allocate a separate exportable one and use `vkGetMemoryAndroidHardwareBufferANDROID`.

**The convert_ycbcr shader scales by 255.** Output is RGB float in the `[0, 255]` range to match `from_pixels`'s convention, NOT `[0, 1]`. If your model expects normalised input, fold the `1/255` into a post-import scale (a `Scale` layer in the graph, or a small custom compute pass) — `record_import_android_hardware_buffer` does not normalise.

**`ex.input(VkMat)` does not auto-convert format.** Unlike `ex.input(Mat)` — which implicitly uploads and re-packs the host blob to whatever `(elempack, dtype)` the next layer expects — `ex.input(VkMat)` stores the user-provided VkMat verbatim into `blob_mats_gpu`. If the network was loaded with `use_fp16_storage` or `use_fp16_packed` (common on the Vulkan path), and `dst` is the raw `fp32, elempack=1` output of `record_import_android_hardware_buffer`, the first conv layer's bottom-blob lookup will see a format mismatch. `NetPrivate::forward_layer` then walks the producer chain looking for an implicit converter that doesn't exist, recurses into itself, and stack-overflows (observed: ~180 frames at `net.cpp:251`). **Pre-cast `dst` via `vkdev->convert_packing(dst, dst_casted, target_elempack, cast_type_to, cmd, opt)` to the format the network expects before calling `ex.input`.**

### troubleshooting

- `ld.lld: undefined symbol: ncnn::VkAndroidHardwareBufferImageAllocator::...` → ncnn was built with `__ANDROID_API__ < 26`. Rebuild ncnn yourself with `-DANDROID_PLATFORM=android-26` or higher; do not rely on the official prebuilt. Symbol-level diagnosis is in [#5531](https://github.com/Tencent/ncnn/issues/5531).
- `VkImageMat::from_android_hardware_buffer` returns an empty mat → check that `AImageReader_newWithUsage` was used (not `AImageReader_new`) and the AHB descriptor actually carries `GPU_SAMPLED_IMAGE` (`desc.usage & 0x100`). [#5190](https://github.com/Tencent/ncnn/issues/5190) is the prior `elemsize`-related bug — fixed in ncnn since.
