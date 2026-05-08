# ncnn vulkan pipeline cache

ncnn has a pipeline cache for vulkan pipeline creation. It avoids creating the same shader module, descriptor set layout, pipeline layout, compute pipeline and descriptor update template repeatedly inside one process.

Creating vulkan compute pipelines may also be expensive because the driver has to translate spir-v to device-specific code. The first model load or first inference may therefore have visible latency.

ncnn can save reusable spir-v data and vulkan driver pipeline cache data to a single file, and load them in the next run.

## motivation

vulkan pipeline creation cost depends heavily on the driver, the gpu and the shader. Some drivers compile quickly, while others may spend a lot of time in `vkCreateComputePipelines()`.

The goal of the pipeline cache is

- reduce cold-start latency
- share one pipeline artifact among layers that request the same pipeline in one process
- reuse compiled spir-v for builtin ncnn shaders
- feed driver pipeline cache data back to vulkan
- reject stale cache data strictly when shader, platform, gpu or driver changes

The cache is only a performance optimization. Loading failure should be treated as cache miss. ncnn will rebuild the pipelines normally.

## architecture overview

The ncnn vulkan pipeline cache has three parts.

```text
live PipelineCache object
  cache_digests        -> keys for ncnn pipeline requests
  cache_artifacts      -> live VkShaderModule / VkPipeline / layouts
  cache_spirv_entries  -> compiled spir-v bytes for builtin ncnn shaders
  vk_pipeline_cache    -> VkPipelineCache object owned by vulkan driver

persistent cache file
  ncnn cache header
  spir-v cache section
  vulkan pipeline cache blob
```

The live `PipelineCache` object is always used for pipeline creation. The cache file stores data that can speed up future pipeline creation, but it does not store live vulkan objects.

### in-memory pipeline object cache

`PipelineCache` keeps a digest for each pipeline request and stores the live vulkan objects created for it. When the same pipeline is requested again in the same process, ncnn reuses the cached artifact directly without calling `vkCreateComputePipelines()` again.

For builtin ncnn shaders, the digest is made from

- shader type index
- option bits that affect shader generation
- local workgroup size
- subgroup size
- specialization constants

For raw spir-v shaders, the digest uses the spir-v data hash instead of shader type index.

The cached artifact contains

- `VkShaderModule`
- `VkDescriptorSetLayout`
- `VkPipelineLayout`
- `VkPipeline`
- `VkDescriptorUpdateTemplateKHR`
- resolved `ShaderInfo`

This part is not written to disk because these objects are valid only for the current `VkDevice`.

The in-memory cache also owns these live vulkan objects. `Pipeline` instances keep the handles, but object destruction is handled by `PipelineCache::clear()` or the `PipelineCache` destructor.

### spir-v cache

Builtin ncnn layer shaders are stored as glsl source in `layer_shader_registry`.

For builtin shader creation, ncnn caches the compiled spir-v bytes in memory and can save them to disk. Each spir-v entry is keyed by

- `shader_type_index`
- option bits that affect shader generation
- per shader source hash
- platform and driver identity from the file header

The hash is per shader entry, not a global registry hash. Updating one shader only invalidates the entries compiled from that shader source.

External spir-v data passed through `Pipeline::create(const uint32_t*)` is not stored in the spir-v section because ncnn has no stable shader registry index for it.

### vulkan driver pipeline cache

ncnn creates a `VkPipelineCache` object and passes it to `vkCreateComputePipelines()`.

When saving, ncnn calls `vkGetPipelineCacheData()` and stores the driver cache blob in the same file. When loading, ncnn validates the blob header and creates a temporary `VkPipelineCache` from it, then merges it into the current cache object.

Some drivers are known to have broken online pipeline cache behavior. On those devices, ncnn does not rely on the driver pipeline cache.

## file format

The cache file is one binary file.

It contains

1. ncnn cache header
2. spir-v cache section
3. vulkan pipeline cache blob

The header records the environment that must match before any cached data can be used.

- magic and cache format version
- ncnn version
- endian marker and pointer size
- vulkan vendor id and device id
- vulkan api version and driver version
- driver id and driver name hash
- gpu device name hash
- pipeline cache uuid
- spir-v section size and hash
- vulkan pipeline cache blob size and hash

The spir-v section contains one header for each cached shader.

- shader type index
- option bits
- per shader source hash
- spir-v byte size
- spir-v data hashes

The vulkan pipeline cache blob is also checked against the standard `VkPipelineCacheHeaderVersionOne` fields.

If any check fails, `load_cache()` returns non-zero and the whole file is rejected. This is intentional. A driver pipeline cache blob may contain pipelines compiled from multiple shaders, so partial loading is not safe for that section.

## implementation flow

The common builtin layer path is

```text
Pipeline::create(shader_type_index, opt, specializations)
  -> PipelineCache::get_pipeline()
     -> find live artifact by cache_digests
     -> return cache_artifacts directly on hit
     -> create_shader_module()
        -> find spir-v by shader_type_index + opt_bits + shader_source_hash
        -> or compile glsl to spir-v and remember it
     -> new_pipeline()
        -> create descriptor set layout
        -> create pipeline layout
        -> vkCreateComputePipelines(..., vk_pipeline_cache, ...)
        -> create descriptor update template
     -> store live artifact in memory
```

The raw spir-v path is similar, but skips the builtin shader source cache. It still uses the in-memory artifact cache and the vulkan driver pipeline cache.

`VulkanDevice::create_pipeline()` has an overload that accepts `VkPipelineCache`, so callers that need direct pipeline creation can still pass their own cache object.

## api usage

For file based usage, get the vulkan device first, set the same device to `Net`, create a `PipelineCache` from that device, load the cache file, and save it after pipelines have been created.

```cpp
ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(0);

ncnn::PipelineCache pipeline_cache(vkdev);

// cache miss is not an error for application logic
pipeline_cache.load_cache("model.ncnn.vkcache");

ncnn::Net net;
net.opt.use_vulkan_compute = true;
net.set_vulkan_device(vkdev);
net.opt.pipeline_cache = &pipeline_cache;

net.load_param("model.param");
net.load_model("model.bin");

// optional warmup inference here to create lazy pipelines

pipeline_cache.save_cache("model.ncnn.vkcache");
```

The memory api is available when the application wants to manage file I/O itself.

```cpp
std::vector<unsigned char> cache_data;
pipeline_cache.save_cache(cache_data);

ncnn::PipelineCache pipeline_cache2(vkdev);
pipeline_cache2.load_cache(cache_data);
```

The `FILE*` and path helpers are available when `NCNN_STDIO` is enabled.

The C api exposes the same ownership model through `ncnn_pipelinecache_t`.

```c
int device_index = 0;

ncnn_pipelinecache_t pipeline_cache = ncnn_pipelinecache_create(device_index);
ncnn_pipelinecache_load(pipeline_cache, "model.ncnn.vkcache");

ncnn_net_t net = ncnn_net_create();
ncnn_net_set_vulkan_device(net, device_index);

ncnn_option_t opt = ncnn_net_get_option(net);
ncnn_option_set_use_vulkan_compute(opt, 1);
ncnn_option_set_pipeline_cache(opt, pipeline_cache);

ncnn_net_load_param(net, "model.param");
ncnn_net_load_model(net, "model.bin");

ncnn_pipelinecache_save(pipeline_cache, "model.ncnn.vkcache");

ncnn_net_destroy(net);
ncnn_pipelinecache_destroy(pipeline_cache);
```

## best practice

### use the same vulkan device

Create `PipelineCache` from the same `VulkanDevice` used by `Net`.

The recommended order is

1. get `VulkanDevice` by `ncnn::get_gpu_device()`
2. call `net.set_vulkan_device(vkdev)`
3. create `PipelineCache pipeline_cache(vkdev)`
4. load cache data by `pipeline_cache.load_cache()`
5. set `net.opt.pipeline_cache = &pipeline_cache`
6. call `load_param()` and `load_model()`

`net.vulkan_device()` returns a valid device after `set_vulkan_device()` or after the net has initialized vulkan internally. Calling `set_vulkan_device()` explicitly makes the ownership and cache-device match clear.

### set the cache before load_model

Most layer pipelines are created during `load_model()`. Set `net.opt.pipeline_cache` before calling `load_model()` if you want cache data to participate in model loading.

For models or usages that create pipelines lazily, run one warmup inference before saving.

### treat load failure as cache miss

Do not make application startup depend on `load_cache()` success.

Cache rejection is expected after

- ncnn library update
- shader source update
- model option change that affects shader generation
- gpu switch
- driver update
- vulkan runtime change
- cache file corruption

The normal policy is

```cpp
if (pipeline_cache.load_cache(cache_path) != 0)
{
    // ignore and rebuild
}
```

### keep one cache file per model and device class

ncnn validates the gpu and driver identity before using the file, but a clear file naming rule still helps deployment and debugging.

For example

```text
model-name.ncnn.vkcache
```

or include your own model version if the application updates models independently.

### do not edit or merge cache files manually

The file stores binary spir-v and driver-owned pipeline cache data. It is not a portable interchange format.

Use `load_cache()` and `save_cache()` only. If the file is rejected, rebuild it.

### avoid concurrent writers

Saving by path writes a unique temporary file and replaces the destination. This avoids partially written cache files, but it does not serialize the final replace among multiple processes.

If several processes may run the same model at the same time, let only one process write the cache file, or write to separate files.

### keep the PipelineCache alive while using the Net

When `net.opt.pipeline_cache` points to an application-created `PipelineCache`, the cache owns the live vulkan pipeline objects returned to ncnn layers.

Keep the `PipelineCache` object alive while the `Net` can still run inference with those pipelines.

## limitations

The cache does not guarantee faster startup on every driver. Some drivers may return small or ineffective pipeline cache data, and some devices disable online pipeline cache usage because of known bugs.

The cache file is not portable across different ncnn builds, different gpu devices or different driver versions.

The cache improves pipeline creation cost. It does not cache uploaded model weights, blob allocators or inference outputs.
