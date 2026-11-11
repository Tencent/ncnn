# vulkan weight storage

Vulkan weights normally stay in device memory. For large models on devices with
little memory, two options reduce resident weight storage at the cost of extra
copies during inference:

- `use_weights_in_host_memory` keeps weights in host memory.
- `use_weights_in_disk` keeps weights in a temporary file, optionally with part of
  the model resident in device and host memory.

Host and file weights are uploaded into temporary Vulkan buffers when a layer uses
them. Host storage prefers a host-visible Vulkan buffer. If its creation or
allocation fails, or no suitable host memory type exists, weights fall back to
ordinary aligned CPU memory, and later weights stay in CPU memory.

| storage_type | weight backing | execution |
| --- | --- | --- |
| 0 | device Vulkan buffer | bound directly |
| 1 | host Vulkan buffer | uploaded before use |
| 2 | CPU memory | uploaded before use |
| 3 | mapped file | uploaded before use |

## usage

Configure storage before loading the model:

```cpp
#include "net.h"

int inference(const ncnn::Mat& input, ncnn::Mat& output)
{
    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.opt.use_weights_in_disk = true;
    net.opt.weight_device_memory_limit = (size_t)512 * 1024 * 1024;
    net.opt.weight_host_memory_limit = (size_t)128 * 1024 * 1024;

    // writable directory on a disk filesystem, copied by net
    net.set_weight_file_directory("/path/to/cache");

    int ret = net.load_param("model.param");
    if (ret != 0)
        return ret;

    ret = net.load_model("model.bin");
    if (ret != 0)
        return ret;

    ncnn::Extractor ex = net.create_extractor();
    ret = ex.input("input", input);
    if (ret != 0)
        return ret;

    return ex.extract("output", output);
}
```

Keep the Net loaded when running inference repeatedly.

## memory settings

| option | meaning |
| --- | --- |
| `use_weights_in_disk` | store weights in a temporary file; takes precedence over `use_weights_in_host_memory` |
| `weight_device_memory_limit` | device memory for resident disk-model weights, zero disables the tier |
| `weight_host_memory_limit` | host memory for resident disk-model weights, zero disables the tier |

With both limits zero, all disk weights stay in the file. Otherwise loading places
weights in loading order into device memory, then host memory, then the file.
Placement accounts for allocation padding. Host Vulkan buffers and CPU blocks share
the host limit. Integrated GPUs and CPU Vulkan devices have a single resident
tier, limited by the smaller positive limit.

If a resident tier fails to allocate, for example when free device memory is below
its limit, it is treated as full: the weight and all later weights go to the next
tier, and the fallback is logged once.

The limits cover model-owned resident weights only. Activations, KV caches,
temporary weights, staging buffers, allocator pool reserves and the OS file cache
are outside them. Each command uploads the weights of an operation into a private
temporary pool and keeps the preceding operation's weights until its work
completes, so leave room for the weights of two adjacent operations.

## file directory

`Net::set_weight_file_directory()` copies a UTF-8 path; Windows also accepts a
`const wchar_t*` path. Set it before loading weights; `Net::clear()` keeps it. A
null or empty path selects the system temporary directory: `TMPDIR` or `/tmp` on
POSIX, `GetTempPathW` on Windows.

Use an application cache directory on a disk filesystem. `/tmp` is often a tmpfs,
which still consumes memory. Android has no usable default directory; pass the
application's cache directory. Temporary files are removed automatically, and file
space is reserved before weights are written.

If the default directory is unavailable and no memory limits are set, weights stay
resident instead. With an explicit directory or memory limits, file creation,
reservation and mapping failures are logged and leave the affected weights empty,
like other weight allocation failures; a failed file creation is logged once.
Finalization failures fail loading.

Disk storage requires `NCNN_STDIO` on Windows, Linux, Android API 21 or newer,
OpenHarmony or Apple platforms. Other builds keep weights resident when no explicit
directory or memory limits are set.

## external commands

External commands need no extra setup. Recording an operation that uses host or
disk weights uploads them within that command, including when several Nets with
different weight storage share the command. Uploads can overlap the preceding
compute work. `submit_and_wait()` waits for pending uploads.

Each command owns its temporary weight pool, separate from the Extractor's
allocators. Reuse a command with `reset()` to keep its pool. If an operation's
temporary weights cannot be allocated, the operation is skipped with a log, like
other recording failures: `record_pipeline()` skips the dispatch, and weight clones
leave their output empty. The command remains usable.

## allocator api

For direct use, configure `VkWeightAllocator` before allocating weights with
`set_weights_in_disk()`, `set_weight_memory_limit()` and optionally
`set_weight_file_directory()` with a UTF-8 path. Upload each weight with
`VkTransfer::record_upload()`, check `submit_and_wait()`, then call
`finalize_weights()` to flush and seal file weights before recording their use with
`VkCompute`. No file weights can be allocated after finalization. Allocation
failures return an empty `VkMat`. `Net` submits uploads whenever more than 256 MiB
is pending after a layer.

`get_weight_memory_usage()` reports the actual device and host resident bytes,
including padding. The host count includes CPU blocks. File mappings, temporary
weights and pool reserves are excluded.

## performance

File speed, host bandwidth and Vulkan queue support decide the cost of host and
disk weights, so measure on the target device. The `perf_weight_storage` target,
built with `NCNN_VULKAN=ON` and `NCNN_BUILD_TESTS=ON`, compares device, host and
disk storage on a fixed workload and reports loading and inference times.

## limitations

- Disk storage supports buffer weights. Image weights are unsupported with resident
  memory limits.
- There is no tensor tiling: the weights of each operation must fit in temporary
  memory together.
- Mapped files consume virtual address space, which can constrain large models in
  32-bit processes. I/O errors after mapping follow the operating system's
  mapped-file behavior.
- Separate Extractors and commands can run concurrently; each Extractor or command
  is used by one thread at a time. Weights must not change while recorded work
  still refers to them.
- `VkBufferMemory` gained a `storage_type` field, so custom allocators must be
  rebuilt with the matching headers. Allocators that do not assign it return
  device buffers.
