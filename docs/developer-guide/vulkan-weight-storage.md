# vulkan weight storage

Vulkan weights normally stay in device memory. `use_weights_in_host_memory` prefers
host memory, while `use_weights_in_disk` keeps weights in a temporary file and
loads them through a cache during inference. Disk storage can reduce the device
and host memory needed for large models, at the cost of file reads and uploads.

## usage

Configure storage before loading the model. The following example loads a model
with disk weights and extracts one output:

```cpp
#include "net.h"

int inference(const ncnn::Mat& input, ncnn::Mat& output)
{
    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.opt.use_weights_in_disk = true;
    net.opt.weight_device_memory_limit = (size_t)512 * 1024 * 1024;
    net.opt.weight_host_memory_limit = (size_t)128 * 1024 * 1024;

    // optional writable directory, copied by net
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

The input and output names depend on the model. Keep the Net loaded when running
inference repeatedly; each call can create a new Extractor.

## memory settings

| option | meaning |
| --- | --- |
| `use_weights_in_disk` | enable temporary file storage; takes precedence over `use_weights_in_host_memory` |
| `weight_device_memory_limit` | device memory available for resident weights and one cache |
| `weight_host_memory_limit` | host memory available for resident weights and one cache |
| `weight_device_cache_size` | device cache capacity; zero selects it automatically |
| `weight_host_cache_size` | host cache capacity; zero selects it automatically |

With both memory limits set to zero, disk weights use the cache without retaining
an additional resident subset. With explicit limits, loading keeps smaller
weights in device memory, then host memory, and leaves the rest in the file.
A model that fits entirely in device memory needs no weight cache.

Discrete GPUs require both memory limits when enabling resident placement.
Integrated GPUs and CPU Vulkan devices use one device allocation accessible by
the host. For these devices, one positive limit is enough; if both are positive,
the smaller value applies.

Automatic cache sizes account for weight sizes, simultaneous bindings, Vulkan
alignment and device limits. A device cache holds complete weights. The host
cache can be smaller than a weight because uploads can use several chunks.
Explicit cache sizes are checked rather than silently reduced.

## file directory

`Net::set_weight_file_directory()` copies a UTF-8 path. Windows also supports a
`const wchar_t*` path. Set it before loading weights; `Net::clear()` retains the
setting. A null or empty path selects the system temporary directory.

POSIX uses `TMPDIR` when set and otherwise uses `tmpfile()`. Windows uses
`GetTempPathW`. Temporary files are removed automatically when their owning
weights are released. Space is reserved before exposing writable mappings.

If the default directory is unavailable and no memory limits were configured,
loading can fall back to resident weights. An explicit directory or explicit
memory limits make file creation failure a load error. File reservation,
mapping and finalization failures also fail loading.

Disk storage requires `NCNN_STDIO` and a supported file mapping platform:
Windows, Linux, Android, OpenHarmony or Apple platforms. Other builds use the
resident fallback when no explicit directory or memory limits are configured.

## external commands

Normal `Mat` extraction manages the cache automatically. Applications that record
several Extractors into one command can configure that command directly:

```cpp
ncnn::Option opt;
opt.weight_device_cache_size = (size_t)64 * 1024 * 1024;
opt.weight_host_cache_size = (size_t)8 * 1024 * 1024;
ncnn::VkCompute cmd(net.vulkan_device(), opt);
```

Continue to set the Extractors' usual blob, workspace and staging allocators.
All participating Nets use the same Vulkan device. An external command has its
own cache settings; options passed to clone or download operations do not change
them.

`cmd.reset()` discards the recording and retains its cache for reuse.
`cmd.clear_weight_cache()` releases the cache after work has completed and the
recording has been reset. Model-owned resident weights remain available.
Keep model weights and their allocators alive until recorded work completes.
Clearing a Net invalidates its weights and existing Extractors.

For direct allocator use, configure `VkWeightAllocator` before allocation with
`set_weights_in_disk()`, `set_weight_memory_limit()` and optionally
`set_weight_file_directory()`. After the initial uploads complete,
`finalize_weights()` flushes the file and plans resident placement. Reset the
`VkTransfer`, call `record_upload(&weights, opt)` to fill those residents, and
finish with `submit_and_wait()`. Net performs this sequence automatically.

## best practice

- Start with automatic cache sizes and set memory limits appropriate for the
  device. Leave room for activations, KV caches and the rest of the application.
- Use a writable application cache directory on a disk filesystem. A temporary
  directory backed by tmpfs still consumes memory.
- Reuse the Net and, for external command recording, reuse the command with
  `reset()` between submissions.
- Measure on the target device. File speed, host bandwidth and Vulkan queue
  support affect the benefit of disk weights.

The `perf_weight_storage` target compares device, host and disk storage using a
fixed workload. Build with `NCNN_VULKAN=ON` and `NCNN_BUILD_TESTS=ON`, then run
`tests/perf/perf_weight_storage` without arguments. It reports model loading and
inference times. `NCNN_BENCHMARK` additionally prints a cache summary with bytes
read, bytes uploaded and elapsed time.

## limitations

Memory limits cover Vulkan weight allocations, including allocation alignment.
They are not a process or device-wide quota. Concurrent commands and cached idle
commands each retain their own cache. Activations, KV caches, CPU model
preparation, OS file cache and driver allocations are outside these limits.

Disk storage currently supports buffer weights. Weight images are unsupported
with explicit resident memory limits. The largest simultaneous set of weight
bindings still needs to fit the device cache.

Mapped files consume virtual address space, which can constrain large models in
32-bit processes. Disk I/O failures after a mapping has been established follow
the operating system's mapped-file error behavior.

Separate Extractors and commands can run concurrently. A single Extractor or
command is used by one thread at a time. Changing model weights while recorded
work still refers to them is unsupported.
