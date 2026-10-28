# vulkan weight storage

Vulkan weights normally stay in device memory. `use_weights_in_host_memory` prefers
host memory. On discrete GPUs, host weights are copied through a device cache
before execution, so this option also consumes device cache memory.
`use_weights_in_disk` keeps weights in a temporary file and loads them through
a cache during inference. Disk storage can reduce the device
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
inference repeatedly. Each Extractor owns its internal command and weight cache;
reuse the Extractor to retain its cache allocations. Clearing or destroying the Extractor
releases it. Net retains model weights and finalized storage settings, not idle
execution caches.

## memory settings

| option | meaning |
| --- | --- |
| `use_weights_in_disk` | enable temporary file storage; takes precedence over `use_weights_in_host_memory` |
| `weight_device_memory_limit` | device memory available for resident weights and one cache |
| `weight_host_memory_limit` | host memory available for resident weights and one cache |
| `weight_device_cache_size` | device cache capacity; zero selects it automatically |
| `weight_host_cache_size` | host cache capacity; zero selects it automatically |

With both memory limits set to zero, disk weights use the cache without retaining
an additional resident subset. With explicit limits, loading first reserves the
configured exchange capacities, then places weights in loading order: device
memory, host memory, and finally the file. Placement uses the remaining budget
and never sorts or relocates the complete model. A model that fits in the
remaining device budget needs no weight cache allocation.

Discrete GPUs require both memory limits when enabling resident placement.
Integrated GPUs and CPU Vulkan devices use one device allocation accessible by
the host. For these devices, one positive limit is enough; if both are positive,
the smaller value applies.

Automatic caches are allocated lazily. The initial device target is 128 MiB
(or the first operation's complete working set if larger), and the separate host
cache target is 16 MiB. Both are clamped to the remaining memory budgets,
including Vulkan allocation padding. Unified-memory devices use only the device
cache. Device-resident weights are not copied into the exchange cache;
host-resident weights still need device cache space on discrete GPUs.

A device cache holds complete weights. The host cache can be smaller than a
weight because uploads use several chunks. Explicit cache sizes apply to
Net-managed and external commands, including when both memory limits are zero.
Sizes are aligned to Vulkan requirements and checked against the memory limits.
An explicit device cache smaller than the weights needed by one operation fails;
it is not replaced by an automatic size.

Once recording references the backing, its capacity is fixed. A later operation
whose simultaneous bindings exceed that capacity fails instead of reallocating
referenced storage or exceeding the budget. Clear the cache after resetting the
command, then recreate the command with a larger explicit cache for such
workloads. Loading-time
maximum working-set derivation is not implemented. Upload order and layer
boundaries do not imply simultaneous use; the actual dispatch bindings determine
which weights must fit together. Loading reserves the configured cache capacity
before allocating residents. Automatic reservation uses the 128 MiB device and
16 MiB host defaults, clamped to the limits. When a limit is smaller than its
default, the exchange reservation can consume the entire budget for that tier.
Use an explicit smaller cache to leave room for residents. Capacity errors may
be reported at inference when the actual dispatch bindings are known.

## file directory

`Net::set_weight_file_directory()` copies a UTF-8 path. Windows also supports a
`const wchar_t*` path. Set it before loading weights; `Net::clear()` retains the
setting. A null or empty path selects the system temporary directory.

POSIX uses `TMPDIR` when set and otherwise uses `/tmp`. Windows uses
`GetTempPathW`. Temporary files are removed automatically when their owning
weights are released. Space is reserved before exposing writable mappings.

If the default directory is unavailable and no memory limits were configured,
loading can fall back to resident weights. An explicit directory or explicit
memory limits make file creation failure a load error. File reservation,
mapping and finalization failures also fail loading.

Disk storage requires `NCNN_STDIO` and a supported file mapping platform:
Windows, Linux, Android API 21 or newer, OpenHarmony or Apple platforms. Other builds use the
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

`cmd.reset()` discards the recording and all weight-content hits while retaining
successful exchange allocations for reuse. The next recording uploads its
weights again. `cmd.clear_weight_cache()` releases the allocations after work
has completed and the recording has been reset. Model-owned resident weights
remain available. Cache allocations and byte-range views are refcounted `VkMat`
resources. Recorded commands retain source handles, but the model backing must
remain valid and immutable until execution completes and the recording is reset.
Allocators must outlive all their views. Destroying or clearing a model allocator
while recorded work refers to it is unsupported.
Clearing a Net invalidates its weights and existing Extractors. To replace or
reload a model, use `clear()`, `load_param()` and `load_model()`. Repeated
`load_model()` calls on finalized weights are not a hot reload interface. After
a loading error, clear and reload the model before inference. A GPU submission
failure can discard the affected command cache; retaining it is not guaranteed.

`VkCompute` owns both upload and compute recording, submission and completion.
Cached dependencies and Vulkan commands are prepared incrementally while
recording, without a speculative whole-command plan or a GPU submission.
Submission fills and flushes bounded staging regions only after their previous
GPU readers have completed, then submits the already-recorded tasks. It does not
copy every weight into a separate retained staging allocation.

Ordinary work before the first paged operation joins the first compute task and
waits for its uploads, unless a cross-queue cache-ownership release must precede
those uploads. There is no artificial empty preliminary submission. Commands
without paged weights retain the normal recording path.

Exchange ranges are assigned in execution order. When the remaining space cannot
hold the next complete dispatch working set, allocation wraps to the start and
discards all prior content hits. The previous readers finish before those bytes
are overwritten. Uploads into unused space can overlap earlier compute work.
Recorded tasks retain owning `VkMat` views of the exchange allocation; reference
counting does not replace GPU completion checks.

Each paged execution uses independent upload and compute command buffers and
synchronization objects. They are not recycled while that execution is running.
Staging-range and cache-overwrite dependencies still wait for their actual users.
Execution resources are released only after all submissions and semaphore waits
complete; the weight cache itself can survive `reset()`. A nonterminal fence
error keeps work pending and prevents reset. Destruction waits for completion
(or device loss), so persistent driver errors can block destruction.

For direct allocator use, configure `VkWeightAllocator` before allocation with
`set_weights_in_disk()`, `set_weight_memory_limit(device_limit, host_limit,
device_cache_size, host_cache_size)` and optionally `set_weight_file_directory()`.
Check the return value of `set_weight_memory_limit()` before allocating weights;
an invalid configuration returns an error and leaves the previous settings intact.
Allocation failures return a null buffer, observed as an empty `VkMat`.
Record each transformed weight with `VkTransfer::record_upload()`, check errors,
and submit and wait before calling `finalize_weights()`. Finalization flushes
and seals file mappings; resident weights have already been uploaded directly
to their final locations. Keep the allocator alive until its views and recorded
work have been released. Uploads use bounded staging and batch small copies
together. Net submits when pending uploads exceed 256 MiB, including retained
CPU snapshots. The check runs after each layer, so the pending total can exceed
the threshold by one layer's uploads. File-backed weights are copied directly
into their mappings and do not retain pending upload snapshots.

`get_weight_memory_usage()` reports actual device and host resident Vulkan
allocation bytes, including padding. It excludes file mappings and command
caches; it does not report past errors or certify that uploads have completed.
Check allocation results and the return value of `finalize_weights()` directly.
Cache capacities come from the command's options. An internal command subtracts
resident allocation usage from positive memory limits before allocating its
cache. Zero limits remain unbounded; exhausted positive limits are not converted
into an unbounded cache budget.

## best practice

- Start with automatic cache sizes and set memory limits appropriate for the
  device. Leave room for activations, KV caches and the rest of the application.
- Use a writable application cache directory on a disk filesystem. A temporary
  directory backed by tmpfs still consumes memory.
- Reuse the Net and Extractor to retain model storage and the internal weight
  cache. For external command recording, reuse the command with `reset()` between
  submissions.
- Measure on the target device. File speed, host bandwidth and Vulkan queue
  support affect the benefit of disk weights.

The `perf_weight_storage` target compares device, host and disk storage using a
fixed workload. Build with `NCNN_VULKAN=ON` and `NCNN_BUILD_TESTS=ON`, then run
`tests/perf/perf_weight_storage` without arguments. It reports model loading and
inference times. `NCNN_BENCHMARK` additionally prints a cache summary with bytes
read, bytes uploaded and elapsed time.

## limitations

Memory limits cover Vulkan weight allocations, including allocation alignment.
They are not a process or device-wide quota. Concurrent commands and live
Extractors with internal commands each retain their own cache. Activations, KV
caches, CPU model preparation, OS file cache and driver allocations are outside
these limits.

Disk storage currently supports buffer weights. Weight images are unsupported
with explicit resident memory limits. The largest simultaneous set of weight
bindings still needs to fit the device cache.

Mapped files consume virtual address space, which can constrain large models in
32-bit processes. Disk I/O failures after a mapping has been established follow
the operating system's mapped-file error behavior.

Separate Extractors and commands can run concurrently. A single Extractor or
command is used by one thread at a time. Changing model weights while recorded
work still refers to them is unsupported.

Applications and custom allocators must be rebuilt with the matching ncnn
headers and library. The added weight metadata changes `VkBufferMemory` layout;
appending fields does not preserve compatibility with older binary allocators.
