# vulkan weight storage

Vulkan weights normally stay in device memory. `use_weights_in_host_memory`
prefers host memory. Host and file weights are resolved into temporary Vulkan
buffers before execution. `use_weights_in_disk` keeps weights in a temporary
file, reducing resident model storage at the cost of file reads and uploads.
Temporary weight buffers use the existing `VkBlobAllocator` implementation.

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
inference repeatedly. Each `Mat` extraction uses a local command. Its default
weight workspace allocator is acquired lazily from the Vulkan device and returned
when the command is destroyed; its implementation is `VkBlobAllocator`.

## memory settings

| option | meaning |
| --- | --- |
| `use_weights_in_disk` | enable temporary file storage; takes precedence over `use_weights_in_host_memory` |
| `weight_device_memory_limit` | limit actual device allocations for resident disk-model weights |
| `weight_host_memory_limit` | limit actual host allocations for resident disk-model weights |

With both limits set to zero, disk weights remain in the file. With explicit
limits, loading places weights in order into device memory, host memory, and
then the file. Placement accounts for Vulkan allocation padding and does not
sort or relocate the model. Device-resident weights need no temporary copy.

Discrete GPUs require both limits when enabling resident placement. Integrated
GPUs and CPU Vulkan devices use one resident allocation tier: one positive limit
is enough, and the smaller value applies when both are positive.

These limits apply to model-owned resident allocations. They do not include
activation buffers, temporary weights, staging, or allocator pool reserves.
There is no separate fixed-capacity weight exchange arena and no
`weight_device_cache_size` or `weight_host_cache_size` setting. Temporary weights
are normal workspace allocations: the allocator can grow its pool, and an
allocation failure is reported by the command. All weights used by one dispatch
remain referenced together until that dispatch has been recorded.

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

Normal `Mat` extraction manages temporary weight storage automatically. An
external command can use a caller-owned allocator:

```cpp
ncnn::VkBlobAllocator weight_workspace(net.vulkan_device());
ncnn::VkCompute cmd(net.vulkan_device());
cmd.workspace_weight_allocator = &weight_workspace;
```

`workspace_weight_allocator` is a non-owning `VkAllocator*`. Null selects the
command's independent default blob allocator. An application can instead point
it to an Extractor's existing `workspace_vkallocator` to share that pool; using
the same implementation does not require sharing the same instance. Configure
it before recording. The allocator must use the same Vulkan device and must
keep pooled backing buffers alive until the recorded work completes. Do not
clear or destroy it while the command refers to its buffers.

Continue to configure each Extractor's blob, workspace and staging allocators
for ordinary layer operations. `cmd.reset()` discards recording state; temporary
weight views return through the selected allocator's normal `fastFree()` path.
Allocator pool reserves follow that allocator's lifetime and `clear()` policy.
There is no `VkCompute::set_weight_memory_limit()` or `clear_weight_cache()`.
Model backing must remain valid and immutable until execution completes and the
recording is reset. Clearing a Net invalidates its weights and existing
Extractors. Use `clear()`, `load_param()` and `load_model()` to replace a model.

The private `record_resolve()` validates a deferred source, creates a temporary
`VkMat` with the selected allocator, records the complete source upload and
returns a view with the original shape, packing and offset. Ordinary buffers
pass through. `record_pipeline()` resolves its local binding array in order.
Multiple views of the same source allocation within a dispatch share one
resolved allocation. Later operations upload the source again; there is no
persistent weight identity cache. `MemoryData` continues to use `record_clone()`
and owns its independent output.

All upload and compute commands for temporary weights use the compute queue
family, so an EXCLUSIVE workspace buffer can also be used by ordinary layers
without queue-family ownership transfers. Queue submissions remain ordered with
semaphores even when different queues are leased from that family. Before an
operation uploads weights, the preceding compute batch is closed and its
completion is connected to the new uploads. This conservatively protects any
workspace address that the allocator may have reused; the allocator does not
track GPU reader serials. It may also reduce upload/compute overlap compared to
a dedicated exchange allocator with per-range dependency tracking.

On unified-memory devices, a mapped target is filled directly after preceding
compute work completes. An unmapped target uses staging. Other devices copy
host-resident Vulkan buffers directly or fill a command-owned staging buffer
from file mappings. File uploads use chunks of at most 16 MiB, waiting for the
previous upload before overwriting staging. These transfer chunks do not slice
the weight's mathematical operation or its shader binding.

Devices with push descriptors encode Vulkan commands immediately. Other devices
create descriptor sets while recording and replay their native commands before
closing the corresponding command buffer. A pipeline bind through its dispatch
remains in one compute batch. Recording allocates resources but does not submit
GPU work or fill deferred file-upload destinations. `submit_and_wait()` executes
the recorded fills, uploads, computation and downloads.

Preparation failures are retained and returned by `submit_and_wait()` without
submitting earlier work. Call `reset()` before recording again after an error.
Ordinary output allocation failures retain the empty-output convention. All
submitted upload and compute work is drained before execution resources are
released. A nonterminal fence error keeps work pending and prevents reset;
destruction waits for completion or device loss.

For direct allocator use, configure `VkWeightAllocator` before allocation with
`set_weights_in_disk()`, `set_weight_memory_limit(device_limit, host_limit)` and optionally `set_weight_file_directory()`.
Check the return value of `set_weight_memory_limit()` before allocating weights;
an invalid configuration returns an error and leaves the previous settings intact.
Allocation failures return a null buffer, observed as an empty `VkMat`.
Record each transformed weight with `VkTransfer::record_upload()`, then check
`submit_and_wait()` before calling `finalize_weights()`. Recording and submission
errors are returned by `submit_and_wait()`. Finalization flushes
and seals file mappings; resident weights have already been uploaded directly
to their final locations. Keep the allocator alive until its views and recorded
work have been released. Uploads use bounded staging and batch small copies
together. Net submits when pending uploads exceed 256 MiB, including retained
CPU snapshots. The check runs after each layer, so the pending total can exceed
the threshold by one layer's uploads. Recording errors are reported at the next
threshold submission or the final submission, so loading may read later layers
before reporting a recording failure. File-backed and host-mappable weights are
copied directly into their mappings while recording and do not retain pending
upload snapshots. Host-mappable Vulkan allocations are flushed at that point;
only unmapped destinations need snapshots and staging transfers.

`get_weight_memory_usage()` reports actual device and host resident Vulkan
allocation bytes, including padding. It excludes file mappings, temporary
weights and workspace pool reserves. Check allocation results and the return
value of `finalize_weights()` directly; memory usage does not certify upload
completion.

## best practice

- Set resident limits while leaving room for temporary weights, allocator pool
  reserves, activations, KV caches and the rest of the application.
- Use a writable application cache directory on a disk filesystem. A temporary
  directory backed by tmpfs still consumes memory.
- Reuse the Net to retain model storage. For external command recording, reuse
  command and workspace allocator with `reset()` between submissions.
- Measure on the target device. File speed, host bandwidth and Vulkan queue
  support affect the benefit of disk weights.

The `perf_weight_storage` target compares device, host and disk storage using a
fixed workload. Build with `NCNN_VULKAN=ON` and `NCNN_BUILD_TESTS=ON`, then run
`tests/perf/perf_weight_storage` without arguments. It reports model loading and
inference times. `NCNN_BENCHMARK` additionally prints an upload summary with bytes
read, bytes uploaded and elapsed time.

## limitations

Memory limits cover resident model-weight Vulkan allocations, including padding.
They are not a process or device-wide quota. Temporary weight workspaces,
activations, KV caches, CPU model preparation, OS file cache and driver
allocations are outside these limits. Concurrent commands normally use separate
workspace allocator instances.

Disk storage currently supports buffer weights. Weight images are unsupported
with explicit resident memory limits. The allocator must be able to hold all
simultaneous weight bindings; there is no tensor tiling for an allocation that
cannot fit in available memory.

Mapped files consume virtual address space, which can constrain large models in
32-bit processes. Disk I/O failures after a mapping has been established follow
the operating system's mapped-file error behavior.

Separate Extractors and commands can run concurrently. A single Extractor or
command is used by one thread at a time. Changing model weights while recorded
work still refers to them is unsupported.

Applications and custom allocators must be rebuilt with the matching ncnn
headers and library. The added weight metadata changes `VkBufferMemory` layout;
appending fields does not preserve compatibility with older binary allocators.
