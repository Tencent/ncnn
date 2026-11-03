# vulkan weight storage

Vulkan weights normally stay in device memory. `use_weights_in_host_memory`
prefers host memory. Host and file weights are resolved into temporary Vulkan
buffers before execution. `use_weights_in_disk` keeps weights in a temporary
file, reducing resident model storage at the cost of file reads and uploads.
Temporary weight buffers use a private, command-owned `VkBlobAllocator` pool,
separate from activation and layer workspace allocations.

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
inference repeatedly. Each `Mat` extraction uses a local command. Its private
weight allocator is created during command initialization; pool memory is
allocated lazily and destroyed with the command. External
commands retain their own pool across `reset()` calls.

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
are command-owned allocations: the private allocator can grow its pool, and an
allocation failure stops the current recording operation. Recording retains the complete
resolved weight group for the preceding operation while allocating the current
group. Allow room for both adjacent complete groups, allocation alignment and
pool reserves, not just the largest individual binding or dispatch.

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

Normal `Mat` extraction selects online weight upload from the Net's
`use_weights_in_host_memory` and `use_weights_in_disk` options. For external
commands, select the mode before recording:

```cpp
ncnn::VkCompute cmd(net.vulkan_device());
cmd.set_use_weight_upload_online(net.opt.use_weights_in_host_memory || net.opt.use_weights_in_disk);
```

The setting defaults to `false` and survives `submit_and_wait()` and `reset()`.
It stays fixed while commands are recorded. Enable it for an external command
that uses deferred host or disk weights, including when several Nets share the
command. With this mode disabled, recording a deferred weight logs the missing
configuration and returns from that operation.

`init()` creates the private weight allocator, prepares the compute and upload
command pools and starts the initial compute command buffer. Both command pools
remain available until command destruction.
Temporary weight and staging allocations are still created only when needed.

Each command owns an independent temporary weight pool. It cannot be replaced
with, or shared with, an Extractor's `workspace_vkallocator`. Continue to
configure each Extractor's blob, workspace and staging allocators for ordinary
layer operations; these settings do not select the temporary weight pool.

`cmd.reset()` waits for submitted work and discards unsubmitted recording state.
Temporary weight views return to the private pool through `fastFree()`; pooled backing allocations remain available
for that command until destruction. File-upload staging also belongs to the
command and is not borrowed from ordinary layer staging. There is no
`VkCompute::set_weight_memory_limit()` or `clear_weight_cache()`.
Model backing must remain valid and immutable until execution completes and the
recording is reset. Clearing a Net invalidates its weights and existing
Extractors. Use `clear()`, `load_param()` and `load_model()` to replace a model.

The private `record_resolve()` creates a temporary `VkMat` from the private
pool, records the complete source upload and
returns a view with the original shape, packing and offset. Ordinary buffers
pass through. `record_pipeline()` first scans for weight bindings that need
resolving. If present, it submits the preceding compute batch, resolves each
weight independently in a second pass, then begins the next compute batch after
waiting for command buffer reuse. Weight-consuming clones use the same batch
transition. Final submission closes the last batch
without opening another. Later operations upload the source again; there is no
persistent weight identity cache. `MemoryData` continues to use `record_clone()`
and owns its independent output.

There is one private pool, not two alternating pools. After a weight-consuming
dispatch or clone is recorded, the command retains its complete resolved group
as owning `VkMat` references. The next such operation keeps that preceding
group alive while allocating and recording its own group, so adjacent groups
cannot overlap in storage. At the end of that recording call, the older group
is released and the command retains the new group. Each retained view keeps its
whole allocation alive, including when a shader binds only a subview. An
operation two generations later may reuse a released range. Submission completion
or reset discards retained references only after pending work is drained.

Temporary weight uploads reuse the existing device transfer-family selector,
which prefers a transfer-only family and otherwise selects a transfer-capable
family. The selected family may be the same as the compute family.
Temporary weight backing buffers use the ordinary EXCLUSIVE blob allocator.
When the two families differ, the final upload chunk releases the complete
weight range to the compute family, and its consumer acquires the same range
after waiting for the upload semaphore. A range's older compute consumer has
completed before the range is reused and every byte overwritten; the upload family can acquire
that range with its old contents discarded. The private file-upload staging
buffer is EXCLUSIVE and used only by the upload family. Ordinary activation and
workspace buffers keep their existing sharing and ownership rules.

Queue-family sharing does not replace synchronization. Before recording weight
group N+1, the preceding compute batch N is submitted. Batch N-1 has already
completed: its completion was required before reusing the compute command buffer
for N. Its temporary ranges can therefore be reused without tracking a batch
number or waiting again. Group N is still referenced while N+1 is allocated, so
these two groups have distinct storage. The compute consumer waits for its own uploads
and the preceding compute batch. These semaphore dependencies keep computation
ordered while permitting upload N+1 to overlap compute N. They remain
conservative batch dependencies, not a per-range reader table or a global
planner. Actual overlap and performance depend on the selected queues, device
and workload; a separate transfer family alone is not a performance guarantee.

On unified-memory devices, a mapped target is filled directly after the older
compute batch N-1 completes, using the same two-generation lifetime rule. An
unmapped target uses staging. Other devices copy host-resident Vulkan buffers
directly or fill a command-owned staging buffer from file mappings. File uploads
use chunks of at most 16 MiB, waiting for the previous upload before overwriting
staging. These transfer chunks do not slice the weight's mathematical operation
or its shader binding. Uploads cover the complete allocation capacity, including
alignment padding, even when the consumer binds a subview. File mappings reserve
that same capacity and initialize the allocation's trailing padding to zero.
Host and file sources use the existing mapped pointer and allocation offset;
the tensor view keeps its separate offset.

Host copies read source and destination mappings directly, including allocation
and chunk offsets. They execute while recording, after any preceding use of the
destination or staging range completes. Non-coherent destinations are flushed
before GPU submission, including alignment padding. Ordinary staging uploads
also call `allocator->flush()` before their compute batch is submitted.

Online upload streams batches from `record_*()`. A new weight group submits the
preceding compute batch, prepares and submits its own uploads, then waits before
resetting the compute command buffer for reuse. Upload chunks similarly wait
only when their upload buffer or staging storage must be reused. Uploads for the
next group can overlap the preceding compute. Push descriptor devices encode
compute commands directly; other devices replay the current batch's logical
commands at submission. A pipeline bind through its dispatch stays in one batch.
CPU metadata for completed recordings is discarded while download postprocessing
remains deferred until `submit_and_wait()`.

Each command owns one compute command buffer and one upload command buffer,
allocated at initialization and retained across reset. Two fences and four binary
semaphores are also reused. The last upload chunk signals readiness once for all
weights in its group. `submit_and_wait()` submits the remaining compute batch,
consumes outstanding chain signals and waits for completion before downloads.
Reset and destruction wait for already submitted work before releasing its
resources; reset discards any unsubmitted recording. An interrupted stream may
require replacing semaphores whose signals have not been consumed.

Recording failures return from the current operation, following the ordinary
`void record_*()` convention. This includes failures from submissions made while
recording. They are not saved as a command-wide error for a later
`submit_and_wait()` call. Reset an incomplete recording before using the command
again. Output allocation failures retain the empty-output convention.
`submit_and_wait()` returns errors from its own recording, submission and waiting
operations. A nonterminal fence error retains pending work and its resources;
reset or destruction must complete the wait before releasing them.

For direct allocator use, configure `VkWeightAllocator` before allocation with
`set_weights_in_disk()`, `set_weight_memory_limit(device_limit, host_limit)` and optionally `set_weight_file_directory()`.
Check the return value of `set_weight_memory_limit()` before allocating weights;
an invalid configuration returns an error and leaves the previous settings intact.
Allocation failures return a null buffer, observed as an empty `VkMat`.
Weight tensors passed to `VkTransfer::record_upload()` have `n == 1`.
Record each transformed weight with `VkTransfer::record_upload()`, then check
`submit_and_wait()` before calling `finalize_weights()`. Complete finalization
before recording weight use with `VkCompute`; command recording relies on this
loading order. Recording failures end the current operation without retaining
an error status. `submit_and_wait()` and `reset()` return errors from their own
operations; a successful final submit does not certify earlier void recording
operations. Net does not aggregate recording failures into a load error.
After a submission or wait failure, discard the command; destruction
drains any work still in flight. Finalization flushes
and seals file mappings; resident weights have already been uploaded directly
to their final locations. Keep the allocator alive until its views and recorded
work have been released. Unmapped destinations are filled through a reusable
staging buffer of at most 8 MiB, reduced to fit the configured host memory limit.
Recording copies source bytes directly into staging, batching small weights.
When staging fills, recording submits and waits before reusing it; large weights
are transferred in chunks during the same call. No whole-weight CPU snapshots or
pending source queue are retained. The final `submit_and_wait()` completes the
remaining commands. Net also submits when pending uploads exceed 256 MiB,
checked after each layer. File-backed and host-mappable weights are copied
directly into their mappings while recording. Host-mappable Vulkan allocations
are flushed at that point. Staging is flushed before submission and remains
alive until all submitted work completes, including after a failed fence wait.

`get_weight_memory_usage()` reports actual device and host resident Vulkan
allocation bytes, including padding. It excludes file mappings, temporary
weights and private pool reserves. Check allocation results and the return
value of `finalize_weights()` directly; memory usage does not certify upload
completion.

## best practice

- Set resident limits while leaving room for two adjacent complete temporary
  weight groups, allocator pool reserves, activations, KV caches and the rest of
  the application.
- Use a writable application cache directory on a disk filesystem. A temporary
  directory backed by tmpfs still consumes memory.
- Reuse the Net to retain model storage. For external command recording, reuse
  the command and its private temporary weight pool with `reset()` between
  submissions.
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
allocations are outside these limits. Concurrent commands always use separate
temporary weight pools, even if their Extractors share ordinary allocators.

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
