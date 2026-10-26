# Vulkan weight storage

Vulkan weights can stay in DEVICE memory, HOST memory, or a temporary prepared
file. The default DEVICE policy is unchanged. File mode allows a model's weights
to exceed device memory by keeping a fixed resident subset and exchanging the
remaining weights through bounded buffers.

## Configuration

Configure the Net before loading its model:

```cpp
net.opt.use_vulkan_compute = true;
net.opt.use_weights_in_host_disk = true;
net.opt.weight_device_budget = (size_t)1536 * 1024 * 1024;
net.opt.weight_host_budget = (size_t)3072 * 1024 * 1024;
```

These capacities are examples. The settings have the following meanings:

| Setting | Meaning |
| --- | --- |
| `use_weights_in_host_disk = false` | Keep native DEVICE/HOST sources; HOST execution can still use DEVICE exchange |
| Both total budgets zero | FILE exchange without adding resident buffers |
| Discrete GPU, explicit totals | Both totals must be positive; resident storage plus one internal command's exchange allocation must fit each total |
| Unified memory, explicit totals | One positive total suffices; when both are positive, use their minimum |
| `weight_device_cache_size = 0` | Choose DEVICE exchange capacity automatically |
| `weight_host_cache_size = 0` | Choose HOST exchange capacity automatically |
| Positive cache size | Explicit exchange capacity, checked against remaining backing capacity |

An all-DEVICE model needs no exchange buffers. Integrated GPUs and CPU Vulkan
devices use mapped DEVICE exchange without a separate HOST copy. Explicit file
mode determines residency independently of `use_weights_in_host_memory`.

Extractor setters configure the capacities and allocators of its internal
command before `Mat` extraction. An explicitly supplied `VkCompute` uses its own
settings, including when several Extractors or Nets record into it. Neither path
changes the Net's resident layout. A complete exchanged weight and each command's
simultaneous weight set must fit DEVICE exchange. HOST exchange can be smaller
than a weight because FILE copies can be split into chunks.

Net's loading budgets count native Vulkan backing requirements, including
alignment, unused resident block capacity, and loading staging. Finalization
reserves space for one planned exchange working set. Internal commands and
`VkCompute(weights, opt)` use the remaining budgets. `VkCompute(vkdev, opt)` owns
an independent exchange budget; zero selects automatic, uncapped capacity. This
allows one external command to serve models with different resident layouts
without silently adopting an Extractor's allocator or shrinking to another Net's
budget.

These are not aggregate quotas. Retained command working sets, including idle
pooled commands, add to each Net's resident allocation. Concurrent commands do
not each receive a reservation from a shared device-wide budget. Activations,
KV caches, CPU model parsing/packing, OS file cache, driver allocations and process
RSS are outside these weight budgets. Applications requiring a device-wide cap
must account for all retained commands and resident stores.

Custom exchange allocators must outlive the commands that use them and return buffers with
compatible memory types, usage flags, ranges, and alignment. Unified exchange
must be mapped. Discrete HOST exchange must use a host heap and permit access by
the transfer family. An exclusive HOST allocation must not be in use by another
family; a concurrent allocation must include the transfer family. Observable
backing requirements are checked; hidden pool reservations and allocator thread
safety remain the caller's responsibility.

## Code organization

All implementation stays in existing source files:

| File | Responsibility |
| --- | --- |
| `allocator.h` | Buffer source metadata, storage configuration and finalization |
| `allocator.cpp` | Temporary mappings, loading directory, resident planning, native allocation and flushes |
| `command.h` | Command recording, cache reuse and standalone command configuration |
| `command.cpp` | Loading uploads, command snapshots, exchange caches and GPU synchronization |
| `net.cpp` | Loading sequence and Net/Extractor ownership |

`WeightFile`, `VkWeightStore`, and the placement records are local to
`allocator.cpp`. The directory only tracks prepared FILE allocations; ordinary
DEVICE allocations do not add entries or scan it when freed. Resident promotion
updates the existing logical `VkBufferMemory` object, preserving VkMat copies,
views, and reference counts. Directory indices are internal planning details.
There is no public weight ID or prepared-store handle. Each weight allocator
keeps its generation in private data. A list of live weight allocators allows
commands to check that generation without dereferencing a destroyed allocator
or changing the public buffer layout. Allocation and release update the private
generation directly, without searching that list.

Recording, dependency analysis and uploads use `VkBufferMemory*` directly. The
command's cache keeps a plain source record for each distinct recorded weight,
with its allocator and generation. Registration evicts a cached range if its
address now identifies a different allocation, preserving any required queue
ownership handoff. Submission checks each recorded allocator once before
reading source metadata. These source records are discarded with the recording;
the retained HOST/DEVICE cache entries keep their own identity fields.

`VkWeightAllocator::finalize_weights()` plans and allocates residents, then sends
source/destination buffer descriptions to one internal loading-upload function
in `command.cpp`. DEVICE uploads finish and release their temporary staging
before HOST residents are allocated. Loading code owns the file and placement;
command code owns copies and queue operations.

`VkComputePrivate` owns dependency analysis, uploads, command buffers, queues,
fences and timing queries. Its compute command pool serves both ordinary and
managed submissions. Its exclusively owned `VkWeightCache` holds exchange
buffers, cached source ranges, resolved settings and the synchronization needed
to retain contents across submissions. Extractors do not own or share this
cache. Internal loading helpers supply layer boundaries to the resident planner.
The allocator-based command constructor queries planned capacities and remaining
budgets without modifying the caller's `Option`. Actual recorded dependencies
determine the minimum exchange capacity needed by a submission.

The weight-specific allocator interface consists of three operations:

```cpp
int set_weights_in_host_disk(bool enable);
int set_weight_budgets(size_t device_budget, size_t host_budget);
int finalize_weights(size_t device_cache_size = 0, size_t host_cache_size = 0);
```

Total budgets are configured before allocation. The optional finalization
arguments reserve exchange space when choosing the resident layout; zero selects
automatic reservations. Net supplies these from its cache settings. Grouping,
cache-limit queries and cold-read controls are internal helpers, with access
confined to the existing allocator implementation class.

Deferred pipelines use the same command snapshot and replay path with push
descriptors and allocated descriptor sets. Ordinary DEVICE commands retain the
direct push-descriptor path. Dependency validation builds a flat list once per
submission; batching accumulates new sources without repeatedly rebuilding the
same dependency union.

## Sources and lifetimes

`VkBufferMemory::mapped_ptr` describes a Vulkan host mapping. `external_ptr` is a
borrowed pointer to source bytes beginning at the logical allocation;
`external_size` bounds those bytes. A VkMat view adds only its view offset.
FILE allocations point to stable file mappings. Native HOST buffers can be copied
directly by Vulkan; HOST sources without a native buffer use their external
bytes. Ordinary DEVICE and activation allocations leave external fields empty.

Net owns prepared mappings and immutable HOST/DEVICE resident storage. Every
`VkCompute` owns one mutable exchange working set. Extractors keep inference
state and weight references; they can record a GPU tensor chain from several
Nets into one command without intermediate downloads or cache rebinding.
Different commands execute independently and share only immutable model weights.
A single Extractor or command must not be used concurrently by multiple threads.
All models and commands in a GPU chain must use the same Vulkan device.

Command constructors save configuration without creating exchange resources.
The first HOST/FILE binding enables deferred weight recording. Retaining an old
working set does not force a later pure-DEVICE recording through replay. The
device-only constructor uses automatic capacities; clone operations can supply
initial exchange settings through their `Option`. For a working set bounded by
one model's planned headroom, construct from its finalized allocator:

```cpp
// After VkTransfer has submitted the prepared weights:
if (weights.finalize_weights(opt.weight_device_cache_size, opt.weight_host_cache_size))
    return -1;

ncnn::VkCompute cmd(weights, opt);
// Record layer forwards, pipelines, clones, and downloads as usual.
```

For multi-Net execution, use `VkCompute(vkdev, opt)` to configure the exchange
working set independently of any one model. External Extractors still need their
usual blob/workspace/staging allocators. Their weight-working-set setters apply
only to their internal commands. No public cache-sharing or cache-binding API is
needed.

`reset()` discards the recording and retains a healthy working set and command
resources. `clear_weight_cache()` releases exchange buffers and associated
execution resources while keeping command configuration; it rejects unfinished
recordings. It does not release Net-owned resident weights. A recoverable upload
or allocation failure invalidates the affected working set, and `reset()` allows
it to be rebuilt on the next submission. Device loss remains a Vulkan error.

Model resources and source bytes must remain valid through recording and
submission. After work completes and the command is reset, its cached copies do
not extend the Net's lifetime. Reloading or freeing model allocations changes
their generation, preventing stale hits even if allocator and allocation
addresses are reused. Changing a model while its commands are pending is invalid.
Borrowed HOST sources outside `VkWeightAllocator` have no tracked generation;
call `clear_weight_cache()` before changing their bytes or reusing their identity.
Custom exchange allocators must remain valid until the command releases its
working set or is destroyed.

For ordinary `Mat` extraction, an Extractor exclusively leases an internal
`VkCompute` for managed weights and uses `reset()` between submissions. Releasing
the Extractor returns a default-allocator command to its Net's pool, which keeps
at most two idle contexts. The pool reuses both arenas and command/synchronization
resources, and never lends an active context twice. Commands using caller-supplied
weight allocators are destroyed on release, so the pool does not extend those
allocators' lifetime requirement. External-command extraction does not create an
internal command. Extractor copies and assignments carry inference state and
configuration, not a command lease.

Recording snapshots source pointers, view ranges, shader metadata, handles,
constants, and dispatch sizes. It does not read file contents or allocate
exchange arenas. Pipeline wrappers need not survive replay, but their Vulkan
resources and source allocations must follow the existing command lifetime
contract. Image command references retain image/view lifetimes.

## File preparation and residency

Upload retains the existing cast, packing, and reorder operations, then copies
the prepared bytes into stable mapped views. Each allocation has its own view;
later file growth does not invalidate earlier pointers. Finalization flushes
those views and makes them read-only before model loading succeeds.

Set `Net::set_weight_file_directory(path)` before `load_model`, or
`VkWeightAllocator::set_weight_file_directory(path)` before enabling disk storage,
to use an application-owned writable directory (for example, Android's cache
directory). Both APIs copy the UTF-8 path; `Net::clear()` retains it. Null or empty
selects the system default. POSIX uses `TMPDIR` when set, otherwise `tmpfile()`;
Windows uses `GetTempPathW`. Files are immediately unlinked on POSIX and marked
for deletion on close on Windows. The allocator closes the file and unmaps its
views when cleared or destroyed. Choose a disk filesystem if the temporary
directory is a tmpfs.

File creation failure may fall back to the existing HOST/DEVICE policy only when
neither a directory nor total budgets were explicitly configured for the Net.
The allocator's `set_weights_in_host_disk` reports creation failure to its caller.
Preparation reserves file space before exposing each writable mapping, so a
space reservation failure returns an error instead of faulting on a mapped write.
Reservation, mapping, flush, and protection errors fail loading. Other faults
during mapped memory access, such as backing-store I/O failures, follow OS
signal/exception behavior.
All live mappings need virtual address space, which can limit large models in
32-bit or otherwise constrained processes. Failed finalization requires clearing
the allocator before reuse.

The resident planner visits live prepared allocations in size order, breaking
ties by loading order. Fitting smaller weights go to DEVICE, then HOST, with the
rest remaining FILE. Unified memory omits separate HOST placement. Resident
blocks target 8 MiB or a dedicated larger weight and shrink to available capacity.
Every proposed allocation is checked against Vulkan memory requirements.

If the full layout fits DEVICE, no exchange space is needed. Otherwise automatic
DEVICE exchange targets the larger of twice the largest weight and the largest
upload group, bounded by the total budget. Net supplies layer groups internally;
manual loads without groups use a conservative combined group for budgeted
placement. Actual commands still validate simultaneous dependencies. Automatic HOST
exchange targets twice the largest remaining weight, capped at one quarter of
the HOST budget and requiring at least one aligned transfer block. Explicit
capacities in the command's `Option` are not enlarged. With automatic command
capacities, an unbudgeted manual load starts from the largest weight and actual
dependencies rather than allocating space for the whole model. Later submissions
can grow the cache within the remaining budget. Growth releases the old arenas
before allocating their replacements and invalidates their cached entries.

DEVICE filling uses mapped memory where possible, otherwise bounded HOST staging
of at most 32 MiB, further limited by budget and the largest fill. HOST residents
are filled through their native mappings. ReBAR device heaps do not count as
independent host memory. VkImage weight paging is unsupported; explicit residency
budgets reject weight images whose allocation cannot be accounted for here.

## Inference and synchronization

Cache entries are keyed by source allocation and allocator generation. DEVICE hits avoid weight reads
and uploads. FILE misses use a FIFO HOST cache or a bounded chunk window. Native
HOST residents upload directly from their buffers. HOST and DEVICE entries are
evicted independently. Unified FILE exchange copies directly into mapped DEVICE
memory, with no additional HOST payload or weight H2D copy.

The scheduler preserves record order and normally uses half the DEVICE arena
per batch. Actual upload/compute overlap depends on queue topology and region
reuse dependencies; ownership returns currently wait behind earlier compute
submissions. A single command can use the
full arena when required. Each command reuses 32 upload slots and 4 compute slots,
with command buffers, fences, and binary semaphores. It uses a dedicated transfer
family or a second same-family queue when available, and supports a single queue.
Acquiring a second same-family queue is nonblocking.

Cross-family reuse has matching transfer-to-compute and compute-to-transfer
release/acquire ranges. Ownership waits use `ALL_COMMANDS`. Same-family paths
use barriers and semaphore ordering without ownership transfers. Host staging
reuse waits for transfer completion; DEVICE reuse waits for the required compute-to-transfer dependency.
The final chunk signals readiness for the complete weight. These synchronization
rules also apply when cache entries are evicted or an operation fails.

## Tests and measurement

`test_weight_execution` links the selected ncnn library, including shared builds.
Its public-interface cases cover Net/Extractor inference, standalone FILE/HOST
sources, mixed pipeline bindings, cross-Net A/B/A chains, mixed storage modes,
command reset and explicit clearing, model reloads, internal context pooling,
automatic cache growth after a smaller submission, and empty optional uploads.
Static builds additionally exercise queue modes, command isolation, cache
lifetime after Extractor destruction, allocator release, cache counters and
recovery from injected allocation failures. `test_weight_storage` and `perf_weight_storage` use static
linkage for internal diagnostics. Shared diagnostic helpers live in the existing
`testutil.h`/`testutil.cpp` and are compiled only for static Vulkan test builds.
No second copy of the core library is built for tests. Benchmark counters remain
inside `command.cpp`; static diagnostic tools observe completed submissions
through an internal callback. There is no
public statistics type or query API, and the callback is not exported by shared
libraries.

For a static build with `NCNN_BUILD_TESTS=ON`:

```sh
cmake --build build --target test_weight_storage test_weight_execution test_command
NCNN_TEST_MODEL_DIR="$PWD/examples" ctest --test-dir build \
    -R '^(test_weight_storage|test_weight_execution|test_command)$' \
    --output-on-failure
```

The performance tool requires `NCNN_VULKAN`, `NCNN_BENCHMARK`, Gemm, and a static
build. Arguments are GPU index, matrix width, input rows, layers, iterations,
queue mode (0 automatic, 1 same-family, 2 single queue), cold flag, and residency
mode (0 FILE, 1 mixed, 2 all resident, 3 ordinary DEVICE):

```sh
./build/tests/perf/perf_weight_storage 0 1024 128 16 5 0 0 1
```

Every iteration checks the output. `load_ms` includes source generation, pipeline
creation, preparation, and resident filling. `finalize_ms` covers final flush,
placement, resident allocation, and loading uploads. `prepared_bytes` is logical
prepared payload; it is not file size including alignment. Layout payload,
unique resident backing requirements, exchange backing, and native loading peaks
are reported separately. Loading peaks are measured by allocation hooks in the
tool rather than production directory queries.

Linux cold mode discards mapped pages and requests file cache eviction before
each timed iteration. A separate sequential scan measures cold read throughput;
`/proc/self/io` reports process physical read bytes. Cache eviction is a request,
and tmpfs cannot measure disk throughput. `upload_timing_available=0` means GPU
upload timing is unavailable on the selected queue, not that no bytes moved.
Counter and timing collection is conditional on `NCNN_BENCHMARK`.
