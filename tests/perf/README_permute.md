# x86 Permute validation and benchmarks

Recorded on 2026-09-20. Kernel baseline: `7d8c096f4`. The baseline benchmark
executable was built at `bfbd26147`, which adds the same measurement interface
without changing that kernel. Results below describe the implementation in this
commit, including the final OpenMP build guard.

Environment: AMD Ryzen 9 9950X exposed through KVM, GCC 15.2, Release build,
ncnn runtime CPU dispatch selecting AVX512, OpenMP enabled. The test build enabled
Input, Cast, Packing and Permute. These measurements do not establish performance
on Intel CPUs or other cache topologies.

## Reproduction

Build the CPU tests and benchmark (in both the baseline and current checkout):

```sh
cmake -S . -B build-permute -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_BUILD_TESTS=ON -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_VULKAN=OFF -DNCNN_RUNTIME_CPU=ON \
  -DNCNN_OPENMP=ON -DNCNN_BF16=ON
cmake --build build-permute -j 4 --target test_permute test_permute_packing perf_permute
ctest --test-dir build-permute -R '^test_permute(_packing)?$' --output-on-failure
```

The explicit case interface preserves the requested input pack:

```text
perf_permute --case dims w h d channel_groups pack bits order threads packing [buffers]
```

`h` in 2D and `channel_groups` in 3D/4D are physical packed extents. For example,
`channel_groups=1, pack=16` represents 16 logical channels. `packing=0` requests
pack1 output for nonidentity permutations. `buffers=1` reuses one input/output
pair; larger values rotate preallocated pairs. Choose a total footprint larger
than the last-level cache to evaluate that working set.

```sh
# Large spatial unpack, one CPU and one input/output pair.
OMP_WAIT_POLICY=PASSIVE taskset -c 0 build-permute/tests/perf/perf_permute \
  --case 3 1024 1024 1 1 8 32 1 1 0

# The same case with four rotating pairs (approximately 256 MiB in total).
OMP_WAIT_POLICY=PASSIVE taskset -c 0 build-permute/tests/perf/perf_permute \
  --case 3 1024 1024 1 1 8 32 1 1 0 4

# Independent depth slices in a single packed channel group.
OMP_WAIT_POLICY=PASSIVE taskset -c 0-3 build-permute/tests/perf/perf_permute \
  --case 4 128 128 64 1 4 32 1 4 1

# Enumerated input packs, all 4D orders, and Net integration cases.
OMP_WAIT_POLICY=PASSIVE taskset -c 0-3 build-permute/tests/perf/perf_permute --packed
```

Each layer invocation warms its buffers and prints the median, minimum and maximum
of seven timing batches. The recorded comparison alternated baseline/current
processes three times per case, reversing their order in the second pair. The
reported result is the median of those three medians. Single-thread cases used CPU
0; four-thread cases used CPUs 0-3. No build, sanitizer or other benchmark was run
concurrently with the recorded comparison.

The complete arguments, timings and individual samples for all 59 layer cases
are in [permute_x86_results.csv](permute_x86_results.csv). Each row can be replayed
using the `--case` arguments above. A ratio above 1 means the current version is
faster. The extra follow-up row records a seven-pair repeat of an apparent small
regression, rather than replacing the original measurement.

## Layer results

Shapes below use the physical extents described above. Times are milliseconds.

| Shape | Input pack / bits | Order | Threads | Packing | Buffers | Baseline | Current | Ratio |
|---|---|---|---|---|---|---|---|---|
| 1024x1024x1 | 8 / 32 | 1 | 1 | 0 | 1 | 5.225281 | 4.322021 | 1.21x |
| 1024x1024x1 | 8 / 16 | 1 | 1 | 0 | 1 | 3.438721 | 2.750977 | 1.25x |
| 1024x1024x1 | 16 / 32 | 1 | 4 | 1 | 1 | 5.377014 | 3.381287 | 1.59x |
| 4096x16x64 | 4 / 32 | 2 | 4 | 1 | 1 | 5.286743 | 3.533752 | 1.50x |
| 128x128x64x1 | 4 / 32 | 1 | 4 | 1 | 1 | 1.093750 | 0.487549 | 2.24x |
| 16x65536 | 4 / 32 | 1 | 4 | 1 | 1 | 0.574280 | 0.330017 | 1.74x |
| 256x256x1 | 1 / 32 | 1 | 1 | 1 | 1 | 0.008579 | 0.003769 | 2.28x |
| 512x512x1 | 1 / 32 | 1 | 1 | 1 | 1 | 0.062378 | 0.024994 | 2.50x |
| 512x512x1 | 1 / 16 | 1 | 1 | 1 | 1 | 0.036598 | 0.011063 | 3.31x |
| 257x257x1 | 1 / 32 | 1 | 1 | 1 | 1 | 0.005251 | 0.005398 | 0.97x |
| 2048x2048x1 | 1 / 32 | 1 | 1 | 1 | 1 | 1.558289 | 1.549255 | 1.01x |
| 1024x1024x1 | 8 / 32 | 1 | 1 | 0 | 4 | 5.169250 | 4.423767 | 1.17x |

The first pass measured the 16-bit, pack1, 257x257 case at 0.002420 ms versus
0.002594 ms, about 7.2% slower. The samples overlapped. Seven additional alternating
pairs gave 0.002358 ms versus 0.002348 ms; the regression did not persist. This case
continues to use the direct transpose path. The full CSV includes slower and flat
cases as well as improvements; these results are not a claim that every shape
becomes faster.

The existing Net case, `(80,1600,32), order=3`, measured 1.8435 -> 1.8379 ms with
one thread and 0.9816 -> 0.9846 ms with four threads: effectively unchanged in this
run. These are the Net harness's 32-extraction averages, not the layer median
protocol. Net loading, extraction and batch correctness checks also passed.

## Kernel and scheduling choices

- Keep the fixed register tiles and adjacent contiguous/stride variants. A cross-
  channel block processes one output channel group. Its interface no longer has
  a dummy exchanged-axis extent or output-group stride. Pack1 variants expose the
  contiguous direction in their argument lists.
- Large spatial unpack uses the direct per-column path for pack8 in both storage
  widths, and pack16 in 16-bit storage, when rows >= 128, input stride >= pack*512
  and output row stride >= 512 scalar elements. Small planes keep the adjacent-
  column path. Other packs retain their measured baseline traversal.
- Register tiles and cache tiles are separate. Pack1 cache blocking is restricted
  to rows and columns in [256,512], both strides <= 1024 scalar elements and both
  strides divisible by 256. FP32 uses 32x32 blocks. AVX512 16-bit uses 64x64 below
  131072 elements and 32x32 otherwise; SSE2/AVX use 32x32. Other shapes keep direct
  traversal. Narrow pack/unpack kernels bypass this cache-policy wrapper.
- Cache experiments compared 32/64/128 square blocks, both traversal orders,
  32x128/128x32 rectangles, odd boundaries and long matrices. Spatial unpack
  experiments compared 1/2/4/8-column groups, 64/128/256-row bounds and paired
  16-row tiles. The implementation retains a small set of measured choices,
  without runtime autotuning.
- Parallel work uses disjoint output regions. Existing channel/depth/slice tasks
  are used first; spatial or input-channel ranges are split only when those tasks
  cannot occupy the requested threads. Scheduling is static. Inputs smaller than
  64 KiB use one thread; additional splits target at least 16 KiB per task.
  Spatial boundaries are multiples of 32 elements and input-channel block
  boundaries are multiples of 16 groups. A build without OpenMP keeps one block.

## Correctness and assembly checks

Passed `test_permute` and `test_permute_packing` in:

- SSE2-only Release, runtime CPU dispatch disabled;
- AVX-only Release, AVX2/AVX512 disabled;
- Release with SSE2/AVX/AVX512 runtime dispatch;
- Debug ASan+UBSan with runtime dispatch;
- SSE2 Release with OpenMP disabled.

Additional checks covered 150 deterministic random shapes, legal packs, both
storage widths and all 2D/3D/4D orders. Permanent tests assert the expected output
pack, preserve raw bit patterns, exercise single channel groups, task/cache
boundaries, unaligned exact-length input buffers, channel padding, 1/2/4-thread
execution, identity aliasing, allocation failures and Net batch handling.

Assembly inspection confirmed the reduced cross-channel signatures and removal
of the one-iteration output-group loops. Fixed SIMD tiles remain direct calls or
inlined blocks; there is no function-pointer dispatch in their inner loops.
