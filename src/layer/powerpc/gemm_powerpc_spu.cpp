// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gemm_powerpc_spu.h"

#if __PPU__

#include <string.h>

#include <sys/spu.h>
#include <sys/thread.h>

// the spe image, produced from gemm_powerpc_spu.c by spu-gcc and bin2s
extern "C" {
    extern const unsigned char gemm_powerpc_spu_bin[];
    extern const unsigned int gemm_powerpc_spu_bin_size;
}

namespace ncnn {

// the descriptor the spe reads. kept as a flat 128 bytes: a struct of these
// fields is 120 bytes on both the ppu and the spu, and a 120 byte dma is
// rounded down to 64, which halts the spu.
struct SpuGemmDesc
{
    unsigned char raw[128];
};

static void desc_u64(SpuGemmDesc* d, int off, unsigned long long v)
{
    memcpy(d->raw + off, &v, 8);
}

static void desc_u32(SpuGemmDesc* d, int off, unsigned int v)
{
    memcpy(d->raw + off, &v, 4);
}

static void desc_f32(SpuGemmDesc* d, int off, float v)
{
    memcpy(d->raw + off, &v, 4);
}

#define DESC_EA_A      0
#define DESC_EA_B      8
#define DESC_EA_C      16
#define DESC_EA_DONE   24
#define DESC_N         36
#define DESC_K         40
#define DESC_ROW_BEGIN 44
#define DESC_ROW_END   48
#define DESC_ALPHA     52
#define DESC_BETA      56
#define DESC_C0        60
// A row stride in bytes: distance from the start of one K-segment of a packed
// A row to the next. each segment is DESC_A_STRIDE/16 vec4 and the segments
// tile the K axis; 0 means the row is contiguous.
#define DESC_A_STRIDE 64

// ps3 has 6 usable spus. the group is created on the first call and kept until
// the process exits: building and tearing it down per gemm costs more than the
// gemm itself once the tile is large.
#define SPU_GEMM_MAX 6

static sysSpuImage g_image;
static int g_image_ready = 0;
static int g_group_ready = 0;
static sys_spu_group_t g_group;
static sys_spu_thread_t g_thread[SPU_GEMM_MAX];

// the spe dma's a word here once, right before its first blocking mailbox read.
// the mailbox drops anything written earlier, so the first command waits for it.
static unsigned int g_ready[SPU_GEMM_MAX][4] __attribute__((aligned(16)));

// one completion flag per worker, each on its own cache line. rpcs3 drops
// stores when two spus dma a flag into the same line.
static volatile unsigned int g_done[SPU_GEMM_MAX][32] __attribute__((aligned(128)));
static SpuGemmDesc g_desc[SPU_GEMM_MAX] __attribute__((aligned(128)));

static int spu_gemm_wait_ready(int nthread)
{
    for (unsigned int spins = 0; spins < 2000000u; spins++)
    {
        int ready = 1;
        for (int w = 0; w < nthread; w++)
            if (g_ready[w][3] == 0)
                ready = 0;
        if (ready)
            return 0;
        if ((spins & 1023u) == 0)
            sysThreadYield();
    }
    return -1;
}

static int spu_gemm_wait_done(int nthread)
{
    // the spe signals completion with a dma into g_done. yielding on every
    // pass costs far more than the gemm once the tile is in registers, so this
    // spins and only gives the emulator a chance to run the spus occasionally.
    for (unsigned int spins = 0; spins < 2000000000u; spins++)
    {
        int all = 1;
        for (int w = 0; w < nthread; w++)
            if (g_done[w][0] == 0)
                all = 0;
        if (all)
            return 0;
        if ((spins & 0x3ffffu) == 0)
            sysThreadYield();
    }
    return -1;
}

// bring the group up once. later calls only check that it is still there.
static int spu_gemm_start(int nthread)
{
    if (g_group_ready)
        return 0;

    if (!g_image_ready)
    {
        int ret = sysSpuInitialize(SPU_GEMM_MAX, 0);
        if (ret != 0)
            return -1;
        ret = sysSpuImageImport(&g_image, gemm_powerpc_spu_bin, 0);
        if (ret != 0)
            return -1;
        g_image_ready = 1;
    }

    sysSpuThreadGroupAttribute grpattr;
    sysSpuThreadGroupAttributeInitialize(grpattr);
    sysSpuThreadGroupAttributeName(grpattr, "ncnnspu");

    int ret = sysSpuThreadGroupCreate(&g_group, nthread, 100, &grpattr);
    if (ret != 0)
        return -1;

    sysSpuThreadAttribute attr;
    sysSpuThreadAttributeInitialize(attr);
    sysSpuThreadAttributeName(attr, "ncnnspu");

    sysSpuThreadArgument arg;
    for (int w = 0; w < nthread; w++)
    {
        sysSpuThreadArgumentInitialize(arg);
        g_ready[w][3] = 0;
        arg.arg0 = (u64)(uintptr_t)&g_desc[w];
        arg.arg1 = (u64)(uintptr_t)g_ready[w];
        ret = sysSpuThreadInitialize(&g_thread[w], g_group, w, &g_image, &attr, &arg);
        if (ret != 0)
            return -1;
    }

    ret = sysSpuThreadGroupStart(g_group);
    if (ret != 0)
        return -1;

    if (spu_gemm_wait_ready(nthread) != 0)
        return -1;

    g_group_ready = nthread;
    return 0;
}

int gemm_transB_pack4_spu(const Mat& A, const Mat& BT, Mat& top_blob, float alpha, float beta, float c0, unsigned int a_stride_bytes)
{
    const int M = A.h * 4;
    const int N = BT.h * 4;
    const int K = A.w;

    // the kernel steps both axes by 4 and has no remainder path
    if (M < 4 || N < 4 || M % 4 != 0 || N % 4 != 0 || K <= 0)
        return -1;

    int nthread = SPU_GEMM_MAX;
    // no point in more workers than there are row groups
    if (nthread > M / 4)
        nthread = M / 4;

    // the group size is fixed at startup. a later call that wants fewer workers
    // just leaves the spare threads blocked on their mailbox.
    if (g_group_ready && nthread > g_group_ready)
        nthread = g_group_ready;

    if (spu_gemm_start(nthread) != 0)
        return -1;

    int begin = 0;
    for (int w = 0; w < nthread; w++)
    {
        int rows = (M - begin) / (nthread - w);
        rows -= rows % 4;
        if (w == nthread - 1)
            rows = M - begin;

        g_done[w][0] = 0;

        SpuGemmDesc* d = &g_desc[w];
        memset(d, 0, sizeof(*d));
        desc_u64(d, DESC_EA_A, (unsigned long long)(uintptr_t)(const float*)A);
        desc_u64(d, DESC_EA_B, (unsigned long long)(uintptr_t)(const float*)BT);
        desc_u64(d, DESC_EA_C, (unsigned long long)(uintptr_t)(float*)top_blob);
        desc_u64(d, DESC_EA_DONE, (unsigned long long)(uintptr_t)g_done[w]);
        desc_u32(d, DESC_N, N);
        desc_u32(d, DESC_K, K);
        desc_u32(d, DESC_ROW_BEGIN, begin);
        desc_u32(d, DESC_ROW_END, begin + rows);
        desc_f32(d, DESC_ALPHA, alpha);
        desc_f32(d, DESC_BETA, beta);
        desc_f32(d, DESC_C0, c0);
        desc_u32(d, DESC_A_STRIDE, a_stride_bytes);

        begin += rows;
    }

    for (int w = 0; w < nthread; w++)
        sysSpuThreadWriteMb(g_thread[w], 1);

    return spu_gemm_wait_done(nthread);
}

} // namespace ncnn

#endif // __PPU__
