// ncnn ps3 SPE gemm kernel
// C[i, j] = alpha * (dot(A[i, k], B[j, k]) + beta * C0), for i in [row_begin, row_end).
//
// A and B are [rows x K] with elempack 4, exactly as an ncnn Mat stores them:
// element (r, k) is lane r%4 of the vec4 at ((r/4)*K + k).
// the result is written back in the same packing, which is what
// gemm_transB_pack4_powerpc stores: element (i, j) at ((i/4)*N + j)*4 + i%4.
//
// the descriptor is a flat 128 bytes. a struct of these fields comes out at 120
// on both the ppu and the spu, and a 120 byte dma is rounded to 64, which halts
// the spu when the local address is only 16 byte aligned.
//
// the thread group is started once and stays up. each mailbox word is one gemm;
// 0 asks the thread to exit. a large tile lives in the local store and the next
// k panel is fetched while the current one is consumed, so the mfc is not waiting
// on a 16 byte transfer for every vec4.
#include <spu_intrinsics.h>
#include <spu_mfcio.h>
#include <sys/spu_thread.h>

#define TAG_GET0 1
#define TAG_GET1 2
#define TAG_PUT  3

// one panel of K. 32 vec4 is 512 bytes per packed row.
#define KB 32
// packed rows of A held at once (4 real rows each) and packed columns of B.
// the accumulator is MR*4*NR vec4 and has to stay in registers: the spu has 128
// of them and spills a larger tile to local store, which serialises every fma
// behind a load and a store. 4*4*4 = 64 leaves the rest for the panel vectors.
#define MR 4
#define NR 4

// flat 128 bytes. a struct of long long + float is 120 on the spu (the compiler
// inserts 4 bytes before the last word and rounds the size up), and mfc_get of
// 120 becomes a 64-byte dma that halts on a 16-byte-aligned local address.
typedef struct
{
    unsigned char raw[128];
} desc_t;

typedef char desc_t_is_128[(sizeof(desc_t) == 128) ? 1 : -1];

static unsigned int d_u32(const desc_t* d, int off)
{
    unsigned int v;
    __builtin_memcpy(&v, d->raw + off, 4);
    return v;
}

static float d_f32(const desc_t* d, int off)
{
    float v;
    __builtin_memcpy(&v, d->raw + off, 4);
    return v;
}

static unsigned long long d_u64(const desc_t* d, int off)
{
    unsigned long long v;
    __builtin_memcpy(&v, d->raw + off, 8);
    return v;
}

#define D_EA_A      0
#define D_EA_B      8
#define D_EA_C      16
#define D_EA_DONE   24
#define D_M         32
#define D_N         36
#define D_K         40
#define D_ROW_BEGIN 44
#define D_ROW_END   48
#define D_ALPHA     52
#define D_BETA      56
#define D_C0        60
// A row stride in bytes, the distance from the start of one K-segment of a
// packed A row to the next. each segment is D_A_STRIDE/16 vec4 and the
// segments tile the K axis; 0 means the row is contiguous.
#define D_A_STRIDE 64

// one allocation. the linker is free to drop another symbol into the gap between
// two aligned arrays, and a dma into the second then overwrites it.
static union
{
    vec_float4 v[1];
    struct
    {
        vec_float4 A[2][MR][KB];
        vec_float4 B[2][NR][KB];
        vec_float4 O[MR][NR];
        // the accumulator is MR*4*NR vec4. that is 16 KB, far more than the
        // default spu stack, so it lives here with the dma buffers.
        vec_float4 ACC[MR * 4][NR];
    } s;
} buf __attribute__((aligned(128)));
#define Abuf buf.s.A
#define Bbuf buf.s.B
#define Out  buf.s.O
#define Acc  buf.s.ACC

#define EA32(x) ((unsigned int)(x))

static void dma_wait(unsigned int tag)
{
    mfc_write_tag_mask(1u << tag);
    spu_mfcstat(MFC_TAG_UPDATE_ALL);
}

// mfc requires the transfer size to be 1, 2, 4, 8 or a multiple of 16, and both
// addresses aligned to that size. ncnn only guarantees 16 byte alignment and a
// row is rarely a multiple of 128, so each chunk is cut to the largest legal
// size the current effective address allows: 128 when it is 128 byte aligned,
// otherwise 16. the local side is always 128 byte aligned, so it never limits.
// the mfc queue holds 16 commands. a transfer that is only 16 byte aligned
// is split into many short pieces, and a panel is one transfer per row, so a
// whole panel queued at once overflows the queue and the spu faults. count the
// commands in flight and drain every few.
static int g_get_queued = 0;

static void dma_get(void* ls, unsigned int ea, unsigned int bytes, unsigned int tag)
{
    unsigned char* p = (unsigned char*)ls;
    while (bytes)
    {
        unsigned int n = bytes;
        if (n > 16384u)
            n = 16384u;
        unsigned int align = ea & -ea;
        if (align < n)
            n = align;
        n &= ~15u;
        if (n == 0)
            n = 16;
        spu_mfcdma32(p, ea, n, tag, MFC_GET_CMD);
        g_get_queued++;
        if (g_get_queued >= 4)
        {
            dma_wait(tag);
            g_get_queued = 0;
        }
        p += n;
        ea += n;
        bytes -= n;
    }
}

static void dma_put(void* ls, unsigned int ea, unsigned int bytes, unsigned int tag)
{
    unsigned char* p = (unsigned char*)ls;
    while (bytes)
    {
        unsigned int n = bytes;
        if (n > 16384u)
            n = 16384u;
        unsigned int align = ea & -ea;
        if (align < n)
            n = align;
        n &= ~15u;
        if (n == 0)
            n = 16;
        spu_mfcdma32(p, ea, n, tag, MFC_PUT_CMD);
        p += n;
        ea += n;
        bytes -= n;
    }
}

// fetch kb vec4 of `rows` packed rows, the panel starting at global k `k0`.
// each packed row is K vec4 apart, so the rows are separate transfers; within a
// row the chunking above does the rest. astride 0 means the row is one
// contiguous run: the panel sits at ea_base + k0*16 + r*K*16 and is a single
// transfer per row, which is exactly the old behaviour. astride non-zero means
// the row is split into segments of astride/16 vec4, segment s at
// r*K*16 + s*astride from ea_base, so the panel is walked segment by segment
// into the contiguous local slot: the vec4 at panel position kk has global k
// k0+kk, lives in segment (k0+kk)/(astride/16) at offset
// (k0+kk)%(astride/16)*16, and a run inside one segment is one transfer.
// the caller guarantees astride is a multiple of 16 and K a multiple of
// astride/16, so the addresses below come out exact.
static void dma_panel(vec_float4* dst, unsigned int ea_base, unsigned int K,
                      int rows, int kb, int k0, unsigned int astride,
                      unsigned int tag)
{
    const unsigned int bytes = (unsigned int)kb * 16u;
    const unsigned int koff = (unsigned int)k0 * 16u;
    const unsigned int seg = astride / 16u;
    g_get_queued = 0;
    for (int r = 0; r < rows; r++)
    {
        const unsigned int ea_row = ea_base + (unsigned int)r * K * 16u;
        if (astride == 0)
        {
            dma_get(dst + r * KB, ea_row + koff, bytes, tag);
            continue;
        }
        for (int kk = 0; kk < kb;)
        {
            const unsigned int g = (unsigned int)(k0 + kk);
            const unsigned int segment = g / seg;
            const unsigned int within = g - segment * seg;
            unsigned int len = seg - within;
            if (len > (unsigned int)(kb - kk))
                len = (unsigned int)(kb - kk);
            dma_get(dst + r * KB + kk,
                    ea_row + segment * astride + within * 16u,
                    len * 16u, tag);
            kk += (int)len;
        }
    }
}

// the full tile is done in assembly. spu-gcc will not keep MR*4*NR
// accumulators in registers no matter the optimisation level: it reloads and
// stores every one around each fma, and that load/store pair is the whole cost.
// gemm_full runs the kb loop with the accumulators pinned to registers and
// returns them in Acc.
void gemm_full(int slot, int kb, vec_float4* a_base, vec_float4* b_base,
               vec_float4* acc_base);

// the trailing panel is shorter than KB and the edge tiles are smaller than
// MR x NR, so those keep the runtime bounds. they are a small fraction of K.
static void consume_partial(int slot, int rows, int cols, int kb)
{
    for (int kk = 0; kk < kb; kk++)
    {
        vec_float4 b0[NR], b1[NR], b2[NR], b3[NR];
        for (int c = 0; c < cols; c++)
        {
            vec_float4 bk = Bbuf[slot][c][kk];
            b0[c] = spu_splats(spu_extract(bk, 0));
            b1[c] = spu_splats(spu_extract(bk, 1));
            b2[c] = spu_splats(spu_extract(bk, 2));
            b3[c] = spu_splats(spu_extract(bk, 3));
        }

        for (int r = 0; r < rows; r++)
        {
            vec_float4 ak = Abuf[slot][r][kk];
            for (int c = 0; c < cols; c++)
            {
                Acc[r * 4 + 0][c] = spu_madd(ak, b0[c], Acc[r * 4 + 0][c]);
                Acc[r * 4 + 1][c] = spu_madd(ak, b1[c], Acc[r * 4 + 1][c]);
                Acc[r * 4 + 2][c] = spu_madd(ak, b2[c], Acc[r * 4 + 2][c]);
                Acc[r * 4 + 3][c] = spu_madd(ak, b3[c], Acc[r * 4 + 3][c]);
            }
        }
    }
}

// one tile: `rows` packed rows of A by `cols` packed columns of B, full K.
static void gemm_tile(const desc_t* d, int i0, int j0, int rows, int cols)
{
    const unsigned int K = d_u32(d, D_K);

    for (int r = 0; r < rows * 4; r++)
        for (int c = 0; c < cols; c++)
            Acc[r][c] = spu_splats(0.f);

    const unsigned int ea_a0 = (unsigned int)d_u64(d, D_EA_A)
                               + (unsigned int)(i0 / 4) * K * 16u;
    const unsigned int ea_b0 = (unsigned int)d_u64(d, D_EA_B)
                               + (unsigned int)(j0 / 4) * K * 16u;
    const unsigned int astride = d_u32(d, D_A_STRIDE);

    int kb0 = 0;
    int kb = (int)K - kb0;
    if (kb > KB)
        kb = KB;

    // dma_panel adds k0*16 to ea_base itself, so A and B both pass their raw
    // base and the panel start. B is always contiguous, astride 0.
    dma_panel(&Abuf[0][0][0], ea_a0, K, rows, kb, kb0, astride, TAG_GET0);
    dma_panel(&Bbuf[0][0][0], ea_b0, K, cols, kb, kb0, 0, TAG_GET0);
    dma_wait(TAG_GET0);

    int slot = 0;
    for (;;)
    {
        const int next = kb0 + kb;
        int kb2 = 0;
        if (next < (int)K)
        {
            kb2 = (int)K - next;
            if (kb2 > KB)
                kb2 = KB;
            dma_panel(&Abuf[slot ^ 1][0][0], ea_a0, K, rows, kb2, next, astride, TAG_GET0 + (slot ^ 1));
            dma_panel(&Bbuf[slot ^ 1][0][0], ea_b0, K, cols, kb2, next, 0, TAG_GET0 + (slot ^ 1));
        }

        if (rows == MR && cols == NR)
            gemm_full(slot, kb, &Abuf[slot][0][0], &Bbuf[slot][0][0], &Acc[0][0]);
        else
            consume_partial(slot, rows, cols, kb);

        if (kb2 == 0)
            break;

        dma_wait(TAG_GET0 + (slot ^ 1));
        kb0 = next;
        kb = kb2;
        slot ^= 1;
    }

    const vec_float4 alpha = spu_splats(d_f32(d, D_ALPHA));
    const vec_float4 beta = spu_splats(d_f32(d, D_BETA));
    const vec_float4 c0 = spu_splats(d_f32(d, D_C0));
    const vec_float4 bias = spu_mul(beta, c0);

    // top is [N x M/4] elempack 4: element (i, j) at ((i/4)*N + j)*4 + i%4,
    // so the four real rows of one column are one contiguous vec4 and the four
    // real columns of a packed column are four contiguous vec4.
    const unsigned int N = d_u32(d, D_N);
    const unsigned int ea_c = (unsigned int)d_u64(d, D_EA_C);

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            for (int t = 0; t < 4; t++)
                Out[r][c * 4 + t] = spu_madd(spu_add(Acc[r * 4 + t][c], bias), alpha, spu_splats(0.f));

            const unsigned int base = (((unsigned int)(i0 / 4) + (unsigned int)r) * N
                                       + (unsigned int)j0 + (unsigned int)c * 4u)
                                      * 16u;
            dma_put(&Out[r][c * 4], ea_c + base, 64, TAG_PUT);
        }
    }
}

int main(unsigned long long ea_desc, unsigned long long ea_ready, unsigned long long a2, unsigned long long a3)
{
    (void)a2;
    (void)a3;

    // the inbound mailbox holds a single word and silently drops whatever is
    // written before this thread reaches its blocking read. tell the ppe we
    // are there before reading, or the first command never arrives.
    static unsigned int ready[4] __attribute__((aligned(16)));
    ready[0] = (unsigned int)ea_desc;
    ready[1] = (unsigned int)(ea_desc >> 32);
    ready[2] = (unsigned int)ea_ready;
    ready[3] = 1;
    spu_mfcdma32(ready, (unsigned int)ea_ready, 16, TAG_PUT, MFC_PUT_CMD);
    dma_wait(TAG_PUT);

    static unsigned int done_local[4] __attribute__((aligned(16)));

    for (;;)
    {
        unsigned int cmd = spu_readch(SPU_RdInMbox);
        if (cmd == 0)
            break;

        // the compiler aligns this to 128 and the dma lands on &desc itself.
        // masking the address down to a 128 boundary points the get at the
        // slot below it, so the fields come back shifted and the tile walks
        // off the end of C.
        desc_t desc __attribute__((aligned(128)));
        spu_mfcdma32(&desc, EA32(ea_desc), sizeof(desc), TAG_GET0, MFC_GET_CMD);
        dma_wait(TAG_GET0);

        const int row_begin = (int)d_u32(&desc, D_ROW_BEGIN);
        const int row_end = (int)d_u32(&desc, D_ROW_END);
        const int n = (int)d_u32(&desc, D_N);

        for (int i0 = row_begin; i0 < row_end;)
        {
            int rows = (row_end - i0) / 4;
            if (rows > MR)
                rows = MR;

            for (int j0 = 0; j0 < n;)
            {
                int cols = (n - j0) / 4;
                if (cols > NR)
                    cols = NR;

                gemm_tile(&desc, i0, j0, rows, cols);
                j0 += cols * 4;
            }
            i0 += rows * 4;
        }

        dma_wait(TAG_PUT);

        done_local[0] = 1;
        done_local[1] = 0;
        done_local[2] = 0;
        done_local[3] = 0;
        spu_mfcdma32(done_local, EA32(d_u64(&desc, D_EA_DONE)), 16, TAG_PUT, MFC_PUT_CMD);
        dma_wait(TAG_PUT);
    }

    spu_thread_exit(0);
    return 0;
}
