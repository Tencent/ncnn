// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layer.h"
#include "layer_type.h"
#include "cpu.h"
#include "net.h"
#include "datareader.h"

#include <stdio.h>
#include <string.h>
#include <vector>

static int permute_max_elempack()
{
#if NCNN_AVX512
    if (ncnn::cpu_support_x86_avx512())
        return 16;
#endif
#if NCNN_AVX
    if (ncnn::cpu_support_x86_avx())
        return 8;
#endif
    return 4;
}

// decode the permutation independently of the implementation's order table
static void permute_order(int dims, int order_type, int* order)
{
    int axes[4] = {0, 1, 2, 3};
    for (int i = dims - 1; i >= 0; i--)
    {
        int f = 1;
        for (int j = 2; j <= i; j++)
            f *= j;
        const int k = i - order_type / f;
        order_type %= f;
        order[i] = axes[k];
        for (int j = k; j < i; j++)
            axes[j] = axes[j + 1];
    }
}

static unsigned char* permute_element(const ncnn::Mat& m, const int* pos)
{
    const int p = m.elempack;
    const size_t size = m.elemsize / p;
    if (m.dims == 2)
        return (unsigned char*)m.row<unsigned char>(pos[1] / p) + pos[0] * m.elemsize + pos[1] % p * size;
    if (m.dims == 3)
        return (unsigned char*)m.channel(pos[2] / p).row<unsigned char>(pos[1]) + pos[0] * m.elemsize + pos[2] % p * size;
    return (unsigned char*)m.channel(pos[3] / p).depth(pos[2]).row<unsigned char>(pos[1]) + pos[0] * m.elemsize + pos[3] % p * size;
}

static int test_permute_packing(ncnn::Layer* op, int dims, int w, int h, int d, int c, int elempack, int bits, bool packing, int threads, bool unaligned, int cstep_padding = 0)
{
    const size_t elemsize = (size_t)(bits / 8) * elempack;
    ncnn::Mat a;
    if (dims == 2) a.create(w, h, elemsize, elempack);
    if (dims == 3) a.create(w, h, c, elemsize, elempack);
    if (dims == 4) a.create(w, h, d, c, elemsize, elempack);

    const size_t cstep = a.cstep + cstep_padding;
    if (dims >= 3) a.cstep = cstep;

    // the external buffer ends at the last valid lane, without allocator overread padding
    // also exercise a row base not aligned to the SIMD register width
    const size_t count = dims == 2 ? (size_t)w * h : a.cstep * (c - 1) + (size_t)w * h * (dims == 4 ? d : 1);
    const int offset = unaligned ? bits / 8 : 0;
    std::vector<unsigned char> storage(count * elemsize + offset);
    if (dims == 2) a = ncnn::Mat(w, h, storage.data() + offset, elemsize, elempack);
    if (dims == 3) a = ncnn::Mat(w, h, c, storage.data() + offset, elemsize, elempack);
    if (dims == 4) a = ncnn::Mat(w, h, d, c, storage.data() + offset, elemsize, elempack);
    if (dims >= 3) a.cstep = cstep;
    unsigned int seed = 7767517;
    for (size_t i = 0; i < count * elemsize; i++)
    {
        seed = seed * 1664525u + 1013904223u;
        ((unsigned char*)a)[i] = seed >> 24;
    }

    int shape[4] = {w, h, dims == 3 ? c : d, c};
    shape[dims - 1] *= elempack;
    ncnn::Option opt;
    opt.num_threads = threads;
    opt.use_packing_layout = packing;
    const int orders = dims == 2 ? 2 : dims == 3 ? 6 : 24;
    for (int o = 0; o < orders; o++)
    {
        ncnn::ParamDict pd;
        pd.set(0, o);
        if (op->load_param(pd) != 0)
            return -1;
        ncnn::Mat out;
        int ret = op->forward(a, out, opt);
        if (ret != 0 || out.empty())
        {
            fprintf(stderr, "permute forward failed dims=%d order=%d pack=%d bits=%d ret=%d\n", dims, o, elempack, bits, ret);
            return -1;
        }
        int order[4];
        permute_order(dims, o, order);
        int actual[4] = {out.w, out.h, dims == 3 ? out.c : out.d, out.c};
        actual[dims - 1] *= out.elempack;
        for (int i = 0; i < dims; i++)
        {
            if (actual[i] != shape[order[i]] || out.elembits() != bits)
            {
                fprintf(stderr, "permute shape failed dims=%d order=%d pack=%d bits=%d\n", dims, o, elempack, bits);
                return -1;
            }
        }
        int expected_pack = elempack;
        if (o != 0)
        {
            if (!packing)
                expected_pack = 1;
            else if (order[dims - 1] != dims - 1)
            {
                const int axis = shape[order[dims - 1]];
                expected_pack = 1;
                for (int p = 4; p <= permute_max_elempack(); p *= 2)
                {
                    if (axis % p == 0)
                        expected_pack = p;
                }
            }
        }
        if (out.elempack != expected_pack || (o == 0 && out.data != a.data))
        {
            fprintf(stderr, "permute packing failed dims=%d order=%d pack=%d expected=%d actual=%d\n", dims, o, elempack, expected_pack, out.elempack);
            return -1;
        }
        size_t total = 1;
        for (int i = 0; i < dims; i++) total *= actual[i];
        for (size_t i = 0; i < total; i++)
        {
            size_t v = i;
            int pos[4] = {0, 0, 0, 0};
            int srcpos[4] = {0, 0, 0, 0};
            for (int j = 0; j < dims; j++)
            {
                pos[j] = v % actual[j];
                v /= actual[j];
                srcpos[order[j]] = pos[j];
            }
            if (memcmp(permute_element(a, srcpos), permute_element(out, pos), bits / 8) != 0)
            {
                fprintf(stderr, "permute data failed dims=%d shape=(%d %d %d %d) order=%d pack=%d outpack=%d bits=%d packing=%d at %zu\n", dims, w, h, d, c, o, elempack, out.elempack, bits, packing, i);
                return -1;
            }
        }
    }
    return 0;
}

class PermuteFailAllocator : public ncnn::Allocator
{
public:
    virtual void* fastMalloc(size_t)
    {
        return 0;
    }
    virtual void fastFree(void*)
    {
    }
};

static int test_permute_allocation(ncnn::Layer* op)
{
    PermuteFailAllocator allocator;
    ncnn::Option opt;
    opt.blob_allocator = &allocator;
    const int packs[] = {1, 4, 8, 16};
    for (int bits = 16; bits <= 32; bits *= 2)
    {
        for (int i = 0; i < 4 && packs[i] <= permute_max_elempack(); i++)
        {
            ncnn::Mat a(8, 8, 2, (size_t)(bits / 8 * packs[i]), packs[i]);
            for (int order = 0; order < 6; order++)
            {
                ncnn::ParamDict pd;
                pd.set(0, order);
                op->load_param(pd);
                ncnn::Mat out;
                int ret = op->forward(a, out, opt);
                if ((order == 0 && (ret != 0 || out.data != a.data)) || (order != 0 && (ret != -100 || !out.empty())))
                {
                    fprintf(stderr, "permute allocation failure was not propagated\n");
                    return -1;
                }
            }
        }
    }
    return 0;
}

static int test_permute_long_records(ncnn::Layer* op)
{
    const int shapes[][5] = {{1048577, 1, 1, 2, 2}, {262145, 2, 2, 2, 8}};
    for (int bits = 16; bits <= 32; bits *= 2)
    {
        for (int i = 0; i < 2; i++)
        {
            const int w = shapes[i][0];
            const int h = shapes[i][1];
            const int d = shapes[i][2];
            const int c = shapes[i][3];
            const int order = shapes[i][4];
            const size_t elemsize = bits / 8;
            ncnn::Mat a(w, h, d, c, elemsize);
            if (a.empty())
                return -1;
            for (size_t j = 0; j < a.total() * elemsize; j++)
                ((unsigned char*)a)[j] = (unsigned char)(j * 131 + j / 13);

            ncnn::Option opt;
            opt.num_threads = 8;
            opt.use_packing_layout = false;
            ncnn::ParamDict pd;
            pd.set(0, order);
            ncnn::Mat out;
            if (op->load_param(pd) != 0 || op->forward(a, out, opt) != 0)
                return -1;
            if (out.dims != 4 || out.w != w || out.h != (order == 2 ? d : c) || out.d != h || out.c != (order == 2 ? c : d) || out.elempack != 1 || out.elembits() != bits)
                return -1;

            for (int q = 0; q < c; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const unsigned char* ptr = a.channel(q).depth(z).row<unsigned char>(y);
                        const unsigned char* outptr = order == 2 ? out.channel(q).depth(y).row<unsigned char>(z) : out.channel(z).depth(y).row<unsigned char>(q);
                        if (memcmp(ptr, outptr, (size_t)w * elemsize) != 0)
                        {
                            fprintf(stderr, "permute long record failed order=%d bits=%d\n", order, bits);
                            return -1;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

#if NCNN_STRING && NCNN_BATCH
static int test_permute_batch(ncnn::Layer* op)
{
    ncnn::Net net;
    net.opt.num_threads = 2;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_bf16_storage = false;
    const char* param = "7767517\n2 2\nInput input 0 1 in\nPermute permute 1 1 in out 0=3\n";
    const unsigned char weights[4] = {0};
    const unsigned char* weights_ptr = weights;
    ncnn::DataReaderFromMemory mb(weights_ptr);
    if (net.load_param_mem(param) != 0 || net.load_model(mb) != 0)
        return -1;
    ncnn::Mat a(7, 8, 2, 16u, 4, 2);
    for (size_t i = 0; i < a.total() * a.elemsize; i++)
        ((unsigned char*)a)[i] = (unsigned char)(i * 131 + i / 13);
    ncnn::Extractor ex = net.create_extractor();
    ncnn::Mat out;
    if (ex.input("in", a) != 0 || ex.extract("out", out, 1) != 0 || out.n != 2)
        return -1;
    ncnn::ParamDict pd;
    pd.set(0, 3);
    op->load_param(pd);
    for (int b = 0; b < 2; b++)
    {
        ncnn::Mat expected;
        if (op->forward(a.batch(b), expected, net.opt) != 0)
            return -1;
        if (out.w != expected.w || out.h != expected.h || out.c != expected.c || out.elempack != expected.elempack)
            return -1;
        const ncnn::Mat actual = out.batch(b);
        for (int q = 0; q < out.c; q++)
        {
            if (memcmp(actual.channel(q), expected.channel(q), (size_t)out.w * out.h * out.elemsize) != 0)
                return -1;
        }
    }
    return 0;
}
#endif // NCNN_STRING && NCNN_BATCH

int main()
{
    ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Permute);
    if (!op)
        return -1;
    if (!op->support_any_packing)
    {
        delete op;
        return 0;
    }
    const int packs[] = {1, 4, 8, 16};
    int ret = 0;
    for (int bits = 16; bits <= 32 && !ret; bits *= 2)
    {
        for (int dims = 2; dims <= 4 && !ret; dims++)
        {
            for (int p = 0; p < 4 && packs[p] <= permute_max_elempack() && !ret; p++)
            {
                ret = test_permute_packing(op, dims, 7, 5, 3, 2, packs[p], bits, true, 1, true)
                      || test_permute_packing(op, dims, 8, 16, 4, 2, packs[p], bits, true, 2, false)
                      || test_permute_packing(op, dims, 12, 12, 4, 2, packs[p], bits, true, 2, true)
                      || test_permute_packing(op, dims, 24, 8, 8, 2, packs[p], bits, true, 2, true)
                      || test_permute_packing(op, dims, 3, 4, 2, 2, packs[p], bits, false, 1, true)
                      || test_permute_packing(op, dims, 1, 1, 1, 1, packs[p], bits, true, 1, true);
            }
        }
    }
    for (int bits = 16; bits <= 32 && !ret; bits *= 2)
    {
        for (int p = 0; p < 4 && packs[p] <= permute_max_elempack() && !ret; p++)
        {
            // exercise pack16 on each output packing axis and unpacked spatial permutations
            ret = test_permute_packing(op, 4, 16, 16, 16, 2, packs[p], bits, true, 2, true)
                  || test_permute_packing(op, 4, 9, 7, 5, 2, packs[p], bits, false, 2, true)
                  || test_permute_packing(op, 3, 17, 19, 1, 2, packs[p], bits, true, 1, true)
                  || test_permute_packing(op, 3, 17, 19, 1, 2, packs[p], bits, false, 1, true)
                  || test_permute_packing(op, 3, 65, 33, 1, 1, packs[p], bits, true, 4, true)
                  || test_permute_packing(op, 3, 65, 33, 1, 1, packs[p], bits, false, 4, true)
                  || test_permute_packing(op, 4, 9, 5, 3, 1, packs[p], bits, true, 4, true)
                  || test_permute_packing(op, 4, 9, 5, 3, 1, packs[p], bits, false, 4, true);
        }
    }
    // single channel groups cover large spatial matrices and multiple slices
    for (int bits = 16; bits <= 32 && !ret; bits *= 2)
    {
        for (int p = 0; p < 4 && packs[p] <= permute_max_elempack() && !ret; p++)
        {
            ret = test_permute_packing(op, 3, 4096, 16, 1, 1, packs[p], bits, true, 4, true)
                  || test_permute_packing(op, 4, 65, 1, 513, 1, packs[p], bits, true, 4, true)
                  || test_permute_packing(op, 4, 65, 1, 513, 1, packs[p], bits, false, 4, true);
        }
    }
    for (int threads = 1; threads <= 4 && !ret; threads *= 2)
        ret = test_permute_packing(op, 3, 256, 128, 1, 1, 4, 32, true, threads, true);
    // large pack1 matrices include square, rectangular and vector-tail cases
    for (int bits = 16; bits <= 32 && !ret; bits *= 2)
    {
        ret = test_permute_packing(op, 3, 256, 256, 1, 1, 1, bits, true, 1, true)
              || test_permute_packing(op, 3, 512, 256, 1, 1, 1, bits, true, 1, true)
              || test_permute_packing(op, 3, 512, 512, 1, 1, 1, bits, true, 1, true)
              || test_permute_packing(op, 3, 257, 256, 1, 1, 1, bits, true, 1, true);
    }
    // degenerate axes and coalesced matrices, with and without channel padding
    // a physical channel group of one is not a scalar channel when packed
    for (int bits = 16; bits <= 32 && !ret; bits *= 2)
    {
        for (int threads = 1; threads <= 4 && !ret; threads *= 2)
        {
            for (int p = 0; p < 4 && packs[p] <= permute_max_elempack() && !ret; p++)
            {
                for (int packing = 0; packing < 2 && !ret; packing++)
                {
                    ret = test_permute_packing(op, 3, 1, 257, 1, 33, packs[p], bits, packing, threads, true)
                          || test_permute_packing(op, 3, 65, 1, 1, 33, packs[p], bits, packing, threads, true)
                          || test_permute_packing(op, 3, 32, 33, 1, 16, packs[p], bits, packing, threads, true)
                          || test_permute_packing(op, 3, 33, 33, 1, 17, packs[p], bits, packing, threads, true)
                          || test_permute_packing(op, 4, 1, 33, 17, 3, packs[p], bits, packing, threads, true)
                          || test_permute_packing(op, 4, 33, 1, 17, 3, packs[p], bits, packing, threads, true)
                          || test_permute_packing(op, 4, 33, 17, 1, 3, packs[p], bits, packing, threads, true)
                          || test_permute_packing(op, 4, 33, 17, 3, 1, packs[p], bits, packing, threads, true);
                }
            }
        }
    }
    // the contiguous-block length is independent of elempack
    // check overlapping head/tail vector copies against exact external buffers at every boundary
    const int widths[] = {2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 19, 31, 32, 33, 63, 64, 65};
    for (int bits = 16; bits <= 32 && !ret; bits *= 2)
    {
        for (int threads = 1; threads <= 4 && !ret; threads *= 4)
        {
            for (int i = 0; i < (int)(sizeof(widths) / sizeof(widths[0])) && !ret; i++)
            {
                ret = test_permute_packing(op, 3, widths[i], 33, 1, 65, 1, bits, false, threads, true)
                      || test_permute_packing(op, 4, widths[i], 33, 3, 7, 1, bits, false, threads, true);
            }
        }
    }
    // long contiguous records cover copies with a vector tail
    for (int bits = 16; bits <= 32 && !ret; bits *= 2)
        ret = test_permute_packing(op, 4, 16385, 1, 1, 2, 1, bits, false, 4, true);
    // degenerate packed planes and a few long records with multiple slices
    for (int bits = 16; bits <= 32 && !ret; bits *= 2)
    {
        for (int p = 0; p < 4 && packs[p] <= permute_max_elempack() && !ret; p++)
        {
            ret = test_permute_packing(op, 3, 4097, 1, 1, 2, packs[p], bits, true, 1, true)
                  || test_permute_packing(op, 3, 1, 4097, 1, 2, packs[p], bits, true, 1, true);
        }
        ret = ret || test_permute_packing(op, 4, 4097, 2, 2, 2, 1, bits, false, 8, true);
    }
    // explicit channel strides must survive every slice and output-group traversal
    for (int bits = 16; bits <= 32 && !ret; bits *= 2)
    {
        for (int p = 0; p < 4 && packs[p] <= permute_max_elempack() && !ret; p++)
        {
            ret = test_permute_packing(op, 4, 8, 16, 4, 17, packs[p], bits, true, 1, true, 5)
                  || test_permute_packing(op, 4, 8, 16, 4, 17, packs[p], bits, true, 8, true, 5)
                  || test_permute_packing(op, 4, 17, 9, 5, 3, packs[p], bits, false, 8, true, 5);
        }
    }
    if (!ret) ret = test_permute_long_records(op);
    // a one-dimensional permutation always aliases the input, including packing disabled
    for (int bits = 16; bits <= 32 && !ret; bits *= 2)
    {
        for (int p = 0; p < 4 && packs[p] <= permute_max_elempack() && !ret; p++)
        {
            ncnn::Mat a(7, (size_t)(bits / 8 * packs[p]), packs[p]);
            for (int packing = 0; packing < 2; packing++)
            {
                for (int order = 0; order < 24; order++)
                {
                    ncnn::ParamDict pd;
                    pd.set(0, order);
                    op->load_param(pd);
                    ncnn::Option opt;
                    opt.use_packing_layout = packing != 0;
                    ncnn::Mat out;
                    if (op->forward(a, out, opt) != 0 || out.data != a.data || out.dims != 1 || out.w != a.w || out.elempack != a.elempack || out.elemsize != a.elemsize)
                        ret = -1;
                }
            }
        }
    }
    // exercise every rectangular tile transition and scalar edge with pack1 output
    const int edges[] = {1, 2, 3, 4, 6, 7, 8, 10, 11, 14, 15, 16, 17, 18, 19, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33};
    const int edge_count = sizeof(edges) / sizeof(edges[0]);
    for (int bits = 16; bits <= 32 && !ret; bits *= 2)
    {
        for (int i = 0; i < edge_count && !ret; i++)
        {
            for (int j = 0; j < edge_count && !ret; j++)
                ret = test_permute_packing(op, 2, edges[i], edges[j], 1, 1, 1, bits, false, 1, true);
        }
    }
    if (!ret) ret = test_permute_allocation(op);
    for (int p = 0; p < 4 && packs[p] <= permute_max_elempack() && !ret; p++)
        ret = test_permute_packing(op, 4, 17, 9, 5, 3, packs[p], 32, true, 2, true);
#if NCNN_STRING && NCNN_BATCH
    if (!ret) ret = test_permute_batch(op);
#endif
    delete op;
    return ret;
}
