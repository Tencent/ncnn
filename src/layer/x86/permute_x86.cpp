// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "permute_x86.h"

#include <string.h>

#include "cpu.h"
#include "x86_usability.h"

namespace ncnn {

#if __SSE2__
// Partial loads/stores must not touch channel padding or the next pack.
static NCNN_FORCEINLINE uint64_t permute_load_tail(const unsigned char* ptr, int size)
{
    uint64_t v = 0;
    switch (size)
    {
    case 2: memcpy(&v, ptr, 2); break;
    case 4: memcpy(&v, ptr, 4); break;
    case 6: memcpy(&v, ptr, 6); break;
    }
    return v;
}

static NCNN_FORCEINLINE void permute_store_tail(unsigned char* ptr, uint64_t v, int size)
{
    switch (size)
    {
    case 2: memcpy(ptr, &v, 2); break;
    case 4: memcpy(ptr, &v, 4); break;
    case 6: memcpy(ptr, &v, 6); break;
    }
}

static NCNN_FORCEINLINE __m128i permute_load(const unsigned char* ptr, int size)
{
    if (size == 16)
        return _mm_loadu_si128((const __m128i*)ptr);

    if (size >= 8)
    {
        __m128i _lo = _mm_loadl_epi64((const __m128i*)ptr);
        uint64_t hi = permute_load_tail(ptr + 8, size - 8);
        return _mm_unpacklo_epi64(_lo, _mm_loadl_epi64((const __m128i*)&hi));
    }

    uint64_t lo = permute_load_tail(ptr, size);
    return _mm_loadl_epi64((const __m128i*)&lo);
}

static NCNN_FORCEINLINE void permute_store(unsigned char* ptr, __m128i _v, int size)
{
    if (size == 16)
    {
        _mm_storeu_si128((__m128i*)ptr, _v);
        return;
    }

    if (size >= 8)
    {
        _mm_storel_epi64((__m128i*)ptr, _v);
        _v = _mm_srli_si128(_v, 8);
        ptr += 8;
        size -= 8;
    }

    uint64_t lo;
    _mm_storel_epi64((__m128i*)&lo, _v);
    permute_store_tail(ptr, lo, size);
}
#endif // __SSE2__

#include "permute_fp32.h"
#include "permute_bf16s_fp16s.h"

typedef void (*permute_transpose_func)(const unsigned char*, size_t, unsigned char*, size_t, int, int);

typedef void (*permute_copy_func)(const unsigned char*, size_t, unsigned char*, size_t, int);

static void permute_copy_64bit(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int count)
{
    int i = 0;
#if __SSE2__
    if (outstride == 8)
    {
        for (; i + 1 < count; i += 2)
        {
            __m128i _r0 = _mm_loadl_epi64((const __m128i*)ptr);
            __m128i _r1 = _mm_loadl_epi64((const __m128i*)(ptr + stride));
            _mm_storeu_si128((__m128i*)outptr, _mm_unpacklo_epi64(_r0, _r1));
            ptr += stride * 2;
            outptr += 16;
        }
    }
    else if (stride == 8)
    {
        for (; i + 1 < count; i += 2)
        {
            __m128i _v = _mm_loadu_si128((const __m128i*)ptr);
            _mm_storel_epi64((__m128i*)outptr, _v);
            _mm_storel_epi64((__m128i*)(outptr + outstride), _mm_srli_si128(_v, 8));
            ptr += 16;
            outptr += outstride * 2;
        }
    }
#endif // __SSE2__
    for (; i < count; i++)
    {
        memcpy(outptr, ptr, 8);
        ptr += stride;
        outptr += outstride;
    }
}

static void permute_copy_128bit(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int count)
{
    for (int i = 0; i < count; i++)
    {
#if __SSE2__
        __m128i _v = _mm_loadu_si128((const __m128i*)ptr);
        _mm_storeu_si128((__m128i*)outptr, _v);
#else
        memcpy(outptr, ptr, 16);
#endif
        ptr += stride;
        outptr += outstride;
    }
}

#if __AVX__
static void permute_copy_256bit(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int count)
{
    for (int i = 0; i < count; i++)
    {
        __m256 _v = _mm256_loadu_ps((const float*)ptr);
        _mm256_storeu_ps((float*)outptr, _v);
        ptr += stride;
        outptr += outstride;
    }
}
#endif // __AVX__

#if __AVX512F__
static void permute_copy_512bit(const unsigned char* ptr, size_t stride, unsigned char* outptr, size_t outstride, int count)
{
    for (int i = 0; i < count; i++)
    {
        __m512i _v = _mm512_loadu_si512(ptr);
        _mm512_storeu_si512(outptr, _v);
        ptr += stride;
        outptr += outstride;
    }
}
#endif // __AVX512F__

// Strides are in scalar lanes. The last logical axis is split into pack
// groups and lanes; cstep is the distance between groups, not logical channels.
static void permute_strides(const Mat& m, size_t* strides)
{
    strides[0] = m.elempack;
    strides[1] = (size_t)m.w * m.elempack;
    if (m.dims == 3)
        strides[2] = m.cstep * m.elempack;
    if (m.dims == 4)
    {
        strides[2] = (size_t)m.w * m.h * m.elempack;
        strides[3] = m.cstep * m.elempack;
    }
}

static NCNN_FORCEINLINE size_t permute_offset(const int* pos, const size_t* strides, int dims, int pack_axis, int elempack)
{
    size_t offset = (size_t)(pos[pack_axis] / elempack) * strides[pack_axis] + pos[pack_axis] % elempack;
    for (int i = 0; i < dims; i++)
    {
        if (i != pack_axis)
            offset += pos[i] * strides[i];
    }
    return offset;
}

// These are the same matrices used by 2d type=1. Only the plane strides differ.
static void permute3d(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int out_elempack = top_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const int elembits = bottom_blob.elembits();

    if (order_type == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned char* ptr = bottom_blob.channel(q);
            unsigned char* outptr = top_blob.channel(q);
            if (elembits == 32)
                permute_transpose_spatial_fp32(ptr, w * elemsize, outptr, h * elemsize, h, w, elempack);
            else
                permute_transpose_spatial_bf16s_fp16s(ptr, w * elemsize, outptr, h * elemsize, h, w, elempack);
        }
        return;
    }

    if (elempack != 1 || out_elempack != 1)
    {
        if (elembits == 32)
            permute3d_cross_fp32(bottom_blob, top_blob, order_type, opt);
        else
            permute3d_cross_bf16s_fp16s(bottom_blob, top_blob, order_type, opt);
        return;
    }

    permute_transpose_func transpose = elembits == 32 ? permute_transpose_pack1_fp32 : permute_transpose_pack1_bf16s_fp16s;
    if (order_type == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < h; q++)
        {
            unsigned char* outptr = top_blob.channel(q);
            for (int c = 0; c < channels; c++)
            {
                const unsigned char* ptr = bottom_blob.channel(c).row<unsigned char>(q);
                memcpy(outptr, ptr, w * elemsize);
                outptr += w * elemsize;
            }
        }
        return;
    }
    if (order_type == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < h; q++)
        {
            const unsigned char* ptr = (const unsigned char*)bottom_blob + q * w * elemsize;
            unsigned char* outptr = top_blob.channel(q);
            transpose(ptr, bottom_blob.cstep * elemsize, outptr, channels * elemsize, channels, w);
        }
        return;
    }
    if (order_type == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned char* ptr = bottom_blob.channel(q);
            unsigned char* outptr = (unsigned char*)top_blob + q * h * elemsize;
            transpose(ptr, w * elemsize, outptr, top_blob.cstep * elemsize, h, w);
        }
        return;
    }
    if (order_type == 5)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < h; q++)
        {
            const unsigned char* ptr = (const unsigned char*)bottom_blob + q * w * elemsize;
            unsigned char* outptr = (unsigned char*)top_blob + q * channels * elemsize;
            transpose(ptr, bottom_blob.cstep * elemsize, outptr, top_blob.cstep * elemsize, channels, w);
        }
        return;
    }
}

// 4d orders 1..5 keep channel packing and permute only the spatial axes.
static void permute4d_spatial(const Mat& bottom_blob, Mat& top_blob, int order_type, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int elempack = bottom_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    typedef void (*transpose_spatial_func)(const unsigned char*, size_t, unsigned char*, size_t, int, int, int);
    transpose_spatial_func transpose = bottom_blob.elembits() == 32 ? permute_transpose_spatial_fp32 : permute_transpose_spatial_bf16s_fp16s;

    if (order_type == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < bottom_blob.c; q++)
        {
            const unsigned char* ptr = bottom_blob.channel(q);
            unsigned char* outptr = top_blob.channel(q);
            for (int z = 0; z < d; z++)
            {
                transpose(ptr, w * elemsize, outptr, h * elemsize, h, w, elempack);
                ptr += (size_t)w * h * elemsize;
                outptr += (size_t)w * h * elemsize;
            }
        }
        return;
    }
    if (order_type == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < bottom_blob.c; q++)
        {
            const unsigned char* ptr = bottom_blob.channel(q);
            unsigned char* outptr = top_blob.channel(q);
            for (int y = 0; y < h; y++)
            {
                for (int z = 0; z < d; z++)
                {
                    memcpy(outptr, ptr + ((size_t)z * h + y) * w * elemsize, w * elemsize);
                    outptr += w * elemsize;
                }
            }
        }
        return;
    }
    if (order_type == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < bottom_blob.c; q++)
        {
            const unsigned char* ptr = bottom_blob.channel(q);
            unsigned char* outptr = top_blob.channel(q);
            for (int y = 0; y < h; y++)
            {
                transpose(ptr, (size_t)w * h * elemsize, outptr, d * elemsize, d, w, elempack);
                ptr += w * elemsize;
                outptr += (size_t)w * d * elemsize;
            }
        }
        return;
    }
    if (order_type == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < bottom_blob.c; q++)
        {
            const unsigned char* ptr = bottom_blob.channel(q);
            unsigned char* outptr = top_blob.channel(q);
            for (int z = 0; z < d; z++)
            {
                transpose(ptr, w * elemsize, outptr, (size_t)h * d * elemsize, h, w, elempack);
                ptr += (size_t)w * h * elemsize;
                outptr += h * elemsize;
            }
        }
        return;
    }
    if (order_type == 5)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < bottom_blob.c; q++)
        {
            const unsigned char* ptr = bottom_blob.channel(q);
            unsigned char* outptr = top_blob.channel(q);
            for (int y = 0; y < h; y++)
            {
                transpose(ptr, (size_t)w * h * elemsize, outptr, (size_t)h * d * elemsize, d, w, elempack);
                ptr += w * elemsize;
                outptr += d * elemsize;
            }
        }
        return;
    }
}

Permute_x86::Permute_x86()
{
#if __SSE2__
    support_packing = true;
    support_any_packing = true;
#endif // __SSE2__
    support_fp16_storage = cpu_support_x86_f16c();
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Permute_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int dims = bottom_blob.dims;
#if __AVX512F__
    const int max_elempack = 16;
#elif __AVX__
    const int max_elempack = 8;
#elif __SSE2__
    const int max_elempack = 4;
#else
    const int max_elempack = 1;
#endif
    if (bottom_blob.elempack > max_elempack || (bottom_blob.elempack != 1 && bottom_blob.elempack != 4 && bottom_blob.elempack != 8 && bottom_blob.elempack != 16))
        return -100;

    if (dims == 1 || order_type == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    if (dims < 2 || dims > 4 || order_type < 0 || order_type >= (dims == 2 ? 2 : dims == 3 ? 6 : 24)) return -1;

    const int elembits = bottom_blob.elembits();
    if (elembits != 16 && elembits != 32)
        return -100;

    static const int orders[24][4] = {
        {0, 1, 2, 3}, // 0
        {1, 0, 2, 3}, // 1
        {0, 2, 1, 3}, // 2
        {2, 0, 1, 3}, // 3
        {1, 2, 0, 3}, // 4
        {2, 1, 0, 3}, // 5
        {0, 1, 3, 2}, // 6
        {1, 0, 3, 2}, // 7
        {0, 3, 1, 2}, // 8
        {3, 0, 1, 2}, // 9
        {1, 3, 0, 2}, // 10
        {3, 1, 0, 2}, // 11
        {0, 2, 3, 1}, // 12
        {2, 0, 3, 1}, // 13
        {0, 3, 2, 1}, // 14
        {3, 0, 2, 1}, // 15
        {2, 3, 0, 1}, // 16
        {3, 2, 0, 1}, // 17
        {1, 2, 3, 0}, // 18
        {2, 1, 3, 0}, // 19
        {1, 3, 2, 0}, // 20
        {3, 1, 2, 0}, // 21
        {2, 3, 1, 0}, // 22
        {3, 2, 1, 0}, // 23
    };
    const int* order = orders[order_type];
    const int elempack = bottom_blob.elempack;
    const size_t lane_size = bottom_blob.elemsize / elempack;
    int shape[4] = {bottom_blob.w, bottom_blob.h, dims == 3 ? bottom_blob.c : bottom_blob.d, bottom_blob.c};
    shape[dims - 1] *= elempack;
    int outshape[4] = {1, 1, 1, 1};
    for (int i = 0; i < dims; i++)
        outshape[i] = shape[order[i]];

    int out_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
        if (order[dims - 1] == dims - 1)
        {
            out_elempack = elempack;
        }
        else
        {
            const int n = outshape[dims - 1];
#if __AVX512F__
            out_elempack = n % 16 == 0 ? 16 : n % 8 == 0 ? 8 : n % 4 == 0   ? 4 : 1;
#elif __AVX__
            out_elempack = n % 8 == 0 ? 8 : n % 4 == 0 ? 4 : 1;
#else
            out_elempack = n % 4 == 0 ? 4 : 1;
#endif
        }
    }
#endif // __SSE2__
    outshape[dims - 1] /= out_elempack;
    const size_t out_elemsize = lane_size * out_elempack;
    if (dims == 2)
        top_blob.create(outshape[0], outshape[1], out_elemsize, out_elempack, opt.blob_allocator);
    if (dims == 3)
        top_blob.create(outshape[0], outshape[1], outshape[2], out_elemsize, out_elempack, opt.blob_allocator);
    if (dims == 4)
        top_blob.create(outshape[0], outshape[1], outshape[2], outshape[3], out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (dims == 2)
    {
        const int w = bottom_blob.w;
        const int h = bottom_blob.h * elempack;
        const size_t stride = (size_t)w * bottom_blob.elemsize;
        const size_t outstride = (size_t)top_blob.w * top_blob.elemsize;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int x = 0; x < w; x += 32)
        {
            const unsigned char* ptr = (const unsigned char*)bottom_blob + x * bottom_blob.elemsize;
            unsigned char* outptr = (unsigned char*)top_blob + (x / out_elempack) * outstride;
            const int cols = std::min(32, w - x);
            if (elembits == 32)
                permute_transpose2d_fp32(ptr, stride, outptr, outstride, h, cols, elempack, out_elempack);
            else
                permute_transpose2d_bf16s_fp16s(ptr, stride, outptr, outstride, h, cols, elempack, out_elempack);
        }
        return 0;
    }

    if (dims == 3 && (order_type != 1 || elempack == out_elempack))
    {
        permute3d(bottom_blob, top_blob, order_type, opt);
        return 0;
    }

    if (dims == 4)
    {
        if (order_type <= 5 && elempack == out_elempack)
        {
            permute4d_spatial(bottom_blob, top_blob, order_type, opt);
            return 0;
        }

        if (order_type == 6 || order_type == 9)
        {
            // w,h are adjacent on both sides: view them as a single width.
            Mat bottom = bottom_blob;
            bottom.dims = 3;
            bottom.w *= bottom.h;
            bottom.h = bottom.d;
            bottom.d = 1;
            Mat top = top_blob;
            top.dims = 3;
            if (order_type == 6)
            {
                top.w *= top.h;
                top.h = top.d;
            }
            else
            {
                top.h *= top.d;
            }
            top.d = 1;
            permute3d(bottom, top, order_type == 6 ? 2 : 3, opt);
            return 0;
        }

        if (order_type == 18 || order_type == 21)
        {
            // h,d are adjacent on both sides: view them as a single height.
            Mat bottom = bottom_blob;
            bottom.dims = 3;
            bottom.h *= bottom.d;
            bottom.d = 1;
            Mat top = top_blob;
            top.dims = 3;
            if (order_type == 18)
            {
                top.w *= top.h;
                top.h = top.d;
            }
            else
            {
                top.h *= top.d;
            }
            top.d = 1;
            permute3d(bottom, top, order_type == 18 ? 4 : 5, opt);
            return 0;
        }

        if (order_type == 14 || order_type == 15 || order_type == 20)
        {
            // Depth is unchanged. Keep the original cstep in the 3d views.
            const int order3d = order_type == 14 ? 2 : order_type == 15 ? 3 : 4;
            Option opt1 = opt;
            opt1.num_threads = 1;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int z = 0; z < bottom_blob.d; z++)
            {
                Mat bottom = bottom_blob;
                bottom.dims = 3;
                bottom.d = 1;
                bottom.data = (unsigned char*)bottom.data + (size_t)z * bottom.w * bottom.h * bottom.elemsize;
                Mat top = top_blob;
                top.dims = 3;
                top.d = 1;
                top.data = (unsigned char*)top.data + (size_t)z * top.w * top.h * top.elemsize;
                permute3d(bottom, top, order3d, opt1);
            }
            return 0;
        }
    }

    const int pack_axis = dims - 1;
    const int out_pack_axis = order[dims - 1];
    const int a = elempack == 1 ? 0 : pack_axis;
    const int b = out_elempack == 1 ? order[0] : out_pack_axis;
    size_t strides[4];
    size_t ds[4];
    size_t outstrides[4];
    permute_strides(bottom_blob, strides);
    permute_strides(top_blob, ds);
    for (int i = 0; i < dims; i++)
        outstrides[order[i]] = ds[i];

    if (elempack == 1 && out_elempack == 1 && order[0] == 0)
    {
        const size_t row_size = (size_t)bottom_blob.w * lane_size;
        const size_t sy = strides[order[1]] * lane_size;
        const size_t sz = dims == 4 ? strides[order[2]] * lane_size : 0;
        const size_t sc = strides[order[dims - 1]] * lane_size;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned char* ptr = (const unsigned char*)bottom_blob + q * sc;
            unsigned char* outptr = top_blob.channel(q);
            for (int z = 0; z < top_blob.d; z++)
            {
                for (int y = 0; y < top_blob.h; y++)
                {
                    memcpy(outptr, ptr + z * sz + y * sy, row_size);
                    outptr += row_size;
                }
            }
        }
        return 0;
    }

    if (dims == 4 && elempack == 1 && out_elempack == 1 && a != b)
    {
        int outer[2];
        int n = 0;
        for (int i = 0; i < dims; i++)
            if (i != a && i != b) outer[n++] = i;
        const int u = outer[0];
        const int v = outer[1];
        permute_transpose_func transpose_plane = elembits == 32 ? permute_transpose_pack1_fp32 : permute_transpose_pack1_bf16s_fp16s;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < shape[v]; q++)
        {
            const unsigned char* ptr = (const unsigned char*)bottom_blob + q * strides[v] * lane_size;
            unsigned char* outptr = (unsigned char*)top_blob + q * outstrides[v] * lane_size;
            for (int i = 0; i < shape[u]; i++)
            {
                transpose_plane(ptr, strides[b] * lane_size, outptr, outstrides[a] * lane_size, shape[b], shape[a]);
                ptr += strides[u] * lane_size;
                outptr += outstrides[u] * lane_size;
            }
        }
        return 0;
    }

    if (a == b)
    {
        // The contiguous logical axis is the same on both sides, but its pack
        // groups can have different strides. Join/split small packs in registers.
        const int pack = elempack > 1 ? elempack : out_elempack > 1 ? out_elempack : shape[a];
        const int groups = shape[a] / pack;
        const size_t bytes = pack * lane_size;
        const size_t srcstep = a == pack_axis ? strides[a] * lane_size : bytes;
        const size_t dststep = a == out_pack_axis ? outstrides[a] * lane_size : bytes;
        permute_copy_func copy = bytes == 8 ? permute_copy_64bit : permute_copy_128bit;
#if __AVX__
        if (bytes == 32) copy = permute_copy_256bit;
#endif
#if __AVX512F__
        if (bytes == 64) copy = permute_copy_512bit;
#endif
        size_t count = 1;
        for (int i = 0; i < dims; i++)
            if (i != a) count *= shape[i];

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int64_t t = 0; t < (int64_t)count; t++)
        {
            size_t v = t;
            int pos[4] = {0, 0, 0, 0};
            for (int i = 0; i < dims; i++)
            {
                if (i == a) continue;
                pos[i] = v % shape[i];
                v /= shape[i];
            }
            const unsigned char* ptr = (const unsigned char*)bottom_blob + permute_offset(pos, strides, dims, pack_axis, elempack) * lane_size;
            unsigned char* outptr = (unsigned char*)top_blob + permute_offset(pos, outstrides, dims, out_pack_axis, out_elempack) * lane_size;
            copy(ptr, srcstep, outptr, dststep, groups);
        }
        return 0;
    }

    int outer[2] = {0, 0};
    int outer_count = 0;
    for (int i = 0; i < dims; i++)
        if (i != a && i != b) outer[outer_count++] = i;
    const int n0 = shape[outer[0]];
    const int n1 = outer_count == 2 ? shape[outer[1]] : 1;
    const size_t s0 = strides[outer[0]] * lane_size;
    const size_t s1 = outer_count == 2 ? strides[outer[1]] * lane_size : 0;
    const size_t d0 = outstrides[outer[0]] * lane_size;
    const size_t d1 = outer_count == 2 ? outstrides[outer[1]] * lane_size : 0;
    const int cols = elempack > 1 ? elempack : shape[a];
    const int rows = out_elempack > 1 ? out_elempack : shape[b];
    const int na = shape[a] / cols;
    const int nb = shape[b] / rows;
    const size_t src_rowstep = strides[b] * lane_size;
    const size_t dst_rowstep = outstrides[a] * lane_size;
    const size_t src_astep = (elempack > 1 ? strides[a] : cols * strides[a]) * lane_size;
    const size_t dst_astep = cols * outstrides[a] * lane_size;
    const size_t src_bstep = rows * strides[b] * lane_size;
    const size_t dst_bstep = (out_elempack > 1 ? outstrides[b] : rows * outstrides[b]) * lane_size;
    typedef void (*transpose_tile_func)(const unsigned char*, size_t, unsigned char*, size_t);
    transpose_tile_func transpose_tile = 0;
    if (elempack == 4 && out_elempack == 4)
        transpose_tile = elembits == 32 ? permute_transpose4x4_fp32 : permute_transpose4x4_bf16s_fp16s;
    if (elempack == 4 && out_elempack == 8)
        transpose_tile = elembits == 32 ? permute_transpose8x4_fp32 : permute_transpose8x4_bf16s_fp16s;
    if (elempack == 4 && out_elempack == 16)
        transpose_tile = elembits == 32 ? permute_transpose16x4_fp32 : permute_transpose16x4_bf16s_fp16s;
    if (elempack == 8 && out_elempack == 4)
        transpose_tile = elembits == 32 ? permute_transpose4x8_fp32 : permute_transpose4x8_bf16s_fp16s;
    if (elempack == 8 && out_elempack == 8)
        transpose_tile = elembits == 32 ? permute_transpose8x8_fp32 : permute_transpose8x8_bf16s_fp16s;
    if (elempack == 8 && out_elempack == 16)
        transpose_tile = elembits == 32 ? permute_transpose16x8_fp32 : permute_transpose16x8_bf16s_fp16s;
    if (elempack == 16 && out_elempack == 4)
        transpose_tile = elembits == 32 ? permute_transpose4x16_fp32 : permute_transpose4x16_bf16s_fp16s;
    if (elempack == 16 && out_elempack == 8)
        transpose_tile = elembits == 32 ? permute_transpose8x16_fp32 : permute_transpose8x16_bf16s_fp16s;
    if (elempack == 16 && out_elempack == 16)
        transpose_tile = elembits == 32 ? permute_transpose16x16_fp32 : permute_transpose16x16_bf16s_fp16s;
    if (transpose_tile)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int64_t q = 0; q < (int64_t)n0 * n1; q++)
        {
            const unsigned char* ptr = (const unsigned char*)bottom_blob + q % n0 * s0 + q / n0 * s1;
            unsigned char* outptr = (unsigned char*)top_blob + q % n0 * d0 + q / n0 * d1;
            for (int y = 0; y < nb; y++)
            {
                const unsigned char* p = ptr + y * src_bstep;
                unsigned char* out = outptr + y * dst_bstep;
                for (int x = 0; x < na; x++)
                {
                    transpose_tile(p, src_rowstep, out, dst_rowstep);
                    p += src_astep;
                    out += dst_astep;
                }
            }
        }
    }
    else
    {
        permute_transpose_func transpose_plane = elembits == 32 ? permute_transpose_pack1_fp32 : permute_transpose_pack1_bf16s_fp16s;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int64_t q = 0; q < (int64_t)n0 * n1; q++)
        {
            const unsigned char* ptr = (const unsigned char*)bottom_blob + q % n0 * s0 + q / n0 * s1;
            unsigned char* outptr = (unsigned char*)top_blob + q % n0 * d0 + q / n0 * d1;
            for (int y = 0; y < nb; y++)
            {
                const unsigned char* p = ptr + y * src_bstep;
                unsigned char* out = outptr + y * dst_bstep;
                for (int x = 0; x < na; x++)
                {
                    transpose_plane(p, src_rowstep, out, dst_rowstep, rows, cols);
                    p += src_astep;
                    out += dst_astep;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
