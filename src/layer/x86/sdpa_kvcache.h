// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static int sdpa_kvcache_capacity(int current_capacity, int new_seqlen, int max_seqlen_hint)
{
    if (current_capacity == 0 && max_seqlen_hint >= new_seqlen && max_seqlen_hint > 0)
        return max_seqlen_hint;

    int capacity = current_capacity > new_seqlen ? current_capacity : new_seqlen;
    int reserve;
    if (current_capacity == 0)
    {
        reserve = capacity < 16 ? 16 - capacity : capacity;
        if (reserve > 256)
            reserve = 256;
    }
    else
    {
        reserve = capacity / 2;
        if (reserve < 16)
            reserve = 16;
    }

    return capacity <= INT_MAX - reserve ? capacity + reserve : capacity;
}

static int sdpa_kvcache_value_panel_width(int remain)
{
#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (remain >= 16)
        return 16;
#endif // __AVX512F__
    if (remain >= 8)
        return 8;
#endif // __AVX__
    if (remain >= 4)
        return 4;
#endif // __SSE2__
    if (remain >= 2)
        return 2;
    return 1;
}

static int sdpa_create_or_grow_kvcache(const Mat& cache, Mat& new_cache, int new_seqlen, int num_kv_heads, int dim, size_t elemsize, int panel_width, const Option& opt)
{
    const int scalar_size = (int)elemsize;
    const int capacity_align = std::max(panel_width, 16 / scalar_size);
    Allocator* allocator = opt.kvcache_allocator;
    const bool reuse = !cache.empty() && cache.allocator == allocator;
    const int current_capacity = cache.empty() ? 0 : (int)(cache.cstep / cache.w);

    if (reuse && new_seqlen <= current_capacity)
    {
        new_cache = cache;
        new_cache.h = new_seqlen;
        return 0;
    }

    int capacity = sdpa_kvcache_capacity(current_capacity, new_seqlen, opt.kvcache_max_seqlen_hint);
    const size_t aligned_capacity = ((size_t)capacity + capacity_align - 1) / capacity_align * capacity_align;
    if (aligned_capacity > INT_MAX)
        return -100;
    capacity = (int)aligned_capacity;

    Mat m(dim, capacity, num_kv_heads, elemsize, 1, allocator);
    if (m.empty())
        return -100;

    m.h = new_seqlen;

    if (!cache.empty())
    {
        const size_t valid_capacity = ((size_t)cache.h + panel_width - 1) / panel_width * panel_width;
        const size_t valid_head_size = (size_t)dim * valid_capacity * elemsize;
        for (int q = 0; q < num_kv_heads; q++)
        {
            const unsigned char* src = (const unsigned char*)cache.data + cache.cstep * q * elemsize;
            unsigned char* dst = (unsigned char*)m.data + m.cstep * q * elemsize;
            memcpy(dst, src, valid_head_size);
        }
    }

    new_cache = m;

    return 0;
}
