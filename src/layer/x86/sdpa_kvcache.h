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

static int sdpa_kvcache(const Mat& query, const Mat& past_key, const Mat& past_value, const Mat& cur_key, const Mat& cur_value, Mat& cached_key, Mat& cached_value, const Mat& attn_mask, Mat& top_blob, float scale, const Option& opt)
{
    const int past_seqlen = past_key.empty() ? 0 : past_key.h;
    const int dst_seqlen = past_seqlen + cur_key.h;
#if __AVX512F__
    const int panel_width = 16;
#elif __AVX__
    const int panel_width = 8;
#elif __SSE2__
    const int panel_width = 4;
#else
    const int panel_width = 2;
#endif

    int ret = sdpa_create_or_grow_kvcache(past_key, cached_key, dst_seqlen, cur_key.c, cur_key.w, cur_key.elemsize, panel_width, opt);
    if (ret != 0)
        return ret;

    ret = sdpa_create_or_grow_kvcache(past_value, cached_value, dst_seqlen, cur_value.c, cur_value.w, cur_value.elemsize, panel_width, opt);
    if (ret != 0)
        return ret;

    const int num_kv_heads = cur_key.c;
    const int first_panel = past_seqlen / panel_width;
    const int num_panels = (past_seqlen % panel_width + cur_key.h + panel_width - 1) / panel_width;
    if (cur_key.h == 1)
    {
        sdpa_append_kvcache_token(cur_key, cur_value, cached_key, cached_value, past_seqlen, panel_width);
    }
    else
    {
        const int num_panel_tasks = std::min(num_panels, std::max(1, (opt.num_threads + num_kv_heads - 1) / num_kv_heads));
        const int num_tasks = num_kv_heads * num_panel_tasks;
        const int nT = cur_key.h >= panel_width ? std::min(opt.num_threads, num_tasks) : 1;

        #pragma omp parallel for num_threads(nT)
        for (int task_id = 0; task_id < num_tasks; task_id++)
        {
            const int g = task_id / num_panel_tasks;
            const int panel_task_id = task_id % num_panel_tasks;
            const int panel_begin_id = panel_task_id * num_panels / num_panel_tasks;
            const int panel_end_id = (panel_task_id + 1) * num_panels / num_panel_tasks;
            const Mat key_head = cur_key.channel(g);
            const Mat value_head = cur_value.channel(g);
            Mat packed_key_head = cached_key.channel(g);
            Mat packed_value_head = cached_value.channel(g);

            for (int panel_offset = panel_begin_id; panel_offset < panel_end_id; panel_offset++)
            {
                const int panel_id = first_panel + panel_offset;
                const int panel_begin = panel_id * panel_width;
                const int n_begin = std::max(past_seqlen, panel_begin);
                const int n_end = std::min(dst_seqlen, panel_begin + panel_width);
                Mat packed_key_tile(cur_key.w * panel_width, (float*)packed_key_head + (size_t)panel_id * cur_key.w * panel_width, 4u);
                Mat packed_value_tile(cur_value.w * panel_width, (float*)packed_value_head + (size_t)panel_id * cur_value.w * panel_width, 4u);

                sdpa_pack_key_tile(key_head, packed_key_tile, n_begin - past_seqlen, n_begin - panel_begin, n_end - n_begin);
                sdpa_pack_value_tile(value_head, packed_value_tile, n_begin - past_seqlen, n_begin - panel_begin, n_end - n_begin);
            }
        }
    }

    if (query.h == 1)
        return sdpa_decode_kvcache(query, cached_key, cached_value, attn_mask, top_blob, scale, opt);

    return sdpa_prefill_packed(query, cached_key, cached_value, attn_mask, top_blob, scale, opt);
}

#if NCNN_BF16
static int sdpa_kvcache_bf16s(const Mat& query, const Mat& past_key, const Mat& past_value, const Mat& cur_key, const Mat& cur_value, Mat& cached_key, Mat& cached_value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
        return sdpa_kvcache_bf16s_avx512bf16(query, past_key, past_value, cur_key, cur_value, cached_key, cached_value, attn_mask_blob, top_blob, scale, opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx2())
        return sdpa_kvcache_bf16s_avx2(query, past_key, past_value, cur_key, cur_value, cached_key, cached_value, attn_mask_blob, top_blob, scale, opt);
#endif

    const int past_seqlen = past_key.empty() ? 0 : past_key.h;
    const int dst_seqlen = past_seqlen + cur_key.h;
#if __AVX512F__
    const int panel_width = 16;
#elif __AVX__
    const int panel_width = 8;
#elif __SSE2__
    const int panel_width = 4;
#else
    const int panel_width = 2;
#endif

    int ret = sdpa_create_or_grow_kvcache(past_key, cached_key, dst_seqlen, cur_key.c, cur_key.w, cur_key.elemsize, panel_width, opt);
    if (ret != 0)
        return ret;

    ret = sdpa_create_or_grow_kvcache(past_value, cached_value, dst_seqlen, cur_value.c, cur_value.w, cur_value.elemsize, panel_width, opt);
    if (ret != 0)
        return ret;

    const int num_kv_heads = cur_key.c;
    const int first_panel = past_seqlen / panel_width;
    const int num_panels = (past_seqlen % panel_width + cur_key.h + panel_width - 1) / panel_width;
    if (cur_key.h == 1)
    {
        sdpa_append_kvcache_token_bf16s(cur_key, cur_value, cached_key, cached_value, past_seqlen, panel_width);
    }
    else
    {
        const int num_panel_tasks = std::min(num_panels, std::max(1, (opt.num_threads + num_kv_heads - 1) / num_kv_heads));
        const int num_tasks = num_kv_heads * num_panel_tasks;
        const int nT = cur_key.h >= panel_width ? std::min(opt.num_threads, num_tasks) : 1;

        #pragma omp parallel for num_threads(nT)
        for (int task_id = 0; task_id < num_tasks; task_id++)
        {
            const int g = task_id / num_panel_tasks;
            const int panel_task_id = task_id % num_panel_tasks;
            const int panel_begin_id = panel_task_id * num_panels / num_panel_tasks;
            const int panel_end_id = (panel_task_id + 1) * num_panels / num_panel_tasks;
            const Mat key_head = cur_key.channel(g);
            const Mat value_head = cur_value.channel(g);
            Mat packed_key_head = cached_key.channel(g);
            Mat packed_value_head = cached_value.channel(g);

            for (int panel_offset = panel_begin_id; panel_offset < panel_end_id; panel_offset++)
            {
                const int panel_id = first_panel + panel_offset;
                const int panel_begin = panel_id * panel_width;
                const int n_begin = std::max(past_seqlen, panel_begin);
                const int n_end = std::min(dst_seqlen, panel_begin + panel_width);
                Mat packed_key_tile(cur_key.w * panel_width, (unsigned short*)packed_key_head + (size_t)panel_id * cur_key.w * panel_width, 2u);
                Mat packed_value_tile(cur_value.w * panel_width, (unsigned short*)packed_value_head + (size_t)panel_id * cur_value.w * panel_width, 2u);
                sdpa_pack_key_tile_bf16s(key_head, packed_key_tile, n_begin - past_seqlen, n_begin - panel_begin, n_end - n_begin);
                sdpa_pack_value_tile_bf16s(value_head, packed_value_tile, n_begin - past_seqlen, n_begin - panel_begin, n_end - n_begin);
            }
        }
    }

    if (query.h == 1)
        return sdpa_decode_kvcache_bf16s(query, cached_key, cached_value, attn_mask_blob, top_blob, scale, opt);

    return sdpa_prefill_packed_bf16s(query, cached_key, cached_value, Mat(), attn_mask_blob, top_blob, scale, opt);
}
#endif // NCNN_BF16
