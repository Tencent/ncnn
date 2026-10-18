// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "sdpa_x86.h"

#include <float.h>
#include <limits.h>

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#endif // __AVX__
#endif // __SSE2__

#include "cpu.h"
#include "x86_usability.h"

namespace ncnn {

#include "sdpa_prefill.h"
#include "sdpa_decode.h"
#if NCNN_BF16
#include "sdpa_prefill_bf16s.h"
#include "sdpa_decode_bf16s.h"
#endif // NCNN_BF16

void sdpa_attention_tile_fma4(const Mat& queryT, const Mat& key_head, const Mat& packed_key_head, const Mat& value_head, const Mat& packed_value_head, const Mat& computation_value_head, const Mat& mask, size_t mask_hstep, const Mat& packed_mask, Mat& scoreT, Mat& outT, Mat& lT, int max_ii)
{
    sdpa_attention_tile(queryT, key_head, packed_key_head, value_head, packed_value_head, computation_value_head, mask, mask_hstep, packed_mask, scoreT, outT, lT, max_ii);
}

void sdpa_decode_tile_fma4(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q0, int max_qq, int g, int block_n, Mat& workspace)
{
    sdpa_decode_tile(query, key, value, attn_mask_blob, top_blob, scale, q0, max_qq, g, block_n, workspace);
}

void sdpa_decode_kvcache_small_tile_fma4(const Mat& query, const Mat& key_cache, const Mat& value_cache, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q0, int max_qq, int g, int block_n, Mat& workspace)
{
    sdpa_decode_kvcache_small_tile(query, key_cache, value_cache, attn_mask_blob, top_blob, scale, q0, max_qq, g, block_n, workspace);
}

#if NCNN_BF16
void sdpa_attention_tile_bf16s_fma4(const Mat& queryT, const Mat& key_head, const Mat& packed_key_head, const Mat& value_head, const Mat& packed_value_head, const Mat& computation_value_head, const Mat& mask, size_t mask_hstep, const Mat& packed_mask, Mat& scoreT, Mat& outT, Mat& lT, int max_ii, float scale)
{
    sdpa_attention_tile_bf16s(queryT, key_head, packed_key_head, value_head, packed_value_head, computation_value_head, mask, mask_hstep, packed_mask, scoreT, outT, lT, max_ii, scale);
}

void sdpa_decode_tile_bf16s_fma4(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q0, int max_qq, int g, int block_n, Mat& workspace)
{
    sdpa_decode_tile_bf16s(query, key, value, attn_mask_blob, top_blob, scale, q0, max_qq, g, block_n, workspace);
}

void sdpa_decode_kvcache_small_tile_bf16s_fma4(const Mat& query, const Mat& key_cache, const Mat& value_cache, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q0, int max_qq, int g, int block_n, Mat& workspace)
{
    sdpa_decode_kvcache_small_tile_bf16s(query, key_cache, value_cache, attn_mask_blob, top_blob, scale, q0, max_qq, g, block_n, workspace);
}
#endif // NCNN_BF16

} // namespace ncnn
