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
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#include "cpu.h"
#include "x86_usability.h"

namespace ncnn {

#include "sdpa_kvcache.h"
#include "sdpa_decode.h"
#include "sdpa_prefill.h"
#include "sdpa_decode_bf16s.h"
#include "sdpa_prefill_bf16s.h"

void sdpa_decode_tile_bf16s_avx2(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q0, int max_qq, int g, int n_begin, int n_end, int block_n, const Mat& packed_query, Mat& workspace, Mat& state)
{
    sdpa_decode_tile_bf16s(query, key, value, attn_mask_blob, top_blob, scale, q0, max_qq, g, n_begin, n_end, block_n, packed_query, workspace, state);
}

void sdpa_decode_kvcache_tile_bf16s_avx2(const Mat& query, const Mat& key_cache, const Mat& value_cache, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q0, int max_qq, int g, int n_begin, int n_end, int block_n, const Mat& packed_query, Mat& workspace, Mat& state)
{
    sdpa_decode_kvcache_tile_bf16s(query, key_cache, value_cache, attn_mask_blob, top_blob, scale, q0, max_qq, g, n_begin, n_end, block_n, packed_query, workspace, state);
}

void sdpa_pack_value_tile_bf16s_fp32_avx2(const Mat& packed_value, Mat& packed_value_fp32, int src_begin, int dst_begin, int max_seqlen, int dst_seqlen)
{
    sdpa_pack_value_tile_bf16s_fp32(packed_value, packed_value_fp32, src_begin, dst_begin, max_seqlen, dst_seqlen);
}

void sdpa_pack_value_tile_bf16s_to_fp32_avx2(const Mat& value, Mat& packed_value_fp32, int src_begin, int dst_begin, int max_seqlen, int dst_seqlen)
{
    sdpa_pack_value_tile_bf16s_to_fp32(value, packed_value_fp32, src_begin, dst_begin, max_seqlen, dst_seqlen);
}

void sdpa_prefill_packed_tile_bf16s_avx2(const Mat& queryT, const Mat& packed_key_head, const Mat& packed_value_head, const Mat& packed_value_fp32_head, const Mat& maskT, Mat& scoreT, Mat& outT, Mat& stateT, int max_ii, int n_begin, int n_end, float scale)
{
    sdpa_prefill_packed_tile_bf16s(queryT, packed_key_head, packed_value_head, packed_value_fp32_head, maskT, scoreT, outT, stateT, max_ii, n_begin, n_end, scale);
}

} // namespace ncnn
