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

#if NCNN_BF16
#include "sdpa_prefill.h"
#include "sdpa_decode.h"
#include "sdpa_prefill_bf16s.h"
#include "sdpa_decode_bf16s.h"
#include "sdpa_kvcache.h"

int sdpa_decode_bf16s_avx2(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
{
    return sdpa_decode_bf16s(query, key, value, attn_mask_blob, top_blob, scale, opt);
}

int sdpa_prefill_bf16s_avx2(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
{
    return sdpa_prefill_bf16s(query, key, value, attn_mask_blob, top_blob, scale, opt);
}

int sdpa_kvcache_bf16s_avx2(const Mat& query, const Mat& past_key, const Mat& past_value, const Mat& cur_key, const Mat& cur_value, Mat& cached_key, Mat& cached_value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
{
    return sdpa_kvcache_bf16s(query, past_key, past_value, cur_key, cur_value, cached_key, cached_value, attn_mask_blob, top_blob, scale, opt);
}
#endif // NCNN_BF16

} // namespace ncnn
