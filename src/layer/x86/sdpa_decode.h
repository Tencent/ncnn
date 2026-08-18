// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void sdpa_dot_product_tile4_fp32(const float* query0, const float* query1, const float* query2, const float* query3, const float* key, int size, float& sum0, float& sum1, float& sum2, float& sum3)
{
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _sum0_avx512 = _mm512_setzero_ps();
    __m512 _sum1_avx512 = _mm512_setzero_ps();
    __m512 _sum2_avx512 = _mm512_setzero_ps();
    __m512 _sum3_avx512 = _mm512_setzero_ps();
#endif // __AVX512F__
    __m256 _sum0_avx = _mm256_setzero_ps();
    __m256 _sum1_avx = _mm256_setzero_ps();
    __m256 _sum2_avx = _mm256_setzero_ps();
    __m256 _sum3_avx = _mm256_setzero_ps();
#endif // __AVX__
    __m128 _sum0 = _mm_setzero_ps();
    __m128 _sum1 = _mm_setzero_ps();
    __m128 _sum2 = _mm_setzero_ps();
    __m128 _sum3 = _mm_setzero_ps();
#endif // __SSE2__
    sum0 = 0.f;
    sum1 = 0.f;
    sum2 = 0.f;
    sum3 = 0.f;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _k = _mm512_loadu_ps(key + i);
        _sum0_avx512 = _mm512_fmadd_ps(_mm512_loadu_ps(query0 + i), _k, _sum0_avx512);
        _sum1_avx512 = _mm512_fmadd_ps(_mm512_loadu_ps(query1 + i), _k, _sum1_avx512);
        _sum2_avx512 = _mm512_fmadd_ps(_mm512_loadu_ps(query2 + i), _k, _sum2_avx512);
        _sum3_avx512 = _mm512_fmadd_ps(_mm512_loadu_ps(query3 + i), _k, _sum3_avx512);
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _k = _mm256_loadu_ps(key + i);
        _sum0_avx = _mm256_comp_fmadd_ps(_mm256_loadu_ps(query0 + i), _k, _sum0_avx);
        _sum1_avx = _mm256_comp_fmadd_ps(_mm256_loadu_ps(query1 + i), _k, _sum1_avx);
        _sum2_avx = _mm256_comp_fmadd_ps(_mm256_loadu_ps(query2 + i), _k, _sum2_avx);
        _sum3_avx = _mm256_comp_fmadd_ps(_mm256_loadu_ps(query3 + i), _k, _sum3_avx);
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _k = _mm_loadu_ps(key + i);
        _sum0 = _mm_comp_fmadd_ps(_mm_loadu_ps(query0 + i), _k, _sum0);
        _sum1 = _mm_comp_fmadd_ps(_mm_loadu_ps(query1 + i), _k, _sum1);
        _sum2 = _mm_comp_fmadd_ps(_mm_loadu_ps(query2 + i), _k, _sum2);
        _sum3 = _mm_comp_fmadd_ps(_mm_loadu_ps(query3 + i), _k, _sum3);
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        const float k = key[i];
        sum0 += query0[i] * k;
        sum1 += query1[i] * k;
        sum2 += query2[i] * k;
        sum3 += query3[i] * k;
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    sum0 += _mm512_comp_reduce_add_ps(_sum0_avx512);
    sum1 += _mm512_comp_reduce_add_ps(_sum1_avx512);
    sum2 += _mm512_comp_reduce_add_ps(_sum2_avx512);
    sum3 += _mm512_comp_reduce_add_ps(_sum3_avx512);
#endif // __AVX512F__
    sum0 += _mm256_reduce_add_ps(_sum0_avx);
    sum1 += _mm256_reduce_add_ps(_sum1_avx);
    sum2 += _mm256_reduce_add_ps(_sum2_avx);
    sum3 += _mm256_reduce_add_ps(_sum3_avx);
#endif // __AVX__
    sum0 += _mm_reduce_add_ps(_sum0);
    sum1 += _mm_reduce_add_ps(_sum1);
    sum2 += _mm_reduce_add_ps(_sum2);
    sum3 += _mm_reduce_add_ps(_sum3);
#endif // __SSE2__
}

static void sdpa_pv_tile4_fp32(float* out0, float* out1, float* out2, float* out3, const float* value, const float* score0, const float* score1, const float* score2, const float* score3, float alpha0, float alpha1, float alpha2, float alpha3, int size, int value_dim)
{
    int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; d + 63 < value_dim; d += 64)
    {
        __m512 _out00 = _mm512_mul_ps(_mm512_loadu_ps(out0 + d), _mm512_set1_ps(alpha0));
        __m512 _out01 = _mm512_mul_ps(_mm512_loadu_ps(out0 + d + 16), _mm512_set1_ps(alpha0));
        __m512 _out02 = _mm512_mul_ps(_mm512_loadu_ps(out0 + d + 32), _mm512_set1_ps(alpha0));
        __m512 _out03 = _mm512_mul_ps(_mm512_loadu_ps(out0 + d + 48), _mm512_set1_ps(alpha0));
        __m512 _out10 = _mm512_mul_ps(_mm512_loadu_ps(out1 + d), _mm512_set1_ps(alpha1));
        __m512 _out11 = _mm512_mul_ps(_mm512_loadu_ps(out1 + d + 16), _mm512_set1_ps(alpha1));
        __m512 _out12 = _mm512_mul_ps(_mm512_loadu_ps(out1 + d + 32), _mm512_set1_ps(alpha1));
        __m512 _out13 = _mm512_mul_ps(_mm512_loadu_ps(out1 + d + 48), _mm512_set1_ps(alpha1));
        __m512 _out20 = _mm512_mul_ps(_mm512_loadu_ps(out2 + d), _mm512_set1_ps(alpha2));
        __m512 _out21 = _mm512_mul_ps(_mm512_loadu_ps(out2 + d + 16), _mm512_set1_ps(alpha2));
        __m512 _out22 = _mm512_mul_ps(_mm512_loadu_ps(out2 + d + 32), _mm512_set1_ps(alpha2));
        __m512 _out23 = _mm512_mul_ps(_mm512_loadu_ps(out2 + d + 48), _mm512_set1_ps(alpha2));
        __m512 _out30 = _mm512_mul_ps(_mm512_loadu_ps(out3 + d), _mm512_set1_ps(alpha3));
        __m512 _out31 = _mm512_mul_ps(_mm512_loadu_ps(out3 + d + 16), _mm512_set1_ps(alpha3));
        __m512 _out32 = _mm512_mul_ps(_mm512_loadu_ps(out3 + d + 32), _mm512_set1_ps(alpha3));
        __m512 _out33 = _mm512_mul_ps(_mm512_loadu_ps(out3 + d + 48), _mm512_set1_ps(alpha3));

        for (int j = 0; j < size; j++)
        {
            const float* vptr = value + (size_t)j * value_dim + d;
            __m512 _v0 = _mm512_loadu_ps(vptr);
            __m512 _v1 = _mm512_loadu_ps(vptr + 16);
            __m512 _v2 = _mm512_loadu_ps(vptr + 32);
            __m512 _v3 = _mm512_loadu_ps(vptr + 48);
            __m512 _p = _mm512_set1_ps(score0[j]);
            _out00 = _mm512_fmadd_ps(_v0, _p, _out00);
            _out01 = _mm512_fmadd_ps(_v1, _p, _out01);
            _out02 = _mm512_fmadd_ps(_v2, _p, _out02);
            _out03 = _mm512_fmadd_ps(_v3, _p, _out03);
            _p = _mm512_set1_ps(score1[j]);
            _out10 = _mm512_fmadd_ps(_v0, _p, _out10);
            _out11 = _mm512_fmadd_ps(_v1, _p, _out11);
            _out12 = _mm512_fmadd_ps(_v2, _p, _out12);
            _out13 = _mm512_fmadd_ps(_v3, _p, _out13);
            _p = _mm512_set1_ps(score2[j]);
            _out20 = _mm512_fmadd_ps(_v0, _p, _out20);
            _out21 = _mm512_fmadd_ps(_v1, _p, _out21);
            _out22 = _mm512_fmadd_ps(_v2, _p, _out22);
            _out23 = _mm512_fmadd_ps(_v3, _p, _out23);
            _p = _mm512_set1_ps(score3[j]);
            _out30 = _mm512_fmadd_ps(_v0, _p, _out30);
            _out31 = _mm512_fmadd_ps(_v1, _p, _out31);
            _out32 = _mm512_fmadd_ps(_v2, _p, _out32);
            _out33 = _mm512_fmadd_ps(_v3, _p, _out33);
        }

        _mm512_storeu_ps(out0 + d, _out00);
        _mm512_storeu_ps(out0 + d + 16, _out01);
        _mm512_storeu_ps(out0 + d + 32, _out02);
        _mm512_storeu_ps(out0 + d + 48, _out03);
        _mm512_storeu_ps(out1 + d, _out10);
        _mm512_storeu_ps(out1 + d + 16, _out11);
        _mm512_storeu_ps(out1 + d + 32, _out12);
        _mm512_storeu_ps(out1 + d + 48, _out13);
        _mm512_storeu_ps(out2 + d, _out20);
        _mm512_storeu_ps(out2 + d + 16, _out21);
        _mm512_storeu_ps(out2 + d + 32, _out22);
        _mm512_storeu_ps(out2 + d + 48, _out23);
        _mm512_storeu_ps(out3 + d, _out30);
        _mm512_storeu_ps(out3 + d + 16, _out31);
        _mm512_storeu_ps(out3 + d + 32, _out32);
        _mm512_storeu_ps(out3 + d + 48, _out33);
    }

    for (; d + 15 < value_dim; d += 16)
    {
        __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(out0 + d), _mm512_set1_ps(alpha0));
        __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(out1 + d), _mm512_set1_ps(alpha1));
        __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(out2 + d), _mm512_set1_ps(alpha2));
        __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(out3 + d), _mm512_set1_ps(alpha3));

        for (int j = 0; j < size; j++)
        {
            __m512 _v = _mm512_loadu_ps(value + (size_t)j * value_dim + d);
            _out0 = _mm512_fmadd_ps(_v, _mm512_set1_ps(score0[j]), _out0);
            _out1 = _mm512_fmadd_ps(_v, _mm512_set1_ps(score1[j]), _out1);
            _out2 = _mm512_fmadd_ps(_v, _mm512_set1_ps(score2[j]), _out2);
            _out3 = _mm512_fmadd_ps(_v, _mm512_set1_ps(score3[j]), _out3);
        }

        _mm512_storeu_ps(out0 + d, _out0);
        _mm512_storeu_ps(out1 + d, _out1);
        _mm512_storeu_ps(out2 + d, _out2);
        _mm512_storeu_ps(out3 + d, _out3);
    }
#endif // __AVX512F__
#if !__AVX512F__
    for (; d + 15 < value_dim; d += 16)
    {
        __m256 _out00 = _mm256_mul_ps(_mm256_loadu_ps(out0 + d), _mm256_set1_ps(alpha0));
        __m256 _out01 = _mm256_mul_ps(_mm256_loadu_ps(out0 + d + 8), _mm256_set1_ps(alpha0));
        __m256 _out10 = _mm256_mul_ps(_mm256_loadu_ps(out1 + d), _mm256_set1_ps(alpha1));
        __m256 _out11 = _mm256_mul_ps(_mm256_loadu_ps(out1 + d + 8), _mm256_set1_ps(alpha1));
        __m256 _out20 = _mm256_mul_ps(_mm256_loadu_ps(out2 + d), _mm256_set1_ps(alpha2));
        __m256 _out21 = _mm256_mul_ps(_mm256_loadu_ps(out2 + d + 8), _mm256_set1_ps(alpha2));
        __m256 _out30 = _mm256_mul_ps(_mm256_loadu_ps(out3 + d), _mm256_set1_ps(alpha3));
        __m256 _out31 = _mm256_mul_ps(_mm256_loadu_ps(out3 + d + 8), _mm256_set1_ps(alpha3));

        for (int j = 0; j < size; j++)
        {
            const float* vptr = value + (size_t)j * value_dim + d;
            __m256 _v0 = _mm256_loadu_ps(vptr);
            __m256 _v1 = _mm256_loadu_ps(vptr + 8);
            __m256 _p = _mm256_set1_ps(score0[j]);
            _out00 = _mm256_comp_fmadd_ps(_v0, _p, _out00);
            _out01 = _mm256_comp_fmadd_ps(_v1, _p, _out01);
            _p = _mm256_set1_ps(score1[j]);
            _out10 = _mm256_comp_fmadd_ps(_v0, _p, _out10);
            _out11 = _mm256_comp_fmadd_ps(_v1, _p, _out11);
            _p = _mm256_set1_ps(score2[j]);
            _out20 = _mm256_comp_fmadd_ps(_v0, _p, _out20);
            _out21 = _mm256_comp_fmadd_ps(_v1, _p, _out21);
            _p = _mm256_set1_ps(score3[j]);
            _out30 = _mm256_comp_fmadd_ps(_v0, _p, _out30);
            _out31 = _mm256_comp_fmadd_ps(_v1, _p, _out31);
        }

        _mm256_storeu_ps(out0 + d, _out00);
        _mm256_storeu_ps(out0 + d + 8, _out01);
        _mm256_storeu_ps(out1 + d, _out10);
        _mm256_storeu_ps(out1 + d + 8, _out11);
        _mm256_storeu_ps(out2 + d, _out20);
        _mm256_storeu_ps(out2 + d + 8, _out21);
        _mm256_storeu_ps(out3 + d, _out30);
        _mm256_storeu_ps(out3 + d + 8, _out31);
    }
#endif // !__AVX512F__

    for (; d + 7 < value_dim; d += 8)
    {
        __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(out0 + d), _mm256_set1_ps(alpha0));
        __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(out1 + d), _mm256_set1_ps(alpha1));
        __m256 _out2 = _mm256_mul_ps(_mm256_loadu_ps(out2 + d), _mm256_set1_ps(alpha2));
        __m256 _out3 = _mm256_mul_ps(_mm256_loadu_ps(out3 + d), _mm256_set1_ps(alpha3));

        for (int j = 0; j < size; j++)
        {
            __m256 _v = _mm256_loadu_ps(value + (size_t)j * value_dim + d);
            _out0 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(score0[j]), _out0);
            _out1 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(score1[j]), _out1);
            _out2 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(score2[j]), _out2);
            _out3 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(score3[j]), _out3);
        }

        _mm256_storeu_ps(out0 + d, _out0);
        _mm256_storeu_ps(out1 + d, _out1);
        _mm256_storeu_ps(out2 + d, _out2);
        _mm256_storeu_ps(out3 + d, _out3);
    }
#endif // __AVX__
#if !__AVX__
    for (; d + 7 < value_dim; d += 8)
    {
        __m128 _out00 = _mm_mul_ps(_mm_loadu_ps(out0 + d), _mm_set1_ps(alpha0));
        __m128 _out01 = _mm_mul_ps(_mm_loadu_ps(out0 + d + 4), _mm_set1_ps(alpha0));
        __m128 _out10 = _mm_mul_ps(_mm_loadu_ps(out1 + d), _mm_set1_ps(alpha1));
        __m128 _out11 = _mm_mul_ps(_mm_loadu_ps(out1 + d + 4), _mm_set1_ps(alpha1));
        __m128 _out20 = _mm_mul_ps(_mm_loadu_ps(out2 + d), _mm_set1_ps(alpha2));
        __m128 _out21 = _mm_mul_ps(_mm_loadu_ps(out2 + d + 4), _mm_set1_ps(alpha2));
        __m128 _out30 = _mm_mul_ps(_mm_loadu_ps(out3 + d), _mm_set1_ps(alpha3));
        __m128 _out31 = _mm_mul_ps(_mm_loadu_ps(out3 + d + 4), _mm_set1_ps(alpha3));

        for (int j = 0; j < size; j++)
        {
            const float* vptr = value + (size_t)j * value_dim + d;
            __m128 _v0 = _mm_loadu_ps(vptr);
            __m128 _v1 = _mm_loadu_ps(vptr + 4);
            __m128 _p = _mm_set1_ps(score0[j]);
            _out00 = _mm_comp_fmadd_ps(_v0, _p, _out00);
            _out01 = _mm_comp_fmadd_ps(_v1, _p, _out01);
            _p = _mm_set1_ps(score1[j]);
            _out10 = _mm_comp_fmadd_ps(_v0, _p, _out10);
            _out11 = _mm_comp_fmadd_ps(_v1, _p, _out11);
            _p = _mm_set1_ps(score2[j]);
            _out20 = _mm_comp_fmadd_ps(_v0, _p, _out20);
            _out21 = _mm_comp_fmadd_ps(_v1, _p, _out21);
            _p = _mm_set1_ps(score3[j]);
            _out30 = _mm_comp_fmadd_ps(_v0, _p, _out30);
            _out31 = _mm_comp_fmadd_ps(_v1, _p, _out31);
        }

        _mm_storeu_ps(out0 + d, _out00);
        _mm_storeu_ps(out0 + d + 4, _out01);
        _mm_storeu_ps(out1 + d, _out10);
        _mm_storeu_ps(out1 + d + 4, _out11);
        _mm_storeu_ps(out2 + d, _out20);
        _mm_storeu_ps(out2 + d + 4, _out21);
        _mm_storeu_ps(out3 + d, _out30);
        _mm_storeu_ps(out3 + d + 4, _out31);
    }
#endif // !__AVX__
    for (; d + 3 < value_dim; d += 4)
    {
        __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(out0 + d), _mm_set1_ps(alpha0));
        __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(out1 + d), _mm_set1_ps(alpha1));
        __m128 _out2 = _mm_mul_ps(_mm_loadu_ps(out2 + d), _mm_set1_ps(alpha2));
        __m128 _out3 = _mm_mul_ps(_mm_loadu_ps(out3 + d), _mm_set1_ps(alpha3));

        for (int j = 0; j < size; j++)
        {
            __m128 _v = _mm_loadu_ps(value + (size_t)j * value_dim + d);
            _out0 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(score0[j]), _out0);
            _out1 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(score1[j]), _out1);
            _out2 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(score2[j]), _out2);
            _out3 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(score3[j]), _out3);
        }

        _mm_storeu_ps(out0 + d, _out0);
        _mm_storeu_ps(out1 + d, _out1);
        _mm_storeu_ps(out2 + d, _out2);
        _mm_storeu_ps(out3 + d, _out3);
    }
#endif // __SSE2__
    for (; d < value_dim; d++)
    {
        float sum0 = out0[d] * alpha0;
        float sum1 = out1[d] * alpha1;
        float sum2 = out2[d] * alpha2;
        float sum3 = out3[d] * alpha3;

        for (int j = 0; j < size; j++)
        {
            const float v = value[(size_t)j * value_dim + d];
            sum0 += score0[j] * v;
            sum1 += score1[j] * v;
            sum2 += score2[j] * v;
            sum3 += score3[j] * v;
        }

        out0[d] = sum0;
        out1[d] = sum1;
        out2[d] = sum2;
        out3[d] = sum3;
    }
}

static void sdpa_decode_tile4_fp32(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q0, int g, int n_begin, int n_end, Mat& workspace, Mat& state)
{
    const int head_dim = query.w;
    const int value_dim = value.w;
    const int block_n = 256;

    const float* query0 = query.channel(q0);
    const float* query1 = query.channel(q0 + 1);
    const float* query2 = query.channel(q0 + 2);
    const float* query3 = query.channel(q0 + 3);
    const float* mask0 = 0;
    const float* mask1 = 0;
    const float* mask2 = 0;
    const float* mask3 = 0;
    if (!attn_mask_blob.empty())
    {
        if (attn_mask_blob.dims == 3)
        {
            mask0 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q0 : 0);
            mask1 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q0 + 1 : 0);
            mask2 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q0 + 2 : 0);
            mask3 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q0 + 3 : 0);
        }
        else
        {
            mask0 = attn_mask_blob;
            mask1 = attn_mask_blob;
            mask2 = attn_mask_blob;
            mask3 = attn_mask_blob;
        }
    }

    float* workspace_ptr = workspace;
    float* score0 = workspace_ptr;
    float* score1 = workspace_ptr + block_n;
    float* score2 = workspace_ptr + block_n * 2;
    float* score3 = workspace_ptr + block_n * 3;
    float* out0 = workspace_ptr + block_n * 4;
    float* out1 = out0 + value_dim;
    float* out2 = out1 + value_dim;
    float* out3 = out2 + value_dim;
    memset(out0, 0, value_dim * sizeof(float));
    memset(out1, 0, value_dim * sizeof(float));
    memset(out2, 0, value_dim * sizeof(float));
    memset(out3, 0, value_dim * sizeof(float));

    const Mat key_head = key.channel(g);
    const Mat value_head = value.channel(g);

    float m0 = -FLT_MAX;
    float m1 = -FLT_MAX;
    float m2 = -FLT_MAX;
    float m3 = -FLT_MAX;
    float l0 = 0.f;
    float l1 = 0.f;
    float l2 = 0.f;
    float l3 = 0.f;

    for (int n = n_begin; n < n_end; n += block_n)
    {
        const int max_jj = std::min(n_end - n, block_n);
        float block_max0 = -FLT_MAX;
        float block_max1 = -FLT_MAX;
        float block_max2 = -FLT_MAX;
        float block_max3 = -FLT_MAX;
        for (int j = 0; j < max_jj; j++)
        {
            float sum0;
            float sum1;
            float sum2;
            float sum3;
            sdpa_dot_product_tile4_fp32(query0, query1, query2, query3, key_head.row(n + j), head_dim, sum0, sum1, sum2, sum3);

            float s0 = sum0 * scale;
            float s1 = sum1 * scale;
            float s2 = sum2 * scale;
            float s3 = sum3 * scale;
            if (mask0)
            {
                s0 += mask0[n + j];
                s1 += mask1[n + j];
                s2 += mask2[n + j];
                s3 += mask3[n + j];
            }
            score0[j] = s0;
            score1[j] = s1;
            score2[j] = s2;
            score3[j] = s3;
            block_max0 = std::max(block_max0, s0);
            block_max1 = std::max(block_max1, s1);
            block_max2 = std::max(block_max2, s2);
            block_max3 = std::max(block_max3, s3);
        }

        const float m_new0 = std::max(m0, block_max0);
        const float m_new1 = std::max(m1, block_max1);
        const float m_new2 = std::max(m2, block_max2);
        const float m_new3 = std::max(m3, block_max3);
        const float alpha0 = l0 == 0.f ? 0.f : expf(m0 - m_new0);
        const float alpha1 = l1 == 0.f ? 0.f : expf(m1 - m_new1);
        const float alpha2 = l2 == 0.f ? 0.f : expf(m2 - m_new2);
        const float alpha3 = l3 == 0.f ? 0.f : expf(m3 - m_new3);
        l0 = l0 * alpha0 + sdpa_exp_submax_fp32(score0, max_jj, m_new0);
        l1 = l1 * alpha1 + sdpa_exp_submax_fp32(score1, max_jj, m_new1);
        l2 = l2 * alpha2 + sdpa_exp_submax_fp32(score2, max_jj, m_new2);
        l3 = l3 * alpha3 + sdpa_exp_submax_fp32(score3, max_jj, m_new3);
        m0 = m_new0;
        m1 = m_new1;
        m2 = m_new2;
        m3 = m_new3;

        sdpa_pv_tile4_fp32(out0, out1, out2, out3, value_head.row(n), score0, score1, score2, score3, alpha0, alpha1, alpha2, alpha3, max_jj, value_dim);
    }

    if (!state.empty())
    {
        float* state0 = state;
        float* state1 = state0 + value_dim + 2;
        float* state2 = state1 + value_dim + 2;
        float* state3 = state2 + value_dim + 2;
        state0[0] = m0;
        state0[1] = l0;
        state1[0] = m1;
        state1[1] = l1;
        state2[0] = m2;
        state2[1] = l2;
        state3[0] = m3;
        state3[1] = l3;
        memcpy(state0 + 2, out0, value_dim * sizeof(float));
        memcpy(state1 + 2, out1, value_dim * sizeof(float));
        memcpy(state2 + 2, out2, value_dim * sizeof(float));
        memcpy(state3 + 2, out3, value_dim * sizeof(float));
    }
    else
    {
        float* output0 = top_blob.channel(q0);
        float* output1 = top_blob.channel(q0 + 1);
        float* output2 = top_blob.channel(q0 + 2);
        float* output3 = top_blob.channel(q0 + 3);
        memcpy(output0, out0, value_dim * sizeof(float));
        memcpy(output1, out1, value_dim * sizeof(float));
        memcpy(output2, out2, value_dim * sizeof(float));
        memcpy(output3, out3, value_dim * sizeof(float));
        if (l0 != 0.f)
            sdpa_normalize_fp32(output0, l0, value_dim);
        if (l1 != 0.f)
            sdpa_normalize_fp32(output1, l1, value_dim);
        if (l2 != 0.f)
            sdpa_normalize_fp32(output2, l2, value_dim);
        if (l3 != 0.f)
            sdpa_normalize_fp32(output3, l3, value_dim);
    }
}

static float sdpa_dot_product_tile1_fp32(const float* query, const float* key, int size)
{
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _sum_avx512 = _mm512_setzero_ps();
#endif // __AVX512F__
    __m256 _sum_avx = _mm256_setzero_ps();
#endif // __AVX__
    __m128 _sum = _mm_setzero_ps();
#endif // __SSE2__
    float sum = 0.f;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
        _sum_avx512 = _mm512_fmadd_ps(_mm512_loadu_ps(query + i), _mm512_loadu_ps(key + i), _sum_avx512);
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
        _sum_avx = _mm256_comp_fmadd_ps(_mm256_loadu_ps(query + i), _mm256_loadu_ps(key + i), _sum_avx);
#endif // __AVX__
    for (; i + 3 < size; i += 4)
        _sum = _mm_comp_fmadd_ps(_mm_loadu_ps(query + i), _mm_loadu_ps(key + i), _sum);
#endif // __SSE2__
    for (; i < size; i++)
        sum += query[i] * key[i];

#if __SSE2__
#if __AVX__
#if __AVX512F__
    sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
    sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
    sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__

    return sum;
}

static void sdpa_pv_tile1_fp32(float* out, const float* value, const float* score, float alpha, int size, int value_dim)
{
    int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; d + 63 < value_dim; d += 64)
    {
        __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(out + d), _mm512_set1_ps(alpha));
        __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(out + d + 16), _mm512_set1_ps(alpha));
        __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(out + d + 32), _mm512_set1_ps(alpha));
        __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(out + d + 48), _mm512_set1_ps(alpha));
        for (int j = 0; j < size; j++)
        {
            const float* vptr = value + (size_t)j * value_dim + d;
            __m512 _p = _mm512_set1_ps(score[j]);
            _out0 = _mm512_fmadd_ps(_mm512_loadu_ps(vptr), _p, _out0);
            _out1 = _mm512_fmadd_ps(_mm512_loadu_ps(vptr + 16), _p, _out1);
            _out2 = _mm512_fmadd_ps(_mm512_loadu_ps(vptr + 32), _p, _out2);
            _out3 = _mm512_fmadd_ps(_mm512_loadu_ps(vptr + 48), _p, _out3);
        }
        _mm512_storeu_ps(out + d, _out0);
        _mm512_storeu_ps(out + d + 16, _out1);
        _mm512_storeu_ps(out + d + 32, _out2);
        _mm512_storeu_ps(out + d + 48, _out3);
    }

    for (; d + 15 < value_dim; d += 16)
    {
        __m512 _out = _mm512_mul_ps(_mm512_loadu_ps(out + d), _mm512_set1_ps(alpha));
        for (int j = 0; j < size; j++)
            _out = _mm512_fmadd_ps(_mm512_loadu_ps(value + (size_t)j * value_dim + d), _mm512_set1_ps(score[j]), _out);
        _mm512_storeu_ps(out + d, _out);
    }
#endif // __AVX512F__
#if !__AVX512F__
    for (; d + 15 < value_dim; d += 16)
    {
        __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(out + d), _mm256_set1_ps(alpha));
        __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(out + d + 8), _mm256_set1_ps(alpha));
        for (int j = 0; j < size; j++)
        {
            const float* vptr = value + (size_t)j * value_dim + d;
            __m256 _p = _mm256_set1_ps(score[j]);
            _out0 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(vptr), _p, _out0);
            _out1 = _mm256_comp_fmadd_ps(_mm256_loadu_ps(vptr + 8), _p, _out1);
        }
        _mm256_storeu_ps(out + d, _out0);
        _mm256_storeu_ps(out + d + 8, _out1);
    }
#endif // !__AVX512F__

    for (; d + 7 < value_dim; d += 8)
    {
        __m256 _out = _mm256_mul_ps(_mm256_loadu_ps(out + d), _mm256_set1_ps(alpha));
        for (int j = 0; j < size; j++)
            _out = _mm256_comp_fmadd_ps(_mm256_loadu_ps(value + (size_t)j * value_dim + d), _mm256_set1_ps(score[j]), _out);
        _mm256_storeu_ps(out + d, _out);
    }
#endif // __AVX__
#if !__AVX__
    for (; d + 7 < value_dim; d += 8)
    {
        __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(out + d), _mm_set1_ps(alpha));
        __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(out + d + 4), _mm_set1_ps(alpha));
        for (int j = 0; j < size; j++)
        {
            const float* vptr = value + (size_t)j * value_dim + d;
            __m128 _p = _mm_set1_ps(score[j]);
            _out0 = _mm_comp_fmadd_ps(_mm_loadu_ps(vptr), _p, _out0);
            _out1 = _mm_comp_fmadd_ps(_mm_loadu_ps(vptr + 4), _p, _out1);
        }
        _mm_storeu_ps(out + d, _out0);
        _mm_storeu_ps(out + d + 4, _out1);
    }
#endif // !__AVX__
    for (; d + 3 < value_dim; d += 4)
    {
        __m128 _out = _mm_mul_ps(_mm_loadu_ps(out + d), _mm_set1_ps(alpha));
        for (int j = 0; j < size; j++)
            _out = _mm_comp_fmadd_ps(_mm_loadu_ps(value + (size_t)j * value_dim + d), _mm_set1_ps(score[j]), _out);
        _mm_storeu_ps(out + d, _out);
    }
#endif // __SSE2__
    for (; d < value_dim; d++)
    {
        float sum = out[d] * alpha;
        for (int j = 0; j < size; j++)
            sum += score[j] * value[(size_t)j * value_dim + d];
        out[d] = sum;
    }
}

static void sdpa_decode_tile1_fp32(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q, int g, int n_begin, int n_end, Mat& workspace, Mat& state)
{
    const int head_dim = query.w;
    const int value_dim = value.w;
    const int block_n = 256;

    const float* query_ptr = query.channel(q);
    const float* mask = 0;
    if (!attn_mask_blob.empty())
    {
        if (attn_mask_blob.dims == 3)
            mask = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q : 0);
        else
            mask = attn_mask_blob;
    }

    float* workspace_ptr = workspace;
    float* score = workspace_ptr;
    float* out = workspace_ptr + block_n * 4;
    memset(out, 0, value_dim * sizeof(float));

    const Mat key_head = key.channel(g);
    const Mat value_head = value.channel(g);

    float m = -FLT_MAX;
    float l = 0.f;

    for (int n = n_begin; n < n_end; n += block_n)
    {
        const int max_jj = std::min(n_end - n, block_n);
        float block_max = -FLT_MAX;
        for (int j = 0; j < max_jj; j++)
        {
            float s = sdpa_dot_product_tile1_fp32(query_ptr, key_head.row(n + j), head_dim) * scale;
            if (mask)
                s += mask[n + j];
            score[j] = s;
            block_max = std::max(block_max, s);
        }

        const float m_new = std::max(m, block_max);
        const float alpha = l == 0.f ? 0.f : expf(m - m_new);
        l = l * alpha + sdpa_exp_submax_fp32(score, max_jj, m_new);
        m = m_new;

        sdpa_pv_tile1_fp32(out, value_head.row(n), score, alpha, max_jj, value_dim);
    }

    if (!state.empty())
    {
        float* state_ptr = state;
        state_ptr[0] = m;
        state_ptr[1] = l;
        memcpy(state_ptr + 2, out, value_dim * sizeof(float));
    }
    else
    {
        float* output = top_blob.channel(q);
        memcpy(output, out, value_dim * sizeof(float));
        if (l != 0.f)
            sdpa_normalize_fp32(output, l, value_dim);
    }
}

static void sdpa_scale_add_fp32(float* out, const float* ptr, float scale, int size)
{
    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _scale_avx512 = _mm512_set1_ps(scale);
    for (; i + 15 < size; i += 16)
        _mm512_storeu_ps(out + i, _mm512_fmadd_ps(_mm512_loadu_ps(ptr + i), _scale_avx512, _mm512_loadu_ps(out + i)));
#endif // __AVX512F__
    __m256 _scale_avx = _mm256_set1_ps(scale);
    for (; i + 7 < size; i += 8)
        _mm256_storeu_ps(out + i, _mm256_comp_fmadd_ps(_mm256_loadu_ps(ptr + i), _scale_avx, _mm256_loadu_ps(out + i)));
#endif // __AVX__
    __m128 _scale = _mm_set1_ps(scale);
    for (; i + 3 < size; i += 4)
        _mm_storeu_ps(out + i, _mm_comp_fmadd_ps(_mm_loadu_ps(ptr + i), _scale, _mm_loadu_ps(out + i)));
#endif // __SSE2__
    for (; i < size; i++)
        out[i] += ptr[i] * scale;
}

static int sdpa_decode_fp32(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
{
    const int num_query_heads = query.c;
    const int num_kv_heads = key.c;
    const int key_seqlen = key.h;
    const int value_dim = value.w;
    const int num_query_heads_per_kv_head = num_query_heads / num_kv_heads;
    const int block_q = 4;
    const int num_qblocks = (num_query_heads_per_kv_head + block_q - 1) / block_q;
    const int num_tasks = num_kv_heads * num_qblocks;

    const int num_threads = std::max(opt.num_threads, 1);
    int num_kv_chunks = 1;
    if (num_tasks < num_threads && key_seqlen >= 512)
    {
        num_kv_chunks = std::min((num_threads + num_tasks - 1) / num_tasks, key_seqlen / 256);
        num_kv_chunks = std::max(num_kv_chunks, 1);
    }

    Mat workspace(4 * (256 + value_dim), 1, num_threads, 4u, opt.workspace_allocator);
    if (workspace.empty())
        return -100;

    Mat partials;
    if (num_kv_chunks > 1)
    {
        partials.create((value_dim + 2) * block_q, 1, num_tasks * num_kv_chunks, 4u, opt.workspace_allocator);
        if (partials.empty())
            return -100;
    }

#pragma omp parallel for num_threads(opt.num_threads)
    for (int ti = 0; ti < num_tasks * num_kv_chunks; ti++)
    {
        const int task_id = ti / num_kv_chunks;
        const int chunk_id = ti % num_kv_chunks;
        const int g = task_id / num_qblocks;
        const int qblock_id = task_id % num_qblocks;
        const int q0 = g * num_query_heads_per_kv_head + qblock_id * block_q;
        const int max_qq = std::min(num_query_heads_per_kv_head - qblock_id * block_q, block_q);
        const int n_begin = chunk_id * key_seqlen / num_kv_chunks;
        const int n_end = (chunk_id + 1) * key_seqlen / num_kv_chunks;

        Mat workspace_tile = workspace.channel(get_omp_thread_num());
        Mat state;
        if (num_kv_chunks > 1)
            state = partials.channel(ti);
        if (max_qq == 4)
        {
            sdpa_decode_tile4_fp32(query, key, value, attn_mask_blob, top_blob, scale, q0, g, n_begin, n_end, workspace_tile, state);
        }
        else
        {
            for (int qq = 0; qq < max_qq; qq++)
            {
                Mat state_q;
                if (!state.empty())
                    state_q = state.range(qq * (value_dim + 2), value_dim + 2);
                sdpa_decode_tile1_fp32(query, key, value, attn_mask_blob, top_blob, scale, q0 + qq, g, n_begin, n_end, workspace_tile, state_q);
            }
        }
    }

    if (num_kv_chunks > 1)
    {
#pragma omp parallel for num_threads(opt.num_threads)
        for (int task_id = 0; task_id < num_tasks; task_id++)
        {
            const int g = task_id / num_qblocks;
            const int qblock_id = task_id % num_qblocks;
            const int q0 = g * num_query_heads_per_kv_head + qblock_id * block_q;
            const int max_qq = std::min(num_query_heads_per_kv_head - qblock_id * block_q, block_q);
            for (int q = 0; q < max_qq; q++)
            {
                float m = -FLT_MAX;
                for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
                {
                    const float* state_q = partials.channel(task_id * num_kv_chunks + chunk_id);
                    state_q += q * (value_dim + 2);
                    m = std::max(m, state_q[0]);
                }

                float* outptr = top_blob.channel(q0 + q);
                memset(outptr, 0, value_dim * sizeof(float));
                float l = 0.f;
                for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
                {
                    const float* state_q = partials.channel(task_id * num_kv_chunks + chunk_id);
                    state_q += q * (value_dim + 2);
                    float partial_scale = state_q[1] == 0.f ? 0.f : expf(state_q[0] - m);
                    l += state_q[1] * partial_scale;
                    sdpa_scale_add_fp32(outptr, state_q + 2, partial_scale, value_dim);
                }
                if (l != 0.f)
                    sdpa_normalize_fp32(outptr, l, value_dim);
            }
        }
    }

    return 0;
}

#if NCNN_BF16
static void sdpa_dot_product_tile4_bf16s(const unsigned short* query0, const unsigned short* query1, const unsigned short* query2, const unsigned short* query3, const unsigned short* key, int size, float& sum0, float& sum1, float& sum2, float& sum3)
{
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _sum0_avx512 = _mm512_setzero_ps();
    __m512 _sum1_avx512 = _mm512_setzero_ps();
    __m512 _sum2_avx512 = _mm512_setzero_ps();
    __m512 _sum3_avx512 = _mm512_setzero_ps();
#endif // __AVX512F__
    __m256 _sum0_avx = _mm256_setzero_ps();
    __m256 _sum1_avx = _mm256_setzero_ps();
    __m256 _sum2_avx = _mm256_setzero_ps();
    __m256 _sum3_avx = _mm256_setzero_ps();
#endif // __AVX__
    __m128 _sum0 = _mm_setzero_ps();
    __m128 _sum1 = _mm_setzero_ps();
    __m128 _sum2 = _mm_setzero_ps();
    __m128 _sum3 = _mm_setzero_ps();
#endif // __SSE2__
    sum0 = 0.f;
    sum1 = 0.f;
    sum2 = 0.f;
    sum3 = 0.f;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _k = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(key + i)));
        _sum0_avx512 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(query0 + i))), _k, _sum0_avx512);
        _sum1_avx512 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(query1 + i))), _k, _sum1_avx512);
        _sum2_avx512 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(query2 + i))), _k, _sum2_avx512);
        _sum3_avx512 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(query3 + i))), _k, _sum3_avx512);
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _k = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(key + i)));
        _sum0_avx = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(query0 + i))), _k, _sum0_avx);
        _sum1_avx = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(query1 + i))), _k, _sum1_avx);
        _sum2_avx = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(query2 + i))), _k, _sum2_avx);
        _sum3_avx = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(query3 + i))), _k, _sum3_avx);
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _k = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(key + i)));
        _sum0 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(query0 + i))), _k, _sum0);
        _sum1 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(query1 + i))), _k, _sum1);
        _sum2 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(query2 + i))), _k, _sum2);
        _sum3 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(query3 + i))), _k, _sum3);
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        const float k = bfloat16_to_float32(key[i]);
        sum0 += bfloat16_to_float32(query0[i]) * k;
        sum1 += bfloat16_to_float32(query1[i]) * k;
        sum2 += bfloat16_to_float32(query2[i]) * k;
        sum3 += bfloat16_to_float32(query3[i]) * k;
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    sum0 += _mm512_comp_reduce_add_ps(_sum0_avx512);
    sum1 += _mm512_comp_reduce_add_ps(_sum1_avx512);
    sum2 += _mm512_comp_reduce_add_ps(_sum2_avx512);
    sum3 += _mm512_comp_reduce_add_ps(_sum3_avx512);
#endif // __AVX512F__
    sum0 += _mm256_reduce_add_ps(_sum0_avx);
    sum1 += _mm256_reduce_add_ps(_sum1_avx);
    sum2 += _mm256_reduce_add_ps(_sum2_avx);
    sum3 += _mm256_reduce_add_ps(_sum3_avx);
#endif // __AVX__
    sum0 += _mm_reduce_add_ps(_sum0);
    sum1 += _mm_reduce_add_ps(_sum1);
    sum2 += _mm_reduce_add_ps(_sum2);
    sum3 += _mm_reduce_add_ps(_sum3);
#endif // __SSE2__
}

static void sdpa_pv_tile4_bf16s(float* out0, float* out1, float* out2, float* out3, const unsigned short* value, const float* score0, const float* score1, const float* score2, const float* score3, float alpha0, float alpha1, float alpha2, float alpha3, int size, int value_dim)
{
    int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; d + 63 < value_dim; d += 64)
    {
        __m512 _out00 = _mm512_mul_ps(_mm512_loadu_ps(out0 + d), _mm512_set1_ps(alpha0));
        __m512 _out01 = _mm512_mul_ps(_mm512_loadu_ps(out0 + d + 16), _mm512_set1_ps(alpha0));
        __m512 _out02 = _mm512_mul_ps(_mm512_loadu_ps(out0 + d + 32), _mm512_set1_ps(alpha0));
        __m512 _out03 = _mm512_mul_ps(_mm512_loadu_ps(out0 + d + 48), _mm512_set1_ps(alpha0));
        __m512 _out10 = _mm512_mul_ps(_mm512_loadu_ps(out1 + d), _mm512_set1_ps(alpha1));
        __m512 _out11 = _mm512_mul_ps(_mm512_loadu_ps(out1 + d + 16), _mm512_set1_ps(alpha1));
        __m512 _out12 = _mm512_mul_ps(_mm512_loadu_ps(out1 + d + 32), _mm512_set1_ps(alpha1));
        __m512 _out13 = _mm512_mul_ps(_mm512_loadu_ps(out1 + d + 48), _mm512_set1_ps(alpha1));
        __m512 _out20 = _mm512_mul_ps(_mm512_loadu_ps(out2 + d), _mm512_set1_ps(alpha2));
        __m512 _out21 = _mm512_mul_ps(_mm512_loadu_ps(out2 + d + 16), _mm512_set1_ps(alpha2));
        __m512 _out22 = _mm512_mul_ps(_mm512_loadu_ps(out2 + d + 32), _mm512_set1_ps(alpha2));
        __m512 _out23 = _mm512_mul_ps(_mm512_loadu_ps(out2 + d + 48), _mm512_set1_ps(alpha2));
        __m512 _out30 = _mm512_mul_ps(_mm512_loadu_ps(out3 + d), _mm512_set1_ps(alpha3));
        __m512 _out31 = _mm512_mul_ps(_mm512_loadu_ps(out3 + d + 16), _mm512_set1_ps(alpha3));
        __m512 _out32 = _mm512_mul_ps(_mm512_loadu_ps(out3 + d + 32), _mm512_set1_ps(alpha3));
        __m512 _out33 = _mm512_mul_ps(_mm512_loadu_ps(out3 + d + 48), _mm512_set1_ps(alpha3));

        for (int j = 0; j < size; j++)
        {
            const unsigned short* vptr = value + (size_t)j * value_dim + d;
            __m512 _v0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)vptr));
            __m512 _v1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(vptr + 16)));
            __m512 _v2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(vptr + 32)));
            __m512 _v3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(vptr + 48)));
            __m512 _p = _mm512_set1_ps(score0[j]);
            _out00 = _mm512_fmadd_ps(_v0, _p, _out00);
            _out01 = _mm512_fmadd_ps(_v1, _p, _out01);
            _out02 = _mm512_fmadd_ps(_v2, _p, _out02);
            _out03 = _mm512_fmadd_ps(_v3, _p, _out03);
            _p = _mm512_set1_ps(score1[j]);
            _out10 = _mm512_fmadd_ps(_v0, _p, _out10);
            _out11 = _mm512_fmadd_ps(_v1, _p, _out11);
            _out12 = _mm512_fmadd_ps(_v2, _p, _out12);
            _out13 = _mm512_fmadd_ps(_v3, _p, _out13);
            _p = _mm512_set1_ps(score2[j]);
            _out20 = _mm512_fmadd_ps(_v0, _p, _out20);
            _out21 = _mm512_fmadd_ps(_v1, _p, _out21);
            _out22 = _mm512_fmadd_ps(_v2, _p, _out22);
            _out23 = _mm512_fmadd_ps(_v3, _p, _out23);
            _p = _mm512_set1_ps(score3[j]);
            _out30 = _mm512_fmadd_ps(_v0, _p, _out30);
            _out31 = _mm512_fmadd_ps(_v1, _p, _out31);
            _out32 = _mm512_fmadd_ps(_v2, _p, _out32);
            _out33 = _mm512_fmadd_ps(_v3, _p, _out33);
        }

        _mm512_storeu_ps(out0 + d, _out00);
        _mm512_storeu_ps(out0 + d + 16, _out01);
        _mm512_storeu_ps(out0 + d + 32, _out02);
        _mm512_storeu_ps(out0 + d + 48, _out03);
        _mm512_storeu_ps(out1 + d, _out10);
        _mm512_storeu_ps(out1 + d + 16, _out11);
        _mm512_storeu_ps(out1 + d + 32, _out12);
        _mm512_storeu_ps(out1 + d + 48, _out13);
        _mm512_storeu_ps(out2 + d, _out20);
        _mm512_storeu_ps(out2 + d + 16, _out21);
        _mm512_storeu_ps(out2 + d + 32, _out22);
        _mm512_storeu_ps(out2 + d + 48, _out23);
        _mm512_storeu_ps(out3 + d, _out30);
        _mm512_storeu_ps(out3 + d + 16, _out31);
        _mm512_storeu_ps(out3 + d + 32, _out32);
        _mm512_storeu_ps(out3 + d + 48, _out33);
    }

    for (; d + 15 < value_dim; d += 16)
    {
        __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(out0 + d), _mm512_set1_ps(alpha0));
        __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(out1 + d), _mm512_set1_ps(alpha1));
        __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(out2 + d), _mm512_set1_ps(alpha2));
        __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(out3 + d), _mm512_set1_ps(alpha3));

        for (int j = 0; j < size; j++)
        {
            const unsigned short* vptr = value + (size_t)j * value_dim + d;
            __m512 _v = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)vptr));
            _out0 = _mm512_fmadd_ps(_v, _mm512_set1_ps(score0[j]), _out0);
            _out1 = _mm512_fmadd_ps(_v, _mm512_set1_ps(score1[j]), _out1);
            _out2 = _mm512_fmadd_ps(_v, _mm512_set1_ps(score2[j]), _out2);
            _out3 = _mm512_fmadd_ps(_v, _mm512_set1_ps(score3[j]), _out3);
        }

        _mm512_storeu_ps(out0 + d, _out0);
        _mm512_storeu_ps(out1 + d, _out1);
        _mm512_storeu_ps(out2 + d, _out2);
        _mm512_storeu_ps(out3 + d, _out3);
    }
#endif // __AVX512F__
#if !__AVX512F__
    for (; d + 15 < value_dim; d += 16)
    {
        __m256 _out00 = _mm256_mul_ps(_mm256_loadu_ps(out0 + d), _mm256_set1_ps(alpha0));
        __m256 _out01 = _mm256_mul_ps(_mm256_loadu_ps(out0 + d + 8), _mm256_set1_ps(alpha0));
        __m256 _out10 = _mm256_mul_ps(_mm256_loadu_ps(out1 + d), _mm256_set1_ps(alpha1));
        __m256 _out11 = _mm256_mul_ps(_mm256_loadu_ps(out1 + d + 8), _mm256_set1_ps(alpha1));
        __m256 _out20 = _mm256_mul_ps(_mm256_loadu_ps(out2 + d), _mm256_set1_ps(alpha2));
        __m256 _out21 = _mm256_mul_ps(_mm256_loadu_ps(out2 + d + 8), _mm256_set1_ps(alpha2));
        __m256 _out30 = _mm256_mul_ps(_mm256_loadu_ps(out3 + d), _mm256_set1_ps(alpha3));
        __m256 _out31 = _mm256_mul_ps(_mm256_loadu_ps(out3 + d + 8), _mm256_set1_ps(alpha3));

        for (int j = 0; j < size; j++)
        {
            const unsigned short* vptr = value + (size_t)j * value_dim + d;
            __m256 _v0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)vptr));
            __m256 _v1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(vptr + 8)));
            __m256 _p = _mm256_set1_ps(score0[j]);
            _out00 = _mm256_comp_fmadd_ps(_v0, _p, _out00);
            _out01 = _mm256_comp_fmadd_ps(_v1, _p, _out01);
            _p = _mm256_set1_ps(score1[j]);
            _out10 = _mm256_comp_fmadd_ps(_v0, _p, _out10);
            _out11 = _mm256_comp_fmadd_ps(_v1, _p, _out11);
            _p = _mm256_set1_ps(score2[j]);
            _out20 = _mm256_comp_fmadd_ps(_v0, _p, _out20);
            _out21 = _mm256_comp_fmadd_ps(_v1, _p, _out21);
            _p = _mm256_set1_ps(score3[j]);
            _out30 = _mm256_comp_fmadd_ps(_v0, _p, _out30);
            _out31 = _mm256_comp_fmadd_ps(_v1, _p, _out31);
        }

        _mm256_storeu_ps(out0 + d, _out00);
        _mm256_storeu_ps(out0 + d + 8, _out01);
        _mm256_storeu_ps(out1 + d, _out10);
        _mm256_storeu_ps(out1 + d + 8, _out11);
        _mm256_storeu_ps(out2 + d, _out20);
        _mm256_storeu_ps(out2 + d + 8, _out21);
        _mm256_storeu_ps(out3 + d, _out30);
        _mm256_storeu_ps(out3 + d + 8, _out31);
    }
#endif // !__AVX512F__

    for (; d + 7 < value_dim; d += 8)
    {
        __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(out0 + d), _mm256_set1_ps(alpha0));
        __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(out1 + d), _mm256_set1_ps(alpha1));
        __m256 _out2 = _mm256_mul_ps(_mm256_loadu_ps(out2 + d), _mm256_set1_ps(alpha2));
        __m256 _out3 = _mm256_mul_ps(_mm256_loadu_ps(out3 + d), _mm256_set1_ps(alpha3));

        for (int j = 0; j < size; j++)
        {
            const unsigned short* vptr = value + (size_t)j * value_dim + d;
            __m256 _v = bfloat2float_avx(_mm_loadu_si128((const __m128i*)vptr));
            _out0 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(score0[j]), _out0);
            _out1 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(score1[j]), _out1);
            _out2 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(score2[j]), _out2);
            _out3 = _mm256_comp_fmadd_ps(_v, _mm256_set1_ps(score3[j]), _out3);
        }

        _mm256_storeu_ps(out0 + d, _out0);
        _mm256_storeu_ps(out1 + d, _out1);
        _mm256_storeu_ps(out2 + d, _out2);
        _mm256_storeu_ps(out3 + d, _out3);
    }
#endif // __AVX__
#if !__AVX__
    for (; d + 7 < value_dim; d += 8)
    {
        __m128 _out00 = _mm_mul_ps(_mm_loadu_ps(out0 + d), _mm_set1_ps(alpha0));
        __m128 _out01 = _mm_mul_ps(_mm_loadu_ps(out0 + d + 4), _mm_set1_ps(alpha0));
        __m128 _out10 = _mm_mul_ps(_mm_loadu_ps(out1 + d), _mm_set1_ps(alpha1));
        __m128 _out11 = _mm_mul_ps(_mm_loadu_ps(out1 + d + 4), _mm_set1_ps(alpha1));
        __m128 _out20 = _mm_mul_ps(_mm_loadu_ps(out2 + d), _mm_set1_ps(alpha2));
        __m128 _out21 = _mm_mul_ps(_mm_loadu_ps(out2 + d + 4), _mm_set1_ps(alpha2));
        __m128 _out30 = _mm_mul_ps(_mm_loadu_ps(out3 + d), _mm_set1_ps(alpha3));
        __m128 _out31 = _mm_mul_ps(_mm_loadu_ps(out3 + d + 4), _mm_set1_ps(alpha3));

        for (int j = 0; j < size; j++)
        {
            const unsigned short* vptr = value + (size_t)j * value_dim + d;
            __m128 _v0 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)vptr));
            __m128 _v1 = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(vptr + 4)));
            __m128 _p = _mm_set1_ps(score0[j]);
            _out00 = _mm_comp_fmadd_ps(_v0, _p, _out00);
            _out01 = _mm_comp_fmadd_ps(_v1, _p, _out01);
            _p = _mm_set1_ps(score1[j]);
            _out10 = _mm_comp_fmadd_ps(_v0, _p, _out10);
            _out11 = _mm_comp_fmadd_ps(_v1, _p, _out11);
            _p = _mm_set1_ps(score2[j]);
            _out20 = _mm_comp_fmadd_ps(_v0, _p, _out20);
            _out21 = _mm_comp_fmadd_ps(_v1, _p, _out21);
            _p = _mm_set1_ps(score3[j]);
            _out30 = _mm_comp_fmadd_ps(_v0, _p, _out30);
            _out31 = _mm_comp_fmadd_ps(_v1, _p, _out31);
        }

        _mm_storeu_ps(out0 + d, _out00);
        _mm_storeu_ps(out0 + d + 4, _out01);
        _mm_storeu_ps(out1 + d, _out10);
        _mm_storeu_ps(out1 + d + 4, _out11);
        _mm_storeu_ps(out2 + d, _out20);
        _mm_storeu_ps(out2 + d + 4, _out21);
        _mm_storeu_ps(out3 + d, _out30);
        _mm_storeu_ps(out3 + d + 4, _out31);
    }
#endif // !__AVX__
    for (; d + 3 < value_dim; d += 4)
    {
        __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(out0 + d), _mm_set1_ps(alpha0));
        __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(out1 + d), _mm_set1_ps(alpha1));
        __m128 _out2 = _mm_mul_ps(_mm_loadu_ps(out2 + d), _mm_set1_ps(alpha2));
        __m128 _out3 = _mm_mul_ps(_mm_loadu_ps(out3 + d), _mm_set1_ps(alpha3));

        for (int j = 0; j < size; j++)
        {
            const unsigned short* vptr = value + (size_t)j * value_dim + d;
            __m128 _v = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)vptr));
            _out0 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(score0[j]), _out0);
            _out1 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(score1[j]), _out1);
            _out2 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(score2[j]), _out2);
            _out3 = _mm_comp_fmadd_ps(_v, _mm_set1_ps(score3[j]), _out3);
        }

        _mm_storeu_ps(out0 + d, _out0);
        _mm_storeu_ps(out1 + d, _out1);
        _mm_storeu_ps(out2 + d, _out2);
        _mm_storeu_ps(out3 + d, _out3);
    }
#endif // __SSE2__
    for (; d < value_dim; d++)
    {
        float sum0 = out0[d] * alpha0;
        float sum1 = out1[d] * alpha1;
        float sum2 = out2[d] * alpha2;
        float sum3 = out3[d] * alpha3;

        for (int j = 0; j < size; j++)
        {
            const float v = bfloat16_to_float32(value[(size_t)j * value_dim + d]);
            sum0 += score0[j] * v;
            sum1 += score1[j] * v;
            sum2 += score2[j] * v;
            sum3 += score3[j] * v;
        }

        out0[d] = sum0;
        out1[d] = sum1;
        out2[d] = sum2;
        out3[d] = sum3;
    }
}

static void sdpa_decode_tile4_bf16s(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q0, int g, int n_begin, int n_end, Mat& workspace, Mat& state)
{
    const int head_dim = query.w;
    const int value_dim = value.w;
    const int block_n = 256;
    const int mask_elembits = attn_mask_blob.empty() ? 0 : attn_mask_blob.elembits();

    const unsigned short* query0 = query.channel(q0);
    const unsigned short* query1 = query.channel(q0 + 1);
    const unsigned short* query2 = query.channel(q0 + 2);
    const unsigned short* query3 = query.channel(q0 + 3);
    Mat mask_head0;
    Mat mask_head1;
    Mat mask_head2;
    Mat mask_head3;
    if (!attn_mask_blob.empty())
    {
        if (attn_mask_blob.dims == 3)
        {
            mask_head0 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q0 : 0);
            mask_head1 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q0 + 1 : 0);
            mask_head2 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q0 + 2 : 0);
            mask_head3 = attn_mask_blob.channel(attn_mask_blob.c > 1 ? q0 + 3 : 0);
        }
        else
        {
            mask_head0 = attn_mask_blob;
            mask_head1 = attn_mask_blob;
            mask_head2 = attn_mask_blob;
            mask_head3 = attn_mask_blob;
        }
    }
    const float* mask0_fp32 = mask_elembits == 32 ? mask_head0 : 0;
    const float* mask1_fp32 = mask_elembits == 32 ? mask_head1 : 0;
    const float* mask2_fp32 = mask_elembits == 32 ? mask_head2 : 0;
    const float* mask3_fp32 = mask_elembits == 32 ? mask_head3 : 0;
    const unsigned short* mask0_bf16 = mask_elembits == 16 ? mask_head0 : 0;
    const unsigned short* mask1_bf16 = mask_elembits == 16 ? mask_head1 : 0;
    const unsigned short* mask2_bf16 = mask_elembits == 16 ? mask_head2 : 0;
    const unsigned short* mask3_bf16 = mask_elembits == 16 ? mask_head3 : 0;

    float* workspace_ptr = workspace;
    float* score0 = workspace_ptr;
    float* score1 = workspace_ptr + block_n;
    float* score2 = workspace_ptr + block_n * 2;
    float* score3 = workspace_ptr + block_n * 3;
    float* out0 = workspace_ptr + block_n * 4;
    float* out1 = out0 + value_dim;
    float* out2 = out1 + value_dim;
    float* out3 = out2 + value_dim;
    memset(out0, 0, value_dim * sizeof(float));
    memset(out1, 0, value_dim * sizeof(float));
    memset(out2, 0, value_dim * sizeof(float));
    memset(out3, 0, value_dim * sizeof(float));

    const Mat key_head = key.channel(g);
    const Mat value_head = value.channel(g);

    float m0 = -FLT_MAX;
    float m1 = -FLT_MAX;
    float m2 = -FLT_MAX;
    float m3 = -FLT_MAX;
    float l0 = 0.f;
    float l1 = 0.f;
    float l2 = 0.f;
    float l3 = 0.f;

    for (int n = n_begin; n < n_end; n += block_n)
    {
        const int max_jj = std::min(n_end - n, block_n);
        float block_max0 = -FLT_MAX;
        float block_max1 = -FLT_MAX;
        float block_max2 = -FLT_MAX;
        float block_max3 = -FLT_MAX;
        for (int j = 0; j < max_jj; j++)
        {
            const unsigned short* kptr = key_head.row<const unsigned short>(n + j);
            float sum0;
            float sum1;
            float sum2;
            float sum3;
            sdpa_dot_product_tile4_bf16s(query0, query1, query2, query3, kptr, head_dim, sum0, sum1, sum2, sum3);

            float s0 = sum0 * scale;
            float s1 = sum1 * scale;
            float s2 = sum2 * scale;
            float s3 = sum3 * scale;
            if (mask0_fp32)
            {
                s0 += mask0_fp32[n + j];
                s1 += mask1_fp32[n + j];
                s2 += mask2_fp32[n + j];
                s3 += mask3_fp32[n + j];
            }
            else if (mask0_bf16)
            {
                s0 += bfloat16_to_float32(mask0_bf16[n + j]);
                s1 += bfloat16_to_float32(mask1_bf16[n + j]);
                s2 += bfloat16_to_float32(mask2_bf16[n + j]);
                s3 += bfloat16_to_float32(mask3_bf16[n + j]);
            }
            score0[j] = s0;
            score1[j] = s1;
            score2[j] = s2;
            score3[j] = s3;
            block_max0 = std::max(block_max0, s0);
            block_max1 = std::max(block_max1, s1);
            block_max2 = std::max(block_max2, s2);
            block_max3 = std::max(block_max3, s3);
        }

        const float m_new0 = std::max(m0, block_max0);
        const float m_new1 = std::max(m1, block_max1);
        const float m_new2 = std::max(m2, block_max2);
        const float m_new3 = std::max(m3, block_max3);
        const float alpha0 = l0 == 0.f ? 0.f : expf(m0 - m_new0);
        const float alpha1 = l1 == 0.f ? 0.f : expf(m1 - m_new1);
        const float alpha2 = l2 == 0.f ? 0.f : expf(m2 - m_new2);
        const float alpha3 = l3 == 0.f ? 0.f : expf(m3 - m_new3);
        l0 = l0 * alpha0 + sdpa_exp_submax_fp32(score0, max_jj, m_new0);
        l1 = l1 * alpha1 + sdpa_exp_submax_fp32(score1, max_jj, m_new1);
        l2 = l2 * alpha2 + sdpa_exp_submax_fp32(score2, max_jj, m_new2);
        l3 = l3 * alpha3 + sdpa_exp_submax_fp32(score3, max_jj, m_new3);
        m0 = m_new0;
        m1 = m_new1;
        m2 = m_new2;
        m3 = m_new3;

        sdpa_pv_tile4_bf16s(out0, out1, out2, out3, value_head.row<const unsigned short>(n), score0, score1, score2, score3, alpha0, alpha1, alpha2, alpha3, max_jj, value_dim);
    }

    if (!state.empty())
    {
        float* state0 = state;
        float* state1 = state0 + value_dim + 2;
        float* state2 = state1 + value_dim + 2;
        float* state3 = state2 + value_dim + 2;
        state0[0] = m0;
        state0[1] = l0;
        state1[0] = m1;
        state1[1] = l1;
        state2[0] = m2;
        state2[1] = l2;
        state3[0] = m3;
        state3[1] = l3;
        memcpy(state0 + 2, out0, value_dim * sizeof(float));
        memcpy(state1 + 2, out1, value_dim * sizeof(float));
        memcpy(state2 + 2, out2, value_dim * sizeof(float));
        memcpy(state3 + 2, out3, value_dim * sizeof(float));
    }
    else
    {
        float* output0 = top_blob.channel(q0);
        float* output1 = top_blob.channel(q0 + 1);
        float* output2 = top_blob.channel(q0 + 2);
        float* output3 = top_blob.channel(q0 + 3);
        memcpy(output0, out0, value_dim * sizeof(float));
        memcpy(output1, out1, value_dim * sizeof(float));
        memcpy(output2, out2, value_dim * sizeof(float));
        memcpy(output3, out3, value_dim * sizeof(float));
        if (l0 != 0.f)
            sdpa_normalize_fp32(output0, l0, value_dim);
        if (l1 != 0.f)
            sdpa_normalize_fp32(output1, l1, value_dim);
        if (l2 != 0.f)
            sdpa_normalize_fp32(output2, l2, value_dim);
        if (l3 != 0.f)
            sdpa_normalize_fp32(output3, l3, value_dim);
    }
}

static float sdpa_dot_product_tile1_bf16s(const unsigned short* query, const unsigned short* key, int size)
{
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _sum_avx512 = _mm512_setzero_ps();
#endif // __AVX512F__
    __m256 _sum_avx = _mm256_setzero_ps();
#endif // __AVX__
    __m128 _sum = _mm_setzero_ps();
#endif // __SSE2__
    float sum = 0.f;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _q = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(query + i)));
        __m512 _k = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(key + i)));
        _sum_avx512 = _mm512_fmadd_ps(_q, _k, _sum_avx512);
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _q = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(query + i)));
        __m256 _k = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(key + i)));
        _sum_avx = _mm256_comp_fmadd_ps(_q, _k, _sum_avx);
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _q = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(query + i)));
        __m128 _k = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(key + i)));
        _sum = _mm_comp_fmadd_ps(_q, _k, _sum);
    }
#endif // __SSE2__
    for (; i < size; i++)
        sum += bfloat16_to_float32(query[i]) * bfloat16_to_float32(key[i]);

#if __SSE2__
#if __AVX__
#if __AVX512F__
    sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
    sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
    sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__

    return sum;
}

static void sdpa_pv_tile1_bf16s(float* out, const unsigned short* value, const float* score, float alpha, int size, int value_dim)
{
    int d = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; d + 63 < value_dim; d += 64)
    {
        __m512 _out0 = _mm512_mul_ps(_mm512_loadu_ps(out + d), _mm512_set1_ps(alpha));
        __m512 _out1 = _mm512_mul_ps(_mm512_loadu_ps(out + d + 16), _mm512_set1_ps(alpha));
        __m512 _out2 = _mm512_mul_ps(_mm512_loadu_ps(out + d + 32), _mm512_set1_ps(alpha));
        __m512 _out3 = _mm512_mul_ps(_mm512_loadu_ps(out + d + 48), _mm512_set1_ps(alpha));
        for (int j = 0; j < size; j++)
        {
            const unsigned short* vptr = value + (size_t)j * value_dim + d;
            __m512 _p = _mm512_set1_ps(score[j]);
            _out0 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)vptr)), _p, _out0);
            _out1 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(vptr + 16))), _p, _out1);
            _out2 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(vptr + 32))), _p, _out2);
            _out3 = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(vptr + 48))), _p, _out3);
        }
        _mm512_storeu_ps(out + d, _out0);
        _mm512_storeu_ps(out + d + 16, _out1);
        _mm512_storeu_ps(out + d + 32, _out2);
        _mm512_storeu_ps(out + d + 48, _out3);
    }

    for (; d + 15 < value_dim; d += 16)
    {
        __m512 _out = _mm512_mul_ps(_mm512_loadu_ps(out + d), _mm512_set1_ps(alpha));
        for (int j = 0; j < size; j++)
        {
            const unsigned short* vptr = value + (size_t)j * value_dim + d;
            _out = _mm512_fmadd_ps(bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)vptr)), _mm512_set1_ps(score[j]), _out);
        }
        _mm512_storeu_ps(out + d, _out);
    }
#endif // __AVX512F__
#if !__AVX512F__
    for (; d + 15 < value_dim; d += 16)
    {
        __m256 _out0 = _mm256_mul_ps(_mm256_loadu_ps(out + d), _mm256_set1_ps(alpha));
        __m256 _out1 = _mm256_mul_ps(_mm256_loadu_ps(out + d + 8), _mm256_set1_ps(alpha));
        for (int j = 0; j < size; j++)
        {
            const unsigned short* vptr = value + (size_t)j * value_dim + d;
            __m256 _p = _mm256_set1_ps(score[j]);
            _out0 = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)vptr)), _p, _out0);
            _out1 = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)(vptr + 8))), _p, _out1);
        }
        _mm256_storeu_ps(out + d, _out0);
        _mm256_storeu_ps(out + d + 8, _out1);
    }
#endif // !__AVX512F__

    for (; d + 7 < value_dim; d += 8)
    {
        __m256 _out = _mm256_mul_ps(_mm256_loadu_ps(out + d), _mm256_set1_ps(alpha));
        for (int j = 0; j < size; j++)
        {
            const unsigned short* vptr = value + (size_t)j * value_dim + d;
            _out = _mm256_comp_fmadd_ps(bfloat2float_avx(_mm_loadu_si128((const __m128i*)vptr)), _mm256_set1_ps(score[j]), _out);
        }
        _mm256_storeu_ps(out + d, _out);
    }
#endif // __AVX__
#if !__AVX__
    for (; d + 7 < value_dim; d += 8)
    {
        __m128 _out0 = _mm_mul_ps(_mm_loadu_ps(out + d), _mm_set1_ps(alpha));
        __m128 _out1 = _mm_mul_ps(_mm_loadu_ps(out + d + 4), _mm_set1_ps(alpha));
        for (int j = 0; j < size; j++)
        {
            const unsigned short* vptr = value + (size_t)j * value_dim + d;
            __m128 _p = _mm_set1_ps(score[j]);
            _out0 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)vptr)), _p, _out0);
            _out1 = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(vptr + 4))), _p, _out1);
        }
        _mm_storeu_ps(out + d, _out0);
        _mm_storeu_ps(out + d + 4, _out1);
    }
#endif // !__AVX__
    for (; d + 3 < value_dim; d += 4)
    {
        __m128 _out = _mm_mul_ps(_mm_loadu_ps(out + d), _mm_set1_ps(alpha));
        for (int j = 0; j < size; j++)
        {
            const unsigned short* vptr = value + (size_t)j * value_dim + d;
            _out = _mm_comp_fmadd_ps(bfloat2float_sse(_mm_loadl_epi64((const __m128i*)vptr)), _mm_set1_ps(score[j]), _out);
        }
        _mm_storeu_ps(out + d, _out);
    }
#endif // __SSE2__
    for (; d < value_dim; d++)
    {
        float sum = out[d] * alpha;
        for (int j = 0; j < size; j++)
            sum += score[j] * bfloat16_to_float32(value[(size_t)j * value_dim + d]);
        out[d] = sum;
    }
}

static void sdpa_decode_tile1_bf16s(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, int q, int g, int n_begin, int n_end, Mat& workspace, Mat& state)
{
    const int head_dim = query.w;
    const int value_dim = value.w;
    const int block_n = 256;

    const unsigned short* query_ptr = query.channel(q);
    Mat mask_head;
    if (!attn_mask_blob.empty())
        mask_head = attn_mask_blob.dims == 3 ? attn_mask_blob.channel(attn_mask_blob.c > 1 ? q : 0) : attn_mask_blob;
    const float* mask_fp32 = mask_head.elembits() == 32 ? mask_head : 0;
    const unsigned short* mask_bf16 = mask_head.elembits() == 16 ? mask_head : 0;

    float* workspace_ptr = workspace;
    float* score = workspace_ptr;
    float* out = workspace_ptr + block_n * 4;
    memset(out, 0, value_dim * sizeof(float));

    const Mat key_head = key.channel(g);
    const Mat value_head = value.channel(g);

    float m = -FLT_MAX;
    float l = 0.f;

    for (int n = n_begin; n < n_end; n += block_n)
    {
        const int max_jj = std::min(n_end - n, block_n);
        float block_max = -FLT_MAX;
        for (int j = 0; j < max_jj; j++)
        {
            float s = sdpa_dot_product_tile1_bf16s(query_ptr, key_head.row<const unsigned short>(n + j), head_dim) * scale;
            if (mask_fp32)
                s += mask_fp32[n + j];
            else if (mask_bf16)
                s += bfloat16_to_float32(mask_bf16[n + j]);
            score[j] = s;
            block_max = std::max(block_max, s);
        }

        const float m_new = std::max(m, block_max);
        const float alpha = l == 0.f ? 0.f : expf(m - m_new);
        l = l * alpha + sdpa_exp_submax_fp32(score, max_jj, m_new);
        m = m_new;

        sdpa_pv_tile1_bf16s(out, value_head.row<const unsigned short>(n), score, alpha, max_jj, value_dim);
    }

    if (!state.empty())
    {
        float* state_ptr = state;
        state_ptr[0] = m;
        state_ptr[1] = l;
        memcpy(state_ptr + 2, out, value_dim * sizeof(float));
    }
    else
    {
        float* output = top_blob.channel(q);
        memcpy(output, out, value_dim * sizeof(float));
        if (l != 0.f)
            sdpa_normalize_fp32(output, l, value_dim);
    }
}

static int sdpa_decode_bf16s(const Mat& query, const Mat& key, const Mat& value, const Mat& attn_mask_blob, Mat& top_blob, float scale, const Option& opt)
{
    const int num_query_heads = query.c;
    const int num_kv_heads = key.c;
    const int key_seqlen = key.h;
    const int value_dim = value.w;
    const int num_query_heads_per_kv_head = num_query_heads / num_kv_heads;
    const int block_q = 4;
    const int num_qblocks = (num_query_heads_per_kv_head + block_q - 1) / block_q;
    const int num_tasks = num_kv_heads * num_qblocks;

    const int num_threads = std::max(opt.num_threads, 1);
    int num_kv_chunks = 1;
    if (num_tasks < num_threads && key_seqlen >= 512)
    {
        num_kv_chunks = std::min((num_threads + num_tasks - 1) / num_tasks, key_seqlen / 256);
        num_kv_chunks = std::max(num_kv_chunks, 1);
    }

    Mat workspace(4 * (256 + value_dim), 1, num_threads, 4u, opt.workspace_allocator);
    if (workspace.empty())
        return -100;

    Mat partials;
    if (num_kv_chunks > 1)
    {
        partials.create((value_dim + 2) * block_q, 1, num_tasks * num_kv_chunks, 4u, opt.workspace_allocator);
        if (partials.empty())
            return -100;
    }

#pragma omp parallel for num_threads(opt.num_threads)
    for (int ti = 0; ti < num_tasks * num_kv_chunks; ti++)
    {
        const int task_id = ti / num_kv_chunks;
        const int chunk_id = ti % num_kv_chunks;
        const int g = task_id / num_qblocks;
        const int qblock_id = task_id % num_qblocks;
        const int q0 = g * num_query_heads_per_kv_head + qblock_id * block_q;
        const int max_qq = std::min(num_query_heads_per_kv_head - qblock_id * block_q, block_q);
        const int n_begin = chunk_id * key_seqlen / num_kv_chunks;
        const int n_end = (chunk_id + 1) * key_seqlen / num_kv_chunks;

        Mat workspace_tile = workspace.channel(get_omp_thread_num());
        Mat state;
        if (num_kv_chunks > 1)
            state = partials.channel(ti);
        if (max_qq == 4)
        {
            sdpa_decode_tile4_bf16s(query, key, value, attn_mask_blob, top_blob, scale, q0, g, n_begin, n_end, workspace_tile, state);
        }
        else
        {
            for (int qq = 0; qq < max_qq; qq++)
            {
                Mat state_q;
                if (!state.empty())
                    state_q = state.range(qq * (value_dim + 2), value_dim + 2);
                sdpa_decode_tile1_bf16s(query, key, value, attn_mask_blob, top_blob, scale, q0 + qq, g, n_begin, n_end, workspace_tile, state_q);
            }
        }
    }

    if (num_kv_chunks > 1)
    {
#pragma omp parallel for num_threads(opt.num_threads)
        for (int task_id = 0; task_id < num_tasks; task_id++)
        {
            const int g = task_id / num_qblocks;
            const int qblock_id = task_id % num_qblocks;
            const int q0 = g * num_query_heads_per_kv_head + qblock_id * block_q;
            const int max_qq = std::min(num_query_heads_per_kv_head - qblock_id * block_q, block_q);
            for (int q = 0; q < max_qq; q++)
            {
                float m = -FLT_MAX;
                for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
                {
                    const float* state_q = partials.channel(task_id * num_kv_chunks + chunk_id);
                    state_q += q * (value_dim + 2);
                    m = std::max(m, state_q[0]);
                }

                float* outptr = top_blob.channel(q0 + q);
                memset(outptr, 0, value_dim * sizeof(float));
                float l = 0.f;
                for (int chunk_id = 0; chunk_id < num_kv_chunks; chunk_id++)
                {
                    const float* state_q = partials.channel(task_id * num_kv_chunks + chunk_id);
                    state_q += q * (value_dim + 2);
                    float partial_scale = state_q[1] == 0.f ? 0.f : expf(state_q[0] - m);
                    l += state_q[1] * partial_scale;
                    sdpa_scale_add_fp32(outptr, state_q + 2, partial_scale, value_dim);
                }
                if (l != 0.f)
                    sdpa_normalize_fp32(outptr, l, value_dim);
            }
        }
    }

    return 0;
}

#endif // NCNN_BF16
