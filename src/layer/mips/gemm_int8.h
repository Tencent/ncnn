// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static NCNN_FORCEINLINE signed char gemm_float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static void pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    signed char* pp = AT;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;
        const signed char* p2 = (const signed char*)A + (i + ii + 2) * A_hstep + k;
        const signed char* p3 = (const signed char*)A + (i + ii + 3) * A_hstep + k;
        const signed char* p4 = (const signed char*)A + (i + ii + 4) * A_hstep + k;
        const signed char* p5 = (const signed char*)A + (i + ii + 5) * A_hstep + k;
        const signed char* p6 = (const signed char*)A + (i + ii + 6) * A_hstep + k;
        const signed char* p7 = (const signed char*)A + (i + ii + 7) * A_hstep + k;

        int kk = 0;
        for (; kk + 15 < max_kk; kk += 16)
        {
            __builtin_prefetch(p0 + 64);
            __builtin_prefetch(p1 + 64);
            __builtin_prefetch(p2 + 64);
            __builtin_prefetch(p3 + 64);
            __builtin_prefetch(p4 + 64);
            __builtin_prefetch(p5 + 64);
            __builtin_prefetch(p6 + 64);
            __builtin_prefetch(p7 + 64);
            v4i32 _r0 = (v4i32)__msa_ld_b(p0, 0);
            v4i32 _r1 = (v4i32)__msa_ld_b(p1, 0);
            v4i32 _r2 = (v4i32)__msa_ld_b(p2, 0);
            v4i32 _r3 = (v4i32)__msa_ld_b(p3, 0);
            v4i32 _r4 = (v4i32)__msa_ld_b(p4, 0);
            v4i32 _r5 = (v4i32)__msa_ld_b(p5, 0);
            v4i32 _r6 = (v4i32)__msa_ld_b(p6, 0);
            v4i32 _r7 = (v4i32)__msa_ld_b(p7, 0);
            transpose4x8_epi32(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            __msa_st_b((v16i8)_r0, pp, 0);
            __msa_st_b((v16i8)_r1, pp + 16, 0);
            __msa_st_b((v16i8)_r2, pp + 32, 0);
            __msa_st_b((v16i8)_r3, pp + 48, 0);
            __msa_st_b((v16i8)_r4, pp + 64, 0);
            __msa_st_b((v16i8)_r5, pp + 80, 0);
            __msa_st_b((v16i8)_r6, pp + 96, 0);
            __msa_st_b((v16i8)_r7, pp + 112, 0);
            pp += 128;
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
            p4 += 16;
            p5 += 16;
            p6 += 16;
            p7 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            __builtin_prefetch(p0 + 32);
            __builtin_prefetch(p1 + 32);
            __builtin_prefetch(p2 + 32);
            __builtin_prefetch(p3 + 32);
            __builtin_prefetch(p4 + 32);
            __builtin_prefetch(p5 + 32);
            __builtin_prefetch(p6 + 32);
            __builtin_prefetch(p7 + 32);
            v4i32 _r0 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p0);
            v4i32 _r1 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p1);
            v4i32 _r2 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p2);
            v4i32 _r3 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p3);
            v4i32 _r4 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p4);
            v4i32 _r5 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p5);
            v4i32 _r6 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p6);
            v4i32 _r7 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p7);
            transpose4x8_epi32(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            __msa_st_b((v16i8)_r0, pp, 0);
            __msa_st_b((v16i8)_r1, pp + 16, 0);
            __msa_st_b((v16i8)_r2, pp + 32, 0);
            __msa_st_b((v16i8)_r3, pp + 48, 0);
            pp += 64;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
            p4 += 8;
            p5 += 8;
            p6 += 8;
            p7 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint64_t v0 = (uint32_t)*(const unsigned int*)p0 | ((uint64_t)(uint32_t)*(const unsigned int*)p1 << 32);
            uint64_t v1 = (uint32_t)*(const unsigned int*)p2 | ((uint64_t)(uint32_t)*(const unsigned int*)p3 << 32);
            uint64_t v2 = (uint32_t)*(const unsigned int*)p4 | ((uint64_t)(uint32_t)*(const unsigned int*)p5 << 32);
            uint64_t v3 = (uint32_t)*(const unsigned int*)p6 | ((uint64_t)(uint32_t)*(const unsigned int*)p7 << 32);
            v16i8 _r0 = (v16i8)__msa_insert_d(__msa_fill_d(0), 0, v0);
            _r0 = (v16i8)__msa_insert_d((v2i64)_r0, 1, v1);
            v16i8 _r1 = (v16i8)__msa_insert_d(__msa_fill_d(0), 0, v2);
            _r1 = (v16i8)__msa_insert_d((v2i64)_r1, 1, v3);
            __msa_st_b(_r0, pp, 0);
            __msa_st_b(_r1, pp + 16, 0);
            pp += 32;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
            p4 += 4;
            p5 += 4;
            p6 += 4;
            p7 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p4[0];
            pp[5] = p5[0];
            pp[6] = p6[0];
            pp[7] = p7[0];
            pp += 8;
            p0++;
            p1++;
            p2++;
            p3++;
            p4++;
            p5++;
            p6++;
            p7++;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;
        const signed char* p2 = (const signed char*)A + (i + ii + 2) * A_hstep + k;
        const signed char* p3 = (const signed char*)A + (i + ii + 3) * A_hstep + k;

        int kk = 0;
        for (; kk + 15 < max_kk; kk += 16)
        {
            __builtin_prefetch(p0 + 64);
            __builtin_prefetch(p1 + 64);
            __builtin_prefetch(p2 + 64);
            __builtin_prefetch(p3 + 64);
            v4i32 _r0 = (v4i32)__msa_ld_b(p0, 0);
            v4i32 _r1 = (v4i32)__msa_ld_b(p1, 0);
            v4i32 _r2 = (v4i32)__msa_ld_b(p2, 0);
            v4i32 _r3 = (v4i32)__msa_ld_b(p3, 0);
            transpose4x4_epi32(_r0, _r1, _r2, _r3);
            __msa_st_b((v16i8)_r0, pp, 0);
            __msa_st_b((v16i8)_r1, pp + 16, 0);
            __msa_st_b((v16i8)_r2, pp + 32, 0);
            __msa_st_b((v16i8)_r3, pp + 48, 0);
            pp += 64;
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            __builtin_prefetch(p0 + 32);
            __builtin_prefetch(p1 + 32);
            __builtin_prefetch(p2 + 32);
            __builtin_prefetch(p3 + 32);
            v4i32 _r0 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p0);
            v4i32 _r1 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p1);
            v4i32 _r2 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p2);
            v4i32 _r3 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p3);
            transpose4x4_epi32(_r0, _r1, _r2, _r3);
            __msa_st_b((v16i8)_r0, pp, 0);
            __msa_st_b((v16i8)_r1, pp + 16, 0);
            pp += 32;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint64_t v0 = (uint32_t)*(const unsigned int*)p0 | ((uint64_t)(uint32_t)*(const unsigned int*)p1 << 32);
            uint64_t v1 = (uint32_t)*(const unsigned int*)p2 | ((uint64_t)(uint32_t)*(const unsigned int*)p3 << 32);
            v16i8 _r0 = (v16i8)__msa_insert_d(__msa_fill_d(0), 0, v0);
            _r0 = (v16i8)__msa_insert_d((v2i64)_r0, 1, v1);
            __msa_st_b(_r0, pp, 0);
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;
        const signed char* p1 = (const signed char*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p1[0];
            pp[5] = p1[1];
            pp[6] = p1[2];
            pp[7] = p1[3];
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const signed char* p0 = (const signed char*)A + (i + ii) * A_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp += 4;
            p0 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    signed char* pp = AT;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* p0 = (const signed char*)A + (size_t)k * A_hstep + i + ii;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + A_hstep * 4);
            const signed char* p1 = p0 + A_hstep;
            const signed char* p2 = p1 + A_hstep;
            const signed char* p3 = p2 + A_hstep;

            v16i8 _r0 = (v16i8)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p0);
            v16i8 _r1 = (v16i8)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p1);
            v16i8 _r2 = (v16i8)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p2);
            v16i8 _r3 = (v16i8)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p3);
            transpose16x4_epi8(_r0, _r1, _r2, _r3);
            __msa_st_b(_r0, pp, 0);
            __msa_st_b(_r1, pp + 16, 0);
            pp += 32;
            p0 += A_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + A_hstep);
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p0[4];
            pp[5] = p0[5];
            pp[6] = p0[6];
            pp[7] = p0[7];
            pp += 8;
            p0 += A_hstep;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* p0 = (const signed char*)A + (size_t)k * A_hstep + i + ii;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + A_hstep * 4);
            const signed char* p1 = p0 + A_hstep;
            const signed char* p2 = p1 + A_hstep;
            const signed char* p3 = p2 + A_hstep;

            v16i8 _r0 = (v16i8)__msa_fill_w(*(const int*)p0);
            v16i8 _r1 = (v16i8)__msa_fill_w(*(const int*)p1);
            v16i8 _r2 = (v16i8)__msa_fill_w(*(const int*)p2);
            v16i8 _r3 = (v16i8)__msa_fill_w(*(const int*)p3);
            transpose16x4_epi8(_r0, _r1, _r2, _r3);
            __msa_st_b(_r0, pp, 0);
            pp += 16;
            p0 += A_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + A_hstep);
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp += 4;
            p0 += A_hstep;
        }
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* p0 = (const signed char*)A + (size_t)k * A_hstep + i + ii;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + A_hstep * 4);
            const signed char* p1 = p0 + A_hstep;
            const signed char* p2 = p1 + A_hstep;
            const signed char* p3 = p2 + A_hstep;
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p0[1];
            pp[5] = p1[1];
            pp[6] = p2[1];
            pp[7] = p3[1];
            pp += 8;
            p0 += A_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + A_hstep);
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp += 2;
            p0 += A_hstep;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const signed char* p0 = (const signed char*)A + (size_t)k * A_hstep + i + ii;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + A_hstep * 4);
            pp[0] = p0[0];
            pp[1] = p0[A_hstep];
            pp[2] = p0[A_hstep * 2];
            pp[3] = p0[A_hstep * 3];
            pp += 4;
            p0 += A_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + A_hstep);
            pp[0] = p0[0];
            pp += 1;
            p0 += A_hstep;
        }
    }
}

static void pack_B_tile_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    signed char* pp = BT;

    int jj = 0;
#if __mips_msa
    for (; jj + 7 < max_jj; jj += 8)
    {
        const signed char* p0 = (const signed char*)B + (j + jj) * B_hstep + k;
        const signed char* p1 = (const signed char*)B + (j + jj + 1) * B_hstep + k;
        const signed char* p2 = (const signed char*)B + (j + jj + 2) * B_hstep + k;
        const signed char* p3 = (const signed char*)B + (j + jj + 3) * B_hstep + k;
        const signed char* p4 = (const signed char*)B + (j + jj + 4) * B_hstep + k;
        const signed char* p5 = (const signed char*)B + (j + jj + 5) * B_hstep + k;
        const signed char* p6 = (const signed char*)B + (j + jj + 6) * B_hstep + k;
        const signed char* p7 = (const signed char*)B + (j + jj + 7) * B_hstep + k;

        int kk = 0;
        for (; kk + 15 < max_kk; kk += 16)
        {
            __builtin_prefetch(p0 + 64);
            __builtin_prefetch(p1 + 64);
            __builtin_prefetch(p2 + 64);
            __builtin_prefetch(p3 + 64);
            __builtin_prefetch(p4 + 64);
            __builtin_prefetch(p5 + 64);
            __builtin_prefetch(p6 + 64);
            __builtin_prefetch(p7 + 64);
            v4i32 _r0 = (v4i32)__msa_ld_b(p0, 0);
            v4i32 _r1 = (v4i32)__msa_ld_b(p1, 0);
            v4i32 _r2 = (v4i32)__msa_ld_b(p2, 0);
            v4i32 _r3 = (v4i32)__msa_ld_b(p3, 0);
            v4i32 _r4 = (v4i32)__msa_ld_b(p4, 0);
            v4i32 _r5 = (v4i32)__msa_ld_b(p5, 0);
            v4i32 _r6 = (v4i32)__msa_ld_b(p6, 0);
            v4i32 _r7 = (v4i32)__msa_ld_b(p7, 0);
            transpose4x8_epi32(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            __msa_st_b((v16i8)_r0, pp, 0);
            __msa_st_b((v16i8)_r1, pp + 16, 0);
            __msa_st_b((v16i8)_r2, pp + 32, 0);
            __msa_st_b((v16i8)_r3, pp + 48, 0);
            __msa_st_b((v16i8)_r4, pp + 64, 0);
            __msa_st_b((v16i8)_r5, pp + 80, 0);
            __msa_st_b((v16i8)_r6, pp + 96, 0);
            __msa_st_b((v16i8)_r7, pp + 112, 0);
            pp += 128;
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
            p4 += 16;
            p5 += 16;
            p6 += 16;
            p7 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            __builtin_prefetch(p0 + 32);
            __builtin_prefetch(p1 + 32);
            __builtin_prefetch(p2 + 32);
            __builtin_prefetch(p3 + 32);
            __builtin_prefetch(p4 + 32);
            __builtin_prefetch(p5 + 32);
            __builtin_prefetch(p6 + 32);
            __builtin_prefetch(p7 + 32);
            v4i32 _r0 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p0);
            v4i32 _r1 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p1);
            v4i32 _r2 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p2);
            v4i32 _r3 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p3);
            v4i32 _r4 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p4);
            v4i32 _r5 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p5);
            v4i32 _r6 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p6);
            v4i32 _r7 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p7);
            transpose4x8_epi32(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            __msa_st_b((v16i8)_r0, pp, 0);
            __msa_st_b((v16i8)_r1, pp + 16, 0);
            __msa_st_b((v16i8)_r2, pp + 32, 0);
            __msa_st_b((v16i8)_r3, pp + 48, 0);
            pp += 64;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
            p4 += 8;
            p5 += 8;
            p6 += 8;
            p7 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint64_t v0 = (uint32_t)*(const unsigned int*)p0 | ((uint64_t)(uint32_t)*(const unsigned int*)p1 << 32);
            uint64_t v1 = (uint32_t)*(const unsigned int*)p2 | ((uint64_t)(uint32_t)*(const unsigned int*)p3 << 32);
            uint64_t v2 = (uint32_t)*(const unsigned int*)p4 | ((uint64_t)(uint32_t)*(const unsigned int*)p5 << 32);
            uint64_t v3 = (uint32_t)*(const unsigned int*)p6 | ((uint64_t)(uint32_t)*(const unsigned int*)p7 << 32);
            v16i8 _r0 = (v16i8)__msa_insert_d(__msa_fill_d(0), 0, v0);
            _r0 = (v16i8)__msa_insert_d((v2i64)_r0, 1, v1);
            v16i8 _r1 = (v16i8)__msa_insert_d(__msa_fill_d(0), 0, v2);
            _r1 = (v16i8)__msa_insert_d((v2i64)_r1, 1, v3);
            __msa_st_b(_r0, pp, 0);
            __msa_st_b(_r1, pp + 16, 0);
            pp += 32;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
            p4 += 4;
            p5 += 4;
            p6 += 4;
            p7 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p4[0];
            pp[5] = p5[0];
            pp[6] = p6[0];
            pp[7] = p7[0];
            pp += 8;
            p0++;
            p1++;
            p2++;
            p3++;
            p4++;
            p5++;
            p6++;
            p7++;
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = (const signed char*)B + (j + jj) * B_hstep + k;
        const signed char* p1 = (const signed char*)B + (j + jj + 1) * B_hstep + k;
        const signed char* p2 = (const signed char*)B + (j + jj + 2) * B_hstep + k;
        const signed char* p3 = (const signed char*)B + (j + jj + 3) * B_hstep + k;

        int kk = 0;
        for (; kk + 15 < max_kk; kk += 16)
        {
            __builtin_prefetch(p0 + 64);
            __builtin_prefetch(p1 + 64);
            __builtin_prefetch(p2 + 64);
            __builtin_prefetch(p3 + 64);
            v4i32 _r0 = (v4i32)__msa_ld_b(p0, 0);
            v4i32 _r1 = (v4i32)__msa_ld_b(p1, 0);
            v4i32 _r2 = (v4i32)__msa_ld_b(p2, 0);
            v4i32 _r3 = (v4i32)__msa_ld_b(p3, 0);
            transpose4x4_epi32(_r0, _r1, _r2, _r3);
            __msa_st_b((v16i8)_r0, pp, 0);
            __msa_st_b((v16i8)_r1, pp + 16, 0);
            __msa_st_b((v16i8)_r2, pp + 32, 0);
            __msa_st_b((v16i8)_r3, pp + 48, 0);
            pp += 64;
            p0 += 16;
            p1 += 16;
            p2 += 16;
            p3 += 16;
        }
        for (; kk + 7 < max_kk; kk += 8)
        {
            __builtin_prefetch(p0 + 32);
            __builtin_prefetch(p1 + 32);
            __builtin_prefetch(p2 + 32);
            __builtin_prefetch(p3 + 32);
            v4i32 _r0 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p0);
            v4i32 _r1 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p1);
            v4i32 _r2 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p2);
            v4i32 _r3 = (v4i32)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p3);
            transpose4x4_epi32(_r0, _r1, _r2, _r3);
            __msa_st_b((v16i8)_r0, pp, 0);
            __msa_st_b((v16i8)_r1, pp + 16, 0);
            pp += 32;
            p0 += 8;
            p1 += 8;
            p2 += 8;
            p3 += 8;
        }
        for (; kk + 3 < max_kk; kk += 4)
        {
            uint64_t v0 = (uint32_t)*(const unsigned int*)p0 | ((uint64_t)(uint32_t)*(const unsigned int*)p1 << 32);
            uint64_t v1 = (uint32_t)*(const unsigned int*)p2 | ((uint64_t)(uint32_t)*(const unsigned int*)p3 << 32);
            v16i8 _r0 = (v16i8)__msa_insert_d(__msa_fill_d(0), 0, v0);
            _r0 = (v16i8)__msa_insert_d((v2i64)_r0, 1, v1);
            __msa_st_b(_r0, pp, 0);
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
#endif // __mips_msa
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = (const signed char*)B + (j + jj) * B_hstep + k;
        const signed char* p1 = (const signed char*)B + (j + jj + 1) * B_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p1[0];
            pp[5] = p1[1];
            pp[6] = p1[2];
            pp[7] = p1[3];
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const signed char* p0 = (const signed char*)B + (j + jj) * B_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp += 4;
            p0 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = p0[0];
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_B_tile_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    signed char* pp = BT;

    int jj = 0;
#if __mips_msa
    for (; jj + 7 < max_jj; jj += 8)
    {
        const signed char* p0 = (const signed char*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + B_hstep * 4);
            const signed char* p1 = p0 + B_hstep;
            const signed char* p2 = p1 + B_hstep;
            const signed char* p3 = p2 + B_hstep;

            v16i8 _r0 = (v16i8)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p0);
            v16i8 _r1 = (v16i8)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p1);
            v16i8 _r2 = (v16i8)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p2);
            v16i8 _r3 = (v16i8)__msa_insert_d(__msa_fill_d(0), 0, *(const int64_t*)p3);
            transpose16x4_epi8(_r0, _r1, _r2, _r3);
            __msa_st_b(_r0, pp, 0);
            __msa_st_b(_r1, pp + 16, 0);
            pp += 32;
            p0 += B_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + B_hstep);
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp[4] = p0[4];
            pp[5] = p0[5];
            pp[6] = p0[6];
            pp[7] = p0[7];
            pp += 8;
            p0 += B_hstep;
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        const signed char* p0 = (const signed char*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + B_hstep * 4);
            const signed char* p1 = p0 + B_hstep;
            const signed char* p2 = p1 + B_hstep;
            const signed char* p3 = p2 + B_hstep;

            v16i8 _r0 = (v16i8)__msa_fill_w(*(const int*)p0);
            v16i8 _r1 = (v16i8)__msa_fill_w(*(const int*)p1);
            v16i8 _r2 = (v16i8)__msa_fill_w(*(const int*)p2);
            v16i8 _r3 = (v16i8)__msa_fill_w(*(const int*)p3);
            transpose16x4_epi8(_r0, _r1, _r2, _r3);
            __msa_st_b(_r0, pp, 0);
            pp += 16;
            p0 += B_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + B_hstep);
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp[2] = p0[2];
            pp[3] = p0[3];
            pp += 4;
            p0 += B_hstep;
        }
    }
#endif // __mips_msa
    for (; jj + 1 < max_jj; jj += 2)
    {
        const signed char* p0 = (const signed char*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + B_hstep * 4);
            const signed char* p1 = p0 + B_hstep;
            const signed char* p2 = p1 + B_hstep;
            const signed char* p3 = p2 + B_hstep;
            pp[0] = p0[0];
            pp[1] = p1[0];
            pp[2] = p2[0];
            pp[3] = p3[0];
            pp[4] = p0[1];
            pp[5] = p1[1];
            pp[6] = p2[1];
            pp[7] = p3[1];
            pp += 8;
            p0 += B_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + B_hstep);
            pp[0] = p0[0];
            pp[1] = p0[1];
            pp += 2;
            p0 += B_hstep;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const signed char* p0 = (const signed char*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + B_hstep * 4);
            pp[0] = p0[0];
            pp[1] = p0[B_hstep];
            pp[2] = p0[B_hstep * 2];
            pp[3] = p0[B_hstep * 3];
            pp += 4;
            p0 += B_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + B_hstep);
            pp[0] = p0[0];
            pp += 1;
            p0 += B_hstep;
        }
    }
}

static void compute_A_tile_int8_scales(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    for (int ii = 0; ii < max_ii; ii++)
    {
        const float* ptr = (const float*)A + (i + ii) * A_hstep;

        float absmax = 0.f;
        for (int kk = 0; kk < A.w; kk++)
        {
            absmax = std::max(absmax, (float)fabs(ptr[kk]));
        }

        float scale = absmax == 0.f ? 1.f : 127.f / absmax;
        scales[i + ii] = scale;
        out_descales[i + ii] = 1.f / (scale * B_scale);
    }
}

static void transpose_compute_A_tile_int8_scales(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;
    const int K = A.dims == 3 ? A.c : A.h;

    for (int ii = 0; ii < max_ii; ii++)
    {
        const float* ptr = (const float*)A + i + ii;

        float absmax = 0.f;
        for (int kk = 0; kk < K; kk++)
        {
            absmax = std::max(absmax, (float)fabs(ptr[0]));
            ptr += A_hstep;
        }

        float scale = absmax == 0.f ? 1.f : 127.f / absmax;
        scales[i + ii] = scale;
        out_descales[i + ii] = 1.f / (scale * B_scale);
    }
}

static void pack_A_tile_fp32_to_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    signed char* pp = AT;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;
        const float* p4 = (const float*)A + (i + ii + 4) * A_hstep + k;
        const float* p5 = (const float*)A + (i + ii + 5) * A_hstep + k;
        const float* p6 = (const float*)A + (i + ii + 6) * A_hstep + k;
        const float* p7 = (const float*)A + (i + ii + 7) * A_hstep + k;
        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];
        const float scale4 = scales[i + ii + 4];
        const float scale5 = scales[i + ii + 5];
        const float scale6 = scales[i + ii + 6];
        const float scale7 = scales[i + ii + 7];
        v4f32 _scale0 = __msa_fill_w_f32(scale0);
        v4f32 _scale1 = __msa_fill_w_f32(scale1);
        v4f32 _scale2 = __msa_fill_w_f32(scale2);
        v4f32 _scale3 = __msa_fill_w_f32(scale3);
        v4f32 _scale4 = __msa_fill_w_f32(scale4);
        v4f32 _scale5 = __msa_fill_w_f32(scale5);
        v4f32 _scale6 = __msa_fill_w_f32(scale6);
        v4f32 _scale7 = __msa_fill_w_f32(scale7);

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + 16);
            __builtin_prefetch(p1 + 16);
            __builtin_prefetch(p2 + 16);
            __builtin_prefetch(p3 + 16);
            __builtin_prefetch(p4 + 16);
            __builtin_prefetch(p5 + 16);
            __builtin_prefetch(p6 + 16);
            __builtin_prefetch(p7 + 16);
            v4f32 _p0 = __msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _scale0);
            v4f32 _p1 = __msa_fmul_w((v4f32)__msa_ld_w(p1, 0), _scale1);
            v4f32 _p2 = __msa_fmul_w((v4f32)__msa_ld_w(p2, 0), _scale2);
            v4f32 _p3 = __msa_fmul_w((v4f32)__msa_ld_w(p3, 0), _scale3);
            v4f32 _p4 = __msa_fmul_w((v4f32)__msa_ld_w(p4, 0), _scale4);
            v4f32 _p5 = __msa_fmul_w((v4f32)__msa_ld_w(p5, 0), _scale5);
            v4f32 _p6 = __msa_fmul_w((v4f32)__msa_ld_w(p6, 0), _scale6);
            v4f32 _p7 = __msa_fmul_w((v4f32)__msa_ld_w(p7, 0), _scale7);

            *((int64_t*)pp) = float2int8(_p0, _p1);
            *((int64_t*)(pp + 8)) = float2int8(_p2, _p3);
            *((int64_t*)(pp + 16)) = float2int8(_p4, _p5);
            *((int64_t*)(pp + 24)) = float2int8(_p6, _p7);
            pp += 32;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
            p4 += 4;
            p5 += 4;
            p6 += 4;
            p7 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p1[0] * scale1);
            pp[2] = gemm_float2int8(p2[0] * scale2);
            pp[3] = gemm_float2int8(p3[0] * scale3);
            pp[4] = gemm_float2int8(p4[0] * scale4);
            pp[5] = gemm_float2int8(p5[0] * scale5);
            pp[6] = gemm_float2int8(p6[0] * scale6);
            pp[7] = gemm_float2int8(p7[0] * scale7);
            pp += 8;
            p0++;
            p1++;
            p2++;
            p3++;
            p4++;
            p5++;
            p6++;
            p7++;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;
        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];
        v4f32 _scale0 = __msa_fill_w_f32(scale0);
        v4f32 _scale1 = __msa_fill_w_f32(scale1);
        v4f32 _scale2 = __msa_fill_w_f32(scale2);
        v4f32 _scale3 = __msa_fill_w_f32(scale3);

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + 16);
            __builtin_prefetch(p1 + 16);
            __builtin_prefetch(p2 + 16);
            __builtin_prefetch(p3 + 16);
            v4f32 _p0 = __msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _scale0);
            v4f32 _p1 = __msa_fmul_w((v4f32)__msa_ld_w(p1, 0), _scale1);
            v4f32 _p2 = __msa_fmul_w((v4f32)__msa_ld_w(p2, 0), _scale2);
            v4f32 _p3 = __msa_fmul_w((v4f32)__msa_ld_w(p3, 0), _scale3);

            *((int64_t*)pp) = float2int8(_p0, _p1);
            *((int64_t*)(pp + 8)) = float2int8(_p2, _p3);
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p1[0] * scale1);
            pp[2] = gemm_float2int8(p2[0] * scale2);
            pp[3] = gemm_float2int8(p3[0] * scale3);
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p0[1] * scale0);
            pp[2] = gemm_float2int8(p0[2] * scale0);
            pp[3] = gemm_float2int8(p0[3] * scale0);
            pp[4] = gemm_float2int8(p1[0] * scale1);
            pp[5] = gemm_float2int8(p1[1] * scale1);
            pp[6] = gemm_float2int8(p1[2] * scale1);
            pp[7] = gemm_float2int8(p1[3] * scale1);
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p1[0] * scale1);
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float scale0 = scales[i + ii];

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p0[1] * scale0);
            pp[2] = gemm_float2int8(p0[2] * scale0);
            pp[3] = gemm_float2int8(p0[3] * scale0);
            pp += 4;
            p0 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_A_tile_fp32_to_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    signed char* pp = AT;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];
        const float scale4 = scales[i + ii + 4];
        const float scale5 = scales[i + ii + 5];
        const float scale6 = scales[i + ii + 6];
        const float scale7 = scales[i + ii + 7];
        v4f32 _scale0 = __msa_fill_w_f32(scale0);
        v4f32 _scale1 = __msa_fill_w_f32(scale1);
        v4f32 _scale2 = __msa_fill_w_f32(scale2);
        v4f32 _scale3 = __msa_fill_w_f32(scale3);
        v4f32 _scale4 = __msa_fill_w_f32(scale4);
        v4f32 _scale5 = __msa_fill_w_f32(scale5);
        v4f32 _scale6 = __msa_fill_w_f32(scale6);
        v4f32 _scale7 = __msa_fill_w_f32(scale7);

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + A_hstep * 4);
            const float* p1 = p0 + A_hstep;
            const float* p2 = p1 + A_hstep;
            const float* p3 = p2 + A_hstep;
            v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
            v4f32 _p1 = (v4f32)__msa_ld_w(p1, 0);
            v4f32 _p2 = (v4f32)__msa_ld_w(p2, 0);
            v4f32 _p3 = (v4f32)__msa_ld_w(p3, 0);
            transpose4x4_ps(_p0, _p1, _p2, _p3);
            _p0 = __msa_fmul_w(_p0, _scale0);
            _p1 = __msa_fmul_w(_p1, _scale1);
            _p2 = __msa_fmul_w(_p2, _scale2);
            _p3 = __msa_fmul_w(_p3, _scale3);

            v4f32 _p4 = (v4f32)__msa_ld_w(p0 + 4, 0);
            v4f32 _p5 = (v4f32)__msa_ld_w(p1 + 4, 0);
            v4f32 _p6 = (v4f32)__msa_ld_w(p2 + 4, 0);
            v4f32 _p7 = (v4f32)__msa_ld_w(p3 + 4, 0);
            transpose4x4_ps(_p4, _p5, _p6, _p7);
            _p4 = __msa_fmul_w(_p4, _scale4);
            _p5 = __msa_fmul_w(_p5, _scale5);
            _p6 = __msa_fmul_w(_p6, _scale6);
            _p7 = __msa_fmul_w(_p7, _scale7);

            *((int64_t*)pp) = float2int8(_p0, _p1);
            *((int64_t*)(pp + 8)) = float2int8(_p2, _p3);
            *((int64_t*)(pp + 16)) = float2int8(_p4, _p5);
            *((int64_t*)(pp + 24)) = float2int8(_p6, _p7);
            pp += 32;
            p0 += A_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + A_hstep);
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p0[1] * scale1);
            pp[2] = gemm_float2int8(p0[2] * scale2);
            pp[3] = gemm_float2int8(p0[3] * scale3);
            pp[4] = gemm_float2int8(p0[4] * scale4);
            pp[5] = gemm_float2int8(p0[5] * scale5);
            pp[6] = gemm_float2int8(p0[6] * scale6);
            pp[7] = gemm_float2int8(p0[7] * scale7);
            pp += 8;
            p0 += A_hstep;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];
        const float scale2 = scales[i + ii + 2];
        const float scale3 = scales[i + ii + 3];
        v4f32 _scale0 = __msa_fill_w_f32(scale0);
        v4f32 _scale1 = __msa_fill_w_f32(scale1);
        v4f32 _scale2 = __msa_fill_w_f32(scale2);
        v4f32 _scale3 = __msa_fill_w_f32(scale3);

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + A_hstep * 4);
            const float* p1 = p0 + A_hstep;
            const float* p2 = p1 + A_hstep;
            const float* p3 = p2 + A_hstep;
            v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
            v4f32 _p1 = (v4f32)__msa_ld_w(p1, 0);
            v4f32 _p2 = (v4f32)__msa_ld_w(p2, 0);
            v4f32 _p3 = (v4f32)__msa_ld_w(p3, 0);
            transpose4x4_ps(_p0, _p1, _p2, _p3);
            _p0 = __msa_fmul_w(_p0, _scale0);
            _p1 = __msa_fmul_w(_p1, _scale1);
            _p2 = __msa_fmul_w(_p2, _scale2);
            _p3 = __msa_fmul_w(_p3, _scale3);

            *((int64_t*)pp) = float2int8(_p0, _p1);
            *((int64_t*)(pp + 8)) = float2int8(_p2, _p3);
            pp += 16;
            p0 += A_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + A_hstep);
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p0[1] * scale1);
            pp[2] = gemm_float2int8(p0[2] * scale2);
            pp[3] = gemm_float2int8(p0[3] * scale3);
            pp += 4;
            p0 += A_hstep;
        }
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
        const float scale0 = scales[i + ii];
        const float scale1 = scales[i + ii + 1];

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + A_hstep * 4);
            const float* p1 = p0 + A_hstep;
            const float* p2 = p1 + A_hstep;
            const float* p3 = p2 + A_hstep;
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p1[0] * scale0);
            pp[2] = gemm_float2int8(p2[0] * scale0);
            pp[3] = gemm_float2int8(p3[0] * scale0);
            pp[4] = gemm_float2int8(p0[1] * scale1);
            pp[5] = gemm_float2int8(p1[1] * scale1);
            pp[6] = gemm_float2int8(p2[1] * scale1);
            pp[7] = gemm_float2int8(p3[1] * scale1);
            pp += 8;
            p0 += A_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + A_hstep);
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p0[1] * scale1);
            pp += 2;
            p0 += A_hstep;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const float* p0 = (const float*)A + (size_t)k * A_hstep + i + ii;
        const float scale0 = scales[i + ii];

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + A_hstep * 4);
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp[1] = gemm_float2int8(p0[A_hstep] * scale0);
            pp[2] = gemm_float2int8(p0[A_hstep * 2] * scale0);
            pp[3] = gemm_float2int8(p0[A_hstep * 3] * scale0);
            pp += 4;
            p0 += A_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + A_hstep);
            pp[0] = gemm_float2int8(p0[0] * scale0);
            pp += 1;
            p0 += A_hstep;
        }
    }
}

static void compute_B_int8_scale(const Mat& B, float& scale)
{
    float absmax = 0.f;

    const int H = B.dims == 3 ? B.c : B.h;
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    for (int y = 0; y < H; y++)
    {
        const float* ptr = (const float*)B + y * B_hstep;
        for (int x = 0; x < B.w; x++)
        {
            absmax = std::max(absmax, (float)fabs(ptr[x]));
        }
    }

    scale = absmax == 0.f ? 1.f : 127.f / absmax;
}

static void pack_B_tile_fp32_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    signed char* pp = BT;

    int jj = 0;
#if __mips_msa
    v4f32 _scale = __msa_fill_w_f32(scale);

    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
        const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
        const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;
        const float* p4 = (const float*)B + (j + jj + 4) * B_hstep + k;
        const float* p5 = (const float*)B + (j + jj + 5) * B_hstep + k;
        const float* p6 = (const float*)B + (j + jj + 6) * B_hstep + k;
        const float* p7 = (const float*)B + (j + jj + 7) * B_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + 16);
            __builtin_prefetch(p1 + 16);
            __builtin_prefetch(p2 + 16);
            __builtin_prefetch(p3 + 16);
            __builtin_prefetch(p4 + 16);
            __builtin_prefetch(p5 + 16);
            __builtin_prefetch(p6 + 16);
            __builtin_prefetch(p7 + 16);
            v4f32 _p0 = __msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _scale);
            v4f32 _p1 = __msa_fmul_w((v4f32)__msa_ld_w(p1, 0), _scale);
            v4f32 _p2 = __msa_fmul_w((v4f32)__msa_ld_w(p2, 0), _scale);
            v4f32 _p3 = __msa_fmul_w((v4f32)__msa_ld_w(p3, 0), _scale);
            v4f32 _p4 = __msa_fmul_w((v4f32)__msa_ld_w(p4, 0), _scale);
            v4f32 _p5 = __msa_fmul_w((v4f32)__msa_ld_w(p5, 0), _scale);
            v4f32 _p6 = __msa_fmul_w((v4f32)__msa_ld_w(p6, 0), _scale);
            v4f32 _p7 = __msa_fmul_w((v4f32)__msa_ld_w(p7, 0), _scale);

            *((int64_t*)pp) = float2int8(_p0, _p1);
            *((int64_t*)(pp + 8)) = float2int8(_p2, _p3);
            *((int64_t*)(pp + 16)) = float2int8(_p4, _p5);
            *((int64_t*)(pp + 24)) = float2int8(_p6, _p7);
            pp += 32;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
            p4 += 4;
            p5 += 4;
            p6 += 4;
            p7 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p1[0] * scale);
            pp[2] = gemm_float2int8(p2[0] * scale);
            pp[3] = gemm_float2int8(p3[0] * scale);
            pp[4] = gemm_float2int8(p4[0] * scale);
            pp[5] = gemm_float2int8(p5[0] * scale);
            pp[6] = gemm_float2int8(p6[0] * scale);
            pp[7] = gemm_float2int8(p7[0] * scale);
            pp += 8;
            p0++;
            p1++;
            p2++;
            p3++;
            p4++;
            p5++;
            p6++;
            p7++;
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
        const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
        const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + 16);
            __builtin_prefetch(p1 + 16);
            __builtin_prefetch(p2 + 16);
            __builtin_prefetch(p3 + 16);
            v4f32 _p0 = __msa_fmul_w((v4f32)__msa_ld_w(p0, 0), _scale);
            v4f32 _p1 = __msa_fmul_w((v4f32)__msa_ld_w(p1, 0), _scale);
            v4f32 _p2 = __msa_fmul_w((v4f32)__msa_ld_w(p2, 0), _scale);
            v4f32 _p3 = __msa_fmul_w((v4f32)__msa_ld_w(p3, 0), _scale);

            *((int64_t*)pp) = float2int8(_p0, _p1);
            *((int64_t*)(pp + 8)) = float2int8(_p2, _p3);
            pp += 16;
            p0 += 4;
            p1 += 4;
            p2 += 4;
            p3 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p1[0] * scale);
            pp[2] = gemm_float2int8(p2[0] * scale);
            pp[3] = gemm_float2int8(p3[0] * scale);
            pp += 4;
            p0++;
            p1++;
            p2++;
            p3++;
        }
    }
#endif // __mips_msa
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p0[1] * scale);
            pp[2] = gemm_float2int8(p0[2] * scale);
            pp[3] = gemm_float2int8(p0[3] * scale);
            pp[4] = gemm_float2int8(p1[0] * scale);
            pp[5] = gemm_float2int8(p1[1] * scale);
            pp[6] = gemm_float2int8(p1[2] * scale);
            pp[7] = gemm_float2int8(p1[3] * scale);
            pp += 8;
            p0 += 4;
            p1 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p1[0] * scale);
            pp += 2;
            p0++;
            p1++;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p0[1] * scale);
            pp[2] = gemm_float2int8(p0[2] * scale);
            pp[3] = gemm_float2int8(p0[3] * scale);
            pp += 4;
            p0 += 4;
        }
        for (; kk < max_kk; kk++)
        {
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp += 1;
            p0++;
        }
    }
}

static void transpose_pack_B_tile_fp32_to_int8(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    const size_t B_hstep = B.dims == 3 ? B.cstep : (size_t)B.w;

    signed char* pp = BT;

    int jj = 0;
#if __mips_msa
    v4f32 _scale = __msa_fill_w_f32(scale);

    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + B_hstep * 4);
            const float* p1 = p0 + B_hstep;
            const float* p2 = p1 + B_hstep;
            const float* p3 = p2 + B_hstep;
            v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
            v4f32 _p1 = (v4f32)__msa_ld_w(p1, 0);
            v4f32 _p2 = (v4f32)__msa_ld_w(p2, 0);
            v4f32 _p3 = (v4f32)__msa_ld_w(p3, 0);
            transpose4x4_ps(_p0, _p1, _p2, _p3);
            _p0 = __msa_fmul_w(_p0, _scale);
            _p1 = __msa_fmul_w(_p1, _scale);
            _p2 = __msa_fmul_w(_p2, _scale);
            _p3 = __msa_fmul_w(_p3, _scale);

            v4f32 _p4 = (v4f32)__msa_ld_w(p0 + 4, 0);
            v4f32 _p5 = (v4f32)__msa_ld_w(p1 + 4, 0);
            v4f32 _p6 = (v4f32)__msa_ld_w(p2 + 4, 0);
            v4f32 _p7 = (v4f32)__msa_ld_w(p3 + 4, 0);
            transpose4x4_ps(_p4, _p5, _p6, _p7);
            _p4 = __msa_fmul_w(_p4, _scale);
            _p5 = __msa_fmul_w(_p5, _scale);
            _p6 = __msa_fmul_w(_p6, _scale);
            _p7 = __msa_fmul_w(_p7, _scale);

            *((int64_t*)pp) = float2int8(_p0, _p1);
            *((int64_t*)(pp + 8)) = float2int8(_p2, _p3);
            *((int64_t*)(pp + 16)) = float2int8(_p4, _p5);
            *((int64_t*)(pp + 24)) = float2int8(_p6, _p7);
            pp += 32;
            p0 += B_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + B_hstep);
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p0[1] * scale);
            pp[2] = gemm_float2int8(p0[2] * scale);
            pp[3] = gemm_float2int8(p0[3] * scale);
            pp[4] = gemm_float2int8(p0[4] * scale);
            pp[5] = gemm_float2int8(p0[5] * scale);
            pp[6] = gemm_float2int8(p0[6] * scale);
            pp[7] = gemm_float2int8(p0[7] * scale);
            pp += 8;
            p0 += B_hstep;
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + B_hstep * 4);
            const float* p1 = p0 + B_hstep;
            const float* p2 = p1 + B_hstep;
            const float* p3 = p2 + B_hstep;
            v4f32 _p0 = (v4f32)__msa_ld_w(p0, 0);
            v4f32 _p1 = (v4f32)__msa_ld_w(p1, 0);
            v4f32 _p2 = (v4f32)__msa_ld_w(p2, 0);
            v4f32 _p3 = (v4f32)__msa_ld_w(p3, 0);
            transpose4x4_ps(_p0, _p1, _p2, _p3);
            _p0 = __msa_fmul_w(_p0, _scale);
            _p1 = __msa_fmul_w(_p1, _scale);
            _p2 = __msa_fmul_w(_p2, _scale);
            _p3 = __msa_fmul_w(_p3, _scale);

            *((int64_t*)pp) = float2int8(_p0, _p1);
            *((int64_t*)(pp + 8)) = float2int8(_p2, _p3);
            pp += 16;
            p0 += B_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + B_hstep);
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p0[1] * scale);
            pp[2] = gemm_float2int8(p0[2] * scale);
            pp[3] = gemm_float2int8(p0[3] * scale);
            pp += 4;
            p0 += B_hstep;
        }
    }
#endif // __mips_msa
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + B_hstep * 4);
            const float* p1 = p0 + B_hstep;
            const float* p2 = p1 + B_hstep;
            const float* p3 = p2 + B_hstep;
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p1[0] * scale);
            pp[2] = gemm_float2int8(p2[0] * scale);
            pp[3] = gemm_float2int8(p3[0] * scale);
            pp[4] = gemm_float2int8(p0[1] * scale);
            pp[5] = gemm_float2int8(p1[1] * scale);
            pp[6] = gemm_float2int8(p2[1] * scale);
            pp[7] = gemm_float2int8(p3[1] * scale);
            pp += 8;
            p0 += B_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + B_hstep);
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p0[1] * scale);
            pp += 2;
            p0 += B_hstep;
        }
    }
    for (; jj < max_jj; jj++)
    {
        const float* p0 = (const float*)B + (size_t)k * B_hstep + j + jj;

        int kk = 0;
        for (; kk + 3 < max_kk; kk += 4)
        {
            __builtin_prefetch(p0 + B_hstep * 4);
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp[1] = gemm_float2int8(p0[B_hstep] * scale);
            pp[2] = gemm_float2int8(p0[B_hstep * 2] * scale);
            pp[3] = gemm_float2int8(p0[B_hstep * 3] * scale);
            pp += 4;
            p0 += B_hstep * 4;
        }
        for (; kk < max_kk; kk++)
        {
            __builtin_prefetch(p0 + B_hstep);
            pp[0] = gemm_float2int8(p0[0] * scale);
            pp += 1;
            p0 += B_hstep;
        }
    }
}

static void gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
    // NCNN_LOGE("gemm_transB_packed_tile_int8 %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const signed char* pAT = AT_tile;
    const signed char* pBT = BT_tile;

    int* outptr = topT_tile;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pB = pBT;
        const v8i16 _one = __msa_fill_h(1);

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
#if NCNN_GNU_INLINE_ASM
            const signed char* pA = pAT;
            int nn = max_kk;

            asm volatile(
                "ldi.h          $w24, 1                         \n"

                "beq            %4, $0, 0f                      \n"
                "nop                                             \n"

                "ld.w           $w0, 0(%3)                      \n"
                "ld.w           $w1, 16(%3)                     \n"
                "ld.w           $w2, 32(%3)                     \n"
                "ld.w           $w3, 48(%3)                     \n"
                "ld.w           $w4, 64(%3)                     \n"
                "ld.w           $w5, 80(%3)                     \n"
                "ld.w           $w6, 96(%3)                     \n"
                "ld.w           $w7, 112(%3)                    \n"
                "ld.w           $w8, 128(%3)                    \n"
                "ld.w           $w9, 144(%3)                    \n"
                "ld.w           $w10, 160(%3)                   \n"
                "ld.w           $w11, 176(%3)                   \n"
                "ld.w           $w12, 192(%3)                   \n"
                "ld.w           $w13, 208(%3)                   \n"
                "ld.w           $w14, 224(%3)                   \n"
                "ld.w           $w15, 240(%3)                   \n"
                "beq            $0, $0, 1f                      \n"
                "nop                                             \n"

                "0:                                              \n"
                "ldi.w          $w0, 0                          \n"
                "ldi.w          $w1, 0                          \n"
                "ldi.w          $w2, 0                          \n"
                "ldi.w          $w3, 0                          \n"
                "ldi.w          $w4, 0                          \n"
                "ldi.w          $w5, 0                          \n"
                "ldi.w          $w6, 0                          \n"
                "ldi.w          $w7, 0                          \n"
                "ldi.w          $w8, 0                          \n"
                "ldi.w          $w9, 0                          \n"
                "ldi.w          $w10, 0                         \n"
                "ldi.w          $w11, 0                         \n"
                "ldi.w          $w12, 0                         \n"
                "ldi.w          $w13, 0                         \n"
                "ldi.w          $w14, 0                         \n"
                "ldi.w          $w15, 0                         \n"

                "1:                                              \n"
                "slti           $8, %2, 4                       \n"
                "bne            $8, $0, 3f                      \n"
                "nop                                             \n"

                "2:                                              \n"
                "pref           6, 64(%0)                       \n"
                "pref           6, 64(%1)                       \n"
                "ld.b           $w16, 0(%0)                     \n"
                "ld.b           $w17, 16(%0)                    \n"
                "ld.b           $w20, 0(%1)                     \n"
                "ld.b           $w21, 16(%1)                    \n"
                "shf.w          $w18, $w16, 0x4e                \n"
                "shf.w          $w19, $w17, 0x4e                \n"
                "shf.w          $w22, $w20, 0x39                \n"
                "shf.w          $w23, $w21, 0x39                \n"
                "dotp_s.h       $w25, $w16, $w20                \n"
                "dotp_s.h       $w26, $w17, $w20                \n"
                "dotp_s.h       $w27, $w18, $w20                \n"
                "dotp_s.h       $w28, $w19, $w20                \n"
                "dotp_s.h       $w29, $w16, $w22                \n"
                "dotp_s.h       $w30, $w17, $w22                \n"
                "dotp_s.h       $w31, $w18, $w22                \n"
                "dotp_s.h       $w20, $w19, $w22                \n"
#if __mips64
                "daddiu         %0, %0, 32                      \n"
                "daddiu         %1, %1, 32                      \n"
#else
                "addiu          %0, %0, 32                      \n"
                "addiu          %1, %1, 32                      \n"
#endif
                "dpadd_s.w      $w0, $w25, $w24                 \n"
                "dpadd_s.w      $w1, $w26, $w24                 \n"
                "dpadd_s.w      $w4, $w27, $w24                 \n"
                "dpadd_s.w      $w5, $w28, $w24                 \n"
                "dpadd_s.w      $w2, $w29, $w24                 \n"
                "dpadd_s.w      $w3, $w30, $w24                 \n"
                "dpadd_s.w      $w6, $w31, $w24                 \n"
                "dpadd_s.w      $w7, $w20, $w24                 \n"
                "dotp_s.h       $w25, $w16, $w21                \n"
                "dotp_s.h       $w26, $w17, $w21                \n"
                "dotp_s.h       $w27, $w18, $w21                \n"
                "dotp_s.h       $w28, $w19, $w21                \n"
                "dotp_s.h       $w29, $w16, $w23                \n"
                "dotp_s.h       $w30, $w17, $w23                \n"
                "dotp_s.h       $w31, $w18, $w23                \n"
                "dotp_s.h       $w21, $w19, $w23                \n"
                "addiu          %2, %2, -4                      \n"
                "slti           $8, %2, 4                       \n"
                "dpadd_s.w      $w8, $w25, $w24                 \n"
                "dpadd_s.w      $w9, $w26, $w24                 \n"
                "dpadd_s.w      $w12, $w27, $w24                \n"
                "dpadd_s.w      $w13, $w28, $w24                \n"
                "dpadd_s.w      $w10, $w29, $w24                \n"
                "dpadd_s.w      $w11, $w30, $w24                \n"
                "dpadd_s.w      $w14, $w31, $w24                \n"
                "dpadd_s.w      $w15, $w21, $w24                \n"
                "beq            $8, $0, 2b                      \n"
                "nop                                             \n"

                "3:                                              \n"
                "beq            %2, $0, 5f                      \n"
                "nop                                             \n"

                "4:                                              \n"
                "lw             $9, 0(%0)                       \n"
                "lw             $10, 4(%0)                      \n"
                "lw             $11, 0(%1)                      \n"
                "lw             $12, 4(%1)                      \n"
                "fill.d         $w16, $9                         \n"
                "fill.d         $w17, $10                        \n"
                "fill.d         $w20, $11                        \n"
                "fill.d         $w21, $12                        \n"
                "clti_s.b       $w24, $w16, 0                   \n"
                "clti_s.b       $w25, $w17, 0                   \n"
                "clti_s.b       $w26, $w20, 0                   \n"
                "clti_s.b       $w27, $w21, 0                   \n"
                "ilvr.b         $w16, $w24, $w16                \n"
                "ilvr.b         $w17, $w25, $w17                \n"
                "ilvr.b         $w20, $w26, $w20                \n"
                "ilvr.b         $w21, $w27, $w21                \n"
                "shf.h          $w18, $w16, 0x4e                \n"
                "shf.h          $w19, $w17, 0x4e                \n"
                "shf.h          $w22, $w20, 0x39                \n"
                "shf.h          $w23, $w21, 0x39                \n"
                "mulv.h         $w24, $w16, $w20                \n"
                "mulv.h         $w25, $w17, $w20                \n"
                "mulv.h         $w26, $w18, $w20                \n"
                "mulv.h         $w27, $w19, $w20                \n"
                "clti_s.h       $w28, $w24, 0                   \n"
                "clti_s.h       $w29, $w25, 0                   \n"
                "clti_s.h       $w30, $w26, 0                   \n"
                "clti_s.h       $w31, $w27, 0                   \n"
                "ilvr.h         $w24, $w28, $w24                \n"
                "ilvr.h         $w25, $w29, $w25                \n"
                "ilvr.h         $w26, $w30, $w26                \n"
                "ilvr.h         $w27, $w31, $w27                \n"
                "addv.w         $w0, $w0, $w24                  \n"
                "addv.w         $w1, $w1, $w25                  \n"
                "addv.w         $w4, $w4, $w26                  \n"
                "addv.w         $w5, $w5, $w27                  \n"
                "mulv.h         $w24, $w16, $w22                \n"
                "mulv.h         $w25, $w17, $w22                \n"
                "mulv.h         $w26, $w18, $w22                \n"
                "mulv.h         $w27, $w19, $w22                \n"
                "clti_s.h       $w28, $w24, 0                   \n"
                "clti_s.h       $w29, $w25, 0                   \n"
                "clti_s.h       $w30, $w26, 0                   \n"
                "clti_s.h       $w31, $w27, 0                   \n"
                "ilvr.h         $w24, $w28, $w24                \n"
                "ilvr.h         $w25, $w29, $w25                \n"
                "ilvr.h         $w26, $w30, $w26                \n"
                "ilvr.h         $w27, $w31, $w27                \n"
                "addv.w         $w2, $w2, $w24                  \n"
                "addv.w         $w3, $w3, $w25                  \n"
                "addv.w         $w6, $w6, $w26                  \n"
                "addv.w         $w7, $w7, $w27                  \n"
                "mulv.h         $w24, $w16, $w21                \n"
                "mulv.h         $w25, $w17, $w21                \n"
                "mulv.h         $w26, $w18, $w21                \n"
                "mulv.h         $w27, $w19, $w21                \n"
                "clti_s.h       $w28, $w24, 0                   \n"
                "clti_s.h       $w29, $w25, 0                   \n"
                "clti_s.h       $w30, $w26, 0                   \n"
                "clti_s.h       $w31, $w27, 0                   \n"
                "ilvr.h         $w24, $w28, $w24                \n"
                "ilvr.h         $w25, $w29, $w25                \n"
                "ilvr.h         $w26, $w30, $w26                \n"
                "ilvr.h         $w27, $w31, $w27                \n"
                "addv.w         $w8, $w8, $w24                  \n"
                "addv.w         $w9, $w9, $w25                  \n"
                "addv.w         $w12, $w12, $w26                \n"
                "addv.w         $w13, $w13, $w27                \n"
                "mulv.h         $w24, $w16, $w23                \n"
                "mulv.h         $w25, $w17, $w23                \n"
                "mulv.h         $w26, $w18, $w23                \n"
                "mulv.h         $w27, $w19, $w23                \n"
#if __mips64
                "daddiu         %0, %0, 8                       \n"
                "daddiu         %1, %1, 8                       \n"
#else
                "addiu          %0, %0, 8                       \n"
                "addiu          %1, %1, 8                       \n"
#endif
                "clti_s.h       $w28, $w24, 0                   \n"
                "clti_s.h       $w29, $w25, 0                   \n"
                "clti_s.h       $w30, $w26, 0                   \n"
                "clti_s.h       $w31, $w27, 0                   \n"
                "ilvr.h         $w24, $w28, $w24                \n"
                "ilvr.h         $w25, $w29, $w25                \n"
                "ilvr.h         $w26, $w30, $w26                \n"
                "ilvr.h         $w27, $w31, $w27                \n"
                "addv.w         $w10, $w10, $w24                \n"
                "addv.w         $w11, $w11, $w25                \n"
                "addv.w         $w14, $w14, $w26                \n"
                "addv.w         $w15, $w15, $w27                \n"
                "addiu          %2, %2, -1                      \n"
                "bne            %2, $0, 4b                      \n"
                "nop                                             \n"

                "5:                                              \n"
                "st.w           $w0, 0(%3)                      \n"
                "st.w           $w1, 16(%3)                     \n"
                "st.w           $w2, 32(%3)                     \n"
                "st.w           $w3, 48(%3)                     \n"
                "st.w           $w4, 64(%3)                     \n"
                "st.w           $w5, 80(%3)                     \n"
                "st.w           $w6, 96(%3)                     \n"
                "st.w           $w7, 112(%3)                    \n"
                "st.w           $w8, 128(%3)                    \n"
                "st.w           $w9, 144(%3)                    \n"
                "st.w           $w10, 160(%3)                   \n"
                "st.w           $w11, 176(%3)                   \n"
                "st.w           $w12, 192(%3)                   \n"
                "st.w           $w13, 208(%3)                   \n"
                "st.w           $w14, 224(%3)                   \n"
                "st.w           $w15, 240(%3)                   \n"
                : "+r"(pA), "+r"(pB), "+r"(nn)
                : "r"(outptr), "r"(k)
                : "memory", "$8", "$9", "$10", "$11", "$12",
                  "$f0", "$f1", "$f2", "$f3", "$f4", "$f5", "$f6", "$f7",
                  "$f8", "$f9", "$f10", "$f11", "$f12", "$f13", "$f14", "$f15",
                  "$f16", "$f17", "$f18", "$f19", "$f20", "$f21", "$f22", "$f23",
                  "$f24", "$f25", "$f26", "$f27", "$f28", "$f29", "$f30", "$f31");

            outptr += 64;
#else
            const signed char* pA = pAT;

            v4i32 _sum00;
            v4i32 _sum01;
            v4i32 _sum10;
            v4i32 _sum11;
            v4i32 _sum20;
            v4i32 _sum21;
            v4i32 _sum30;
            v4i32 _sum31;
            v4i32 _sum40;
            v4i32 _sum41;
            v4i32 _sum50;
            v4i32 _sum51;
            v4i32 _sum60;
            v4i32 _sum61;
            v4i32 _sum70;
            v4i32 _sum71;

            if (k == 0)
            {
                _sum00 = __msa_fill_w(0);
                _sum01 = __msa_fill_w(0);
                _sum10 = __msa_fill_w(0);
                _sum11 = __msa_fill_w(0);
                _sum20 = __msa_fill_w(0);
                _sum21 = __msa_fill_w(0);
                _sum30 = __msa_fill_w(0);
                _sum31 = __msa_fill_w(0);
                _sum40 = __msa_fill_w(0);
                _sum41 = __msa_fill_w(0);
                _sum50 = __msa_fill_w(0);
                _sum51 = __msa_fill_w(0);
                _sum60 = __msa_fill_w(0);
                _sum61 = __msa_fill_w(0);
                _sum70 = __msa_fill_w(0);
                _sum71 = __msa_fill_w(0);
            }
            else
            {
                _sum00 = __msa_ld_w(outptr, 0);
                _sum01 = __msa_ld_w(outptr + 4, 0);
                _sum10 = __msa_ld_w(outptr + 8, 0);
                _sum11 = __msa_ld_w(outptr + 12, 0);
                _sum20 = __msa_ld_w(outptr + 16, 0);
                _sum21 = __msa_ld_w(outptr + 20, 0);
                _sum30 = __msa_ld_w(outptr + 24, 0);
                _sum31 = __msa_ld_w(outptr + 28, 0);
                _sum40 = __msa_ld_w(outptr + 32, 0);
                _sum41 = __msa_ld_w(outptr + 36, 0);
                _sum50 = __msa_ld_w(outptr + 40, 0);
                _sum51 = __msa_ld_w(outptr + 44, 0);
                _sum60 = __msa_ld_w(outptr + 48, 0);
                _sum61 = __msa_ld_w(outptr + 52, 0);
                _sum70 = __msa_ld_w(outptr + 56, 0);
                _sum71 = __msa_ld_w(outptr + 60, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(pA + 64);
                __builtin_prefetch(pB + 64);
                v16i8 _pA0 = __msa_ld_b(pA, 0);
                v16i8 _pA1 = __msa_ld_b(pA + 16, 0);
                v16i8 _pA0r = (v16i8)__msa_shf_w((v4i32)_pA0, _MSA_SHUFFLE(1, 0, 3, 2));
                v16i8 _pA1r = (v16i8)__msa_shf_w((v4i32)_pA1, _MSA_SHUFFLE(1, 0, 3, 2));
                v16i8 _pB0 = __msa_ld_b(pB, 0);
                v16i8 _pB1 = __msa_ld_b(pB + 16, 0);
                v16i8 _pB0r = (v16i8)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                v16i8 _pB1r = (v16i8)__msa_shf_w((v4i32)_pB1, _MSA_SHUFFLE(0, 3, 2, 1));

                _sum00 = __msa_dpadd_s_w(_sum00, __msa_dotp_s_h(_pA0, _pB0), _one);
                _sum01 = __msa_dpadd_s_w(_sum01, __msa_dotp_s_h(_pA1, _pB0), _one);
                _sum10 = __msa_dpadd_s_w(_sum10, __msa_dotp_s_h(_pA0, _pB0r), _one);
                _sum11 = __msa_dpadd_s_w(_sum11, __msa_dotp_s_h(_pA1, _pB0r), _one);
                _sum20 = __msa_dpadd_s_w(_sum20, __msa_dotp_s_h(_pA0r, _pB0), _one);
                _sum21 = __msa_dpadd_s_w(_sum21, __msa_dotp_s_h(_pA1r, _pB0), _one);
                _sum30 = __msa_dpadd_s_w(_sum30, __msa_dotp_s_h(_pA0r, _pB0r), _one);
                _sum31 = __msa_dpadd_s_w(_sum31, __msa_dotp_s_h(_pA1r, _pB0r), _one);
                _sum40 = __msa_dpadd_s_w(_sum40, __msa_dotp_s_h(_pA0, _pB1), _one);
                _sum41 = __msa_dpadd_s_w(_sum41, __msa_dotp_s_h(_pA1, _pB1), _one);
                _sum50 = __msa_dpadd_s_w(_sum50, __msa_dotp_s_h(_pA0, _pB1r), _one);
                _sum51 = __msa_dpadd_s_w(_sum51, __msa_dotp_s_h(_pA1, _pB1r), _one);
                _sum60 = __msa_dpadd_s_w(_sum60, __msa_dotp_s_h(_pA0r, _pB1), _one);
                _sum61 = __msa_dpadd_s_w(_sum61, __msa_dotp_s_h(_pA1r, _pB1), _one);
                _sum70 = __msa_dpadd_s_w(_sum70, __msa_dotp_s_h(_pA0r, _pB1r), _one);
                _sum71 = __msa_dpadd_s_w(_sum71, __msa_dotp_s_h(_pA1r, _pB1r), _one);

                pA += 32;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                v8i16 _pA0 = (v8i16)__msa_fill_d(*(int*)pA);
                _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA0, 0), (v16i8)_pA0);
                v8i16 _pA1 = (v8i16)__msa_fill_d(*(int*)(pA + 4));
                _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA1, 0), (v16i8)_pA1);
                v8i16 _pA0r = __msa_shf_h(_pA0, _MSA_SHUFFLE(1, 0, 3, 2));
                v8i16 _pA1r = __msa_shf_h(_pA1, _MSA_SHUFFLE(1, 0, 3, 2));
                v8i16 _pB0 = (v8i16)__msa_fill_d(*(int*)pB);
                _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB0, 0), (v16i8)_pB0);
                v8i16 _pB1 = (v8i16)__msa_fill_d(*(int*)(pB + 4));
                _pB1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB1, 0), (v16i8)_pB1);
                v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                v8i16 _pB1r = __msa_shf_h(_pB1, _MSA_SHUFFLE(0, 3, 2, 1));

                v8i16 _s00 = __msa_mulv_h(_pA0, _pB0);
                v8i16 _s01 = __msa_mulv_h(_pA1, _pB0);
                v8i16 _s10 = __msa_mulv_h(_pA0, _pB0r);
                v8i16 _s11 = __msa_mulv_h(_pA1, _pB0r);
                v8i16 _s20 = __msa_mulv_h(_pA0r, _pB0);
                v8i16 _s21 = __msa_mulv_h(_pA1r, _pB0);
                v8i16 _s30 = __msa_mulv_h(_pA0r, _pB0r);
                v8i16 _s31 = __msa_mulv_h(_pA1r, _pB0r);
                v8i16 _s40 = __msa_mulv_h(_pA0, _pB1);
                v8i16 _s41 = __msa_mulv_h(_pA1, _pB1);
                v8i16 _s50 = __msa_mulv_h(_pA0, _pB1r);
                v8i16 _s51 = __msa_mulv_h(_pA1, _pB1r);
                v8i16 _s60 = __msa_mulv_h(_pA0r, _pB1);
                v8i16 _s61 = __msa_mulv_h(_pA1r, _pB1);
                v8i16 _s70 = __msa_mulv_h(_pA0r, _pB1r);
                v8i16 _s71 = __msa_mulv_h(_pA1r, _pB1r);
                _sum00 = __msa_addv_w(_sum00, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s00, 0), _s00));
                _sum01 = __msa_addv_w(_sum01, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s01, 0), _s01));
                _sum10 = __msa_addv_w(_sum10, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s10, 0), _s10));
                _sum11 = __msa_addv_w(_sum11, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s11, 0), _s11));
                _sum20 = __msa_addv_w(_sum20, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s20, 0), _s20));
                _sum21 = __msa_addv_w(_sum21, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s21, 0), _s21));
                _sum30 = __msa_addv_w(_sum30, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s30, 0), _s30));
                _sum31 = __msa_addv_w(_sum31, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s31, 0), _s31));
                _sum40 = __msa_addv_w(_sum40, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s40, 0), _s40));
                _sum41 = __msa_addv_w(_sum41, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s41, 0), _s41));
                _sum50 = __msa_addv_w(_sum50, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s50, 0), _s50));
                _sum51 = __msa_addv_w(_sum51, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s51, 0), _s51));
                _sum60 = __msa_addv_w(_sum60, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s60, 0), _s60));
                _sum61 = __msa_addv_w(_sum61, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s61, 0), _s61));
                _sum70 = __msa_addv_w(_sum70, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s70, 0), _s70));
                _sum71 = __msa_addv_w(_sum71, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s71, 0), _s71));

                pA += 8;
                pB += 8;
            }

            __msa_st_w(_sum00, outptr, 0);
            __msa_st_w(_sum01, outptr + 4, 0);
            __msa_st_w(_sum10, outptr + 8, 0);
            __msa_st_w(_sum11, outptr + 12, 0);
            __msa_st_w(_sum20, outptr + 16, 0);
            __msa_st_w(_sum21, outptr + 20, 0);
            __msa_st_w(_sum30, outptr + 24, 0);
            __msa_st_w(_sum31, outptr + 28, 0);
            __msa_st_w(_sum40, outptr + 32, 0);
            __msa_st_w(_sum41, outptr + 36, 0);
            __msa_st_w(_sum50, outptr + 40, 0);
            __msa_st_w(_sum51, outptr + 44, 0);
            __msa_st_w(_sum60, outptr + 48, 0);
            __msa_st_w(_sum61, outptr + 52, 0);
            __msa_st_w(_sum70, outptr + 56, 0);
            __msa_st_w(_sum71, outptr + 60, 0);

            outptr += 64;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            v4i32 _sum00;
            v4i32 _sum01;
            v4i32 _sum10;
            v4i32 _sum11;
            v4i32 _sum20;
            v4i32 _sum21;
            v4i32 _sum30;
            v4i32 _sum31;

            if (k == 0)
            {
                _sum00 = __msa_fill_w(0);
                _sum01 = __msa_fill_w(0);
                _sum10 = __msa_fill_w(0);
                _sum11 = __msa_fill_w(0);
                _sum20 = __msa_fill_w(0);
                _sum21 = __msa_fill_w(0);
                _sum30 = __msa_fill_w(0);
                _sum31 = __msa_fill_w(0);
            }
            else
            {
                _sum00 = __msa_ld_w(outptr, 0);
                _sum01 = __msa_ld_w(outptr + 4, 0);
                _sum10 = __msa_ld_w(outptr + 8, 0);
                _sum11 = __msa_ld_w(outptr + 12, 0);
                _sum20 = __msa_ld_w(outptr + 16, 0);
                _sum21 = __msa_ld_w(outptr + 20, 0);
                _sum30 = __msa_ld_w(outptr + 24, 0);
                _sum31 = __msa_ld_w(outptr + 28, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(pA + 64);
                __builtin_prefetch(pB + 32);
                v16i8 _pA0 = __msa_ld_b(pA, 0);
                v16i8 _pA1 = __msa_ld_b(pA + 16, 0);
                v16i8 _pA0r = (v16i8)__msa_shf_w((v4i32)_pA0, _MSA_SHUFFLE(1, 0, 3, 2));
                v16i8 _pA1r = (v16i8)__msa_shf_w((v4i32)_pA1, _MSA_SHUFFLE(1, 0, 3, 2));
                v16i8 _pB0 = __msa_ld_b(pB, 0);
                v16i8 _pB0r = (v16i8)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));

                _sum00 = __msa_dpadd_s_w(_sum00, __msa_dotp_s_h(_pA0, _pB0), _one);
                _sum01 = __msa_dpadd_s_w(_sum01, __msa_dotp_s_h(_pA1, _pB0), _one);
                _sum10 = __msa_dpadd_s_w(_sum10, __msa_dotp_s_h(_pA0, _pB0r), _one);
                _sum11 = __msa_dpadd_s_w(_sum11, __msa_dotp_s_h(_pA1, _pB0r), _one);
                _sum20 = __msa_dpadd_s_w(_sum20, __msa_dotp_s_h(_pA0r, _pB0), _one);
                _sum21 = __msa_dpadd_s_w(_sum21, __msa_dotp_s_h(_pA1r, _pB0), _one);
                _sum30 = __msa_dpadd_s_w(_sum30, __msa_dotp_s_h(_pA0r, _pB0r), _one);
                _sum31 = __msa_dpadd_s_w(_sum31, __msa_dotp_s_h(_pA1r, _pB0r), _one);

                pA += 32;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                v8i16 _pA0 = (v8i16)__msa_fill_d(*(int*)pA);
                _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA0, 0), (v16i8)_pA0);
                v8i16 _pA1 = (v8i16)__msa_fill_d(*(int*)(pA + 4));
                _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA1, 0), (v16i8)_pA1);
                v8i16 _pA0r = __msa_shf_h(_pA0, _MSA_SHUFFLE(1, 0, 3, 2));
                v8i16 _pA1r = __msa_shf_h(_pA1, _MSA_SHUFFLE(1, 0, 3, 2));
                v8i16 _pB0 = (v8i16)__msa_fill_d(*(int*)pB);
                _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB0, 0), (v16i8)_pB0);
                v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));

                v8i16 _s00 = __msa_mulv_h(_pA0, _pB0);
                v8i16 _s01 = __msa_mulv_h(_pA1, _pB0);
                v8i16 _s10 = __msa_mulv_h(_pA0, _pB0r);
                v8i16 _s11 = __msa_mulv_h(_pA1, _pB0r);
                v8i16 _s20 = __msa_mulv_h(_pA0r, _pB0);
                v8i16 _s21 = __msa_mulv_h(_pA1r, _pB0);
                v8i16 _s30 = __msa_mulv_h(_pA0r, _pB0r);
                v8i16 _s31 = __msa_mulv_h(_pA1r, _pB0r);
                _sum00 = __msa_addv_w(_sum00, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s00, 0), _s00));
                _sum01 = __msa_addv_w(_sum01, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s01, 0), _s01));
                _sum10 = __msa_addv_w(_sum10, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s10, 0), _s10));
                _sum11 = __msa_addv_w(_sum11, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s11, 0), _s11));
                _sum20 = __msa_addv_w(_sum20, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s20, 0), _s20));
                _sum21 = __msa_addv_w(_sum21, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s21, 0), _s21));
                _sum30 = __msa_addv_w(_sum30, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s30, 0), _s30));
                _sum31 = __msa_addv_w(_sum31, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s31, 0), _s31));

                pA += 8;
                pB += 4;
            }

            __msa_st_w(_sum00, outptr, 0);
            __msa_st_w(_sum01, outptr + 4, 0);
            __msa_st_w(_sum10, outptr + 8, 0);
            __msa_st_w(_sum11, outptr + 12, 0);
            __msa_st_w(_sum20, outptr + 16, 0);
            __msa_st_w(_sum21, outptr + 20, 0);
            __msa_st_w(_sum30, outptr + 24, 0);
            __msa_st_w(_sum31, outptr + 28, 0);

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            v4i32 _sum00;
            v4i32 _sum01;
            v4i32 _sum10;
            v4i32 _sum11;

            if (k == 0)
            {
                _sum00 = __msa_fill_w(0);
                _sum01 = __msa_fill_w(0);
                _sum10 = __msa_fill_w(0);
                _sum11 = __msa_fill_w(0);
            }
            else
            {
                _sum00 = __msa_ld_w(outptr, 0);
                _sum01 = __msa_ld_w(outptr + 4, 0);
                _sum10 = __msa_ld_w(outptr + 8, 0);
                _sum11 = __msa_ld_w(outptr + 12, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(pA + 64);
                __builtin_prefetch(pB + 16);
                v16i8 _pA0 = __msa_ld_b(pA, 0);
                v16i8 _pA1 = __msa_ld_b(pA + 16, 0);
                v16i8 _pB0 = (v16i8)__msa_fill_d(*(const int64_t*)pB);
                v16i8 _pB0r = (v16i8)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));

                _sum00 = __msa_dpadd_s_w(_sum00, __msa_dotp_s_h(_pA0, _pB0), _one);
                _sum01 = __msa_dpadd_s_w(_sum01, __msa_dotp_s_h(_pA1, _pB0), _one);
                _sum10 = __msa_dpadd_s_w(_sum10, __msa_dotp_s_h(_pA0, _pB0r), _one);
                _sum11 = __msa_dpadd_s_w(_sum11, __msa_dotp_s_h(_pA1, _pB0r), _one);

                pA += 32;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                v8i16 _pA0 = (v8i16)__msa_fill_d(*(int*)pA);
                _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA0, 0), (v16i8)_pA0);
                v8i16 _pA1 = (v8i16)__msa_fill_d(*(int*)(pA + 4));
                _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA1, 0), (v16i8)_pA1);
                int b01 = (unsigned char)pB[0] | ((unsigned char)pB[1] << 8);
                v8i16 _pB0 = (v8i16)__msa_fill_w(b01);
                _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB0, 0), (v16i8)_pB0);
                _pB0 = __msa_shf_h(_pB0, _MSA_SHUFFLE(1, 0, 1, 0));
                v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));

                v8i16 _s00 = __msa_mulv_h(_pA0, _pB0);
                v8i16 _s01 = __msa_mulv_h(_pA1, _pB0);
                v8i16 _s10 = __msa_mulv_h(_pA0, _pB0r);
                v8i16 _s11 = __msa_mulv_h(_pA1, _pB0r);
                _sum00 = __msa_addv_w(_sum00, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s00, 0), _s00));
                _sum01 = __msa_addv_w(_sum01, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s01, 0), _s01));
                _sum10 = __msa_addv_w(_sum10, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s10, 0), _s10));
                _sum11 = __msa_addv_w(_sum11, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s11, 0), _s11));

                pA += 8;
                pB += 2;
            }

            __msa_st_w(_sum00, outptr, 0);
            __msa_st_w(_sum01, outptr + 4, 0);
            __msa_st_w(_sum10, outptr + 8, 0);
            __msa_st_w(_sum11, outptr + 12, 0);

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            v4i32 _sum00;
            v4i32 _sum01;

            if (k == 0)
            {
                _sum00 = __msa_fill_w(0);
                _sum01 = __msa_fill_w(0);
            }
            else
            {
                _sum00 = __msa_ld_w(outptr, 0);
                _sum01 = __msa_ld_w(outptr + 4, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(pA + 64);
                __builtin_prefetch(pB + 16);
                v16i8 _pA0 = __msa_ld_b(pA, 0);
                v16i8 _pA1 = __msa_ld_b(pA + 16, 0);

                v16i8 _pB0 = (v16i8)__msa_fill_w(*(const int*)pB);
                _sum00 = __msa_dpadd_s_w(_sum00, __msa_dotp_s_h(_pA0, _pB0), _one);
                _sum01 = __msa_dpadd_s_w(_sum01, __msa_dotp_s_h(_pA1, _pB0), _one);

                pA += 32;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                v8i16 _pA0 = (v8i16)__msa_fill_d(*(int*)pA);
                _pA0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA0, 0), (v16i8)_pA0);
                v8i16 _pA1 = (v8i16)__msa_fill_d(*(int*)(pA + 4));
                _pA1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA1, 0), (v16i8)_pA1);

                v8i16 _pB0 = __msa_fill_h(pB[0]);
                v8i16 _s0 = __msa_mulv_h(_pA0, _pB0);
                v8i16 _s1 = __msa_mulv_h(_pA1, _pB0);
                _sum00 = __msa_addv_w(_sum00, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum01 = __msa_addv_w(_sum01, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                pA += 8;
                pB += 1;
            }

            __msa_st_w(_sum00, outptr, 0);
            __msa_st_w(_sum01, outptr + 4, 0);

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pB = pBT;
        const v8i16 _one = __msa_fill_h(1);

        int jj = 0;
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            v4i32 _sum0;
            v4i32 _sum1;
            v4i32 _sum2;
            v4i32 _sum3;
            v4i32 _sum4;
            v4i32 _sum5;
            v4i32 _sum6;
            v4i32 _sum7;

            if (k == 0)
            {
                _sum0 = __msa_fill_w(0);
                _sum1 = __msa_fill_w(0);
                _sum2 = __msa_fill_w(0);
                _sum3 = __msa_fill_w(0);
                _sum4 = __msa_fill_w(0);
                _sum5 = __msa_fill_w(0);
                _sum6 = __msa_fill_w(0);
                _sum7 = __msa_fill_w(0);
            }
            else
            {
                _sum0 = __msa_ld_w(outptr, 0);
                _sum1 = __msa_ld_w(outptr + 4, 0);
                _sum2 = __msa_ld_w(outptr + 8, 0);
                _sum3 = __msa_ld_w(outptr + 12, 0);
                _sum4 = __msa_ld_w(outptr + 16, 0);
                _sum5 = __msa_ld_w(outptr + 20, 0);
                _sum6 = __msa_ld_w(outptr + 24, 0);
                _sum7 = __msa_ld_w(outptr + 28, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(pA + 64);
                __builtin_prefetch(pB + 64);
                v16i8 _pA = __msa_ld_b(pA, 0);
                v16i8 _pAr = (v16i8)__msa_shf_w((v4i32)_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                v16i8 _pB0 = __msa_ld_b(pB, 0);
                v16i8 _pB1 = __msa_ld_b(pB + 16, 0);
                v16i8 _pB0r = (v16i8)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                v16i8 _pB1r = (v16i8)__msa_shf_w((v4i32)_pB1, _MSA_SHUFFLE(0, 3, 2, 1));

                _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB0r), _one);
                _sum2 = __msa_dpadd_s_w(_sum2, __msa_dotp_s_h(_pAr, _pB0), _one);
                _sum3 = __msa_dpadd_s_w(_sum3, __msa_dotp_s_h(_pAr, _pB0r), _one);
                _sum4 = __msa_dpadd_s_w(_sum4, __msa_dotp_s_h(_pA, _pB1), _one);
                _sum5 = __msa_dpadd_s_w(_sum5, __msa_dotp_s_h(_pA, _pB1r), _one);
                _sum6 = __msa_dpadd_s_w(_sum6, __msa_dotp_s_h(_pAr, _pB1), _one);
                _sum7 = __msa_dpadd_s_w(_sum7, __msa_dotp_s_h(_pAr, _pB1r), _one);

                pA += 16;
                pB += 32;
            }
            for (; kk < max_kk; kk += 1)
            {
                v8i16 _pA = (v8i16)__msa_fill_d(*(int*)pA);
                _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);
                v8i16 _pAr = __msa_shf_h(_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                v8i16 _pB0 = (v8i16)__msa_fill_d(*(int*)pB);
                _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB0, 0), (v16i8)_pB0);
                v8i16 _pB1 = (v8i16)__msa_fill_d(*(int*)(pB + 4));
                _pB1 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB1, 0), (v16i8)_pB1);
                v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));
                v8i16 _pB1r = __msa_shf_h(_pB1, _MSA_SHUFFLE(0, 3, 2, 1));

                v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                v8i16 _s1 = __msa_mulv_h(_pA, _pB0r);
                v8i16 _s2 = __msa_mulv_h(_pAr, _pB0);
                v8i16 _s3 = __msa_mulv_h(_pAr, _pB0r);
                v8i16 _s4 = __msa_mulv_h(_pA, _pB1);
                v8i16 _s5 = __msa_mulv_h(_pA, _pB1r);
                v8i16 _s6 = __msa_mulv_h(_pAr, _pB1);
                v8i16 _s7 = __msa_mulv_h(_pAr, _pB1r);

                _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s2, 0), _s2));
                _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s3, 0), _s3));
                _sum4 = __msa_addv_w(_sum4, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s4, 0), _s4));
                _sum5 = __msa_addv_w(_sum5, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s5, 0), _s5));
                _sum6 = __msa_addv_w(_sum6, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s6, 0), _s6));
                _sum7 = __msa_addv_w(_sum7, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s7, 0), _s7));

                pA += 4;
                pB += 8;
            }

            __msa_st_w(_sum0, outptr, 0);
            __msa_st_w(_sum1, outptr + 4, 0);
            __msa_st_w(_sum2, outptr + 8, 0);
            __msa_st_w(_sum3, outptr + 12, 0);
            __msa_st_w(_sum4, outptr + 16, 0);
            __msa_st_w(_sum5, outptr + 20, 0);
            __msa_st_w(_sum6, outptr + 24, 0);
            __msa_st_w(_sum7, outptr + 28, 0);

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            v4i32 _sum0;
            v4i32 _sum1;
            v4i32 _sum2;
            v4i32 _sum3;

            if (k == 0)
            {
                _sum0 = __msa_fill_w(0);
                _sum1 = __msa_fill_w(0);
                _sum2 = __msa_fill_w(0);
                _sum3 = __msa_fill_w(0);
            }
            else
            {
                _sum0 = __msa_ld_w(outptr, 0);
                _sum1 = __msa_ld_w(outptr + 4, 0);
                _sum2 = __msa_ld_w(outptr + 8, 0);
                _sum3 = __msa_ld_w(outptr + 12, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(pA + 64);
                __builtin_prefetch(pB + 32);
                v16i8 _pA = __msa_ld_b(pA, 0);
                v16i8 _pAr = (v16i8)__msa_shf_w((v4i32)_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                v16i8 _pB0 = __msa_ld_b(pB, 0);
                v16i8 _pB0r = (v16i8)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));

                _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB0r), _one);
                _sum2 = __msa_dpadd_s_w(_sum2, __msa_dotp_s_h(_pAr, _pB0), _one);
                _sum3 = __msa_dpadd_s_w(_sum3, __msa_dotp_s_h(_pAr, _pB0r), _one);

                pA += 16;
                pB += 16;
            }
            for (; kk < max_kk; kk += 1)
            {
                v8i16 _pA = (v8i16)__msa_fill_d(*(int*)pA);
                _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);
                v8i16 _pAr = __msa_shf_h(_pA, _MSA_SHUFFLE(1, 0, 3, 2));
                v8i16 _pB0 = (v8i16)__msa_fill_d(*(int*)pB);
                _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB0, 0), (v16i8)_pB0);
                v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));

                v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                v8i16 _s1 = __msa_mulv_h(_pA, _pB0r);
                v8i16 _s2 = __msa_mulv_h(_pAr, _pB0);
                v8i16 _s3 = __msa_mulv_h(_pAr, _pB0r);

                _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));
                _sum2 = __msa_addv_w(_sum2, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s2, 0), _s2));
                _sum3 = __msa_addv_w(_sum3, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s3, 0), _s3));

                pA += 4;
                pB += 4;
            }

            __msa_st_w(_sum0, outptr, 0);
            __msa_st_w(_sum1, outptr + 4, 0);
            __msa_st_w(_sum2, outptr + 8, 0);
            __msa_st_w(_sum3, outptr + 12, 0);

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pAT;

            v4i32 _sum0;
            v4i32 _sum1;

            if (k == 0)
            {
                _sum0 = __msa_fill_w(0);
                _sum1 = __msa_fill_w(0);
            }
            else
            {
                _sum0 = __msa_ld_w(outptr, 0);
                _sum1 = __msa_ld_w(outptr + 4, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(pA + 32);
                __builtin_prefetch(pB + 16);
                v16i8 _pA = __msa_ld_b(pA, 0);
                v16i8 _pB0 = (v16i8)__msa_fill_d(*(const int64_t*)pB);
                v16i8 _pB0r = (v16i8)__msa_shf_w((v4i32)_pB0, _MSA_SHUFFLE(0, 3, 2, 1));

                _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB0r), _one);

                pA += 16;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                v8i16 _pA = (v8i16)__msa_fill_d(*(int*)pA);
                _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);

                int b01 = (unsigned char)pB[0] | ((unsigned char)pB[1] << 8);
                v8i16 _pB0 = (v8i16)__msa_fill_w(b01);
                _pB0 = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pB0, 0), (v16i8)_pB0);
                _pB0 = __msa_shf_h(_pB0, _MSA_SHUFFLE(1, 0, 1, 0));
                v8i16 _pB0r = __msa_shf_h(_pB0, _MSA_SHUFFLE(0, 3, 2, 1));

                v8i16 _s0 = __msa_mulv_h(_pA, _pB0);
                v8i16 _s1 = __msa_mulv_h(_pA, _pB0r);

                _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));
                _sum1 = __msa_addv_w(_sum1, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s1, 0), _s1));

                pA += 4;
                pB += 2;
            }

            __msa_st_w(_sum0, outptr, 0);
            __msa_st_w(_sum1, outptr + 4, 0);

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            const signed char* pA = pAT;

            v4i32 _sum0;

            if (k == 0)
            {
                _sum0 = __msa_fill_w(0);
            }
            else
            {
                _sum0 = __msa_ld_w(outptr, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(pA + 32);
                __builtin_prefetch(pB + 16);
                v16i8 _pA = __msa_ld_b(pA, 0);

                v16i8 _pB0 = (v16i8)__msa_fill_w(*(const int*)pB);
                _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);

                pA += 16;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                v8i16 _pA = (v8i16)__msa_fill_d(*(int*)pA);
                _pA = (v8i16)__msa_ilvr_b(__msa_clti_s_b((v16i8)_pA, 0), (v16i8)_pA);

                v8i16 _pB0 = __msa_fill_h(pB[0]);

                v8i16 _s0 = __msa_mulv_h(_pA, _pB0);

                _sum0 = __msa_addv_w(_sum0, (v4i32)__msa_ilvr_h(__msa_clti_s_h(_s0, 0), _s0));

                pA += 4;
                pB += 1;
            }

            __msa_st_w(_sum0, outptr, 0);

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if __mips_msa
        const v8i16 _one = __msa_fill_h(1);

        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            v4i32 _sum0;
            v4i32 _sum1;
            v4i32 _sum2;
            v4i32 _sum3;

            if (k == 0)
            {
                _sum0 = __msa_fill_w(0);
                _sum1 = __msa_fill_w(0);
                _sum2 = __msa_fill_w(0);
                _sum3 = __msa_fill_w(0);
            }
            else
            {
                _sum0 = __msa_ld_w(outptr, 0);
                _sum1 = __msa_ld_w(outptr + 4, 0);
                _sum2 = __msa_ld_w(outptr + 8, 0);
                _sum3 = __msa_ld_w(outptr + 12, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(pA + 32);
                __builtin_prefetch(pB + 64);
                v16i8 _pA = (v16i8)__msa_fill_d(*(const int64_t*)pA);
                v16i8 _pB0 = __msa_ld_b(pB, 0);
                v16i8 _pB1 = __msa_ld_b(pB + 16, 0);

                v16i8 _pB01 = (v16i8)__msa_ilvr_w((v4i32)_pB0, (v4i32)_pB0);
                v16i8 _pB23 = (v16i8)__msa_ilvl_w((v4i32)_pB0, (v4i32)_pB0);
                v16i8 _pB45 = (v16i8)__msa_ilvr_w((v4i32)_pB1, (v4i32)_pB1);
                v16i8 _pB67 = (v16i8)__msa_ilvl_w((v4i32)_pB1, (v4i32)_pB1);

                _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB01), _one);
                _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB23), _one);
                _sum2 = __msa_dpadd_s_w(_sum2, __msa_dotp_s_h(_pA, _pB45), _one);
                _sum3 = __msa_dpadd_s_w(_sum3, __msa_dotp_s_h(_pA, _pB67), _one);

                pA += 8;
                pB += 32;
            }

            int sum00 = __msa_copy_s_w(_sum0, 0);
            int sum01 = __msa_copy_s_w(_sum0, 1);
            int sum10 = __msa_copy_s_w(_sum0, 2);
            int sum11 = __msa_copy_s_w(_sum0, 3);
            int sum20 = __msa_copy_s_w(_sum1, 0);
            int sum21 = __msa_copy_s_w(_sum1, 1);
            int sum30 = __msa_copy_s_w(_sum1, 2);
            int sum31 = __msa_copy_s_w(_sum1, 3);
            int sum40 = __msa_copy_s_w(_sum2, 0);
            int sum41 = __msa_copy_s_w(_sum2, 1);
            int sum50 = __msa_copy_s_w(_sum2, 2);
            int sum51 = __msa_copy_s_w(_sum2, 3);
            int sum60 = __msa_copy_s_w(_sum3, 0);
            int sum61 = __msa_copy_s_w(_sum3, 1);
            int sum70 = __msa_copy_s_w(_sum3, 2);
            int sum71 = __msa_copy_s_w(_sum3, 3);

            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum01 += pA[1] * pB[0];
                sum10 += pA[0] * pB[1];
                sum11 += pA[1] * pB[1];
                sum20 += pA[0] * pB[2];
                sum21 += pA[1] * pB[2];
                sum30 += pA[0] * pB[3];
                sum31 += pA[1] * pB[3];
                sum40 += pA[0] * pB[4];
                sum41 += pA[1] * pB[4];
                sum50 += pA[0] * pB[5];
                sum51 += pA[1] * pB[5];
                sum60 += pA[0] * pB[6];
                sum61 += pA[1] * pB[6];
                sum70 += pA[0] * pB[7];
                sum71 += pA[1] * pB[7];

                pA += 2;
                pB += 8;
            }

            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[2] = sum10;
            outptr[3] = sum11;
            outptr[4] = sum20;
            outptr[5] = sum21;
            outptr[6] = sum30;
            outptr[7] = sum31;
            outptr[8] = sum40;
            outptr[9] = sum41;
            outptr[10] = sum50;
            outptr[11] = sum51;
            outptr[12] = sum60;
            outptr[13] = sum61;
            outptr[14] = sum70;
            outptr[15] = sum71;

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            v4i32 _sum0;
            v4i32 _sum1;

            if (k == 0)
            {
                _sum0 = __msa_fill_w(0);
                _sum1 = __msa_fill_w(0);
            }
            else
            {
                _sum0 = __msa_ld_w(outptr, 0);
                _sum1 = __msa_ld_w(outptr + 4, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(pA + 32);
                __builtin_prefetch(pB + 32);
                v16i8 _pA = (v16i8)__msa_fill_d(*(const int64_t*)pA);
                v16i8 _pB0 = __msa_ld_b(pB, 0);
                v16i8 _pB01 = (v16i8)__msa_ilvr_w((v4i32)_pB0, (v4i32)_pB0);
                v16i8 _pB23 = (v16i8)__msa_ilvl_w((v4i32)_pB0, (v4i32)_pB0);

                _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB01), _one);
                _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB23), _one);

                pA += 8;
                pB += 16;
            }

            int sum00 = __msa_copy_s_w(_sum0, 0);
            int sum01 = __msa_copy_s_w(_sum0, 1);
            int sum10 = __msa_copy_s_w(_sum0, 2);
            int sum11 = __msa_copy_s_w(_sum0, 3);
            int sum20 = __msa_copy_s_w(_sum1, 0);
            int sum21 = __msa_copy_s_w(_sum1, 1);
            int sum30 = __msa_copy_s_w(_sum1, 2);
            int sum31 = __msa_copy_s_w(_sum1, 3);

            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum01 += pA[1] * pB[0];
                sum10 += pA[0] * pB[1];
                sum11 += pA[1] * pB[1];
                sum20 += pA[0] * pB[2];
                sum21 += pA[1] * pB[2];
                sum30 += pA[0] * pB[3];
                sum31 += pA[1] * pB[3];

                pA += 2;
                pB += 4;
            }

            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[2] = sum10;
            outptr[3] = sum11;
            outptr[4] = sum20;
            outptr[5] = sum21;
            outptr[6] = sum30;
            outptr[7] = sum31;

            outptr += 8;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            int sum00;
            int sum01;
            int sum10;
            int sum11;

            if (k == 0)
            {
                sum00 = 0;
                sum01 = 0;
                sum10 = 0;
                sum11 = 0;
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                sum00 += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                sum01 += pA[4] * pB[0] + pA[5] * pB[1] + pA[6] * pB[2] + pA[7] * pB[3];
                sum10 += pA[0] * pB[4] + pA[1] * pB[5] + pA[2] * pB[6] + pA[3] * pB[7];
                sum11 += pA[4] * pB[4] + pA[5] * pB[5] + pA[6] * pB[6] + pA[7] * pB[7];

                pA += 8;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum01 += pA[1] * pB[0];
                sum10 += pA[0] * pB[1];
                sum11 += pA[1] * pB[1];

                pA += 2;
                pB += 2;
            }

            outptr[0] = sum00;
            outptr[1] = sum01;
            outptr[2] = sum10;
            outptr[3] = sum11;

            outptr += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            int sum0;
            int sum1;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                sum0 += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                sum1 += pA[4] * pB[0] + pA[5] * pB[1] + pA[6] * pB[2] + pA[7] * pB[3];

                pA += 8;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];
                pA += 2;
                pB += 1;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
        const signed char* pB = pBT;

        int jj = 0;
#if __mips_msa
        const v8i16 _one = __msa_fill_h(1);

        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pA = pAT;

            v4i32 _sum0;
            v4i32 _sum1;

            if (k == 0)
            {
                _sum0 = __msa_fill_w(0);
                _sum1 = __msa_fill_w(0);
            }
            else
            {
                _sum0 = __msa_ld_w(outptr, 0);
                _sum1 = __msa_ld_w(outptr + 4, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 64);
                v16i8 _pA = (v16i8)__msa_fill_w(*(const int*)pA);
                v16i8 _pB0 = __msa_ld_b(pB, 0);
                v16i8 _pB1 = __msa_ld_b(pB + 16, 0);

                _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);
                _sum1 = __msa_dpadd_s_w(_sum1, __msa_dotp_s_h(_pA, _pB1), _one);

                pA += 4;
                pB += 32;
            }

            int sum0 = __msa_copy_s_w(_sum0, 0);
            int sum1 = __msa_copy_s_w(_sum0, 1);
            int sum2 = __msa_copy_s_w(_sum0, 2);
            int sum3 = __msa_copy_s_w(_sum0, 3);
            int sum4 = __msa_copy_s_w(_sum1, 0);
            int sum5 = __msa_copy_s_w(_sum1, 1);
            int sum6 = __msa_copy_s_w(_sum1, 2);
            int sum7 = __msa_copy_s_w(_sum1, 3);

            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];
                sum2 += pA[0] * pB[2];
                sum3 += pA[0] * pB[3];
                sum4 += pA[0] * pB[4];
                sum5 += pA[0] * pB[5];
                sum6 += pA[0] * pB[6];
                sum7 += pA[0] * pB[7];

                pA += 1;
                pB += 8;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr[2] = sum2;
            outptr[3] = sum3;
            outptr[4] = sum4;
            outptr[5] = sum5;
            outptr[6] = sum6;
            outptr[7] = sum7;

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pAT;

            v4i32 _sum0;

            if (k == 0)
            {
                _sum0 = __msa_fill_w(0);
            }
            else
            {
                _sum0 = __msa_ld_w(outptr, 0);
            }

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                __builtin_prefetch(pA + 16);
                __builtin_prefetch(pB + 32);
                v16i8 _pA = (v16i8)__msa_fill_w(*(const int*)pA);
                v16i8 _pB0 = __msa_ld_b(pB, 0);

                _sum0 = __msa_dpadd_s_w(_sum0, __msa_dotp_s_h(_pA, _pB0), _one);

                pA += 4;
                pB += 16;
            }

            int sum0 = __msa_copy_s_w(_sum0, 0);
            int sum1 = __msa_copy_s_w(_sum0, 1);
            int sum2 = __msa_copy_s_w(_sum0, 2);
            int sum3 = __msa_copy_s_w(_sum0, 3);

            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];
                sum2 += pA[0] * pB[2];
                sum3 += pA[0] * pB[3];

                pA += 1;
                pB += 4;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;
            outptr[2] = sum2;
            outptr[3] = sum3;

            outptr += 4;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            int sum0;
            int sum1;

            if (k == 0)
            {
                sum0 = 0;
                sum1 = 0;
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                sum0 += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];
                sum1 += pA[0] * pB[4] + pA[1] * pB[5] + pA[2] * pB[6] + pA[3] * pB[7];

                pA += 4;
                pB += 8;
            }
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];

                pA += 1;
                pB += 2;
            }

            outptr[0] = sum0;
            outptr[1] = sum1;

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            int sum;

            if (k == 0)
            {
                sum = 0;
            }
            else
            {
                sum = outptr[0];
            }

            const signed char* pA = pAT;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                sum += pA[0] * pB[0] + pA[1] * pB[1] + pA[2] * pB[2] + pA[3] * pB[3];

                pA += 4;
                pB += 4;
            }
            for (; kk < max_kk; kk += 1)
            {
                sum += pA[0] * pB[0];
                pA += 1;
                pB += 1;
            }

            outptr[0] = sum;

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void unpack_output_tile_int32_to_fp32(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta, int output_transpose)
{
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;

    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;
    const float* descale_ptr = descales;

    const int* pp = topT;

    int ii = 0;
#if __mips_msa
    for (; ii + 7 < max_ii; ii += 8)
    {
        int jj = 0;
        v4f32 _descale0 = (v4f32)__msa_ld_w(descale_ptr + i + ii, 0);
        v4f32 _descale1 = (v4f32)__msa_ld_w(descale_ptr + i + ii + 4, 0);

        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + i + ii;
        }
        else
        {
            p0 = (float*)top_blob + (i + ii) * out_hstep + j;
        }

        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (size_t)(i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        for (; jj + 7 < max_jj; jj += 8)
        {
            __builtin_prefetch(pp + 64);
            v4i32 _sum0 = __msa_ld_w(pp, 0);
            v4i32 _sum8 = __msa_ld_w(pp + 4, 0);
            v4i32 _sum1 = __msa_ld_w(pp + 8, 0);
            v4i32 _sum9 = __msa_ld_w(pp + 12, 0);
            v4i32 _sum2 = __msa_ld_w(pp + 16, 0);
            v4i32 _suma = __msa_ld_w(pp + 20, 0);
            v4i32 _sum3 = __msa_ld_w(pp + 24, 0);
            v4i32 _sumb = __msa_ld_w(pp + 28, 0);

            _sum2 = __msa_shf_w(_sum2, _MSA_SHUFFLE(1, 0, 3, 2));
            _sum3 = __msa_shf_w(_sum3, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
            _sum1 = __msa_shf_w(_sum1, _MSA_SHUFFLE(2, 1, 0, 3));
            _sum2 = __msa_shf_w(_sum2, _MSA_SHUFFLE(1, 0, 3, 2));
            _sum3 = __msa_shf_w(_sum3, _MSA_SHUFFLE(0, 3, 2, 1));

            _suma = __msa_shf_w(_suma, _MSA_SHUFFLE(1, 0, 3, 2));
            _sumb = __msa_shf_w(_sumb, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum8, _sum9, _suma, _sumb);
            _sum9 = __msa_shf_w(_sum9, _MSA_SHUFFLE(2, 1, 0, 3));
            _suma = __msa_shf_w(_suma, _MSA_SHUFFLE(1, 0, 3, 2));
            _sumb = __msa_shf_w(_sumb, _MSA_SHUFFLE(0, 3, 2, 1));

            v4i32 _sum4 = __msa_ld_w(pp + 32, 0);
            v4i32 _sumc = __msa_ld_w(pp + 36, 0);
            v4i32 _sum5 = __msa_ld_w(pp + 40, 0);
            v4i32 _sumd = __msa_ld_w(pp + 44, 0);
            v4i32 _sum6 = __msa_ld_w(pp + 48, 0);
            v4i32 _sume = __msa_ld_w(pp + 52, 0);
            v4i32 _sum7 = __msa_ld_w(pp + 56, 0);
            v4i32 _sumf = __msa_ld_w(pp + 60, 0);

            _sum6 = __msa_shf_w(_sum6, _MSA_SHUFFLE(1, 0, 3, 2));
            _sum7 = __msa_shf_w(_sum7, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum4, _sum5, _sum6, _sum7);
            _sum5 = __msa_shf_w(_sum5, _MSA_SHUFFLE(2, 1, 0, 3));
            _sum6 = __msa_shf_w(_sum6, _MSA_SHUFFLE(1, 0, 3, 2));
            _sum7 = __msa_shf_w(_sum7, _MSA_SHUFFLE(0, 3, 2, 1));

            _sume = __msa_shf_w(_sume, _MSA_SHUFFLE(1, 0, 3, 2));
            _sumf = __msa_shf_w(_sumf, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sumc, _sumd, _sume, _sumf);
            _sumd = __msa_shf_w(_sumd, _MSA_SHUFFLE(2, 1, 0, 3));
            _sume = __msa_shf_w(_sume, _MSA_SHUFFLE(1, 0, 3, 2));
            _sumf = __msa_shf_w(_sumf, _MSA_SHUFFLE(0, 3, 2, 1));

            v4f32 _f0 = (v4f32)__msa_ffint_s_w(_sum0);
            v4f32 _f4 = (v4f32)__msa_ffint_s_w(_sum4);
            v4f32 _f1 = (v4f32)__msa_ffint_s_w(_sum1);
            v4f32 _f5 = (v4f32)__msa_ffint_s_w(_sum5);
            v4f32 _f2 = (v4f32)__msa_ffint_s_w(_sum2);
            v4f32 _f6 = (v4f32)__msa_ffint_s_w(_sum6);
            v4f32 _f3 = (v4f32)__msa_ffint_s_w(_sum3);
            v4f32 _f7 = (v4f32)__msa_ffint_s_w(_sum7);
            v4f32 _f8 = (v4f32)__msa_ffint_s_w(_sum8);
            v4f32 _fc = (v4f32)__msa_ffint_s_w(_sumc);
            v4f32 _f9 = (v4f32)__msa_ffint_s_w(_sum9);
            v4f32 _fd = (v4f32)__msa_ffint_s_w(_sumd);
            v4f32 _fa = (v4f32)__msa_ffint_s_w(_suma);
            v4f32 _fe = (v4f32)__msa_ffint_s_w(_sume);
            v4f32 _fb = (v4f32)__msa_ffint_s_w(_sumb);
            v4f32 _ff = (v4f32)__msa_ffint_s_w(_sumf);

            _f0 = __msa_fmul_w(_f0, (v4f32)__msa_splati_w((v4i32)_descale0, 0));
            _f4 = __msa_fmul_w(_f4, (v4f32)__msa_splati_w((v4i32)_descale0, 0));
            _f1 = __msa_fmul_w(_f1, (v4f32)__msa_splati_w((v4i32)_descale0, 1));
            _f5 = __msa_fmul_w(_f5, (v4f32)__msa_splati_w((v4i32)_descale0, 1));
            _f2 = __msa_fmul_w(_f2, (v4f32)__msa_splati_w((v4i32)_descale0, 2));
            _f6 = __msa_fmul_w(_f6, (v4f32)__msa_splati_w((v4i32)_descale0, 2));
            _f3 = __msa_fmul_w(_f3, (v4f32)__msa_splati_w((v4i32)_descale0, 3));
            _f7 = __msa_fmul_w(_f7, (v4f32)__msa_splati_w((v4i32)_descale0, 3));
            _f8 = __msa_fmul_w(_f8, (v4f32)__msa_splati_w((v4i32)_descale1, 0));
            _fc = __msa_fmul_w(_fc, (v4f32)__msa_splati_w((v4i32)_descale1, 0));
            _f9 = __msa_fmul_w(_f9, (v4f32)__msa_splati_w((v4i32)_descale1, 1));
            _fd = __msa_fmul_w(_fd, (v4f32)__msa_splati_w((v4i32)_descale1, 1));
            _fa = __msa_fmul_w(_fa, (v4f32)__msa_splati_w((v4i32)_descale1, 2));
            _fe = __msa_fmul_w(_fe, (v4f32)__msa_splati_w((v4i32)_descale1, 2));
            _fb = __msa_fmul_w(_fb, (v4f32)__msa_splati_w((v4i32)_descale1, 3));
            _ff = __msa_fmul_w(_ff, (v4f32)__msa_splati_w((v4i32)_descale1, 3));

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    v4f32 _c0 = __msa_fill_w_f32(pC[0] * beta);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f4 = __msa_fadd_w(_f4, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                    _f5 = __msa_fadd_w(_f5, _c0);
                    _f2 = __msa_fadd_w(_f2, _c0);
                    _f6 = __msa_fadd_w(_f6, _c0);
                    _f3 = __msa_fadd_w(_f3, _c0);
                    _f7 = __msa_fadd_w(_f7, _c0);
                    _f8 = __msa_fadd_w(_f8, _c0);
                    _fc = __msa_fadd_w(_fc, _c0);
                    _f9 = __msa_fadd_w(_f9, _c0);
                    _fd = __msa_fadd_w(_fd, _c0);
                    _fa = __msa_fadd_w(_fa, _c0);
                    _fe = __msa_fadd_w(_fe, _c0);
                    _fb = __msa_fadd_w(_fb, _c0);
                    _ff = __msa_fadd_w(_ff, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c0 = __msa_fill_w_f32(pC[0] * beta);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f4 = __msa_fadd_w(_f4, _c0);
                    _c0 = __msa_fill_w_f32(pC[1] * beta);
                    _f1 = __msa_fadd_w(_f1, _c0);
                    _f5 = __msa_fadd_w(_f5, _c0);
                    _c0 = __msa_fill_w_f32(pC[2] * beta);
                    _f2 = __msa_fadd_w(_f2, _c0);
                    _f6 = __msa_fadd_w(_f6, _c0);
                    _c0 = __msa_fill_w_f32(pC[3] * beta);
                    _f3 = __msa_fadd_w(_f3, _c0);
                    _f7 = __msa_fadd_w(_f7, _c0);
                    _c0 = __msa_fill_w_f32(pC[4] * beta);
                    _f8 = __msa_fadd_w(_f8, _c0);
                    _fc = __msa_fadd_w(_fc, _c0);
                    _c0 = __msa_fill_w_f32(pC[5] * beta);
                    _f9 = __msa_fadd_w(_f9, _c0);
                    _fd = __msa_fadd_w(_fd, _c0);
                    _c0 = __msa_fill_w_f32(pC[6] * beta);
                    _fa = __msa_fadd_w(_fa, _c0);
                    _fe = __msa_fadd_w(_fe, _c0);
                    _c0 = __msa_fill_w_f32(pC[7] * beta);
                    _fb = __msa_fadd_w(_fb, _c0);
                    _ff = __msa_fadd_w(_ff, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta));
                    _f4 = __msa_fadd_w(_f4, __msa_fmul_w((v4f32)__msa_ld_w(pC + 4, 0), _beta));
                    _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep, 0), _beta));
                    _f5 = __msa_fadd_w(_f5, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep + 4, 0), _beta));
                    _f2 = __msa_fadd_w(_f2, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 2, 0), _beta));
                    _f6 = __msa_fadd_w(_f6, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 2 + 4, 0), _beta));
                    _f3 = __msa_fadd_w(_f3, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 3, 0), _beta));
                    _f7 = __msa_fadd_w(_f7, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 3 + 4, 0), _beta));
                    _f8 = __msa_fadd_w(_f8, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 4, 0), _beta));
                    _fc = __msa_fadd_w(_fc, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 4 + 4, 0), _beta));
                    _f9 = __msa_fadd_w(_f9, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 5, 0), _beta));
                    _fd = __msa_fadd_w(_fd, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 5 + 4, 0), _beta));
                    _fa = __msa_fadd_w(_fa, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 6, 0), _beta));
                    _fe = __msa_fadd_w(_fe, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 6 + 4, 0), _beta));
                    _fb = __msa_fadd_w(_fb, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 7, 0), _beta));
                    _ff = __msa_fadd_w(_ff, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 7 + 4, 0), _beta));
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    v4f32 _c0 = __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta);
                    v4f32 _c1 = __msa_fmul_w((v4f32)__msa_ld_w(pC + 4, 0), _beta);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f4 = __msa_fadd_w(_f4, _c1);
                    _f1 = __msa_fadd_w(_f1, _c0);
                    _f5 = __msa_fadd_w(_f5, _c1);
                    _f2 = __msa_fadd_w(_f2, _c0);
                    _f6 = __msa_fadd_w(_f6, _c1);
                    _f3 = __msa_fadd_w(_f3, _c0);
                    _f7 = __msa_fadd_w(_f7, _c1);
                    _f8 = __msa_fadd_w(_f8, _c0);
                    _fc = __msa_fadd_w(_fc, _c1);
                    _f9 = __msa_fadd_w(_f9, _c0);
                    _fd = __msa_fadd_w(_fd, _c1);
                    _fa = __msa_fadd_w(_fa, _c0);
                    _fe = __msa_fadd_w(_fe, _c1);
                    _fb = __msa_fadd_w(_fb, _c0);
                    _ff = __msa_fadd_w(_ff, _c1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f4 = __msa_fmul_w(_f4, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
                _f5 = __msa_fmul_w(_f5, _alpha);
                _f2 = __msa_fmul_w(_f2, _alpha);
                _f6 = __msa_fmul_w(_f6, _alpha);
                _f3 = __msa_fmul_w(_f3, _alpha);
                _f7 = __msa_fmul_w(_f7, _alpha);
                _f8 = __msa_fmul_w(_f8, _alpha);
                _fc = __msa_fmul_w(_fc, _alpha);
                _f9 = __msa_fmul_w(_f9, _alpha);
                _fd = __msa_fmul_w(_fd, _alpha);
                _fa = __msa_fmul_w(_fa, _alpha);
                _fe = __msa_fmul_w(_fe, _alpha);
                _fb = __msa_fmul_w(_fb, _alpha);
                _ff = __msa_fmul_w(_ff, _alpha);
            }

            if (output_transpose)
            {
                transpose4x4_ps(_f0, _f1, _f2, _f3);
                transpose4x4_ps(_f4, _f5, _f6, _f7);
                transpose4x4_ps(_f8, _f9, _fa, _fb);
                transpose4x4_ps(_fc, _fd, _fe, _ff);

                __msa_st_w((v4i32)_f0, p0, 0);
                __msa_st_w((v4i32)_f8, p0 + 4, 0);
                __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
                __msa_st_w((v4i32)_f9, p0 + out_hstep + 4, 0);
                __msa_st_w((v4i32)_f2, p0 + out_hstep * 2, 0);
                __msa_st_w((v4i32)_fa, p0 + out_hstep * 2 + 4, 0);
                __msa_st_w((v4i32)_f3, p0 + out_hstep * 3, 0);
                __msa_st_w((v4i32)_fb, p0 + out_hstep * 3 + 4, 0);
                __msa_st_w((v4i32)_f4, p0 + out_hstep * 4, 0);
                __msa_st_w((v4i32)_fc, p0 + out_hstep * 4 + 4, 0);
                __msa_st_w((v4i32)_f5, p0 + out_hstep * 5, 0);
                __msa_st_w((v4i32)_fd, p0 + out_hstep * 5 + 4, 0);
                __msa_st_w((v4i32)_f6, p0 + out_hstep * 6, 0);
                __msa_st_w((v4i32)_fe, p0 + out_hstep * 6 + 4, 0);
                __msa_st_w((v4i32)_f7, p0 + out_hstep * 7, 0);
                __msa_st_w((v4i32)_ff, p0 + out_hstep * 7 + 4, 0);
                p0 += out_hstep * 8;
            }
            else
            {
                __msa_st_w((v4i32)_f0, p0, 0);
                __msa_st_w((v4i32)_f4, p0 + 4, 0);
                __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
                __msa_st_w((v4i32)_f5, p0 + out_hstep + 4, 0);
                __msa_st_w((v4i32)_f2, p0 + out_hstep * 2, 0);
                __msa_st_w((v4i32)_f6, p0 + out_hstep * 2 + 4, 0);
                __msa_st_w((v4i32)_f3, p0 + out_hstep * 3, 0);
                __msa_st_w((v4i32)_f7, p0 + out_hstep * 3 + 4, 0);
                __msa_st_w((v4i32)_f8, p0 + out_hstep * 4, 0);
                __msa_st_w((v4i32)_fc, p0 + out_hstep * 4 + 4, 0);
                __msa_st_w((v4i32)_f9, p0 + out_hstep * 5, 0);
                __msa_st_w((v4i32)_fd, p0 + out_hstep * 5 + 4, 0);
                __msa_st_w((v4i32)_fa, p0 + out_hstep * 6, 0);
                __msa_st_w((v4i32)_fe, p0 + out_hstep * 6 + 4, 0);
                __msa_st_w((v4i32)_fb, p0 + out_hstep * 7, 0);
                __msa_st_w((v4i32)_ff, p0 + out_hstep * 7 + 4, 0);
                p0 += 8;
            }
            pp += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __builtin_prefetch(pp + 32);
            v4i32 _sum0 = __msa_ld_w(pp, 0);
            v4i32 _sum1 = __msa_ld_w(pp + 8, 0);
            v4i32 _sum2 = __msa_ld_w(pp + 16, 0);
            v4i32 _sum3 = __msa_ld_w(pp + 24, 0);

            _sum2 = __msa_shf_w(_sum2, _MSA_SHUFFLE(1, 0, 3, 2));
            _sum3 = __msa_shf_w(_sum3, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
            _sum1 = __msa_shf_w(_sum1, _MSA_SHUFFLE(2, 1, 0, 3));
            _sum2 = __msa_shf_w(_sum2, _MSA_SHUFFLE(1, 0, 3, 2));
            _sum3 = __msa_shf_w(_sum3, _MSA_SHUFFLE(0, 3, 2, 1));

            v4i32 _sum4 = __msa_ld_w(pp + 4, 0);
            v4i32 _sum5 = __msa_ld_w(pp + 12, 0);
            v4i32 _sum6 = __msa_ld_w(pp + 20, 0);
            v4i32 _sum7 = __msa_ld_w(pp + 28, 0);

            _sum6 = __msa_shf_w(_sum6, _MSA_SHUFFLE(1, 0, 3, 2));
            _sum7 = __msa_shf_w(_sum7, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum4, _sum5, _sum6, _sum7);
            _sum5 = __msa_shf_w(_sum5, _MSA_SHUFFLE(2, 1, 0, 3));
            _sum6 = __msa_shf_w(_sum6, _MSA_SHUFFLE(1, 0, 3, 2));
            _sum7 = __msa_shf_w(_sum7, _MSA_SHUFFLE(0, 3, 2, 1));

            v4f32 _f0 = (v4f32)__msa_ffint_s_w(_sum0);
            v4f32 _f1 = (v4f32)__msa_ffint_s_w(_sum1);
            v4f32 _f2 = (v4f32)__msa_ffint_s_w(_sum2);
            v4f32 _f3 = (v4f32)__msa_ffint_s_w(_sum3);
            v4f32 _f4 = (v4f32)__msa_ffint_s_w(_sum4);
            v4f32 _f5 = (v4f32)__msa_ffint_s_w(_sum5);
            v4f32 _f6 = (v4f32)__msa_ffint_s_w(_sum6);
            v4f32 _f7 = (v4f32)__msa_ffint_s_w(_sum7);

            _f0 = __msa_fmul_w(_f0, (v4f32)__msa_splati_w((v4i32)_descale0, 0));
            _f1 = __msa_fmul_w(_f1, (v4f32)__msa_splati_w((v4i32)_descale0, 1));
            _f2 = __msa_fmul_w(_f2, (v4f32)__msa_splati_w((v4i32)_descale0, 2));
            _f3 = __msa_fmul_w(_f3, (v4f32)__msa_splati_w((v4i32)_descale0, 3));
            _f4 = __msa_fmul_w(_f4, (v4f32)__msa_splati_w((v4i32)_descale1, 0));
            _f5 = __msa_fmul_w(_f5, (v4f32)__msa_splati_w((v4i32)_descale1, 1));
            _f6 = __msa_fmul_w(_f6, (v4f32)__msa_splati_w((v4i32)_descale1, 2));
            _f7 = __msa_fmul_w(_f7, (v4f32)__msa_splati_w((v4i32)_descale1, 3));

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    v4f32 _c0 = __msa_fill_w_f32(pC[0] * beta);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                    _f2 = __msa_fadd_w(_f2, _c0);
                    _f3 = __msa_fadd_w(_f3, _c0);
                    _f4 = __msa_fadd_w(_f4, _c0);
                    _f5 = __msa_fadd_w(_f5, _c0);
                    _f6 = __msa_fadd_w(_f6, _c0);
                    _f7 = __msa_fadd_w(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(pC[0] * beta));
                    _f1 = __msa_fadd_w(_f1, __msa_fill_w_f32(pC[1] * beta));
                    _f2 = __msa_fadd_w(_f2, __msa_fill_w_f32(pC[2] * beta));
                    _f3 = __msa_fadd_w(_f3, __msa_fill_w_f32(pC[3] * beta));
                    _f4 = __msa_fadd_w(_f4, __msa_fill_w_f32(pC[4] * beta));
                    _f5 = __msa_fadd_w(_f5, __msa_fill_w_f32(pC[5] * beta));
                    _f6 = __msa_fadd_w(_f6, __msa_fill_w_f32(pC[6] * beta));
                    _f7 = __msa_fadd_w(_f7, __msa_fill_w_f32(pC[7] * beta));
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta));
                    _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep, 0), _beta));
                    _f2 = __msa_fadd_w(_f2, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 2, 0), _beta));
                    _f3 = __msa_fadd_w(_f3, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 3, 0), _beta));
                    _f4 = __msa_fadd_w(_f4, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 4, 0), _beta));
                    _f5 = __msa_fadd_w(_f5, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 5, 0), _beta));
                    _f6 = __msa_fadd_w(_f6, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 6, 0), _beta));
                    _f7 = __msa_fadd_w(_f7, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 7, 0), _beta));
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c0 = __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                    _f2 = __msa_fadd_w(_f2, _c0);
                    _f3 = __msa_fadd_w(_f3, _c0);
                    _f4 = __msa_fadd_w(_f4, _c0);
                    _f5 = __msa_fadd_w(_f5, _c0);
                    _f6 = __msa_fadd_w(_f6, _c0);
                    _f7 = __msa_fadd_w(_f7, _c0);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
                _f2 = __msa_fmul_w(_f2, _alpha);
                _f3 = __msa_fmul_w(_f3, _alpha);
                _f4 = __msa_fmul_w(_f4, _alpha);
                _f5 = __msa_fmul_w(_f5, _alpha);
                _f6 = __msa_fmul_w(_f6, _alpha);
                _f7 = __msa_fmul_w(_f7, _alpha);
            }

            if (output_transpose)
            {
                transpose4x4_ps(_f0, _f1, _f2, _f3);
                transpose4x4_ps(_f4, _f5, _f6, _f7);

                __msa_st_w((v4i32)_f0, p0, 0);
                __msa_st_w((v4i32)_f4, p0 + 4, 0);
                __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
                __msa_st_w((v4i32)_f5, p0 + out_hstep + 4, 0);
                __msa_st_w((v4i32)_f2, p0 + out_hstep * 2, 0);
                __msa_st_w((v4i32)_f6, p0 + out_hstep * 2 + 4, 0);
                __msa_st_w((v4i32)_f3, p0 + out_hstep * 3, 0);
                __msa_st_w((v4i32)_f7, p0 + out_hstep * 3 + 4, 0);
                p0 += out_hstep * 4;
            }
            else
            {
                __msa_st_w((v4i32)_f0, p0, 0);
                __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
                __msa_st_w((v4i32)_f2, p0 + out_hstep * 2, 0);
                __msa_st_w((v4i32)_f3, p0 + out_hstep * 3, 0);
                __msa_st_w((v4i32)_f4, p0 + out_hstep * 4, 0);
                __msa_st_w((v4i32)_f5, p0 + out_hstep * 5, 0);
                __msa_st_w((v4i32)_f6, p0 + out_hstep * 6, 0);
                __msa_st_w((v4i32)_f7, p0 + out_hstep * 7, 0);
                p0 += 4;
            }
            pp += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __builtin_prefetch(pp + 16);
            v4i32 _sum0 = __msa_ld_w(pp, 0);
            v4i32 _sum1 = __msa_ld_w(pp + 4, 0);
            v4i32 _sum2 = __msa_ld_w(pp + 8, 0);
            v4i32 _sum3 = __msa_ld_w(pp + 12, 0);

            v4i32 _sum0e = __msa_shf_w(_sum0, _MSA_SHUFFLE(3, 1, 2, 0));
            v4i32 _sum0o = __msa_shf_w(_sum0, _MSA_SHUFFLE(2, 0, 3, 1));
            v4i32 _sum2e = __msa_shf_w(_sum2, _MSA_SHUFFLE(3, 1, 2, 0));
            v4i32 _sum2o = __msa_shf_w(_sum2, _MSA_SHUFFLE(2, 0, 3, 1));
            v4i32 _sum4e = __msa_shf_w(_sum1, _MSA_SHUFFLE(3, 1, 2, 0));
            v4i32 _sum4o = __msa_shf_w(_sum1, _MSA_SHUFFLE(2, 0, 3, 1));
            v4i32 _sum6e = __msa_shf_w(_sum3, _MSA_SHUFFLE(3, 1, 2, 0));
            v4i32 _sum6o = __msa_shf_w(_sum3, _MSA_SHUFFLE(2, 0, 3, 1));

            v4f32 _f0 = (v4f32)__msa_ffint_s_w((v4i32)__msa_ilvr_w(_sum2o, _sum0e));
            v4f32 _f1 = (v4f32)__msa_ffint_s_w((v4i32)__msa_ilvr_w(_sum0o, _sum2e));
            v4f32 _f4 = (v4f32)__msa_ffint_s_w((v4i32)__msa_ilvr_w(_sum6o, _sum4e));
            v4f32 _f5 = (v4f32)__msa_ffint_s_w((v4i32)__msa_ilvr_w(_sum4o, _sum6e));

            _f0 = __msa_fmul_w(_f0, _descale0);
            _f1 = __msa_fmul_w(_f1, _descale0);
            _f4 = __msa_fmul_w(_f4, _descale1);
            _f5 = __msa_fmul_w(_f5, _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    v4f32 _c0 = __msa_fill_w_f32(pC[0] * beta);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                    _f4 = __msa_fadd_w(_f4, _c0);
                    _f5 = __msa_fadd_w(_f5, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    v4f32 _c0 = __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta);
                    v4f32 _c1 = __msa_fmul_w((v4f32)__msa_ld_w(pC + 4, 0), _beta);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                    _f4 = __msa_fadd_w(_f4, _c1);
                    _f5 = __msa_fadd_w(_f5, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    v4i32 _c0 = __msa_fill_w(((const int*)pC)[0]);
                    _c0 = __msa_insert_w(_c0, 1, ((const int*)(pC + c_hstep))[0]);
                    _c0 = __msa_insert_w(_c0, 2, ((const int*)(pC + c_hstep * 2))[0]);
                    _c0 = __msa_insert_w(_c0, 3, ((const int*)(pC + c_hstep * 3))[0]);
                    v4i32 _c1 = __msa_fill_w(((const int*)pC)[1]);
                    _c1 = __msa_insert_w(_c1, 1, ((const int*)(pC + c_hstep))[1]);
                    _c1 = __msa_insert_w(_c1, 2, ((const int*)(pC + c_hstep * 2))[1]);
                    _c1 = __msa_insert_w(_c1, 3, ((const int*)(pC + c_hstep * 3))[1]);
                    v4i32 _c4 = __msa_fill_w(((const int*)(pC + c_hstep * 4))[0]);
                    _c4 = __msa_insert_w(_c4, 1, ((const int*)(pC + c_hstep * 5))[0]);
                    _c4 = __msa_insert_w(_c4, 2, ((const int*)(pC + c_hstep * 6))[0]);
                    _c4 = __msa_insert_w(_c4, 3, ((const int*)(pC + c_hstep * 7))[0]);
                    v4i32 _c5 = __msa_fill_w(((const int*)(pC + c_hstep * 4))[1]);
                    _c5 = __msa_insert_w(_c5, 1, ((const int*)(pC + c_hstep * 5))[1]);
                    _c5 = __msa_insert_w(_c5, 2, ((const int*)(pC + c_hstep * 6))[1]);
                    _c5 = __msa_insert_w(_c5, 3, ((const int*)(pC + c_hstep * 7))[1]);
                    _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)_c0, _beta));
                    _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)_c1, _beta));
                    _f4 = __msa_fadd_w(_f4, __msa_fmul_w((v4f32)_c4, _beta));
                    _f5 = __msa_fadd_w(_f5, __msa_fmul_w((v4f32)_c5, _beta));
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c0 = __msa_fill_w_f32(pC[0] * beta);
                    v4f32 _c1 = __msa_fill_w_f32(pC[1] * beta);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f4 = __msa_fadd_w(_f4, _c0);
                    _f1 = __msa_fadd_w(_f1, _c1);
                    _f5 = __msa_fadd_w(_f5, _c1);
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
                _f4 = __msa_fmul_w(_f4, _alpha);
                _f5 = __msa_fmul_w(_f5, _alpha);
            }

            if (output_transpose)
            {
                __msa_st_w((v4i32)_f0, p0, 0);
                __msa_st_w((v4i32)_f4, p0 + 4, 0);
                __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
                __msa_st_w((v4i32)_f5, p0 + out_hstep + 4, 0);
                p0 += out_hstep * 2;
            }
            else
            {
                v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_f1, (v4i32)_f0);
                v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_f1, (v4i32)_f0);
                v4f32 _tmp2 = (v4f32)__msa_ilvr_w((v4i32)_f5, (v4i32)_f4);
                v4f32 _tmp3 = (v4f32)__msa_ilvl_w((v4i32)_f5, (v4i32)_f4);

                *(int64_t*)p0 = __msa_copy_s_d((v2i64)_tmp0, 0);
                *(int64_t*)(p0 + out_hstep) = __msa_copy_s_d((v2i64)_tmp0, 1);
                *(int64_t*)(p0 + out_hstep * 2) = __msa_copy_s_d((v2i64)_tmp1, 0);
                *(int64_t*)(p0 + out_hstep * 3) = __msa_copy_s_d((v2i64)_tmp1, 1);
                *(int64_t*)(p0 + out_hstep * 4) = __msa_copy_s_d((v2i64)_tmp2, 0);
                *(int64_t*)(p0 + out_hstep * 5) = __msa_copy_s_d((v2i64)_tmp2, 1);
                *(int64_t*)(p0 + out_hstep * 6) = __msa_copy_s_d((v2i64)_tmp3, 0);
                *(int64_t*)(p0 + out_hstep * 7) = __msa_copy_s_d((v2i64)_tmp3, 1);
                p0 += 2;
            }
            pp += 16;
        }
        for (; jj < max_jj; jj++)
        {
            __builtin_prefetch(pp + 8);
            v4f32 _f0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(pp, 0));
            v4f32 _f4 = (v4f32)__msa_ffint_s_w(__msa_ld_w(pp + 4, 0));

            _f0 = __msa_fmul_w(_f0, _descale0);
            _f4 = __msa_fmul_w(_f4, _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    v4f32 _c0 = __msa_fill_w_f32(pC[0] * beta);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f4 = __msa_fadd_w(_f4, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta));
                    _f4 = __msa_fadd_w(_f4, __msa_fmul_w((v4f32)__msa_ld_w(pC + 4, 0), _beta));
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    v4i32 _c0 = __msa_fill_w(((const int*)pC)[0]);
                    _c0 = __msa_insert_w(_c0, 1, ((const int*)(pC + c_hstep))[0]);
                    _c0 = __msa_insert_w(_c0, 2, ((const int*)(pC + c_hstep * 2))[0]);
                    _c0 = __msa_insert_w(_c0, 3, ((const int*)(pC + c_hstep * 3))[0]);
                    v4i32 _c4 = __msa_fill_w(((const int*)(pC + c_hstep * 4))[0]);
                    _c4 = __msa_insert_w(_c4, 1, ((const int*)(pC + c_hstep * 5))[0]);
                    _c4 = __msa_insert_w(_c4, 2, ((const int*)(pC + c_hstep * 6))[0]);
                    _c4 = __msa_insert_w(_c4, 3, ((const int*)(pC + c_hstep * 7))[0]);
                    _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)_c0, _beta));
                    _f4 = __msa_fadd_w(_f4, __msa_fmul_w((v4f32)_c4, _beta));
                    pC += 1;
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c0 = __msa_fill_w_f32(pC[0] * beta);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f4 = __msa_fadd_w(_f4, _c0);
                    pC += 1;
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f4 = __msa_fmul_w(_f4, _alpha);
            }

            if (output_transpose)
            {
                __msa_st_w((v4i32)_f0, p0, 0);
                __msa_st_w((v4i32)_f4, p0 + 4, 0);
                p0 += out_hstep;
            }
            else
            {
                *(int*)p0 = __msa_copy_s_w((v4i32)_f0, 0);
                *(int*)(p0 + out_hstep) = __msa_copy_s_w((v4i32)_f0, 1);
                *(int*)(p0 + out_hstep * 2) = __msa_copy_s_w((v4i32)_f0, 2);
                *(int*)(p0 + out_hstep * 3) = __msa_copy_s_w((v4i32)_f0, 3);
                *(int*)(p0 + out_hstep * 4) = __msa_copy_s_w((v4i32)_f4, 0);
                *(int*)(p0 + out_hstep * 5) = __msa_copy_s_w((v4i32)_f4, 1);
                *(int*)(p0 + out_hstep * 6) = __msa_copy_s_w((v4i32)_f4, 2);
                *(int*)(p0 + out_hstep * 7) = __msa_copy_s_w((v4i32)_f4, 3);
                p0 += 1;
            }
            pp += 8;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        int jj = 0;
        v4f32 _descale0 = (v4f32)__msa_ld_w(descale_ptr + i + ii, 0);

        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + i + ii;
        }
        else
        {
            p0 = (float*)top_blob + (i + ii) * out_hstep + j;
        }

        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (size_t)(i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        for (; jj + 7 < max_jj; jj += 8)
        {
            __builtin_prefetch(pp + 32);
            v4i32 _sum0 = __msa_ld_w(pp, 0);
            v4i32 _sum1 = __msa_ld_w(pp + 4, 0);
            v4i32 _sum2 = __msa_ld_w(pp + 8, 0);
            v4i32 _sum3 = __msa_ld_w(pp + 12, 0);

            _sum2 = __msa_shf_w(_sum2, _MSA_SHUFFLE(1, 0, 3, 2));
            _sum3 = __msa_shf_w(_sum3, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
            _sum1 = __msa_shf_w(_sum1, _MSA_SHUFFLE(2, 1, 0, 3));
            _sum2 = __msa_shf_w(_sum2, _MSA_SHUFFLE(1, 0, 3, 2));
            _sum3 = __msa_shf_w(_sum3, _MSA_SHUFFLE(0, 3, 2, 1));

            v4i32 _sum4 = __msa_ld_w(pp + 16, 0);
            v4i32 _sum5 = __msa_ld_w(pp + 20, 0);
            v4i32 _sum6 = __msa_ld_w(pp + 24, 0);
            v4i32 _sum7 = __msa_ld_w(pp + 28, 0);

            _sum6 = __msa_shf_w(_sum6, _MSA_SHUFFLE(1, 0, 3, 2));
            _sum7 = __msa_shf_w(_sum7, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum4, _sum5, _sum6, _sum7);
            _sum5 = __msa_shf_w(_sum5, _MSA_SHUFFLE(2, 1, 0, 3));
            _sum6 = __msa_shf_w(_sum6, _MSA_SHUFFLE(1, 0, 3, 2));
            _sum7 = __msa_shf_w(_sum7, _MSA_SHUFFLE(0, 3, 2, 1));

            v4f32 _f0 = (v4f32)__msa_ffint_s_w(_sum0);
            v4f32 _f4 = (v4f32)__msa_ffint_s_w(_sum4);
            v4f32 _f1 = (v4f32)__msa_ffint_s_w(_sum1);
            v4f32 _f5 = (v4f32)__msa_ffint_s_w(_sum5);
            v4f32 _f2 = (v4f32)__msa_ffint_s_w(_sum2);
            v4f32 _f6 = (v4f32)__msa_ffint_s_w(_sum6);
            v4f32 _f3 = (v4f32)__msa_ffint_s_w(_sum3);
            v4f32 _f7 = (v4f32)__msa_ffint_s_w(_sum7);

            _f0 = __msa_fmul_w(_f0, (v4f32)__msa_splati_w((v4i32)_descale0, 0));
            _f4 = __msa_fmul_w(_f4, (v4f32)__msa_splati_w((v4i32)_descale0, 0));
            _f1 = __msa_fmul_w(_f1, (v4f32)__msa_splati_w((v4i32)_descale0, 1));
            _f5 = __msa_fmul_w(_f5, (v4f32)__msa_splati_w((v4i32)_descale0, 1));
            _f2 = __msa_fmul_w(_f2, (v4f32)__msa_splati_w((v4i32)_descale0, 2));
            _f6 = __msa_fmul_w(_f6, (v4f32)__msa_splati_w((v4i32)_descale0, 2));
            _f3 = __msa_fmul_w(_f3, (v4f32)__msa_splati_w((v4i32)_descale0, 3));
            _f7 = __msa_fmul_w(_f7, (v4f32)__msa_splati_w((v4i32)_descale0, 3));

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    v4f32 _c0 = __msa_fill_w_f32(pC[0] * beta);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f4 = __msa_fadd_w(_f4, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                    _f5 = __msa_fadd_w(_f5, _c0);
                    _f2 = __msa_fadd_w(_f2, _c0);
                    _f6 = __msa_fadd_w(_f6, _c0);
                    _f3 = __msa_fadd_w(_f3, _c0);
                    _f7 = __msa_fadd_w(_f7, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c0 = __msa_fill_w_f32(pC[0] * beta);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f4 = __msa_fadd_w(_f4, _c0);
                    _c0 = __msa_fill_w_f32(pC[1] * beta);
                    _f1 = __msa_fadd_w(_f1, _c0);
                    _f5 = __msa_fadd_w(_f5, _c0);
                    _c0 = __msa_fill_w_f32(pC[2] * beta);
                    _f2 = __msa_fadd_w(_f2, _c0);
                    _f6 = __msa_fadd_w(_f6, _c0);
                    _c0 = __msa_fill_w_f32(pC[3] * beta);
                    _f3 = __msa_fadd_w(_f3, _c0);
                    _f7 = __msa_fadd_w(_f7, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta));
                    _f4 = __msa_fadd_w(_f4, __msa_fmul_w((v4f32)__msa_ld_w(pC + 4, 0), _beta));
                    _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep, 0), _beta));
                    _f5 = __msa_fadd_w(_f5, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep + 4, 0), _beta));
                    _f2 = __msa_fadd_w(_f2, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 2, 0), _beta));
                    _f6 = __msa_fadd_w(_f6, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 2 + 4, 0), _beta));
                    _f3 = __msa_fadd_w(_f3, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 3, 0), _beta));
                    _f7 = __msa_fadd_w(_f7, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 3 + 4, 0), _beta));
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    v4f32 _c0 = __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta);
                    v4f32 _c1 = __msa_fmul_w((v4f32)__msa_ld_w(pC + 4, 0), _beta);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f4 = __msa_fadd_w(_f4, _c1);
                    _f1 = __msa_fadd_w(_f1, _c0);
                    _f5 = __msa_fadd_w(_f5, _c1);
                    _f2 = __msa_fadd_w(_f2, _c0);
                    _f6 = __msa_fadd_w(_f6, _c1);
                    _f3 = __msa_fadd_w(_f3, _c0);
                    _f7 = __msa_fadd_w(_f7, _c1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f4 = __msa_fmul_w(_f4, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
                _f5 = __msa_fmul_w(_f5, _alpha);
                _f2 = __msa_fmul_w(_f2, _alpha);
                _f6 = __msa_fmul_w(_f6, _alpha);
                _f3 = __msa_fmul_w(_f3, _alpha);
                _f7 = __msa_fmul_w(_f7, _alpha);
            }

            if (output_transpose)
            {
                transpose4x4_ps(_f0, _f1, _f2, _f3);
                transpose4x4_ps(_f4, _f5, _f6, _f7);

                __msa_st_w((v4i32)_f0, p0, 0);
                __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
                __msa_st_w((v4i32)_f2, p0 + out_hstep * 2, 0);
                __msa_st_w((v4i32)_f3, p0 + out_hstep * 3, 0);
                __msa_st_w((v4i32)_f4, p0 + out_hstep * 4, 0);
                __msa_st_w((v4i32)_f5, p0 + out_hstep * 5, 0);
                __msa_st_w((v4i32)_f6, p0 + out_hstep * 6, 0);
                __msa_st_w((v4i32)_f7, p0 + out_hstep * 7, 0);
                p0 += out_hstep * 8;
            }
            else
            {
                __msa_st_w((v4i32)_f0, p0, 0);
                __msa_st_w((v4i32)_f4, p0 + 4, 0);
                __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
                __msa_st_w((v4i32)_f5, p0 + out_hstep + 4, 0);
                __msa_st_w((v4i32)_f2, p0 + out_hstep * 2, 0);
                __msa_st_w((v4i32)_f6, p0 + out_hstep * 2 + 4, 0);
                __msa_st_w((v4i32)_f3, p0 + out_hstep * 3, 0);
                __msa_st_w((v4i32)_f7, p0 + out_hstep * 3 + 4, 0);
                p0 += 8;
            }
            pp += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __builtin_prefetch(pp + 16);
            v4i32 _sum0 = __msa_ld_w(pp, 0);
            v4i32 _sum1 = __msa_ld_w(pp + 4, 0);
            v4i32 _sum2 = __msa_ld_w(pp + 8, 0);
            v4i32 _sum3 = __msa_ld_w(pp + 12, 0);

            _sum2 = __msa_shf_w(_sum2, _MSA_SHUFFLE(1, 0, 3, 2));
            _sum3 = __msa_shf_w(_sum3, _MSA_SHUFFLE(1, 0, 3, 2));
            transpose4x4_epi32(_sum0, _sum1, _sum2, _sum3);
            _sum1 = __msa_shf_w(_sum1, _MSA_SHUFFLE(2, 1, 0, 3));
            _sum2 = __msa_shf_w(_sum2, _MSA_SHUFFLE(1, 0, 3, 2));
            _sum3 = __msa_shf_w(_sum3, _MSA_SHUFFLE(0, 3, 2, 1));

            v4f32 _f0 = (v4f32)__msa_ffint_s_w(_sum0);
            v4f32 _f1 = (v4f32)__msa_ffint_s_w(_sum1);
            v4f32 _f2 = (v4f32)__msa_ffint_s_w(_sum2);
            v4f32 _f3 = (v4f32)__msa_ffint_s_w(_sum3);

            _f0 = __msa_fmul_w(_f0, (v4f32)__msa_splati_w((v4i32)_descale0, 0));
            _f1 = __msa_fmul_w(_f1, (v4f32)__msa_splati_w((v4i32)_descale0, 1));
            _f2 = __msa_fmul_w(_f2, (v4f32)__msa_splati_w((v4i32)_descale0, 2));
            _f3 = __msa_fmul_w(_f3, (v4f32)__msa_splati_w((v4i32)_descale0, 3));

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    v4f32 _c0 = __msa_fill_w_f32(pC[0] * beta);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                    _f2 = __msa_fadd_w(_f2, _c0);
                    _f3 = __msa_fadd_w(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(pC[0] * beta));
                    _f1 = __msa_fadd_w(_f1, __msa_fill_w_f32(pC[1] * beta));
                    _f2 = __msa_fadd_w(_f2, __msa_fill_w_f32(pC[2] * beta));
                    _f3 = __msa_fadd_w(_f3, __msa_fill_w_f32(pC[3] * beta));
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta));
                    _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep, 0), _beta));
                    _f2 = __msa_fadd_w(_f2, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 2, 0), _beta));
                    _f3 = __msa_fadd_w(_f3, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep * 3, 0), _beta));
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c0 = __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                    _f2 = __msa_fadd_w(_f2, _c0);
                    _f3 = __msa_fadd_w(_f3, _c0);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
                _f2 = __msa_fmul_w(_f2, _alpha);
                _f3 = __msa_fmul_w(_f3, _alpha);
            }

            if (output_transpose)
            {
                transpose4x4_ps(_f0, _f1, _f2, _f3);

                __msa_st_w((v4i32)_f0, p0, 0);
                __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
                __msa_st_w((v4i32)_f2, p0 + out_hstep * 2, 0);
                __msa_st_w((v4i32)_f3, p0 + out_hstep * 3, 0);
                p0 += out_hstep * 4;
            }
            else
            {
                __msa_st_w((v4i32)_f0, p0, 0);
                __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
                __msa_st_w((v4i32)_f2, p0 + out_hstep * 2, 0);
                __msa_st_w((v4i32)_f3, p0 + out_hstep * 3, 0);
                p0 += 4;
            }
            pp += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            __builtin_prefetch(pp + 8);
            v4i32 _sum0 = __msa_ld_w(pp, 0);
            v4i32 _sum1 = __msa_ld_w(pp + 4, 0);

            v4i32 _sum0e = __msa_shf_w(_sum0, _MSA_SHUFFLE(3, 1, 2, 0));
            v4i32 _sum0o = __msa_shf_w(_sum0, _MSA_SHUFFLE(2, 0, 3, 1));
            v4i32 _sum1e = __msa_shf_w(_sum1, _MSA_SHUFFLE(3, 1, 2, 0));
            v4i32 _sum1o = __msa_shf_w(_sum1, _MSA_SHUFFLE(2, 0, 3, 1));

            v4f32 _f0 = (v4f32)__msa_ffint_s_w((v4i32)__msa_ilvr_w(_sum1o, _sum0e));
            v4f32 _f1 = (v4f32)__msa_ffint_s_w((v4i32)__msa_ilvr_w(_sum0o, _sum1e));

            _f0 = __msa_fmul_w(_f0, _descale0);
            _f1 = __msa_fmul_w(_f1, _descale0);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    v4f32 _c0 = __msa_fill_w_f32(pC[0] * beta);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c0 = __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    v4i32 _c0 = __msa_fill_w(((const int*)pC)[0]);
                    _c0 = __msa_insert_w(_c0, 1, ((const int*)(pC + c_hstep))[0]);
                    _c0 = __msa_insert_w(_c0, 2, ((const int*)(pC + c_hstep * 2))[0]);
                    _c0 = __msa_insert_w(_c0, 3, ((const int*)(pC + c_hstep * 3))[0]);
                    v4i32 _c1 = __msa_fill_w(((const int*)pC)[1]);
                    _c1 = __msa_insert_w(_c1, 1, ((const int*)(pC + c_hstep))[1]);
                    _c1 = __msa_insert_w(_c1, 2, ((const int*)(pC + c_hstep * 2))[1]);
                    _c1 = __msa_insert_w(_c1, 3, ((const int*)(pC + c_hstep * 3))[1]);
                    _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)_c0, _beta));
                    _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)_c1, _beta));
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(pC[0] * beta));
                    _f1 = __msa_fadd_w(_f1, __msa_fill_w_f32(pC[1] * beta));
                    pC += 2;
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
            }

            if (output_transpose)
            {
                __msa_st_w((v4i32)_f0, p0, 0);
                __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
                p0 += out_hstep * 2;
            }
            else
            {
                v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_f1, (v4i32)_f0);
                v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_f1, (v4i32)_f0);

                *(int64_t*)p0 = __msa_copy_s_d((v2i64)_tmp0, 0);
                *(int64_t*)(p0 + out_hstep) = __msa_copy_s_d((v2i64)_tmp0, 1);
                *(int64_t*)(p0 + out_hstep * 2) = __msa_copy_s_d((v2i64)_tmp1, 0);
                *(int64_t*)(p0 + out_hstep * 3) = __msa_copy_s_d((v2i64)_tmp1, 1);
                p0 += 2;
            }
            pp += 8;
        }
        for (; jj < max_jj; jj++)
        {
            __builtin_prefetch(pp + 4);
            v4f32 _f0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(pp, 0));

            _f0 = __msa_fmul_w(_f0, _descale0);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(pC[0] * beta));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), __msa_fill_w_f32(beta)));
                }
                if (broadcast_type_C == 3)
                {
                    v4i32 _c0 = __msa_fill_w(((const int*)pC)[0]);
                    _c0 = __msa_insert_w(_c0, 1, ((const int*)(pC + c_hstep))[0]);
                    _c0 = __msa_insert_w(_c0, 2, ((const int*)(pC + c_hstep * 2))[0]);
                    _c0 = __msa_insert_w(_c0, 3, ((const int*)(pC + c_hstep * 3))[0]);
                    _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)_c0, __msa_fill_w_f32(beta)));
                    pC += 1;
                }
                if (broadcast_type_C == 4)
                {
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(pC[0] * beta));
                    pC += 1;
                }
            }

            if (alpha != 1.f)
            {
                _f0 = __msa_fmul_w(_f0, __msa_fill_w_f32(alpha));
            }

            if (output_transpose)
            {
                __msa_st_w((v4i32)_f0, p0, 0);
                p0 += out_hstep;
            }
            else
            {
                *(int*)p0 = __msa_copy_s_w((v4i32)_f0, 0);
                *(int*)(p0 + out_hstep) = __msa_copy_s_w((v4i32)_f0, 1);
                *(int*)(p0 + out_hstep * 2) = __msa_copy_s_w((v4i32)_f0, 2);
                *(int*)(p0 + out_hstep * 3) = __msa_copy_s_w((v4i32)_f0, 3);
                p0 += 1;
            }
            pp += 4;
        }
    }
#endif // __mips_msa
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + i + ii;
        }
        else
        {
            p0 = (float*)top_blob + (i + ii) * out_hstep + j;
        }

        const float descale0 = descale_ptr[i + ii];
        const float descale1 = descale_ptr[i + ii + 1];

        float c0 = 0.f;
        float c1 = 0.f;
        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
                c1 = pC[1] * beta;
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (size_t)(i + ii) * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if __mips_msa
        v4f32 _descale0 = __msa_fill_w_f32(descale0);
        v4f32 _descale1 = __msa_fill_w_f32(descale1);
        for (; jj + 7 < max_jj; jj += 8)
        {
            __builtin_prefetch(pp + 16);
            v4i32 _sum0 = __msa_ld_w(pp, 0);
            v4i32 _sum1 = __msa_ld_w(pp + 4, 0);
            v4i32 _sum2 = __msa_ld_w(pp + 8, 0);
            v4i32 _sum3 = __msa_ld_w(pp + 12, 0);

            v4i32 _sum00 = __msa_pckev_w(_sum1, _sum0);
            v4i32 _sum10 = __msa_pckod_w(_sum1, _sum0);
            v4i32 _sum01 = __msa_pckev_w(_sum3, _sum2);
            v4i32 _sum11 = __msa_pckod_w(_sum3, _sum2);

            v4f32 _f0 = (v4f32)__msa_ffint_s_w(_sum00);
            v4f32 _f1 = (v4f32)__msa_ffint_s_w(_sum01);
            v4f32 _f2 = (v4f32)__msa_ffint_s_w(_sum10);
            v4f32 _f3 = (v4f32)__msa_ffint_s_w(_sum11);

            _f0 = __msa_fmul_w(_f0, _descale0);
            _f1 = __msa_fmul_w(_f1, _descale0);
            _f2 = __msa_fmul_w(_f2, _descale1);
            _f3 = __msa_fmul_w(_f3, _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    v4f32 _c0 = __msa_fill_w_f32(c0);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                    _f2 = __msa_fadd_w(_f2, _c0);
                    _f3 = __msa_fadd_w(_f3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(c0));
                    _f1 = __msa_fadd_w(_f1, __msa_fill_w_f32(c0));
                    _f2 = __msa_fadd_w(_f2, __msa_fill_w_f32(c1));
                    _f3 = __msa_fadd_w(_f3, __msa_fill_w_f32(c1));
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta));
                    _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)__msa_ld_w(pC + 4, 0), _beta));
                    _f2 = __msa_fadd_w(_f2, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep, 0), _beta));
                    _f3 = __msa_fadd_w(_f3, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep + 4, 0), _beta));
                    pC += 8;
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    v4f32 _c0 = __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta);
                    v4f32 _c1 = __msa_fmul_w((v4f32)__msa_ld_w(pC + 4, 0), _beta);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c1);
                    _f2 = __msa_fadd_w(_f2, _c0);
                    _f3 = __msa_fadd_w(_f3, _c1);
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
                _f2 = __msa_fmul_w(_f2, _alpha);
                _f3 = __msa_fmul_w(_f3, _alpha);
            }

            if (output_transpose)
            {
                v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_f2, (v4i32)_f0);
                v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_f2, (v4i32)_f0);
                v4f32 _tmp2 = (v4f32)__msa_ilvr_w((v4i32)_f3, (v4i32)_f1);
                v4f32 _tmp3 = (v4f32)__msa_ilvl_w((v4i32)_f3, (v4i32)_f1);

                *(int64_t*)p0 = __msa_copy_s_d((v2i64)_tmp0, 0);
                *(int64_t*)(p0 + out_hstep) = __msa_copy_s_d((v2i64)_tmp0, 1);
                *(int64_t*)(p0 + out_hstep * 2) = __msa_copy_s_d((v2i64)_tmp1, 0);
                *(int64_t*)(p0 + out_hstep * 3) = __msa_copy_s_d((v2i64)_tmp1, 1);
                *(int64_t*)(p0 + out_hstep * 4) = __msa_copy_s_d((v2i64)_tmp2, 0);
                *(int64_t*)(p0 + out_hstep * 5) = __msa_copy_s_d((v2i64)_tmp2, 1);
                *(int64_t*)(p0 + out_hstep * 6) = __msa_copy_s_d((v2i64)_tmp3, 0);
                *(int64_t*)(p0 + out_hstep * 7) = __msa_copy_s_d((v2i64)_tmp3, 1);
                p0 += out_hstep * 8;
            }
            else
            {
                __msa_st_w((v4i32)_f0, p0, 0);
                __msa_st_w((v4i32)_f1, p0 + 4, 0);
                __msa_st_w((v4i32)_f2, p0 + out_hstep, 0);
                __msa_st_w((v4i32)_f3, p0 + out_hstep + 4, 0);
                p0 += 8;
            }
            pp += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __builtin_prefetch(pp + 8);
            v4i32 _sum0 = __msa_ld_w(pp, 0);
            v4i32 _sum1 = __msa_ld_w(pp + 4, 0);

            v4i32 _sum00 = __msa_pckev_w(_sum1, _sum0);
            v4i32 _sum10 = __msa_pckod_w(_sum1, _sum0);

            v4f32 _f0 = (v4f32)__msa_ffint_s_w(_sum00);
            v4f32 _f1 = (v4f32)__msa_ffint_s_w(_sum10);

            _f0 = __msa_fmul_w(_f0, _descale0);
            _f1 = __msa_fmul_w(_f1, _descale1);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    v4f32 _c0 = __msa_fill_w_f32(c0);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(c0));
                    _f1 = __msa_fadd_w(_f1, __msa_fill_w_f32(c1));
                }
                if (broadcast_type_C == 3)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta));
                    _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)__msa_ld_w(pC + c_hstep, 0), _beta));
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    v4f32 _c0 = __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), __msa_fill_w_f32(beta));
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
            }

            if (output_transpose)
            {
                v4f32 _tmp0 = (v4f32)__msa_ilvr_w((v4i32)_f1, (v4i32)_f0);
                v4f32 _tmp1 = (v4f32)__msa_ilvl_w((v4i32)_f1, (v4i32)_f0);

                *(int64_t*)p0 = __msa_copy_s_d((v2i64)_tmp0, 0);
                *(int64_t*)(p0 + out_hstep) = __msa_copy_s_d((v2i64)_tmp0, 1);
                *(int64_t*)(p0 + out_hstep * 2) = __msa_copy_s_d((v2i64)_tmp1, 0);
                *(int64_t*)(p0 + out_hstep * 3) = __msa_copy_s_d((v2i64)_tmp1, 1);
                p0 += out_hstep * 4;
            }
            else
            {
                __msa_st_w((v4i32)_f0, p0, 0);
                __msa_st_w((v4i32)_f1, p0 + out_hstep, 0);
                p0 += 4;
            }
            pp += 8;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f00 = pp[0] * descale0;
            float f10 = pp[1] * descale1;
            float f01 = pp[2] * descale0;
            float f11 = pp[3] * descale1;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    f00 += c0;
                    f01 += c0;
                    f10 += c0;
                    f11 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f00 += c0;
                    f01 += c0;
                    f10 += c1;
                    f11 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    f00 += pC[0] * beta;
                    f01 += pC[1] * beta;
                    f10 += pC[c_hstep] * beta;
                    f11 += pC[c_hstep + 1] * beta;
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    f00 += pC[0] * beta;
                    f01 += pC[1] * beta;
                    f10 += pC[0] * beta;
                    f11 += pC[1] * beta;
                    pC += 2;
                }
            }

            f00 *= alpha;
            f01 *= alpha;
            f10 *= alpha;
            f11 *= alpha;

            if (output_transpose)
            {
                p0[0] = f00;
                p0[1] = f10;
                p0[out_hstep] = f01;
                p0[out_hstep + 1] = f11;
                p0 += out_hstep * 2;
            }
            else
            {
                p0[0] = f00;
                p0[1] = f01;
                p0[out_hstep] = f10;
                p0[out_hstep + 1] = f11;
                p0 += 2;
            }
            pp += 4;
        }
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0] * descale0;
            float f1 = pp[1] * descale1;

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    f0 += c0;
                    f1 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                    f1 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    f0 += pC[0] * beta;
                    f1 += pC[c_hstep] * beta;
                    pC += 1;
                }
                if (broadcast_type_C == 4)
                {
                    f0 += pC[0] * beta;
                    f1 += pC[0] * beta;
                    pC += 1;
                }
            }

            f0 *= alpha;
            f1 *= alpha;

            if (output_transpose)
            {
                p0[0] = f0;
                p0[1] = f1;
                p0 += out_hstep;
            }
            else
            {
                p0[0] = f0;
                p0[out_hstep] = f1;
                p0 += 1;
            }
            pp += 2;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const int row = i + ii;
        float* p0;
        if (output_transpose)
        {
            p0 = (float*)top_blob + j * out_hstep + row;
        }
        else
        {
            p0 = (float*)top_blob + row * out_hstep + j;
        }

        const float descale = descale_ptr[row];

        float c0 = 0.f;
        const float* pC = C;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0] * beta;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + row;
                c0 = pC[0] * beta;
            }
            if (broadcast_type_C == 3)
            {
                pC = (const float*)C + (size_t)row * c_hstep + j;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)C + j;
            }
        }

        int jj = 0;
#if __mips_msa
        v4f32 _descale = __msa_fill_w_f32(descale);
        for (; jj + 7 < max_jj; jj += 8)
        {
            __builtin_prefetch(pp + 8);
            v4i32 _sum0 = __msa_ld_w(pp, 0);
            v4i32 _sum1 = __msa_ld_w(pp + 4, 0);

            v4f32 _f0 = (v4f32)__msa_ffint_s_w(_sum0);
            v4f32 _f1 = (v4f32)__msa_ffint_s_w(_sum1);

            _f0 = __msa_fmul_w(_f0, _descale);
            _f1 = __msa_fmul_w(_f1, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    v4f32 _c0 = __msa_fill_w_f32(c0);
                    _f0 = __msa_fadd_w(_f0, _c0);
                    _f1 = __msa_fadd_w(_f1, _c0);
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    v4f32 _beta = __msa_fill_w_f32(beta);
                    _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), _beta));
                    _f1 = __msa_fadd_w(_f1, __msa_fmul_w((v4f32)__msa_ld_w(pC + 4, 0), _beta));
                    pC += 8;
                }
            }

            if (alpha != 1.f)
            {
                v4f32 _alpha = __msa_fill_w_f32(alpha);
                _f0 = __msa_fmul_w(_f0, _alpha);
                _f1 = __msa_fmul_w(_f1, _alpha);
            }

            if (output_transpose)
            {
                float sum0[4];
                float sum1[4];
                __msa_st_w((v4i32)_f0, sum0, 0);
                __msa_st_w((v4i32)_f1, sum1, 0);

                p0[0] = sum0[0];
                p0[out_hstep] = sum0[1];
                p0[out_hstep * 2] = sum0[2];
                p0[out_hstep * 3] = sum0[3];
                p0[out_hstep * 4] = sum1[0];
                p0[out_hstep * 5] = sum1[1];
                p0[out_hstep * 6] = sum1[2];
                p0[out_hstep * 7] = sum1[3];
                p0 += out_hstep * 8;
            }
            else
            {
                __msa_st_w((v4i32)_f0, p0, 0);
                __msa_st_w((v4i32)_f1, p0 + 4, 0);
                p0 += 8;
            }
            pp += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            __builtin_prefetch(pp + 4);
            v4i32 _sum0 = __msa_ld_w(pp, 0);
            v4f32 _f0 = (v4f32)__msa_ffint_s_w(_sum0);

            _f0 = __msa_fmul_w(_f0, _descale);

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _f0 = __msa_fadd_w(_f0, __msa_fill_w_f32(c0));
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    _f0 = __msa_fadd_w(_f0, __msa_fmul_w((v4f32)__msa_ld_w(pC, 0), __msa_fill_w_f32(beta)));
                    pC += 4;
                }
            }

            if (alpha != 1.f)
            {
                _f0 = __msa_fmul_w(_f0, __msa_fill_w_f32(alpha));
            }

            if (output_transpose)
            {
                float sum0[4];
                __msa_st_w((v4i32)_f0, sum0, 0);
                p0[0] = sum0[0];
                p0[out_hstep] = sum0[1];
                p0[out_hstep * 2] = sum0[2];
                p0[out_hstep * 3] = sum0[3];
                p0 += out_hstep * 4;
            }
            else
            {
                __msa_st_w((v4i32)_f0, p0, 0);
                p0 += 4;
            }
            pp += 4;
        }
#endif // __mips_msa
        for (; jj + 1 < max_jj; jj += 2)
        {
            float f0 = pp[0] * descale;
            float f1 = pp[1] * descale;

            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                    f1 += c0;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    f0 += pC[0] * beta;
                    f1 += pC[1] * beta;
                    pC += 2;
                }
            }

            f0 *= alpha;
            f1 *= alpha;

            if (output_transpose)
            {
                p0[0] = f0;
                p0[out_hstep] = f1;
                p0 += out_hstep * 2;
            }
            else
            {
                p0[0] = f0;
                p0[1] = f1;
                p0 += 2;
            }
            pp += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0] * descale;
            if (pC)
            {
                if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    f0 += c0;
                }
                if (broadcast_type_C == 3 || broadcast_type_C == 4)
                {
                    f0 += pC[0] * beta;
                    pC += 1;
                }
            }

            f0 *= alpha;

            p0[0] = f0;

            if (output_transpose)
                p0 += out_hstep;
            else
                p0 += 1;
            pp += 1;
        }
    }
}

static void get_optimal_tile_mnk_int8(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    const int l2_cache_size_int8 = (int)(get_cpu_level2_cache_size() / sizeof(signed char));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    {
#if __mips_msa
        int tile_size = (l2_cache_size_int8 - 16) / 8;
        TILE_K = std::max(8, tile_size / 8 * 8);
#else
        int tile_size = (l2_cache_size_int8 - 2) / 3;
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

        if (K > 0)
        {
            int nn_K = (K + TILE_K - 1) / TILE_K;
#if __mips_msa
            TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
#else
            TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif
        }
    }

#if __mips_msa
    TILE_M = 8;
#else
    TILE_M = 2;
#endif
    if (M > 0)
    {
        TILE_M *= std::min(nT, get_physical_cpu_count());
        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __mips_msa
        TILE_M = std::max(8, std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8));
#else
        TILE_M = std::max(2, std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2));
#endif
        if (nT > 1)
        {
#if __mips_msa
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
        }
    }

    if (N > 0)
    {
        int tile_size = TILE_K >= K ? (l2_cache_size_int8 - TILE_M * TILE_K) / std::max(1, TILE_K) : (l2_cache_size_int8 - TILE_M * TILE_K) / std::max(1, TILE_M + TILE_K);
#if __mips_msa
        TILE_N = std::max(8, tile_size / 8 * 8);
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::max(8, std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8));
#else
        TILE_N = std::max(1, tile_size);
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::max(1, std::min(TILE_N, (N + nn_N - 1) / nn_N));
#endif
    }
    else
    {
#if __mips_msa
        TILE_N = 8;
#else
        TILE_N = 1;
#endif
    }

    if (constant_TILE_M > 0)
    {
#if __mips_msa
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif
    }
    if (constant_TILE_N > 0)
    {
#if __mips_msa
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
#else
        TILE_N = constant_TILE_N;
#endif
    }
    if (constant_TILE_K > 0)
    {
#if __mips_msa
        TILE_K = (constant_TILE_K + 7) / 8 * 8;
#else
        TILE_K = (constant_TILE_K + 1) / 2 * 2;
#endif
    }
}

struct gemm_mips_int8_omp_args
{
    int TILE_M;
    int TILE_N;
    int TILE_K;
    int broadcast_type_C;
    int transA;
    int output_transpose;
    float alpha;
    float beta;
};

static int gemm_mips_int8(const Mat& A, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int transA, int transB, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h);
    const int K = transA ? (A.dims == 3 ? A.c : A.h) : A.w;
    const int N = transB ? (B.dims == 3 ? B.c : B.h) : B.w;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat ATX(TILE_K * TILE_M, nn_K, nT, 1u, opt.workspace_allocator);
    if (ATX.empty())
        return -100;

    Mat BT(TILE_K * TILE_N, nn_K, nn_N, 1u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    Mat A_int8_scales(M, 4u, opt.workspace_allocator);
    if (A_int8_scales.empty())
        return -100;

    float B_int8_scale;
    compute_B_int8_scale(B, B_int8_scale);

    Mat output_descales(M, 4u, opt.workspace_allocator);
    if (output_descales.empty())
        return -100;

    const int nn_NK = nn_N * nn_K;
    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min(N - j, TILE_N);
        const int max_kk = std::min(K - k, TILE_K);

        Mat BT_tile = BT.channel(ppj).row_range(ppk, 1);

        if (transB)
            pack_B_tile_fp32_to_int8(B, BT_tile, j, max_jj, k, max_kk, B_int8_scale);
        else
            transpose_pack_B_tile_fp32_to_int8(B, BT_tile, j, max_jj, k, max_kk, B_int8_scale);
    }

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    const struct gemm_mips_int8_omp_args args = {TILE_M, TILE_N, TILE_K, broadcast_type_C, transA, output_transpose, alpha, beta};

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int TILE_M = args.TILE_M;
        const int TILE_N = args.TILE_N;
        const int TILE_K = args.TILE_K;
        const int broadcast_type_C = args.broadcast_type_C;
        const int transA = args.transA;
        const int output_transpose = args.output_transpose;
        const float alpha = args.alpha;
        const float beta = args.beta;

        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min(K - k, TILE_K);

                Mat AT_tile = ATX.channel(get_omp_thread_num()).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                if (j == 0)
                {
                    if (k == 0)
                    {
                        if (transA)
                            transpose_compute_A_tile_int8_scales(A, A_int8_scales, B_int8_scale, output_descales, i, max_ii);
                        else
                            compute_A_tile_int8_scales(A, A_int8_scales, B_int8_scale, output_descales, i, max_ii);
                    }

                    if (transA)
                        transpose_pack_A_tile_fp32_to_int8(A, AT_tile, i, max_ii, k, max_kk, A_int8_scales);
                    else
                        pack_A_tile_fp32_to_int8(A, AT_tile, i, max_ii, k, max_kk, A_int8_scales);
                }

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_int32_to_fp32(topT_tile, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, output_descales, alpha, beta, output_transpose);
        }
    }

    return 0;
}

static int gemm_AT_mips_int8(const Mat& AT, const Mat& A_int8_scales, const Mat& B, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int K, int transB, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int N = transB ? (B.dims == 3 ? B.c : B.h) : B.w;

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat BT(TILE_K * TILE_N, nn_K, nn_N, 1u, opt.workspace_allocator);
    if (BT.empty())
        return -100;

    float B_int8_scale;
    compute_B_int8_scale(B, B_int8_scale);

    Mat output_descales(M, 4u, opt.workspace_allocator);
    if (output_descales.empty())
        return -100;

    for (int i = 0; i < M; i++)
    {
        output_descales[i] = 1.f / (A_int8_scales[i] * B_int8_scale);
    }

    const int nn_NK = nn_N * nn_K;
    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min(N - j, TILE_N);
        const int max_kk = std::min(K - k, TILE_K);

        Mat BT_tile = BT.channel(ppj).row_range(ppk, 1);

        if (transB)
            pack_B_tile_fp32_to_int8(B, BT_tile, j, max_jj, k, max_kk, B_int8_scale);
        else
            transpose_pack_B_tile_fp32_to_int8(B, BT_tile, j, max_jj, k, max_kk, B_int8_scale);
    }

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    const struct gemm_mips_int8_omp_args args = {TILE_M, TILE_N, TILE_K, broadcast_type_C, 0, output_transpose, alpha, beta};

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int TILE_M = args.TILE_M;
        const int TILE_N = args.TILE_N;
        const int TILE_K = args.TILE_K;
        const int broadcast_type_C = args.broadcast_type_C;
        const int output_transpose = args.output_transpose;
        const float alpha = args.alpha;
        const float beta = args.beta;

        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min(K - k, TILE_K);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_int32_to_fp32(topT_tile, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, output_descales, alpha, beta, output_transpose);
        }
    }

    return 0;
}

static int gemm_BT_mips_int8(const Mat& A, const Mat& BT, float B_int8_scale, const Mat& C, Mat& top_blob, int broadcast_type_C, int N, int K, int transA, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    const int M = transA ? A.w : (A.dims == 3 ? A.c : A.h);

    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    Mat A_int8_scales(M, 4u, opt.workspace_allocator);
    if (A_int8_scales.empty())
        return -100;

    Mat output_descales(M, 4u, opt.workspace_allocator);
    if (output_descales.empty())
        return -100;

    Mat ATX(TILE_K * TILE_M, nn_K, nT, 1u, opt.workspace_allocator);
    if (ATX.empty())
        return -100;

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    const struct gemm_mips_int8_omp_args args = {TILE_M, TILE_N, TILE_K, broadcast_type_C, transA, output_transpose, alpha, beta};

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int TILE_M = args.TILE_M;
        const int TILE_N = args.TILE_N;
        const int TILE_K = args.TILE_K;
        const int broadcast_type_C = args.broadcast_type_C;
        const int transA = args.transA;
        const int output_transpose = args.output_transpose;
        const float alpha = args.alpha;
        const float beta = args.beta;

        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min(K - k, TILE_K);

                Mat AT_tile = ATX.channel(get_omp_thread_num()).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                if (j == 0)
                {
                    if (k == 0)
                    {
                        if (transA)
                            transpose_compute_A_tile_int8_scales(A, A_int8_scales, B_int8_scale, output_descales, i, max_ii);
                        else
                            compute_A_tile_int8_scales(A, A_int8_scales, B_int8_scale, output_descales, i, max_ii);
                    }

                    if (transA)
                        transpose_pack_A_tile_fp32_to_int8(A, AT_tile, i, max_ii, k, max_kk, A_int8_scales);
                    else
                        pack_A_tile_fp32_to_int8(A, AT_tile, i, max_ii, k, max_kk, A_int8_scales);
                }

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_int32_to_fp32(topT_tile, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, output_descales, alpha, beta, output_transpose);
        }
    }

    return 0;
}

static int gemm_AT_BT_mips_int8(const Mat& AT, const Mat& A_int8_scales, const Mat& BT, float B_int8_scale, const Mat& C, Mat& top_blob, int broadcast_type_C, int M, int N, int K, int output_transpose, float alpha, float beta, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int nT, const Option& opt)
{
    int TILE_M, TILE_N, TILE_K;
    get_optimal_tile_mnk_int8(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    Mat output_descales(M, 4u, opt.workspace_allocator);
    if (output_descales.empty())
        return -100;

    for (int i = 0; i < M; i++)
    {
        output_descales[i] = 1.f / (A_int8_scales[i] * B_int8_scale);
    }

    Mat topT(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);
    if (topT.empty())
        return -100;

    const struct gemm_mips_int8_omp_args args = {TILE_M, TILE_N, TILE_K, broadcast_type_C, 0, output_transpose, alpha, beta};

    #pragma omp parallel for num_threads(nT)
    for (int ppi = 0; ppi < nn_M; ppi++)
    {
        const int TILE_M = args.TILE_M;
        const int TILE_N = args.TILE_N;
        const int TILE_K = args.TILE_K;
        const int broadcast_type_C = args.broadcast_type_C;
        const int output_transpose = args.output_transpose;
        const float alpha = args.alpha;
        const float beta = args.beta;

        const int i = ppi * TILE_M;
        const int max_ii = std::min(M - i, TILE_M);

        Mat topT_tile = topT.channel(get_omp_thread_num());

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min(N - j, TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min(K - k, TILE_K);

                Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);
                Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
            }

            unpack_output_tile_int32_to_fp32(topT_tile, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, output_descales, alpha, beta, output_transpose);
        }
    }

    return 0;
}
