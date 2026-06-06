// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void pack_B_tile_fp16sa(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
    const size_t vl16 = __riscv_vsetvl_e16m2(16);
    const size_t vl8 = __riscv_vsetvl_e16m1(8);
    const size_t vl4 = __riscv_vsetvl_e16m1(4);
#endif

    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + 15 < max_jj; jj += 16)
    {
        if (elempack == packn)
        {
            const int q = (j + jj) / packn * packn;
            const int r = (j + jj) % packn;
            const unsigned short* p0 = (const unsigned short*)B + q * B_hstep + k * packn + r;

            if (packn >= 16)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    __riscv_vse16_v_u16m1(pp, __riscv_vle16_v_u16m1(p0, vl16), vl16);
                    pp += 16;
                    p0 += packn;
                }
            }
            if (packn == 8)
            {
                const unsigned short* p1 = (const unsigned short*)B + (q + 8) * B_hstep + k * 8;

                for (int kk = 0; kk < max_kk; kk++)
                {
                    __riscv_vse16_v_u16m1(pp, __riscv_vle16_v_u16m1(p0, vl8), vl8);
                    __riscv_vse16_v_u16m1(pp + 8, __riscv_vle16_v_u16m1(p1, vl8), vl8);
                    pp += 16;
                    p0 += 8;
                    p1 += 8;
                }
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __riscv_vse16_v_u16m2(pp, __riscv_vlse16_v_u16m2(p0, B_hstep * sizeof(unsigned short), vl16), vl16);
                pp += 16;
                p0++;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == packn)
        {
            const int q = (j + jj) / packn * packn;
            const int r = (j + jj) % packn;
            const unsigned short* p0 = (const unsigned short*)B + q * B_hstep + k * packn + r;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vle16_v_u16m1(p0, vl8), vl8);
                pp += 8;
                p0 += packn;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vlse16_v_u16m1(p0, B_hstep * sizeof(unsigned short), vl8), vl8);
                pp += 8;
                p0++;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == packn)
        {
            const int q = (j + jj) / packn * packn;
            const int r = (j + jj) % packn;
            const unsigned short* p0 = (const unsigned short*)B + q * B_hstep + k * packn + r;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vle16_v_u16m1(p0, vl4), vl4);
                pp += 4;
                p0 += packn;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

            for (int kk = 0; kk < max_kk; kk++)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vlse16_v_u16m1(p0, B_hstep * sizeof(unsigned short), vl4), vl4);
                pp += 4;
                p0++;
            }
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            const int q = (j + jj) / packn * packn;
            const int r = (j + jj) % packn;
            const unsigned short* p0 = (const unsigned short*)B + q * B_hstep + k * packn + r;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += packn;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;

            int kk = 0;
#if __riscv_vector
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                vuint16m1_t v0 = __riscv_vle16_v_u16m1(p0, vl);
                vuint16m1_t v1 = __riscv_vle16_v_u16m1(p1, vl);
                __riscv_vsseg2e16_v_u16m1x2(pp, __riscv_vcreate_v_u16m1x2(v0, v1), vl);
                pp += packn * 2;
                p0 += packn;
                p1 += packn;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0++;
                p1++;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            const int q = (j + jj) / packn * packn;
            const int r = (j + jj) % packn;
            const unsigned short* p0 = (const unsigned short*)B + q * B_hstep + k * packn + r;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += packn;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

            int kk = 0;
#if __riscv_vector
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vle16_v_u16m1(p0, vl), vl);
                pp += packn;
                p0 += packn;
            }
#endif // __riscv_vector
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_B_tile_fp16sa(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
    const size_t vl16 = __riscv_vsetvl_e16m2(16);
    const size_t vl8 = __riscv_vsetvl_e16m1(8);
    const size_t vl4 = __riscv_vsetvl_e16m1(4);
#endif

    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + 15 < max_jj; jj += 16)
    {
        if (elempack == packn)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * packn;

            if (packn >= 16)
            {
                int kk = 0;
                for (; kk + (packn - 1) < max_kk; kk += packn)
                {
                    // transposeNx16
                    for (int l = 0; l < packn; l++)
                    {
                        __riscv_vse16_v_u16m1(pp, __riscv_vlse16_v_u16m1(p0 + l, packn * sizeof(unsigned short), vl16), vl16);
                        pp += 16;
                    }

                    p0 += B_hstep * packn;
                }
            }
            if (packn == 8)
            {
                const unsigned short* p1 = p0 + 8 * 8;

                int kk = 0;
                for (; kk + 7 < max_kk; kk += 8)
                {
                    // transpose8x16
                    for (int l = 0; l < 8; l++)
                    {
                        __riscv_vse16_v_u16m1(pp, __riscv_vlse16_v_u16m1(p0 + l, 8 * sizeof(unsigned short), vl8), vl8);
                        __riscv_vse16_v_u16m1(pp + 8, __riscv_vlse16_v_u16m1(p1 + l, 8 * sizeof(unsigned short), vl8), vl8);
                        pp += 16;
                    }

                    p0 += B_hstep * 8;
                    p1 += B_hstep * 8;
                }
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            for (int kk = 0; kk < max_kk; kk++)
            {
                __riscv_vse16_v_u16m2(pp, __riscv_vle16_v_u16m2(p0, vl16), vl16);
                pp += 16;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == packn)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                // transposeNx8
                for (int l = 0; l < packn; l++)
                {
                    __riscv_vse16_v_u16m1(pp, __riscv_vlse16_v_u16m1(p0 + l, packn * sizeof(unsigned short), vl8), vl8);
                    pp += 8;
                }

                p0 += B_hstep * packn;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vle16_v_u16m1(p0, vl8), vl8);
                pp += 8;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == packn)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                // transposeNx4
                for (int l = 0; l < packn; l++)
                {
                    __riscv_vse16_v_u16m1(pp, __riscv_vlse16_v_u16m1(p0 + l, packn * sizeof(unsigned short), vl4), vl4);
                    pp += 4;
                }
                p0 += B_hstep * packn;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vle16_v_u16m1(p0, vl4), vl4);
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                vuint16m1_t v0 = __riscv_vle16_v_u16m1(p0, vl);
                vuint16m1_t v1 = __riscv_vle16_v_u16m1(p0 + packn, vl);
                __riscv_vsseg2e16_v_u16m1x2(pp, __riscv_vcreate_v_u16m1x2(v0, v1), vl);
                pp += packn * 2;
                p0 += B_hstep * packn;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += B_hstep;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * packn;

            int kk = 0;
            for (; kk + (packn - 1) < max_kk; kk += packn)
            {
                __riscv_vse16_v_u16m1(pp, __riscv_vle16_v_u16m1(p0, vl), vl);
                pp += packn;
                p0 += B_hstep * packn;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += B_hstep;
            }
        }
    }
}

static void transpose_unpack_output_tile_fp16sa(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
    const size_t vl16 = __riscv_vsetvl_e16m2(16);
    const size_t vl8 = __riscv_vsetvl_e16m1(8);
    const size_t vl4 = __riscv_vsetvl_e16m1(4);
#endif

#if __riscv_vector
    const int out_elempack = top_blob.elempack;
#endif
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const __fp16* pp = topT;

    int ii = 0;
#if __riscv_vector
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        if (out_elempack == packn)
        {
            int jj = 0;

            const int r0 = j % packn;
            if (r0 != 0)
            {
                const int nn = std::min(packn - r0, max_jj);
                __fp16* p0 = (__fp16*)top_blob + (j / packn * packn) * out_hstep + r0 + (i + ii) * packn;

                for (; jj < nn; jj++)
                {
                    __riscv_vsse16_v_f16m1(p0, packn * sizeof(__fp16), __riscv_vle16_v_f16m1(pp, vl), vl);
                    pp += packn;
                    p0++;
                }
            }

            __fp16* p0 = (__fp16*)top_blob + (j + jj) * out_hstep + (i + ii) * packn;

            for (; jj + (packn - 1) < max_jj; jj += packn)
            {
                // transposeNxN
                for (int l = 0; l < packn; l++)
                {
                    __riscv_vsse16_v_f16m1(p0 + l, packn * sizeof(__fp16), __riscv_vle16_v_f16m1(pp, vl), vl);
                    pp += packn;
                }

                p0 += out_hstep * packn;
            }

            for (; jj < max_jj; jj++)
            {
                __riscv_vsse16_v_f16m1(p0, packn * sizeof(__fp16), __riscv_vle16_v_f16m1(pp, vl), vl);
                pp += packn;
                p0++;
            }
        }
        if (out_elempack == 1)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                __riscv_vse16_v_f16m1(p0, __riscv_vle16_v_f16m1(pp, vl), vl);
                pp += packn;
                p0 += out_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __riscv_vector
        if (out_elempack == packn)
        {
            int jj = 0;

            for (; jj + 15 < max_jj; jj += 16)
            {
                const int out_j = j + jj;
                __fp16* p0 = (__fp16*)top_blob + (out_j / packn * packn) * out_hstep + out_j % packn + (i + ii) * packn;
                if (packn == 8)
                {
                    __riscv_vse16_v_f16m1(p0, __riscv_vle16_v_f16m1(pp, vl8), vl8);
                    __riscv_vse16_v_f16m1(p0 + packn, __riscv_vle16_v_f16m1(pp + 16, vl8), vl8);
                    p0 += out_hstep * 8;
                    __riscv_vse16_v_f16m1(p0, __riscv_vle16_v_f16m1(pp + 8, vl8), vl8);
                    __riscv_vse16_v_f16m1(p0 + packn, __riscv_vle16_v_f16m1(pp + 24, vl8), vl8);
                }
                if (packn >= 16)
                {
                    __riscv_vse16_v_f16m2(p0, __riscv_vle16_v_f16m2(pp, vl16), vl16);
                    __riscv_vse16_v_f16m2(p0 + packn, __riscv_vle16_v_f16m2(pp + 16, vl16), vl16);
                }
                pp += 16 * 2;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const int out_j = j + jj;
                __fp16* p0 = (__fp16*)top_blob + (out_j / packn * packn) * out_hstep + out_j % packn + (i + ii) * packn;
                __riscv_vse16_v_f16m1(p0, __riscv_vle16_v_f16m1(pp, vl8), vl8);
                __riscv_vse16_v_f16m1(p0 + packn, __riscv_vle16_v_f16m1(pp + 8, vl8), vl8);
                pp += 8 * 2;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const int out_j = j + jj;
                __fp16* p0 = (__fp16*)top_blob + (out_j / packn * packn) * out_hstep + out_j % packn + (i + ii) * packn;
                __riscv_vse16_v_f16m1(p0, __riscv_vle16_v_f16m1(pp, vl4), vl4);
                __riscv_vse16_v_f16m1(p0 + packn, __riscv_vle16_v_f16m1(pp + 4, vl4), vl4);
                pp += 4 * 2;
            }
            for (; jj + 1 < max_jj; jj += 2)
            {
                const int out_j = j + jj;
                __fp16* p0 = (__fp16*)top_blob + (out_j / packn * packn) * out_hstep + out_j % packn + (i + ii) * packn;
                p0[0] = pp[0];
                p0[1] = pp[1];
                p0[packn] = pp[2];
                p0[packn + 1] = pp[3];
                pp += 2 * 2;
            }
            for (; jj < max_jj; jj += 1)
            {
                const int out_j = j + jj;
                __fp16* p0 = (__fp16*)top_blob + (out_j / packn * packn) * out_hstep + out_j % packn + (i + ii) * packn;
                p0[0] = pp[0];
                p0[packn] = pp[1];
                pp += 2;
            }
        }
        if (out_elempack == 1)
#endif // __riscv_vector
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii);

            int jj = 0;
#if __riscv_vector
            for (; jj + 15 < max_jj; jj += 16)
            {
                __riscv_vsse16_v_f16m2(p0, out_hstep * sizeof(__fp16), __riscv_vle16_v_f16m2(pp, vl16), vl16);
                __riscv_vsse16_v_f16m2(p0 + 1, out_hstep * sizeof(__fp16), __riscv_vle16_v_f16m2(pp + 16, vl16), vl16);
                pp += 16 * 2;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                __riscv_vsse16_v_f16m1(p0, out_hstep * sizeof(__fp16), __riscv_vle16_v_f16m1(pp, vl8), vl8);
                __riscv_vsse16_v_f16m1(p0 + 1, out_hstep * sizeof(__fp16), __riscv_vle16_v_f16m1(pp + 8, vl8), vl8);
                pp += 8 * 2;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                __riscv_vsse16_v_f16m1(p0, out_hstep * sizeof(__fp16), __riscv_vle16_v_f16m1(pp, vl4), vl4);
                __riscv_vsse16_v_f16m1(p0 + 1, out_hstep * sizeof(__fp16), __riscv_vle16_v_f16m1(pp + 4, vl4), vl4);
                pp += 4 * 2;
                p0 += out_hstep * 4;
            }
#endif // __riscv_vector
            for (; jj + 1 < max_jj; jj += 2)
            {
                p0[0] = pp[0];
                p0[out_hstep] = pp[1];
                p0[1] = pp[2];
                p0[out_hstep + 1] = pp[3];
                pp += 2 * 2;
                p0 += out_hstep * 2;
            }
            for (; jj < max_jj; jj += 1)
            {
                p0[0] = pp[0];
                p0[1] = pp[1];
                pp += 2;
                p0 += out_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
#if __riscv_vector
        if (out_elempack == packn)
        {
            int jj = 0;

            for (; jj + 15 < max_jj; jj += 16)
            {
                const int out_j = j + jj;
                __fp16* p0 = (__fp16*)top_blob + (out_j / packn * packn) * out_hstep + out_j % packn + (i + ii) * packn;
                if (packn == 8)
                {
                    __riscv_vse16_v_f16m1(p0, __riscv_vle16_v_f16m1(pp, vl8), vl8);
                    p0 += out_hstep * 8;
                    __riscv_vse16_v_f16m1(p0, __riscv_vle16_v_f16m1(pp + 8, vl8), vl8);
                }
                if (packn >= 16)
                {
                    __riscv_vse16_v_f16m2(p0, __riscv_vle16_v_f16m2(pp, vl16), vl16);
                }
                pp += 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                const int out_j = j + jj;
                __fp16* p0 = (__fp16*)top_blob + (out_j / packn * packn) * out_hstep + out_j % packn + (i + ii) * packn;
                __riscv_vse16_v_f16m1(p0, __riscv_vle16_v_f16m1(pp, vl8), vl8);
                pp += 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                const int out_j = j + jj;
                __fp16* p0 = (__fp16*)top_blob + (out_j / packn * packn) * out_hstep + out_j % packn + (i + ii) * packn;
                __riscv_vse16_v_f16m1(p0, __riscv_vle16_v_f16m1(pp, vl4), vl4);
                pp += 4;
            }
            for (; jj < max_jj; jj += 1)
            {
                const int out_j = j + jj;
                __fp16* p0 = (__fp16*)top_blob + (out_j / packn * packn) * out_hstep + out_j % packn + (i + ii) * packn;
                p0[0] = pp[0];
                pp += 1;
            }
        }
        if (out_elempack == 1)
#endif // __riscv_vector
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii);

            int jj = 0;
#if __riscv_vector
            for (; jj + 15 < max_jj; jj += 16)
            {
                __riscv_vsse16_v_f16m2(p0, out_hstep * sizeof(__fp16), __riscv_vle16_v_f16m2(pp, vl16), vl16);
                pp += 16;
                p0 += out_hstep * 16;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                __riscv_vsse16_v_f16m1(p0, out_hstep * sizeof(__fp16), __riscv_vle16_v_f16m1(pp, vl8), vl8);
                pp += 8;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                __riscv_vsse16_v_f16m1(p0, out_hstep * sizeof(__fp16), __riscv_vle16_v_f16m1(pp, vl4), vl4);
                pp += 4;
                p0 += out_hstep * 4;
            }
#endif // __riscv_vector
            for (; jj < max_jj; jj += 1)
            {
                p0[0] = pp[0];
                pp += 1;
                p0 += out_hstep;
            }
        }
    }
}

static void gemm_transB_packed_tile_fp16sa(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, float alpha, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
    const size_t vl16 = __riscv_vsetvl_e16m2(16);
    const size_t vl8 = __riscv_vsetvl_e16m1(8);
    const size_t vl4 = __riscv_vsetvl_e16m1(4);
#endif

#if __riscv_vector
    const int out_elempack = top_blob.elempack;
#endif
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;
    const __fp16 _alpha = (__fp16)alpha;

    const __fp16* pAT = AT_tile;
    const __fp16* pBT = BT_tile;
    const __fp16* pC = CT_tile;

    __fp16* outptr = topT_tile;

    int ii = 0;
#if __riscv_vector
    for (; ii + (packn - 1) < max_ii; ii += packn)
    {
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const __fp16* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const __fp16*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const __fp16*)CT_tile + j;
            }
        }

        int jj = 0;
        for (; jj + 15 < max_jj; jj += 16)
        {
            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;
            vfloat16m1_t _sum2;
            vfloat16m1_t _sum3;
            vfloat16m1_t _sum4;
            vfloat16m1_t _sum5;
            vfloat16m1_t _sum6;
            vfloat16m1_t _sum7;
            vfloat16m1_t _sum8;
            vfloat16m1_t _sum9;
            vfloat16m1_t _suma;
            vfloat16m1_t _sumb;
            vfloat16m1_t _sumc;
            vfloat16m1_t _sumd;
            vfloat16m1_t _sume;
            vfloat16m1_t _sumf;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                        _sum8 = _sum0;
                        _sum9 = _sum0;
                        _suma = _sum0;
                        _sumb = _sum0;
                        _sumc = _sum0;
                        _sumd = _sum0;
                        _sume = _sum0;
                        _sumf = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vle16_v_f16m1(pC, vl);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                        _sum8 = _sum0;
                        _sum9 = _sum0;
                        _suma = _sum0;
                        _sumb = _sum0;
                        _sumc = _sum0;
                        _sumd = _sum0;
                        _sume = _sum0;
                        _sumf = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = __riscv_vle16_v_f16m1(pC, vl);
                        _sum1 = __riscv_vle16_v_f16m1(pC + packn, vl);
                        _sum2 = __riscv_vle16_v_f16m1(pC + packn * 2, vl);
                        _sum3 = __riscv_vle16_v_f16m1(pC + packn * 3, vl);
                        _sum4 = __riscv_vle16_v_f16m1(pC + packn * 4, vl);
                        _sum5 = __riscv_vle16_v_f16m1(pC + packn * 5, vl);
                        _sum6 = __riscv_vle16_v_f16m1(pC + packn * 6, vl);
                        _sum7 = __riscv_vle16_v_f16m1(pC + packn * 7, vl);
                        _sum8 = __riscv_vle16_v_f16m1(pC + packn * 8, vl);
                        _sum9 = __riscv_vle16_v_f16m1(pC + packn * 9, vl);
                        _suma = __riscv_vle16_v_f16m1(pC + packn * 10, vl);
                        _sumb = __riscv_vle16_v_f16m1(pC + packn * 11, vl);
                        _sumc = __riscv_vle16_v_f16m1(pC + packn * 12, vl);
                        _sumd = __riscv_vle16_v_f16m1(pC + packn * 13, vl);
                        _sume = __riscv_vle16_v_f16m1(pC + packn * 14, vl);
                        _sumf = __riscv_vle16_v_f16m1(pC + packn * 15, vl);
                        pC += packn * 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl);
                        _sum1 = __riscv_vfmv_v_f_f16m1(pC[1], vl);
                        _sum2 = __riscv_vfmv_v_f_f16m1(pC[2], vl);
                        _sum3 = __riscv_vfmv_v_f_f16m1(pC[3], vl);
                        _sum4 = __riscv_vfmv_v_f_f16m1(pC[4], vl);
                        _sum5 = __riscv_vfmv_v_f_f16m1(pC[5], vl);
                        _sum6 = __riscv_vfmv_v_f_f16m1(pC[6], vl);
                        _sum7 = __riscv_vfmv_v_f_f16m1(pC[7], vl);
                        _sum8 = __riscv_vfmv_v_f_f16m1(pC[8], vl);
                        _sum9 = __riscv_vfmv_v_f_f16m1(pC[9], vl);
                        _suma = __riscv_vfmv_v_f_f16m1(pC[10], vl);
                        _sumb = __riscv_vfmv_v_f_f16m1(pC[11], vl);
                        _sumc = __riscv_vfmv_v_f_f16m1(pC[12], vl);
                        _sumd = __riscv_vfmv_v_f_f16m1(pC[13], vl);
                        _sume = __riscv_vfmv_v_f_f16m1(pC[14], vl);
                        _sumf = __riscv_vfmv_v_f_f16m1(pC[15], vl);
                        pC += 16;
                    }
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                    _sum8 = _sum0;
                    _sum9 = _sum0;
                    _suma = _sum0;
                    _sumb = _sum0;
                    _sumc = _sum0;
                    _sumd = _sum0;
                    _sume = _sum0;
                    _sumf = _sum0;
                }
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl);
                _sum1 = __riscv_vle16_v_f16m1(outptr + packn, vl);
                _sum2 = __riscv_vle16_v_f16m1(outptr + packn * 2, vl);
                _sum3 = __riscv_vle16_v_f16m1(outptr + packn * 3, vl);
                _sum4 = __riscv_vle16_v_f16m1(outptr + packn * 4, vl);
                _sum5 = __riscv_vle16_v_f16m1(outptr + packn * 5, vl);
                _sum6 = __riscv_vle16_v_f16m1(outptr + packn * 6, vl);
                _sum7 = __riscv_vle16_v_f16m1(outptr + packn * 7, vl);
                _sum8 = __riscv_vle16_v_f16m1(outptr + packn * 8, vl);
                _sum9 = __riscv_vle16_v_f16m1(outptr + packn * 9, vl);
                _suma = __riscv_vle16_v_f16m1(outptr + packn * 10, vl);
                _sumb = __riscv_vle16_v_f16m1(outptr + packn * 11, vl);
                _sumc = __riscv_vle16_v_f16m1(outptr + packn * 12, vl);
                _sumd = __riscv_vle16_v_f16m1(outptr + packn * 13, vl);
                _sume = __riscv_vle16_v_f16m1(outptr + packn * 14, vl);
                _sumf = __riscv_vle16_v_f16m1(outptr + packn * 15, vl);
            }

            const __fp16* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _pA = __riscv_vle16_v_f16m1(pA, vl);
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pB[1], _pA, vl);
                _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pB[2], _pA, vl);
                _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pB[3], _pA, vl);
                _sum4 = __riscv_vfmacc_vf_f16m1(_sum4, pB[4], _pA, vl);
                _sum5 = __riscv_vfmacc_vf_f16m1(_sum5, pB[5], _pA, vl);
                _sum6 = __riscv_vfmacc_vf_f16m1(_sum6, pB[6], _pA, vl);
                _sum7 = __riscv_vfmacc_vf_f16m1(_sum7, pB[7], _pA, vl);
                _sum8 = __riscv_vfmacc_vf_f16m1(_sum8, pB[8], _pA, vl);
                _sum9 = __riscv_vfmacc_vf_f16m1(_sum9, pB[9], _pA, vl);
                _suma = __riscv_vfmacc_vf_f16m1(_suma, pB[10], _pA, vl);
                _sumb = __riscv_vfmacc_vf_f16m1(_sumb, pB[11], _pA, vl);
                _sumc = __riscv_vfmacc_vf_f16m1(_sumc, pB[12], _pA, vl);
                _sumd = __riscv_vfmacc_vf_f16m1(_sumd, pB[13], _pA, vl);
                _sume = __riscv_vfmacc_vf_f16m1(_sume, pB[14], _pA, vl);
                _sumf = __riscv_vfmacc_vf_f16m1(_sumf, pB[15], _pA, vl);
                pA += packn;
                pB += 16;
            }

            if (alpha != 1.f)
            {
                _sum0 = __riscv_vfmul_vf_f16m1(_sum0, _alpha, vl);
                _sum1 = __riscv_vfmul_vf_f16m1(_sum1, _alpha, vl);
                _sum2 = __riscv_vfmul_vf_f16m1(_sum2, _alpha, vl);
                _sum3 = __riscv_vfmul_vf_f16m1(_sum3, _alpha, vl);
                _sum4 = __riscv_vfmul_vf_f16m1(_sum4, _alpha, vl);
                _sum5 = __riscv_vfmul_vf_f16m1(_sum5, _alpha, vl);
                _sum6 = __riscv_vfmul_vf_f16m1(_sum6, _alpha, vl);
                _sum7 = __riscv_vfmul_vf_f16m1(_sum7, _alpha, vl);
                _sum8 = __riscv_vfmul_vf_f16m1(_sum8, _alpha, vl);
                _sum9 = __riscv_vfmul_vf_f16m1(_sum9, _alpha, vl);
                _suma = __riscv_vfmul_vf_f16m1(_suma, _alpha, vl);
                _sumb = __riscv_vfmul_vf_f16m1(_sumb, _alpha, vl);
                _sumc = __riscv_vfmul_vf_f16m1(_sumc, _alpha, vl);
                _sumd = __riscv_vfmul_vf_f16m1(_sumd, _alpha, vl);
                _sume = __riscv_vfmul_vf_f16m1(_sume, _alpha, vl);
                _sumf = __riscv_vfmul_vf_f16m1(_sumf, _alpha, vl);
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse16_v_f16m1(outptr0, _sum0, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn, _sum1, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 2, _sum2, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 3, _sum3, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 4, _sum4, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 5, _sum5, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 6, _sum6, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 7, _sum7, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 8, _sum8, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 9, _sum9, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 10, _suma, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 11, _sumb, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 12, _sumc, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 13, _sumd, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 14, _sume, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 15, _sumf, vl);
                    outptr0 += packn * 16;
                }
                if (out_elempack == 1)
                {
                    vfloat16m1x8_t _sum01 = __riscv_vcreate_v_f16m1x8(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
                    vfloat16m1x8_t _sum23 = __riscv_vcreate_v_f16m1x8(_sum8, _sum9, _suma, _sumb, _sumc, _sumd, _sume, _sumf);
                    __riscv_vssseg8e16_v_f16m1x8(outptr0, out_hstep * sizeof(__fp16), _sum01, vl);
                    __riscv_vssseg8e16_v_f16m1x8(outptr0 + 8, out_hstep * sizeof(__fp16), _sum23, vl);
                    outptr0 += 16;
                }
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl);
                __riscv_vse16_v_f16m1(outptr + packn, _sum1, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 2, _sum2, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 3, _sum3, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 4, _sum4, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 5, _sum5, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 6, _sum6, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 7, _sum7, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 8, _sum8, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 9, _sum9, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 10, _suma, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 11, _sumb, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 12, _sumc, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 13, _sumd, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 14, _sume, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 15, _sumf, vl);
            }

            outptr += packn * 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;
            vfloat16m1_t _sum2;
            vfloat16m1_t _sum3;
            vfloat16m1_t _sum4;
            vfloat16m1_t _sum5;
            vfloat16m1_t _sum6;
            vfloat16m1_t _sum7;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vle16_v_f16m1(pC, vl);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = __riscv_vle16_v_f16m1(pC, vl);
                        _sum1 = __riscv_vle16_v_f16m1(pC + packn, vl);
                        _sum2 = __riscv_vle16_v_f16m1(pC + packn * 2, vl);
                        _sum3 = __riscv_vle16_v_f16m1(pC + packn * 3, vl);
                        _sum4 = __riscv_vle16_v_f16m1(pC + packn * 4, vl);
                        _sum5 = __riscv_vle16_v_f16m1(pC + packn * 5, vl);
                        _sum6 = __riscv_vle16_v_f16m1(pC + packn * 6, vl);
                        _sum7 = __riscv_vle16_v_f16m1(pC + packn * 7, vl);
                        pC += packn * 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl);
                        _sum1 = __riscv_vfmv_v_f_f16m1(pC[1], vl);
                        _sum2 = __riscv_vfmv_v_f_f16m1(pC[2], vl);
                        _sum3 = __riscv_vfmv_v_f_f16m1(pC[3], vl);
                        _sum4 = __riscv_vfmv_v_f_f16m1(pC[4], vl);
                        _sum5 = __riscv_vfmv_v_f_f16m1(pC[5], vl);
                        _sum6 = __riscv_vfmv_v_f_f16m1(pC[6], vl);
                        _sum7 = __riscv_vfmv_v_f_f16m1(pC[7], vl);
                        pC += 8;
                    }
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                }
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl);
                _sum1 = __riscv_vle16_v_f16m1(outptr + packn, vl);
                _sum2 = __riscv_vle16_v_f16m1(outptr + packn * 2, vl);
                _sum3 = __riscv_vle16_v_f16m1(outptr + packn * 3, vl);
                _sum4 = __riscv_vle16_v_f16m1(outptr + packn * 4, vl);
                _sum5 = __riscv_vle16_v_f16m1(outptr + packn * 5, vl);
                _sum6 = __riscv_vle16_v_f16m1(outptr + packn * 6, vl);
                _sum7 = __riscv_vle16_v_f16m1(outptr + packn * 7, vl);
            }

            const __fp16* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _pA = __riscv_vle16_v_f16m1(pA, vl);
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pB[1], _pA, vl);
                _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pB[2], _pA, vl);
                _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pB[3], _pA, vl);
                _sum4 = __riscv_vfmacc_vf_f16m1(_sum4, pB[4], _pA, vl);
                _sum5 = __riscv_vfmacc_vf_f16m1(_sum5, pB[5], _pA, vl);
                _sum6 = __riscv_vfmacc_vf_f16m1(_sum6, pB[6], _pA, vl);
                _sum7 = __riscv_vfmacc_vf_f16m1(_sum7, pB[7], _pA, vl);
                pA += packn;
                pB += 8;
            }

            if (alpha != 1.f)
            {
                _sum0 = __riscv_vfmul_vf_f16m1(_sum0, _alpha, vl);
                _sum1 = __riscv_vfmul_vf_f16m1(_sum1, _alpha, vl);
                _sum2 = __riscv_vfmul_vf_f16m1(_sum2, _alpha, vl);
                _sum3 = __riscv_vfmul_vf_f16m1(_sum3, _alpha, vl);
                _sum4 = __riscv_vfmul_vf_f16m1(_sum4, _alpha, vl);
                _sum5 = __riscv_vfmul_vf_f16m1(_sum5, _alpha, vl);
                _sum6 = __riscv_vfmul_vf_f16m1(_sum6, _alpha, vl);
                _sum7 = __riscv_vfmul_vf_f16m1(_sum7, _alpha, vl);
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse16_v_f16m1(outptr0, _sum0, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn, _sum1, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 2, _sum2, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 3, _sum3, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 4, _sum4, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 5, _sum5, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 6, _sum6, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 7, _sum7, vl);
                    outptr0 += packn * 8;
                }
                if (out_elempack == 1)
                {
                    vfloat16m1x8_t _sum = __riscv_vcreate_v_f16m1x8(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);
                    __riscv_vssseg8e16_v_f16m1x8(outptr0, out_hstep * sizeof(__fp16), _sum, vl);
                    outptr0 += 8;
                }
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl);
                __riscv_vse16_v_f16m1(outptr + packn, _sum1, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 2, _sum2, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 3, _sum3, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 4, _sum4, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 5, _sum5, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 6, _sum6, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 7, _sum7, vl);
            }

            outptr += packn * 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;
            vfloat16m1_t _sum2;
            vfloat16m1_t _sum3;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vle16_v_f16m1(pC, vl);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = __riscv_vle16_v_f16m1(pC, vl);
                        _sum1 = __riscv_vle16_v_f16m1(pC + packn, vl);
                        _sum2 = __riscv_vle16_v_f16m1(pC + packn * 2, vl);
                        _sum3 = __riscv_vle16_v_f16m1(pC + packn * 3, vl);
                        pC += packn * 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl);
                        _sum1 = __riscv_vfmv_v_f_f16m1(pC[1], vl);
                        _sum2 = __riscv_vfmv_v_f_f16m1(pC[2], vl);
                        _sum3 = __riscv_vfmv_v_f_f16m1(pC[3], vl);
                        pC += 4;
                    }
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl);
                _sum1 = __riscv_vle16_v_f16m1(outptr + packn, vl);
                _sum2 = __riscv_vle16_v_f16m1(outptr + packn * 2, vl);
                _sum3 = __riscv_vle16_v_f16m1(outptr + packn * 3, vl);
            }

            const __fp16* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _pA = __riscv_vle16_v_f16m1(pA, vl);
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pB[1], _pA, vl);
                _sum2 = __riscv_vfmacc_vf_f16m1(_sum2, pB[2], _pA, vl);
                _sum3 = __riscv_vfmacc_vf_f16m1(_sum3, pB[3], _pA, vl);
                pA += packn;
                pB += 4;
            }

            if (alpha != 1.f)
            {
                _sum0 = __riscv_vfmul_vf_f16m1(_sum0, _alpha, vl);
                _sum1 = __riscv_vfmul_vf_f16m1(_sum1, _alpha, vl);
                _sum2 = __riscv_vfmul_vf_f16m1(_sum2, _alpha, vl);
                _sum3 = __riscv_vfmul_vf_f16m1(_sum3, _alpha, vl);
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse16_v_f16m1(outptr0, _sum0, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn, _sum1, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 2, _sum2, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn * 3, _sum3, vl);
                    outptr0 += packn * 4;
                }
                if (out_elempack == 1)
                {
                    vfloat16m1x4_t _sum = __riscv_vcreate_v_f16m1x4(_sum0, _sum1, _sum2, _sum3);
                    __riscv_vssseg4e16_v_f16m1x4(outptr0, out_hstep * sizeof(__fp16), _sum, vl);
                    outptr0 += 4;
                }
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl);
                __riscv_vse16_v_f16m1(outptr + packn, _sum1, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 2, _sum2, vl);
                __riscv_vse16_v_f16m1(outptr + packn * 3, _sum3, vl);
            }

            outptr += packn * 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vle16_v_f16m1(pC, vl);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = __riscv_vle16_v_f16m1(pC, vl);
                        _sum1 = __riscv_vle16_v_f16m1(pC + packn, vl);
                        pC += packn * 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl);
                        _sum1 = __riscv_vfmv_v_f_f16m1(pC[1], vl);
                        pC += 2;
                    }
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);
                    _sum1 = _sum0;
                }
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl);
                _sum1 = __riscv_vle16_v_f16m1(outptr + packn, vl);
            }

            const __fp16* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _pA = __riscv_vle16_v_f16m1(pA, vl);
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pB[1], _pA, vl);
                pA += packn;
                pB += 2;
            }

            if (alpha != 1.f)
            {
                _sum0 = __riscv_vfmul_vf_f16m1(_sum0, _alpha, vl);
                _sum1 = __riscv_vfmul_vf_f16m1(_sum1, _alpha, vl);
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse16_v_f16m1(outptr0, _sum0, vl);
                    __riscv_vse16_v_f16m1(outptr0 + packn, _sum1, vl);
                    outptr0 += packn * 2;
                }
                if (out_elempack == 1)
                {
                    vfloat16m1x2_t _sum = __riscv_vcreate_v_f16m1x2(_sum0, _sum1);
                    __riscv_vssseg2e16_v_f16m1x2(outptr0, out_hstep * sizeof(__fp16), _sum, vl);
                    outptr0 += 2;
                }
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl);
                __riscv_vse16_v_f16m1(outptr + packn, _sum1, vl);
            }

            outptr += packn * 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            vfloat16m1_t _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl);
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                        _sum0 = __riscv_vle16_v_f16m1(pC, vl);
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = __riscv_vle16_v_f16m1(pC, vl);
                        pC += packn;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl);
                        pC += 1;
                    }
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);
                }
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl);
            }

            const __fp16* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _pA = __riscv_vle16_v_f16m1(pA, vl);
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pB[0], _pA, vl);
                pA += packn;
                pB += 1;
            }

            if (alpha != 1.f)
            {
                _sum0 = __riscv_vfmul_vf_f16m1(_sum0, _alpha, vl);
            }

            if (k_end)
            {
                if (out_elempack == packn)
                {
                    __riscv_vse16_v_f16m1(outptr0, _sum0, vl);
                    outptr0 += packn;
                }
                if (out_elempack == 1)
                {
                    __riscv_vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0, vl);
                    outptr0++;
                }
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl);
            }

            outptr += packn;
        }

        pAT += max_kk * packn;
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j;

        const __fp16* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pC = (const __fp16*)CT_tile + i + ii;
            if (broadcast_type_C == 4)
                pC = (const __fp16*)CT_tile + j;
        }

        int jj = 0;
#if __riscv_vector
        for (; jj + 15 < max_jj; jj += 16)
        {
            vfloat16m2_t _sum0;
            vfloat16m2_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m2(pC[0], vl16);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m2(pC[0], vl16);
                        _sum1 = __riscv_vfmv_v_f_f16m2(pC[1], vl16);
                    }
                    if (broadcast_type_C == 3)
                    {
                        vfloat16m2x2_t _s0 = __riscv_vlseg2e16_v_f16m2x2(pC, vl16);
                        _sum0 = __riscv_vget_v_f16m2x2_f16m2(_s0, 0);
                        _sum1 = __riscv_vget_v_f16m2x2_f16m2(_s0, 1);
                        pC += 16 * 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vle16_v_f16m2(pC, vl16);
                        _sum1 = _sum0;
                        pC += 16;
                    }
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f16m2((__fp16)0.f, vl16);
                    _sum1 = _sum0;
                }
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m2(outptr, vl16);
                _sum1 = __riscv_vle16_v_f16m2(outptr + 16, vl16);
            }

            const __fp16* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m2_t _pB = __riscv_vle16_v_f16m2(pB, vl16);
                _sum0 = __riscv_vfmacc_vf_f16m2(_sum0, pA[0], _pB, vl16);
                _sum1 = __riscv_vfmacc_vf_f16m2(_sum1, pA[1], _pB, vl16);
                pA += 2;
                pB += 16;
            }

            if (alpha != 1.f)
            {
                _sum0 = __riscv_vfmul_vf_f16m2(_sum0, _alpha, vl16);
                _sum1 = __riscv_vfmul_vf_f16m2(_sum1, _alpha, vl16);
            }

            if (k_end)
            {
                __riscv_vse16_v_f16m2(outptr0, _sum0, vl16);
                __riscv_vse16_v_f16m2(outptr0 + out_hstep, _sum1, vl16);
                outptr0 += 16;
            }
            else
            {
                __riscv_vse16_v_f16m2(outptr, _sum0, vl16);
                __riscv_vse16_v_f16m2(outptr + 16, _sum1, vl16);
            }

            outptr += 16 * 2;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl);
                        _sum1 = __riscv_vfmv_v_f_f16m1(pC[1], vl);
                    }
                    if (broadcast_type_C == 3)
                    {
                        vfloat16m1x2_t _s0 = __riscv_vlseg2e16_v_f16m1x2(pC, vl8);
                        _sum0 = __riscv_vget_v_f16m1x2_f16m1(_s0, 0);
                        _sum1 = __riscv_vget_v_f16m1x2_f16m1(_s0, 1);
                        pC += 8 * 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vle16_v_f16m1(pC, vl8);
                        _sum1 = _sum0;
                        pC += 8;
                    }
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);
                    _sum1 = _sum0;
                }
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl8);
                _sum1 = __riscv_vle16_v_f16m1(outptr + 8, vl8);
            }

            const __fp16* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _pB = __riscv_vle16_v_f16m1(pB, vl8);
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _pB, vl8);
                _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _pB, vl8);
                pA += 2;
                pB += 8;
            }

            if (alpha != 1.f)
            {
                _sum0 = __riscv_vfmul_vf_f16m1(_sum0, _alpha, vl8);
                _sum1 = __riscv_vfmul_vf_f16m1(_sum1, _alpha, vl8);
            }

            if (k_end)
            {
                __riscv_vse16_v_f16m1(outptr0, _sum0, vl8);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep, _sum1, vl8);
                outptr0 += 8;
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl8);
                __riscv_vse16_v_f16m1(outptr + 8, _sum1, vl8);
            }

            outptr += 8 * 2;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat16m1_t _sum0;
            vfloat16m1_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl4);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = __riscv_vfmv_v_f_f16m1(pC[0], vl4);
                        _sum1 = __riscv_vfmv_v_f_f16m1(pC[1], vl4);
                    }
                    if (broadcast_type_C == 3)
                    {
                        vfloat16m1x2_t _s0 = __riscv_vlseg2e16_v_f16m1x2(pC, vl4);
                        _sum0 = __riscv_vget_v_f16m1x2_f16m1(_s0, 0);
                        _sum1 = __riscv_vget_v_f16m1x2_f16m1(_s0, 1);
                        pC += 4 * 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = __riscv_vle16_v_f16m1(pC, vl4);
                        _sum1 = _sum0;
                        pC += 4;
                    }
                }
                else
                {
                    _sum0 = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl4);
                    _sum1 = _sum0;
                }
            }
            else
            {
                _sum0 = __riscv_vle16_v_f16m1(outptr, vl4);
                _sum1 = __riscv_vle16_v_f16m1(outptr + 4, vl4);
            }

            const __fp16* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _pB = __riscv_vle16_v_f16m1(pB, vl4);
                _sum0 = __riscv_vfmacc_vf_f16m1(_sum0, pA[0], _pB, vl4);
                _sum1 = __riscv_vfmacc_vf_f16m1(_sum1, pA[1], _pB, vl4);
                pA += 2;
                pB += 4;
            }

            if (alpha != 1.f)
            {
                _sum0 = __riscv_vfmul_vf_f16m1(_sum0, _alpha, vl4);
                _sum1 = __riscv_vfmul_vf_f16m1(_sum1, _alpha, vl4);
            }

            if (k_end)
            {
                __riscv_vse16_v_f16m1(outptr0, _sum0, vl4);
                __riscv_vse16_v_f16m1(outptr0 + out_hstep, _sum1, vl4);
                outptr0 += 4;
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum0, vl4);
                __riscv_vse16_v_f16m1(outptr + 4, _sum1, vl4);
            }

            outptr += 4 * 2;
        }
#endif // __riscv_vector
        for (; jj + 1 < max_jj; jj += 2)
        {
            __fp16 sum00;
            __fp16 sum01;
            __fp16 sum10;
            __fp16 sum11;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        sum00 = pC[0];
                        sum01 = pC[0];
                        sum10 = pC[0];
                        sum11 = pC[0];
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum00 = pC[0];
                        sum01 = pC[1];
                        sum10 = pC[0];
                        sum11 = pC[1];
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum00 = pC[0];
                        sum01 = pC[1];
                        sum10 = pC[2];
                        sum11 = pC[3];
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum00 = pC[0];
                        sum01 = pC[0];
                        sum10 = pC[1];
                        sum11 = pC[1];
                        pC += 2;
                    }
                }
                else
                {
                    sum00 = (__fp16)0.f;
                    sum01 = (__fp16)0.f;
                    sum10 = (__fp16)0.f;
                    sum11 = (__fp16)0.f;
                }
            }
            else
            {
                sum00 = outptr[0];
                sum10 = outptr[1];
                sum01 = outptr[2];
                sum11 = outptr[3];
            }

            const __fp16* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                sum00 = (__fp16)(sum00 + pA[0] * pB[0]);
                sum01 = (__fp16)(sum01 + pA[1] * pB[0]);
                sum10 = (__fp16)(sum10 + pA[0] * pB[1]);
                sum11 = (__fp16)(sum11 + pA[1] * pB[1]);
                pA += 2;
                pB += 2;
            }

            if (alpha != 1.f)
            {
                sum00 = (__fp16)(sum00 * _alpha);
                sum01 = (__fp16)(sum01 * _alpha);
                sum10 = (__fp16)(sum10 * _alpha);
                sum11 = (__fp16)(sum11 * _alpha);
            }

            if (k_end)
            {
                outptr0[0] = sum00;
                outptr0[1] = sum10;
                outptr0[out_hstep] = sum01;
                outptr0[out_hstep + 1] = sum11;
                outptr0 += 2;
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum10;
                outptr[2] = sum01;
                outptr[3] = sum11;
            }

            outptr += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            __fp16 sum0;
            __fp16 sum1;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        sum0 = pC[0];
                        sum1 = pC[0];
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum0 = pC[0];
                        sum1 = pC[1];
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum0 = pC[0];
                        sum1 = pC[1];
                        pC += 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum0 = pC[0];
                        sum1 = pC[0];
                        pC += 1;
                    }
                }
                else
                {
                    sum0 = (__fp16)0.f;
                    sum1 = (__fp16)0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const __fp16* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                sum0 = (__fp16)(sum0 + pA[0] * pB[0]);
                sum1 = (__fp16)(sum1 + pA[1] * pB[0]);
                pA += 2;
                pB += 1;
            }

            if (alpha != 1.f)
            {
                sum0 = (__fp16)(sum0 * _alpha);
                sum1 = (__fp16)(sum1 * _alpha);
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[out_hstep] = sum1;
                outptr0++;
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j;

        const __fp16* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
                pC = (const __fp16*)CT_tile + i + ii;
            if (broadcast_type_C == 4)
                pC = (const __fp16*)CT_tile + j;
        }

        int jj = 0;
#if __riscv_vector
        for (; jj + 15 < max_jj; jj += 16)
        {
            vfloat16m2_t _sum;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = __riscv_vfmv_v_f_f16m2(pC[0], vl16);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = __riscv_vle16_v_f16m2(pC, vl16);
                        pC += 16;
                    }
                }
                else
                {
                    _sum = __riscv_vfmv_v_f_f16m2((__fp16)0.f, vl16);
                }
            }
            else
            {
                _sum = __riscv_vle16_v_f16m2(outptr, vl16);
            }

            const __fp16* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m2_t _pB = __riscv_vle16_v_f16m2(pB, vl16);
                _sum = __riscv_vfmacc_vf_f16m2(_sum, pA[0], _pB, vl16);
                pA += 1;
                pB += 16;
            }

            if (alpha != 1.f)
            {
                _sum = __riscv_vfmul_vf_f16m2(_sum, _alpha, vl16);
            }

            if (k_end)
            {
                __riscv_vse16_v_f16m2(outptr0, _sum, vl16);
                outptr0 += 16;
            }
            else
            {
                __riscv_vse16_v_f16m2(outptr, _sum, vl16);
            }

            outptr += 16;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat16m1_t _sum;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = __riscv_vfmv_v_f_f16m1(pC[0], vl8);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = __riscv_vle16_v_f16m1(pC, vl8);
                        pC += 8;
                    }
                }
                else
                {
                    _sum = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl8);
                }
            }
            else
            {
                _sum = __riscv_vle16_v_f16m1(outptr, vl8);
            }

            const __fp16* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _pB = __riscv_vle16_v_f16m1(pB, vl8);
                _sum = __riscv_vfmacc_vf_f16m1(_sum, pA[0], _pB, vl8);
                pA += 1;
                pB += 8;
            }

            if (alpha != 1.f)
            {
                _sum = __riscv_vfmul_vf_f16m1(_sum, _alpha, vl8);
            }

            if (k_end)
            {
                __riscv_vse16_v_f16m1(outptr0, _sum, vl8);
                outptr0 += 8;
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum, vl8);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat16m1_t _sum;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = __riscv_vfmv_v_f_f16m1(pC[0], vl4);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = __riscv_vle16_v_f16m1(pC, vl4);
                        pC += 4;
                    }
                }
                else
                {
                    _sum = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl4);
                }
            }
            else
            {
                _sum = __riscv_vle16_v_f16m1(outptr, vl4);
            }

            const __fp16* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                vfloat16m1_t _pB = __riscv_vle16_v_f16m1(pB, vl4);
                _sum = __riscv_vfmacc_vf_f16m1(_sum, pA[0], _pB, vl4);
                pA += 1;
                pB += 4;
            }

            if (alpha != 1.f)
            {
                _sum = __riscv_vfmul_vf_f16m1(_sum, _alpha, vl4);
            }

            if (k_end)
            {
                __riscv_vse16_v_f16m1(outptr0, _sum, vl4);
                outptr0 += 4;
            }
            else
            {
                __riscv_vse16_v_f16m1(outptr, _sum, vl4);
            }

            outptr += 4;
        }
#endif // __riscv_vector
        for (; jj + 1 < max_jj; jj += 2)
        {
            __fp16 sum0;
            __fp16 sum1;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum0 = pC[0];
                        sum1 = pC[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum0 = pC[0];
                        sum1 = pC[1];
                        pC += 2;
                    }
                }
                else
                {
                    sum0 = (__fp16)0.f;
                    sum1 = (__fp16)0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const __fp16* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                sum0 = (__fp16)(sum0 + pA[0] * pB[0]);
                sum1 = (__fp16)(sum1 + pA[0] * pB[1]);
                pA += 1;
                pB += 2;
            }

            if (alpha != 1.f)
            {
                sum0 = (__fp16)(sum0 * _alpha);
                sum1 = (__fp16)(sum1 * _alpha);
            }

            if (k_end)
            {
                outptr0[0] = sum0;
                outptr0[1] = sum1;
                outptr0 += 2;
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            __fp16 sum;

            if (k == 0)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum = pC[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum = pC[0];
                        pC += 1;
                    }
                }
                else
                {
                    sum = (__fp16)0.f;
                }
            }
            else
            {
                sum = outptr[0];
            }

            const __fp16* pA = pAT;
            for (int kk = 0; kk < max_kk; kk++)
            {
                sum = (__fp16)(sum + pA[0] * pB[0]);
                pA += 1;
                pB += 1;
            }

            if (alpha != 1.f)
            {
                sum = (__fp16)(sum * _alpha);
            }

            if (k_end)
            {
                outptr0[0] = sum;
                outptr0++;
            }
            else
            {
                outptr[0] = sum;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void get_optimal_tile_mnk_fp16sa(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    int tile_size = (int)sqrtf((float)l2_cache_size / 3 / sizeof(__fp16));

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
    const int packn_n = 16;
#else
    const int packn = 4;
    const int packn_n = 4;
#endif

    TILE_M = std::max(packn, tile_size / packn * packn);
    TILE_N = std::max(packn_n, tile_size / packn_n * packn_n);
    TILE_K = std::max(packn, tile_size / packn * packn);

    if (K > 0)
    {
        int nn_K = (K + TILE_K - 1) / TILE_K;
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + (packn - 1)) / packn * packn);

        if (nn_K == 1)
        {
            tile_size = (int)((float)l2_cache_size / 2 / sizeof(__fp16) / TILE_K);
            TILE_M = std::max(packn, tile_size / packn * packn);
            TILE_N = std::max(packn_n, tile_size / packn_n * packn_n);
        }
    }

    TILE_M *= std::min(nT, get_physical_cpu_count());

    if (M > 0)
    {
        int nn_M = (M + TILE_M - 1) / TILE_M;
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + (packn - 1)) / packn * packn);
    }

    if (N > 0)
    {
        int nn_N = (N + TILE_N - 1) / TILE_N;
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + (packn_n - 1)) / packn_n * packn_n);
    }

    if (nT > 1)
    {
        TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + (packn - 1)) / packn * packn);
    }

    // always take constant TILE_M/N/K value when provided
    if (constant_TILE_M > 0)
    {
        TILE_M = (constant_TILE_M + (packn - 1)) / packn * packn;
    }

    if (constant_TILE_N > 0)
    {
        TILE_N = (constant_TILE_N + (packn_n - 1)) / packn_n * packn_n;
    }

    if (constant_TILE_K > 0)
    {
        TILE_K = (constant_TILE_K + (packn - 1)) / packn * packn;
    }
}
