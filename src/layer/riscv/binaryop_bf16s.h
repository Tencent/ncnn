// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
void binary_op_vector_bf16s_zvfbfmin(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int aw, int bw, int ap, int bp, int op_type);
void binary_op_vector_scalar_b_bf16s_zvfbfmin(const unsigned short* ptr, float b, unsigned short* outptr, int size, int op_type);
#endif

template<typename Op>
static void binary_op_vector_no_broadcast_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size)
{
    const Op op;

#if __riscv_vector
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e16m4(n);
        vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vl), vl);
        vfloat32m8_t _p1 = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr1, vl), vl);
        vfloat32m8_t _outp = op(_p, _p1, vl);
        __riscv_vse16_v_u16m4(outptr, float2bfloat_rvv(_outp, vl), vl);
        n -= vl;
        ptr += vl;
        ptr1 += vl;
        outptr += vl;
    }
#else
    for (int i = 0; i < size; i++)
    {
        *outptr = float32_to_bfloat16(op(bfloat16_to_float32(*ptr), bfloat16_to_float32(*ptr1)));
        ptr += 1;
        ptr1 += 1;
        outptr += 1;
    }
#endif
}

template<typename Op>
static void binary_op_vector_scalar_b_bf16s(const unsigned short* ptr, float b, unsigned short* outptr, int size)
{
    const Op op;

#if __riscv_vector
    int n = size;
    while (n > 0)
    {
        size_t vl = __riscv_vsetvl_e16m4(n);
        vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vl), vl);
        vfloat32m8_t _outp = op(_p, b, vl);
        __riscv_vse16_v_u16m4(outptr, float2bfloat_rvv(_outp, vl), vl);
        n -= vl;
        ptr += vl;
        outptr += vl;
    }
#else
    for (int i = 0; i < size; i++)
    {
        *outptr = float32_to_bfloat16(op(bfloat16_to_float32(*ptr), b));
        ptr += 1;
        outptr += 1;
    }
#endif
}

template<typename Op>
static void binary_op_vector_broadcast_b_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size, int elempack)
{
    if (elempack == 1)
    {
        // broadcast single b
        return binary_op_vector_scalar_b_bf16s<Op>(ptr, bfloat16_to_float32(*ptr1), outptr, size);
    }

#if __riscv_vector
    const Op op;

    // b holds one packed vector, repeated for every w position
    const size_t vl = __riscv_vsetvl_e16m4(elempack);
    vfloat32m8_t _bx = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr1, vl), vl);

    int n = size;
    while (n > 0)
    {
        vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vl), vl);
        vfloat32m8_t _outp = op(_p, _bx, vl);
        __riscv_vse16_v_u16m4(outptr, float2bfloat_rvv(_outp, vl), vl);
        n -= vl;
        ptr += vl;
        outptr += vl;
    }
#endif // __riscv_vector
}

template<typename Op>
static void binary_op_vector_broadcast_a_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size, int elempack)
{
    const Op op;

    const float a = bfloat16_to_float32(*ptr);

#if __riscv_vector
    if (elempack == 1)
    {
        // broadcast single a
        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m4(n);
            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr1, vl), vl);
            vfloat32m8_t _outp = op(a, _p, vl);
            __riscv_vse16_v_u16m4(outptr, float2bfloat_rvv(_outp, vl), vl);
            n -= vl;
            ptr1 += vl;
            outptr += vl;
        }
        return;
    }

    // a holds one packed vector, repeated for every w position
    const size_t vl = __riscv_vsetvl_e16m4(elempack);
    vfloat32m8_t _ax = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vl), vl);

    int n = size;
    while (n > 0)
    {
        vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr1, vl), vl);
        vfloat32m8_t _outp = op(_ax, _p, vl);
        __riscv_vse16_v_u16m4(outptr, float2bfloat_rvv(_outp, vl), vl);
        n -= vl;
        ptr1 += vl;
        outptr += vl;
    }
#else
    for (int i = 0; i < size; i++)
    {
        *outptr = float32_to_bfloat16(op(a, bfloat16_to_float32(*ptr1)));
        ptr1 += 1;
        outptr += 1;
    }
#endif
}

template<typename Op>
static void binary_op_vector_broadcast_pb_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

#if __riscv_vector
    // if (elempack == packn)
    {
        const size_t vl = __riscv_vsetvl_e16m4(elempack);
        for (int i = 0; i < w; i++)
        {
            vfloat32m8_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vl), vl);
            vfloat32m8_t _outp = op(_p, bfloat16_to_float32(*ptr1), vl);
            __riscv_vse16_v_u16m4(outptr, float2bfloat_rvv(_outp, vl), vl);
            ptr += vl;
            ptr1 += 1;
            outptr += vl;
        }
    }
#endif // __riscv_vector
}

template<typename Op>
static void binary_op_vector_broadcast_pb_b_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    binary_op_vector_scalar_b_bf16s<Op>(ptr, bfloat16_to_float32(*ptr1), outptr, w * elempack);
}

template<typename Op>
static void binary_op_vector_broadcast_pb_a_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

#if __riscv_vector
    // if (elempack == packn)
    {
        const size_t vl = __riscv_vsetvl_e16m4(elempack);
        vfloat32m8_t _ax = bfloat2float_rvv(__riscv_vle16_v_u16m4(ptr, vl), vl);
        for (int i = 0; i < w; i++)
        {
            vfloat32m8_t _outp = op(_ax, bfloat16_to_float32(*ptr1), vl);
            __riscv_vse16_v_u16m4(outptr, float2bfloat_rvv(_outp, vl), vl);
            ptr1 += 1;
            outptr += vl;
        }
    }
#endif // __riscv_vector
}

template<typename Op>
static void binary_op_vector_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int aw, int bw, int ap, int bp)
{
    const int w = std::max(aw, bw);
    const int elempack = std::max(ap, bp);
    const int size = w * elempack;

    if (ap == bp)
    {
        if (aw == bw)
        {
            // no broadcast
            return binary_op_vector_no_broadcast_bf16s<Op>(ptr, ptr1, outptr, size);
        }

        if (bw == 1)
        {
            // broadcast single b
            return binary_op_vector_broadcast_b_bf16s<Op>(ptr, ptr1, outptr, size, elempack);
        }

        if (aw == 1)
        {
            // broadcast single a
            return binary_op_vector_broadcast_a_bf16s<Op>(ptr, ptr1, outptr, size, elempack);
        }
    }

    if (bp == 1)
    {
        if (aw == bw)
        {
            // broadcast pack1 b
            return binary_op_vector_broadcast_pb_bf16s<Op>(ptr, ptr1, outptr, w, elempack);
        }

        if (bw == 1)
        {
            // broadcast pack1 single b
            return binary_op_vector_broadcast_pb_b_bf16s<Op>(ptr, ptr1, outptr, w, elempack);
        }

        if (aw == 1)
        {
            // broadcast single a and pack1 b
            return binary_op_vector_broadcast_pb_a_bf16s<Op>(ptr, ptr1, outptr, w, elempack);
        }
    }

    // shall never reach here
}
