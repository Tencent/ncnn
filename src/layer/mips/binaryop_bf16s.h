// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

template<typename Op>
static void binary_op_vector_no_broadcast_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size)
{
    const Op op;

    int i = 0;
#if __mips_msa
    for (; i + 3 < size; i += 4)
    {
        v4f32 _p = bfloat2float_msa(ptr);
        v4f32 _b = bfloat2float_msa(ptr1);
        v4f32 _outp = op(_p, _b);
        float2bfloat_msa_store(_outp, outptr);
        ptr += 4;
        ptr1 += 4;
        outptr += 4;
    }
#endif // __mips_msa
    for (; i < size; i++)
    {
        *outptr++ = float32_to_bfloat16(op(bfloat16_to_float32(*ptr++), bfloat16_to_float32(*ptr1++)));
    }
}

template<typename Op>
static void binary_op_vector_broadcast_b_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size, int elempack)
{
    const Op op;

    const float b = bfloat16_to_float32(*ptr1);

    int i = 0;
#if __mips_msa
    v4f32 _b_128 = (elempack == 4) ? bfloat2float_msa(ptr1) : __msa_fill_w_f32(b);
    for (; i + 3 < size; i += 4)
    {
        v4f32 _p = bfloat2float_msa(ptr);
        v4f32 _outp = op(_p, _b_128);
        float2bfloat_msa_store(_outp, outptr);
        ptr += 4;
        outptr += 4;
    }
#endif // __mips_msa
    for (; i < size; i++)
    {
        *outptr++ = float32_to_bfloat16(op(bfloat16_to_float32(*ptr++), b));
    }
}

template<typename Op>
static void binary_op_vector_broadcast_a_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int size, int elempack)
{
    const Op op;

    const float a = bfloat16_to_float32(*ptr);

    int i = 0;
#if __mips_msa
    v4f32 _a_128 = (elempack == 4) ? bfloat2float_msa(ptr) : __msa_fill_w_f32(a);
    for (; i + 3 < size; i += 4)
    {
        v4f32 _b = bfloat2float_msa(ptr1);
        v4f32 _outp = op(_a_128, _b);
        float2bfloat_msa_store(_outp, outptr);
        ptr1 += 4;
        outptr += 4;
    }
#endif // __mips_msa
    for (; i < size; i++)
    {
        *outptr++ = float32_to_bfloat16(op(a, bfloat16_to_float32(*ptr1++)));
    }
}

template<typename Op>
static void binary_op_vector_broadcast_pb_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

#if __mips_msa
    if (elempack == 4)
    {
        for (int i = 0; i < w; i++)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            v4f32 _b = __msa_fill_w_f32(bfloat16_to_float32(*ptr1));
            v4f32 _outp = op(_p, _b);
            float2bfloat_msa_store(_outp, outptr);
            ptr += 4;
            ptr1 += 1;
            outptr += 4;
        }
        return;
    }
#endif // __mips_msa
}

template<typename Op>
static void binary_op_vector_broadcast_pb_b_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

    const int size = w * elempack;
    const float b = bfloat16_to_float32(*ptr1);

    int i = 0;
#if __mips_msa
    {
        v4f32 _b_128 = __msa_fill_w_f32(b);
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            v4f32 _outp = op(_p, _b_128);
            float2bfloat_msa_store(_outp, outptr);
            ptr += 4;
            outptr += 4;
        }
    }
#endif // __mips_msa
    for (; i < size; i++)
    {
        *outptr++ = float32_to_bfloat16(op(bfloat16_to_float32(*ptr++), b));
    }
}

template<typename Op>
static void binary_op_vector_broadcast_pb_a_bf16s(const unsigned short* ptr, const unsigned short* ptr1, unsigned short* outptr, int w, int elempack)
{
    const Op op;

#if __mips_msa
    if (elempack == 4)
    {
        v4f32 _p = bfloat2float_msa(ptr);
        for (int i = 0; i < w; i++)
        {
            v4f32 _b = __msa_fill_w_f32(bfloat16_to_float32(*ptr1));
            v4f32 _outp = op(_p, _b);
            float2bfloat_msa_store(_outp, outptr);
            ptr1 += 1;
            outptr += 4;
        }
        return;
    }
#endif // __mips_msa
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
            return binary_op_vector_no_broadcast_bf16s<Op>(ptr, ptr1, outptr, size);
        if (bw == 1)
            return binary_op_vector_broadcast_b_bf16s<Op>(ptr, ptr1, outptr, size, elempack);
        if (aw == 1)
            return binary_op_vector_broadcast_a_bf16s<Op>(ptr, ptr1, outptr, size, elempack);
    }

    if (bp == 1)
    {
        if (aw == bw)
            return binary_op_vector_broadcast_pb_bf16s<Op>(ptr, ptr1, outptr, w, elempack);
        if (bw == 1)
            return binary_op_vector_broadcast_pb_b_bf16s<Op>(ptr, ptr1, outptr, w, elempack);
        if (aw == 1)
            return binary_op_vector_broadcast_pb_a_bf16s<Op>(ptr, ptr1, outptr, w, elempack);
    }
}
