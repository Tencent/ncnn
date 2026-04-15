// Tencent is pleased to support the open source community by making ncnn available.
//
//                    Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

static void softmax_bf16s_msa(unsigned short* _ptr, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // reduce max
#if __mips_msa
    v4f32 _max = (v4f32)__msa_fill_w_f32(-FLT_MAX);
#endif // __mips_msa
    float max = -FLT_MAX;
    {
        const unsigned short* ptr = _ptr;

        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            _max = __msa_fmax_w(_max, _p);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            max = std::max(max, bfloat16_to_float32(*ptr++));
        }
    }

#if __mips_msa
    if (elempack == 1)
    {
        max = std::max(max, __msa_reduce_fmax_w(_max));

        _max = (v4f32)__msa_fill_w_f32(max);
    }
#endif // __mips_msa

    // reduce exp(x - max) and store back to bf16
#if __mips_msa
    v4f32 _sum = (v4f32)__msa_fill_w(0);
#endif // __mips_msa
    float sum = 0.f;
    {
        unsigned short* ptr = _ptr;

        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            _p = __msa_fsub_w(_p, _max);
            _p = exp_ps(_p);
            float2bfloat_msa_store(_p, ptr);
            _sum = __msa_fadd_w(_sum, _p);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            float v = expf(bfloat16_to_float32(*ptr) - max);
            *ptr = float32_to_bfloat16(v);
            sum += v;
            ptr++;
        }
    }

#if __mips_msa
    if (elempack == 4)
    {
        // reciprocal per-lane
        v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
        _sum = __msa_fdiv_w(_one, _sum);
    }
#endif // __mips_msa
    if (elempack == 1)
    {
#if __mips_msa
        sum += __msa_reduce_fadd_w(_sum);
#endif // __mips_msa

        sum = 1.f / sum;

#if __mips_msa
        _sum = (v4f32)__msa_fill_w_f32(sum);
#endif // __mips_msa
    }

    // div sum (multiply by reciprocal)
    {
        unsigned short* ptr = _ptr;

        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            _p = __msa_fmul_w(_p, _sum);
            float2bfloat_msa_store(_p, ptr);
            ptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * sum);
            ptr++;
        }
    }
}

#if __mips_msa
static void softmax_bf16s_pack4_msa(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
        for (; j < size1; j++)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            *maxptr = std::max(*maxptr, __msa_reduce_fmax_w(_p));
            ptr += 4;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
        for (; j < size1; j++)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            v4f32 _max = (v4f32)__msa_fill_w_f32(*maxptr);
            _p = exp_ps(__msa_fsub_w(_p, _max));
            float2bfloat_msa_store(_p, ptr);
            *sumptr += __msa_reduce_fadd_w(_p);
            ptr += 4;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _sum = (v4f32)__msa_ld_w(sumptr, 0);
            v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
            _sum = __msa_fdiv_w(_one, _sum);
            __msa_st_w((v4i32)_sum, sumptr, 0);
            sumptr += 4;
        }
        for (; j < size1; j++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
        for (; j < size1; j++)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            v4f32 _sum = (v4f32)__msa_fill_w_f32(*sumptr);
            _p = __msa_fmul_w(_p, _sum);
            float2bfloat_msa_store(_p, ptr);
            ptr += 4;
            sumptr++;
        }
    }
}
#endif // __mips_msa

static void softmax_bf16s_pack1_msa(unsigned short* _ptr, int elemcount, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int j = 0;
#if __mips_msa
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            v4f32 _max = (v4f32)__msa_ld_w(maxptr, 0);
            _max = __msa_fmax_w(_max, _p);
            __msa_st_w((v4i32)_max, maxptr, 0);
            ptr += 4;
            maxptr += 4;
        }
#endif // __mips_msa
        for (; j < size1; j++)
        {
            *maxptr = std::max(*maxptr, bfloat16_to_float32(*ptr));
            ptr++;
            maxptr++;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int j = 0;
#if __mips_msa
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            v4f32 _max = (v4f32)__msa_ld_w(maxptr, 0);
            v4f32 _sum = (v4f32)__msa_ld_w(sumptr, 0);
            _p = __msa_fsub_w(_p, _max);
            _p = exp_ps(_p);
            float2bfloat_msa_store(_p, ptr);
            _sum = __msa_fadd_w(_sum, _p);
            __msa_st_w((v4i32)_sum, sumptr, 0);
            ptr += 4;
            maxptr += 4;
            sumptr += 4;
        }
#endif // __mips_msa
        for (; j < size1; j++)
        {
            float v = expf(bfloat16_to_float32(*ptr) - *maxptr);
            *ptr = float32_to_bfloat16(v);
            *sumptr += v;
            ptr++;
            maxptr++;
            sumptr++;
        }
    }

    {
        float* sumptr = _sumptr;
        int j = 0;
#if __mips_msa
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _sum = (v4f32)__msa_ld_w(sumptr, 0);
            v4f32 _one = (v4f32)__msa_fill_w_f32(1.f);
            _sum = __msa_fdiv_w(_one, _sum);
            __msa_st_w((v4i32)_sum, sumptr, 0);
            sumptr += 4;
        }
#endif // __mips_msa
        for (; j < size1; j++)
        {
            *sumptr = 1.f / *sumptr;
            sumptr++;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* sumptr = _sumptr;

        int j = 0;
#if __mips_msa
        for (; j + 3 < size1; j += 4)
        {
            v4f32 _p = bfloat2float_msa(ptr);
            v4f32 _sum = (v4f32)__msa_ld_w(sumptr, 0);
            _p = __msa_fmul_w(_p, _sum);
            float2bfloat_msa_store(_p, ptr);
            ptr += 4;
            sumptr += 4;
        }
#endif // __mips_msa
        for (; j < size1; j++)
        {
            *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * *sumptr);
            ptr++;
            sumptr++;
        }
    }
}

static void softmax_bf16s_msa_dispatch(unsigned short* _ptr, int elemcount, int elempack, size_t stride, int size1, float* _maxptr, float* _sumptr)
{
    // init max
    {
        float* maxptr = _maxptr;

        int j = 0;
#if __mips_msa
        v4f32 _negmax = (v4f32)__msa_fill_w_f32(-FLT_MAX);
        for (; j + 3 < size1; j += 4)
        {
            __msa_st_w((v4i32)_negmax, maxptr, 0);
            maxptr += 4;
        }
#endif // __mips_msa
        for (; j < size1; j++)
        {
            *maxptr++ = -FLT_MAX;
        }
    }

    // init sum
    {
        float* sumptr = _sumptr;

        int j = 0;
#if __mips_msa
        v4i32 _zero = __msa_fill_w(0);
        for (; j + 3 < size1; j += 4)
        {
            __msa_st_w(_zero, sumptr, 0);
            sumptr += 4;
        }
#endif // __mips_msa
        for (; j < size1; j++)
        {
            *sumptr++ = 0.f;
        }
    }

#if __mips_msa
    if (elempack == 4)
    {
        softmax_bf16s_pack4_msa(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
#endif // __mips_msa
    if (elempack == 1)
    {
        softmax_bf16s_pack1_msa(_ptr, elemcount, stride, size1, _maxptr, _sumptr);
    }
}
