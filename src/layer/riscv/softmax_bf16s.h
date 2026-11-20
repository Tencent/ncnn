// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
int softmax_bf16s_zvfbfmin(Mat& bottom_top_blob, int axis, const Option& opt);
#endif

static void softmax_bf16s_pack(unsigned short* _ptr, int elemcount, size_t stride, int size, float* _maxptr, float* _sumptr);

// softmax over a contiguous run of elemcount * elempack bf16 values
// when packed, the elempack lanes of every packed element are independent
// batches and must be reduced separately
static void softmax_bf16s(unsigned short* _ptr, int elemcount, int elempack, float* _scratch)
{
    const int size = elemcount * elempack;

    if (elempack > 1)
    {
        softmax_bf16s_pack(_ptr, elemcount, elempack, elempack, _scratch, _scratch + elempack);
        return;
    }

#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    unsigned long old_fxcr = csrr_fxcr();
    // enable BF16 with round-to-nearest-even
    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

    const size_t vlm4 = __riscv_vsetvlmax_e32m4();

    // reduce max
    vfloat32m4_t _max = __riscv_vfmv_v_f_f32m4(-FLT_MAX, vlm4);
    {
        const unsigned short* ptr = _ptr;

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);

            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            _max = __riscv_vfmax_vv_f32m4(_max, _p, vl);

            ptr += vl;
            n -= vl;
        }
    }

    vfloat32m1_t _max1 = __riscv_vfmv_s_f_f32m1(-FLT_MAX, __riscv_vsetvlmax_e32m1());
    _max1 = __riscv_vfredmax_vs_f32m4_f32m1(_max, _max1, vlm4);
    const float max = __riscv_vfmv_f_s_f32m1_f32(_max1);
    const vfloat32m4_t _vmax = __riscv_vfmv_v_f_f32m4(max, vlm4);

    // reduce exp(x - max)
    vfloat32m4_t _sum = __riscv_vfmv_v_f_f32m4(0.f, vlm4);
    {
        const unsigned short* ptr = _ptr;

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);

            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            _p = exp_ps(__riscv_vfsub_vv_f32m4(_p, _vmax, vl), vl);
            _sum = __riscv_vfadd_vv_f32m4(_sum, _p, vl);

            ptr += vl;
            n -= vl;
        }
    }

    vfloat32m1_t _sum1 = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
    _sum1 = __riscv_vfredusum_vs_f32m4_f32m1(_sum, _sum1, vlm4);
    const float sum = __riscv_vfmv_f_s_f32m1_f32(_sum1);
    const vfloat32m4_t _isum = __riscv_vfmv_v_f_f32m4(1.f / sum, vlm4);

    // div sum
    {
        unsigned short* ptr = _ptr;

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);

            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            _p = exp_ps(__riscv_vfsub_vv_f32m4(_p, _vmax, vl), vl);
            _p = __riscv_vfmul_vv_f32m4(_p, _isum, vl);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);

            ptr += vl;
            n -= vl;
        }
    }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    // restore FXCR while retaining accrued exception flags, including FE
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
    float max = -FLT_MAX;
    {
        const unsigned short* ptr = _ptr;

        for (int i = 0; i < size; i++)
        {
            max = std::max(max, bfloat16_to_float32(*ptr++));
        }
    }

    // reduce exp(x - max)
    float sum = 0.f;
    {
        const unsigned short* ptr = _ptr;

        for (int i = 0; i < size; i++)
        {
            sum += expf(bfloat16_to_float32(*ptr++) - max);
        }
    }

    sum = 1.f / sum;

    // div sum
    {
        unsigned short* ptr = _ptr;

        for (int i = 0; i < size; i++)
        {
            float v = expf(bfloat16_to_float32(*ptr) - max) * sum;
            *ptr++ = float32_to_bfloat16(v);
        }
    }
#endif // __riscv_vector
}

// softmax across elemcount values with stride between them
// max and reciprocal sum are kept in the fp32 scratch buffer _maxptr / _sumptr
static void softmax_bf16s_pack(unsigned short* _ptr, int elemcount, size_t stride, int size, float* _maxptr, float* _sumptr)
{
#if __riscv_vector
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    unsigned long old_fxcr = csrr_fxcr();
    // enable BF16 with round-to-nearest-even
    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

    // init max / sum
    {
        float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        vfloat32m4_t _negmax = __riscv_vfmv_v_f_f32m4(-FLT_MAX, __riscv_vsetvlmax_e32m4());
        vfloat32m4_t _zero = __riscv_vfmv_v_f_f32m4(0.f, __riscv_vsetvlmax_e32m4());

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m4(n);

            __riscv_vse32_v_f32m4(maxptr, _negmax, vl);
            __riscv_vse32_v_f32m4(sumptr, _zero, vl);

            maxptr += vl;
            sumptr += vl;
            n -= vl;
        }
    }

    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;
        float* maxptr = _maxptr;

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);

            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            vfloat32m4_t _max = __riscv_vle32_v_f32m4(maxptr, vl);
            _max = __riscv_vfmax_vv_f32m4(_max, _p, vl);
            __riscv_vse32_v_f32m4(maxptr, _max, vl);

            ptr += vl;
            maxptr += vl;
            n -= vl;
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        float* sumptr = _sumptr;

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);

            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            vfloat32m4_t _max = __riscv_vle32_v_f32m4(maxptr, vl);
            _p = exp_ps(__riscv_vfsub_vv_f32m4(_p, _max, vl), vl);
            vfloat32m4_t _sum = __riscv_vle32_v_f32m4(sumptr, vl);
            _sum = __riscv_vfadd_vv_f32m4(_sum, _p, vl);
            __riscv_vse32_v_f32m4(sumptr, _sum, vl);

            ptr += vl;
            maxptr += vl;
            sumptr += vl;
            n -= vl;
        }
    }

    // reciprocal sum
    {
        float* sumptr = _sumptr;

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m4(n);

            vfloat32m4_t _sum = __riscv_vle32_v_f32m4(sumptr, vl);
            _sum = __riscv_vfrdiv_vf_f32m4(_sum, 1.f, vl);
            __riscv_vse32_v_f32m4(sumptr, _sum, vl);

            sumptr += vl;
            n -= vl;
        }
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;
        const float* maxptr = _maxptr;
        const float* sumptr = _sumptr;

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);

            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            vfloat32m4_t _max = __riscv_vle32_v_f32m4(maxptr, vl);
            vfloat32m4_t _sum = __riscv_vle32_v_f32m4(sumptr, vl);
            _p = exp_ps(__riscv_vfsub_vv_f32m4(_p, _max, vl), vl);
            _p = __riscv_vfmul_vv_f32m4(_p, _sum, vl);
            __riscv_vse16_v_u16m2(ptr, float2bfloat_rvv(_p, vl), vl);

            ptr += vl;
            maxptr += vl;
            sumptr += vl;
            n -= vl;
        }
    }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    // restore FXCR while retaining accrued exception flags, including FE
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
#else  // __riscv_vector
    // init max / sum
    for (int j = 0; j < size; j++)
    {
        _maxptr[j] = -FLT_MAX;
        _sumptr[j] = 0.f;
    }

    // reduce max
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;

        for (int j = 0; j < size; j++)
        {
            _maxptr[j] = std::max(_maxptr[j], bfloat16_to_float32(ptr[j]));
        }
    }

    // reduce exp(x - max)
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;

        for (int j = 0; j < size; j++)
        {
            _sumptr[j] += expf(bfloat16_to_float32(ptr[j]) - _maxptr[j]);
        }
    }

    // reciprocal sum
    for (int j = 0; j < size; j++)
    {
        _sumptr[j] = 1.f / _sumptr[j];
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;

        for (int j = 0; j < size; j++)
        {
            float v = expf(bfloat16_to_float32(ptr[j]) - _maxptr[j]) * _sumptr[j];
            ptr[j] = float32_to_bfloat16(v);
        }
    }
#endif // __riscv_vector
}


#if __riscv_vector
// softmax across a packed bf16 axis
// every group of elempack consecutive bf16 values holds elempack values of the
// reduced axis, so after the per lane max/sum accumulation the elempack lanes of
// each position have to be folded back to one scalar (like fp32 softmax_packn)
static void softmax_bf16s_packn(unsigned short* _ptr, int elemcount, size_t stride, int size1, int elempack, float* _maxptr)
{
    const int size = size1 * elempack;

    float* maxptr = _maxptr;
    float* sumptr = maxptr + size;
    float* invsumptr = sumptr + size;

    const size_t vlm4 = __riscv_vsetvlmax_e32m4();
    const size_t vlm2 = __riscv_vsetvlmax_e32m2();
    const size_t vlm1 = __riscv_vsetvlmax_e32m1();

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    unsigned long old_fxcr = csrr_fxcr();
    // enable BF16 with round-to-nearest-even
    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif

    // init max / sum
    {
        vfloat32m4_t _negmax = __riscv_vfmv_v_f_f32m4(-FLT_MAX, vlm4);
        vfloat32m4_t _zero = __riscv_vfmv_v_f_f32m4(0.f, vlm4);

        float* mp = maxptr;
        float* sp = sumptr;

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e32m4(n);

            __riscv_vse32_v_f32m4(mp, _negmax, vl);
            __riscv_vse32_v_f32m4(sp, _zero, vl);

            mp += vl;
            sp += vl;
            n -= vl;
        }
    }

    // reduce max per lane
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;
        float* mp = maxptr;

        int n = size;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e16m2(n);

            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
            vfloat32m4_t _max = __riscv_vle32_v_f32m4(mp, vl);
            _max = __riscv_vfmax_vv_f32m4(_max, _p, vl);
            __riscv_vse32_v_f32m4(mp, _max, vl);

            ptr += vl;
            mp += vl;
            n -= vl;
        }
    }

    // fold the elempack lanes of every position to one max
    for (int j = 0; j < size1; j++)
    {
        vfloat32m2_t _max = __riscv_vle32_v_f32m2(maxptr + j * elempack, vlm2);
        vfloat32m1_t _acc = __riscv_vfmv_s_f_f32m1(-FLT_MAX, vlm1);
        _acc = __riscv_vfredmax_vs_f32m2_f32m1(_max, _acc, vlm2);
        maxptr[j * elempack] = __riscv_vfmv_f_s_f32m1_f32(_acc);
    }

    // reduce exp(x - max) per lane
    for (int i = 0; i < elemcount; i++)
    {
        const unsigned short* ptr = _ptr + i * stride;

        int j = 0;
        while (j + 1 < size1)
        {
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + j * elempack, vlm4), vlm4);
            vfloat32m2_t _m0 = __riscv_vfmv_v_f_f32m2(maxptr[j * elempack], vlm2);
            vfloat32m2_t _m1 = __riscv_vfmv_v_f_f32m2(maxptr[(j + 1) * elempack], vlm2);
            vfloat32m4_t _max = __riscv_vcreate_v_f32m2_f32m4(_m0, _m1);
            _p = exp_ps(__riscv_vfsub_vv_f32m4(_p, _max, vlm4), vlm4);

            vfloat32m4_t _sum = __riscv_vle32_v_f32m4(sumptr + j * elempack, vlm4);
            _sum = __riscv_vfadd_vv_f32m4(_sum, _p, vlm4);
            __riscv_vse32_v_f32m4(sumptr + j * elempack, _sum, vlm4);

            j += 2;
        }
        if (j < size1)
        {
            vfloat32m2_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m1(ptr + j * elempack, vlm2), vlm2);
            vfloat32m2_t _max = __riscv_vfmv_v_f_f32m2(maxptr[j * elempack], vlm2);
            _p = exp_ps(__riscv_vfsub_vv_f32m2(_p, _max, vlm2), vlm2);

            vfloat32m2_t _sum = __riscv_vle32_v_f32m2(sumptr + j * elempack, vlm2);
            _sum = __riscv_vfadd_vv_f32m2(_sum, _p, vlm2);
            __riscv_vse32_v_f32m2(sumptr + j * elempack, _sum, vlm2);
        }
    }

    // fold the sums of every position and keep the reciprocal
    for (int j = 0; j < size1; j++)
    {
        vfloat32m2_t _sum = __riscv_vle32_v_f32m2(sumptr + j * elempack, vlm2);
        vfloat32m1_t _acc = __riscv_vfmv_s_f_f32m1(0.f, vlm1);
        _acc = __riscv_vfredusum_vs_f32m2_f32m1(_sum, _acc, vlm2);
        const float sum = __riscv_vfmv_f_s_f32m1_f32(_acc);
        invsumptr[j] = 1.f / sum;
    }

    // div sum
    for (int i = 0; i < elemcount; i++)
    {
        unsigned short* ptr = _ptr + i * stride;

        int j = 0;
        while (j + 1 < size1)
        {
            vfloat32m4_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + j * elempack, vlm4), vlm4);
            vfloat32m2_t _m0 = __riscv_vfmv_v_f_f32m2(maxptr[j * elempack], vlm2);
            vfloat32m2_t _m1 = __riscv_vfmv_v_f_f32m2(maxptr[(j + 1) * elempack], vlm2);
            vfloat32m4_t _max = __riscv_vcreate_v_f32m2_f32m4(_m0, _m1);
            vfloat32m2_t _is0 = __riscv_vfmv_v_f_f32m2(invsumptr[j], vlm2);
            vfloat32m2_t _is1 = __riscv_vfmv_v_f_f32m2(invsumptr[j + 1], vlm2);
            vfloat32m4_t _isum = __riscv_vcreate_v_f32m2_f32m4(_is0, _is1);

            _p = exp_ps(__riscv_vfsub_vv_f32m4(_p, _max, vlm4), vlm4);
            _p = __riscv_vfmul_vv_f32m4(_p, _isum, vlm4);
            __riscv_vse16_v_u16m2(ptr + j * elempack, float2bfloat_rvv(_p, vlm4), vlm4);

            j += 2;
        }
        if (j < size1)
        {
            vfloat32m2_t _p = bfloat2float_rvv(__riscv_vle16_v_u16m1(ptr + j * elempack, vlm2), vlm2);
            vfloat32m2_t _max = __riscv_vfmv_v_f_f32m2(maxptr[j * elempack], vlm2);
            vfloat32m2_t _isum = __riscv_vfmv_v_f_f32m2(invsumptr[j], vlm2);

            _p = exp_ps(__riscv_vfsub_vv_f32m2(_p, _max, vlm2), vlm2);
            _p = __riscv_vfmul_vv_f32m2(_p, _isum, vlm2);
            __riscv_vse16_v_u16m1(ptr + j * elempack, float2bfloat_rvv(_p, vlm2), vlm2);
        }
    }

#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
    // restore FXCR while retaining accrued exception flags, including FE
    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
}
#endif // __riscv_vector

static int softmax_bf16s(Mat& bottom_top_blob, int axis, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ZVFBFMIN && __riscv_vector && !__riscv_zvfbfmin
    if (cpu_support_riscv_zvfbfmin())
    {
        return softmax_bf16s_zvfbfmin(bottom_top_blob, axis, opt);
    }
#endif

    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        unsigned short* ptr = bottom_top_blob;

        const int size = w * elempack;

        softmax_bf16s(ptr, size, 1, 0);

        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
        const int size = w;
        const int sizen = (size + (opt.num_threads - 1)) / opt.num_threads;
        const size_t stride = (size_t)w * elempack;

        Mat maxsum(sizen * (elempack + 1), 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        const int nn_size = (size + sizen - 1) / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());

            unsigned short* ptr = (unsigned short*)bottom_top_blob + i * elempack;

#if __riscv_vector
            if (elempack > 1)
            {
                softmax_bf16s_packn(ptr, h, stride, size1, elempack, maxsumptr);
            }
            else
#endif // __riscv_vector
            {
                float* maxptr = maxsumptr;
                float* sumptr = maxptr + size1;

                softmax_bf16s_pack(ptr, h, stride, size1, maxptr, sumptr);
            }
        }

        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
        Mat maxsum;
        if (elempack > 1)
        {
            maxsum.create(2 * elempack, opt.num_threads, 4u, opt.workspace_allocator);
            if (maxsum.empty())
                return -100;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);

            float* scratch = elempack > 1 ? maxsum.channel(get_omp_thread_num()) : 0;

            softmax_bf16s(ptr, w, elempack, scratch);
        }

        return 0;
    }

    if ((dims == 3 || dims == 4) && positive_axis == 0)
    {
        const int size = w * h * d;
        const int sizen = (size + (opt.num_threads - 1)) / opt.num_threads;
        const size_t stride = bottom_top_blob.cstep * elempack;

        Mat maxsum(sizen * (elempack + 1), 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        const int nn_size = (size + sizen - 1) / sizen;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            const int i = ii * sizen;
            const int size1 = std::min(sizen, size - i);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());

            unsigned short* ptr = (unsigned short*)bottom_top_blob + i * elempack;

#if __riscv_vector
            if (elempack > 1)
            {
                softmax_bf16s_packn(ptr, channels, stride, size1, elempack, maxsumptr);
            }
            else
#endif // __riscv_vector
            {
                float* maxptr = maxsumptr;
                float* sumptr = maxptr + size1;

                softmax_bf16s_pack(ptr, channels, stride, size1, maxptr, sumptr);
            }
        }

        return 0;
    }

    if ((dims == 3 && positive_axis == 1) || (dims == 4 && positive_axis == 2))
    {
        const int size = w * elempack;

        Mat maxsum(size, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < d; i++)
            {
                unsigned short* ptr = bottom_top_blob.channel(q).depth(i);

                float* maxsumptr = maxsum.channel(get_omp_thread_num());
                float* maxptr = maxsumptr;
                float* sumptr = maxptr + size;

                softmax_bf16s_pack(ptr, h, size, size, maxptr, sumptr);
            }
        }

        return 0;
    }

    if (dims == 3 && positive_axis == 2)
    {
        Mat maxsum;
        if (elempack > 1)
        {
            maxsum.create(2 * elempack, opt.num_threads, 4u, opt.workspace_allocator);
            if (maxsum.empty())
                return -100;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            float* scratch = elempack > 1 ? maxsum.channel(get_omp_thread_num()) : 0;

            for (int i = 0; i < h; i++)
            {
                softmax_bf16s(ptr, w, elempack, scratch);
                ptr += w * elempack;
            }
        }

        return 0;
    }

    if (dims == 4 && positive_axis == 1)
    {
        const int size = w * h * elempack;

        Mat maxsum(size, 2, opt.num_threads, 4u, opt.workspace_allocator);
        if (maxsum.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            float* maxsumptr = maxsum.channel(get_omp_thread_num());
            float* maxptr = maxsumptr;
            float* sumptr = maxptr + size;

            softmax_bf16s_pack(ptr, d, size, size, maxptr, sumptr);
        }

        return 0;
    }

    if (dims == 4 && positive_axis == 3)
    {
        Mat maxsum;
        if (elempack > 1)
        {
            maxsum.create(2 * elempack, opt.num_threads, 4u, opt.workspace_allocator);
            if (maxsum.empty())
                return -100;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);

            float* scratch = elempack > 1 ? maxsum.channel(get_omp_thread_num()) : 0;

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    softmax_bf16s(ptr, w, elempack, scratch);
                    ptr += w * elempack;
                }
            }
        }

        return 0;
    }

    return 0;
}
