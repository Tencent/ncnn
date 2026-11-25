// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

template<typename Conv>
static void glu_vector_16bit(const unsigned short* ptr, const unsigned short* gateptr, unsigned short* outptr, int size)
{
    int i = 0;
#if __ARM_NEON
#if __aarch64__
    for (; i + 15 < size; i += 16)
    {
        float32x4_t _p0 = Conv::load4(ptr);
        float32x4_t _p1 = Conv::load4(ptr + 4);
        float32x4_t _p2 = Conv::load4(ptr + 8);
        float32x4_t _p3 = Conv::load4(ptr + 12);
        float32x4_t _g0 = Conv::load4(gateptr);
        float32x4_t _g1 = Conv::load4(gateptr + 4);
        float32x4_t _g2 = Conv::load4(gateptr + 8);
        float32x4_t _g3 = Conv::load4(gateptr + 12);
        Conv::store4(outptr, vmulq_f32(_p0, sigmoid_ps(_g0)));
        Conv::store4(outptr + 4, vmulq_f32(_p1, sigmoid_ps(_g1)));
        Conv::store4(outptr + 8, vmulq_f32(_p2, sigmoid_ps(_g2)));
        Conv::store4(outptr + 12, vmulq_f32(_p3, sigmoid_ps(_g3)));
        ptr += 16;
        gateptr += 16;
        outptr += 16;
    }
#endif // __aarch64__
    for (; i + 7 < size; i += 8)
    {
        float32x4_t _p0 = Conv::load4(ptr);
        float32x4_t _p1 = Conv::load4(ptr + 4);
        float32x4_t _g0 = Conv::load4(gateptr);
        float32x4_t _g1 = Conv::load4(gateptr + 4);
        Conv::store4(outptr, vmulq_f32(_p0, sigmoid_ps(_g0)));
        Conv::store4(outptr + 4, vmulq_f32(_p1, sigmoid_ps(_g1)));
        ptr += 8;
        gateptr += 8;
        outptr += 8;
    }
    for (; i + 3 < size; i += 4)
    {
        float32x4_t _p = Conv::load4(ptr);
        float32x4_t _g = Conv::load4(gateptr);
        Conv::store4(outptr, vmulq_f32(_p, sigmoid_ps(_g)));
        ptr += 4;
        gateptr += 4;
        outptr += 4;
    }
#endif // __ARM_NEON
    for (; i < size; i++)
    {
        Conv::store1(outptr, Conv::load1(ptr) / (1.f + expf(-Conv::load1(gateptr))));
        ptr++;
        gateptr++;
        outptr++;
    }
}

template<typename Conv>
static int glu_forward_16bit(const Mat& bottom_blob, Mat& top_blob, int axis, const Option& opt)
{
    const int dims = bottom_blob.dims;
    const int positive_axis = dims == 1 ? 0 : axis < 0 ? dims + axis
                                                       : axis;
    if (positive_axis < 0 || positive_axis >= dims)
        return -100;

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int c = bottom_blob.c;
    const size_t elemsize = 2u;

    if (dims == 1)
    {
        const int outw = w / 2;
        top_blob.create(outw, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const unsigned short* ptr = bottom_blob;
        glu_vector_16bit<Conv>(ptr, ptr + outw, top_blob, outw);
        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
        const int outh = h / 2;
        top_blob.create(w, outh, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int size = w * outh;
        const unsigned short* ptr = bottom_blob;
        glu_vector_16bit<Conv>(ptr, ptr + size, top_blob, size);
        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
        const int outw = w / 2;
        top_blob.create(outw, h, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            const unsigned short* ptr = bottom_blob.row<unsigned short>(i);
            unsigned short* outptr = top_blob.row<unsigned short>(i);
            glu_vector_16bit<Conv>(ptr, ptr + outw, outptr, outw);
        }
        return 0;
    }

    if (dims == 3 && positive_axis == 0)
    {
        const int outc = c / 2;
        top_blob.create(w, h, outc, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int size = w * h;
        const size_t offset = (size_t)outc * bottom_blob.cstep;

    #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < outc; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            unsigned short* outptr = top_blob.channel(q);
            glu_vector_16bit<Conv>(ptr, ptr + offset, outptr, size);
        }
        return 0;
    }

    if (dims == 3 && positive_axis == 1)
    {
        const int outh = h / 2;
        top_blob.create(w, outh, c, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int size = w * outh;

    #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            unsigned short* outptr = top_blob.channel(q);
            glu_vector_16bit<Conv>(ptr, ptr + size, outptr, size);
        }
        return 0;
    }

    if (dims == 3 && positive_axis == 2)
    {
        const int outw = w / 2;
        top_blob.create(outw, h, c, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
        for (int qi = 0; qi < c * h; qi++)
        {
            const int q = qi / h;
            const int i = qi - q * h;
            const unsigned short* ptr = bottom_blob.channel(q).row<unsigned short>(i);
            unsigned short* outptr = top_blob.channel(q).row<unsigned short>(i);
            glu_vector_16bit<Conv>(ptr, ptr + outw, outptr, outw);
        }
        return 0;
    }

    if (dims == 4 && positive_axis == 0)
    {
        const int outc = c / 2;
        top_blob.create(w, h, d, outc, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int size = w * h * d;
        const size_t offset = (size_t)outc * bottom_blob.cstep;

    #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < outc; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            unsigned short* outptr = top_blob.channel(q);
            glu_vector_16bit<Conv>(ptr, ptr + offset, outptr, size);
        }
        return 0;
    }

    if (dims == 4 && positive_axis == 1)
    {
        const int outd = d / 2;
        top_blob.create(w, h, outd, c, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int size = w * h * outd;

    #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            unsigned short* outptr = top_blob.channel(q);
            glu_vector_16bit<Conv>(ptr, ptr + size, outptr, size);
        }
        return 0;
    }

    if (dims == 4 && positive_axis == 2)
    {
        const int outh = h / 2;
        top_blob.create(w, outh, d, c, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int size = w * outh;

    #pragma omp parallel for num_threads(opt.num_threads)
        for (int qz = 0; qz < c * d; qz++)
        {
            const int q = qz / d;
            const int z = qz - q * d;
            const unsigned short* ptr = bottom_blob.channel(q).depth(z);
            unsigned short* outptr = top_blob.channel(q).depth(z);
            glu_vector_16bit<Conv>(ptr, ptr + size, outptr, size);
        }
        return 0;
    }

    if (dims == 4 && positive_axis == 3)
    {
        const int outw = w / 2;
        top_blob.create(outw, h, d, c, elemsize, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
        for (int qzi = 0; qzi < c * d * h; qzi++)
        {
            const int q = qzi / (d * h);
            const int zi = qzi - q * (d * h);
            const int z = zi / h;
            const int i = zi - z * h;
            const unsigned short* ptr = bottom_blob.channel(q).depth(z).row<unsigned short>(i);
            unsigned short* outptr = top_blob.channel(q).depth(z).row<unsigned short>(i);
            glu_vector_16bit<Conv>(ptr, ptr + outw, outptr, outw);
        }
        return 0;
    }

    return -100;
}

template<typename Conv>
static int glu_forward_16bit_dispatch(const Mat& bottom_blob, Mat& top_blob, int axis, const Option& opt)
{
    if (bottom_blob.elempack == 1)
        return glu_forward_16bit<Conv>(bottom_blob, top_blob, axis, opt);

    Option opt_pack1 = opt;
    opt_pack1.blob_allocator = opt.workspace_allocator;

    Mat bottom_blob_pack1;
    convert_packing(bottom_blob, bottom_blob_pack1, 1, opt_pack1);
    if (bottom_blob_pack1.empty())
        return -100;

    return glu_forward_16bit<Conv>(bottom_blob_pack1, top_blob, axis, opt);
}
