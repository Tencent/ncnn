// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

#include "cpu.h"

#include <string.h>

namespace ncnn {

Reshape_riscv::Reshape_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Reshape_riscv::load_param(const ParamDict& pd)
{
    return Reshape::load_param(pd);
}

int Reshape_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    if (batch_mode == 0)
    {
        int elembits = bottom_blob.elembits();

        if (elembits == 16 && (opt.use_fp16_storage || opt.use_bf16_storage))
            return forward_bf16s_fp16s(bottom_blobs, top_blobs, opt);

        if (elembits != 32)
            return Reshape::forward(bottom_blobs, top_blobs, opt);

        int outw = w;
        int outh = h;
        int outd = d;
        int outc = c;

        if (!shape_expr.empty())
        {
            int er = eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
            if (er != 0)
                return -1;
        }

        if (ndim == 1)
        {
            flatten(bottom_blob, top_blob, opt);
            if (top_blob.empty())
                return -100;

            return 0;
        }

        const int dims = bottom_blob.dims;
        const int elempack = bottom_blob.elempack;
        const size_t elemsize = bottom_blob.elemsize;

        const int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;

#if __riscv_vector
        const int packn = csrr_vlenb() / 4;
#endif // __riscv_vector

        if (ndim == 2)
        {
            if (outw == 0)
                outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
            if (outh == 0)
                outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

            if (outw == -1)
                outw = total / outh;
            if (outh == -1)
                outh = total / outw;

            int out_elempack = 1;
#if __riscv_vector
            if (opt.use_packing_layout)
            {
                out_elempack = outh % packn == 0 ? packn : 1;
            }
#endif // __riscv_vector
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (dims == 2 && bottom_blob.h * elempack == outh && elempack == out_elempack)
            {
                top_blob = bottom_blob;
                return 0;
            }

            if (out_elempack == 1)
            {
                flatten(bottom_blob, top_blob, opt);
                if (top_blob.empty())
                    return -100;

                top_blob.dims = 2;
                top_blob.w = outw;
                top_blob.h = outh;
                top_blob.cstep = top_blob.cstep * top_blob.elempack;
                top_blob.elemsize = out_elemsize;
                top_blob.elempack = out_elempack;

                return 0;
            }

            Mat bottom_blob_flattened = bottom_blob;
            {
                Option opt_flatten = opt;
                opt_flatten.blob_allocator = opt.workspace_allocator;

                flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
                if (bottom_blob_flattened.empty())
                    return -100;
            }

            top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

#if __riscv_vector
            if (out_elempack == packn)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < top_blob.h; i++)
                {
                    const float* ptr = (const float*)bottom_blob_flattened + outw * i * packn;
                    float* outptr = top_blob.row(i);

                    for (int j = 0; j < outw; j++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(packn);

                        vfloat32m1_t _p = __riscv_vlse32_v_f32m1(ptr + j, outw * sizeof(float), vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }
#endif // __riscv_vector
        }

        if (ndim == 3 || ndim == 4)
        {
            if (ndim == 3)
            {
                if (outw == 0)
                    outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
                if (outh == 0)
                    outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
                if (outc == 0)
                    outc = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

                if (outw == -1)
                    outw = total / outc / outh;
                if (outh == -1)
                    outh = total / outc / outw;
                if (outc == -1)
                    outc = total / outh / outw;

                outd = 1;
            }
            else
            {
                if (outw == 0)
                    outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
                if (outh == 0)
                    outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
                if (outd == 0)
                    outd = bottom_blob.d;
                if (outc == 0)
                    outc = (dims == 3 || dims == 4) ? bottom_blob.c * elempack : bottom_blob.c;

                if (outw == -1)
                    outw = total / outc / outd / outh;
                if (outh == -1)
                    outh = total / outc / outd / outw;
                if (outd == -1)
                    outd = total / outc / outh / outw;
                if (outc == -1)
                    outc = total / outd / outh / outw;
            }

            int out_elempack = 1;
#if __riscv_vector
            if (opt.use_packing_layout)
            {
                out_elempack = outc % packn == 0 ? packn : 1;
            }
#endif // __riscv_vector
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if ((dims == 3 || dims == 4) && bottom_blob.c * elempack == outc && elempack == out_elempack)
            {
                top_blob = bottom_blob;
                top_blob.dims = ndim;
                top_blob.w = outw;
                top_blob.h = outh;
                top_blob.d = outd;
                return 0;
            }

            Mat bottom_blob_flattened = bottom_blob;
            {
                Option opt_flatten = opt;
                opt_flatten.blob_allocator = opt.workspace_allocator;

                flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
                if (bottom_blob_flattened.empty())
                    return -100;
            }

            if (ndim == 3)
                top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            else
                top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);

            if (top_blob.empty())
                return -100;

            int size = top_blob.w * top_blob.h * top_blob.d;

#if __riscv_vector
            if (out_elempack == packn)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* ptr = (const float*)bottom_blob_flattened + size * q * packn;
                    float* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(packn);

                        vfloat32m1_t _p = __riscv_vlse32_v_f32m1(ptr + i, size * sizeof(float), vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }
#endif // __riscv_vector

            if (out_elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    const float* ptr = (const float*)bottom_blob_flattened + size * q;
                    float* outptr = top_blob.channel(q);

                    int i = 0;
#if __riscv_vector
                    int n = size;
                    while (n > 0)
                    {
                        size_t vl = __riscv_vsetvl_e32m8(n);

                        vfloat32m8_t _p = __riscv_vle32_v_f32m8(ptr, vl);
                        __riscv_vse32_v_f32m8(outptr, _p, vl);

                        ptr += vl;
                        outptr += vl;
                        n -= vl;
                        i += vl;
                    }
#endif // __riscv_vector
                    for (; i < size; i++)
                    {
                        *outptr++ = *ptr++;
                    }
                }
            }
        }

        return 0;
    }

    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

    if (!shape_expr.empty())
    {
        int er = eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
        if (er != 0)
            return -1;
    }

    if (batch_mode == 2 && (outw == -1 || outh == -1 || outd == -1 || outc == -1))
        return -1;

    const int dims = bottom_blob.dims;
    const int elempack = bottom_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const size_t scalar_elemsize = elemsize / elempack;

    int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;
    if (batch_mode == 1)
        total *= bottom_blob.n;

    if (ndim == 0)
        return -1;

    if (ndim == 1)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outw == -1)
            outw = total;
    }
    if (ndim == 2)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;
    }
    if (ndim == 3)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (outc == 0)
            outc = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;
        if (outw == -1)
            outw = total / outc / outh;
        if (outh == -1)
            outh = total / outc / outw;
        if (outc == -1)
            outc = total / outh / outw;
    }
    if (ndim == 4)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (outc == 0)
            outc = (dims == 3 || dims == 4) ? bottom_blob.c * elempack : bottom_blob.c;
        if (outd == 0)
            outd = bottom_blob.d;
        if (outw == -1)
            outw = total / outc / outd / outh;
        if (outh == -1)
            outh = total / outc / outd / outw;
        if (outd == -1)
            outd = total / outc / outh / outw;
        if (outc == -1)
            outc = total / outd / outh / outw;
    }

    int out_elempack = 1;
#if __riscv_vector
    int packn = csrr_vlenb() / 4;
    if (bottom_blob.elembits() == 16)
        packn = csrr_vlenb() / 2;
    if (bottom_blob.elembits() == 8)
        packn = csrr_vlenb();

    if (opt.use_packing_layout)
    {
        if (ndim == 1)
            out_elempack = outw % packn == 0 ? packn : 1;
        if (ndim == 2)
            out_elempack = outh % packn == 0 ? packn : 1;
        if (ndim == 3 || ndim == 4)
            out_elempack = outc % packn == 0 ? packn : 1;
    }
#endif // __riscv_vector
    const size_t out_elemsize = scalar_elemsize * out_elempack;

    int shape[4] = {0, 0, 0, 0};
    if (ndim == 1)
        shape[0] = outw;
    if (ndim == 2)
    {
        shape[0] = outh;
        shape[1] = outw;
    }
    if (ndim == 3)
    {
        shape[0] = outc;
        shape[1] = outh;
        shape[2] = outw;
    }
    if (ndim == 4)
    {
        shape[0] = outc;
        shape[1] = outd;
        shape[2] = outh;
        shape[3] = outw;
    }

    if (batch_mode == 1)
    {
        if (ndim == 1)
            top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 2)
            top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 3)
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 4)
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);

        if (top_blob.empty())
            return -100;

        if (batch_axis == 0 && elempack == out_elempack && dims == 1 && ndim == 1 && top_blob.w == bottom_blob.w * bottom_blob.n)
        {
            const size_t size = (size_t)bottom_blob.w * elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int b = 0; b < bottom_blob.n; b++)
            {
                const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * bottom_blob.nstep * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * bottom_blob.w * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == out_elempack && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && top_blob.h == bottom_blob.h * bottom_blob.n)
        {
            const size_t size = (size_t)bottom_blob.w * bottom_blob.h * elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int b = 0; b < bottom_blob.n; b++)
            {
                const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * bottom_blob.nstep * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * bottom_blob.w * bottom_blob.h * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == out_elempack && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && top_blob.c == bottom_blob.c * bottom_blob.n)
        {
            const size_t size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d * elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < bottom_blob.n * bottom_blob.c; bq++)
            {
                const int b = bq / bottom_blob.c;
                const int q = bq - b * bottom_blob.c;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)q * bottom_blob.cstep) * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + (size_t)bq * top_blob.cstep * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == out_elempack && dims == 2 && ndim == 3 && bottom_blob.w == top_blob.w && top_blob.h == bottom_blob.n && top_blob.c == bottom_blob.h)
        {
            const size_t size = (size_t)bottom_blob.w * elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < bottom_blob.n * bottom_blob.h; bq++)
            {
                const int b = bq / bottom_blob.h;
                const int q = bq - b * bottom_blob.h;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)q * bottom_blob.w) * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + ((size_t)q * top_blob.cstep + (size_t)b * top_blob.w) * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == out_elempack && dims == 3 && ndim == 4 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && top_blob.d == bottom_blob.n && top_blob.c == bottom_blob.c)
        {
            const size_t size = (size_t)bottom_blob.w * bottom_blob.h * elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < bottom_blob.n * bottom_blob.c; bq++)
            {
                const int b = bq / bottom_blob.c;
                const int q = bq - b * bottom_blob.c;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)q * bottom_blob.cstep) * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + ((size_t)q * top_blob.cstep + (size_t)b * top_blob.w * top_blob.h) * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

#if __riscv_vector
        if (batch_axis == 1 && elempack == 1 && out_elempack == packn && dims == 2 && ndim == 3 && bottom_blob.w == top_blob.w && top_blob.h == bottom_blob.n && top_blob.c * packn == bottom_blob.h)
        {
            const int size = bottom_blob.w;

            if (scalar_elemsize == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < bottom_blob.n * top_blob.c; bq++)
                {
                    const int b = bq / top_blob.c;
                    const int q = bq - b * top_blob.c;

                    const float* ptr = (const float*)bottom_blob + (size_t)b * bottom_blob.nstep + (size_t)(q * packn) * bottom_blob.w;
                    float* outptr = (float*)top_blob + ((size_t)q * top_blob.cstep + (size_t)b * top_blob.w) * packn;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(packn);

                        vfloat32m1_t _p = __riscv_vlse32_v_f32m1(ptr + i, bottom_blob.w * sizeof(float), vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }
            if (scalar_elemsize == 2)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < bottom_blob.n * top_blob.c; bq++)
                {
                    const int b = bq / top_blob.c;
                    const int q = bq - b * top_blob.c;

                    const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)b * bottom_blob.nstep + (size_t)(q * packn) * bottom_blob.w;
                    unsigned short* outptr = (unsigned short*)top_blob + ((size_t)q * top_blob.cstep + (size_t)b * top_blob.w) * packn;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e16m1(packn);

                        vuint16m1_t _p = __riscv_vlse16_v_u16m1(ptr + i, bottom_blob.w * sizeof(unsigned short), vl);
                        __riscv_vse16_v_u16m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == packn && out_elempack == 1 && dims == 2 && ndim == 3 && bottom_blob.w == top_blob.w && top_blob.h == bottom_blob.n && top_blob.c == bottom_blob.h * packn)
        {
            const int size = bottom_blob.w;

            if (scalar_elemsize == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < bottom_blob.n * bottom_blob.h; bq++)
                {
                    const int b = bq / bottom_blob.h;
                    const int q = bq - b * bottom_blob.h;

                    const float* ptr = (const float*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)q * bottom_blob.w) * packn;
                    float* outptr = (float*)top_blob + (size_t)(q * packn) * top_blob.cstep + (size_t)b * top_blob.w;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(packn);

                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
                        __riscv_vsse32_v_f32m1(outptr, top_blob.cstep * sizeof(float), _p, vl);

                        ptr += vl;
                        outptr++;
                    }
                }
            }
            if (scalar_elemsize == 2)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < bottom_blob.n * bottom_blob.h; bq++)
                {
                    const int b = bq / bottom_blob.h;
                    const int q = bq - b * bottom_blob.h;

                    const unsigned short* ptr = (const unsigned short*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)q * bottom_blob.w) * packn;
                    unsigned short* outptr = (unsigned short*)top_blob + (size_t)(q * packn) * top_blob.cstep + (size_t)b * top_blob.w;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e16m1(packn);

                        vuint16m1_t _p = __riscv_vle16_v_u16m1(ptr, vl);
                        __riscv_vsse16_v_u16m1(outptr, top_blob.cstep * sizeof(unsigned short), _p, vl);

                        ptr += vl;
                        outptr++;
                    }
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == 1 && out_elempack == packn && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && top_blob.h * packn == bottom_blob.h * bottom_blob.n)
        {
            if (scalar_elemsize == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < top_blob.h; i++)
                {
                    const int y = i * packn;
                    const int b = y / bottom_blob.h;
                    const int sq = y - b * bottom_blob.h;

                    const float* ptr = (const float*)bottom_blob + (size_t)b * bottom_blob.nstep + (size_t)sq * bottom_blob.w;
                    float* outptr = top_blob.row(i);

                    for (int j = 0; j < bottom_blob.w; j++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(packn);

                        vfloat32m1_t _p = __riscv_vlse32_v_f32m1(ptr + j, bottom_blob.w * sizeof(float), vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }
            if (scalar_elemsize == 2)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < top_blob.h; i++)
                {
                    const int y = i * packn;
                    const int b = y / bottom_blob.h;
                    const int sq = y - b * bottom_blob.h;

                    const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)b * bottom_blob.nstep + (size_t)sq * bottom_blob.w;
                    unsigned short* outptr = top_blob.row<unsigned short>(i);

                    for (int j = 0; j < bottom_blob.w; j++)
                    {
                        size_t vl = __riscv_vsetvl_e16m1(packn);

                        vuint16m1_t _p = __riscv_vlse16_v_u16m1(ptr + j, bottom_blob.w * sizeof(unsigned short), vl);
                        __riscv_vse16_v_u16m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == packn && out_elempack == 1 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && top_blob.h == bottom_blob.h * bottom_blob.n * packn)
        {
            if (scalar_elemsize == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bi = 0; bi < bottom_blob.n * bottom_blob.h; bi++)
                {
                    const int b = bi / bottom_blob.h;
                    const int i = bi - b * bottom_blob.h;
                    const int y = bi * packn;

                    const float* ptr = (const float*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)i * bottom_blob.w) * packn;
                    float* outptr = (float*)top_blob + (size_t)y * top_blob.w;

                    for (int j = 0; j < bottom_blob.w; j++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(packn);

                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
                        __riscv_vsse32_v_f32m1(outptr, top_blob.w * sizeof(float), _p, vl);

                        ptr += vl;
                        outptr++;
                    }
                }
            }
            if (scalar_elemsize == 2)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bi = 0; bi < bottom_blob.n * bottom_blob.h; bi++)
                {
                    const int b = bi / bottom_blob.h;
                    const int i = bi - b * bottom_blob.h;
                    const int y = bi * packn;

                    const unsigned short* ptr = (const unsigned short*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)i * bottom_blob.w) * packn;
                    unsigned short* outptr = (unsigned short*)top_blob + (size_t)y * top_blob.w;

                    for (int j = 0; j < bottom_blob.w; j++)
                    {
                        size_t vl = __riscv_vsetvl_e16m1(packn);

                        vuint16m1_t _p = __riscv_vle16_v_u16m1(ptr, vl);
                        __riscv_vsse16_v_u16m1(outptr, top_blob.w * sizeof(unsigned short), _p, vl);

                        ptr += vl;
                        outptr++;
                    }
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == 1 && out_elempack == packn && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && top_blob.c * packn == bottom_blob.c * bottom_blob.n)
        {
            const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

            if (scalar_elemsize == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    const int bq = q * packn;
                    const int b = bq / bottom_blob.c;
                    const int sq = bq - b * bottom_blob.c;

                    const float* ptr = (const float*)bottom_blob + (size_t)b * bottom_blob.nstep + (size_t)sq * bottom_blob.cstep;
                    float* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(packn);

                        vfloat32m1_t _p = __riscv_vlse32_v_f32m1(ptr + i, bottom_blob.cstep * sizeof(float), vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }
            if (scalar_elemsize == 2)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < top_blob.c; q++)
                {
                    const int bq = q * packn;
                    const int b = bq / bottom_blob.c;
                    const int sq = bq - b * bottom_blob.c;

                    const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)b * bottom_blob.nstep + (size_t)sq * bottom_blob.cstep;
                    unsigned short* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e16m1(packn);

                        vuint16m1_t _p = __riscv_vlse16_v_u16m1(ptr + i, bottom_blob.cstep * sizeof(unsigned short), vl);
                        __riscv_vse16_v_u16m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == 1 && out_elempack == packn && dims == 3 && ndim == 4 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && top_blob.d == bottom_blob.n && top_blob.c * packn == bottom_blob.c)
        {
            const int size = bottom_blob.w * bottom_blob.h;

            if (scalar_elemsize == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < bottom_blob.n * top_blob.c; bq++)
                {
                    const int b = bq / top_blob.c;
                    const int q = bq - b * top_blob.c;

                    const float* ptr = (const float*)bottom_blob + (size_t)b * bottom_blob.nstep + (size_t)(q * packn) * bottom_blob.cstep;
                    float* outptr = (float*)top_blob + ((size_t)q * top_blob.cstep + (size_t)b * top_blob.w * top_blob.h) * packn;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(packn);

                        vfloat32m1_t _p = __riscv_vlse32_v_f32m1(ptr + i, bottom_blob.cstep * sizeof(float), vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }
            if (scalar_elemsize == 2)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < bottom_blob.n * top_blob.c; bq++)
                {
                    const int b = bq / top_blob.c;
                    const int q = bq - b * top_blob.c;

                    const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)b * bottom_blob.nstep + (size_t)(q * packn) * bottom_blob.cstep;
                    unsigned short* outptr = (unsigned short*)top_blob + ((size_t)q * top_blob.cstep + (size_t)b * top_blob.w * top_blob.h) * packn;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e16m1(packn);

                        vuint16m1_t _p = __riscv_vlse16_v_u16m1(ptr + i, bottom_blob.cstep * sizeof(unsigned short), vl);
                        __riscv_vse16_v_u16m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }

            return 0;
        }
#endif // __riscv_vector

        size_t prefix = 1;
        for (int i = 0; i < batch_axis; i++)
            prefix *= shape[i];

        size_t suffix = 1;
        if (batch_axis == 0)
            suffix = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;
        else
        {
            for (int i = batch_axis + 1; i < ndim; i++)
                suffix *= shape[i];
        }

        const size_t bottom_channel_size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d;
        const size_t top_channel_size = (size_t)top_blob.w * top_blob.h * top_blob.d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < (int)prefix; pp++)
        {
            for (int b = 0; b < bottom_blob.n; b++)
            {
                for (size_t s = 0; s < suffix; s++)
                {
                    const size_t srci = batch_axis == 0 ? s : (size_t)pp * suffix + s;
                    const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * bottom_blob.nstep * elemsize;
                    if (dims == 1)
                    {
                        const int x = srci / elempack;
                        const int k = srci % elempack;
                        ptr += (size_t)x * elemsize + k * scalar_elemsize;
                    }
                    else if (dims == 2)
                    {
                        const int x = srci % bottom_blob.w;
                        const int y = srci / bottom_blob.w;
                        const int y0 = y / elempack;
                        const int k = y % elempack;
                        ptr += ((size_t)y0 * bottom_blob.w + x) * elemsize + k * scalar_elemsize;
                    }
                    else
                    {
                        const int q = srci / bottom_channel_size;
                        const size_t r = srci - (size_t)q * bottom_channel_size;
                        const int q0 = q / elempack;
                        const int k = q % elempack;
                        ptr += ((size_t)q0 * bottom_blob.cstep + r) * elemsize + k * scalar_elemsize;
                    }

                    const size_t dsti = batch_axis == 0 ? (size_t)b * suffix + s : ((size_t)pp * bottom_blob.n + b) * suffix + s;
                    unsigned char* outptr = (unsigned char*)top_blob;
                    if (top_blob.dims == 1)
                    {
                        const int x = dsti / out_elempack;
                        const int k = dsti % out_elempack;
                        outptr += (size_t)x * out_elemsize + k * scalar_elemsize;
                    }
                    else if (top_blob.dims == 2)
                    {
                        const int x = dsti % top_blob.w;
                        const int y = dsti / top_blob.w;
                        const int y0 = y / out_elempack;
                        const int k = y % out_elempack;
                        outptr += ((size_t)y0 * top_blob.w + x) * out_elemsize + k * scalar_elemsize;
                    }
                    else
                    {
                        const int q = dsti / top_channel_size;
                        const size_t r = dsti - (size_t)q * top_channel_size;
                        const int q0 = q / out_elempack;
                        const int k = q % out_elempack;
                        outptr += ((size_t)q0 * top_blob.cstep + r) * out_elemsize + k * scalar_elemsize;
                    }

                    memcpy(outptr, ptr, scalar_elemsize);
                }
            }
        }

        return 0;
    }

    if (batch_mode == 2)
    {
        if (bottom_blob.n != 1)
            return -1;

        size_t out_total = outw;
        if (ndim == 2)
            out_total *= outh;
        if (ndim == 3)
            out_total *= (size_t)outh * outc;
        if (ndim == 4)
            out_total *= (size_t)outh * outd * outc;

        if (out_total == 0)
            return -1;

        const size_t bottom_total = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;
        const int batch = bottom_total / out_total;
        if ((size_t)batch * out_total != bottom_total)
            return -1;

        if (ndim == 1)
            top_blob.create_batch(outw / out_elempack, batch, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 2)
            top_blob.create_batch(outw, outh / out_elempack, batch, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 3)
            top_blob.create_batch(outw, outh, outc / out_elempack, batch, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 4)
            top_blob.create_batch(outw, outh, outd, outc / out_elempack, batch, out_elemsize, out_elempack, opt.blob_allocator);

        if (top_blob.empty())
            return -100;

        if (batch_axis == 0 && elempack == out_elempack && dims == 1 && ndim == 1 && bottom_blob.w == top_blob.w * batch)
        {
            const size_t size = (size_t)top_blob.w * out_elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int b = 0; b < batch; b++)
            {
                const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * top_blob.w * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * top_blob.nstep * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == out_elempack && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h * batch)
        {
            const size_t size = (size_t)top_blob.w * top_blob.h * out_elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int b = 0; b < batch; b++)
            {
                const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * top_blob.w * top_blob.h * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * top_blob.nstep * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == out_elempack && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c == top_blob.c * batch)
        {
            const size_t size = (size_t)top_blob.w * top_blob.h * top_blob.d * out_elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)bq * bottom_blob.cstep * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == out_elempack && dims == 3 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == batch && bottom_blob.c == top_blob.h)
        {
            const size_t size = (size_t)top_blob.w * out_elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.h; bq++)
            {
                const int b = bq / top_blob.h;
                const int q = bq - b * top_blob.h;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)q * bottom_blob.cstep + (size_t)b * bottom_blob.w) * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.w) * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == out_elempack && dims == 4 && ndim == 3 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == batch && bottom_blob.c == top_blob.c)
        {
            const size_t size = (size_t)top_blob.w * top_blob.h * out_elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)q * bottom_blob.cstep + (size_t)b * bottom_blob.w * bottom_blob.h) * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

#if __riscv_vector
        if (batch_axis == 1 && elempack == 1 && out_elempack == packn && dims == 3 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == batch && bottom_blob.c == top_blob.h * packn)
        {
            const int size = top_blob.w;

            if (scalar_elemsize == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < batch * top_blob.h; bq++)
                {
                    const int b = bq / top_blob.h;
                    const int q = bq - b * top_blob.h;

                    const float* ptr = (const float*)bottom_blob + (size_t)(q * packn) * bottom_blob.cstep + (size_t)b * bottom_blob.w;
                    float* outptr = (float*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.w) * packn;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(packn);

                        vfloat32m1_t _p = __riscv_vlse32_v_f32m1(ptr + i, bottom_blob.cstep * sizeof(float), vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }
            if (scalar_elemsize == 2)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < batch * top_blob.h; bq++)
                {
                    const int b = bq / top_blob.h;
                    const int q = bq - b * top_blob.h;

                    const unsigned short* ptr = (const unsigned short*)bottom_blob + (size_t)(q * packn) * bottom_blob.cstep + (size_t)b * bottom_blob.w;
                    unsigned short* outptr = (unsigned short*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.w) * packn;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e16m1(packn);

                        vuint16m1_t _p = __riscv_vlse16_v_u16m1(ptr + i, bottom_blob.cstep * sizeof(unsigned short), vl);
                        __riscv_vse16_v_u16m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == packn && out_elempack == 1 && dims == 3 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == batch && bottom_blob.c * packn == top_blob.h)
        {
            const int size = top_blob.w;

            if (scalar_elemsize == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < batch * bottom_blob.c; bq++)
                {
                    const int b = bq / bottom_blob.c;
                    const int q = bq - b * bottom_blob.c;

                    const float* ptr = (const float*)bottom_blob + ((size_t)q * bottom_blob.cstep + (size_t)b * bottom_blob.w) * packn;
                    float* outptr = (float*)top_blob + (size_t)b * top_blob.nstep + (size_t)(q * packn) * top_blob.w;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(packn);

                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
                        __riscv_vsse32_v_f32m1(outptr, top_blob.w * sizeof(float), _p, vl);

                        ptr += vl;
                        outptr++;
                    }
                }
            }
            if (scalar_elemsize == 2)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < batch * bottom_blob.c; bq++)
                {
                    const int b = bq / bottom_blob.c;
                    const int q = bq - b * bottom_blob.c;

                    const unsigned short* ptr = (const unsigned short*)bottom_blob + ((size_t)q * bottom_blob.cstep + (size_t)b * bottom_blob.w) * packn;
                    unsigned short* outptr = (unsigned short*)top_blob + (size_t)b * top_blob.nstep + (size_t)(q * packn) * top_blob.w;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e16m1(packn);

                        vuint16m1_t _p = __riscv_vle16_v_u16m1(ptr, vl);
                        __riscv_vsse16_v_u16m1(outptr, top_blob.w * sizeof(unsigned short), _p, vl);

                        ptr += vl;
                        outptr++;
                    }
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == 1 && out_elempack == packn && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h * batch * packn)
        {
            if (scalar_elemsize == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bi = 0; bi < batch * top_blob.h; bi++)
                {
                    const int b = bi / top_blob.h;
                    const int i = bi - b * top_blob.h;
                    const int y = bi * packn;

                    const float* ptr = bottom_blob.row(y);
                    float* outptr = (float*)top_blob + ((size_t)b * top_blob.nstep + (size_t)i * top_blob.w) * packn;

                    for (int j = 0; j < bottom_blob.w; j++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(packn);

                        vfloat32m1_t _p = __riscv_vlse32_v_f32m1(ptr + j, bottom_blob.w * sizeof(float), vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }
            if (scalar_elemsize == 2)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bi = 0; bi < batch * top_blob.h; bi++)
                {
                    const int b = bi / top_blob.h;
                    const int i = bi - b * top_blob.h;
                    const int y = bi * packn;

                    const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                    unsigned short* outptr = (unsigned short*)top_blob + ((size_t)b * top_blob.nstep + (size_t)i * top_blob.w) * packn;

                    for (int j = 0; j < bottom_blob.w; j++)
                    {
                        size_t vl = __riscv_vsetvl_e16m1(packn);

                        vuint16m1_t _p = __riscv_vlse16_v_u16m1(ptr + j, bottom_blob.w * sizeof(unsigned short), vl);
                        __riscv_vse16_v_u16m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == packn && out_elempack == 1 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h * packn == top_blob.h * batch)
        {
            if (scalar_elemsize == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < bottom_blob.h; i++)
                {
                    const int y = i * packn;
                    const int b = y / top_blob.h;
                    const int sq = y - b * top_blob.h;

                    const float* ptr = bottom_blob.row(i);
                    float* outptr = (float*)top_blob + (size_t)b * top_blob.nstep + (size_t)sq * top_blob.w;

                    for (int j = 0; j < bottom_blob.w; j++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(packn);

                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
                        __riscv_vsse32_v_f32m1(outptr, top_blob.w * sizeof(float), _p, vl);

                        ptr += vl;
                        outptr++;
                    }
                }
            }
            if (scalar_elemsize == 2)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < bottom_blob.h; i++)
                {
                    const int y = i * packn;
                    const int b = y / top_blob.h;
                    const int sq = y - b * top_blob.h;

                    const unsigned short* ptr = bottom_blob.row<const unsigned short>(i);
                    unsigned short* outptr = (unsigned short*)top_blob + (size_t)b * top_blob.nstep + (size_t)sq * top_blob.w;

                    for (int j = 0; j < bottom_blob.w; j++)
                    {
                        size_t vl = __riscv_vsetvl_e16m1(packn);

                        vuint16m1_t _p = __riscv_vle16_v_u16m1(ptr, vl);
                        __riscv_vsse16_v_u16m1(outptr, top_blob.w * sizeof(unsigned short), _p, vl);

                        ptr += vl;
                        outptr++;
                    }
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == 1 && out_elempack == packn && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c == top_blob.c * batch * packn)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            if (scalar_elemsize == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < batch * top_blob.c; bq++)
                {
                    const int b = bq / top_blob.c;
                    const int q = bq - b * top_blob.c;
                    const int sq = b * top_blob.c * packn + q * packn;

                    const float* ptr = bottom_blob.channel(sq);
                    float* outptr = (float*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * packn;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(packn);

                        vfloat32m1_t _p = __riscv_vlse32_v_f32m1(ptr + i, bottom_blob.cstep * sizeof(float), vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }
            if (scalar_elemsize == 2)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < batch * top_blob.c; bq++)
                {
                    const int b = bq / top_blob.c;
                    const int q = bq - b * top_blob.c;
                    const int sq = b * top_blob.c * packn + q * packn;

                    const unsigned short* ptr = bottom_blob.channel(sq);
                    unsigned short* outptr = (unsigned short*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * packn;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e16m1(packn);

                        vuint16m1_t _p = __riscv_vlse16_v_u16m1(ptr + i, bottom_blob.cstep * sizeof(unsigned short), vl);
                        __riscv_vse16_v_u16m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == 1 && out_elempack == packn && dims == 4 && ndim == 3 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == batch && bottom_blob.c == top_blob.c * packn)
        {
            const int size = top_blob.w * top_blob.h;

            if (scalar_elemsize == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < batch * top_blob.c; bq++)
                {
                    const int b = bq / top_blob.c;
                    const int q = bq - b * top_blob.c;

                    const float* ptr = (const float*)bottom_blob + ((size_t)(q * packn) * bottom_blob.cstep + (size_t)b * bottom_blob.w * bottom_blob.h);
                    float* outptr = (float*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * packn;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(packn);

                        vfloat32m1_t _p = __riscv_vlse32_v_f32m1(ptr + i, bottom_blob.cstep * sizeof(float), vl);
                        __riscv_vse32_v_f32m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }
            if (scalar_elemsize == 2)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < batch * top_blob.c; bq++)
                {
                    const int b = bq / top_blob.c;
                    const int q = bq - b * top_blob.c;

                    const unsigned short* ptr = (const unsigned short*)bottom_blob + ((size_t)(q * packn) * bottom_blob.cstep + (size_t)b * bottom_blob.w * bottom_blob.h);
                    unsigned short* outptr = (unsigned short*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * packn;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e16m1(packn);

                        vuint16m1_t _p = __riscv_vlse16_v_u16m1(ptr + i, bottom_blob.cstep * sizeof(unsigned short), vl);
                        __riscv_vse16_v_u16m1(outptr, _p, vl);

                        outptr += vl;
                    }
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == packn && out_elempack == 1 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c * packn == top_blob.c * batch)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            if (scalar_elemsize == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < bottom_blob.c; q++)
                {
                    const int bq = q * packn;
                    const int b = bq / top_blob.c;
                    const int sq = bq - b * top_blob.c;

                    const float* ptr = bottom_blob.channel(q);
                    float* outptr = (float*)top_blob + (size_t)b * top_blob.nstep + (size_t)sq * top_blob.cstep;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(packn);

                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
                        __riscv_vsse32_v_f32m1(outptr, top_blob.cstep * sizeof(float), _p, vl);

                        ptr += vl;
                        outptr++;
                    }
                }
            }
            if (scalar_elemsize == 2)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < bottom_blob.c; q++)
                {
                    const int bq = q * packn;
                    const int b = bq / top_blob.c;
                    const int sq = bq - b * top_blob.c;

                    const unsigned short* ptr = bottom_blob.channel(q);
                    unsigned short* outptr = (unsigned short*)top_blob + (size_t)b * top_blob.nstep + (size_t)sq * top_blob.cstep;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e16m1(packn);

                        vuint16m1_t _p = __riscv_vle16_v_u16m1(ptr, vl);
                        __riscv_vsse16_v_u16m1(outptr, top_blob.cstep * sizeof(unsigned short), _p, vl);

                        ptr += vl;
                        outptr++;
                    }
                }
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == packn && out_elempack == 1 && dims == 4 && ndim == 3 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == batch && bottom_blob.c * packn == top_blob.c)
        {
            const int size = top_blob.w * top_blob.h;

            if (scalar_elemsize == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < batch * bottom_blob.c; bq++)
                {
                    const int b = bq / bottom_blob.c;
                    const int q = bq - b * bottom_blob.c;

                    const float* ptr = (const float*)bottom_blob + ((size_t)q * bottom_blob.cstep + (size_t)b * bottom_blob.w * bottom_blob.h) * packn;
                    float* outptr = (float*)top_blob + (size_t)b * top_blob.nstep + (size_t)(q * packn) * top_blob.cstep;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e32m1(packn);

                        vfloat32m1_t _p = __riscv_vle32_v_f32m1(ptr, vl);
                        __riscv_vsse32_v_f32m1(outptr, top_blob.cstep * sizeof(float), _p, vl);

                        ptr += vl;
                        outptr++;
                    }
                }
            }
            if (scalar_elemsize == 2)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < batch * bottom_blob.c; bq++)
                {
                    const int b = bq / bottom_blob.c;
                    const int q = bq - b * bottom_blob.c;

                    const unsigned short* ptr = (const unsigned short*)bottom_blob + ((size_t)q * bottom_blob.cstep + (size_t)b * bottom_blob.w * bottom_blob.h) * packn;
                    unsigned short* outptr = (unsigned short*)top_blob + (size_t)b * top_blob.nstep + (size_t)(q * packn) * top_blob.cstep;

                    for (int i = 0; i < size; i++)
                    {
                        size_t vl = __riscv_vsetvl_e16m1(packn);

                        vuint16m1_t _p = __riscv_vle16_v_u16m1(ptr, vl);
                        __riscv_vsse16_v_u16m1(outptr, top_blob.cstep * sizeof(unsigned short), _p, vl);

                        ptr += vl;
                        outptr++;
                    }
                }
            }

            return 0;
        }
#endif // __riscv_vector

        size_t prefix = 1;
        for (int i = 0; i < batch_axis; i++)
            prefix *= shape[i];

        size_t suffix = 1;
        if (batch_axis == 0)
            suffix = out_total;
        else
        {
            for (int i = batch_axis; i < ndim; i++)
                suffix *= shape[i];
        }

        const size_t bottom_channel_size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d;
        const size_t top_channel_size = (size_t)top_blob.w * top_blob.h * top_blob.d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < (int)prefix; pp++)
        {
            for (int b = 0; b < batch; b++)
            {
                for (size_t s = 0; s < suffix; s++)
                {
                    const size_t srci = batch_axis == 0 ? (size_t)b * suffix + s : ((size_t)pp * batch + b) * suffix + s;
                    const unsigned char* ptr = (const unsigned char*)bottom_blob;
                    if (dims == 1)
                    {
                        const int x = srci / elempack;
                        const int k = srci % elempack;
                        ptr += (size_t)x * elemsize + k * scalar_elemsize;
                    }
                    else if (dims == 2)
                    {
                        const int x = srci % bottom_blob.w;
                        const int y = srci / bottom_blob.w;
                        const int y0 = y / elempack;
                        const int k = y % elempack;
                        ptr += ((size_t)y0 * bottom_blob.w + x) * elemsize + k * scalar_elemsize;
                    }
                    else
                    {
                        const int q = srci / bottom_channel_size;
                        const size_t r = srci - (size_t)q * bottom_channel_size;
                        const int q0 = q / elempack;
                        const int k = q % elempack;
                        ptr += ((size_t)q0 * bottom_blob.cstep + r) * elemsize + k * scalar_elemsize;
                    }

                    const size_t dsti = batch_axis == 0 ? s : (size_t)pp * suffix + s;
                    unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * top_blob.nstep * out_elemsize;
                    if (top_blob.dims == 1)
                    {
                        const int x = dsti / out_elempack;
                        const int k = dsti % out_elempack;
                        outptr += (size_t)x * out_elemsize + k * scalar_elemsize;
                    }
                    else if (top_blob.dims == 2)
                    {
                        const int x = dsti % top_blob.w;
                        const int y = dsti / top_blob.w;
                        const int y0 = y / out_elempack;
                        const int k = y % out_elempack;
                        outptr += ((size_t)y0 * top_blob.w + x) * out_elemsize + k * scalar_elemsize;
                    }
                    else
                    {
                        const int q = dsti / top_channel_size;
                        const size_t r = dsti - (size_t)q * top_channel_size;
                        const int q0 = q / out_elempack;
                        const int k = q % out_elempack;
                        outptr += ((size_t)q0 * top_blob.cstep + r) * out_elemsize + k * scalar_elemsize;
                    }

                    memcpy(outptr, ptr, scalar_elemsize);
                }
            }
        }

        return 0;
    }

    return -1;
}

int Reshape_riscv::forward_bf16s_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

    if (!shape_expr.empty())
    {
        int er = eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
        if (er != 0)
            return -1;
    }

    if (ndim == 1)
    {
        flatten(bottom_blob, top_blob, opt);
        if (top_blob.empty())
            return -100;

        return 0;
    }

    const int dims = bottom_blob.dims;
    const int elempack = bottom_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;

    const int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
#endif // __riscv_vector

    if (ndim == 2)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;

        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = outh % packn == 0 ? packn : 1;
        }
#endif // __riscv_vector
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && bottom_blob.h * elempack == outh && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (out_elempack == 1)
        {
            flatten(bottom_blob, top_blob, opt);
            if (top_blob.empty())
                return -100;

            top_blob.dims = 2;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.cstep = top_blob.cstep * top_blob.elempack;
            top_blob.elemsize = out_elemsize;
            top_blob.elempack = out_elempack;

            return 0;
        }

        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __riscv_vector
        if (out_elempack == packn)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob_flattened + outw * i * packn;
                unsigned short* outptr = top_blob.row<unsigned short>(i);

                for (int j = 0; j < outw; j++)
                {
                    size_t vl = __riscv_vsetvl_e16m1(packn);

                    vuint16m1_t _p = __riscv_vlse16_v_u16m1(ptr + j, outw * sizeof(unsigned short), vl);
                    __riscv_vse16_v_u16m1(outptr, _p, vl);

                    outptr += vl;
                }
            }
        }
#endif // __riscv_vector
    }

    if (ndim == 3 || ndim == 4)
    {
        if (ndim == 3)
        {
            if (outw == 0)
                outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
            if (outh == 0)
                outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
            if (outc == 0)
                outc = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

            if (outw == -1)
                outw = total / outc / outh;
            if (outh == -1)
                outh = total / outc / outw;
            if (outc == -1)
                outc = total / outh / outw;

            outd = 1;
        }
        else
        {
            if (outw == 0)
                outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
            if (outh == 0)
                outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
            if (outd == 0)
                outd = bottom_blob.d;
            if (outc == 0)
                outc = (dims == 3 || dims == 4) ? bottom_blob.c * elempack : bottom_blob.c;

            if (outw == -1)
                outw = total / outc / outd / outh;
            if (outh == -1)
                outh = total / outc / outd / outw;
            if (outd == -1)
                outd = total / outc / outh / outw;
            if (outc == -1)
                outc = total / outd / outh / outw;
        }

        int out_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            out_elempack = outc % packn == 0 ? packn : 1;
        }
#endif // __riscv_vector
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if ((dims == 3 || dims == 4) && bottom_blob.c * elempack == outc && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.dims = ndim;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.d = outd;
            return 0;
        }

        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        if (ndim == 3)
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        else
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);

        if (top_blob.empty())
            return -100;

        int size = top_blob.w * top_blob.h * top_blob.d;

#if __riscv_vector
        if (out_elempack == packn)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob_flattened + size * q * packn;
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    size_t vl = __riscv_vsetvl_e16m1(packn);

                    vuint16m1_t _p = __riscv_vlse16_v_u16m1(ptr + i, size * sizeof(unsigned short), vl);
                    __riscv_vse16_v_u16m1(outptr, _p, vl);

                    outptr += vl;
                }
            }
        }
#endif // __riscv_vector

        if (out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob_flattened + size * q;
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m8(n);

                    vuint16m8_t _p = __riscv_vle16_v_u16m8(ptr, vl);
                    __riscv_vse16_v_u16m8(outptr, _p, vl);

                    ptr += vl;
                    outptr += vl;
                    n -= vl;
                    i += vl;
                }
#endif // __riscv_vector
                for (; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
