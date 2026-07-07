// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

#include "cpu.h"
#include "expression.h"

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

int Reshape_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

#if NCNN_BATCH
    if (input_batch_axis != 233 || output_batch_axis != 233)
        return forward_batch(bottom_blobs, top_blobs, opt);
#endif

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

int Reshape_riscv::forward_bf16s_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_BATCH
    if (input_batch_axis != 233 || output_batch_axis != 233)
        return Reshape::forward(bottom_blobs, top_blobs, opt);
#endif

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

#if NCNN_BATCH
int Reshape_riscv::forward_batch(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    Mat input_shape;
    Mat output_shape;
    int input_axis = 233;
    int output_axis = 233;
    size_t input_total = 0;
    if (resolve_batch_shape(bottom_blobs, input_shape, output_shape, input_axis, output_axis, input_total) != 0)
        return -1;

    const size_t scalar_elemsize = bottom_blob.elemsize / bottom_blob.elempack;

    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / (int)scalar_elemsize;
        out_elempack = (output_shape.dims == 1 ? output_shape.w : output_shape.dims == 2 ? output_shape.h : output_shape.c) % packn == 0 ? packn : 1;
    }
#endif // __riscv_vector

    const size_t out_elemsize = scalar_elemsize * out_elempack;

    bool reshape_zero_copy = input_axis == output_axis && output_shape.n == bottom_blob.n && out_elempack == bottom_blob.elempack;
    if (reshape_zero_copy && bottom_blob.elempack != 1)
    {
        const int pack_axis_size = bottom_blob.dims == 1 ? bottom_blob.w * bottom_blob.elempack : bottom_blob.dims == 2 ? bottom_blob.h * bottom_blob.elempack : bottom_blob.c * bottom_blob.elempack;
        const int out_pack_axis_size = output_shape.dims == 1 ? output_shape.w : output_shape.dims == 2 ? output_shape.h : output_shape.c;
        reshape_zero_copy = pack_axis_size == out_pack_axis_size;
    }

    if (reshape_zero_copy)
    {
        if (output_shape.dims == 1)
            top_blob = bottom_blob.reshape(output_shape.w / out_elempack, opt.blob_allocator);
        if (output_shape.dims == 2)
            top_blob = bottom_blob.reshape(output_shape.w, output_shape.h / out_elempack, opt.blob_allocator);
        if (output_shape.dims == 3)
            top_blob = bottom_blob.reshape(output_shape.w, output_shape.h, output_shape.c / out_elempack, opt.blob_allocator);
        if (output_shape.dims == 4)
            top_blob = bottom_blob.reshape(output_shape.w, output_shape.h, output_shape.d, output_shape.c / out_elempack, opt.blob_allocator);

        if (top_blob.empty())
            return -100;

        return 0;
    }

    if (output_shape.dims == 1)
        top_blob.create(output_shape.w / out_elempack, out_elemsize, out_elempack, output_shape.n, opt.blob_allocator);
    if (output_shape.dims == 2)
        top_blob.create(output_shape.w, output_shape.h / out_elempack, out_elemsize, out_elempack, output_shape.n, opt.blob_allocator);
    if (output_shape.dims == 3)
        top_blob.create(output_shape.w, output_shape.h, output_shape.c / out_elempack, out_elemsize, out_elempack, output_shape.n, opt.blob_allocator);
    if (output_shape.dims == 4)
        top_blob.create(output_shape.w, output_shape.h, output_shape.d, output_shape.c / out_elempack, out_elemsize, out_elempack, output_shape.n, opt.blob_allocator);

    if (top_blob.empty())
        return -100;

    if (out_elempack == bottom_blob.elempack)
    {
        if (output_shape.dims == bottom_blob.dims)
        {
            if (input_axis == 0 && output_axis == 233)
            {
                if (bottom_blob.dims == 1 && top_blob.w == bottom_blob.w * bottom_blob.n)
                {
                    const size_t size = (size_t)bottom_blob.w * bottom_blob.elemsize;
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int b = 0; b < bottom_blob.n; b++)
                    {
                        const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * bottom_blob.nstep * bottom_blob.elemsize;
                        unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * bottom_blob.w * top_blob.elemsize;
                        memcpy(outptr, ptr, size);
                    }
                    return 0;
                }
                if (bottom_blob.dims == 2 && top_blob.w == bottom_blob.w && top_blob.h == bottom_blob.h * bottom_blob.n)
                {
                    const size_t size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.elemsize;
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int b = 0; b < bottom_blob.n; b++)
                    {
                        const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * bottom_blob.nstep * bottom_blob.elemsize;
                        unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * bottom_blob.w * bottom_blob.h * top_blob.elemsize;
                        memcpy(outptr, ptr, size);
                    }
                    return 0;
                }
                if ((bottom_blob.dims == 3 || bottom_blob.dims == 4) && top_blob.w == bottom_blob.w && top_blob.h == bottom_blob.h && top_blob.d == bottom_blob.d && top_blob.c == bottom_blob.c * bottom_blob.n)
                {
                    const size_t size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.elemsize;
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int bq = 0; bq < bottom_blob.n * bottom_blob.c; bq++)
                    {
                        const int b = bq / bottom_blob.c;
                        const int q = bq - b * bottom_blob.c;
                        const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)q * bottom_blob.cstep) * bottom_blob.elemsize;
                        unsigned char* outptr = (unsigned char*)top_blob + (size_t)bq * top_blob.cstep * top_blob.elemsize;
                        memcpy(outptr, ptr, size);
                    }
                    return 0;
                }
            }
            if (input_axis == 233 && output_axis == 0)
            {
                if (bottom_blob.dims == 1 && bottom_blob.w == top_blob.w * top_blob.n)
                {
                    const size_t size = (size_t)top_blob.w * top_blob.elemsize;
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int b = 0; b < top_blob.n; b++)
                    {
                        const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * top_blob.w * bottom_blob.elemsize;
                        unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * top_blob.nstep * top_blob.elemsize;
                        memcpy(outptr, ptr, size);
                    }
                    return 0;
                }
                if (bottom_blob.dims == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h * top_blob.n)
                {
                    const size_t size = (size_t)top_blob.w * top_blob.h * top_blob.elemsize;
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int b = 0; b < top_blob.n; b++)
                    {
                        const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * top_blob.w * top_blob.h * bottom_blob.elemsize;
                        unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * top_blob.nstep * top_blob.elemsize;
                        memcpy(outptr, ptr, size);
                    }
                    return 0;
                }
                if ((bottom_blob.dims == 3 || bottom_blob.dims == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c == top_blob.c * top_blob.n)
                {
                    const size_t size = (size_t)top_blob.w * top_blob.h * top_blob.d * top_blob.elemsize;
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int bq = 0; bq < top_blob.n * top_blob.c; bq++)
                    {
                        const int b = bq / top_blob.c;
                        const int q = bq - b * top_blob.c;
                        const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)(b * top_blob.c + q) * bottom_blob.cstep * bottom_blob.elemsize;
                        unsigned char* outptr = (unsigned char*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * top_blob.elemsize;
                        memcpy(outptr, ptr, size);
                    }
                    return 0;
                }
            }
        }
        if (input_axis == 1 && output_axis == 233)
        {
            if (bottom_blob.dims == 2 && top_blob.dims == 3 && top_blob.w == bottom_blob.w && top_blob.h == bottom_blob.n && top_blob.c == bottom_blob.h)
            {
                const size_t size = (size_t)bottom_blob.w * bottom_blob.elemsize;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < bottom_blob.n * bottom_blob.h; bq++)
                {
                    const int b = bq / bottom_blob.h;
                    const int q = bq - b * bottom_blob.h;
                    const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)q * bottom_blob.w) * bottom_blob.elemsize;
                    unsigned char* outptr = (unsigned char*)top_blob + ((size_t)q * top_blob.cstep + (size_t)b * top_blob.w) * top_blob.elemsize;
                    memcpy(outptr, ptr, size);
                }
                return 0;
            }
            if (bottom_blob.dims == 3 && top_blob.dims == 4 && top_blob.w == bottom_blob.w && top_blob.h == bottom_blob.h && top_blob.d == bottom_blob.n && top_blob.c == bottom_blob.c)
            {
                const size_t size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.elemsize;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < bottom_blob.n * bottom_blob.c; bq++)
                {
                    const int b = bq / bottom_blob.c;
                    const int q = bq - b * bottom_blob.c;
                    const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)q * bottom_blob.cstep) * bottom_blob.elemsize;
                    unsigned char* outptr = (unsigned char*)top_blob + ((size_t)q * top_blob.cstep + (size_t)b * top_blob.w * top_blob.h) * top_blob.elemsize;
                    memcpy(outptr, ptr, size);
                }
                return 0;
            }
        }
        if (input_axis == 233 && output_axis == 1)
        {
            if (bottom_blob.dims == 3 && top_blob.dims == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.n && bottom_blob.c == top_blob.h)
            {
                const size_t size = (size_t)top_blob.w * top_blob.elemsize;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < top_blob.n * top_blob.h; bq++)
                {
                    const int b = bq / top_blob.h;
                    const int q = bq - b * top_blob.h;
                    const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)q * bottom_blob.cstep + (size_t)b * bottom_blob.w) * bottom_blob.elemsize;
                    unsigned char* outptr = (unsigned char*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.w) * top_blob.elemsize;
                    memcpy(outptr, ptr, size);
                }
                return 0;
            }
            if (bottom_blob.dims == 4 && top_blob.dims == 3 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.n && bottom_blob.c == top_blob.c)
            {
                const size_t size = (size_t)top_blob.w * top_blob.h * top_blob.elemsize;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int bq = 0; bq < top_blob.n * top_blob.c; bq++)
                {
                    const int b = bq / top_blob.c;
                    const int q = bq - b * top_blob.c;
                    const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)q * bottom_blob.cstep + (size_t)b * bottom_blob.w * bottom_blob.h) * bottom_blob.elemsize;
                    unsigned char* outptr = (unsigned char*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * top_blob.elemsize;
                    memcpy(outptr, ptr, size);
                }
                return 0;
            }
        }
    }

    copy_batch_reshape(bottom_blob, top_blob, input_shape, input_axis, output_shape, output_axis, input_total, scalar_elemsize, opt);

    return 0;
}
#endif // NCNN_BATCH

} // namespace ncnn
