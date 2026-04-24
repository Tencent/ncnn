// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

namespace ncnn {

Reshape_mips::Reshape_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Reshape_mips::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    const int elembits = bottom_blob.elembits();
    if (elembits == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);

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
#if __mips_msa
        if (opt.use_packing_layout)
            out_elempack = outh % 4 == 0 ? 4 : 1;
#endif // __mips_msa
        const size_t out_elemsize = elemsize / elempack * out_elempack;

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

#if __mips_msa
        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + outw * i * 4;
                const float* ptr1 = (const float*)bottom_blob_flattened + outw * (i * 4 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + outw * (i * 4 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + outw * (i * 4 + 3);
                float* outptr = top_blob.row(i);

                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    v4f32 _row0 = (v4f32)__msa_ld_w(ptr0, 0);
                    v4f32 _row1 = (v4f32)__msa_ld_w(ptr1, 0);
                    v4f32 _row2 = (v4f32)__msa_ld_w(ptr2, 0);
                    v4f32 _row3 = (v4f32)__msa_ld_w(ptr3, 0);

                    v4i32 _row01r = __msa_ilvr_w((v4i32)_row1, (v4i32)_row0);
                    v4i32 _row01l = __msa_ilvl_w((v4i32)_row1, (v4i32)_row0);
                    v4i32 _row23r = __msa_ilvr_w((v4i32)_row3, (v4i32)_row2);
                    v4i32 _row23l = __msa_ilvl_w((v4i32)_row3, (v4i32)_row2);
                    v2i64 _row0123_0 = __msa_ilvr_d((v2i64)_row23r, (v2i64)_row01r);
                    v2i64 _row0123_1 = __msa_ilvl_d((v2i64)_row23r, (v2i64)_row01r);
                    v2i64 _row0123_2 = __msa_ilvr_d((v2i64)_row23l, (v2i64)_row01l);
                    v2i64 _row0123_3 = __msa_ilvl_d((v2i64)_row23l, (v2i64)_row01l);

                    __msa_st_w((v4i32)_row0123_0, outptr, 0);
                    __msa_st_w((v4i32)_row0123_1, outptr + 4, 0);
                    __msa_st_w((v4i32)_row0123_2, outptr + 4 * 2, 0);
                    __msa_st_w((v4i32)_row0123_3, outptr + 4 * 3, 0);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }
#endif // __mips_msa

        return 0;
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

        outd = 1;
    }

    if (ndim == 4)
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
#if __mips_msa
    if (opt.use_packing_layout)
        out_elempack = outc % 4 == 0 ? 4 : 1;
#endif // __mips_msa
    const size_t out_elemsize = elemsize / elempack * out_elempack;

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
    else // if (ndim == 4)
        top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int size = top_blob.w * top_blob.h * top_blob.d;

#if __mips_msa
    if (out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr0 = (const float*)bottom_blob_flattened + size * q * 4;
            const float* ptr1 = (const float*)bottom_blob_flattened + size * (q * 4 + 1);
            const float* ptr2 = (const float*)bottom_blob_flattened + size * (q * 4 + 2);
            const float* ptr3 = (const float*)bottom_blob_flattened + size * (q * 4 + 3);
            float* outptr = top_blob.channel(q);

            int i = 0;
#if __mips_msa
            for (; i + 3 < size; i += 4)
            {
                v4f32 _row0 = (v4f32)__msa_ld_w(ptr0, 0);
                v4f32 _row1 = (v4f32)__msa_ld_w(ptr1, 0);
                v4f32 _row2 = (v4f32)__msa_ld_w(ptr2, 0);
                v4f32 _row3 = (v4f32)__msa_ld_w(ptr3, 0);

                v4i32 _row01r = __msa_ilvr_w((v4i32)_row1, (v4i32)_row0);
                v4i32 _row01l = __msa_ilvl_w((v4i32)_row1, (v4i32)_row0);
                v4i32 _row23r = __msa_ilvr_w((v4i32)_row3, (v4i32)_row2);
                v4i32 _row23l = __msa_ilvl_w((v4i32)_row3, (v4i32)_row2);
                v2i64 _row0123_0 = __msa_ilvr_d((v2i64)_row23r, (v2i64)_row01r);
                v2i64 _row0123_1 = __msa_ilvl_d((v2i64)_row23r, (v2i64)_row01r);
                v2i64 _row0123_2 = __msa_ilvr_d((v2i64)_row23l, (v2i64)_row01l);
                v2i64 _row0123_3 = __msa_ilvl_d((v2i64)_row23l, (v2i64)_row01l);

                __msa_st_w((v4i32)_row0123_0, outptr, 0);
                __msa_st_w((v4i32)_row0123_1, outptr + 4, 0);
                __msa_st_w((v4i32)_row0123_2, outptr + 4 * 2, 0);
                __msa_st_w((v4i32)_row0123_3, outptr + 4 * 3, 0);

                ptr0 += 4;
                ptr1 += 4;
                ptr2 += 4;
                ptr3 += 4;
                outptr += 16;
            }
#endif // __mips_msa
            for (; i < size; i++)
            {
                outptr[0] = *ptr0++;
                outptr[1] = *ptr1++;
                outptr[2] = *ptr2++;
                outptr[3] = *ptr3++;

                outptr += 4;
            }
        }

        return 0;
    }
#endif // __mips_msa

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        const float* ptr = (const float*)bottom_blob_flattened + size * q;
        float* outptr = top_blob.channel(q);

        int i = 0;
#if __mips_msa
        for (; i + 3 < size; i += 4)
        {
            __msa_st_w(__msa_ld_w(ptr, 0), outptr, 0);
            ptr += 4;
            outptr += 4;
        }
#endif // __mips_msa
        for (; i < size; i++)
        {
            *outptr++ = *ptr++;
        }
    }

    return 0;
}

int Reshape_mips::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
#if __mips_msa
        if (opt.use_packing_layout)
            out_elempack = outh % 4 == 0 ? 4 : 1;
#endif // __mips_msa
        const size_t out_elemsize = elemsize / elempack * out_elempack;

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

#if __mips_msa
        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const unsigned short* ptr0 = (const unsigned short*)bottom_blob_flattened + outw * i * 4;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob_flattened + outw * (i * 4 + 1);
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob_flattened + outw * (i * 4 + 2);
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob_flattened + outw * (i * 4 + 3);
                unsigned short* outptr = top_blob.row<unsigned short>(i);

                for (int j = 0; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }
#endif // __mips_msa

        return 0;
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

        outd = 1;
    }

    if (ndim == 4)
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
#if __mips_msa
    if (opt.use_packing_layout)
        out_elempack = outc % 4 == 0 ? 4 : 1;
#endif // __mips_msa
    const size_t out_elemsize = elemsize / elempack * out_elempack;

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
    else // if (ndim == 4)
        top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int size = top_blob.w * top_blob.h * top_blob.d;

#if __mips_msa
    if (out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr0 = (const unsigned short*)bottom_blob_flattened + size * q * 4;
            const unsigned short* ptr1 = (const unsigned short*)bottom_blob_flattened + size * (q * 4 + 1);
            const unsigned short* ptr2 = (const unsigned short*)bottom_blob_flattened + size * (q * 4 + 2);
            const unsigned short* ptr3 = (const unsigned short*)bottom_blob_flattened + size * (q * 4 + 3);
            unsigned short* outptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                outptr[0] = *ptr0++;
                outptr[1] = *ptr1++;
                outptr[2] = *ptr2++;
                outptr[3] = *ptr3++;

                outptr += 4;
            }
        }

        return 0;
    }
#endif // __mips_msa

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < top_blob.c; q++)
    {
        const unsigned short* ptr = (const unsigned short*)bottom_blob_flattened + size * q;
        unsigned short* outptr = top_blob.channel(q);

        memcpy(outptr, ptr, (size_t)size * sizeof(unsigned short));
    }

    return 0;
}

} // namespace ncnn
