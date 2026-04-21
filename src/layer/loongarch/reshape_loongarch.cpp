// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape_loongarch.h"

#if __loongarch_sx
#include "loongarch_usability.h"
#endif // __loongarch_sx

namespace ncnn {

Reshape_loongarch::Reshape_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Reshape_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
#if __loongarch_sx
        if (opt.use_packing_layout)
        {
#if __loongarch_asx
            out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
            out_elempack = outh % 4 == 0 ? 4 : 1;
#endif // __loongarch_asx
        }
#endif // __loongarch_sx
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

#if __loongarch_asx
        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + outw * i * 8;
                const float* ptr1 = (const float*)bottom_blob_flattened + outw * (i * 8 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + outw * (i * 8 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + outw * (i * 8 + 3);
                const float* ptr4 = (const float*)bottom_blob_flattened + outw * (i * 8 + 4);
                const float* ptr5 = (const float*)bottom_blob_flattened + outw * (i * 8 + 5);
                const float* ptr6 = (const float*)bottom_blob_flattened + outw * (i * 8 + 6);
                const float* ptr7 = (const float*)bottom_blob_flattened + outw * (i * 8 + 7);
                float* outptr = top_blob.row(i);

                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    __m256 _row0 = (__m256)__lasx_xvld(ptr0, 0);
                    __m256 _row1 = (__m256)__lasx_xvld(ptr1, 0);
                    __m256 _row2 = (__m256)__lasx_xvld(ptr2, 0);
                    __m256 _row3 = (__m256)__lasx_xvld(ptr3, 0);
                    __m256 _row4 = (__m256)__lasx_xvld(ptr4, 0);
                    __m256 _row5 = (__m256)__lasx_xvld(ptr5, 0);
                    __m256 _row6 = (__m256)__lasx_xvld(ptr6, 0);
                    __m256 _row7 = (__m256)__lasx_xvld(ptr7, 0);

                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

                    __lasx_xvst((__m256i)_row0, outptr, 0);
                    __lasx_xvst((__m256i)_row1, outptr + 8, 0);
                    __lasx_xvst((__m256i)_row2, outptr + 8 * 2, 0);
                    __lasx_xvst((__m256i)_row3, outptr + 8 * 3, 0);
                    __lasx_xvst((__m256i)_row4, outptr + 8 * 4, 0);
                    __lasx_xvst((__m256i)_row5, outptr + 8 * 5, 0);
                    __lasx_xvst((__m256i)_row6, outptr + 8 * 6, 0);
                    __lasx_xvst((__m256i)_row7, outptr + 8 * 7, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    outptr += 64;
                }
                for (; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }
#endif // __loongarch_asx

#if __loongarch_sx
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
                    __m128 _row0 = (__m128)__lsx_vld(ptr0, 0);
                    __m128 _row1 = (__m128)__lsx_vld(ptr1, 0);
                    __m128 _row2 = (__m128)__lsx_vld(ptr2, 0);
                    __m128 _row3 = (__m128)__lsx_vld(ptr3, 0);

                    __m128i _row01r = __lsx_vilvl_w((__m128i)_row1, (__m128i)_row0);
                    __m128i _row01l = __lsx_vilvh_w((__m128i)_row1, (__m128i)_row0);
                    __m128i _row23r = __lsx_vilvl_w((__m128i)_row3, (__m128i)_row2);
                    __m128i _row23l = __lsx_vilvh_w((__m128i)_row3, (__m128i)_row2);
                    __m128i _row0123_0 = __lsx_vilvl_d(_row23r, _row01r);
                    __m128i _row0123_1 = __lsx_vilvh_d(_row23r, _row01r);
                    __m128i _row0123_2 = __lsx_vilvl_d(_row23l, _row01l);
                    __m128i _row0123_3 = __lsx_vilvh_d(_row23l, _row01l);

                    __lsx_vst(_row0123_0, outptr, 0);
                    __lsx_vst(_row0123_1, outptr + 4, 0);
                    __lsx_vst(_row0123_2, outptr + 4 * 2, 0);
                    __lsx_vst(_row0123_3, outptr + 4 * 3, 0);

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
#endif // __loongarch_sx
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
        else // if (ndim == 4)
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
#if __loongarch_sx
        if (opt.use_packing_layout)
        {
#if __loongarch_asx
            out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#else
            out_elempack = outc % 4 == 0 ? 4 : 1;
#endif // __loongarch_asx
        }
#endif // __loongarch_sx
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

#if __loongarch_asx
        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + size * q * 8;
                const float* ptr1 = (const float*)bottom_blob_flattened + size * (q * 8 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + size * (q * 8 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + size * (q * 8 + 3);
                const float* ptr4 = (const float*)bottom_blob_flattened + size * (q * 8 + 4);
                const float* ptr5 = (const float*)bottom_blob_flattened + size * (q * 8 + 5);
                const float* ptr6 = (const float*)bottom_blob_flattened + size * (q * 8 + 6);
                const float* ptr7 = (const float*)bottom_blob_flattened + size * (q * 8 + 7);
                float* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 7 < size; i += 8)
                {
                    __m256 _row0 = (__m256)__lasx_xvld(ptr0, 0);
                    __m256 _row1 = (__m256)__lasx_xvld(ptr1, 0);
                    __m256 _row2 = (__m256)__lasx_xvld(ptr2, 0);
                    __m256 _row3 = (__m256)__lasx_xvld(ptr3, 0);
                    __m256 _row4 = (__m256)__lasx_xvld(ptr4, 0);
                    __m256 _row5 = (__m256)__lasx_xvld(ptr5, 0);
                    __m256 _row6 = (__m256)__lasx_xvld(ptr6, 0);
                    __m256 _row7 = (__m256)__lasx_xvld(ptr7, 0);

                    transpose8x8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);

                    __lasx_xvst((__m256i)_row0, outptr, 0);
                    __lasx_xvst((__m256i)_row1, outptr + 8, 0);
                    __lasx_xvst((__m256i)_row2, outptr + 8 * 2, 0);
                    __lasx_xvst((__m256i)_row3, outptr + 8 * 3, 0);
                    __lasx_xvst((__m256i)_row4, outptr + 8 * 4, 0);
                    __lasx_xvst((__m256i)_row5, outptr + 8 * 5, 0);
                    __lasx_xvst((__m256i)_row6, outptr + 8 * 6, 0);
                    __lasx_xvst((__m256i)_row7, outptr + 8 * 7, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    outptr += 64;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }
#endif // __loongarch_asx

#if __loongarch_sx
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
                for (; i + 3 < size; i += 4)
                {
                    __m128 _row0 = (__m128)__lsx_vld(ptr0, 0);
                    __m128 _row1 = (__m128)__lsx_vld(ptr1, 0);
                    __m128 _row2 = (__m128)__lsx_vld(ptr2, 0);
                    __m128 _row3 = (__m128)__lsx_vld(ptr3, 0);

                    __m128i _row01r = __lsx_vilvl_w((__m128i)_row1, (__m128i)_row0);
                    __m128i _row01l = __lsx_vilvh_w((__m128i)_row1, (__m128i)_row0);
                    __m128i _row23r = __lsx_vilvl_w((__m128i)_row3, (__m128i)_row2);
                    __m128i _row23l = __lsx_vilvh_w((__m128i)_row3, (__m128i)_row2);
                    __m128i _row0123_0 = __lsx_vilvl_d(_row23r, _row01r);
                    __m128i _row0123_1 = __lsx_vilvh_d(_row23r, _row01r);
                    __m128i _row0123_2 = __lsx_vilvl_d(_row23l, _row01l);
                    __m128i _row0123_3 = __lsx_vilvh_d(_row23l, _row01l);

                    __lsx_vst(_row0123_0, outptr, 0);
                    __lsx_vst(_row0123_1, outptr + 4, 0);
                    __lsx_vst(_row0123_2, outptr + 4 * 2, 0);
                    __lsx_vst(_row0123_3, outptr + 4 * 3, 0);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
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
#endif // __loongarch_sx

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < top_blob.c; q++)
        {
            const float* ptr = (const float*)bottom_blob_flattened + size * q;
            float* outptr = top_blob.channel(q);

            int i = 0;
#if __loongarch_asx
            for (; i + 7 < size; i += 8)
            {
                __lasx_xvst(__lasx_xvld(ptr, 0), outptr, 0);
                ptr += 8;
                outptr += 8;
            }
#endif // __loongarch_asx
#if __loongarch_sx
            for (; i + 3 < size; i += 4)
            {
                __lsx_vst(__lsx_vld(ptr, 0), outptr, 0);
                ptr += 4;
                outptr += 4;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                *outptr++ = *ptr++;
            }
        }
    }

    return 0;
}

int Reshape_loongarch::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
#if __loongarch_sx
        if (opt.use_packing_layout)
        {
#if __loongarch_asx
            out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
            out_elempack = outh % 4 == 0 ? 4 : 1;
#endif // __loongarch_asx
        }
#endif // __loongarch_sx
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

#if __loongarch_asx
        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const unsigned short* ptr0 = (const unsigned short*)bottom_blob_flattened + outw * i * 8;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 1);
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 2);
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 3);
                const unsigned short* ptr4 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 4);
                const unsigned short* ptr5 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 5);
                const unsigned short* ptr6 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 6);
                const unsigned short* ptr7 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 7);
                unsigned short* outptr = top_blob.row<unsigned short>(i);

                for (int j = 0; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }
#endif // __loongarch_asx

#if __loongarch_sx
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
#endif // __loongarch_sx
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
        else // if (ndim == 4)
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
#if __loongarch_sx
        if (opt.use_packing_layout)
        {
#if __loongarch_asx
            out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#else
            out_elempack = outc % 4 == 0 ? 4 : 1;
#endif // __loongarch_asx
        }
#endif // __loongarch_sx
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

#if __loongarch_asx
        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr0 = (const unsigned short*)bottom_blob_flattened + size * q * 8;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 1);
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 2);
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 3);
                const unsigned short* ptr4 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 4);
                const unsigned short* ptr5 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 5);
                const unsigned short* ptr6 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 6);
                const unsigned short* ptr7 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 7);
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }
#endif // __loongarch_asx

#if __loongarch_sx
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
#endif // __loongarch_sx

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < top_blob.c; q++)
        {
            const unsigned short* ptr = (const unsigned short*)bottom_blob_flattened + size * q;
            unsigned short* outptr = top_blob.channel(q);

            memcpy(outptr, ptr, (size_t)size * sizeof(unsigned short));
        }
    }

    return 0;
}

} // namespace ncnn
