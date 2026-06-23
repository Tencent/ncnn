// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "cpu.h"

#include <string.h>

namespace ncnn {

Reshape_arm::Reshape_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Reshape_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    if (batch_mode != 0)
    {
        if (batch_axis == 0)
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

        if (batch_mode == 2 && (outw == -1 || outh == -1 || outd == -1 || outc == -1))
            return -1;

        int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c;
        if (batch_mode == 1)
            total *= bottom_blob.n;

        if (ndim == 0)
            return -1;

        if (ndim == 1)
        {
            if (outw == 0)
                outw = bottom_blob.w;
            if (outw == -1)
                outw = total;
        }
        if (ndim == 2)
        {
            if (outw == 0)
                outw = bottom_blob.w;
            if (outh == 0)
                outh = bottom_blob.h;
            if (outw == -1)
                outw = total / outh;
            if (outh == -1)
                outh = total / outw;
        }
        if (ndim == 3)
        {
            if (outw == 0)
                outw = bottom_blob.w;
            if (outh == 0)
                outh = bottom_blob.h;
            if (outc == 0)
                outc = bottom_blob.c;
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
                outw = bottom_blob.w;
            if (outh == 0)
                outh = bottom_blob.h;
            if (outc == 0)
                outc = bottom_blob.c;
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
            if (bottom_blob.elempack != 1)
                return -1;

            size_t prefix = 1;
            for (int i = 0; i < batch_axis; i++)
                prefix *= shape[i];

            size_t suffix = 1;
            for (int i = batch_axis + 1; i < ndim; i++)
                suffix *= shape[i];

            if (ndim == 1)
                top_blob.create(outw, bottom_blob.elemsize, 1, opt.blob_allocator);
            if (ndim == 2)
                top_blob.create(outw, outh, bottom_blob.elemsize, 1, opt.blob_allocator);
            if (ndim == 3)
                top_blob.create(outw, outh, outc, bottom_blob.elemsize, 1, opt.blob_allocator);
            if (ndim == 4)
                top_blob.create(outw, outh, outd, outc, bottom_blob.elemsize, 1, opt.blob_allocator);

            if (top_blob.empty())
                return -100;

            const size_t bottom_channel_size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d;
            const size_t top_channel_size = (size_t)top_blob.w * top_blob.h * top_blob.d;
            const size_t suffix_bytes = suffix * bottom_blob.elemsize;
            for (size_t p = 0; p < prefix; p++)
            {
                const size_t srci = p * suffix;
                const int sq = srci / bottom_channel_size;
                const size_t sr = srci - (size_t)sq * bottom_channel_size;

                for (int b = 0; b < bottom_blob.n; b++)
                {
                    const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)sq * bottom_blob.cstep + sr) * bottom_blob.elemsize;

                    const size_t dsti = (p * bottom_blob.n + b) * suffix;
                    const int dq = dsti / top_channel_size;
                    const size_t dr = dsti - (size_t)dq * top_channel_size;
                    unsigned char* outptr = (unsigned char*)top_blob + ((size_t)dq * top_blob.cstep + dr) * bottom_blob.elemsize;

                    memcpy(outptr, ptr, suffix_bytes);
                }
            }

            return 0;
        }

        if (batch_mode == 2)
        {
            if (bottom_blob.n != 1 || bottom_blob.elempack != 1)
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

            const size_t bottom_total = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c;
            const int batch = bottom_total / out_total;
            if ((size_t)batch * out_total != bottom_total)
                return -1;

            size_t prefix = 1;
            for (int i = 0; i < batch_axis; i++)
                prefix *= shape[i];

            size_t suffix = 1;
            for (int i = batch_axis; i < ndim; i++)
                suffix *= shape[i];

            if (ndim == 1)
                top_blob.create_batch(outw, batch, bottom_blob.elemsize, 1, opt.blob_allocator);
            if (ndim == 2)
                top_blob.create_batch(outw, outh, batch, bottom_blob.elemsize, 1, opt.blob_allocator);
            if (ndim == 3)
                top_blob.create_batch(outw, outh, outc, batch, bottom_blob.elemsize, 1, opt.blob_allocator);
            if (ndim == 4)
                top_blob.create_batch(outw, outh, outd, outc, batch, bottom_blob.elemsize, 1, opt.blob_allocator);

            if (top_blob.empty())
                return -100;

            const size_t bottom_channel_size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d;
            const size_t top_channel_size = (size_t)top_blob.w * top_blob.h * top_blob.d;
            const size_t suffix_bytes = suffix * bottom_blob.elemsize;
            for (size_t p = 0; p < prefix; p++)
            {
                for (int b = 0; b < batch; b++)
                {
                    const size_t srci = (p * batch + b) * suffix;
                    const int sq = srci / bottom_channel_size;
                    const size_t sr = srci - (size_t)sq * bottom_channel_size;
                    const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)sq * bottom_blob.cstep + sr) * bottom_blob.elemsize;

                    const size_t dsti = p * suffix;
                    const int dq = dsti / top_channel_size;
                    const size_t dr = dsti - (size_t)dq * top_channel_size;
                    unsigned char* outptr = (unsigned char*)top_blob + ((size_t)b * top_blob.nstep + (size_t)dq * top_blob.cstep + dr) * bottom_blob.elemsize;

                    memcpy(outptr, ptr, suffix_bytes);
                }
            }

            return 0;
        }
    }

    int elembits = bottom_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blobs, top_blobs, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blobs, top_blobs, opt);
#endif

    // resolve out shape
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
        // flatten
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

        int out_elempack = opt.use_packing_layout && outh % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && bottom_blob.h * elempack == outh && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (out_elempack == 1)
        {
            // flatten
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

        // flatten
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

        // assert out_elempack == 4

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < top_blob.h; i++)
        {
            const float* ptr0 = (const float*)bottom_blob_flattened + outw * i * 4;
            const float* ptr1 = (const float*)bottom_blob_flattened + outw * (i * 4 + 1);
            const float* ptr2 = (const float*)bottom_blob_flattened + outw * (i * 4 + 2);
            const float* ptr3 = (const float*)bottom_blob_flattened + outw * (i * 4 + 3);
            float* outptr = top_blob.row(i);

            int j = 0;
#if __ARM_NEON
            for (; j + 3 < outw; j += 4)
            {
                float32x4x4_t _v4;
                _v4.val[0] = vld1q_f32(ptr0);
                _v4.val[1] = vld1q_f32(ptr1);
                _v4.val[2] = vld1q_f32(ptr2);
                _v4.val[3] = vld1q_f32(ptr3);

                vst4q_f32(outptr, _v4);

                ptr0 += 4;
                ptr1 += 4;
                ptr2 += 4;
                ptr3 += 4;
                outptr += 16;
            }
#endif
            for (; j < outw; j++)
            {
                outptr[0] = *ptr0++;
                outptr[1] = *ptr1++;
                outptr[2] = *ptr2++;
                outptr[3] = *ptr3++;

                outptr += 4;
            }
        }
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

        int out_elempack = opt.use_packing_layout && outc % 4 == 0 ? 4 : 1;
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

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        if (ndim == 3)
        {
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        }
        else // if (ndim == 4)
        {
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        }
        if (top_blob.empty())
            return -100;

        int size = top_blob.w * top_blob.h * top_blob.d;

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
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4x4_t _v4;
                    _v4.val[0] = vld1q_f32(ptr0);
                    _v4.val[1] = vld1q_f32(ptr1);
                    _v4.val[2] = vld1q_f32(ptr2);
                    _v4.val[3] = vld1q_f32(ptr3);

                    vst4q_f32(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
#endif
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }
        }

        if (out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob_flattened + size * q;
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _v = vld1q_f32(ptr);
                    vst1q_f32(outptr, _v);
                    ptr += 4;
                    outptr += 4;
                }
#endif
                for (; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }
        }
    }

    return 0;
}

int Reshape_arm::forward_bf16s_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (batch_mode != 0)
        return forward(bottom_blobs, top_blobs, opt);

    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    // resolve out shape
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
        // flatten
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
        if (opt.use_packing_layout)
        {
#if NCNN_ARM82
            out_elempack = support_fp16_storage && opt.use_fp16_arithmetic && outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
            out_elempack = outh % 4 == 0 ? 4 : 1;
#endif
        }
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && bottom_blob.h * elempack == outh && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (out_elempack == 1)
        {
            // flatten
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

        // flatten
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

#if NCNN_ARM82
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

                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    uint16x8_t _p01 = vcombine_u16(vld1_u16(ptr0), vld1_u16(ptr1));
                    uint16x8_t _p23 = vcombine_u16(vld1_u16(ptr2), vld1_u16(ptr3));
                    uint16x8_t _p45 = vcombine_u16(vld1_u16(ptr4), vld1_u16(ptr5));
                    uint16x8_t _p67 = vcombine_u16(vld1_u16(ptr6), vld1_u16(ptr7));

                    uint16x8x2_t _p0415 = vzipq_u16(_p01, _p45);
                    uint16x8x2_t _p2637 = vzipq_u16(_p23, _p67);

                    uint16x8x4_t _v4;
                    _v4.val[0] = _p0415.val[0];
                    _v4.val[1] = _p0415.val[1];
                    _v4.val[2] = _p2637.val[0];
                    _v4.val[3] = _p2637.val[1];

                    vst4q_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    ptr4 += 4;
                    ptr5 += 4;
                    ptr6 += 4;
                    ptr7 += 4;
                    outptr += 32;
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
        }
#endif // NCNN_ARM82

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

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < outw; j += 4)
                {
                    uint16x4x4_t _v4;
                    _v4.val[0] = vld1_u16(ptr0);
                    _v4.val[1] = vld1_u16(ptr1);
                    _v4.val[2] = vld1_u16(ptr2);
                    _v4.val[3] = vld1_u16(ptr3);

                    vst4_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
#endif
                for (; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }
        }
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
        if (opt.use_packing_layout)
        {
#if NCNN_ARM82
            out_elempack = support_fp16_storage && opt.use_fp16_arithmetic && outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#else
            out_elempack = outc % 4 == 0 ? 4 : 1;
#endif
        }
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

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        if (ndim == 3)
        {
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        }
        else // if (ndim == 4)
        {
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        }
        if (top_blob.empty())
            return -100;

        int size = top_blob.w * top_blob.h * top_blob.d;

#if NCNN_ARM82
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

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    uint16x8_t _p01 = vcombine_u16(vld1_u16(ptr0), vld1_u16(ptr1));
                    uint16x8_t _p23 = vcombine_u16(vld1_u16(ptr2), vld1_u16(ptr3));
                    uint16x8_t _p45 = vcombine_u16(vld1_u16(ptr4), vld1_u16(ptr5));
                    uint16x8_t _p67 = vcombine_u16(vld1_u16(ptr6), vld1_u16(ptr7));

                    uint16x8x2_t _p0415 = vzipq_u16(_p01, _p45);
                    uint16x8x2_t _p2637 = vzipq_u16(_p23, _p67);

                    uint16x8x4_t _v4;
                    _v4.val[0] = _p0415.val[0];
                    _v4.val[1] = _p0415.val[1];
                    _v4.val[2] = _p2637.val[0];
                    _v4.val[3] = _p2637.val[1];

                    vst4q_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    ptr4 += 4;
                    ptr5 += 4;
                    ptr6 += 4;
                    ptr7 += 4;
                    outptr += 32;
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
        }
#endif // NCNN_ARM82

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

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    uint16x4x4_t _v4;
                    _v4.val[0] = vld1_u16(ptr0);
                    _v4.val[1] = vld1_u16(ptr1);
                    _v4.val[2] = vld1_u16(ptr2);
                    _v4.val[3] = vld1_u16(ptr3);

                    vst4_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
#endif
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }
        }

        if (out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob_flattened + size * q;
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    uint16x4_t _v = vld1_u16(ptr);
                    vst1_u16(outptr, _v);
                    ptr += 4;
                    outptr += 4;
                }
#endif
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
