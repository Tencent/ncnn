// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "shufflechannel_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

ShuffleChannel_loongarch::ShuffleChannel_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int ShuffleChannel_loongarch::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_BF16
    int elembits = bottom_blob.elembits();
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

    int logical_channels = channels * elempack;
    if (logical_channels % group != 0)
        return -100;

    int _group = reverse ? logical_channels / group : group;

    if (_group == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    if (elempack == 1)
        return ShuffleChannel::forward(bottom_blob, top_blob, opt);

#if __loongarch_sx
#if __loongarch_asx
    if (elempack == 8)
    {
        int channels_per_group = channels / _group;

        if (_group == 2 && channels % _group == 0)
        {
            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr0 = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                float* outptr0 = top_blob.channel(q * 2);
                float* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    __m256 _p0 = (__m256)__lasx_xvld(ptr0, 0);
                    __m256 _p1 = (__m256)__lasx_xvld(ptr1, 0);

                    transpose8x2_ps(_p0, _p1);

                    __lasx_xvst((__m256i)_p0, outptr0, 0);
                    __lasx_xvst((__m256i)_p1, outptr1, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                }
            }

            return 0;
        }

        if (_group == 4 && channels % _group == 0)
        {
            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr0 = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                const float* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                const float* ptr3 = bottom_blob.channel(channels_per_group * 3 + q);
                float* outptr0 = top_blob.channel(q * 4);
                float* outptr1 = top_blob.channel(q * 4 + 1);
                float* outptr2 = top_blob.channel(q * 4 + 2);
                float* outptr3 = top_blob.channel(q * 4 + 3);

                for (int i = 0; i < size; i++)
                {
                    __m256 _p0 = (__m256)__lasx_xvld(ptr0, 0);
                    __m256 _p1 = (__m256)__lasx_xvld(ptr1, 0);
                    __m256 _p2 = (__m256)__lasx_xvld(ptr2, 0);
                    __m256 _p3 = (__m256)__lasx_xvld(ptr3, 0);

                    transpose8x4_ps(_p0, _p1, _p2, _p3);

                    __lasx_xvst((__m256i)_p0, outptr0, 0);
                    __lasx_xvst((__m256i)_p1, outptr1, 0);
                    __lasx_xvst((__m256i)_p2, outptr2, 0);
                    __lasx_xvst((__m256i)_p3, outptr3, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                    outptr3 += 8;
                }
            }

            return 0;
        }

        if (_group == 8 && channels % _group == 0)
        {
            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr0 = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                const float* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                const float* ptr3 = bottom_blob.channel(channels_per_group * 3 + q);
                const float* ptr4 = bottom_blob.channel(channels_per_group * 4 + q);
                const float* ptr5 = bottom_blob.channel(channels_per_group * 5 + q);
                const float* ptr6 = bottom_blob.channel(channels_per_group * 6 + q);
                const float* ptr7 = bottom_blob.channel(channels_per_group * 7 + q);
                float* outptr0 = top_blob.channel(q * 8);
                float* outptr1 = top_blob.channel(q * 8 + 1);
                float* outptr2 = top_blob.channel(q * 8 + 2);
                float* outptr3 = top_blob.channel(q * 8 + 3);
                float* outptr4 = top_blob.channel(q * 8 + 4);
                float* outptr5 = top_blob.channel(q * 8 + 5);
                float* outptr6 = top_blob.channel(q * 8 + 6);
                float* outptr7 = top_blob.channel(q * 8 + 7);

                for (int i = 0; i < size; i++)
                {
                    __m256 _p0 = (__m256)__lasx_xvld(ptr0, 0);
                    __m256 _p1 = (__m256)__lasx_xvld(ptr1, 0);
                    __m256 _p2 = (__m256)__lasx_xvld(ptr2, 0);
                    __m256 _p3 = (__m256)__lasx_xvld(ptr3, 0);
                    __m256 _p4 = (__m256)__lasx_xvld(ptr4, 0);
                    __m256 _p5 = (__m256)__lasx_xvld(ptr5, 0);
                    __m256 _p6 = (__m256)__lasx_xvld(ptr6, 0);
                    __m256 _p7 = (__m256)__lasx_xvld(ptr7, 0);

                    transpose8x8_ps(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);

                    __lasx_xvst((__m256i)_p0, outptr0, 0);
                    __lasx_xvst((__m256i)_p1, outptr1, 0);
                    __lasx_xvst((__m256i)_p2, outptr2, 0);
                    __lasx_xvst((__m256i)_p3, outptr3, 0);
                    __lasx_xvst((__m256i)_p4, outptr4, 0);
                    __lasx_xvst((__m256i)_p5, outptr5, 0);
                    __lasx_xvst((__m256i)_p6, outptr6, 0);
                    __lasx_xvst((__m256i)_p7, outptr7, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                    outptr3 += 8;
                    outptr4 += 8;
                    outptr5 += 8;
                    outptr6 += 8;
                    outptr7 += 8;
                }
            }

            return 0;
        }
    }
#endif // __loongarch_asx

    if (elempack == 4)
    {
        int channels_per_group = channels / _group;

        if (_group == 2 && channels % _group != 0)
        {
            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            __m128i _mask_2301 = __lsx_vreplgr2vr_w(2);
            _mask_2301 = __lsx_vinsgr2vr_w(_mask_2301, 3, 1);
            _mask_2301 = __lsx_vinsgr2vr_w(_mask_2301, 4, 2);
            _mask_2301 = __lsx_vinsgr2vr_w(_mask_2301, 5, 3);

            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr0 = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                const float* ptr2 = bottom_blob.channel(channels_per_group + q + 1);
                float* outptr0 = top_blob.channel(q * 2);
                float* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    __m128 _p0 = (__m128)__lsx_vld(ptr0, 0);
                    __m128 _p1 = (__m128)__lsx_vld(ptr1, 0);
                    __m128 _p2 = (__m128)__lsx_vld(ptr2, 0);

                    __m128 _p12 = (__m128)__lsx_vshuf_w(_mask_2301, (__m128i)_p2, (__m128i)_p1);

                    __m128 _lo = (__m128)__lsx_vilvr_w((__m128i)_p12, (__m128i)_p0);
                    __m128 _hi = (__m128)__lsx_vilvh_w((__m128i)_p12, (__m128i)_p0);

                    __lsx_vst(_lo, outptr0, 0);
                    __lsx_vst(_hi, outptr1, 0);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }

            // handle the last channel
            {
                const float* ptr0 = bottom_blob.channel(channels_per_group);
                const float* ptr1 = bottom_blob.channel(channels_per_group * 2);
                float* outptr = top_blob.channel(channels_per_group * 2);

                ptr1 += 2;

                for (int i = 0; i < size; i++)
                {
                    __m128 _p0 = (__m128)__lsx_vld(ptr0, 0);
                    __m128 _p1 = (__m128)__lsx_vldrepl_d((void*)ptr1, 0);

                    __m128 _lo = (__m128)__lsx_vilvr_w((__m128i)_p1, (__m128i)_p0);

                    __lsx_vst(_lo, outptr, 0);

                    ptr0 += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            return 0;
        }

        if (_group <= 4 && channels % _group == 0)
        {
            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (_group == 2)
            {
                for (int q = 0; q < channels_per_group; q++)
                {
                    const float* ptr0 = bottom_blob.channel(q);
                    const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                    float* outptr0 = top_blob.channel(q * 2);
                    float* outptr1 = top_blob.channel(q * 2 + 1);

                    for (int i = 0; i < size; i++)
                    {
                        __m128 _p0 = (__m128)__lsx_vld(ptr0, 0);
                        __m128 _p1 = (__m128)__lsx_vld(ptr1, 0);

                        __m128 _lo = (__m128)__lsx_vilvr_w((__m128i)_p1, (__m128i)_p0);
                        __m128 _hi = (__m128)__lsx_vilvh_w((__m128i)_p1, (__m128i)_p0);

                        __lsx_vst(_lo, outptr0, 0);
                        __lsx_vst(_hi, outptr1, 0);

                        ptr0 += 4;
                        ptr1 += 4;
                        outptr0 += 4;
                        outptr1 += 4;
                    }
                }

                return 0;
            }

            if (_group == 3)
            {
                __m128i _mask_0481 = __lsx_vreplgr2vr_w(0);
                _mask_0481 = __lsx_vinsgr2vr_w(_mask_0481, 1, 1);
                _mask_0481 = __lsx_vinsgr2vr_w(_mask_0481, 4, 2);
                _mask_0481 = __lsx_vinsgr2vr_w(_mask_0481, 2, 3);

                __m128i _mask_5926 = __lsx_vreplgr2vr_w(2);
                _mask_5926 = __lsx_vinsgr2vr_w(_mask_5926, 3, 1);
                _mask_5926 = __lsx_vinsgr2vr_w(_mask_5926, 4, 2);
                _mask_5926 = __lsx_vinsgr2vr_w(_mask_5926, 5, 3);

                __m128i _mask_a37b = __lsx_vreplgr2vr_w(1);
                _mask_a37b = __lsx_vinsgr2vr_w(_mask_a37b, 6, 1);
                _mask_a37b = __lsx_vinsgr2vr_w(_mask_a37b, 2, 2);
                _mask_a37b = __lsx_vinsgr2vr_w(_mask_a37b, 3, 3);

                for (int q = 0; q < channels_per_group; q++)
                {
                    const float* ptr0 = bottom_blob.channel(q);
                    const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                    const float* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                    float* outptr0 = top_blob.channel(q * 3);
                    float* outptr1 = top_blob.channel(q * 3 + 1);
                    float* outptr2 = top_blob.channel(q * 3 + 2);

                    for (int i = 0; i < size; i++)
                    {
                        __m128 _p0 = (__m128)__lsx_vld(ptr0, 0);
                        __m128 _p1 = (__m128)__lsx_vld(ptr1, 0);
                        __m128 _p2 = (__m128)__lsx_vld(ptr2, 0);

                        __m128 _0415 = (__m128)__lsx_vilvr_w((__m128i)_p1, (__m128i)_p0);
                        __m128 _2637 = (__m128)__lsx_vilvh_w((__m128i)_p1, (__m128i)_p0);
                        __m128 _4859 = (__m128)__lsx_vilvr_w((__m128i)_p2, (__m128i)_p1);
                        __m128 _6a7b = (__m128)__lsx_vilvh_w((__m128i)_p2, (__m128i)_p1);

                        __m128 _0481 = (__m128)__lsx_vshuf_w(_mask_0481, (__m128i)_p2, (__m128i)_0415);
                        __m128 _5926 = (__m128)__lsx_vshuf_w(_mask_5926, (__m128i)_2637, (__m128i)_4859);
                        __m128 _a37b = (__m128)__lsx_vshuf_w(_mask_a37b, (__m128i)_2637, (__m128i)_6a7b);

                        __lsx_vst(_0481, outptr0, 0);
                        __lsx_vst(_5926, outptr1, 0);
                        __lsx_vst(_a37b, outptr2, 0);

                        ptr0 += 4;
                        ptr1 += 4;
                        ptr2 += 4;
                        outptr0 += 4;
                        outptr1 += 4;
                        outptr2 += 4;
                    }
                }

                return 0;
            }

            if (_group == 4)
            {
                for (int q = 0; q < channels_per_group; q++)
                {
                    const float* ptr0 = bottom_blob.channel(q);
                    const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                    const float* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                    const float* ptr3 = bottom_blob.channel(channels_per_group * 3 + q);
                    float* outptr0 = top_blob.channel(q * 4);
                    float* outptr1 = top_blob.channel(q * 4 + 1);
                    float* outptr2 = top_blob.channel(q * 4 + 2);
                    float* outptr3 = top_blob.channel(q * 4 + 3);

                    for (int i = 0; i < size; i++)
                    {
                        __m128 _p0 = (__m128)__lsx_vld(ptr0, 0);
                        __m128 _p1 = (__m128)__lsx_vld(ptr1, 0);
                        __m128 _p2 = (__m128)__lsx_vld(ptr2, 0);
                        __m128 _p3 = (__m128)__lsx_vld(ptr3, 0);

                        transpose4x4_ps(_p0, _p1, _p2, _p3);

                        __lsx_vst(_p0, outptr0, 0);
                        __lsx_vst(_p1, outptr1, 0);
                        __lsx_vst(_p2, outptr2, 0);
                        __lsx_vst(_p3, outptr3, 0);

                        ptr0 += 4;
                        ptr1 += 4;
                        ptr2 += 4;
                        ptr3 += 4;
                        outptr0 += 4;
                        outptr1 += 4;
                        outptr2 += 4;
                        outptr3 += 4;
                    }
                }

                return 0;
            }
        }
    }
#endif // __loongarch_sx

    int channels_per_group = logical_channels / _group;
    size_t lane_size = elemsize / elempack;

    top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    for (int i = 0; i < _group; i++)
    {
        for (int j = 0; j < channels_per_group; j++)
        {
            int src_c = channels_per_group * i + j;
            int dst_c = _group * j + i;

            int src_q = src_c / elempack;
            int src_lane = src_c % elempack;
            int dst_q = dst_c / elempack;
            int dst_lane = dst_c % elempack;

            const unsigned char* ptr = bottom_blob.channel(src_q);
            unsigned char* outptr = top_blob.channel(dst_q);

            ptr += src_lane * lane_size;
            outptr += dst_lane * lane_size;

            for (int k = 0; k < size; k++)
            {
                memcpy(outptr, ptr, lane_size);

                ptr += elemsize;
                outptr += elemsize;
            }
        }
    }

    return 0;
}

int ShuffleChannel_loongarch::forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

    int logical_channels = channels * elempack;
    if (logical_channels % group != 0)
        return -100;

    int _group = reverse ? logical_channels / group : group;

    if (_group == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    if (elempack == 1)
        return ShuffleChannel::forward(bottom_blob, top_blob, opt);

#if __loongarch_sx
#if __loongarch_asx
    if (elempack == 8)
    {
        int channels_per_group = channels / _group;

        if (_group == 2 && channels % _group == 0)
        {
            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            for (int q = 0; q < channels_per_group; q++)
            {
                const unsigned short* ptr0 = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + q);
                unsigned short* outptr0 = top_blob.channel(q * 2);
                unsigned short* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    __m128i _p0 = __lsx_vld(ptr0, 0);
                    __m128i _p1 = __lsx_vld(ptr1, 0);

                    __m128i _lo = __lsx_vilvr_h(_p1, _p0);
                    __m128i _hi = __lsx_vilvh_h(_p1, _p0);

                    __lsx_vst(_lo, outptr0, 0);
                    __lsx_vst(_hi, outptr1, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                }
            }

            return 0;
        }

        if (_group == 4 && channels % _group == 0)
        {
            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            for (int q = 0; q < channels_per_group; q++)
            {
                const unsigned short* ptr0 = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + q);
                const unsigned short* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                const unsigned short* ptr3 = bottom_blob.channel(channels_per_group * 3 + q);
                unsigned short* outptr0 = top_blob.channel(q * 4);
                unsigned short* outptr1 = top_blob.channel(q * 4 + 1);
                unsigned short* outptr2 = top_blob.channel(q * 4 + 2);
                unsigned short* outptr3 = top_blob.channel(q * 4 + 3);

                for (int i = 0; i < size; i++)
                {
                    __m128i _p0 = __lsx_vld(ptr0, 0);
                    __m128i _p1 = __lsx_vld(ptr1, 0);
                    __m128i _p2 = __lsx_vld(ptr2, 0);
                    __m128i _p3 = __lsx_vld(ptr3, 0);

                    transpose8x4_epi16(_p0, _p1, _p2, _p3);

                    __lsx_vst(_p0, outptr0, 0);
                    __lsx_vst(_p1, outptr1, 0);
                    __lsx_vst(_p2, outptr2, 0);
                    __lsx_vst(_p3, outptr3, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                    outptr3 += 8;
                }
            }

            return 0;
        }

        if (_group == 8 && channels % _group == 0)
        {
            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            for (int q = 0; q < channels_per_group; q++)
            {
                const unsigned short* ptr0 = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + q);
                const unsigned short* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                const unsigned short* ptr3 = bottom_blob.channel(channels_per_group * 3 + q);
                const unsigned short* ptr4 = bottom_blob.channel(channels_per_group * 4 + q);
                const unsigned short* ptr5 = bottom_blob.channel(channels_per_group * 5 + q);
                const unsigned short* ptr6 = bottom_blob.channel(channels_per_group * 6 + q);
                const unsigned short* ptr7 = bottom_blob.channel(channels_per_group * 7 + q);
                unsigned short* outptr0 = top_blob.channel(q * 8);
                unsigned short* outptr1 = top_blob.channel(q * 8 + 1);
                unsigned short* outptr2 = top_blob.channel(q * 8 + 2);
                unsigned short* outptr3 = top_blob.channel(q * 8 + 3);
                unsigned short* outptr4 = top_blob.channel(q * 8 + 4);
                unsigned short* outptr5 = top_blob.channel(q * 8 + 5);
                unsigned short* outptr6 = top_blob.channel(q * 8 + 6);
                unsigned short* outptr7 = top_blob.channel(q * 8 + 7);

                for (int i = 0; i < size; i++)
                {
                    __m128i _p0 = __lsx_vld(ptr0, 0);
                    __m128i _p1 = __lsx_vld(ptr1, 0);
                    __m128i _p2 = __lsx_vld(ptr2, 0);
                    __m128i _p3 = __lsx_vld(ptr3, 0);
                    __m128i _p4 = __lsx_vld(ptr4, 0);
                    __m128i _p5 = __lsx_vld(ptr5, 0);
                    __m128i _p6 = __lsx_vld(ptr6, 0);
                    __m128i _p7 = __lsx_vld(ptr7, 0);

                    transpose8x8_epi16(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7);

                    __lsx_vst(_p0, outptr0, 0);
                    __lsx_vst(_p1, outptr1, 0);
                    __lsx_vst(_p2, outptr2, 0);
                    __lsx_vst(_p3, outptr3, 0);
                    __lsx_vst(_p4, outptr4, 0);
                    __lsx_vst(_p5, outptr5, 0);
                    __lsx_vst(_p6, outptr6, 0);
                    __lsx_vst(_p7, outptr7, 0);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    ptr3 += 8;
                    ptr4 += 8;
                    ptr5 += 8;
                    ptr6 += 8;
                    ptr7 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
                    outptr3 += 8;
                    outptr4 += 8;
                    outptr5 += 8;
                    outptr6 += 8;
                    outptr7 += 8;
                }
            }

            return 0;
        }
    }
#endif // __loongarch_asx

    if (elempack == 4)
    {
        int channels_per_group = channels / _group;

        if (_group == 2 && channels % _group != 0)
        {
            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            __m128i _mask_2301 = __lsx_vreplgr2vr_h(2);
            _mask_2301 = __lsx_vinsgr2vr_h(_mask_2301, 3, 1);
            _mask_2301 = __lsx_vinsgr2vr_h(_mask_2301, 8, 2);
            _mask_2301 = __lsx_vinsgr2vr_h(_mask_2301, 9, 3);

            for (int q = 0; q < channels_per_group; q++)
            {
                const unsigned short* ptr0 = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + q);
                const unsigned short* ptr2 = bottom_blob.channel(channels_per_group + q + 1);
                unsigned short* outptr0 = top_blob.channel(q * 2);
                unsigned short* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    __m128i _p0 = __lsx_vldrepl_d((void*)ptr0, 0);
                    __m128i _p1 = __lsx_vldrepl_d((void*)ptr1, 0);
                    __m128i _p2 = __lsx_vldrepl_d((void*)ptr2, 0);

                    __m128i _p12 = __lsx_vshuf_h(_mask_2301, _p2, _p1);

                    __m128i _p01 = __lsx_vilvr_h(_p12, _p0);

                    __lsx_vstelm_d(_p01, outptr0, 0, 0);
                    __lsx_vstelm_d(_p01, outptr1, 0, 1);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }

            // handle the last channel
            {
                const unsigned short* ptr0 = bottom_blob.channel(channels_per_group);
                const unsigned short* ptr1 = bottom_blob.channel(channels_per_group * 2);
                unsigned short* outptr = top_blob.channel(channels_per_group * 2);

                ptr1 += 2;

                for (int i = 0; i < size; i++)
                {
                    __m128i _p0 = __lsx_vldrepl_d((void*)ptr0, 0);
                    __m128i _p1 = __lsx_vldrepl_w((void*)ptr1, 0);

                    __m128i _p01 = __lsx_vilvr_h(_p1, _p0);

                    __lsx_vstelm_d(_p01, outptr, 0, 0);

                    ptr0 += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            return 0;
        }

        if (_group <= 4 && channels % _group == 0)
        {
            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (_group == 2)
            {
                for (int q = 0; q < channels_per_group; q++)
                {
                    const unsigned short* ptr0 = bottom_blob.channel(q);
                    const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + q);
                    unsigned short* outptr0 = top_blob.channel(q * 2);
                    unsigned short* outptr1 = top_blob.channel(q * 2 + 1);

                    for (int i = 0; i < size; i++)
                    {
                        __m128i _p0 = __lsx_vldrepl_d((void*)ptr0, 0);
                        __m128i _p1 = __lsx_vldrepl_d((void*)ptr1, 0);

                        __m128i _p01 = __lsx_vilvr_h(_p1, _p0);

                        __lsx_vstelm_d(_p01, outptr0, 0, 0);
                        __lsx_vstelm_d(_p01, outptr1, 0, 1);

                        ptr0 += 4;
                        ptr1 += 4;
                        outptr0 += 4;
                        outptr1 += 4;
                    }
                }

                return 0;
            }

            if (_group == 3)
            {
                __m128i _mask_0481 = __lsx_vreplgr2vr_h(0);
                _mask_0481 = __lsx_vinsgr2vr_h(_mask_0481, 1, 1);
                _mask_0481 = __lsx_vinsgr2vr_h(_mask_0481, 8, 2);
                _mask_0481 = __lsx_vinsgr2vr_h(_mask_0481, 2, 3);

                __m128i _mask_5926 = __lsx_vreplgr2vr_h(3);
                _mask_5926 = __lsx_vinsgr2vr_h(_mask_5926, 9, 1);
                _mask_5926 = __lsx_vinsgr2vr_h(_mask_5926, 4, 2);
                _mask_5926 = __lsx_vinsgr2vr_h(_mask_5926, 5, 3);

                __m128i _mask_a37b = __lsx_vreplgr2vr_h(10);
                _mask_a37b = __lsx_vinsgr2vr_h(_mask_a37b, 6, 1);
                _mask_a37b = __lsx_vinsgr2vr_h(_mask_a37b, 7, 2);
                _mask_a37b = __lsx_vinsgr2vr_h(_mask_a37b, 11, 3);

                for (int q = 0; q < channels_per_group; q++)
                {
                    const unsigned short* ptr0 = bottom_blob.channel(q);
                    const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + q);
                    const unsigned short* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                    unsigned short* outptr0 = top_blob.channel(q * 3);
                    unsigned short* outptr1 = top_blob.channel(q * 3 + 1);
                    unsigned short* outptr2 = top_blob.channel(q * 3 + 2);

                    for (int i = 0; i < size; i++)
                    {
                        __m128i _p0 = __lsx_vldrepl_d((void*)ptr0, 0);
                        __m128i _p1 = __lsx_vldrepl_d((void*)ptr1, 0);
                        __m128i _p2 = __lsx_vldrepl_d((void*)ptr2, 0);

                        __m128i _p01 = __lsx_vilvr_h(_p1, _p0);

                        __m128i _0481 = __lsx_vshuf_h(_mask_0481, _p2, _p01);
                        __m128i _5926 = __lsx_vshuf_h(_mask_5926, _p2, _p01);
                        __m128i _a37b = __lsx_vshuf_h(_mask_a37b, _p2, _p01);

                        __lsx_vstelm_d(_0481, outptr0, 0, 0);
                        __lsx_vstelm_d(_5926, outptr1, 0, 0);
                        __lsx_vstelm_d(_a37b, outptr2, 0, 0);

                        ptr0 += 4;
                        ptr1 += 4;
                        ptr2 += 4;
                        outptr0 += 4;
                        outptr1 += 4;
                        outptr2 += 4;
                    }
                }

                return 0;
            }

            if (_group == 4)
            {
                for (int q = 0; q < channels_per_group; q++)
                {
                    const unsigned short* ptr0 = bottom_blob.channel(q);
                    const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + q);
                    const unsigned short* ptr2 = bottom_blob.channel(channels_per_group * 2 + q);
                    const unsigned short* ptr3 = bottom_blob.channel(channels_per_group * 3 + q);
                    unsigned short* outptr0 = top_blob.channel(q * 4);
                    unsigned short* outptr1 = top_blob.channel(q * 4 + 1);
                    unsigned short* outptr2 = top_blob.channel(q * 4 + 2);
                    unsigned short* outptr3 = top_blob.channel(q * 4 + 3);

                    for (int i = 0; i < size; i++)
                    {
                        __m128i _p0 = __lsx_vldrepl_d((void*)ptr0, 0);
                        __m128i _p1 = __lsx_vldrepl_d((void*)ptr1, 0);
                        __m128i _p2 = __lsx_vldrepl_d((void*)ptr2, 0);
                        __m128i _p3 = __lsx_vldrepl_d((void*)ptr3, 0);

                        __m128i _p01 = __lsx_vilvr_h(_p1, _p0);
                        __m128i _p23 = __lsx_vilvr_h(_p3, _p2);

                        __m128i _p02 = __lsx_vilvr_w(_p23, _p01);
                        __m128i _p13 = __lsx_vilvh_w(_p23, _p01);

                        __lsx_vstelm_d(_p02, outptr0, 0, 0);
                        __lsx_vstelm_d(_p02, outptr1, 0, 1);
                        __lsx_vstelm_d(_p13, outptr2, 0, 0);
                        __lsx_vstelm_d(_p13, outptr3, 0, 1);

                        ptr0 += 4;
                        ptr1 += 4;
                        ptr2 += 4;
                        ptr3 += 4;
                        outptr0 += 4;
                        outptr1 += 4;
                        outptr2 += 4;
                        outptr3 += 4;
                    }
                }

                return 0;
            }
        }
    }
#endif // __loongarch_sx

    int channels_per_group = logical_channels / _group;
    size_t lane_size = elemsize / elempack;

    top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    for (int i = 0; i < _group; i++)
    {
        for (int j = 0; j < channels_per_group; j++)
        {
            int src_c = channels_per_group * i + j;
            int dst_c = _group * j + i;

            int src_q = src_c / elempack;
            int src_lane = src_c % elempack;
            int dst_q = dst_c / elempack;
            int dst_lane = dst_c % elempack;

            const unsigned char* ptr = bottom_blob.channel(src_q);
            unsigned char* outptr = top_blob.channel(dst_q);

            ptr += src_lane * lane_size;
            outptr += dst_lane * lane_size;

            for (int k = 0; k < size; k++)
            {
                memcpy(outptr, ptr, lane_size);

                ptr += elemsize;
                outptr += elemsize;
            }
        }
    }

    return 0;
}

} // namespace ncnn
