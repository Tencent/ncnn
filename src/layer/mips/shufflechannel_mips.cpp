// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "shufflechannel_mips.h"

#include <stdint.h>

#if __mips_msa
#include <msa.h>
#include "mips_usability.h"
#endif // __mips_msa

namespace ncnn {

ShuffleChannel_mips::ShuffleChannel_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int ShuffleChannel_mips::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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

#if __mips_msa
    if (elempack == 4)
    {
        int channels_per_group = channels / _group;

        if (_group == 2 && channels % _group != 0)
        {
            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr0 = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                const float* ptr2 = bottom_blob.channel(channels_per_group + q + 1);
                float* outptr0 = top_blob.channel(q * 2);
                float* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    v4f32 _p0 = (v4f32)__msa_ld_w(ptr0, 0);
                    v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                    v4f32 _p2 = (v4f32)__msa_ld_w(ptr2, 0);

                    v4f32 _p12 = (v4f32)__msa_sldi_b((v16i8)_p2, (v16i8)_p1, 8);

                    v4f32 _lo = (v4f32)__msa_ilvr_w((v4i32)_p12, (v4i32)_p0);
                    v4f32 _hi = (v4f32)__msa_ilvl_w((v4i32)_p12, (v4i32)_p0);

                    __msa_st_w((v4i32)_lo, outptr0, 0);
                    __msa_st_w((v4i32)_hi, outptr1, 0);

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
                    v4f32 _p0 = (v4f32)__msa_ld_w(ptr0, 0);
                    v4f32 _p1 = (v4f32)__msa_loadl_d(ptr1);

                    v4f32 _lo = (v4f32)__msa_ilvr_w((v4i32)_p1, (v4i32)_p0);

                    __msa_st_w((v4i32)_lo, outptr, 0);

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
                        v4f32 _p0 = (v4f32)__msa_ld_w(ptr0, 0);
                        v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);

                        v4f32 _lo = (v4f32)__msa_ilvr_w((v4i32)_p1, (v4i32)_p0);
                        v4f32 _hi = (v4f32)__msa_ilvl_w((v4i32)_p1, (v4i32)_p0);

                        __msa_st_w((v4i32)_lo, outptr0, 0);
                        __msa_st_w((v4i32)_hi, outptr1, 0);

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
                v4i32 _mask_0481 = __msa_set_w(0, 1, 4, 2);
                v4i32 _mask_5926 = __msa_set_w(2, 3, 4, 5);
                v4i32 _mask_a37b = __msa_set_w(1, 6, 2, 3);

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
                        v4f32 _p0 = (v4f32)__msa_ld_w(ptr0, 0);
                        v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                        v4f32 _p2 = (v4f32)__msa_ld_w(ptr2, 0);

                        v4f32 _0415 = (v4f32)__msa_ilvr_w((v4i32)_p1, (v4i32)_p0);
                        v4f32 _2637 = (v4f32)__msa_ilvl_w((v4i32)_p1, (v4i32)_p0);
                        v4f32 _4859 = (v4f32)__msa_ilvr_w((v4i32)_p2, (v4i32)_p1);
                        v4f32 _6a7b = (v4f32)__msa_ilvl_w((v4i32)_p2, (v4i32)_p1);

                        v4f32 _0481 = (v4f32)__msa_vshf_w(_mask_0481, (v4i32)_p2, (v4i32)_0415);
                        v4f32 _5926 = (v4f32)__msa_vshf_w(_mask_5926, (v4i32)_2637, (v4i32)_4859);
                        v4f32 _a37b = (v4f32)__msa_vshf_w(_mask_a37b, (v4i32)_2637, (v4i32)_6a7b);

                        __msa_st_w((v4i32)_0481, outptr0, 0);
                        __msa_st_w((v4i32)_5926, outptr1, 0);
                        __msa_st_w((v4i32)_a37b, outptr2, 0);

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
                        v4f32 _p0 = (v4f32)__msa_ld_w(ptr0, 0);
                        v4f32 _p1 = (v4f32)__msa_ld_w(ptr1, 0);
                        v4f32 _p2 = (v4f32)__msa_ld_w(ptr2, 0);
                        v4f32 _p3 = (v4f32)__msa_ld_w(ptr3, 0);

                        transpose4x4_ps(_p0, _p1, _p2, _p3);

                        __msa_st_w((v4i32)_p0, outptr0, 0);
                        __msa_st_w((v4i32)_p1, outptr1, 0);
                        __msa_st_w((v4i32)_p2, outptr2, 0);
                        __msa_st_w((v4i32)_p3, outptr3, 0);

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
#endif // __mips_msa

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

int ShuffleChannel_mips::forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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

#if __mips_msa
    if (elempack == 4)
    {
        int channels_per_group = channels / _group;

        if (_group == 2 && channels % _group != 0)
        {
            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            v8i16 _mask_2301 = __msa_fill_h(2);
            _mask_2301 = __msa_insert_h(_mask_2301, 1, 3);
            _mask_2301 = __msa_insert_h(_mask_2301, 2, 8);
            _mask_2301 = __msa_insert_h(_mask_2301, 3, 9);

            for (int q = 0; q < channels_per_group; q++)
            {
                const unsigned short* ptr0 = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob.channel(channels_per_group + q);
                const unsigned short* ptr2 = bottom_blob.channel(channels_per_group + q + 1);
                unsigned short* outptr0 = top_blob.channel(q * 2);
                unsigned short* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    v8i16 _p0 = (v8i16)__msa_loadl_d(ptr0);
                    v8i16 _p1 = (v8i16)__msa_loadl_d(ptr1);
                    v8i16 _p2 = (v8i16)__msa_loadl_d(ptr2);

                    v8i16 _p12 = __msa_vshf_h(_mask_2301, _p2, _p1);

                    v8i16 _p01 = (v8i16)__msa_ilvr_h(_p12, _p0);

                    *(uint64_t*)outptr0 = __msa_copy_s_d((v2i64)_p01, 0);
                    *(uint64_t*)outptr1 = __msa_copy_s_d((v2i64)_p01, 1);

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
                    v8i16 _p0 = (v8i16)__msa_loadl_d(ptr0);
                    v8i16 _p1 = (v8i16)__msa_fill_w(*(const int*)ptr1);

                    v8i16 _p01 = (v8i16)__msa_ilvr_h(_p1, _p0);

                    *(uint64_t*)outptr = __msa_copy_s_d((v2i64)_p01, 0);

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
                        v8i16 _p0 = (v8i16)__msa_loadl_d(ptr0);
                        v8i16 _p1 = (v8i16)__msa_loadl_d(ptr1);

                        v8i16 _p01 = (v8i16)__msa_ilvr_h(_p1, _p0);

                        *(uint64_t*)outptr0 = __msa_copy_s_d((v2i64)_p01, 0);
                        *(uint64_t*)outptr1 = __msa_copy_s_d((v2i64)_p01, 1);

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
                v8i16 _mask_0481 = __msa_fill_h(0);
                _mask_0481 = __msa_insert_h(_mask_0481, 1, 1);
                _mask_0481 = __msa_insert_h(_mask_0481, 2, 8);
                _mask_0481 = __msa_insert_h(_mask_0481, 3, 2);

                v8i16 _mask_5926 = __msa_fill_h(3);
                _mask_5926 = __msa_insert_h(_mask_5926, 1, 9);
                _mask_5926 = __msa_insert_h(_mask_5926, 2, 4);
                _mask_5926 = __msa_insert_h(_mask_5926, 3, 5);

                v8i16 _mask_a37b = __msa_fill_h(10);
                _mask_a37b = __msa_insert_h(_mask_a37b, 1, 6);
                _mask_a37b = __msa_insert_h(_mask_a37b, 2, 7);
                _mask_a37b = __msa_insert_h(_mask_a37b, 3, 11);

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
                        v8i16 _p0 = (v8i16)__msa_loadl_d(ptr0);
                        v8i16 _p1 = (v8i16)__msa_loadl_d(ptr1);
                        v8i16 _p2 = (v8i16)__msa_loadl_d(ptr2);

                        v8i16 _p01 = (v8i16)__msa_ilvr_h(_p1, _p0);

                        v8i16 _0481 = __msa_vshf_h(_mask_0481, _p2, _p01);
                        v8i16 _5926 = __msa_vshf_h(_mask_5926, _p2, _p01);
                        v8i16 _a37b = __msa_vshf_h(_mask_a37b, _p2, _p01);

                        *(uint64_t*)outptr0 = __msa_copy_s_d((v2i64)_0481, 0);
                        *(uint64_t*)outptr1 = __msa_copy_s_d((v2i64)_5926, 0);
                        *(uint64_t*)outptr2 = __msa_copy_s_d((v2i64)_a37b, 0);

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
                        v8i16 _p0 = (v8i16)__msa_loadl_d(ptr0);
                        v8i16 _p1 = (v8i16)__msa_loadl_d(ptr1);
                        v8i16 _p2 = (v8i16)__msa_loadl_d(ptr2);
                        v8i16 _p3 = (v8i16)__msa_loadl_d(ptr3);

                        v8i16 _p01 = (v8i16)__msa_ilvr_h(_p1, _p0);
                        v8i16 _p23 = (v8i16)__msa_ilvr_h(_p3, _p2);

                        v8i16 _p02 = (v8i16)__msa_ilvr_w((v4i32)_p23, (v4i32)_p01);
                        v8i16 _p13 = (v8i16)__msa_ilvl_w((v4i32)_p23, (v4i32)_p01);

                        *(uint64_t*)outptr0 = __msa_copy_s_d((v2i64)_p02, 0);
                        *(uint64_t*)outptr1 = __msa_copy_s_d((v2i64)_p02, 1);
                        *(uint64_t*)outptr2 = __msa_copy_s_d((v2i64)_p13, 0);
                        *(uint64_t*)outptr3 = __msa_copy_s_d((v2i64)_p13, 1);

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
#endif // __mips_msa

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
