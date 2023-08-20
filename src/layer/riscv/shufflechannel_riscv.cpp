// Xavier Hsinyuan is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 Xavier Hsinyuan <me@lstlx.com>. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "shufflechannel_riscv.h"
#include "layer_type.h"
#include <cstdint>

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#include <assert.h>
#endif // __riscv_vector

namespace ncnn {

ShuffleChannel_riscv::ShuffleChannel_riscv()
{
#if __riscv_vector
    support_packing = true;
    // TODO support float16
#endif // __riscv_vector
}

#if __riscv_vector
static NCNN_FORCEINLINE int create_bitmask(int group, uint8_t (&bitarray)[32], int packn)
{
    const int uint8_bitsize = sizeof(uint8_t) * 8;
    assert(group > 0);
    /* Unlikely to have a processor w/ vlenb >= 128B */
    assert(packn > 0 && packn <= 32);
    assert(uint8_size == 8);
    for (int i = 0; i < 32; i++)
    {
        bitarray[i] = 0;
    }
    for (int i = 0; i < packn * group; i += group)
    {
        bitarray[i / uint8_bitsize] |= (1 << (i % uint8_bitsize));
    }
    return 0;
}
#endif

int ShuffleChannel_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

    int elempack = bottom_blob.elempack;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int _group = reverse ? channels * elempack / group : group;

    if (_group == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    int channels_per_group = channels / _group;

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    // TODO pack1
    if (elempack == packn)
    {
        static uint8_t bitmask[32];
        // follow arm version
        if (_group == 2 && channels % _group != 0)
        {
            top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            // create mask
            memset(bitmask, 0b01010101 /* little endian*/, 32);

            size_t vl = vsetvl_e32m4(packn * 3);
            vbool8_t _mask = vlm_v_b8(bitmask, vl);
            vuint32m4_t _idx = viota_m_u32m4(_mask, vl);
            vuint32m4_t _idx_shifted = vslideup_vx_u32m4(vundefined_u32m4(), vadd_vx_u32m4_m(_mask, vundefined_u32m4(), _idx, packn + packn / 2, vl), 1, vl);
            _idx = vmerge_vvm_u32m4(_mask, _idx_shifted, _idx, vl);

            int size = w * h;

            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr0 = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob.channel(channels_per_group + q);
                const float* ptr2 = bottom_blob.channel(channels_per_group + q + 1);
                float* outptr0 = top_blob.channel(q * 2);
                float* outptr1 = top_blob.channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
                    vl = vsetvl_e32m2(packn);
                    vfloat32m1_t _p0 = vle32_v_f32m1(ptr0, vl);
                    vfloat32m1_t _p1 = vle32_v_f32m1(ptr1, vl);
                    vfloat32m1_t _p2 = vle32_v_f32m1(ptr2, vl);

                    vfloat32m4_t _p012 = vundefined_f32m4();
                    _p012 = vset_v_f32m1_f32m4(_p012, 0, _p0);
                    _p012 = vset_v_f32m1_f32m4(_p012, 1, _p1);
                    _p012 = vset_v_f32m1_f32m4(_p012, 2, _p2);

                    vl = vsetvl_e32m2(packn * 3);
                    vfloat32m4_t _p01 = vrgather_vv_f32m4(_p012, _idx, vl);

                    vl = vsetvl_e32m2(packn);
                    vse32_v_f32m1(outptr0, vget_v_f32m4_f32m1(_p01, 0), vl);
                    vse32_v_f32m1(outptr1, vget_v_f32m4_f32m1(_p01, 1), vl);

                    ptr0 += packn;
                    ptr1 += packn;
                    ptr2 += packn;
                    outptr0 += packn;
                    outptr1 += packn;
                }
            }

            // handle the last channel
            {
                size_t vl = vsetvl_e32m2(packn * 2);
                vbool16_t _mask = vlm_v_b16(bitmask, vl);
                vuint32m2_t _idx = viota_m_u32m2(_mask, vl);
                vuint32m2_t _idx_shifted = vslideup_vx_u32m2(vundefined_u32m2(), vadd_vx_u32m2_m(_mask, vundefined_u32m2(), _idx, packn, vl), 1, vl);
                _idx = vmerge_vvm_u32m2(_mask, _idx_shifted, _idx, vl);

                const float* ptr0 = bottom_blob.channel(channels_per_group);
                const float* ptr1 = bottom_blob.channel(channels_per_group + channels_per_group);
                float* outptr0 = top_blob.channel(channels_per_group * 2);

                ptr1 += packn / 2;

                for (int i = 0; i < size; i++)
                {
                    vl = vsetvl_e32m1(packn);
                    vfloat32m1_t _p0 = vle32_v_f32m1(ptr0, vl);
                    vfloat32m1_t _p1 = vle32_v_f32m1(ptr1, vl);
                    vfloat32m2_t _p01 = vset_v_f32m1_f32m2(vundefined_f32m2(), 0, _p0);
                    _p01 = vset_v_f32m1_f32m2(_p01, 1, _p1);

                    vl = vsetvl_e32m2(packn * 2);
                    _p01 = vrgather_vv_f32m2(_p01, _idx, vl);

                    vl = vsetvl_e32m1(packn);
                    vse32_v_f32m1(outptr0, vget_v_f32m2_f32m1(_p01, 0), vl);

                    ptr0 += packn;
                    ptr1 += packn;
                    outptr0 += packn;
                }
            }

            return 0;
        }
        // group too large or shuffle inside elempack
        if (_group > elempack || (_group > 8 && elempack > 8) || channels % _group != 0)
        {
            // convert to pack1
            Mat bottom_blob_unpacked;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
            // convert packing won't change w,h
            int channels_unpacked = bottom_blob_unpacked.c;
            size_t elemsize_unpacked = bottom_blob_unpacked.elemsize;
            int _group_unpack = reverse ? channels_unpacked / group : group;
            if (channels_unpacked % group != 0)
            {
                // reject invalid group
                return -100;
            }
            Mat top_blob_unpacked;
            top_blob_unpacked.create(w, h, channels_unpacked, elemsize_unpacked, opt.blob_allocator);

            int channels_unpacked_per_group = channels_unpacked / _group_unpack;
            const size_t feature_sz = (size_t)w * h;
            for (int i = 0; i < _group_unpack; i++)
            {
                for (int j = 0; j < channels_unpacked_per_group; j++)
                {
                    float* p_dst = top_blob_unpacked.channel(_group_unpack * j + i);
                    const float* p_src = bottom_blob_unpacked.channel(channels_unpacked_per_group * i + j);
                    int n = feature_sz;
                    while (n > 0)
                    {
                        size_t vl = vsetvl_e32m8(n);
                        vfloat32m8_t _src = vle32_v_f32m8(p_src, vl);
                        vse32_v_f32m8(p_dst, _src, vl);
                        n -= vl;
                        p_src += vl;
                        p_dst += vl;
                    }
                }
            }
            convert_packing(top_blob_unpacked, top_blob, elempack, opt);
            return 0;
        }
        top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
        const int size = w * h;
        create_bitmask(_group, bitmask, packn);
        if (_group == 4 && packn == 4)
        {
            const size_t vl = vsetvl_e32m1(packn);
            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr0 = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob.channel(q + channels_per_group * 1);
                ptrdiff_t ptrdiff01 = (ptr1 - ptr0) * sizeof(float);

                float* outptr0 = top_blob.channel(q * 4);
                float* outptr1 = top_blob.channel(q * 4 + 1);
                float* outptr2 = top_blob.channel(q * 4 + 2);
                float* outptr3 = top_blob.channel(q * 4 + 3);
                for (int i = 0; i < size; i++)
                {
                    vfloat32m1_t _p0;
                    vfloat32m1_t _p1;
                    vfloat32m1_t _p2;
                    vfloat32m1_t _p3;
                    vlsseg4e32_v_f32m1(&_p0, &_p1, &_p2, &_p3, ptr0, ptrdiff01, vl);
                    vse32_v_f32m1(outptr0, _p0, vl);
                    vse32_v_f32m1(outptr1, _p1, vl);
                    vse32_v_f32m1(outptr2, _p2, vl);
                    vse32_v_f32m1(outptr3, _p3, vl);

                    ptr0 += 4;
                    ptr1 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
            }
            return 0;
        }
        else if (_group == 8 && packn == 8)
        {
            const size_t vl = vsetvl_e32m1(packn);
            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr0 = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob.channel(q + channels_per_group * 1);
                ptrdiff_t ptrdiff01 = (ptr1 - ptr0) * sizeof(float);

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
                    vfloat32m1_t _p0;
                    vfloat32m1_t _p1;
                    vfloat32m1_t _p2;
                    vfloat32m1_t _p3;
                    vfloat32m1_t _p4;
                    vfloat32m1_t _p5;
                    vfloat32m1_t _p6;
                    vfloat32m1_t _p7;
                    vlsseg8e32_v_f32m1(&_p0, &_p1, &_p2, &_p3, &_p4, &_p5, &_p6, &_p7, ptr0, ptrdiff01, vl);
                    vse32_v_f32m1(outptr0, _p0, vl);
                    vse32_v_f32m1(outptr1, _p1, vl);
                    vse32_v_f32m1(outptr2, _p2, vl);
                    vse32_v_f32m1(outptr3, _p3, vl);
                    vse32_v_f32m1(outptr4, _p4, vl);
                    vse32_v_f32m1(outptr5, _p5, vl);
                    vse32_v_f32m1(outptr6, _p6, vl);
                    vse32_v_f32m1(outptr7, _p7, vl);
                    ptr0 += 8;
                    ptr1 += 8;
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
        if (_group <= 4)
        {
            size_t vl;
            vl = vsetvl_e32m4(_group * elempack);
            // create bitmask
            vbool8_t _mask = vlm_v_b8(bitmask, vl);
            vuint32m4_t _idx_init = viota_m_u32m4(_mask, vl);
            vuint32m4_t _idx = vadd_vx_u32m4(_idx_init, (_group - 1) * elempack, vl);
            for (int shift = _group - 2; shift >= 0; shift--)
            {
                vuint32m4_t _idx_lower = vadd_vx_u32m4(_idx_init, shift * elempack, vl);
                _idx = vslideup_vx_u32m4(vundefined_u32m4(), _idx, 1, vl);
                _idx = vmerge_vvm_u32m4(_mask, _idx, _idx_lower, vl);
            }

            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr1 = NULL;
                const float* ptr2 = NULL;
                const float* ptr3 = NULL;
                const float* ptr4 = NULL;
                float* outptr1 = NULL;
                float* outptr2 = NULL;
                float* outptr3 = NULL;
                float* outptr4 = NULL;
                switch (_group)
                {
                case 4:
                    ptr4 = bottom_blob.channel(q + channels_per_group * 3);
                    outptr4 = top_blob.channel(q * _group + 3);
                case 3:
                    ptr3 = bottom_blob.channel(q + channels_per_group * 2);
                    outptr3 = top_blob.channel(q * _group + 2);
                case 2:
                    ptr2 = bottom_blob.channel(q + channels_per_group);
                    outptr2 = top_blob.channel(q * _group + 1);
                    ptr1 = bottom_blob.channel(q);
                    outptr1 = top_blob.channel(q * _group);
                    break;
                }

                for (int i = 0; i < size; i++)
                {
                    vfloat32m4_t _src = vundefined_f32m4();
                    vl = vsetvl_e32m1(elemsize);
                    switch (_group)
                    {
                    case 4:
                        _src = vset_v_f32m1_f32m4(_src, 3, vle32_v_f32m1(ptr4, vl));
                    case 3:
                        _src = vset_v_f32m1_f32m4(_src, 2, vle32_v_f32m1(ptr3, vl));
                    case 2:
                        _src = vset_v_f32m1_f32m4(_src, 1, vle32_v_f32m1(ptr2, vl));
                        _src = vset_v_f32m1_f32m4(_src, 0, vle32_v_f32m1(ptr1, vl));
                        break;
                    }
                    vl = vsetvl_e32m4(_group * elempack);
                    vfloat32m4_t _dst = vrgather_vv_f32m4(_src, _idx, vl);
                    vl = vsetvl_e32m1(elempack);

                    switch (_group)
                    {
                    case 4:
                        vse32_v_f32m1(outptr4, vget_v_f32m4_f32m1(_dst, 3), vl);
                        outptr4 += elempack;
                        ptr4 += elempack;
                    case 3:
                        vse32_v_f32m1(outptr3, vget_v_f32m4_f32m1(_dst, 2), vl);
                        outptr3 += elempack;
                        ptr3 += elempack;
                    case 2:
                        vse32_v_f32m1(outptr2, vget_v_f32m4_f32m1(_dst, 1), vl);
                        outptr2 += elempack;
                        ptr2 += elempack;
                        vse32_v_f32m1(outptr1, vget_v_f32m4_f32m1(_dst, 0), vl);
                        outptr1 += elempack;
                        ptr1 += elempack;
                        break;
                    }
                }
            }
        }
        else /* if (_group <= 8) */
        {
            size_t vl;
            vl = vsetvl_e32m8(_group * elempack);
            // create bitmask
            vbool4_t _mask = vlm_v_b4(bitmask, vl);
            vuint32m8_t _idx_init = viota_m_u32m8(_mask, vl);
            vuint32m8_t _idx = vadd_vx_u32m8(_idx_init, (_group - 1) * elempack, vl);
            for (int shift = _group - 2; shift >= 0; shift--)
            {
                vuint32m8_t _idx_lower = vadd_vx_u32m8(_idx_init, shift * elempack, vl);
                _idx = vslideup_vx_u32m8(vundefined_u32m8(), _idx, 1, vl);
                _idx = vmerge_vvm_u32m8(_mask, _idx, _idx_lower, vl);
            }

            for (int q = 0; q < channels_per_group; q++)
            {
                const float* ptr1 = NULL;
                const float* ptr2 = NULL;
                const float* ptr3 = NULL;
                const float* ptr4 = NULL;
                const float* ptr5 = NULL;
                const float* ptr6 = NULL;
                const float* ptr7 = NULL;
                const float* ptr8 = NULL;
                float* outptr1 = NULL;
                float* outptr2 = NULL;
                float* outptr3 = NULL;
                float* outptr4 = NULL;
                float* outptr5 = NULL;
                float* outptr6 = NULL;
                float* outptr7 = NULL;
                float* outptr8 = NULL;
                switch (_group)
                {
                case 8:
                    ptr8 = bottom_blob.channel(q + channels_per_group * 7);
                    outptr8 = top_blob.channel(q * _group + 7);
                case 7:
                    ptr7 = bottom_blob.channel(q + channels_per_group * 6);
                    outptr7 = top_blob.channel(q * _group + 6);
                case 6:
                    ptr6 = bottom_blob.channel(q + channels_per_group * 5);
                    outptr6 = top_blob.channel(q * _group + 5);
                case 5:
                    ptr5 = bottom_blob.channel(q + channels_per_group * 4);
                    outptr5 = top_blob.channel(q * _group + 4);
                    ptr4 = bottom_blob.channel(q + channels_per_group * 3);
                    outptr4 = top_blob.channel(q * _group + 3);
                    ptr3 = bottom_blob.channel(q + channels_per_group * 2);
                    outptr3 = top_blob.channel(q * _group + 2);
                    ptr2 = bottom_blob.channel(q + channels_per_group);
                    outptr2 = top_blob.channel(q * _group + 1);
                    ptr1 = bottom_blob.channel(q);
                    outptr1 = top_blob.channel(q * _group);
                    break;
                }

                for (int i = 0; i < size; i++)
                {
                    vfloat32m8_t _src = vundefined_f32m8();
                    vl = vsetvl_e32m1(elemsize);
                    switch (_group)
                    {
                    case 8:
                        _src = vset_v_f32m1_f32m8(_src, 7, vle32_v_f32m1(ptr8, vl));
                    case 7:
                        _src = vset_v_f32m1_f32m8(_src, 6, vle32_v_f32m1(ptr7, vl));
                    case 6:
                        _src = vset_v_f32m1_f32m8(_src, 5, vle32_v_f32m1(ptr6, vl));
                    case 5:
                        _src = vset_v_f32m1_f32m8(_src, 4, vle32_v_f32m1(ptr5, vl));
                        _src = vset_v_f32m1_f32m8(_src, 3, vle32_v_f32m1(ptr4, vl));
                        _src = vset_v_f32m1_f32m8(_src, 2, vle32_v_f32m1(ptr3, vl));
                        _src = vset_v_f32m1_f32m8(_src, 1, vle32_v_f32m1(ptr2, vl));
                        _src = vset_v_f32m1_f32m8(_src, 0, vle32_v_f32m1(ptr1, vl));
                        break;
                    }
                    vl = vsetvl_e32m8(_group * elempack);
                    vfloat32m8_t _dst = vrgather_vv_f32m8(_src, _idx, vl);
                    vl = vsetvl_e32m1(elempack);

                    switch (_group)
                    {
                    case 8:
                        vse32_v_f32m1(outptr8, vget_v_f32m8_f32m1(_dst, 7), vl);
                        outptr8 += elempack;
                        ptr8 += elempack;
                    case 7:
                        vse32_v_f32m1(outptr7, vget_v_f32m8_f32m1(_dst, 6), vl);
                        outptr7 += elempack;
                        ptr7 += elempack;
                    case 6:
                        vse32_v_f32m1(outptr6, vget_v_f32m8_f32m1(_dst, 5), vl);
                        outptr6 += elempack;
                        ptr6 += elempack;
                    case 5:
                        vse32_v_f32m1(outptr5, vget_v_f32m8_f32m1(_dst, 4), vl);
                        outptr5 += elempack;
                        ptr5 += elempack;
                        vse32_v_f32m1(outptr4, vget_v_f32m8_f32m1(_dst, 3), vl);
                        outptr4 += elempack;
                        ptr4 += elempack;
                        vse32_v_f32m1(outptr3, vget_v_f32m8_f32m1(_dst, 2), vl);
                        outptr3 += elempack;
                        ptr3 += elempack;
                        vse32_v_f32m1(outptr2, vget_v_f32m8_f32m1(_dst, 1), vl);
                        outptr2 += elempack;
                        ptr2 += elempack;
                        vse32_v_f32m1(outptr1, vget_v_f32m8_f32m1(_dst, 0), vl);
                        outptr1 += elempack;
                        ptr1 += elempack;
                        break;
                    }
                }
            }
        }

        return 0;
    }
#endif

#if __riscv_vector
    if (elempack == 1)
    {
#endif
        if (channels % group != 0)
        {
            // reject invalid group
            return -100;
        }

        top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __riscv_vector
        const size_t feature_sz = (size_t)w * h;
#else
        const size_t feature_sz = (size_t)w * h * elemsize;
#endif
        for (int i = 0; i < _group; i++)
        {
            for (int j = 0; j < channels_per_group; j++)
            {
#if __riscv_vector
                float* p_dst = top_blob.channel(_group * j + i);
                const float* p_src = bottom_blob.channel(channels_per_group * i + j);
                int n = feature_sz;
                while (n > 0)
                {
                    size_t vl = vsetvl_e32m8(n);
                    vfloat32m8_t _src = vle32_v_f32m8(p_src, vl);
                    vse32_v_f32m8(p_dst, _src, vl);
                    n -= vl;
                    p_src += vl;
                    p_dst += vl;
                }
#else
                int src_q = channels_per_group * i + j;
                int dst_q = _group * j + i;
                memcpy(top_blob.channel(dst_q), bottom_blob.channel(src_q), feature_sz);
#endif // __riscv_vector
            }
        }
#if __riscv_vector
    }
#endif // __riscv_vector

    return 0;
}
} // namespace ncnn