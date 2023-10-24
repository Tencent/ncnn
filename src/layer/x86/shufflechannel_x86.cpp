// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "shufflechannel_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

namespace ncnn {

ShuffleChannel_x86::ShuffleChannel_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int ShuffleChannel_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();
    if (elembits != 32)
    {
        NCNN_LOGE("Elembits = %d is not implemented yet.", elembits);
        return -100;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

    int _group = reverse ? channels * elempack / group : group;
    int channels_per_group = channels / _group;

    if (_group == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        __m512i _idxlo = _mm512_set_epi64(
                             0x1700000007, 0x1600000006,
                             0x1500000005, 0x1400000004,
                             0x1300000003, 0x1200000002,
                             0x1100000001, 0x1000000000);
        __m512i _idxhi = _mm512_set_epi64(
                             0x1f0000000f, 0x1e0000000e,
                             0x1d0000000d, 0x1c0000000c,
                             0x1b0000000b, 0x1a0000000a,
                             0x1900000009, 0x1800000008);

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
                    __m512 _p0 = _mm512_loadu_ps(ptr0);
                    __m512 _p1 = _mm512_loadu_ps(ptr1);
                    __m512 _p2 = _mm512_loadu_ps(ptr2);

                    __m512 _p12 = _mm512_castsi512_ps(
                                      _mm512_alignr_epi64(_mm512_castps_si512(_p2), _mm512_castps_si512(_p1), 4));

                    __m512 _lo = _mm512_permutex2var_ps(_p0, _idxlo, _p12);
                    __m512 _hi = _mm512_permutex2var_ps(_p0, _idxhi, _p12);

                    _mm512_storeu_ps(outptr0, _lo);
                    _mm512_storeu_ps(outptr1, _hi);

                    ptr0 += 16;
                    ptr1 += 16;
                    ptr2 += 16;
                    outptr0 += 16;
                    outptr1 += 16;
                }
            }

            // handle the last channel
            {
                const float* ptr0 = bottom_blob.channel(channels_per_group);
                const float* ptr1 = bottom_blob.channel(channels_per_group * 2);
                float* outptr = top_blob.channel(channels_per_group * 2);

                ptr1 += 8;

                for (int i = 0; i < size; i++)
                {
                    __m256 _p0 = _mm256_loadu_ps(ptr0);
                    __m256 _p1 = _mm256_loadu_ps(ptr1);

                    __m256 _lo = _mm256_unpacklo_ps(_p0, _p1);
                    __m256 _hi = _mm256_unpackhi_ps(_p0, _p1);

                    __m256 _lo_ = _mm256_permute2f128_ps(_lo, _hi, 0x20);
                    __m256 _hi_ = _mm256_permute2f128_ps(_lo, _hi, 0x31);

                    _mm256_storeu_ps(outptr, _lo_);
                    _mm256_storeu_ps(outptr + 8, _hi_);

                    ptr0 += 16;
                    ptr1 += 16;
                    outptr += 16;
                }
            }

            return 0;
        }
        if (_group > 4 || channels % _group != 0)
        {
            // slow path for too large group or shuffle inside elempack
            Option opt_pack = opt;
            opt_pack.blob_allocator = opt.workspace_allocator;

            Mat bottom_blob_unpacked;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);

            Mat top_blob_unpacked;
            int ret = ShuffleChannel::forward(bottom_blob_unpacked, top_blob_unpacked, opt_pack);
            if (ret != 0)
                return ret;

            convert_packing(top_blob_unpacked, top_blob, elempack, opt);

            return 0;
        }

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
                    __m512 _p0 = _mm512_loadu_ps(ptr0);
                    __m512 _p1 = _mm512_loadu_ps(ptr1);

                    __m512 _lo = _mm512_permutex2var_ps(_p0, _idxlo, _p1);
                    __m512 _hi = _mm512_permutex2var_ps(_p0, _idxhi, _p1);

                    _mm512_storeu_ps(outptr0, _lo);
                    _mm512_storeu_ps(outptr1, _hi);

                    ptr0 += 16;
                    ptr1 += 16;
                    outptr0 += 16;
                    outptr1 += 16;
                }
            }

            return 0;
        }
        if (_group == 3)
        {
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
                    // TODO Naive implementation
                    /*
                    0123456789abcdef        0gw1hx2iy3jz4kA5
                    ghijklmnopqrstuv  --->  lB6mC7nD8oE9pFaq
                    wxyzABCDEFGHIJKL        GbrHcsIdtJeuKfvL
                    */

                    outptr0[0] = ptr0[0];
                    outptr0[1] = ptr1[0];
                    outptr0[2] = ptr2[0];
                    outptr0[3] = ptr0[1];
                    outptr0[4] = ptr1[1];
                    outptr0[5] = ptr2[1];
                    outptr0[6] = ptr0[2];
                    outptr0[7] = ptr1[2];
                    outptr0[8] = ptr2[2];
                    outptr0[9] = ptr0[3];
                    outptr0[10] = ptr1[3];
                    outptr0[11] = ptr2[3];
                    outptr0[12] = ptr0[4];
                    outptr0[13] = ptr1[4];
                    outptr0[14] = ptr2[4];
                    outptr0[15] = ptr0[5];

                    outptr1[0] = ptr1[5];
                    outptr1[1] = ptr2[5];
                    outptr1[2] = ptr0[6];
                    outptr1[3] = ptr1[6];
                    outptr1[4] = ptr2[6];
                    outptr1[5] = ptr0[7];
                    outptr1[6] = ptr1[7];
                    outptr1[7] = ptr2[7];
                    outptr1[8] = ptr0[8];
                    outptr1[9] = ptr1[8];
                    outptr1[10] = ptr2[8];
                    outptr1[11] = ptr0[9];
                    outptr1[12] = ptr1[9];
                    outptr1[13] = ptr2[9];
                    outptr1[14] = ptr0[10];
                    outptr1[15] = ptr1[10];

                    outptr2[0] = ptr2[10];
                    outptr2[1] = ptr0[11];
                    outptr2[2] = ptr1[11];
                    outptr2[3] = ptr2[11];
                    outptr2[4] = ptr0[12];
                    outptr2[5] = ptr1[12];
                    outptr2[6] = ptr2[12];
                    outptr2[7] = ptr0[13];
                    outptr2[8] = ptr1[13];
                    outptr2[9] = ptr2[13];
                    outptr2[10] = ptr0[14];
                    outptr2[11] = ptr1[14];
                    outptr2[12] = ptr2[14];
                    outptr2[13] = ptr0[15];
                    outptr2[14] = ptr1[15];
                    outptr2[15] = ptr2[15];

                    ptr0 += 16;
                    ptr1 += 16;
                    ptr2 += 16;
                    outptr0 += 16;
                    outptr1 += 16;
                    outptr2 += 16;
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
                    __m512 _p0 = _mm512_loadu_ps(ptr0);
                    __m512 _p1 = _mm512_loadu_ps(ptr1);
                    __m512 _p2 = _mm512_loadu_ps(ptr2);
                    __m512 _p3 = _mm512_loadu_ps(ptr3);

                    __m512 _lo02 = _mm512_permutex2var_ps(_p0, _idxlo, _p2);
                    __m512 _hi02 = _mm512_permutex2var_ps(_p0, _idxhi, _p2);
                    __m512 _lo13 = _mm512_permutex2var_ps(_p1, _idxlo, _p3);
                    __m512 _hi13 = _mm512_permutex2var_ps(_p1, _idxhi, _p3);

                    __m512 _lolo = _mm512_permutex2var_ps(_lo02, _idxlo, _lo13);
                    __m512 _lohi = _mm512_permutex2var_ps(_lo02, _idxhi, _lo13);
                    __m512 _hilo = _mm512_permutex2var_ps(_hi02, _idxlo, _hi13);
                    __m512 _hihi = _mm512_permutex2var_ps(_hi02, _idxhi, _hi13);

                    _mm512_storeu_ps(outptr0, _lolo);
                    _mm512_storeu_ps(outptr1, _lohi);
                    _mm512_storeu_ps(outptr2, _hilo);
                    _mm512_storeu_ps(outptr3, _hihi);

                    ptr0 += 16;
                    ptr1 += 16;
                    ptr2 += 16;
                    ptr3 += 16;
                    outptr0 += 16;
                    outptr1 += 16;
                    outptr2 += 16;
                    outptr3 += 16;
                }
            }

            return 0;
        }
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
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

                ptr1 += 4;

                for (int i = 0; i < size; i++)
                {
                    __m256 _p0 = _mm256_loadu_ps(ptr0);

                    __m256 _p1 = _mm256_castps128_ps256(_mm_loadu_ps(ptr1));
                    _p1 = _mm256_insertf128_ps(_p1, _mm_loadu_ps(ptr2), 1);

                    __m256 _lo = _mm256_unpacklo_ps(_p0, _p1);
                    __m256 _hi = _mm256_unpackhi_ps(_p0, _p1);

                    __m256 _lo_ = _mm256_permute2f128_ps(_lo, _hi, 0x20);
                    __m256 _hi_ = _mm256_permute2f128_ps(_lo, _hi, 0x31);

                    _mm256_storeu_ps(outptr0, _lo_);
                    _mm256_storeu_ps(outptr1, _hi_);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                }
            }

            // handle the last channel
            {
                const float* ptr0 = bottom_blob.channel(channels_per_group);
                const float* ptr1 = bottom_blob.channel(channels_per_group * 2);
                float* outptr = top_blob.channel(channels_per_group * 2);

                ptr1 += 4;

                for (int i = 0; i < size; i++)
                {
                    __m128 _p0 = _mm_loadu_ps(ptr0);
                    __m128 _p1 = _mm_loadu_ps(ptr1);

                    __m128 _lo = _mm_unpacklo_ps(_p0, _p1);
                    __m128 _hi = _mm_unpackhi_ps(_p0, _p1);

                    _mm_storeu_ps(outptr, _lo);
                    _mm_storeu_ps(outptr + 4, _hi);

                    ptr0 += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
            }

            return 0;
        }
        if (_group > 4 || channels % _group != 0)
        {
            // slow path for too large group or shuffle inside elempack
            Option opt_pack = opt;
            opt_pack.blob_allocator = opt.workspace_allocator;

            Mat bottom_blob_unpacked;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);

            Mat top_blob_unpacked;
            int ret = ShuffleChannel::forward(bottom_blob_unpacked, top_blob_unpacked, opt_pack);
            if (ret != 0)
                return ret;

            convert_packing(top_blob_unpacked, top_blob, elempack, opt);

            return 0;
        }

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
                    __m256 _p0 = _mm256_loadu_ps(ptr0);
                    __m256 _p1 = _mm256_loadu_ps(ptr1);

                    __m256 _lo = _mm256_unpacklo_ps(_p0, _p1);
                    __m256 _hi = _mm256_unpackhi_ps(_p0, _p1);

                    __m256 _lo_ = _mm256_permute2f128_ps(_lo, _hi, 0x20);
                    __m256 _hi_ = _mm256_permute2f128_ps(_lo, _hi, 0x31);

                    _mm256_storeu_ps(outptr0, _lo_);
                    _mm256_storeu_ps(outptr1, _hi_);

                    ptr0 += 8;
                    ptr1 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                }
            }

            return 0;
        }
        if (_group == 3)
        {
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
                    // TODO figure out a faster way
                    /*
                    01234567        08g19h2a
                    89abcdef  --->  i3bj4ck5
                    ghijklmn        dl6em7fn
                    */

                    __m256 _p0 = _mm256_loadu_ps(ptr0); // 01234567
                    __m256 _p1 = _mm256_loadu_ps(ptr1); // 89abcdef
                    __m256 _p2 = _mm256_loadu_ps(ptr2); // ghijklmn

                    __m256 _08194c5d = _mm256_unpacklo_ps(_p0, _p1);
                    __m256 _2a3b6e7f = _mm256_unpackhi_ps(_p0, _p1);
                    __m256 _8g9hckdl = _mm256_unpacklo_ps(_p1, _p2);
                    __m256 _aibjemfn = _mm256_unpackhi_ps(_p1, _p2);
                    __m256 _0g1h4k5l = _mm256_unpacklo_ps(_p0, _p2);
                    __m256 _2i3j6m7n = _mm256_unpackhi_ps(_p0, _p2);

                    __m256 _i3g1m7k5 = _mm256_shuffle_ps(_2i3j6m7n, _0g1h4k5l, _MM_SHUFFLE(2, 1, 2, 1));

                    __m256 _9h2adl6e = _mm256_shuffle_ps(_8g9hckdl, _2a3b6e7f, _MM_SHUFFLE(1, 0, 3, 2));
                    __m256 _08g14ck5 = _mm256_shuffle_ps(_08194c5d, _i3g1m7k5, _MM_SHUFFLE(3, 2, 1, 0));
                    __m256 _i3bjm7fn = _mm256_shuffle_ps(_i3g1m7k5, _aibjemfn, _MM_SHUFFLE(3, 2, 1, 0));

                    __m256 _08g19h2a = _mm256_permute2f128_ps(_08g14ck5, _9h2adl6e, 0x20); // 0 2
                    __m256 _i3bj4ck5 = _mm256_permute2f128_ps(_i3bjm7fn, _08g14ck5, 0x30); // 0 3
                    __m256 _dl6em7fn = _mm256_permute2f128_ps(_9h2adl6e, _i3bjm7fn, 0x31); // 1 3

                    _mm256_storeu_ps(outptr0, _08g19h2a);
                    _mm256_storeu_ps(outptr1, _i3bj4ck5);
                    _mm256_storeu_ps(outptr2, _dl6em7fn);

                    ptr0 += 8;
                    ptr1 += 8;
                    ptr2 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                    outptr2 += 8;
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
                    __m256 _p0 = _mm256_loadu_ps(ptr0);
                    __m256 _p1 = _mm256_loadu_ps(ptr1);
                    __m256 _p2 = _mm256_loadu_ps(ptr2);
                    __m256 _p3 = _mm256_loadu_ps(ptr3);

                    __m256 _lo02 = _mm256_unpacklo_ps(_p0, _p2);
                    __m256 _hi02 = _mm256_unpackhi_ps(_p0, _p2);
                    __m256 _lo13 = _mm256_unpacklo_ps(_p1, _p3);
                    __m256 _hi13 = _mm256_unpackhi_ps(_p1, _p3);

                    __m256 _lolo = _mm256_unpacklo_ps(_lo02, _lo13);
                    __m256 _lohi = _mm256_unpackhi_ps(_lo02, _lo13);
                    __m256 _hilo = _mm256_unpacklo_ps(_hi02, _hi13);
                    __m256 _hihi = _mm256_unpackhi_ps(_hi02, _hi13);

                    __m256 _lolo_ = _mm256_permute2f128_ps(_lolo, _lohi, 0x20);
                    __m256 _lohi_ = _mm256_permute2f128_ps(_hilo, _hihi, 0x20);
                    __m256 _hilo_ = _mm256_permute2f128_ps(_lolo, _lohi, 0x31);
                    __m256 _hihi_ = _mm256_permute2f128_ps(_hilo, _hihi, 0x31);

                    _mm256_storeu_ps(outptr0, _lolo_);
                    _mm256_storeu_ps(outptr1, _lohi_);
                    _mm256_storeu_ps(outptr2, _hilo_);
                    _mm256_storeu_ps(outptr3, _hihi_);

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
    }
#endif // __AVX__
    if (elempack == 4)
    {
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
                    __m128 _p0 = _mm_loadu_ps(ptr0);
                    __m128 _p1 = _mm_loadu_ps(ptr1);
                    __m128 _p2 = _mm_loadu_ps(ptr2);

                    __m128 _p12 = _mm_shuffle_ps(_p1, _p2, _MM_SHUFFLE(1, 0, 3, 2));

                    __m128 _lo = _mm_unpacklo_ps(_p0, _p12);
                    __m128 _hi = _mm_unpackhi_ps(_p0, _p12);

                    _mm_storeu_ps(outptr0, _lo);
                    _mm_storeu_ps(outptr1, _hi);

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
                    __m128 _p0 = _mm_loadu_ps(ptr0);
                    __m128 _p1 = _mm_loadu_ps(ptr1);

                    __m128 _lo = _mm_unpacklo_ps(_p0, _p1);

                    _mm_storeu_ps(outptr, _lo);

                    ptr0 += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            return 0;
        }
        if (_group > 4 || channels % _group != 0)
        {
            // slow path for too large group or shuffle inside elempack
            Option opt_pack = opt;
            opt_pack.blob_allocator = opt.workspace_allocator;

            Mat bottom_blob_unpacked;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack);

            Mat top_blob_unpacked;
            int ret = ShuffleChannel::forward(bottom_blob_unpacked, top_blob_unpacked, opt_pack);
            if (ret != 0)
                return ret;

            convert_packing(top_blob_unpacked, top_blob, elempack, opt);

            return 0;
        }

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
                    __m128 _p0 = _mm_loadu_ps(ptr0);
                    __m128 _p1 = _mm_loadu_ps(ptr1);

                    __m128 _lo = _mm_unpacklo_ps(_p0, _p1);
                    __m128 _hi = _mm_unpackhi_ps(_p0, _p1);

                    _mm_storeu_ps(outptr0, _lo);
                    _mm_storeu_ps(outptr1, _hi);

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
                    __m128 _p0 = _mm_loadu_ps(ptr0);
                    __m128 _p1 = _mm_loadu_ps(ptr1);
                    __m128 _p2 = _mm_loadu_ps(ptr2);

                    __m128 _0415 = _mm_unpacklo_ps(_p0, _p1);
                    __m128 _2637 = _mm_unpackhi_ps(_p0, _p1);
                    __m128 _4859 = _mm_unpacklo_ps(_p1, _p2);
                    __m128 _6a7b = _mm_unpackhi_ps(_p1, _p2);

                    __m128 _138a = _mm_shuffle_ps(_p0, _p2, _MM_SHUFFLE(2, 0, 3, 1));

                    __m128 _0481 = _mm_shuffle_ps(_0415, _138a, _MM_SHUFFLE(0, 2, 1, 0));
                    __m128 _5926 = _mm_shuffle_ps(_4859, _2637, _MM_SHUFFLE(1, 0, 3, 2));
                    __m128 _a37b = _mm_shuffle_ps(_138a, _6a7b, _MM_SHUFFLE(3, 2, 1, 3));

                    _mm_storeu_ps(outptr0, _0481);
                    _mm_storeu_ps(outptr1, _5926);
                    _mm_storeu_ps(outptr2, _a37b);

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
                    __m128 _p0 = _mm_loadu_ps(ptr0);
                    __m128 _p1 = _mm_loadu_ps(ptr1);
                    __m128 _p2 = _mm_loadu_ps(ptr2);
                    __m128 _p3 = _mm_loadu_ps(ptr3);

                    __m128 _lo02 = _mm_unpacklo_ps(_p0, _p2);
                    __m128 _hi02 = _mm_unpackhi_ps(_p0, _p2);
                    __m128 _lo13 = _mm_unpacklo_ps(_p1, _p3);
                    __m128 _hi13 = _mm_unpackhi_ps(_p1, _p3);

                    __m128 _lolo = _mm_unpacklo_ps(_lo02, _lo13);
                    __m128 _lohi = _mm_unpackhi_ps(_lo02, _lo13);
                    __m128 _hilo = _mm_unpacklo_ps(_hi02, _hi13);
                    __m128 _hihi = _mm_unpackhi_ps(_hi02, _hi13);

                    _mm_storeu_ps(outptr0, _lolo);
                    _mm_storeu_ps(outptr1, _lohi);
                    _mm_storeu_ps(outptr2, _hilo);
                    _mm_storeu_ps(outptr3, _hihi);

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
#endif // __SSE2__

    return ShuffleChannel::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
