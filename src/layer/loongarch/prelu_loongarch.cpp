// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "prelu_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

PReLU_loongarch::PReLU_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int PReLU_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        int w = bottom_top_blob.w * elempack;

#if __loongarch_sx
        int nn_w = w / 4;
        int remain_w_start = nn_w * 4;
#else
        int remain_w_start = 0;
#endif // __loongarch_sx

        float* ptr = bottom_top_blob;

        if (num_slope > 1)
        {
            const float* slope = slope_data;

#if __loongarch_sx
#if __loongarch_asx
            int nn_w8 = w / 8;
            int remain_w_start8 = nn_w8 * 8;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < nn_w8; i++)
            {
                float* ptr0 = ptr + i * 8;

                __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
                __m256 _zero = (__m256)__lasx_xvreplgr2vr_w(0);
                __m256 _slope = (__m256)__lasx_xvld(slope + i * 8, 0);
                __m256i _lemask = __lasx_xvfcmp_cle_s(_p, _zero);
                __m256 _ps = __lasx_xvfmul_s(_p, _slope);
                _p = (__m256)__lasx_xvbitsel_v((__m256i)_p, (__m256i)_ps, (__m256i)_lemask);
                __lasx_xvst(_p, ptr0, 0);
            }
#endif // __loongarch_asx
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = remain_w_start8; i < nn_w; i++)
            {
                float* ptr0 = ptr + i * 4;

                __m128 _p = (__m128)__lsx_vld(ptr0, 0);
                __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _slope = (__m128)__lsx_vld(slope + i * 4, 0);
                __m128i _lemask = __lsx_vfcmp_cle_s(_p, _zero);
                __m128 _ps = __lsx_vfmul_s(_p, _slope);
                _p = (__m128)__lsx_vbitsel_v((__m128i)_p, (__m128i)_ps, (__m128i)_lemask);
                __lsx_vst(_p, ptr0, 0);
            }
#endif // __loongarch_sx

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = remain_w_start; i < w; i++)
            {
                float v = ptr[i];
                if (v < 0.f)
                    ptr[i] = v * slope[i];
            }
        }
        else
        {
            const float slope = slope_data[0];

#if __loongarch_sx
#if __loongarch_asx
            int nn_w8 = w / 8;
            int remain_w_start8 = nn_w8 * 8;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < nn_w8; i++)
            {
                float* ptr0 = ptr + i * 8;

                __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
                __m256 _zero = (__m256)__lasx_xvreplgr2vr_w(0);
                __m256 _slope = (__m256)__lasx_xvreplfr2vr_s(slope);
                __m256i _lemask = __lasx_xvfcmp_cle_s(_p, _zero);
                __m256 _ps = __lasx_xvfmul_s(_p, _slope);
                _p = (__m256)__lasx_xvbitsel_v((__m256i)_p, (__m256i)_ps, (__m256i)_lemask);
                __lasx_xvst(_p, ptr0, 0);
            }
#endif // __loongarch_asx
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = remain_w_start8; i < nn_w; i++)
            {
                float* ptr0 = ptr + i * 4;

                __m128 _p = (__m128)__lsx_vld(ptr0, 0);
                __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _slope = (__m128)__lsx_vreplfr2vr_s(slope);
                __m128i _lemask = __lsx_vfcmp_cle_s(_p, _zero);
                __m128 _ps = __lsx_vfmul_s(_p, _slope);
                _p = (__m128)__lsx_vbitsel_v((__m128i)_p, (__m128i)_ps, (__m128i)_lemask);
                __lsx_vst(_p, ptr0, 0);
            }
#endif // __loongarch_sx

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = remain_w_start; i < w; i++)
            {
                float v = ptr[i];
                if (v < 0.f)
                    ptr[i] = v * slope;
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w * elempack;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            const float slope = num_slope > 1 ? slope_data[i] : slope_data[0];

            int j = 0;
#if __loongarch_sx
#if __loongarch_asx
            __m256 _zero8 = (__m256)__lasx_xvreplgr2vr_w(0);
            __m256 _slope8 = (elempack == 4 && num_slope > 1) ? (__m256)__lasx_xvld((const float*)slope_data + i * 4, 0) : (__m256)__lasx_xvreplfr2vr_s(slope);
            int nn_w8 = w / 8;
            int remain_w_start8 = nn_w8 * 8;
            for (; j < remain_w_start8; j += 8)
            {
                __builtin_prefetch(ptr + j + 32);
                __m256 _p = (__m256)__lasx_xvld(ptr + j, 0);
                __m256i _lemask = __lasx_xvfcmp_cle_s(_p, _zero8);
                __m256 _ps = __lasx_xvfmul_s(_p, _slope8);
                _p = (__m256)__lasx_xvbitsel_v((__m256i)_p, (__m256i)_ps, (__m256i)_lemask);
                __lasx_xvst(_p, ptr + j, 0);
            }
#endif // __loongarch_asx
            __m128 _zero4 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _slope4 = (elempack == 4 && num_slope > 1) ? (__m128)__lsx_vld((const float*)slope_data + i * 4, 0) : (__m128)__lsx_vreplfr2vr_s(slope);
            int nn_w4 = w / 4;
            int remain_w_start4 = nn_w4 * 4;
            for (; j < remain_w_start4; j += 4)
            {
                __builtin_prefetch(ptr + j + 16);
                __m128 _p = (__m128)__lsx_vld(ptr + j, 0);
                __m128i _lemask = __lsx_vfcmp_cle_s(_p, _zero4);
                __m128 _ps = __lsx_vfmul_s(_p, _slope4);
                _p = (__m128)__lsx_vbitsel_v((__m128i)_p, (__m128i)_ps, (__m128i)_lemask);
                __lsx_vst(_p, ptr + j, 0);
            }
#endif // __loongarch_sx
            for (; j < w; j++)
            {
                float v = ptr[j];
                if (v < 0.f)
                    ptr[j] = v * slope;
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h * elempack;

        const float* slope_data_ptr = slope_data;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float slope = num_slope > 1 ? slope_data_ptr[q] : slope_data_ptr[0];

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            __m256 _zero8 = (__m256)__lasx_xvreplgr2vr_w(0);
            __m256 _slope8 = (elempack == 4 && num_slope > 1) ? (__m256)__lasx_xvld((const float*)slope_data + q * 4, 0) : (__m256)__lasx_xvreplfr2vr_s(slope);
            int nn_size8 = size / 8;
            int remain_size_start8 = nn_size8 * 8;
            for (; i < remain_size_start8; i += 8)
            {
                __builtin_prefetch(ptr + i + 32);
                __m256 _p = (__m256)__lasx_xvld(ptr + i, 0);
                __m256i _lemask = __lasx_xvfcmp_cle_s(_p, _zero8);
                __m256 _ps = __lasx_xvfmul_s(_p, _slope8);
                _p = (__m256)__lasx_xvbitsel_v((__m256i)_p, (__m256i)_ps, (__m256i)_lemask);
                __lasx_xvst(_p, ptr + i, 0);
            }
#endif // __loongarch_asx
            __m128 _zero4 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _slope4 = (elempack == 4 && num_slope > 1) ? (__m128)__lsx_vld((const float*)slope_data + q * 4, 0) : (__m128)__lsx_vreplfr2vr_s(slope);
            int nn_size4 = size / 4;
            int remain_size_start4 = nn_size4 * 4;
            for (; i < remain_size_start4; i += 4)
            {
                __builtin_prefetch(ptr + i + 16);
                __m128 _p = (__m128)__lsx_vld(ptr + i, 0);
                __m128i _lemask = __lsx_vfcmp_cle_s(_p, _zero4);
                __m128 _ps = __lsx_vfmul_s(_p, _slope4);
                _p = (__m128)__lsx_vbitsel_v((__m128i)_p, (__m128i)_ps, (__m128i)_lemask);
                __lsx_vst(_p, ptr + i, 0);
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr *= slope;

                ptr++;
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int PReLU_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        int w = bottom_top_blob.w * elempack;

        unsigned short* ptr = (unsigned short*)bottom_top_blob;

        if (num_slope > 1)
        {
            const float* slope = slope_data;

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            for (; i + 7 < w; i += 8)
            {
                __m256 _p = bfloat2float_avx((__m128i*)(ptr + i));
                __m256 _zero = (__m256)__lasx_xvreplgr2vr_w(0);
                __m256 _slope = (__m256)__lasx_xvld(slope + i, 0);
                __m256i _lemask = __lasx_xvfcmp_cle_s(_p, _zero);
                __m256 _ps = __lasx_xvfmul_s(_p, _slope);
                _p = (__m256)__lasx_xvbitsel_v((__m256i)_p, (__m256i)_ps, (__m256i)_lemask);
                __lasx_xvst(float2bfloat_avx(_p), ptr + i, 0);
            }
#endif // __loongarch_asx
            for (; i + 3 < w; i += 4)
            {
                __m128 _p = bfloat2float_sse((__m128i*)(ptr + i));
                __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _slope = (__m128)__lsx_vld(slope + i, 0);
                __m128i _lemask = __lsx_vfcmp_cle_s(_p, _zero);
                __m128 _ps = __lsx_vfmul_s(_p, _slope);
                _p = (__m128)__lsx_vbitsel_v((__m128i)_p, (__m128i)_ps, (__m128i)_lemask);
                __lsx_vst(float2bfloat_sse(_p), ptr + i, 0);
            }
#endif // __loongarch_sx
            for (; i < w; i++)
            {
                float v = bfloat16_to_float32(ptr[i]);
                if (v < 0.f)
                    ptr[i] = float32_to_bfloat16(v * slope[i]);
                else
                    ptr[i] = float32_to_bfloat16(v);
            }
        }
        else
        {
            const float slope = slope_data[0];

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            for (; i + 7 < w; i += 8)
            {
                __m256 _p = bfloat2float_avx((__m128i*)(ptr + i));
                __m256 _zero = (__m256)__lasx_xvreplgr2vr_w(0);
                __m256 _slope = (__m256)__lasx_xvreplfr2vr_s(slope);
                __m256i _lemask = __lasx_xvfcmp_cle_s(_p, _zero);
                __m256 _ps = __lasx_xvfmul_s(_p, _slope);
                _p = (__m256)__lasx_xvbitsel_v((__m256i)_p, (__m256i)_ps, (__m256i)_lemask);
                __lasx_xvst(float2bfloat_avx(_p), ptr + i, 0);
            }
#endif // __loongarch_asx
            for (; i + 3 < w; i += 4)
            {
                __m128 _p = bfloat2float_sse((__m128i*)(ptr + i));
                __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
                __m128 _slope = (__m128)__lsx_vreplfr2vr_s(slope);
                __m128i _lemask = __lsx_vfcmp_cle_s(_p, _zero);
                __m128 _ps = __lsx_vfmul_s(_p, _slope);
                _p = (__m128)__lsx_vbitsel_v((__m128i)_p, (__m128i)_ps, (__m128i)_lemask);
                __lsx_vst(float2bfloat_sse(_p), ptr + i, 0);
            }
#endif // __loongarch_sx
            for (; i < w; i++)
            {
                float v = bfloat16_to_float32(ptr[i]);
                if (v < 0.f)
                    ptr[i] = float32_to_bfloat16(v * slope);
                else
                    ptr[i] = float32_to_bfloat16(v);
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w * elempack;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);

            const float slope = num_slope > 1 ? slope_data[i] : slope_data[0];

            int j = 0;
#if __loongarch_sx
#if __loongarch_asx
            __m256 _zero8 = (__m256)__lasx_xvreplgr2vr_w(0);
            __m256 _slope8 = (elempack == 4 && num_slope > 1) ? (__m256)__lasx_xvld((const float*)slope_data + i * 4, 0) : (__m256)__lasx_xvreplfr2vr_s(slope);
            for (; j + 7 < w; j += 8)
            {
                __m256 _p = bfloat2float_avx((__m128i*)(ptr + j));
                __m256i _lemask = __lasx_xvfcmp_cle_s(_p, _zero8);
                __m256 _ps = __lasx_xvfmul_s(_p, _slope8);
                _p = (__m256)__lasx_xvbitsel_v((__m256i)_p, (__m256i)_ps, (__m256i)_lemask);
                __lasx_xvst(float2bfloat_avx(_p), ptr + j, 0);
            }
#endif // __loongarch_asx
            __m128 _zero4 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _slope4 = (elempack == 4 && num_slope > 1) ? (__m128)__lsx_vld((const float*)slope_data + i * 4, 0) : (__m128)__lsx_vreplfr2vr_s(slope);
            for (; j + 3 < w; j += 4)
            {
                __m128 _p = bfloat2float_sse((__m128i*)(ptr + j));
                __m128i _lemask = __lsx_vfcmp_cle_s(_p, _zero4);
                __m128 _ps = __lsx_vfmul_s(_p, _slope4);
                _p = (__m128)__lsx_vbitsel_v((__m128i)_p, (__m128i)_ps, (__m128i)_lemask);
                __lsx_vst(float2bfloat_sse(_p), ptr + j, 0);
            }
#endif // __loongarch_sx
            for (; j < w; j++)
            {
                float v = bfloat16_to_float32(ptr[j]);
                if (v < 0.f)
                    ptr[j] = float32_to_bfloat16(v * slope);
                else
                    ptr[j] = float32_to_bfloat16(v);
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h * elempack;

        const float* slope_data_ptr = slope_data;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel<unsigned short>(q);
            float slope = num_slope > 1 ? slope_data_ptr[q] : slope_data_ptr[0];

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            __m256 _zero8 = (__m256)__lasx_xvreplgr2vr_w(0);
            __m256 _slope8 = (elempack == 4 && num_slope > 1) ? (__m256)__lasx_xvld((const float*)slope_data + q * 4, 0) : (__m256)__lasx_xvreplfr2vr_s(slope);
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx((__m128i*)(ptr + i));
                __m256i _lemask = __lasx_xvfcmp_cle_s(_p, _zero8);
                __m256 _ps = __lasx_xvfmul_s(_p, _slope8);
                _p = (__m256)__lasx_xvbitsel_v((__m256i)_p, (__m256i)_ps, (__m256i)_lemask);
                __lasx_xvst(float2bfloat_avx(_p), ptr + i, 0);
            }
#endif // __loongarch_asx
            __m128 _zero4 = (__m128)__lsx_vreplgr2vr_w(0);
            __m128 _slope4 = (elempack == 4 && num_slope > 1) ? (__m128)__lsx_vld((const float*)slope_data + q * 4, 0) : (__m128)__lsx_vreplfr2vr_s(slope);
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse((__m128i*)(ptr + i));
                __m128i _lemask = __lsx_vfcmp_cle_s(_p, _zero4);
                __m128 _ps = __lsx_vfmul_s(_p, _slope4);
                _p = (__m128)__lsx_vbitsel_v((__m128i)_p, (__m128i)_ps, (__m128i)_lemask);
                __lsx_vst(float2bfloat_sse(_p), ptr + i, 0);
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                unsigned short* p = ptr + i;
                float v = bfloat16_to_float32(*p);
                if (v < 0)
                    *p = float32_to_bfloat16(v * slope);
                else
                    *p = float32_to_bfloat16(v);
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
