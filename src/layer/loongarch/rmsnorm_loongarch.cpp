// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rmsnorm_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

RMSNorm_loongarch::RMSNorm_loongarch()
{
#if __loongarch_sx
    support_packing = true;
    support_any_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static void rmsnorm_loongarch(float* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // compute rms
#if __loongarch_sx
#if __loongarch_asx
    __m256 _rms_lasx = (__m256)__lasx_xvreplfr2vr_s(0.f);
#endif // __loongarch_asx
#endif // __loongarch_sx
#if __loongarch_sx
    __m128 _rms = (__m128)__lsx_vreplfr2vr_s(0.f);
#endif // __loongarch_sx
    float rms = 0.f;
    {
        const float* ptr0 = ptr;
        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
            _rms_lasx = __lasx_xvfmadd_s(_p, _p, _rms_lasx);
            ptr0 += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0, 0);
            _rms = __lsx_vfmadd_s(_p, _p, _rms);
            ptr0 += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            rms += ptr0[0] * ptr0[0];
            ptr0++;
        }
    }

#if __loongarch_sx
#if __loongarch_asx
    if (elempack == 8)
    {
        __m256 _elemcount = (__m256)__lasx_xvreplfr2vr_s((float)elemcount);
        __m256 _eps = (__m256)__lasx_xvreplfr2vr_s(eps);
        _rms_lasx = __lasx_xvfdiv_s(_rms_lasx, _elemcount);
        _rms_lasx = __lasx_xvfadd_s(_rms_lasx, _eps);
        _rms_lasx = __lasx_xvfrsqrt_s(_rms_lasx);
    }
#endif // __loongarch_asx
    if (elempack == 4)
    {
#if __loongarch_asx
        {
            __m128 _rms0 = __lasx_extract_128_lo_s(_rms_lasx);
            __m128 _rms1 = __lasx_extract_128_hi_s(_rms_lasx);
            _rms = __lsx_vfadd_s(_rms, _rms0);
            _rms = __lsx_vfadd_s(_rms, _rms1);
        }
#endif // __loongarch_asx
        __m128 _elemcount = (__m128)__lsx_vreplfr2vr_s((float)elemcount);
        __m128 _eps = (__m128)__lsx_vreplfr2vr_s(eps);
        _rms = __lsx_vfdiv_s(_rms, _elemcount);
        _rms = __lsx_vfadd_s(_rms, _eps);
        _rms = __lsx_vfrsqrt_s(_rms);
#if __loongarch_asx
        _rms_lasx = __lasx_concat_128_s(_rms, _rms);
#endif // __loongarch_asx
    }
#endif // __loongarch_sx
    if (elempack == 1)
    {
#if __loongarch_sx
#if __loongarch_asx
        rms += __lasx_reduce_fadd_s(_rms_lasx);
#endif // __loongarch_asx
        rms += __lsx_reduce_fadd_s(_rms);
#endif // __loongarch_sx

        rms = 1.f / sqrtf(rms / elemcount + eps);
#if __loongarch_sx
        _rms = __lsx_vreplfr2vr_s(rms);
#if __loongarch_asx
        _rms_lasx = __lasx_xvreplfr2vr_s(rms);
#endif // __loongarch_asx
#endif // __loongarch_sx
    }
    if (gamma_ptr)
    {
        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                __m256 _gamma = (__m256)__lasx_xvreplfr2vr_s(gamma_ptr[0]);
                _p = __lasx_xvfmul_s(_p, _rms_lasx);
                _p = __lasx_xvfmul_s(_p, _gamma);
                __lasx_xvst(_p, ptr, 0);
                ptr += 8;
                gamma_ptr += 1;
            }
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
#if __loongarch_asx
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                __m128 _gamma0 = (__m128)__lsx_vreplfr2vr_s(gamma_ptr[0]);
                __m128 _gamma1 = (__m128)__lsx_vreplfr2vr_s(gamma_ptr[1]);
                __m256 _gamma_lasx = __lasx_concat_128_s(_gamma0, _gamma1);
                _p = __lasx_xvfmul_s(_p, _rms_lasx);
                _p = __lasx_xvfmul_s(_p, _gamma_lasx);
                __lasx_xvst(_p, ptr, 0);
                ptr += 8;
                gamma_ptr += 2;
            }
#endif // __loongarch_asx
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = (__m128)__lsx_vld(ptr, 0);
                __m128 _gamma = (__m128)__lsx_vreplfr2vr_s(gamma_ptr[0]);
                _p = __lsx_vfmul_s(_p, _rms);
                _p = __lsx_vfmul_s(_p, _gamma);
                __lsx_vst(_p, ptr, 0);
                ptr += 4;
                gamma_ptr += 1;
            }
        }
        if (elempack == 1)
        {
#if __loongarch_asx
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                __m256 _gamma = (__m256)__lasx_xvld(gamma_ptr, 0);
                _p = __lasx_xvfmul_s(_p, _rms_lasx);
                _p = __lasx_xvfmul_s(_p, _gamma);
                __lasx_xvst(_p, ptr, 0);
                ptr += 8;
                gamma_ptr += 8;
            }
#endif // __loongarch_asx
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = (__m128)__lsx_vld(ptr, 0);
                __m128 _gamma = (__m128)__lsx_vld(gamma_ptr, 0);
                _p = __lsx_vfmul_s(_p, _rms);
                _p = __lsx_vfmul_s(_p, _gamma);
                __lsx_vst(_p, ptr, 0);
                ptr += 4;
                gamma_ptr += 4;
            }
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            ptr[0] = (ptr[0] * rms) * gamma_ptr[0];
            ptr++;
            gamma_ptr++;
        }
    }
    else
    {
        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            _p = __lasx_xvfmul_s(_p, _rms_lasx);
            __lasx_xvst(_p, ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            _p = __lsx_vfmul_s(_p, _rms);
            __lsx_vst(_p, ptr, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            ptr[0] = ptr[0] * rms;
            ptr++;
        }
    }
}

int RMSNorm_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        rmsnorm_loongarch(ptr, gamma_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            rmsnorm_loongarch(ptr, gamma_data, eps, w, elempack);
        }
    }

    if (dims == 3)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
                    rmsnorm_loongarch(ptr, gamma_data, eps, w, elempack);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                rmsnorm_loongarch(ptr, gamma_data, eps, w * h, elempack);
            }
        }
    }

    if (dims == 4)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float* ptr = bottom_top_blob.channel(q).depth(z).row(i);
                        rmsnorm_loongarch(ptr, gamma_data, eps, w, elempack);
                    }
                }
            }
        }
        else if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    float* ptr = bottom_top_blob.channel(q).depth(z);
                    rmsnorm_loongarch(ptr, gamma_data, eps, w * h, elempack);
                }
            }
        }
        else // if (affine_size == w * h * d)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                rmsnorm_loongarch(ptr, gamma_data, eps, w * h * d, elempack);
            }
        }
    }

    return 0;
}

#if NCNN_BF16
static void rmsnorm_loongarch_bf16(unsigned short* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

    // compute rms
#if __loongarch_sx
#if __loongarch_asx
    __m256 _rms_lasx = (__m256)__lasx_xvreplfr2vr_s(0.f);
#endif // __loongarch_asx
    __m128 _rms = (__m128)__lsx_vreplfr2vr_s(0.f);
#if !__loongarch_asx
    __m128 _rms1 = (__m128)__lsx_vreplfr2vr_s(0.f);
#endif // !__loongarch_asx
#endif // __loongarch_sx
    float rms = 0.f;
    {
        const unsigned short* ptr0 = ptr;
        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i*)ptr0);
            _rms_lasx = __lasx_xvfmadd_s(_p, _p, _rms_lasx);
            ptr0 += 8;
        }
#else
        __m128i _zero_bf16 = __lsx_vreplgr2vr_w(0);
        for (; i + 7 < size; i += 8)
        {
            __m128i _p01 = __lsx_vld(ptr0, 0);
            __m128 _p0 = (__m128)__lsx_vilvl_h(_p01, _zero_bf16);
            __m128 _p1 = (__m128)__lsx_vilvh_h(_p01, _zero_bf16);
            _rms = __lsx_vfmadd_s(_p0, _p0, _rms);
            _rms1 = __lsx_vfmadd_s(_p1, _p1, _rms1);
            ptr0 += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx(__lsx_vldrepl_d(ptr0, 0));
            _rms = __lsx_vfmadd_s(_p, _p, _rms);
            ptr0 += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(ptr0[0]);
            rms += v * v;
            ptr0++;
        }
    }

#if __loongarch_sx
#if __loongarch_asx
    if (elempack == 8)
    {
        __m256 _elemcount = (__m256)__lasx_xvreplfr2vr_s((float)elemcount);
        __m256 _eps = (__m256)__lasx_xvreplfr2vr_s(eps);
        _rms_lasx = __lasx_xvfdiv_s(_rms_lasx, _elemcount);
        _rms_lasx = __lasx_xvfadd_s(_rms_lasx, _eps);
        _rms_lasx = __lasx_xvfrsqrt_s(_rms_lasx);
    }
#else
    if (elempack == 8)
    {
        __m128 _elemcount = (__m128)__lsx_vreplfr2vr_s((float)elemcount);
        __m128 _eps = (__m128)__lsx_vreplfr2vr_s(eps);
        _rms = __lsx_vfdiv_s(_rms, _elemcount);
        _rms1 = __lsx_vfdiv_s(_rms1, _elemcount);
        _rms = __lsx_vfadd_s(_rms, _eps);
        _rms1 = __lsx_vfadd_s(_rms1, _eps);
        _rms = __lsx_vfrsqrt_s(_rms);
        _rms1 = __lsx_vfrsqrt_s(_rms1);
    }
#endif // __loongarch_asx
    if (elempack == 4)
    {
#if __loongarch_asx
        {
            __m128 _rms0 = __lasx_extract_128_lo_s(_rms_lasx);
            __m128 _rms1 = __lasx_extract_128_hi_s(_rms_lasx);
            _rms = __lsx_vfadd_s(_rms, _rms0);
            _rms = __lsx_vfadd_s(_rms, _rms1);
        }
#else
        _rms = __lsx_vfadd_s(_rms, _rms1);
#endif // __loongarch_asx

        __m128 _elemcount = (__m128)__lsx_vreplfr2vr_s((float)elemcount);
        __m128 _eps = (__m128)__lsx_vreplfr2vr_s(eps);
        _rms = __lsx_vfdiv_s(_rms, _elemcount);
        _rms = __lsx_vfadd_s(_rms, _eps);
        _rms = __lsx_vfrsqrt_s(_rms);
#if __loongarch_asx
        _rms_lasx = __lasx_concat_128_s(_rms, _rms);
#else
        _rms1 = _rms;
#endif // __loongarch_asx
    }
#endif // __loongarch_sx
    if (elempack == 1)
    {
#if __loongarch_sx
#if __loongarch_asx
        rms += __lasx_reduce_fadd_s(_rms_lasx);
#else
        rms += __lsx_reduce_fadd_s(_rms1);
#endif // __loongarch_asx
        rms += __lsx_reduce_fadd_s(_rms);
#endif // __loongarch_sx

        rms = 1.f / sqrtf(rms / elemcount + eps);
#if __loongarch_sx
        _rms = __lsx_vreplfr2vr_s(rms);
#if __loongarch_asx
        _rms_lasx = __lasx_xvreplfr2vr_s(rms);
#else
        _rms1 = _rms;
#endif // __loongarch_asx
#endif // __loongarch_sx
    }
    if (gamma_ptr)
    {
        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_lasx((__m128i*)ptr);
                __m256 _gamma = (__m256)__lasx_xvreplfr2vr_s(gamma_ptr[0]);
                _p = __lasx_xvfmul_s(_p, _rms_lasx);
                _p = __lasx_xvfmul_s(_p, _gamma);
                __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
                ptr += 8;
                gamma_ptr += 1;
            }
        }
#else
        __m128i _zero_bf16 = __lsx_vreplgr2vr_w(0);
        if (elempack == 8)
        {
            for (; i + 7 < size; i += 8)
            {
                __m128i _p01 = __lsx_vld(ptr, 0);
                __m128 _p0 = (__m128)__lsx_vilvl_h(_p01, _zero_bf16);
                __m128 _p1 = (__m128)__lsx_vilvh_h(_p01, _zero_bf16);
                __m128 _gamma = (__m128)__lsx_vreplfr2vr_s(gamma_ptr[0]);
                _p0 = __lsx_vfmul_s(_p0, _rms);
                _p1 = __lsx_vfmul_s(_p1, _rms1);
                _p0 = __lsx_vfmul_s(_p0, _gamma);
                _p1 = __lsx_vfmul_s(_p1, _gamma);
                __lsx_vst(float2bfloat_lsx(_p0, _p1), ptr, 0);
                ptr += 8;
                gamma_ptr += 1;
            }
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
#if __loongarch_asx
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_lasx((__m128i*)ptr);
                __m128 _gamma0 = (__m128)__lsx_vreplfr2vr_s(gamma_ptr[0]);
                __m128 _gamma1 = (__m128)__lsx_vreplfr2vr_s(gamma_ptr[1]);
                __m256 _gamma_lasx = __lasx_concat_128_s(_gamma0, _gamma1);
                _p = __lasx_xvfmul_s(_p, _rms_lasx);
                _p = __lasx_xvfmul_s(_p, _gamma_lasx);
                __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
                ptr += 8;
                gamma_ptr += 2;
            }
#endif // __loongarch_asx
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_lsx(__lsx_vldrepl_d(ptr, 0));
                __m128 _gamma = (__m128)__lsx_vreplfr2vr_s(gamma_ptr[0]);
                _p = __lsx_vfmul_s(_p, _rms);
                _p = __lsx_vfmul_s(_p, _gamma);
                __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
                ptr += 4;
                gamma_ptr += 1;
            }
        }
        if (elempack == 1)
        {
#if __loongarch_asx
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_lasx((__m128i*)ptr);
                __m256 _gamma = (__m256)__lasx_xvld(gamma_ptr, 0);
                _p = __lasx_xvfmul_s(_p, _rms_lasx);
                _p = __lasx_xvfmul_s(_p, _gamma);
                __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
                ptr += 8;
                gamma_ptr += 8;
            }
#endif // __loongarch_asx
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_lsx(__lsx_vldrepl_d(ptr, 0));
                __m128 _gamma = (__m128)__lsx_vld(gamma_ptr, 0);
                _p = __lsx_vfmul_s(_p, _rms);
                _p = __lsx_vfmul_s(_p, _gamma);
                __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
                ptr += 4;
                gamma_ptr += 4;
            }
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            ptr[0] = float32_to_bfloat16(bfloat16_to_float32(ptr[0]) * rms * gamma_ptr[0]);
            ptr++;
            gamma_ptr++;
        }
    }
    else
    {
        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_lasx((__m128i*)ptr);
            _p = __lasx_xvfmul_s(_p, _rms_lasx);
            __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
            ptr += 8;
        }
#else
        __m128i _zero_bf16 = __lsx_vreplgr2vr_w(0);
        for (; i + 7 < size; i += 8)
        {
            __m128i _p01 = __lsx_vld(ptr, 0);
            __m128 _p0 = (__m128)__lsx_vilvl_h(_p01, _zero_bf16);
            __m128 _p1 = (__m128)__lsx_vilvh_h(_p01, _zero_bf16);
            _p0 = __lsx_vfmul_s(_p0, _rms);
            _p1 = __lsx_vfmul_s(_p1, _rms1);
            __lsx_vst(float2bfloat_lsx(_p0, _p1), ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_lsx(__lsx_vldrepl_d(ptr, 0));
            _p = __lsx_vfmul_s(_p, _rms);
            __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            ptr[0] = float32_to_bfloat16(bfloat16_to_float32(ptr[0]) * rms);
            ptr++;
        }
    }
}

int RMSNorm_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        unsigned short* ptr = (unsigned short*)bottom_top_blob;
        rmsnorm_loongarch_bf16(ptr, gamma_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = (unsigned short*)bottom_top_blob.row(i);
            rmsnorm_loongarch_bf16(ptr, gamma_data, eps, w, elempack);
        }
    }

    if (dims == 3)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob.channel(q).row(i);
                    rmsnorm_loongarch_bf16(ptr, gamma_data, eps, w, elempack);
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr = (unsigned short*)bottom_top_blob.channel(q);
                rmsnorm_loongarch_bf16(ptr, gamma_data, eps, w * h, elempack);
            }
        }
    }

    if (dims == 4)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        unsigned short* ptr = (unsigned short*)bottom_top_blob.channel(q).depth(z).row(i);
                        rmsnorm_loongarch_bf16(ptr, gamma_data, eps, w, elempack);
                    }
                }
            }
        }
        else if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int z = 0; z < d; z++)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob.channel(q).depth(z);
                    rmsnorm_loongarch_bf16(ptr, gamma_data, eps, w * h, elempack);
                }
            }
        }
        else // if (affine_size == w * h * d)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                unsigned short* ptr = (unsigned short*)bottom_top_blob.channel(q);
                rmsnorm_loongarch_bf16(ptr, gamma_data, eps, w * h * d, elempack);
            }
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
