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
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

static void rmsnorm_loongarch(float* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

#if __loongarch_asx
    if (elempack == 8)
    {
        __m256 _rms = (__m256)__lasx_xvreplfr2vr_s(0.f);
        const float* ptr0 = ptr;
        for (int i = 0; i < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
            _rms = __lasx_xvfmadd_s(_p, _p, _rms);
            ptr0 += 8;
        }

        float rms_data[8];
        __lasx_xvst(_rms, rms_data, 0);
        for (int i = 0; i < 8; i++)
        {
            rms_data[i] = 1.f / sqrtf(rms_data[i] / elemcount + eps);
        }
        _rms = (__m256)__lasx_xvld(rms_data, 0);

        if (gamma_ptr)
        {
            for (int i = 0; i < size; i += 8)
            {
                __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                __m256 _gamma = (__m256)__lasx_xvreplfr2vr_s(gamma_ptr[0]);
                _p = __lasx_xvfmul_s(_p, _rms);
                _p = __lasx_xvfmul_s(_p, _gamma);
                __lasx_xvst(_p, ptr, 0);
                ptr += 8;
                gamma_ptr += 1;
            }
        }
        else
        {
            for (int i = 0; i < size; i += 8)
            {
                __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                _p = __lasx_xvfmul_s(_p, _rms);
                __lasx_xvst(_p, ptr, 0);
                ptr += 8;
            }
        }

        return;
    }
#endif // __loongarch_asx

#if __loongarch_sx
    if (elempack == 4)
    {
        __m128 _rms = (__m128)__lsx_vreplfr2vr_s(0.f);
        const float* ptr0 = ptr;
        for (int i = 0; i < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0, 0);
            _rms = __lsx_vfmadd_s(_p, _p, _rms);
            ptr0 += 4;
        }

        float rms_data[4];
        __lsx_vst(_rms, rms_data, 0);
        for (int i = 0; i < 4; i++)
        {
            rms_data[i] = 1.f / sqrtf(rms_data[i] / elemcount + eps);
        }
        _rms = (__m128)__lsx_vld(rms_data, 0);

        if (gamma_ptr)
        {
            for (int i = 0; i < size; i += 4)
            {
                __m128 _p = (__m128)__lsx_vld(ptr, 0);
                __m128 _gamma = __lsx_vreplfr2vr_s(gamma_ptr[0]);
                _p = __lsx_vfmul_s(_p, _rms);
                _p = __lsx_vfmul_s(_p, _gamma);
                __lsx_vst(_p, ptr, 0);
                ptr += 4;
                gamma_ptr += 1;
            }
        }
        else
        {
            for (int i = 0; i < size; i += 4)
            {
                __m128 _p = (__m128)__lsx_vld(ptr, 0);
                _p = __lsx_vfmul_s(_p, _rms);
                __lsx_vst(_p, ptr, 0);
                ptr += 4;
            }
        }

        return;
    }
#endif // __loongarch_sx

    float rms = 0.f;
    {
        const float* ptr0 = ptr;
        int i = 0;
#if __loongarch_asx
        __m256 _rms8 = (__m256)__lasx_xvreplfr2vr_s(0.f);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
            _rms8 = __lasx_xvfmadd_s(_p, _p, _rms8);
            ptr0 += 8;
        }
        rms += __lasx_reduce_fadd_s(_rms8);
#endif // __loongarch_asx
#if __loongarch_sx
        __m128 _rms4 = (__m128)__lsx_vreplfr2vr_s(0.f);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr0, 0);
            _rms4 = __lsx_vfmadd_s(_p, _p, _rms4);
            ptr0 += 4;
        }
        rms += __lsx_reduce_fadd_s(_rms4);
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            rms += ptr0[0] * ptr0[0];
            ptr0++;
        }
    }

    rms = 1.f / sqrtf(rms / elemcount + eps);

    if (gamma_ptr)
    {
        int i = 0;
#if __loongarch_asx
        __m256 _rms8 = __lasx_xvreplfr2vr_s(rms);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            __m256 _gamma = (__m256)__lasx_xvld(gamma_ptr, 0);
            _p = __lasx_xvfmul_s(_p, _rms8);
            _p = __lasx_xvfmul_s(_p, _gamma);
            __lasx_xvst(_p, ptr, 0);
            ptr += 8;
            gamma_ptr += 8;
        }
#endif // __loongarch_asx
#if __loongarch_sx
        __m128 _rms4 = __lsx_vreplfr2vr_s(rms);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            __m128 _gamma = (__m128)__lsx_vld(gamma_ptr, 0);
            _p = __lsx_vfmul_s(_p, _rms4);
            _p = __lsx_vfmul_s(_p, _gamma);
            __lsx_vst(_p, ptr, 0);
            ptr += 4;
            gamma_ptr += 4;
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
#if __loongarch_asx
        __m256 _rms8 = __lasx_xvreplfr2vr_s(rms);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = (__m256)__lasx_xvld(ptr, 0);
            _p = __lasx_xvfmul_s(_p, _rms8);
            __lasx_xvst(_p, ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
#if __loongarch_sx
        __m128 _rms4 = __lsx_vreplfr2vr_s(rms);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = (__m128)__lsx_vld(ptr, 0);
            _p = __lsx_vfmul_s(_p, _rms4);
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

    return 0;
}

#if NCNN_BF16
int RMSNorm_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    Option opt_cast = opt;
    opt_cast.blob_allocator = opt.workspace_allocator;

    Mat bottom_top_blob_fp32;
    cast_bfloat16_to_float32(bottom_top_blob, bottom_top_blob_fp32, opt_cast);
    if (bottom_top_blob_fp32.empty())
        return -100;

    int ret = forward_inplace(bottom_top_blob_fp32, opt);
    if (ret != 0)
        return ret;

    cast_float32_to_bfloat16(bottom_top_blob_fp32, bottom_top_blob, opt);
    if (bottom_top_blob.empty())
        return -100;

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
