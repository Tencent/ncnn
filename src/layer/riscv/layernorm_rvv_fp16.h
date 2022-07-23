// Xavier Hsinyuan is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 Xavier Hsinyuan <me@lstlx.com>. All rights reserved.
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

// fp16s
static inline int layernorm_rvv_pack1_fp16s_procedure(int size, __fp16* ptr, const float* gamma_data, const float* beta_data, float eps, int affine)
{
    float sum = 0.f;
    float sqsum = 0.f;
    vfloat32m1_t _sum = vfmv_s_f_f32m1(vundefined_f32m1(), 0.f, vsetvlmax_e32m1());
    vfloat32m1_t _sqsum = vfmv_s_f_f32m1(vundefined_f32m1(), 0.f, vsetvlmax_e32m1());
    {
        int n = size;
        __fp16* ptr_sum = ptr;
        while (n > 0)
        {
            word_type vl = vsetvl_e16m4(n);
            vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr_sum, vl), vl);
            _sum = vfredusum_vs_f32m8_f32m1(_sum, _p, /* scalar */ _sum, vl);
            // _sqsum = vfredusum_vs_f32m8_f32m1(_sqsum, vfmul_vv_f32m8(_p, _p, vl), /* scalar */ _sqsum, vl);
            ptr_sum += vl;
            n -= vl;
        }
    }
    sum = vfmv_f_s_f32m1_f32(_sum);
    float mean = sum / size;

    {
        int n = size;
        __fp16* ptr_sqsum = ptr;
        while (n > 0)
        {
            word_type vl = vsetvl_e16m4(n);
            vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr_sqsum, vl), vl);
            _p = vfsub_vf_f32m8(_p, mean, vl);
            _sqsum = vfredusum_vs_f32m8_f32m1(_sqsum, vfmul_vv_f32m8(_p, _p, vl), /* scalar */ _sqsum, vl);
            n -= vl;
            ptr_sqsum += vl;
        }
    }
    sqsum = vfmv_f_s_f32m1_f32(_sqsum);
    float var = sqsum / size;
    // the var maybe minus due to accuracy
    //float var = sqsum / size - mean * mean;
    float a = static_cast<float>(1.f / (sqrt(var + eps)));
    float b = -mean * a;

    {
        int n = size;
        __fp16* ptr_store = ptr;
        const float* ptr_gamma = gamma_data;
        const float* ptr_beta = beta_data;
        if (affine)
        {
            while (n > 0)
            {
                word_type vl = vsetvl_e16m4(n);
                vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr_store, vl), vl);
                _p = vfmul_vf_f32m8(_p, a, vl);
                vfloat32m8_t _gamma = vle32_v_f32m8(ptr_gamma, vl);
                _p = vfadd_vf_f32m8(_p, b, vl);
                vfloat32m8_t _beta = vle32_v_f32m8(ptr_beta, vl);
                _p = vfmadd_vv_f32m8(_p, _gamma, _beta, vl);
                vse16_v_f16m4(ptr_store, vfncvt_f_f_w_f16m4(_p, vl), vl);

                n -= vl;
                ptr_store += vl;
                ptr_gamma += vl;
                ptr_beta += vl;
            }
        }
        else
        {
            while (n > 0)
            {
                word_type vl = vsetvl_e16m4(n);
                vfloat32m8_t _p = vfwcvt_f_f_v_f32m8(vle16_v_f16m4(ptr_store, vl), vl);
                _p = vfmul_vf_f32m8(_p, a, vl);
                _p = vfadd_vf_f32m8(_p, b, vl);
                vse16_v_f16m4(ptr_store, vfncvt_f_f_w_f16m4(_p, vl), vl);
                n -= vl;
                ptr_store += vl;
            }
        }
    }
    return 0;
}

static inline int layernorm_rvv_packn_fp16s_procedure(int size, __fp16* ptr, const float* gamma_data, const float* beta_data, float eps, int affine, const word_type vl)
{
    // mean and var
    // f16m1 => f32m2
    vfloat32m2_t _sum = vfmv_v_f_f32m2(0.f, vl);
    vfloat32m2_t _sqsum = vfmv_v_f_f32m2(0.f, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat32m2_t _p = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr + vl * i, vl), vl);
        _sum = vfadd_vv_f32m2(_p, _sum, vl);
    }
    vfloat32m2_t _mean = vfdiv_vf_f32m2(_sum, (float)size, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat32m2_t _p = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr + vl * i, vl), vl);
        _p = vfsub_vv_f32m2(_p, _mean, vl);
        _sqsum = vfmacc_vv_f32m2(_sqsum, _p, _p, vl);
    }
    vfloat32m2_t _var = vfdiv_vf_f32m2(_sqsum, (float)size, vl);

    // the var maybe minus due to accuracy
    //float var = sqsum / size - mean * mean;
    vfloat32m2_t _a = vfrdiv_vf_f32m2(vfsqrt_v_f32m2(vfadd_vf_f32m2(_var, eps, vl), vl), 1.f, vl);
    // how about vfrsqrt7.v?
    vfloat32m2_t _b = vfmul_vv_f32m2(vfsgnjn_vv_f32m2(_mean, _mean, vl), _a, vl);
    if (affine)
    {
        for (int i = 0; i < size; i++)
        {
            const int offset = vl * i;
            vfloat32m2_t _p = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr + offset, vl), vl);
            _p = vfmadd_vv_f32m2(_p, _a, _b, vl);
            _p = vfmul_vf_f32m2(_p, gamma_data[i], vl);
            _p = vfadd_vf_f32m2(_p, beta_data[i], vl);
            vse16_v_f16m1(ptr + offset, vfncvt_f_f_w_f16m1(_p, vl), vl);
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            const int offset = vl * i;
            vfloat32m2_t _p = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr + offset, vl), vl);
            _p = vfmadd_vv_f32m2(_p, _a, _b, vl);
            vse16_v_f16m1(ptr + offset, vfncvt_f_f_w_f16m1(_p, vl), vl);
        }
    }

    return 0;
}

// fp16sa

static inline int layernorm_rvv_pack1_fp16sa_procedure(int size, __fp16* ptr, const float* gamma_data, const float* beta_data, float eps, int affine)
{
    __fp16 sum = 0.f;
    __fp16 sqsum = 0.f;
    vfloat16m1_t _sum = vfmv_s_f_f16m1(vundefined_f16m1(), 0.f, vsetvlmax_e32m1());
    vfloat16m1_t _sqsum = vfmv_s_f_f16m1(vundefined_f16m1(), 0.f, vsetvlmax_e32m1());
    {
        int n = size;
        __fp16* ptr_sum = ptr;
        while (n > 0)
        {
            word_type vl = vsetvl_e16m8(n);
            vfloat16m8_t _p = vle16_v_f16m8(ptr_sum, vl);
            _sum = vfredusum_vs_f16m8_f16m1(_sum, _p, /* scalar */ _sum, vl);
            // _sqsum = vfredusum_vs_f32m8_f32m1(_sqsum, vfmul_vv_f32m8(_p, _p, vl), /* scalar */ _sqsum, vl);
            ptr_sum += vl;
            n -= vl;
        }
    }
    sum = vfmv_f_s_f16m1_f16(_sum);
    __fp16 mean = sum / size;

    {
        int n = size;
        __fp16* ptr_sqsum = ptr;
        while (n > 0)
        {
            word_type vl = vsetvl_e16m8(n);
            vfloat16m8_t _p = vle16_v_f16m8(ptr_sqsum, vl);
            _p = vfsub_vf_f16m8(_p, mean, vl);
            _sqsum = vfredusum_vs_f16m8_f16m1(_sqsum, vfmul_vv_f16m8(_p, _p, vl), /* scalar */ _sqsum, vl);
            n -= vl;
            ptr_sqsum += vl;
        }
    }
    sqsum = vfmv_f_s_f16m1_f16(_sqsum);
    __fp16 var = sqsum / size;
    // the var maybe minus due to accuracy
    //float var = sqsum / size - mean * mean;
    __fp16 a = static_cast<__fp16>(1.f / (sqrt(var + eps)));
    __fp16 b = static_cast<__fp16>(-mean * a);

    {
        int n = size;
        __fp16* ptr_store = ptr;
        const float* ptr_gamma = gamma_data;
        const float* ptr_beta = beta_data;
        if (affine)
        {
            while (n > 0)
            {
                word_type vl = vsetvl_e16m4(n);
                vfloat16m4_t _p = vle16_v_f16m4(ptr_store, vl);
                _p = vfmul_vf_f16m4(_p, a, vl);
                vfloat16m4_t _gamma = vfncvt_f_f_w_f16m4(vle32_v_f32m8(ptr_gamma, vl), vl);
                _p = vfadd_vf_f16m4(_p, b, vl);
                vfloat16m4_t _beta = vfncvt_f_f_w_f16m4(vle32_v_f32m8(ptr_beta, vl), vl);
                _p = vfmadd_vv_f16m4(_p, _gamma, _beta, vl);
                vse16_v_f16m4(ptr_store, _p, vl);

                n -= vl;
                ptr_store += vl;
                ptr_gamma += vl;
                ptr_beta += vl;
            }
        }
        else
        {
            while (n > 0)
            {
                word_type vl = vsetvl_e16m8(n);
                vfloat16m8_t _p = vle16_v_f16m8(ptr_store, vl);
                _p = vfmul_vf_f16m8(_p, a, vl);
                _p = vfadd_vf_f16m8(_p, b, vl);
                vse16_v_f16m8(ptr_store, _p, vl);
                n -= vl;
                ptr_store += vl;
            }
        }
    }
    return 0;
}

static inline int layernorm_rvv_packn_fp16sa_procedure(int size, __fp16* ptr, const float* gamma_data, const float* beta_data, float eps, int affine, const word_type vl)
{
    // mean and var
    vfloat16m1_t _sum = vfmv_v_f_f16m1(0.f, vl);
    vfloat16m1_t _sqsum = vfmv_v_f_f16m1(0.f, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat16m1_t _p = vle16_v_f16m1(ptr + vl * i, vl);
        _sum = vfadd_vv_f16m1(_p, _sum, vl);
        // _sqsum = vfmadd_vv_f16m1(_p,_p,_sqsum,vl);
    }
    vfloat16m1_t _mean = vfdiv_vf_f16m1(_sum, size, vl);
    for (int i = 0; i < size; i++)
    {
        vfloat16m1_t _p = vle16_v_f16m1(ptr + vl * i, vl);
        _p = vfsub_vv_f16m1(_p, _mean, vl);
        _sqsum = vfmacc_vv_f16m1(_sqsum, _p, _p, vl);
    }
    vfloat16m1_t _var = vfdiv_vf_f16m1(_sqsum, size, vl);

    // the var maybe minus due to accuracy
    //float var = sqsum / size - mean * mean;
    vfloat16m1_t _a = vfrdiv_vf_f16m1(vfsqrt_v_f16m1(vfadd_vf_f16m1(_var, eps, vl), vl), 1.f, vl);
    // how about vfrsqrt7.v?
    vfloat16m1_t _b = vfmul_vv_f16m1(vfsgnjn_vv_f16m1(_mean, _mean, vl), _a, vl);
    if (affine)
    {
        for (int i = 0; i < size; i++)
        {
            const int offset = vl * i;
            vfloat16m1_t _p = vle16_v_f16m1(ptr + offset, vl);
            _p = vfmadd_vv_f16m1(_p, _a, _b, vl);
            _p = vfmul_vf_f16m1(_p, gamma_data[i], vl);
            _p = vfadd_vf_f16m1(_p, beta_data[i], vl);
            vse16_v_f16m1(ptr + offset, _p, vl);
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            const int offset = vl * i;
            vfloat16m1_t _p = vle16_v_f16m1(ptr + offset, vl);
            _p = vfmadd_vv_f16m1(_p, _a, _b, vl);
            vse16_v_f16m1(ptr + offset, _p, vl);
        }
    }

    return 0;
}
