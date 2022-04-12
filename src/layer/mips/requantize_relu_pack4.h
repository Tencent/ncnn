// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

static void requantize_relu_pack4_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& scale_in_data, const Mat& scale_out_data, const Mat& bias_data, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;
    int outc = top_blob.c;
    int out_elempack = top_blob.elempack;

    int scale_in_data_size = scale_in_data.w;
    int scale_out_data_size = scale_out_data.w;
    int bias_data_size = bias_data.w;

    // int8(relu(v * scale_in) * scale_out)
    // int8_relu(v * (scale_in * scale_out))

    // int8(relu(v * scale_in + bias) * scale_out)
    // int8_relu(v * (scale_in * scale_out) + (bias * scale_out))

    if (out_elempack == 8)
    {
        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const int* intptr0 = bottom_blob.channel(q * 2);
                const int* intptr1 = bottom_blob.channel(q * 2 + 1);
                signed char* ptr = top_blob.channel(q);

                v4f32 _scale_in0 = scale_in_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_in_data[0]) : (v4f32)__msa_ld_w((const float*)scale_in_data + q * 8, 0);
                v4f32 _scale_in1 = scale_in_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_in_data[0]) : (v4f32)__msa_ld_w((const float*)scale_in_data + q * 8 + 4, 0);
                v4f32 _scale_out0 = scale_out_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_out_data[0]) : (v4f32)__msa_ld_w((const float*)scale_out_data + q * 8, 0);
                v4f32 _scale_out1 = scale_out_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_out_data[0]) : (v4f32)__msa_ld_w((const float*)scale_out_data + q * 8 + 4, 0);

                v4f32 _scale0 = __msa_fmul_w(_scale_in0, _scale_out0);
                v4f32 _scale1 = __msa_fmul_w(_scale_in1, _scale_out1);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __builtin_prefetch(intptr0 + 64);
                    __builtin_prefetch(intptr1 + 64);
                    v4f32 _v00 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr0, 0));
                    v4f32 _v01 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr0 + 4, 0));
                    v4f32 _v02 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr0 + 8, 0));
                    v4f32 _v03 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr0 + 12, 0));
                    v4f32 _v10 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr1, 0));
                    v4f32 _v11 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr1 + 4, 0));
                    v4f32 _v12 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr1 + 8, 0));
                    v4f32 _v13 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr1 + 12, 0));
                    _v00 = __msa_fmul_w(_v00, _scale0);
                    _v01 = __msa_fmul_w(_v01, _scale0);
                    _v02 = __msa_fmul_w(_v02, _scale0);
                    _v03 = __msa_fmul_w(_v03, _scale0);
                    _v10 = __msa_fmul_w(_v10, _scale1);
                    _v11 = __msa_fmul_w(_v11, _scale1);
                    _v12 = __msa_fmul_w(_v12, _scale1);
                    _v13 = __msa_fmul_w(_v13, _scale1);
                    *((int64_t*)ptr) = float2int8relu(_v00, _v10);
                    *((int64_t*)(ptr + 8)) = float2int8relu(_v01, _v11);
                    *((int64_t*)(ptr + 16)) = float2int8relu(_v02, _v12);
                    *((int64_t*)(ptr + 24)) = float2int8relu(_v03, _v13);

                    intptr0 += 16;
                    intptr1 += 16;
                    ptr += 32;
                }
                for (; i < size; i++)
                {
                    __builtin_prefetch(intptr0 + 16);
                    __builtin_prefetch(intptr1 + 16);
                    v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr0, 0));
                    v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr1, 0));
                    _v0 = __msa_fmul_w(_v0, _scale0);
                    _v1 = __msa_fmul_w(_v1, _scale1);
                    *((int64_t*)ptr) = float2int8relu(_v0, _v1);

                    intptr0 += 4;
                    intptr1 += 4;
                    ptr += 8;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const int* intptr0 = bottom_blob.channel(q * 2);
                const int* intptr1 = bottom_blob.channel(q * 2 + 1);
                signed char* ptr = top_blob.channel(q);

                v4f32 _scale_in0 = scale_in_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_in_data[0]) : (v4f32)__msa_ld_w((const float*)scale_in_data + q * 8, 0);
                v4f32 _scale_in1 = scale_in_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_in_data[0]) : (v4f32)__msa_ld_w((const float*)scale_in_data + q * 8 + 4, 0);
                v4f32 _scale_out0 = scale_out_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_out_data[0]) : (v4f32)__msa_ld_w((const float*)scale_out_data + q * 8, 0);
                v4f32 _scale_out1 = scale_out_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_out_data[0]) : (v4f32)__msa_ld_w((const float*)scale_out_data + q * 8 + 4, 0);
                v4f32 _bias0 = bias_data_size == 1 ? (v4f32)__msa_fill_w_f32(bias_data[0]) : (v4f32)__msa_ld_w((const float*)bias_data + q * 8, 0);
                v4f32 _bias1 = bias_data_size == 1 ? (v4f32)__msa_fill_w_f32(bias_data[0]) : (v4f32)__msa_ld_w((const float*)bias_data + q * 8 + 4, 0);

                v4f32 _scale0 = __msa_fmul_w(_scale_in0, _scale_out0);
                v4f32 _scale1 = __msa_fmul_w(_scale_in1, _scale_out1);
                _bias0 = __msa_fmul_w(_bias0, _scale_out0);
                _bias1 = __msa_fmul_w(_bias1, _scale_out1);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    __builtin_prefetch(intptr0 + 64);
                    __builtin_prefetch(intptr1 + 64);
                    v4f32 _v00 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr0, 0));
                    v4f32 _v01 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr0 + 4, 0));
                    v4f32 _v02 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr0 + 8, 0));
                    v4f32 _v03 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr0 + 12, 0));
                    v4f32 _v10 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr1, 0));
                    v4f32 _v11 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr1 + 4, 0));
                    v4f32 _v12 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr1 + 8, 0));
                    v4f32 _v13 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr1 + 12, 0));
                    _v00 = __msa_fmadd_w(_bias0, _v00, _scale0);
                    _v01 = __msa_fmadd_w(_bias0, _v01, _scale0);
                    _v02 = __msa_fmadd_w(_bias0, _v02, _scale0);
                    _v03 = __msa_fmadd_w(_bias0, _v03, _scale0);
                    _v10 = __msa_fmadd_w(_bias1, _v10, _scale1);
                    _v11 = __msa_fmadd_w(_bias1, _v11, _scale1);
                    _v12 = __msa_fmadd_w(_bias1, _v12, _scale1);
                    _v13 = __msa_fmadd_w(_bias1, _v13, _scale1);
                    *((int64_t*)ptr) = float2int8relu(_v00, _v10);
                    *((int64_t*)(ptr + 8)) = float2int8relu(_v01, _v11);
                    *((int64_t*)(ptr + 16)) = float2int8relu(_v02, _v12);
                    *((int64_t*)(ptr + 24)) = float2int8relu(_v03, _v13);

                    intptr0 += 16;
                    intptr1 += 16;
                    ptr += 32;
                }
                for (; i + 1 < size; i += 2)
                {
                    __builtin_prefetch(intptr0 + 32);
                    __builtin_prefetch(intptr1 + 32);
                    v4f32 _v00 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr0, 0));
                    v4f32 _v01 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr0 + 4, 0));
                    v4f32 _v10 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr1, 0));
                    v4f32 _v11 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr1 + 4, 0));
                    _v00 = __msa_fmadd_w(_bias0, _v00, _scale0);
                    _v01 = __msa_fmadd_w(_bias0, _v01, _scale0);
                    _v10 = __msa_fmadd_w(_bias1, _v10, _scale1);
                    _v11 = __msa_fmadd_w(_bias1, _v11, _scale1);
                    *((int64_t*)ptr) = float2int8relu(_v00, _v10);
                    *((int64_t*)(ptr + 8)) = float2int8relu(_v01, _v11);

                    intptr0 += 8;
                    intptr1 += 8;
                    ptr += 16;
                }
                for (; i < size; i++)
                {
                    __builtin_prefetch(intptr0 + 16);
                    __builtin_prefetch(intptr1 + 16);
                    v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr0, 0));
                    v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr1, 0));
                    _v0 = __msa_fmadd_w(_bias0, _v0, _scale0);
                    _v1 = __msa_fmadd_w(_bias1, _v1, _scale1);
                    *((int64_t*)ptr) = float2int8relu(_v0, _v1);

                    intptr0 += 4;
                    intptr1 += 4;
                    ptr += 8;
                }
            }
        }
    }
    if (out_elempack == 1)
    {
        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                signed char* ptr0 = top_blob.channel(q * 4);
                signed char* ptr1 = top_blob.channel(q * 4 + 1);
                signed char* ptr2 = top_blob.channel(q * 4 + 2);
                signed char* ptr3 = top_blob.channel(q * 4 + 3);

                v4f32 _scale_in = scale_in_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_in_data[0]) : (v4f32)__msa_ld_w((const float*)scale_in_data + q * 4, 0);
                v4f32 _scale_out = scale_out_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_out_data[0]) : (v4f32)__msa_ld_w((const float*)scale_out_data + q * 4, 0);

                v4f32 _scale = __msa_fmul_w(_scale_in, _scale_out);

                int i = 0;
                for (; i < size; i++)
                {
                    __builtin_prefetch(intptr + 16);
                    v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                    _v = __msa_fmul_w(_v, _scale);
                    v16i8 v = float2int8relu(_v);
                    ptr0[0] = v[0];
                    ptr1[0] = v[1];
                    ptr2[0] = v[2];
                    ptr3[0] = v[3];

                    intptr += 4;
                    ptr0 += 1;
                    ptr1 += 1;
                    ptr2 += 1;
                    ptr3 += 1;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                signed char* ptr0 = top_blob.channel(q * 4);
                signed char* ptr1 = top_blob.channel(q * 4 + 1);
                signed char* ptr2 = top_blob.channel(q * 4 + 2);
                signed char* ptr3 = top_blob.channel(q * 4 + 3);

                v4f32 _scale_in = scale_in_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_in_data[0]) : (v4f32)__msa_ld_w((const float*)scale_in_data + q * 4, 0);
                v4f32 _scale_out = scale_out_data_size == 1 ? (v4f32)__msa_fill_w_f32(scale_out_data[0]) : (v4f32)__msa_ld_w((const float*)scale_out_data + q * 4, 0);
                v4f32 _bias = bias_data_size == 1 ? (v4f32)__msa_fill_w_f32(bias_data[0]) : (v4f32)__msa_ld_w((const float*)bias_data + q * 4, 0);

                v4f32 _scale = __msa_fmul_w(_scale_in, _scale_out);
                _bias = __msa_fmul_w(_bias, _scale_out);

                int i = 0;
                for (; i < size; i++)
                {
                    __builtin_prefetch(intptr + 16);
                    v4f32 _v = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                    _v = __msa_fmadd_w(_bias, _v, _scale);
                    v16i8 v = float2int8relu(_v);
                    ptr0[0] = v[0];
                    ptr1[0] = v[1];
                    ptr2[0] = v[2];
                    ptr3[0] = v[3];

                    intptr += 4;
                    ptr0 += 1;
                    ptr1 += 1;
                    ptr2 += 1;
                    ptr3 += 1;
                }
            }
        }
    }
}
