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

static void requantize_relu_pack8_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& scale_in_data, const Mat& scale_out_data, const Mat& bias_data, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    int scale_in_data_size = scale_in_data.w;
    int scale_out_data_size = scale_out_data.w;
    int bias_data_size = bias_data.w;

    // int8(relu(v * scale_in) * scale_out)
    // int8_relu(v * (scale_in * scale_out))

    // int8(relu(v * scale_in + bias) * scale_out)
    // int8_relu(v * (scale_in * scale_out) + (bias * scale_out))

    if (bias_data_size == 0)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
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
                __builtin_prefetch(intptr + 128);
                v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
                v4f32 _v2 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 8, 0));
                v4f32 _v3 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 12, 0));
                v4f32 _v4 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 16, 0));
                v4f32 _v5 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 20, 0));
                v4f32 _v6 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 24, 0));
                v4f32 _v7 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 28, 0));
                _v0 = __msa_fmul_w(_v0, _scale0);
                _v1 = __msa_fmul_w(_v1, _scale1);
                _v2 = __msa_fmul_w(_v2, _scale0);
                _v3 = __msa_fmul_w(_v3, _scale1);
                _v4 = __msa_fmul_w(_v4, _scale0);
                _v5 = __msa_fmul_w(_v5, _scale1);
                _v6 = __msa_fmul_w(_v6, _scale0);
                _v7 = __msa_fmul_w(_v7, _scale1);
                *((int64_t*)ptr) = float2int8relu(_v0, _v1);
                *((int64_t*)(ptr + 8)) = float2int8relu(_v2, _v3);
                *((int64_t*)(ptr + 16)) = float2int8relu(_v4, _v5);
                *((int64_t*)(ptr + 24)) = float2int8relu(_v6, _v7);

                intptr += 32;
                ptr += 32;
            }
            for (; i + 1 < size; i += 2)
            {
                __builtin_prefetch(intptr + 64);
                v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
                v4f32 _v2 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 8, 0));
                v4f32 _v3 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 12, 0));
                _v0 = __msa_fmul_w(_v0, _scale0);
                _v1 = __msa_fmul_w(_v1, _scale1);
                _v2 = __msa_fmul_w(_v2, _scale0);
                _v3 = __msa_fmul_w(_v3, _scale1);
                *((int64_t*)ptr) = float2int8relu(_v0, _v1);
                *((int64_t*)(ptr + 8)) = float2int8relu(_v2, _v3);

                intptr += 16;
                ptr += 16;
            }
            for (; i < size; i++)
            {
                __builtin_prefetch(intptr + 32);
                v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
                _v0 = __msa_fmul_w(_v0, _scale0);
                _v1 = __msa_fmul_w(_v1, _scale1);
                *((int64_t*)ptr) = float2int8relu(_v0, _v1);

                intptr += 8;
                ptr += 8;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
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
                __builtin_prefetch(intptr + 128);
                v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
                v4f32 _v2 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 8, 0));
                v4f32 _v3 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 12, 0));
                v4f32 _v4 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 16, 0));
                v4f32 _v5 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 20, 0));
                v4f32 _v6 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 24, 0));
                v4f32 _v7 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 28, 0));
                _v0 = __msa_fmadd_w(_bias0, _v0, _scale0);
                _v1 = __msa_fmadd_w(_bias1, _v1, _scale1);
                _v2 = __msa_fmadd_w(_bias0, _v2, _scale0);
                _v3 = __msa_fmadd_w(_bias1, _v3, _scale1);
                _v4 = __msa_fmadd_w(_bias0, _v4, _scale0);
                _v5 = __msa_fmadd_w(_bias1, _v5, _scale1);
                _v6 = __msa_fmadd_w(_bias0, _v6, _scale0);
                _v7 = __msa_fmadd_w(_bias1, _v7, _scale1);
                *((int64_t*)ptr) = float2int8relu(_v0, _v1);
                *((int64_t*)(ptr + 8)) = float2int8relu(_v2, _v3);
                *((int64_t*)(ptr + 16)) = float2int8relu(_v4, _v5);
                *((int64_t*)(ptr + 24)) = float2int8relu(_v6, _v7);

                intptr += 32;
                ptr += 32;
            }
            for (; i + 1 < size; i += 2)
            {
                __builtin_prefetch(intptr + 64);
                v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
                v4f32 _v2 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 8, 0));
                v4f32 _v3 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 12, 0));
                _v0 = __msa_fmadd_w(_bias0, _v0, _scale0);
                _v1 = __msa_fmadd_w(_bias1, _v1, _scale1);
                _v2 = __msa_fmadd_w(_bias0, _v2, _scale0);
                _v3 = __msa_fmadd_w(_bias1, _v3, _scale1);
                *((int64_t*)ptr) = float2int8relu(_v0, _v1);
                *((int64_t*)(ptr + 8)) = float2int8relu(_v2, _v3);

                intptr += 16;
                ptr += 16;
            }
            for (; i < size; i++)
            {
                __builtin_prefetch(intptr + 32);
                v4f32 _v0 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr, 0));
                v4f32 _v1 = (v4f32)__msa_ffint_s_w(__msa_ld_w(intptr + 4, 0));
                _v0 = __msa_fmadd_w(_bias0, _v0, _scale0);
                _v1 = __msa_fmadd_w(_bias1, _v1, _scale1);
                *((int64_t*)ptr) = float2int8relu(_v0, _v1);

                intptr += 8;
                ptr += 8;
            }
        }
    }
}
