// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

static void requantize_leakyrelu_pack8_lsx(const Mat& bottom_blob, Mat& top_blob, const Mat& scale_in_data, const Mat& scale_out_data, const Mat& bias_data, float slope, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    int scale_in_data_size = scale_in_data.w;
    int scale_out_data_size = scale_out_data.w;
    int bias_data_size = bias_data.w;

    // int8(leakyrelu(v * scale_in, slope) * scale_out)
    // int8_leakyrelu(v * (scale_in * scale_out), slope)

    // int8(leakyrelu(v * scale_in + bias, slope) * scale_out)
    // int8_leakyrelu(v * (scale_in * scale_out) + (bias * scale_out), slope)

    if (bias_data_size == 0)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const int* intptr = bottom_blob.channel(q);
            signed char* ptr = top_blob.channel(q);

            __m128 _scale_in0 = scale_in_data_size == 1 ? (__m128)__lsx_vreplfr2vr_s(scale_in_data[0]) : (__m128)__lsx_vld((const float*)scale_in_data + q * 8, 0);
            __m128 _scale_in1 = scale_in_data_size == 1 ? (__m128)__lsx_vreplfr2vr_s(scale_in_data[0]) : (__m128)__lsx_vld((const float*)scale_in_data + q * 8 + 4, 0);
            __m128 _scale_out0 = scale_out_data_size == 1 ? (__m128)__lsx_vreplfr2vr_s(scale_out_data[0]) : (__m128)__lsx_vld((const float*)scale_out_data + q * 8, 0);
            __m128 _scale_out1 = scale_out_data_size == 1 ? (__m128)__lsx_vreplfr2vr_s(scale_out_data[0]) : (__m128)__lsx_vld((const float*)scale_out_data + q * 8 + 4, 0);

            __m128 _scale0 = __lsx_vfmul_s(_scale_in0, _scale_out0);
            __m128 _scale1 = __lsx_vfmul_s(_scale_in1, _scale_out1);
            __m128 _slope = (__m128)__lsx_vreplfr2vr_s(slope);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(intptr + 128);
                __m128 _v0 = __lsx_vffint_s_w(__lsx_vld(intptr, 0));
                __m128 _v1 = __lsx_vffint_s_w(__lsx_vld(intptr + 4, 0));
                __m128 _v2 = __lsx_vffint_s_w(__lsx_vld(intptr + 8, 0));
                __m128 _v3 = __lsx_vffint_s_w(__lsx_vld(intptr + 12, 0));
                __m128 _v4 = __lsx_vffint_s_w(__lsx_vld(intptr + 16, 0));
                __m128 _v5 = __lsx_vffint_s_w(__lsx_vld(intptr + 20, 0));
                __m128 _v6 = __lsx_vffint_s_w(__lsx_vld(intptr + 24, 0));
                __m128 _v7 = __lsx_vffint_s_w(__lsx_vld(intptr + 28, 0));
                _v0 = __lsx_vfmul_s(_v0, _scale0);
                _v1 = __lsx_vfmul_s(_v1, _scale1);
                _v2 = __lsx_vfmul_s(_v2, _scale0);
                _v3 = __lsx_vfmul_s(_v3, _scale1);
                _v4 = __lsx_vfmul_s(_v4, _scale0);
                _v5 = __lsx_vfmul_s(_v5, _scale1);
                _v6 = __lsx_vfmul_s(_v6, _scale0);
                _v7 = __lsx_vfmul_s(_v7, _scale1);
                *((int64_t*)ptr) = float2int8leakyrelu(_v0, _v1, _slope);
                *((int64_t*)(ptr + 8)) = float2int8leakyrelu(_v2, _v3, _slope);
                *((int64_t*)(ptr + 16)) = float2int8leakyrelu(_v4, _v5, _slope);
                *((int64_t*)(ptr + 24)) = float2int8leakyrelu(_v6, _v7, _slope);

                intptr += 32;
                ptr += 32;
            }
            for (; i + 1 < size; i += 2)
            {
                __builtin_prefetch(intptr + 64);
                __m128 _v0 = __lsx_vffint_s_w(__lsx_vld(intptr, 0));
                __m128 _v1 = __lsx_vffint_s_w(__lsx_vld(intptr + 4, 0));
                __m128 _v2 = __lsx_vffint_s_w(__lsx_vld(intptr + 8, 0));
                __m128 _v3 = __lsx_vffint_s_w(__lsx_vld(intptr + 12, 0));
                _v0 = __lsx_vfmul_s(_v0, _scale0);
                _v1 = __lsx_vfmul_s(_v1, _scale1);
                _v2 = __lsx_vfmul_s(_v2, _scale0);
                _v3 = __lsx_vfmul_s(_v3, _scale1);
                *((int64_t*)ptr) = float2int8leakyrelu(_v0, _v1, _slope);
                *((int64_t*)(ptr + 8)) = float2int8leakyrelu(_v2, _v3, _slope);

                intptr += 16;
                ptr += 16;
            }
            for (; i < size; i++)
            {
                __builtin_prefetch(intptr + 32);
                __m128 _v0 = __lsx_vffint_s_w(__lsx_vld(intptr, 0));
                __m128 _v1 = __lsx_vffint_s_w(__lsx_vld(intptr + 4, 0));
                _v0 = __lsx_vfmul_s(_v0, _scale0);
                _v1 = __lsx_vfmul_s(_v1, _scale1);
                *((int64_t*)ptr) = float2int8leakyrelu(_v0, _v1, _slope);

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

            __m128 _scale_in0 = scale_in_data_size == 1 ? (__m128)__lsx_vreplfr2vr_s(scale_in_data[0]) : (__m128)__lsx_vld((const float*)scale_in_data + q * 8, 0);
            __m128 _scale_in1 = scale_in_data_size == 1 ? (__m128)__lsx_vreplfr2vr_s(scale_in_data[0]) : (__m128)__lsx_vld((const float*)scale_in_data + q * 8 + 4, 0);
            __m128 _scale_out0 = scale_out_data_size == 1 ? (__m128)__lsx_vreplfr2vr_s(scale_out_data[0]) : (__m128)__lsx_vld((const float*)scale_out_data + q * 8, 0);
            __m128 _scale_out1 = scale_out_data_size == 1 ? (__m128)__lsx_vreplfr2vr_s(scale_out_data[0]) : (__m128)__lsx_vld((const float*)scale_out_data + q * 8 + 4, 0);
            __m128 _bias0 = bias_data_size == 1 ? (__m128)__lsx_vreplfr2vr_s(bias_data[0]) : (__m128)__lsx_vld((const float*)bias_data + q * 8, 0);
            __m128 _bias1 = bias_data_size == 1 ? (__m128)__lsx_vreplfr2vr_s(bias_data[0]) : (__m128)__lsx_vld((const float*)bias_data + q * 8 + 4, 0);

            __m128 _scale0 = __lsx_vfmul_s(_scale_in0, _scale_out0);
            __m128 _scale1 = __lsx_vfmul_s(_scale_in1, _scale_out1);
            _bias0 = __lsx_vfmul_s(_bias0, _scale_out0);
            _bias1 = __lsx_vfmul_s(_bias1, _scale_out1);
            __m128 _slope = (__m128)__lsx_vreplfr2vr_s(slope);

            int i = 0;
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(intptr + 128);
                __m128 _v0 = __lsx_vffint_s_w(__lsx_vld(intptr, 0));
                __m128 _v1 = __lsx_vffint_s_w(__lsx_vld(intptr + 4, 0));
                __m128 _v2 = __lsx_vffint_s_w(__lsx_vld(intptr + 8, 0));
                __m128 _v3 = __lsx_vffint_s_w(__lsx_vld(intptr + 12, 0));
                __m128 _v4 = __lsx_vffint_s_w(__lsx_vld(intptr + 16, 0));
                __m128 _v5 = __lsx_vffint_s_w(__lsx_vld(intptr + 20, 0));
                __m128 _v6 = __lsx_vffint_s_w(__lsx_vld(intptr + 24, 0));
                __m128 _v7 = __lsx_vffint_s_w(__lsx_vld(intptr + 28, 0));
                _v0 = __lsx_vfmadd_s(_scale0, _v0, _bias0);
                _v1 = __lsx_vfmadd_s(_scale1, _v1, _bias1);
                _v2 = __lsx_vfmadd_s(_scale0, _v2, _bias0);
                _v3 = __lsx_vfmadd_s(_scale1, _v3, _bias1);
                _v4 = __lsx_vfmadd_s(_scale0, _v4, _bias0);
                _v5 = __lsx_vfmadd_s(_scale1, _v5, _bias1);
                _v6 = __lsx_vfmadd_s(_scale0, _v6, _bias0);
                _v7 = __lsx_vfmadd_s(_scale1, _v7, _bias1);
                *((int64_t*)ptr) = float2int8leakyrelu(_v0, _v1, _slope);
                *((int64_t*)(ptr + 8)) = float2int8leakyrelu(_v2, _v3, _slope);
                *((int64_t*)(ptr + 16)) = float2int8leakyrelu(_v4, _v5, _slope);
                *((int64_t*)(ptr + 24)) = float2int8leakyrelu(_v6, _v7, _slope);

                intptr += 32;
                ptr += 32;
            }
            for (; i + 1 < size; i += 2)
            {
                __builtin_prefetch(intptr + 64);
                __m128 _v0 = __lsx_vffint_s_w(__lsx_vld(intptr, 0));
                __m128 _v1 = __lsx_vffint_s_w(__lsx_vld(intptr + 4, 0));
                __m128 _v2 = __lsx_vffint_s_w(__lsx_vld(intptr + 8, 0));
                __m128 _v3 = __lsx_vffint_s_w(__lsx_vld(intptr + 12, 0));
                _v0 = __lsx_vfmadd_s(_scale0, _v0, _bias0);
                _v1 = __lsx_vfmadd_s(_scale1, _v1, _bias1);
                _v2 = __lsx_vfmadd_s(_scale0, _v2, _bias0);
                _v3 = __lsx_vfmadd_s(_scale1, _v3, _bias1);
                *((int64_t*)ptr) = float2int8leakyrelu(_v0, _v1, _slope);
                *((int64_t*)(ptr + 8)) = float2int8leakyrelu(_v2, _v3, _slope);

                intptr += 16;
                ptr += 16;
            }
            for (; i < size; i++)
            {
                __builtin_prefetch(intptr + 32);
                __m128 _v0 = __lsx_vffint_s_w(__lsx_vld(intptr, 0));
                __m128 _v1 = __lsx_vffint_s_w(__lsx_vld(intptr + 4, 0));
                _v0 = __lsx_vfmadd_s(_scale0, _v0, _bias0);
                _v1 = __lsx_vfmadd_s(_scale1, _v1, _bias1);
                *((int64_t*)ptr) = float2int8leakyrelu(_v0, _v1, _slope);

                intptr += 8;
                ptr += 8;
            }
        }
    }
}
