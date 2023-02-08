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

#if NCNN_RUNTIME_CPU && NCNN_F16C && __AVX__ && !__F16C__
void cast_fp32_to_fp16_sse_f16c(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
void cast_fp16_to_fp32_sse_f16c(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
#endif

static void cast_fp32_to_fp16_sse(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_F16C && __AVX__ && !__F16C__
    if (ncnn::cpu_support_x86_f16c())
    {
        cast_fp32_to_fp16_sse_f16c(bottom_blob, top_blob, opt);
        return;
    }
#endif

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        unsigned short* outptr = top_blob.channel(q);

        int i = 0;
#if __F16C__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _v_fp32 = _mm512_loadu_ps(ptr);
            __m256i _v_fp16 = _mm512_cvtps_ph(_v_fp32, _MM_ROUND_NEAREST | _MM_FROUND_NO_EXC);
            _mm256_storeu_si256((__m256i*)outptr, _v_fp16);
            ptr += 16;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _v_fp32 = _mm256_loadu_ps(ptr);
            __m128i _v_fp16 = _mm256_cvtps_ph(_v_fp32, _MM_ROUND_NEAREST | _MM_FROUND_NO_EXC);
            _mm_storeu_si128((__m128i*)outptr, _v_fp16);
            ptr += 8;
            outptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __m128 _v_fp32 = _mm_loadu_ps(ptr);
            __m128i _v_fp16 = _mm_cvtps_ph(_v_fp32, _MM_ROUND_NEAREST | _MM_FROUND_NO_EXC);
            _mm_storel_epi64((__m128i*)outptr, _v_fp16);
            ptr += 4;
            outptr += 4;
        }
#endif // __F16C__
        for (; i < size; i++)
        {
            *outptr++ = float32_to_float16(*ptr++);
        }
    }
}

static void cast_fp16_to_fp32_sse(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_F16C && __AVX__ && !__F16C__
    if (ncnn::cpu_support_x86_f16c())
    {
        cast_fp16_to_fp32_sse_f16c(bottom_blob, top_blob, opt);
        return;
    }
#endif

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const unsigned short* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

        int i = 0;
#if __F16C__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m256i _v_fp16 = _mm256_loadu_si256((const __m256i*)ptr);
            __m512 _v_fp32 = _mm512_cvtph_ps(_v_fp16);
            _mm512_storeu_ps(outptr, _v_fp32);
            ptr += 16;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m128i _v_fp16 = _mm_loadu_si128((const __m128i*)ptr);
            __m256 _v_fp32 = _mm256_cvtph_ps(_v_fp16);
            _mm256_storeu_ps(outptr, _v_fp32);
            ptr += 8;
            outptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            __m128i _v_fp16 = _mm_loadl_epi64((const __m128i*)ptr);
            __m128 _v_fp32 = _mm_cvtph_ps(_v_fp16);
            _mm_storeu_ps(outptr, _v_fp32);
            ptr += 4;
            outptr += 4;
        }
#endif // __F16C__
        for (; i < size; i++)
        {
            *outptr++ = float16_to_float32(*ptr++);
        }
    }
}
