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

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void cast_fp32_to_bf16_sse_avx512bf16(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
void cast_bf16_to_fp32_sse_avx512bf16(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
void cast_fp32_to_bf16_sse_avx2(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
void cast_bf16_to_fp32_sse_avx2(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
#endif

static void cast_fp32_to_bf16_sse(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        cast_fp32_to_bf16_sse_avx512bf16(bottom_blob, top_blob, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        cast_fp32_to_bf16_sse_avx2(bottom_blob, top_blob, opt);
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
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 31 < size; i += 32)
        {
            _mm512_storeu_si512((__m512i*)outptr, float2bfloat_avx512(_mm512_loadu_ps(ptr), _mm512_loadu_ps(ptr + 16)));
            ptr += 32;
            outptr += 32;
        }
#endif // __AVX512F__
        for (; i + 15 < size; i += 16)
        {
#if __AVX512F__
            _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_mm512_loadu_ps(ptr)));
#else
            _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx(_mm256_loadu_ps(ptr), _mm256_loadu_ps(ptr + 8)));
#endif
            ptr += 16;
            outptr += 16;
        }
#endif // __AVX__
        for (; i + 7 < size; i += 8)
        {
#if __AVX__
            _mm_store_si128((__m128i*)outptr, float2bfloat_avx(_mm256_loadu_ps(ptr)));
#else
            _mm_store_si128((__m128i*)outptr, float2bfloat_sse(_mm_loadu_ps(ptr), _mm_loadu_ps(ptr + 4)));
#endif
            ptr += 8;
            outptr += 8;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *outptr++ = float32_to_bfloat16(*ptr++);
        }
    }
}

static void cast_bf16_to_fp32_sse(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        cast_bf16_to_fp32_sse_avx512bf16(bottom_blob, top_blob, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        cast_bf16_to_fp32_sse_avx2(bottom_blob, top_blob, opt);
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
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            _mm512_storeu_ps(outptr, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr)));
            ptr += 16;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            _mm256_storeu_ps(outptr, bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr)));
            ptr += 8;
            outptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            _mm_storeu_ps(outptr, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr)));
            ptr += 4;
            outptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *outptr++ = bfloat16_to_float32(*ptr++);
        }
    }
}
