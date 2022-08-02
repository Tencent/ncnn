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

#if __AVX__
static inline __m256 bfloat2float_avx(__m128i v0)
{
    __m128i zero = _mm_set1_epi32(0);
    __m128i a = _mm_slli_epi32(_mm_unpacklo_epi16(v0, zero), 16);
    __m128i b = _mm_slli_epi32(_mm_unpackhi_epi16(v0, zero), 16);
    __m256i ab = _mm256_set1_epi32(0);
    ab = _mm256_insertf128_si256(ab, a, 0); // insert in low 128-bit lane
    ab = _mm256_insertf128_si256(ab, b, 1); // insert in high 128-bit lane
    return _mm256_castsi256_ps(ab);
}
#if __AVX2__
static inline __m256i float2bfloat_avx(__m256 v0, __m256 v1)
{
    __m256i a = _mm256_castps_si256(v0);
    a = _mm256_srli_epi32(a, 16);
    __m256i b = _mm256_castps_si256(v1);
    b = _mm256_srli_epi32(b, 16);
    __m256i abab = _mm256_packus_epi32(a, b);
    return _mm256_permutevar8x32_epi32(abab, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
}
static inline __m128i float2bfloat_avx(__m256 v0)
{
    __m256i a = _mm256_castps_si256(v0);
    a = _mm256_srli_epi32(a, 16);
    __m256i aaaa = _mm256_packus_epi32(a, a);
    return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(aaaa, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7)));
}
#endif
#endif // __AVX__

static void cast_fp32_to_bf16_sse(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        cast_fp32_to_bf16_sse_avx512bf16(bottom_blob, top_blob, opt);
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
#if __AVX512BF16__
        for (; i + 15 < size; i += 16)
        {
            __m512 _v_fp32 = _mm512_loadu_ps(ptr);
            __m256bh _v_bf16 = _mm512_cvtneps_pbh(_v_fp32);
            _mm256_storeu_si256((__m256i*)outptr, (__m256i)_v_bf16);

            ptr += 16;
            outptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            __m256 _v_fp32 = _mm256_loadu_ps(ptr);
            __m128bh _v_bf16 = _mm256_cvtneps_pbh(_v_fp32);
            _mm_storeu_si128((__m128i*)outptr, (__m128i)_v_bf16);

            ptr += 8;
            outptr += 8;
        }
#elif __AVX2__
        for (; i + 15 < size; i += 16)
        {
            _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx(_mm256_loadu_ps(ptr), _mm256_loadu_ps(ptr + 8)));
            ptr += 16;
            outptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            _mm_store_si128((__m128i*)outptr, float2bfloat_avx(_mm256_loadu_ps(ptr)));
            ptr += 8;
            outptr += 8;
        }
#endif
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
#if __AVX512BF16__
        for (; i + 15 < size; i += 16)
        {
            __m256bh _v_bf16 = (__m256bh)_mm256_loadu_si256((const __m256i*)ptr);
            __m512 _v_fp32 = _mm512_cvtpbh_ps(_v_bf16);
            _mm512_storeu_ps(outptr, _v_fp32);

            ptr += 16;
            outptr += 16;
        }
        for (; i + 7 < size; i += 8)
        {
            __m128bh _v_bf16 = (__m128bh)_mm_loadu_si128((const __m128i*)ptr);
            __m256 _v_fp32 = _mm256_cvtpbh_ps(_v_bf16);
            _mm256_storeu_ps(outptr, _v_fp32);

            ptr += 8;
            outptr += 8;
        }
#elif __AVX__
        for (; i + 7 < size; i += 8)
        {
            _mm256_storeu_ps(outptr, bfloat2float_avx(_mm_lddqu_si128((__m128i const*)ptr)));
            ptr += 8;
            outptr += 8;
        }
#endif
        for (; i < size; i++)
        {
            *outptr++ = bfloat16_to_float32(*ptr++);
        }
    }
}
