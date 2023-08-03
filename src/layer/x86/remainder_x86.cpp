#include "remainder_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"

namespace ncnn {

Remainder_x86::Remainder_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int Remainder_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h * d * elempack;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // first blob
    const Mat& bottom_blob1 = bottom_blobs[1];
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        const float* ptr1 = bottom_blob1.channel(q);
        float* outptr = top_blob.channel(q);

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _p1 = _mm512_loadu_ps(ptr1);
            // TODO: Instruction for remainder
            // _p = xxxxx(_p, _p1);
            // _mm512_storeu_ps(outptr, _p);

            ptr += 16;
            ptr1 += 16;
            outptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _p1 = _mm256_loadu_ps(ptr1);
            // TODO: Instruction for remainder
            // _p = xxxxx(_p, _p1);
            // _mm256_storeu_ps(outptr, _p);

            ptr += 8;
            ptr1 += 8;
            outptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_load_ps(ptr);
            __m128 _p1 = _mm_load_ps(ptr1);
            // TODO: Instruction for remainder
            // _p = xxxxx(_p, _p1);
            // _mm_store_ps(outptr, _p);

            ptr += 4;
            ptr1 += 4;
            outptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *outptr = remainderf(*ptr, *ptr1);

            ptr++;
            ptr1++;
            outptr++;
        }
    }

    for (size_t b = 2; b < bottom_blobs.size(); b++)
    {
        const Mat& bottom_blob2 = bottom_blobs[b];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob2.channel(q);
            float* outptr = top_blob.channel(q);

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(outptr);
                __m512 _p1 = _mm512_loadu_ps(ptr);
                // TODO: Instruction for remainder
                // _p = xxxxx(_p, _p1);
                // _mm512_storeu_ps(outptr, _p);

                ptr += 16;
                outptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(outptr);
                __m256 _p1 = _mm256_loadu_ps(ptr);
                // TODO: Instruction for remainder
                // _p = xxxxx(_p, _p1);
                // _mm256_storeu_ps(outptr, _p);

                ptr += 8;
                outptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_load_ps(outptr);
                __m128 _p1 = _mm_load_ps(ptr);
                // TODO: Instruction for remainder
                // _p = xxxxx(_p, _p1);
                // _mm_store_ps(outptr, _p);

                ptr += 4;
                outptr += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                *outptr = remainderf(*outptr, *ptr);

                ptr++;
                outptr++;
            }
        }
    }

    return 0;
}

} // namespace ncnn
