// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

RotaryEmbed_powerpc::RotaryEmbed_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

#if __ALTIVEC__
// vectorized fp32 rope kernel, identical semantics to the generic implementation
// with 4-wide elementwise inner loops
static int rotaryembed_forward_fp32(const Mat& bottom_blob, const Mat& cos_cache, const Mat& sin_cache, Mat& top_blob, int interleaved, const Option& opt)
{
    const int embed_dim = bottom_blob.w;
    const int seqlen = bottom_blob.h;
    const int num_heads = bottom_blob.c;

    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_heads; q++)
    {
        const Mat head = bottom_blob.channel(q);
        Mat out_head = top_blob.channel(q);

        for (int i = 0; i < seqlen; i++)
        {
            if (interleaved)
            {
                const float* ptr = head.row(i);
                const float* cos_ptr = cos_cache.row(i);
                const float* sin_ptr = sin_cache.row(i);
                float* outptr = out_head.row(i);

                const v16u8 _perm_even = {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
                const v16u8 _perm_odd = {4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};

                int j = 0;
                for (; j + 4 <= embed_dim / 2; j += 4)
                {
                    // 8 floats, four (x0,x1) pairs across two vectors
                    v4f32 _a = __ncnn_altivec_load(ptr + j * 2);
                    v4f32 _b = __ncnn_altivec_load(ptr + j * 2 + 4);

                    v4f32 _x0 = vec_perm(_a, _b, _perm_even);
                    v4f32 _x1 = vec_perm(_a, _b, _perm_odd);

                    v4f32 _cos = __ncnn_altivec_load(cos_ptr + j);
                    v4f32 _sin = __ncnn_altivec_load(sin_ptr + j);

                    v4f32 _oe = vec_nmsub(_x1, _sin, __ncnn_altivec_mul(_x0, _cos));
                    v4f32 _oo = vec_madd(_x0, _sin, __ncnn_altivec_mul(_x1, _cos));

                    // re-interleave the four output pairs
                    __ncnn_altivec_store(outptr + j * 2, vec_mergeh(_oe, _oo));
                    __ncnn_altivec_store(outptr + j * 2 + 4, vec_mergel(_oe, _oo));
                }
                ptr = head.row(i) + (size_t)j * 2;
                outptr = out_head.row(i) + (size_t)j * 2;
                cos_ptr = cos_cache.row(i) + j;
                sin_ptr = sin_cache.row(i) + j;
                for (; j < embed_dim / 2; j++)
                {
                    const float x0 = ptr[0];
                    const float x1 = ptr[1];
                    const float cos_val = *cos_ptr++;
                    const float sin_val = *sin_ptr++;
                    outptr[0] = x0 * cos_val - x1 * sin_val;
                    outptr[1] = x0 * sin_val + x1 * cos_val;
                    ptr += 2;
                    outptr += 2;
                }
            }
            else
            {
                const int half = embed_dim / 2;
                // A full-width (embed_dim) cache carries independent cos/sin for the two halves
                // (2D / vision rope where the halves differ); a half-width (embed_dim/2) cache is
                // the standard-rope form whose halves are identical, so the second half reuses the
                // first. This keeps existing half-width callers bit-identical.
                const int cw = cos_cache.w == embed_dim ? half : 0;
                const float* ptr0 = head.row(i);
                const float* ptr1 = ptr0 + half;
                const float* cos_ptr0 = cos_cache.row(i);
                const float* sin_ptr0 = sin_cache.row(i);
                const float* cos_ptr1 = cos_ptr0 + cw;
                const float* sin_ptr1 = sin_ptr0 + cw;
                float* outptr0 = out_head.row(i);
                float* outptr1 = outptr0 + half;

                int j = 0;
                for (; j + 4 <= half; j += 4)
                {
                    v4f32 _x0 = __ncnn_altivec_load(ptr0 + j);
                    v4f32 _x1 = __ncnn_altivec_load(ptr1 + j);
                    v4f32 _cos0 = __ncnn_altivec_load(cos_ptr0 + j);
                    v4f32 _sin0 = __ncnn_altivec_load(sin_ptr0 + j);
                    v4f32 _cos1 = __ncnn_altivec_load(cos_ptr1 + j);
                    v4f32 _sin1 = __ncnn_altivec_load(sin_ptr1 + j);

                    v4f32 _o0 = vec_nmsub(_x1, _sin0, __ncnn_altivec_mul(_x0, _cos0));
                    v4f32 _o1 = vec_madd(_x0, _sin1, __ncnn_altivec_mul(_x1, _cos1));

                    __ncnn_altivec_store(outptr0 + j, _o0);
                    __ncnn_altivec_store(outptr1 + j, _o1);
                }
                for (; j < half; j++)
                {
                    const float x0 = ptr0[j];
                    const float x1 = ptr1[j];
                    outptr0[j] = x0 * cos_ptr0[j] - x1 * sin_ptr0[j];
                    outptr1[j] = x0 * sin_ptr1[j] + x1 * cos_ptr1[j];
                }
            }
        }
    }

    return 0;
}
#endif // __ALTIVEC__

int RotaryEmbed_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // assert bottom_blobs.size() == 3

    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& cos_cache = bottom_blobs[1];
    const Mat& sin_cache = bottom_blobs[2];
    Mat& top_blob = top_blobs[0];

#if __ALTIVEC__
    if (bottom_blob.elembits() == 32)
    {
        // the vectorized kernel is not packing aware, unpack any packed blob first
        // note: the cos/sin caches are 2D and get packed along h, so all three
        // inputs must be unpacked
        bool any_packed = bottom_blob.elempack != 1 || cos_cache.elempack != 1 || sin_cache.elempack != 1;
        if (any_packed)
        {
            Mat bottom_blob_unpacked;
            Mat cos_cache_unpacked;
            Mat sin_cache_unpacked;
            if (bottom_blob.elempack != 1)
            {
                convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
                if (bottom_blob_unpacked.empty())
                    return -100;
            }
            if (cos_cache.elempack != 1)
            {
                convert_packing(cos_cache, cos_cache_unpacked, 1, opt);
                if (cos_cache_unpacked.empty())
                    return -100;
            }
            if (sin_cache.elempack != 1)
            {
                convert_packing(sin_cache, sin_cache_unpacked, 1, opt);
                if (sin_cache_unpacked.empty())
                    return -100;
            }

            return rotaryembed_forward_fp32(
                       bottom_blob.elempack != 1 ? bottom_blob_unpacked : bottom_blob,
                       cos_cache.elempack != 1 ? cos_cache_unpacked : cos_cache,
                       sin_cache.elempack != 1 ? sin_cache_unpacked : sin_cache,
                       top_blob, interleaved, opt);
        }

        return rotaryembed_forward_fp32(bottom_blob, cos_cache, sin_cache, top_blob, interleaved, opt);
    }
#endif // __ALTIVEC__

    // non-fp32 or altivec unavailable
    if (bottom_blob.elempack != 1 || cos_cache.elempack != 1 || sin_cache.elempack != 1)
    {
        Mat bottom_blob_unpacked;
        Mat cos_cache_unpacked;
        Mat sin_cache_unpacked;
        if (bottom_blob.elempack != 1)
        {
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
            if (bottom_blob_unpacked.empty())
                return -100;
        }
        if (cos_cache.elempack != 1)
        {
            convert_packing(cos_cache, cos_cache_unpacked, 1, opt);
            if (cos_cache_unpacked.empty())
                return -100;
        }
        if (sin_cache.elempack != 1)
        {
            convert_packing(sin_cache, sin_cache_unpacked, 1, opt);
            if (sin_cache_unpacked.empty())
                return -100;
        }

        std::vector<Mat> bottom_blobs_unpacked(3);
        bottom_blobs_unpacked[0] = bottom_blob.elempack != 1 ? bottom_blob_unpacked : bottom_blob;
        bottom_blobs_unpacked[1] = cos_cache.elempack != 1 ? cos_cache_unpacked : cos_cache;
        bottom_blobs_unpacked[2] = sin_cache.elempack != 1 ? sin_cache_unpacked : sin_cache;

        return RotaryEmbed::forward(bottom_blobs_unpacked, top_blobs, opt);
    }

    return RotaryEmbed::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
