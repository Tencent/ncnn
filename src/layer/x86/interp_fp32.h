// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
int interp_forward_fma(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, int outw, int outh, int resize_type, int align_corner, float height_scale, float width_scale, int output_height, int output_width, int has_size_expr, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
int interp_forward_fma4(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, int outw, int outh, int resize_type, int align_corner, float height_scale, float width_scale, int output_height, int output_width, int has_size_expr, const Option& opt);
#endif

static int interp_forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, int outw, int outh, int resize_type, int align_corner, float height_scale, float width_scale, int output_height, int output_width, int has_size_expr, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
        return interp_forward_fma(bottom_blobs, top_blobs, outw, outh, resize_type, align_corner, height_scale, width_scale, output_height, output_width, has_size_expr, opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
        return interp_forward_fma4(bottom_blobs, top_blobs, outw, outh, resize_type, align_corner, height_scale, width_scale, output_height, output_width, has_size_expr, opt);
#endif

    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    const int h = bottom_blob.h;
    const int w = bottom_blob.w;
    const int channels = bottom_blob.c;
    const int dims = bottom_blob.dims;
    const size_t elemsize = bottom_blob.elemsize;
    const int elempack = bottom_blob.elempack;

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                __m512 _v = _mm512_loadu_ps((const float*)bottom_blob + q * 16);
                top_blob_c.fill(_v);
            }

            return 0;
        }
#endif // __AVX512F__

        if (elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                __m256 _v = _mm256_load_ps((const float*)bottom_blob + q * 8);
                top_blob_c.fill(_v);
            }

            return 0;
        }
#endif // __AVX__

        if (elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                __m128 _v = _mm_load_ps((const float*)bottom_blob + q * 4);
                top_blob_c.fill(_v);
            }

            return 0;
        }
#endif // __SSE2__

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < w; q++)
        {
            Mat top_blob_c = top_blob.channel(q);
            const float v = bottom_blob[q];
            top_blob_c.fill(v);
        }

        return 0;
    }

    if (dims == 2)
    {
        if (outw == w)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (resize_type == 1) // nearest
        {
            const float ws = (output_width || has_size_expr) ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.row(y);
                float* outptr = top_blob.row(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    const float* Sp = ptr + in_x * elempack;

                    int ep = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    for (; ep + 15 < elempack; ep += 16)
                    {
                        __m512 _p = _mm512_load_ps(Sp + ep);
                        _mm512_store_ps(outptr + ep, _p);
                    }
#endif // __AVX512F__
                    for (; ep + 7 < elempack; ep += 8)
                    {
                        __m256 _p = _mm256_load_ps(Sp + ep);
                        _mm256_store_ps(outptr + ep, _p);
                    }
#endif // __AVX__
                    for (; ep + 3 < elempack; ep += 4)
                    {
                        __m128 _p = _mm_load_ps(Sp + ep);
                        _mm_store_ps(outptr + ep, _p);
                    }
#endif // __SSE2__
                    for (; ep < elempack; ep++)
                    {
                        outptr[ep] = Sp[ep];
                    }

                    outptr += elempack;
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outw * 2];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            linear_coeffs(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.row(y);
                float* outptr = top_blob.row(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x] * elempack;
                    const float* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];

                    int ep = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    {
                        __m512 _a0 = _mm512_set1_ps(a0);
                        __m512 _a1 = _mm512_set1_ps(a1);
                        for (; ep + 15 < elempack; ep += 16)
                        {
                            __m512 _S0 = _mm512_load_ps(Sp + ep);
                            __m512 _S1 = _mm512_load_ps(Sp + ep + elempack);
                            __m512 _p = _mm512_mul_ps(_S0, _a0);
                            _p = _mm512_fmadd_ps(_S1, _a1, _p);
                            _mm512_store_ps(outptr + ep, _p);
                        }
                    }
#endif // __AVX512F__
                    {
                        __m256 _a0 = _mm256_set1_ps(a0);
                        __m256 _a1 = _mm256_set1_ps(a1);
                        for (; ep + 7 < elempack; ep += 8)
                        {
                            __m256 _S0 = _mm256_load_ps(Sp + ep);
                            __m256 _S1 = _mm256_load_ps(Sp + ep + elempack);
                            __m256 _p = _mm256_mul_ps(_S0, _a0);
                            _p = _mm256_comp_fmadd_ps(_S1, _a1, _p);
                            _mm256_store_ps(outptr + ep, _p);
                        }
                    }
#endif // __AVX__
                    {
                        __m128 _a0 = _mm_set1_ps(a0);
                        __m128 _a1 = _mm_set1_ps(a1);
                        for (; ep + 3 < elempack; ep += 4)
                        {
                            __m128 _S0 = _mm_load_ps(Sp + ep);
                            __m128 _S1 = _mm_load_ps(Sp + ep + elempack);
                            __m128 _p = _mm_mul_ps(_S0, _a0);
                            _p = _mm_comp_fmadd_ps(_S1, _a1, _p);
                            _mm_store_ps(outptr + ep, _p);
                        }
                    }
#endif // __SSE2__
                    for (; ep < elempack; ep++)
                    {
                        outptr[ep] = Sp[ep] * a0 + Sp[ep + elempack] * a1;
                    }

                    alphap += 2;
                    outptr += elempack;
                }
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outw * 4];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            cubic_coeffs(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.row(y);
                float* outptr = top_blob.row(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x] * elempack;
                    const float* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    float a2 = alphap[2];
                    float a3 = alphap[3];

                    int ep = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    {
                        __m512 _a0 = _mm512_set1_ps(a0);
                        __m512 _a1 = _mm512_set1_ps(a1);
                        __m512 _a2 = _mm512_set1_ps(a2);
                        __m512 _a3 = _mm512_set1_ps(a3);
                        for (; ep + 15 < elempack; ep += 16)
                        {
                            __m512 _S0 = _mm512_load_ps(Sp + ep - elempack);
                            __m512 _S1 = _mm512_load_ps(Sp + ep);
                            __m512 _S2 = _mm512_load_ps(Sp + ep + elempack);
                            __m512 _S3 = _mm512_load_ps(Sp + ep + elempack * 2);
                            __m512 _p = _mm512_mul_ps(_S0, _a0);
                            _p = _mm512_fmadd_ps(_S1, _a1, _p);
                            _p = _mm512_fmadd_ps(_S2, _a2, _p);
                            _p = _mm512_fmadd_ps(_S3, _a3, _p);
                            _mm512_store_ps(outptr + ep, _p);
                        }
                    }
#endif // __AVX512F__
                    {
                        __m256 _a0 = _mm256_set1_ps(a0);
                        __m256 _a1 = _mm256_set1_ps(a1);
                        __m256 _a2 = _mm256_set1_ps(a2);
                        __m256 _a3 = _mm256_set1_ps(a3);
                        for (; ep + 7 < elempack; ep += 8)
                        {
                            __m256 _S0 = _mm256_load_ps(Sp + ep - elempack);
                            __m256 _S1 = _mm256_load_ps(Sp + ep);
                            __m256 _S2 = _mm256_load_ps(Sp + ep + elempack);
                            __m256 _S3 = _mm256_load_ps(Sp + ep + elempack * 2);
                            __m256 _p = _mm256_mul_ps(_S0, _a0);
                            _p = _mm256_comp_fmadd_ps(_S1, _a1, _p);
                            _p = _mm256_comp_fmadd_ps(_S2, _a2, _p);
                            _p = _mm256_comp_fmadd_ps(_S3, _a3, _p);
                            _mm256_store_ps(outptr + ep, _p);
                        }
                    }
#endif // __AVX__
                    {
                        __m128 _a0 = _mm_set1_ps(a0);
                        __m128 _a1 = _mm_set1_ps(a1);
                        __m128 _a2 = _mm_set1_ps(a2);
                        __m128 _a3 = _mm_set1_ps(a3);
                        for (; ep + 3 < elempack; ep += 4)
                        {
                            __m128 _S0 = _mm_load_ps(Sp + ep - elempack);
                            __m128 _S1 = _mm_load_ps(Sp + ep);
                            __m128 _S2 = _mm_load_ps(Sp + ep + elempack);
                            __m128 _S3 = _mm_load_ps(Sp + ep + elempack * 2);
                            __m128 _p = _mm_mul_ps(_S0, _a0);
                            _p = _mm_comp_fmadd_ps(_S1, _a1, _p);
                            _p = _mm_comp_fmadd_ps(_S2, _a2, _p);
                            _p = _mm_comp_fmadd_ps(_S3, _a3, _p);
                            _mm_store_ps(outptr + ep, _p);
                        }
                    }
#endif // __SSE2__
                    for (; ep < elempack; ep++)
                    {
                        outptr[ep] = Sp[ep - elempack] * a0 + Sp[ep] * a1 + Sp[ep + elempack] * a2 + Sp[ep + elempack * 2] * a3;
                    }

                    alphap += 4;
                    outptr += elempack;
                }
            }

            delete[] buf;
        }

        return 0;
    }

    if (outw == w && outh == h)
    {
        top_blob = bottom_blob;
        return 0;
    }

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (resize_type == 1) // nearest
    {
        const float hs = (output_height || has_size_expr) ? h / (float)outh : 1.f / height_scale;
        const float ws = (output_width || has_size_expr) ? w / (float)outw : 1.f / width_scale;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            for (int y = 0; y < outh; y++)
            {
                int in_y = std::min((int)(y * hs), (h - 1));

                const float* ptr = src.row(in_y);
                float* outptr = dst.row(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    const float* Sp = ptr + in_x * elempack;

                    memcpy(outptr, Sp, elempack * sizeof(float));

                    outptr += elempack;
                }
            }
        }
    }

    if (resize_type == 2) // bilinear
    {
        int* buf = new int[outw + outh + outw * 2 + outh * 2];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
        float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

        linear_coeffs(w, outw, xofs, alpha, align_corner);
        linear_coeffs(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

#if __SSE2__
#if __AVX__
#if __AVX512F__
            if (elempack == 16)
            {
                resize_bilinear_image_pack16(src, dst, alpha, xofs, beta, yofs);
            }
#endif // __AVX512F__
            if (elempack == 8)
            {
                resize_bilinear_image_pack8(src, dst, alpha, xofs, beta, yofs);
            }
#endif // __AVX__
            if (elempack == 4)
            {
                resize_bilinear_image_pack4(src, dst, alpha, xofs, beta, yofs);
            }
#endif // __SSE2__
            if (elempack == 1)
            {
                resize_bilinear_image(src, dst, alpha, xofs, beta, yofs);
            }
        }

        delete[] buf;
    }

    if (resize_type == 3) // bicubic
    {
        int* buf = new int[outw + outh + outw * 4 + outh * 4];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
        float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

        cubic_coeffs(w, outw, xofs, alpha, align_corner);
        cubic_coeffs(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

#if __SSE2__
#if __AVX__
#if __AVX512F__
            if (elempack == 16)
            {
                resize_bicubic_image_pack16(src, dst, alpha, xofs, beta, yofs);
            }
#endif // __AVX512F__
            if (elempack == 8)
            {
                resize_bicubic_image_pack8(src, dst, alpha, xofs, beta, yofs);
            }
#endif // __AVX__
            if (elempack == 4)
            {
                resize_bicubic_image_pack4(src, dst, alpha, xofs, beta, yofs);
            }
#endif // __SSE2__
            if (elempack == 1)
            {
                resize_bicubic_image(src, dst, alpha, xofs, beta, yofs);
            }
        }

        delete[] buf;
    }

    return 0;
}
