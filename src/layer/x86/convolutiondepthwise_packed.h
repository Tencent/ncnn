// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__
void convolutiondepthwise_packed8_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int bias_term, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt);
#endif

static void convolutiondepthwise_packed8(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int bias_term, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__
    if (ncnn::cpu_support_x86_fma())
    {
        convolutiondepthwise_packed8_fma(bottom_blob, top_blob, weight_data, bias_data, bias_term, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
        return;
    }
#endif

    const int w = bottom_blob.w;
    const int channels = bottom_blob.c;
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < channels; g++)
    {
        float* outptr = top_blob.channel(g);
        const float* kptr = (const float*)weight_data + maxk * g * 8;
        const Mat m = bottom_blob.channel(g);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                __m256 _sum = _mm256_set1_ps(0.f);

                if (bias_term)
                {
                    _sum = _mm256_loadu_ps(((const float*)bias_data) + g * 8);
                }

                const float* sptr = m.row(i * stride_h) + j * stride_w * 8;

                for (int k = 0; k < maxk; k++)
                {
                    __m256 _val = _mm256_loadu_ps(sptr + space_ofs[k] * 8);
                    __m256 _w = _mm256_loadu_ps(kptr + k * 8);
                    _sum = _mm256_comp_fmadd_ps(_val, _w, _sum);
                }

                _mm256_storeu_ps(outptr + j * 8, _sum);
            }

            outptr += outw * 8;
        }
    }
}
