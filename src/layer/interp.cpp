// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "interp.h"

namespace ncnn {

Interp::Interp()
{
    one_blob_only = true;
    support_inplace = false;
}

int Interp::load_param(const ParamDict& pd)
{
    resize_type = pd.get(0, 0);
    height_scale = pd.get(1, 1.f);
    width_scale = pd.get(2, 1.f);
    output_height = pd.get(3, 0);
    output_width = pd.get(4, 0);
    dynamic_target_size = pd.get(5, 0);
    align_corner = pd.get(6, 0);

    if (resize_type < 0 || resize_type > 3)
    {
        NCNN_LOGE("unsupported resize type %d", resize_type);
        return -1;
    }

    if (dynamic_target_size == 1)
    {
        one_blob_only = false;
    }

    return 0;
}

#if defined(__GNUC__) && defined(__powerpc__) && defined(__ALTIVEC__)
// NOTE gcc altivec optimized version produce wrong result
// so I have to disable vectorize here  --- nihui
__attribute__((optimize("no-tree-vectorize")))
#endif
static void
linear_coeffs(int w, int outw, int* xofs, float* alpha, int align_corner)
{
    double scale = (double)w / outw;
    if (align_corner)
    {
        scale = (double)(w - 1) / (outw - 1);
    }

    for (int dx = 0; dx < outw; dx++)
    {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        if (align_corner)
        {
            fx = static_cast<float>(dx * scale);
        }

        int sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= w - 1)
        {
            sx = w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        alpha[dx * 2] = 1.f - fx;
        alpha[dx * 2 + 1] = fx;
    }
}

static void resize_bilinear_image(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w);
    Mat rowsbuf1(w);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const float* S1 = src.row(sy + 1);

            const float* alphap = alpha;
            float* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const float* S0 = src.row(sy);
            const float* S1 = src.row(sy + 1);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows0p[dx] = S0p[0] * a0 + S0p[1] * a1;
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst.row(dy);
        for (int dx = 0; dx < w; dx++)
        {
            //             D[x] = rows0[x]*b0 + rows1[x]*b1;
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }

        beta += 2;
    }
}

static inline void interpolate_cubic(float fx, float* coeffs)
{
    const float A = -0.75f;

    float fx0 = fx + 1;
    float fx1 = fx;
    float fx2 = 1 - fx;
    // float fx3 = 2 - fx;

    coeffs[0] = A * fx0 * fx0 * fx0 - 5 * A * fx0 * fx0 + 8 * A * fx0 - 4 * A;
    coeffs[1] = (A + 2) * fx1 * fx1 * fx1 - (A + 3) * fx1 * fx1 + 1;
    coeffs[2] = (A + 2) * fx2 * fx2 * fx2 - (A + 3) * fx2 * fx2 + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

static void cubic_coeffs(int w, int outw, int* xofs, float* alpha)
{
    double scale = (double)w / outw;

    for (int dx = 0; dx < outw; dx++)
    {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        int sx = static_cast<int>(floor(fx));
        fx -= sx;

        interpolate_cubic(fx, alpha + dx * 4);

        if (sx <= -1)
        {
            sx = 1;
            alpha[dx * 4 + 0] = 1.f - alpha[dx * 4 + 3];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 3];
            alpha[dx * 4 + 2] = 0.f;
            alpha[dx * 4 + 3] = 0.f;
        }
        if (sx == 0)
        {
            sx = 1;
            alpha[dx * 4 + 0] = alpha[dx * 4 + 0] + alpha[dx * 4 + 1];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 2];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 3];
            alpha[dx * 4 + 3] = 0.f;
        }
        if (sx == w - 2)
        {
            sx = w - 3;
            alpha[dx * 4 + 3] = alpha[dx * 4 + 2] + alpha[dx * 4 + 3];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 1];
            alpha[dx * 4 + 1] = alpha[dx * 4 + 0];
            alpha[dx * 4 + 0] = 0.f;
        }
        if (sx >= w - 1)
        {
            sx = w - 3;
            alpha[dx * 4 + 3] = 1.f - alpha[dx * 4 + 0];
            alpha[dx * 4 + 2] = alpha[dx * 4 + 0];
            alpha[dx * 4 + 1] = 0.f;
            alpha[dx * 4 + 0] = 0.f;
        }

        xofs[dx] = sx;
    }
}

static void resize_bicubic_image(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    // loop body
    Mat rowsbuf0(w);
    Mat rowsbuf1(w);
    Mat rowsbuf2(w);
    Mat rowsbuf3(w);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;
    float* rows2 = rowsbuf2;
    float* rows3 = rowsbuf3;

    int prev_sy1 = -3;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows2;
            rows2 = rows3;
            rows3 = rows0_old;
            const float* S3 = src.row(sy + 2);

            const float* alphap = alpha;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 2)
        {
            // hresize two rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            rows0 = rows2;
            rows1 = rows3;
            rows2 = rows0_old;
            rows3 = rows1_old;
            const float* S2 = src.row(sy + 1);
            const float* S3 = src.row(sy + 2);

            const float* alphap = alpha;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 3)
        {
            // hresize three rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            float* rows2_old = rows2;
            rows0 = rows3;
            rows1 = rows0_old;
            rows2 = rows1_old;
            rows3 = rows2_old;
            const float* S1 = src.row(sy);
            const float* S2 = src.row(sy + 1);
            const float* S3 = src.row(sy + 2);

            const float* alphap = alpha;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows1p[dx] = S1p[-1] * a0 + S1p[0] * a1 + S1p[1] * a2 + S1p[2] * a3;
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }
        else
        {
            // hresize four rows
            const float* S0 = src.row(sy - 1);
            const float* S1 = src.row(sy);
            const float* S2 = src.row(sy + 1);
            const float* S3 = src.row(sy + 2);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;
                const float* S2p = S2 + sx;
                const float* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows0p[dx] = S0p[-1] * a0 + S0p[0] * a1 + S0p[1] * a2 + S0p[2] * a3;
                rows1p[dx] = S1p[-1] * a0 + S1p[0] * a1 + S1p[1] * a2 + S1p[2] * a3;
                rows2p[dx] = S2p[-1] * a0 + S2p[0] * a1 + S2p[1] * a2 + S2p[2] * a3;
                rows3p[dx] = S3p[-1] * a0 + S3p[0] * a1 + S3p[1] * a2 + S3p[2] * a3;

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];
        float b2 = beta[2];
        float b3 = beta[3];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* rows2p = rows2;
        float* rows3p = rows3;
        float* Dp = dst.row(dy);
        for (int dx = 0; dx < w; dx++)
        {
            //             D[x] = rows0[x]*b0 + rows1[x]*b1 + rows2[x]*b2 + rows3[x]*b3;
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1 + *rows2p++ * b2 + *rows3p++ * b3;
        }

        beta += 4;
    }
}

int Interp::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    int outw = output_width;
    int outh = output_height;
    if (bottom_blob.dims == 1)
    {
        w = 1;
        h = 1;
    }
    if (outw == 0 || outh == 0)
    {
        outw = static_cast<int>(w * width_scale);
        outh = static_cast<int>(h * height_scale);
    }

    Mat reference_blob;
    reference_blob.w = outw;
    reference_blob.h = outh;

    std::vector<Mat> bottom_blobs(2);
    bottom_blobs[0] = bottom_blob;
    bottom_blobs[1] = reference_blob;

    std::vector<Mat> top_blobs(1);

    int ret = forward(bottom_blobs, top_blobs, opt);

    top_blob = top_blobs[0];

    return ret;
}

int Interp::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int outw = reference_blob.w;
    int outh = reference_blob.h;

    if (bottom_blob.dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < w; q++)
        {
            Mat top_blob_c = top_blob.channel(q);
            const float v = bottom_blob[q];
            top_blob_c.fill(v);
        }

        return 0;
    }

    if (outw == w && outh == h)
    {
        top_blob = bottom_blob;
        return 0;
    }

    top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (resize_type == 1) // nearest
    {
        const float hs = outh ? h / (float)outh : 1.f / height_scale;
        const float ws = outw ? w / (float)outw : 1.f / width_scale;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);
            for (int y = 0; y < outh; y++)
            {
                int in_y = std::min((int)(y * hs), (h - 1));
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    *outptr++ = ptr[in_y * w + in_x];
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
        for (int q = 0; q < channels; ++q)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bilinear_image(src, dst, alpha, xofs, beta, yofs);
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

        cubic_coeffs(w, outw, xofs, alpha);
        cubic_coeffs(h, outh, yofs, beta);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bicubic_image(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    return 0;
}

} // namespace ncnn
