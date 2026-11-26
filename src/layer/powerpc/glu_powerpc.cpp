// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "glu_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#include "powerpc_mathfun.h"
#endif // __ALTIVEC__

namespace ncnn {

GLU_powerpc::GLU_powerpc()
{
}

#if __ALTIVEC__
static void glu_sigmoid_mul_powerpc(const float* in_ptr, float* out_ptr, int n, int sigmoid_offset)
{
    int i = 0;
    v4f32 _zero = vec_splats(0.f);
    v4f32 _one = vec_splats(1.f);
    for (; i + 3 < n; i += 4)
    {
        v4f32 _x = __ncnn_altivec_load(in_ptr + i);
        v4f32 _sig_in = __ncnn_altivec_load(in_ptr + i + sigmoid_offset);

        // sigmoid = 1 / (1 + exp(-x))
        v4f32 _sigmoid = __ncnn_altivec_div2(_one, vec_add(_one, exp_ps(vec_sub(_zero, _sig_in))));

        v4f32 _out = __ncnn_altivec_mul(_x, _sigmoid);

        __ncnn_altivec_store(out_ptr + i, _out);
    }
    for (; i < n; i++)
    {
        float sigmoid = 1.f / (1.f + expf(-in_ptr[i + sigmoid_offset]));
        out_ptr[i] = in_ptr[i] * sigmoid;
    }
}
#endif // __ALTIVEC__

int GLU_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1)
    {   // ignore axis
        int w = bottom_blob.w;
        int out_w = w / 2;
        top_blob.create(out_w, sizeof(float), opt.blob_allocator);

        const float* in_ptr = bottom_blob;
        float* out_ptr = top_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int x = 0; x < out_w; ++x)
        {
            float sigmoid = 1.f / (1.f + expf(-in_ptr[x + out_w]));

            out_ptr[x] = in_ptr[x] * sigmoid;
        }

        return 0;
    } // if (dims == 1)

    if (dims == 2 && positive_axis == 0)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int out_w = w;
        int out_h = h / 2;
        top_blob.create(out_w, out_h, sizeof(float), opt.blob_allocator);

        int offset = out_w * out_h;

        int size = offset;
        const float* in_ptr = bottom_blob;
        float* out_ptr = top_blob;

#if __ALTIVEC__
        glu_sigmoid_mul_powerpc(in_ptr, out_ptr, size, offset);
        return 0;
#else
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; ++i)
        {
            float sigmoid = 1.f / (1.f + expf(-in_ptr[i + offset]));
            out_ptr[i] = in_ptr[i] * sigmoid;
        }

        return 0;
#endif
    } // if (dims == 2 && positive_axis == 0)

    if (dims == 2 && positive_axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int out_w = w / 2;
        int out_h = h;

        top_blob.create(out_w, out_h, sizeof(float), opt.blob_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; ++y)
        {
            const float* in_ptr = bottom_blob.row(y);
            float* out_ptr = top_blob.row(y);

#if __ALTIVEC__
            glu_sigmoid_mul_powerpc(in_ptr, out_ptr, out_w, out_w);
#else
            for (int x = 0; x < out_w; ++x)
            {
                float sigmoid = 1.f / (1.f + expf(-in_ptr[x + out_w]));
                out_ptr[x] = in_ptr[x] * sigmoid;
            }
#endif
        }

        return 0;
    } // if (dims == 2 && positive_axis == 1)

    if (dims == 3 && positive_axis == 0)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int c = bottom_blob.c;

        int out_w = w;
        int out_h = h;
        int out_c = c / 2;

        top_blob.create(out_w, out_h, out_c, sizeof(float), opt.blob_allocator);

        size_t offset = out_c * bottom_blob.cstep;
        int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < out_c; ++q)
        {
            const float* in_ptr = bottom_blob.channel(q);
            float* out_ptr = top_blob.channel(q);

#if __ALTIVEC__
            glu_sigmoid_mul_powerpc(in_ptr, out_ptr, size, offset);
#else
            for (int i = 0; i < size; ++i)
            {
                float sigmoid = 1.f / (1.f + expf(-in_ptr[i + offset]));
                out_ptr[i] = in_ptr[i] * sigmoid;
            }
#endif
        }
        return 0;
    } //   if (dims == 3 && positive_axis == 0) {

    if (dims == 3 && positive_axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int c = bottom_blob.c;

        int out_w = w;
        int out_h = h / 2;
        int out_c = c;

        top_blob.create(out_w, out_h, out_c, sizeof(float), opt.blob_allocator);

        int offset = out_h * out_w;
        int size = offset;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            const float* in_ptr = bottom_blob.channel(q);
            float* out_ptr = top_blob.channel(q);

#if __ALTIVEC__
            glu_sigmoid_mul_powerpc(in_ptr, out_ptr, size, offset);
#else
            for (int i = 0; i < size; ++i)
            {
                float sigmoid = 1.f / (1.f + expf(-in_ptr[i + offset]));
                out_ptr[i] = in_ptr[i] * sigmoid;
            }
#endif
        }
        return 0;
    } // if (dims == 3 && positive_axis == 1)

    if (dims == 3 && positive_axis == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int c = bottom_blob.c;

        int out_w = w / 2;
        int out_h = h;
        int out_c = c;

        top_blob.create(out_w, out_h, out_c, sizeof(float), opt.blob_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            const float* in_ptr = bottom_blob.channel(q);
            float* out_ptr = top_blob.channel(q);
            for (int y = 0; y < h; ++y)
            {
#if __ALTIVEC__
                glu_sigmoid_mul_powerpc(in_ptr, out_ptr, out_w, out_w);
#else
                for (int x = 0; x < out_w; ++x)
                {
                    float sigmoid = 1.f / (1.f + expf(-in_ptr[x + out_w]));
                    out_ptr[x] = in_ptr[x] * sigmoid;
                }
#endif
                in_ptr += w;
                out_ptr += out_w;
            }
        }
        return 0;
    } // if (dims == 3 && positive_axis == 2)

    if (dims == 4 && positive_axis == 0)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int c = bottom_blob.c;

        int out_w = w;
        int out_h = h;
        int out_d = d;
        int out_c = c / 2;

        top_blob.create(out_w, out_h, out_d, out_c, sizeof(float), opt.blob_allocator);

        size_t offset = out_c * bottom_blob.cstep;
        int size = w * h * d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < out_c; ++q)
        {
            const float* in_ptr = bottom_blob.channel(q);
            float* out_ptr = top_blob.channel(q);

#if __ALTIVEC__
            glu_sigmoid_mul_powerpc(in_ptr, out_ptr, size, offset);
#else
            for (int i = 0; i < size; ++i)
            {
                float sigmoid = 1.f / (1.f + expf(-in_ptr[i + offset]));
                out_ptr[i] = in_ptr[i] * sigmoid;
            }
#endif
        }
        return 0;
    } // if (dims == 4 && positive_axis == 0)

    if (dims == 4 && positive_axis == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int c = bottom_blob.c;

        int out_w = w;
        int out_h = h;
        int out_d = d / 2;
        int out_c = c;

        top_blob.create(out_w, out_h, out_d, out_c, sizeof(float), opt.blob_allocator);

        int offset = out_d * out_h * out_w;
        int size = offset;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            const float* in_ptr = bottom_blob.channel(q);
            float* out_ptr = top_blob.channel(q);

#if __ALTIVEC__
            glu_sigmoid_mul_powerpc(in_ptr, out_ptr, size, offset);
#else
            for (int i = 0; i < size; ++i)
            {
                float sigmoid = 1.f / (1.f + expf(-in_ptr[i + offset]));
                out_ptr[i] = in_ptr[i] * sigmoid;
            }
#endif
        }
        return 0;
    } // if (dims == 4 && positive_axis == 1)

    if (dims == 4 && positive_axis == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int c = bottom_blob.c;

        int out_w = w;
        int out_h = h / 2;
        int out_d = d;
        int out_c = c;

        top_blob.create(out_w, out_h, out_d, out_c, sizeof(float), opt.blob_allocator);

        int offset = out_h * out_w;
        int size = offset;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            for (int z = 0; z < d; ++z)
            {
                const float* in_ptr = bottom_blob.channel(q).depth(z);
                float* out_ptr = top_blob.channel(q).depth(z);

#if __ALTIVEC__
                glu_sigmoid_mul_powerpc(in_ptr, out_ptr, size, offset);
#else
                for (int i = 0; i < size; ++i)
                {
                    float sigmoid = 1.f / (1.f + expf(-in_ptr[i + offset]));
                    out_ptr[i] = in_ptr[i] * sigmoid;
                }
#endif
            }
        }
        return 0;
    } // if (dims == 4 && positive_axis == 2)

    if (dims == 4 && positive_axis == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int c = bottom_blob.c;

        int out_w = w / 2;
        int out_h = h;
        int out_d = d;
        int out_c = c;

        top_blob.create(out_w, out_h, out_d, out_c, sizeof(float), opt.blob_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            for (int z = 0; z < d; ++z)
            {
                const Mat m = bottom_blob.channel(q).depth(z);
                Mat outm = top_blob.channel(q).depth(z);
                for (int y = 0; y < h; ++y)
                {
                    const float* in_ptr = m.row(y);
                    float* out_ptr = outm.row(y);
#if __ALTIVEC__
                    glu_sigmoid_mul_powerpc(in_ptr, out_ptr, out_w, out_w);
#else
                    for (int x = 0; x < out_w; ++x)
                    {
                        float sigmoid = 1.f / (1.f + expf(-in_ptr[x + out_w]));
                        out_ptr[x] = in_ptr[x] * sigmoid;
                    }
#endif
                }
            }
        }
        return 0;
    } // if (dims == 4 && positive_axis == 3)

    return -100;
}

} // namespace ncnn
