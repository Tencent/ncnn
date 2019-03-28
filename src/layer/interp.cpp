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
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(Interp);

Interp::Interp()
{
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = true;

#if NCNN_VULKAN
    pipeline_interp = 0;
    pipeline_interp_pack4 = 0;
#endif // NCNN_VULKAN
}

int Interp::load_param(const ParamDict& pd)
{
    resize_type = pd.get(0, 0);
    height_scale = pd.get(1, 1.f);
    width_scale = pd.get(2, 1.f);
    output_height = pd.get(3, 0);
    output_width = pd.get(4, 0);

    return 0;
}

static void resize_bilinear_image(const Mat& src, Mat& dst)
{
    int w = dst.w;
    int h = dst.w;

    double scale_x = (double)src.w / w;
    double scale_y = (double)src.h / h;

    int* buf = new int[w + h + w*2 + h*2];

    int* xofs = buf;//new int[w];
    int* yofs = buf + w;//new int[h];

    float* alpha = (float*)(buf + w + h);//new float[w * 2];
    float* beta = (float*)(buf + w + h + w*2);//new float[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

    for (int dx = 0; dx < w; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = floor(fx);
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= src.w - 1)
        {
            sx = src.w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        alpha[dx*2    ] = 1.f - fx;
        alpha[dx*2 + 1] = fx;
    }

    for (int dy = 0; dy < h; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = floor(fy);
        fy -= sy;

        if (sy < 0)
        {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= src.h - 1)
        {
            sy = src.h - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        beta[dy*2    ] = 1.f - fy;
        beta[dy*2 + 1] = fy;
    }

    // loop body
    Mat rowsbuf0(w + 1);
    Mat rowsbuf1(w + 1);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++ )
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
            const float* S1 = src.row(sy+1);

            const float* alphap = alpha;
            float* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = S1p[0]*a0 + S1p[1]*a1;

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const float* S0 = src.row(sy);
            const float* S1 = src.row(sy+1);

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
                rows0p[dx] = S0p[0]*a0 + S0p[1]*a1;
                rows1p[dx] = S1p[0]*a0 + S1p[1]*a1;

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

    delete[] buf;
}

int Interp::forward(const Mat &bottom_blob, Mat &top_blob, const Option& opt) const
{
    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int c = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int oh = output_height;
    int ow = output_width;
    if (bottom_blob.dims == 1)
    {
        h = 1;
        w = 1;
        c = bottom_blob.w;
    }
    if (oh == 0 || ow == 0)
    {
        oh = h * height_scale;
        ow = w * width_scale;
    }
    if (oh == h && ow == w)
    {
        top_blob = bottom_blob;
        return 0;
    }
    top_blob.create(ow, oh, c, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (bottom_blob.dims == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            Mat top_blob_c = top_blob.channel(q);
            const float *ptr = ((const float*)bottom_blob.data + q);
            top_blob_c.fill(*ptr);
        }
        return 0;
    }

    if (resize_type == 1)//nearest
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            const float *ptr = bottom_blob.channel(q);
            float *output_ptr = top_blob.channel(q);
            for (int y = 0; y < oh; ++y)
            {
                const int in_y = std::min((int) (y / height_scale), (h - 1));
                for (int x = 0; x < ow; ++x)
                {
                    const int in_x = std::min((int) (x / width_scale), (w - 1));
                    output_ptr[ow * y + x] = ptr[in_y * w + in_x];
                }
            }
        }
        return 0;

    }
    else if (resize_type == 2)// bilinear
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bilinear_image(src, dst);
        }

        return 0;
    }
    else
    {
        fprintf(stderr, "unsupported resize type %d %d %d\n", resize_type, oh, ow);
        return -233;
    }
}

#if NCNN_VULKAN
int Interp::create_pipeline()
{
    pipeline_interp = new Pipeline(vkdev);
    pipeline_interp->set_optimal_local_size_xyz();

    std::vector<vk_specialization_type> specializations(1);
    specializations[0].i = resize_type;

    pipeline_interp->create("interp", specializations, 2, 12);

    // pack4
    {
        pipeline_interp_pack4 = new Pipeline(vkdev);
        pipeline_interp_pack4->set_optimal_local_size_xyz();
        pipeline_interp_pack4->create("interp_pack4", specializations, 2, 12);
    }

    return 0;
}

int Interp::destroy_pipeline()
{
    delete pipeline_interp;
    pipeline_interp = 0;

    delete pipeline_interp_pack4;
    pipeline_interp_pack4 = 0;

    return 0;
}

int Interp::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int packing = bottom_blob.packing;

    int outw = output_width;
    int outh = output_height;
    if (outw == 0 || outh == 0)
    {
        outw = w * width_scale;
        outh = h * height_scale;
    }

    if (outh == h && outw == w)
    {
        top_blob = bottom_blob;
        return 0;
    }

    top_blob.create(outw, outh, channels, elemsize, packing, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

//     fprintf(stderr, "Interp::forward %p %p\n", bottom_blob.buffer(), top_blob.buffer());

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(12);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = bottom_blob.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;
    constants[10].f = w / (float)outw;
    constants[11].f = h / (float)outh;

    const Pipeline* pipeline = packing == 4 ? pipeline_interp_pack4 : pipeline_interp;

    // record
    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
