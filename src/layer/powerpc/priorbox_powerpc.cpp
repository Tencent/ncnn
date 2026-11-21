// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "priorbox_powerpc.h"

#include <math.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

PriorBox_powerpc::PriorBox_powerpc()
{
    // one_blob_only = false, support_inplace = false, support_packing = false, all inherited
}

#if __ALTIVEC__

// emit the 4 corner coordinates {cx - cw, cy - ch, cx + cw, cy + ch} of one prior box
static NCNN_FORCEINLINE void priorbox_emit_box(float* box, float cx, float cy, float cw, float ch)
{
    const v4f32 _base = (v4f32) {
        cx, cy, cx, cy
    };
    const v4f32 _delta = (v4f32) {
        cw, ch, cw, ch
    };
    const v4f32 _sign = (v4f32) {
        -1.f, -1.f, 1.f, 1.f
        };

    __ncnn_altivec_store(box, vec_madd(_delta, _sign, _base));
}

// emit the 4 corner coordinates normalized by the image size
// box[0] = (cx - bw * 0.5f) / image_w etc, matching the generic implementation
static NCNN_FORCEINLINE void priorbox_emit_box_norm(float* box, float cx, float cy, float bw, float bh,
        const v4f32& _image, const v4f32& _sign)
{
    const v4f32 _base = (v4f32) {
        cx, cy, cx, cy
    };
    const v4f32 _half = (v4f32) {
        bw * 0.5f, bh * 0.5f, bw * 0.5f, bh * 0.5f
    };

    const v4f32 _corner = vec_madd(_half, _sign, _base);

    __ncnn_altivec_store(box, __ncnn_altivec_div2(_corner, _image));
}

static void priorbox_clip_inplace(float* box, int n)
{
    const v4f32 _zero = vec_splats(0.f);
    const v4f32 _one = vec_splats(1.f);

    int i = 0;
    for (; i + 4 <= n; i += 4)
    {
        v4f32 _v = __ncnn_altivec_load(box + i);
        _v = vec_max(_v, _zero);
        _v = vec_min(_v, _one);
        __ncnn_altivec_store(box + i, _v);
    }
    for (; i < n; i++)
    {
        box[i] = std::min(std::max(box[i], 0.f), 1.f);
    }
}

#endif // __ALTIVEC__

int PriorBox_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if __ALTIVEC__
    int w = bottom_blobs[0].w;
    int h = bottom_blobs[0].h;

    if (bottom_blobs.size() == 1 && image_width == -233 && image_height == -233 && max_sizes.empty())
    {
        // mxnet style _contrib_MultiBoxPrior
        float step_w = step_width;
        float step_h = step_height;
        if (step_w == -233)
            step_w = 1.f / (float)w;
        if (step_h == -233)
            step_h = 1.f / (float)h;

        int num_sizes = min_sizes.w;
        int num_ratios = aspect_ratios.w;

        int num_prior = num_sizes - 1 + num_ratios;

        Mat& top_blob = top_blobs[0];
        top_blob.create(4 * w * h * num_prior, 4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* box = (float*)top_blob + i * w * num_prior * 4;

            float center_x = offset * step_w;
            float center_y = offset * step_h + i * step_h;

            for (int j = 0; j < w; j++)
            {
                // ratio = 1, various sizes
                for (int k = 0; k < num_sizes; k++)
                {
                    float size = min_sizes[k];
                    float cw = size * h / w / 2;
                    float ch = size / 2;

                    priorbox_emit_box(box, center_x, center_y, cw, ch);
                    box += 4;
                }

                // various ratios, size = min_size = size[0]
                float size = min_sizes[0];
                for (int p = 1; p < num_ratios; p++)
                {
                    float ratio = sqrtf(aspect_ratios[p]);
                    float cw = size * h / w * ratio / 2;
                    float ch = size / ratio / 2;

                    priorbox_emit_box(box, center_x, center_y, cw, ch);
                    box += 4;
                }

                center_x += step_w;
            }
        }

        if (clip)
        {
            priorbox_clip_inplace(top_blob, top_blob.w);
        }

        return 0;
    }

    int image_w = image_width;
    int image_h = image_height;
    if (image_w == -233)
        image_w = bottom_blobs[1].w;
    if (image_h == -233)
        image_h = bottom_blobs[1].h;

    float step_w = step_width;
    float step_h = step_height;
    if (step_w == -233)
    {
        step_w = (float)image_w / w;
        if (step_mmdetection)
            step_w = ceilf((float)image_w / w);
    }
    if (step_h == -233)
    {
        step_h = (float)image_h / h;
        if (step_mmdetection)
            step_h = ceilf((float)image_h / h);
    }

    int num_min_size = min_sizes.w;
    int num_max_size = max_sizes.w;
    int num_aspect_ratio = aspect_ratios.w;

    int num_prior = num_min_size * num_aspect_ratio + num_min_size + num_max_size;
    if (flip)
        num_prior += num_min_size * num_aspect_ratio;

    Mat& top_blob = top_blobs[0];
    top_blob.create(4 * w * h * num_prior, 2, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const v4f32 _image = (v4f32) {
        (float)image_w, (float)image_h, (float)image_w, (float)image_h
    };
    const v4f32 _sign = (v4f32) {
        -1.f, -1.f, 1.f, 1.f
        };

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < h; i++)
    {
        float* box = (float*)top_blob + i * w * num_prior * 4;

        float center_x = offset * step_w;
        float center_y = offset * step_h + i * step_h;
        if (center_mmdetection)
        {
            center_x = offset * (step_w - 1);
            center_y = offset * (step_h - 1) + i * step_h;
        }

        for (int j = 0; j < w; j++)
        {
            float box_w;
            float box_h;

            for (int k = 0; k < num_min_size; k++)
            {
                float min_size = min_sizes[k];

                // min size box
                box_w = box_h = min_size;

                priorbox_emit_box_norm(box, center_x, center_y, box_w, box_h, _image, _sign);
                box += 4;

                if (num_max_size > 0)
                {
                    float max_size = max_sizes[k];

                    // max size box
                    box_w = box_h = sqrtf(min_size * max_size);

                    priorbox_emit_box_norm(box, center_x, center_y, box_w, box_h, _image, _sign);
                    box += 4;
                }

                // all aspect_ratios
                for (int p = 0; p < num_aspect_ratio; p++)
                {
                    float ar = aspect_ratios[p];

                    box_w = min_size * sqrtf(ar);
                    box_h = min_size / sqrtf(ar);

                    priorbox_emit_box_norm(box, center_x, center_y, box_w, box_h, _image, _sign);
                    box += 4;

                    if (flip)
                    {
                        priorbox_emit_box_norm(box, center_x, center_y, box_h, box_w, _image, _sign);
                        box += 4;
                    }
                }
            }

            center_x += step_w;
        }
    }

    if (clip)
    {
        priorbox_clip_inplace(top_blob, top_blob.w);
    }

    // set variance, one vector stores the 4 variance values of one prior box
    const v4f32 _variance = (v4f32) {
        variances[0], variances[1], variances[2], variances[3]
    };
    float* var = top_blob.row(1);
    for (int i = 0; i < top_blob.w / 4; i++)
    {
        __ncnn_altivec_store(var, _variance);
        var += 4;
    }

    return 0;
#else
    // no altivec, fall back to the generic implementation
    return PriorBox::forward(bottom_blobs, top_blobs, opt);
#endif // __ALTIVEC__
}

} // namespace ncnn
