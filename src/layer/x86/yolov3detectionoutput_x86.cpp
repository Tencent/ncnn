// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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
#if __AVX__
#include <immintrin.h>
#endif

#include "yolov3detectionoutput_x86.h"

#include <float.h>
#include <math.h>

namespace ncnn {

Yolov3DetectionOutput_x86::Yolov3DetectionOutput_x86()
{
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

int Yolov3DetectionOutput_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    // gather all box
    std::vector<BBoxRect> all_bbox_rects;

    for (size_t b = 0; b < bottom_blobs.size(); b++)
    {
        std::vector<std::vector<BBoxRect> > all_box_bbox_rects;
        all_box_bbox_rects.resize(num_box);
        const Mat& bottom_top_blobs = bottom_blobs[b];

        int w = bottom_top_blobs.w;
        int h = bottom_top_blobs.h;
        int channels = bottom_top_blobs.c;
        //printf("%d %d %d\n", w, h, channels);
        const int channels_per_box = channels / num_box;

        // anchor coord + box score + num_class
        if (channels_per_box != 4 + 1 + num_class)
            return -1;
        size_t mask_offset = b * num_box;
        int net_w = (int)(anchors_scale[b] * w);
        int net_h = (int)(anchors_scale[b] * h);
        //printf("%d %d\n", net_w, net_h);

        //printf("%d %d %d\n", w, h, channels);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < num_box; pp++)
        {
            int p = pp * channels_per_box;
            int biases_index = static_cast<int>(mask[pp + mask_offset]);
            //printf("%d\n", biases_index);
            const float bias_w = biases[biases_index * 2];
            const float bias_h = biases[biases_index * 2 + 1];
            //printf("%f %f\n", bias_w, bias_h);
            const float* xptr = bottom_top_blobs.channel(p);
            const float* yptr = bottom_top_blobs.channel(p + 1);
            const float* wptr = bottom_top_blobs.channel(p + 2);
            const float* hptr = bottom_top_blobs.channel(p + 3);

            const float* box_score_ptr = bottom_top_blobs.channel(p + 4);

            // softmax class scores
            Mat scores = bottom_top_blobs.channel_range(p + 5, num_class);
            //softmax->forward_inplace(scores, opt);

            const int cs = scores.cstep;

#if __AVX__
            const __m256i vi = _mm256_setr_epi32(
                                   0, cs * 1, cs * 2, cs * 3, cs * 4, cs * 5, cs * 6, cs * 7);
#endif

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
#if 0
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int q = 0; q < num_class; q++)
                    {
                        float score = scores.channel(q).row(i)[j];
                        if (score > class_score)
                        {
                            class_index = q;
                            class_score = score;
                        }
                    }
#else
                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    float* ptr = ((float*)scores.data) + i * w + j;
                    float* end = ptr + num_class * cs;
                    int q = 0;
#if __AVX__
                    float* end8 = ptr + (num_class & -8) * cs;
                    unsigned long index;

                    for (; ptr < end8; ptr += 8 * cs, q += 8)
                    {
                        __m256 p = _mm256_i32gather_ps(ptr, vi, 4);
                        __m256 t = _mm256_max_ps(p, _mm256_permute2f128_ps(p, p, 1));
                        t = _mm256_max_ps(t, _mm256_permute_ps(t, 0x4e));
                        t = _mm256_max_ps(t, _mm256_permute_ps(t, 0xb1));
                        float score = _mm_cvtss_f32(_mm256_extractf128_ps(t, 0));

                        if (score > class_score)
                        {
                            __m256 mi = _mm256_cmp_ps(p, t, _CMP_EQ_OQ);
                            int mask = _mm256_movemask_ps(mi);
#ifdef _MSC_VER
                            BitScanForward(&index, mask);
#else
                            index = __builtin_ctz(mask);
#endif
                            class_index = q + index;
                            class_score = score;
                        }
                    }
#endif

                    for (; ptr < end; ptr += cs, q++)
                    {
                        if (*ptr > class_score)
                        {
                            class_index = q;
                            class_score = *ptr;
                        }
                    }
#endif
                    //sigmoid(box_score) * sigmoid(class_score)
                    float confidence = 1.f / ((1.f + exp(-box_score_ptr[0]) * (1.f + exp(-class_score))));
                    if (confidence >= confidence_threshold)
                    {
                        // region box
                        float bbox_cx = (j + sigmoid(xptr[0])) / w;
                        float bbox_cy = (i + sigmoid(yptr[0])) / h;
                        float bbox_w = static_cast<float>(exp(wptr[0]) * bias_w / net_w);
                        float bbox_h = static_cast<float>(exp(hptr[0]) * bias_h / net_h);

                        float bbox_xmin = bbox_cx - bbox_w * 0.5f;
                        float bbox_ymin = bbox_cy - bbox_h * 0.5f;
                        float bbox_xmax = bbox_cx + bbox_w * 0.5f;
                        float bbox_ymax = bbox_cy + bbox_h * 0.5f;

                        float area = bbox_w * bbox_h;

                        BBoxRect c = {confidence, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, area, class_index};
                        all_box_bbox_rects[pp].push_back(c);
                    }

                    xptr++;
                    yptr++;
                    wptr++;
                    hptr++;

                    box_score_ptr++;
                }
            }
        }

        for (int i = 0; i < num_box; i++)
        {
            const std::vector<BBoxRect>& box_bbox_rects = all_box_bbox_rects[i];

            all_bbox_rects.insert(all_bbox_rects.end(), box_bbox_rects.begin(), box_bbox_rects.end());
        }
    }

    // global sort inplace
    qsort_descent_inplace(all_bbox_rects);

    // apply nms
    std::vector<size_t> picked;
    nms_sorted_bboxes(all_bbox_rects, picked, nms_threshold);

    // select
    std::vector<BBoxRect> bbox_rects;

    for (size_t i = 0; i < picked.size(); i++)
    {
        size_t z = picked[i];
        bbox_rects.push_back(all_bbox_rects[z]);
    }

    // fill result
    int num_detected = static_cast<int>(bbox_rects.size());
    if (num_detected == 0)
        return 0;

    Mat& top_blob = top_blobs[0];
    top_blob.create(6, num_detected, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    for (int i = 0; i < num_detected; i++)
    {
        const BBoxRect& r = bbox_rects[i];
        float score = r.score;
        float* outptr = top_blob.row(i);

        outptr[0] = static_cast<float>(r.label + 1); // +1 for prepend background class
        outptr[1] = score;
        outptr[2] = r.xmin;
        outptr[3] = r.ymin;
        outptr[4] = r.xmax;
        outptr[5] = r.ymax;
    }

    return 0;
}

} // namespace ncnn
