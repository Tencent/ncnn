// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "yolov3detectionoutput.h"

#include "layer_type.h"

#include <float.h>
#include <math.h>

namespace ncnn {

Yolov3DetectionOutput::Yolov3DetectionOutput()
{
    one_blob_only = false;
    support_inplace = false;

    //softmax = ncnn::create_layer(ncnn::LayerType::Softmax);

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 0); // axis

    //softmax->load_param(pd);
}

Yolov3DetectionOutput::~Yolov3DetectionOutput()
{
    //delete softmax;
}

int Yolov3DetectionOutput::load_param(const ParamDict& pd)
{
    num_class = pd.get(0, 20);
    num_box = pd.get(1, 5);
    confidence_threshold = pd.get(2, 0.01f);
    nms_threshold = pd.get(3, 0.45f);
    biases = pd.get(4, Mat());
    mask = pd.get(5, Mat());
    anchors_scale = pd.get(6, Mat());
    return 0;
}

static inline float intersection_area(const Yolov3DetectionOutput::BBoxRect& a, const Yolov3DetectionOutput::BBoxRect& b)
{
    if (a.xmin > b.xmax || a.xmax < b.xmin || a.ymin > b.ymax || a.ymax < b.ymin)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin);
    float inter_height = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin);

    return inter_width * inter_height;
}

void Yolov3DetectionOutput::qsort_descent_inplace(std::vector<BBoxRect>& datas, int left, int right) const
{
    int i = left;
    int j = right;
    float p = datas[(left + right) / 2].score;

    while (i <= j)
    {
        while (datas[i].score > p)
            i++;

        while (datas[j].score < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(datas[i], datas[j]);

            i++;
            j--;
        }
    }

    if (left < j)
        qsort_descent_inplace(datas, left, j);

    if (i < right)
        qsort_descent_inplace(datas, i, right);
}

void Yolov3DetectionOutput::qsort_descent_inplace(std::vector<BBoxRect>& datas) const
{
    if (datas.empty())
        return;

    qsort_descent_inplace(datas, 0, static_cast<int>(datas.size() - 1));
}

void Yolov3DetectionOutput::nms_sorted_bboxes(std::vector<BBoxRect>& bboxes, std::vector<size_t>& picked, float nms_threshold) const
{
    picked.clear();

    const size_t n = bboxes.size();

    for (size_t i = 0; i < n; i++)
    {
        const BBoxRect& a = bboxes[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const BBoxRect& b = bboxes[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = a.area + b.area - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area > nms_threshold * union_area)
            {
                keep = 0;
                break;
            }
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

int Yolov3DetectionOutput::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    // find class index with max class score
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
