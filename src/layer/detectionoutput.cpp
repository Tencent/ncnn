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

#include "detectionoutput.h"

#include <math.h>

namespace ncnn {

DetectionOutput::DetectionOutput()
{
    one_blob_only = false;
    support_inplace = false;
}

int DetectionOutput::load_param(const ParamDict& pd)
{
    num_class = pd.get(0, 0);
    nms_threshold = pd.get(1, 0.05f);
    nms_top_k = pd.get(2, 300);
    keep_top_k = pd.get(3, 100);
    confidence_threshold = pd.get(4, 0.5f);
    variances[0] = pd.get(5, 0.1f);
    variances[1] = pd.get(6, 0.1f);
    variances[2] = pd.get(7, 0.2f);
    variances[3] = pd.get(8, 0.2f);

    return 0;
}

struct BBoxRect
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    int label;
};

static inline float intersection_area(const BBoxRect& a, const BBoxRect& b)
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

template<typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores, int left, int right)
{
    int i = left;
    int j = right;
    float p = scores[(left + right) / 2];

    while (i <= j)
    {
        while (scores[i] > p)
            i++;

        while (scores[j] < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(datas[i], datas[j]);
            std::swap(scores[i], scores[j]);

            i++;
            j--;
        }
    }

    if (left < j)
        qsort_descent_inplace(datas, scores, left, j);

    if (i < right)
        qsort_descent_inplace(datas, scores, i, right);
}

template<typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores)
{
    if (datas.empty() || scores.empty())
        return;

    qsort_descent_inplace(datas, scores, 0, static_cast<int>(scores.size() - 1));
}

static void nms_sorted_bboxes(const std::vector<BBoxRect>& bboxes, std::vector<size_t>& picked, float nms_threshold)
{
    picked.clear();

    const size_t n = bboxes.size();

    std::vector<float> areas(n);
    for (size_t i = 0; i < n; i++)
    {
        const BBoxRect& r = bboxes[i];

        float width = r.xmax - r.xmin;
        float height = r.ymax - r.ymin;

        areas[i] = width * height;
    }

    for (size_t i = 0; i < n; i++)
    {
        const BBoxRect& a = bboxes[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const BBoxRect& b = bboxes[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

int DetectionOutput::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& location = bottom_blobs[0];
    const Mat& confidence = bottom_blobs[1];
    const Mat& priorbox = bottom_blobs[2];

    bool mxnet_ssd_style = num_class == -233;

    // mxnet-ssd _contrib_MultiBoxDetection
    const int num_prior = mxnet_ssd_style ? priorbox.h : priorbox.w / 4;

    int num_class_copy = mxnet_ssd_style ? confidence.h : num_class;

    // apply location with priorbox
    Mat bboxes;
    bboxes.create(4, num_prior, 4u, opt.workspace_allocator);
    if (bboxes.empty())
        return -100;

    const float* location_ptr = location;
    const float* priorbox_ptr = priorbox.row(0);
    const float* variance_ptr = mxnet_ssd_style ? 0 : priorbox.row(1);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < num_prior; i++)
    {
        const float* loc = location_ptr + i * 4;
        const float* pb = priorbox_ptr + i * 4;
        const float* var = variance_ptr ? variance_ptr + i * 4 : variances;

        float* bbox = bboxes.row(i);

        // CENTER_SIZE
        float pb_w = pb[2] - pb[0];
        float pb_h = pb[3] - pb[1];
        float pb_cx = (pb[0] + pb[2]) * 0.5f;
        float pb_cy = (pb[1] + pb[3]) * 0.5f;

        float bbox_cx = var[0] * loc[0] * pb_w + pb_cx;
        float bbox_cy = var[1] * loc[1] * pb_h + pb_cy;
        float bbox_w = static_cast<float>(exp(var[2] * loc[2]) * pb_w);
        float bbox_h = static_cast<float>(exp(var[3] * loc[3]) * pb_h);

        bbox[0] = bbox_cx - bbox_w * 0.5f;
        bbox[1] = bbox_cy - bbox_h * 0.5f;
        bbox[2] = bbox_cx + bbox_w * 0.5f;
        bbox[3] = bbox_cy + bbox_h * 0.5f;
    }

    // sort and nms for each class
    std::vector<std::vector<BBoxRect> > all_class_bbox_rects;
    std::vector<std::vector<float> > all_class_bbox_scores;
    all_class_bbox_rects.resize(num_class_copy);
    all_class_bbox_scores.resize(num_class_copy);

    // start from 1 to ignore background class
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 1; i < num_class_copy; i++)
    {
        // filter by confidence_threshold
        std::vector<BBoxRect> class_bbox_rects;
        std::vector<float> class_bbox_scores;

        for (int j = 0; j < num_prior; j++)
        {
            // prob data layout
            // caffe-ssd = num_class x num_prior
            // mxnet-ssd = num_prior x num_class
            float score = mxnet_ssd_style ? confidence[i * num_prior + j] : confidence[j * num_class_copy + i];

            if (score > confidence_threshold)
            {
                const float* bbox = bboxes.row(j);
                BBoxRect c = {bbox[0], bbox[1], bbox[2], bbox[3], i};
                class_bbox_rects.push_back(c);
                class_bbox_scores.push_back(score);
            }
        }

        // sort inplace
        qsort_descent_inplace(class_bbox_rects, class_bbox_scores);

        // keep nms_top_k
        if (nms_top_k < (int)class_bbox_rects.size())
        {
            class_bbox_rects.resize(nms_top_k);
            class_bbox_scores.resize(nms_top_k);
        }

        // apply nms
        std::vector<size_t> picked;
        nms_sorted_bboxes(class_bbox_rects, picked, nms_threshold);

        // select
        for (size_t j = 0; j < picked.size(); j++)
        {
            size_t z = picked[j];
            all_class_bbox_rects[i].push_back(class_bbox_rects[z]);
            all_class_bbox_scores[i].push_back(class_bbox_scores[z]);
        }
    }

    // gather all class
    std::vector<BBoxRect> bbox_rects;
    std::vector<float> bbox_scores;

    for (int i = 1; i < num_class_copy; i++)
    {
        const std::vector<BBoxRect>& class_bbox_rects = all_class_bbox_rects[i];
        const std::vector<float>& class_bbox_scores = all_class_bbox_scores[i];

        bbox_rects.insert(bbox_rects.end(), class_bbox_rects.begin(), class_bbox_rects.end());
        bbox_scores.insert(bbox_scores.end(), class_bbox_scores.begin(), class_bbox_scores.end());
    }

    // global sort inplace
    qsort_descent_inplace(bbox_rects, bbox_scores);

    // keep_top_k
    if (keep_top_k < (int)bbox_rects.size())
    {
        bbox_rects.resize(keep_top_k);
        bbox_scores.resize(keep_top_k);
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
        float score = bbox_scores[i];
        float* outptr = top_blob.row(i);

        outptr[0] = static_cast<float>(r.label);
        outptr[1] = score;
        outptr[2] = r.xmin;
        outptr[3] = r.ymin;
        outptr[4] = r.xmax;
        outptr[5] = r.ymax;
    }

    return 0;
}

} // namespace ncnn
