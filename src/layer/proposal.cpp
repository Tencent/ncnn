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

#include "proposal.h"

#include <math.h>

namespace ncnn {

Proposal::Proposal()
{
    one_blob_only = false;
    support_inplace = false;

    // TODO load from param
    ratios.create(3);
    ratios[0] = 0.5f;
    ratios[1] = 1.f;
    ratios[2] = 2.f;

    scales.create(3);
    scales[0] = 8.f;
    scales[1] = 16.f;
    scales[2] = 32.f;
}

static Mat generate_anchors(int base_size, const Mat& ratios, const Mat& scales)
{
    int num_ratio = ratios.w;
    int num_scale = scales.w;

    Mat anchors;
    anchors.create(4, num_ratio * num_scale);

    const float cx = base_size * 0.5f;
    const float cy = base_size * 0.5f;

    for (int i = 0; i < num_ratio; i++)
    {
        float ar = ratios[i];

        int r_w = static_cast<int>(round(base_size / sqrt(ar)));
        int r_h = static_cast<int>(round(r_w * ar)); //round(base_size * sqrt(ar));

        for (int j = 0; j < num_scale; j++)
        {
            float scale = scales[j];

            float rs_w = r_w * scale;
            float rs_h = r_h * scale;

            float* anchor = anchors.row(i * num_scale + j);

            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }

    return anchors;
}

int Proposal::load_param(const ParamDict& pd)
{
    feat_stride = pd.get(0, 16);
    base_size = pd.get(1, 16);
    pre_nms_topN = pd.get(2, 6000);
    after_nms_topN = pd.get(3, 300);
    nms_thresh = pd.get(4, 0.7f);
    min_size = pd.get(5, 16);

    //     Mat ratio;
    //     Mat scale;

    anchors = generate_anchors(base_size, ratios, scales);

    return 0;
}

struct Rect
{
    float x1;
    float y1;
    float x2;
    float y2;
};

static inline float intersection_area(const Rect& a, const Rect& b)
{
    if (a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    float inter_height = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);

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

static void nms_sorted_bboxes(const std::vector<Rect>& bboxes, std::vector<size_t>& picked, float nms_threshold)
{
    picked.clear();

    const size_t n = bboxes.size();

    std::vector<float> areas(n);
    for (size_t i = 0; i < n; i++)
    {
        const Rect& r = bboxes[i];

        float width = r.x2 - r.x1;
        float height = r.y2 - r.y1;

        areas[i] = width * height;
    }

    for (size_t i = 0; i < n; i++)
    {
        const Rect& a = bboxes[i];

        int keep = 1;
        for (size_t j = 0; j < picked.size(); j++)
        {
            const Rect& b = bboxes[picked[j]];

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

int Proposal::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& score_blob = bottom_blobs[0];
    const Mat& bbox_blob = bottom_blobs[1];
    const Mat& im_info_blob = bottom_blobs[2];

    int w = score_blob.w;
    int h = score_blob.h;

    // generate proposals from bbox deltas and shifted anchors
    const int num_anchors = anchors.h;

    Mat proposals;
    proposals.create(4, w * h, num_anchors);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_anchors; q++)
    {
        const float* bbox_xptr = bbox_blob.channel(q * 4);
        const float* bbox_yptr = bbox_blob.channel(q * 4 + 1);
        const float* bbox_wptr = bbox_blob.channel(q * 4 + 2);
        const float* bbox_hptr = bbox_blob.channel(q * 4 + 3);

        Mat pbs = proposals.channel(q);

        const float* anchor = anchors.row(q);

        // shifted anchor
        float anchor_y = anchor[1];

        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];

        for (int i = 0; i < h; i++)
        {
            float anchor_x = anchor[0];

            for (int j = 0; j < w; j++)
            {
                float* pb = pbs.row(i * w + j);

                // apply center size
                float dx = bbox_xptr[j];
                float dy = bbox_yptr[j];
                float dw = bbox_wptr[j];
                float dh = bbox_hptr[j];

                float cx = anchor_x + anchor_w * 0.5f;
                float cy = anchor_y + anchor_h * 0.5f;

                float pb_cx = cx + anchor_w * dx;
                float pb_cy = cy + anchor_h * dy;

                float pb_w = static_cast<float>(anchor_w * exp(dw));
                float pb_h = static_cast<float>(anchor_h * exp(dh));

                pb[0] = pb_cx - pb_w * 0.5f;
                pb[1] = pb_cy - pb_h * 0.5f;
                pb[2] = pb_cx + pb_w * 0.5f;
                pb[3] = pb_cy + pb_h * 0.5f;

                anchor_x += feat_stride;
            }

            bbox_xptr += w;
            bbox_yptr += w;
            bbox_wptr += w;
            bbox_hptr += w;

            anchor_y += feat_stride;
        }
    }

    // clip predicted boxes to image
    float im_w = im_info_blob[1];
    float im_h = im_info_blob[0];

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_anchors; q++)
    {
        Mat pbs = proposals.channel(q);

        for (int i = 0; i < w * h; i++)
        {
            float* pb = pbs.row(i);

            // clip box
            pb[0] = std::max(std::min(pb[0], im_w - 1), 0.f);
            pb[1] = std::max(std::min(pb[1], im_h - 1), 0.f);
            pb[2] = std::max(std::min(pb[2], im_w - 1), 0.f);
            pb[3] = std::max(std::min(pb[3], im_h - 1), 0.f);
        }
    }

    // remove predicted boxes with either height or width < threshold
    std::vector<Rect> proposal_boxes;
    std::vector<float> scores;

    float im_scale = im_info_blob[2];
    float min_boxsize = min_size * im_scale;

    for (int q = 0; q < num_anchors; q++)
    {
        Mat pbs = proposals.channel(q);
        const float* scoreptr = score_blob.channel(q + num_anchors);

        for (int i = 0; i < w * h; i++)
        {
            float* pb = pbs.row(i);

            float pb_w = pb[2] - pb[0] + 1;
            float pb_h = pb[3] - pb[1] + 1;

            if (pb_w >= min_boxsize && pb_h >= min_boxsize)
            {
                Rect r = {pb[0], pb[1], pb[2], pb[3]};
                proposal_boxes.push_back(r);
                scores.push_back(scoreptr[i]);
            }
        }
    }

    // sort all (proposal, score) pairs by score from highest to lowest
    qsort_descent_inplace(proposal_boxes, scores);

    // take top pre_nms_topN
    if (pre_nms_topN > 0 && pre_nms_topN < (int)proposal_boxes.size())
    {
        proposal_boxes.resize(pre_nms_topN);
        scores.resize(pre_nms_topN);
    }

    // apply nms with nms_thresh
    std::vector<size_t> picked;
    nms_sorted_bboxes(proposal_boxes, picked, nms_thresh);

    // take after_nms_topN
    int picked_count = std::min((int)picked.size(), after_nms_topN);

    // return the top proposals
    Mat& roi_blob = top_blobs[0];
    roi_blob.create(4, 1, picked_count);
    if (roi_blob.empty())
        return -100;

    for (int i = 0; i < picked_count; i++)
    {
        float* outptr = roi_blob.channel(i);

        outptr[0] = proposal_boxes[picked[i]].x1;
        outptr[1] = proposal_boxes[picked[i]].y1;
        outptr[2] = proposal_boxes[picked[i]].x2;
        outptr[3] = proposal_boxes[picked[i]].y2;
    }

    if (top_blobs.size() > 1)
    {
        Mat& roi_score_blob = top_blobs[1];
        roi_score_blob.create(1, 1, picked_count);
        if (roi_score_blob.empty())
            return -100;

        for (int i = 0; i < picked_count; i++)
        {
            float* outptr = roi_score_blob.channel(i);
            outptr[0] = scores[picked[i]];
        }
    }

    return 0;
}

} // namespace ncnn
