// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "proposal_powerpc.h"

#include <math.h>

#if __ALTIVEC__
#include "powerpc_mathfun.h"
#include "powerpc_usability.h"
#endif // __ALTIVEC__

#include <algorithm>
#include <vector>

namespace ncnn {

Proposal_powerpc::Proposal_powerpc()
{
    // one_blob_only = false, support_inplace = false, support_packing = false, all inherited
    // anchors are generated in Proposal::load_param, no create_pipeline needed
}

#if __ALTIVEC__

struct ProposalRect
{
    float x1;
    float y1;
    float x2;
    float y2;
};

static inline float proposal_intersection_area(const ProposalRect& a, const ProposalRect& b)
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
static void proposal_qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores, int left, int right)
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
        proposal_qsort_descent_inplace(datas, scores, left, j);

    if (i < right)
        proposal_qsort_descent_inplace(datas, scores, i, right);
}

template<typename T>
static void proposal_qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores)
{
    if (datas.empty() || scores.empty())
        return;

    proposal_qsort_descent_inplace(datas, scores, 0, static_cast<int>(scores.size() - 1));
}

static void proposal_nms_sorted_bboxes(const std::vector<ProposalRect>& bboxes, std::vector<size_t>& picked, float nms_threshold)
{
    picked.clear();

    const size_t n = bboxes.size();

    std::vector<float> areas(n);
    for (size_t i = 0; i < n; i++)
    {
        const ProposalRect& r = bboxes[i];

        float width = r.x2 - r.x1;
        float height = r.y2 - r.y1;

        areas[i] = width * height;
    }

    for (size_t i = 0; i < n; i++)
    {
        const ProposalRect& a = bboxes[i];

        int keep = 1;
        for (size_t j = 0; j < picked.size(); j++)
        {
            const ProposalRect& b = bboxes[picked[j]];

            // intersection over union
            float inter_area = proposal_intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

#endif // __ALTIVEC__

int Proposal_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if __ALTIVEC__
    const Mat& score_blob = bottom_blobs[0];
    const Mat& bbox_blob = bottom_blobs[1];
    const Mat& im_info_blob = bottom_blobs[2];

    int w = score_blob.w;
    int h = score_blob.h;

    // generate proposals from bbox deltas and shifted anchors
    const int num_anchors = anchors.h;

    Mat proposals;
    proposals.create(4, w * h, num_anchors);

    const v4f32 _sign = (v4f32) {
        -1.f, -1.f, 1.f, 1.f
        };

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

                // exp of the width / height deltas, lanes {exp(dw), exp(dh), exp(dw), exp(dh)}
                // scaled by the anchor size: pb_w = anchor_w * expf(dw), pb_h = anchor_h * expf(dh)
                const v4f32 _dh = (v4f32) {
                    dw, dh, dw, dh
                };
                const v4f32 _anchor_wh = (v4f32) {
                    anchor_w, anchor_h, anchor_w, anchor_h
                };
                const v4f32 _pb_wh = exp_ps(_dh);
                const v4f32 _half_wh = __ncnn_altivec_mul(_pb_wh, __ncnn_altivec_mul(_anchor_wh, vec_splats(0.5f)));

                const v4f32 _base = (v4f32) {
                    cx, cy, cx, cy
                };
                const v4f32 _shift = (v4f32) {
                    anchor_w * dx, anchor_h * dy, anchor_w * dx, anchor_h * dy
                };

                const v4f32 _center = vec_add(_shift, _base);

                // pb[0] = pb_cx - pb_w * 0.5f, pb[1] = pb_cy - pb_h * 0.5f
                // pb[2] = pb_cx + pb_w * 0.5f, pb[3] = pb_cy + pb_h * 0.5f
                __ncnn_altivec_store(pb, vec_madd(_half_wh, _sign, _center));

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

    const v4f32 _im_wh = (v4f32) {
        im_w - 1, im_h - 1, im_w - 1, im_h - 1
    };
    const v4f32 _zero = vec_splats(0.f);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < num_anchors; q++)
    {
        // all w * h boxes of one anchor channel are contiguous, clamp x pairs and y pairs lane-wise
        float* pb = proposals.channel(q);

        int n = w * h;
        for (int i = 0; i < n; i++)
        {
            v4f32 _v = __ncnn_altivec_load(pb);
            _v = vec_max(_v, _zero);
            _v = vec_min(_v, _im_wh);
            __ncnn_altivec_store(pb, _v);

            pb += 4;
        }
    }

    // remove predicted boxes with either height or width < threshold
    // the branchy filter with push_back stays scalar
    std::vector<ProposalRect> proposal_boxes;
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
                ProposalRect r = {pb[0], pb[1], pb[2], pb[3]};
                proposal_boxes.push_back(r);
                scores.push_back(scoreptr[i]);
            }
        }
    }

    // sort all (proposal, score) pairs by score from highest to lowest
    proposal_qsort_descent_inplace(proposal_boxes, scores);

    // take top pre_nms_topN
    if (pre_nms_topN > 0 && pre_nms_topN < (int)proposal_boxes.size())
    {
        proposal_boxes.resize(pre_nms_topN);
        scores.resize(pre_nms_topN);
    }

    // apply nms with nms_thresh
    std::vector<size_t> picked;
    proposal_nms_sorted_bboxes(proposal_boxes, picked, nms_thresh);

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
#else
    // no altivec, fall back to the generic implementation
    return Proposal::forward(bottom_blobs, top_blobs, opt);
#endif // __ALTIVEC__
}

} // namespace ncnn
