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
#include <algorithm>
#include <math.h>
#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Yolov3DetectionOutput)

Yolov3DetectionOutput::Yolov3DetectionOutput()
{
	one_blob_only = false;
	support_inplace = false;
	
    //softmax = ncnn::create_layer(ncnn::LayerType::Softmax);

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 0);// axis

    //softmax->load_param(pd);
}

Yolov3DetectionOutput::~Yolov3DetectionOutput()
{
    //delete softmax;
}

int Yolov3DetectionOutput::load_param(const ParamDict& pd)
{
    num_class = pd.get(0, 20);
	//printf("%d\n", num_class);
    num_box = pd.get(1, 5);
	//printf("%d\n", num_box);
    confidence_threshold = pd.get(2, 0.01f);
	//printf("%f\n", confidence_threshold);
    nms_threshold = pd.get(3, 0.45f);
	//printf("%f\n", nms_threshold);
    biases = pd.get(4, Mat());
	//printf("%f %f\n", biases[0], biases[1]);
	mask = pd.get(5, Mat());
	//printf("%f %f %f\n", mask[0], mask[1], mask[2]);
	anchors_scale = pd.get(6, Mat());
	//printf("%f %f\n", anchors_scale[0], anchors_scale[1]);
	mask_group_num = pd.get(7,2);
	//printf("%d\n", mask_group_num);
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

template <typename T>
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

template <typename T>
static void qsort_descent_inplace(std::vector<T>& datas, std::vector<float>& scores)
{
    if (datas.empty() || scores.empty())
        return;

    qsort_descent_inplace(datas, scores, 0, scores.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<BBoxRect>& bboxes, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = bboxes.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        const BBoxRect& r = bboxes[i];

        float width = r.xmax - r.xmin;
        float height = r.ymax - r.ymin;

        areas[i] = width * height;
    }

    for (int i = 0; i < n; i++)
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

static inline float sigmoid(float x)
{
    return 1.f / (1.f + exp(-x));
}

int Yolov3DetectionOutput::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
	//const Mat& bottom_top_blob2 = **bottom_top_blob[0];

	//int w2 = bottom_top_blob2.w;
	//int h2 = bottom_top_blob2.h;
	//int channels2 = bottom_top_blob2.c;

	//printf("%d %d %d\n", w2, h2, channels2);

	// gather all box
	std::vector<BBoxRect> all_bbox_rects;
	std::vector<float> all_bbox_scores;

	for (size_t b = 0; b < bottom_blobs.size(); b++)
	{
		std::vector< std::vector<BBoxRect> > all_box_bbox_rects;
		std::vector< std::vector<float> > all_box_bbox_scores;
		all_box_bbox_rects.resize(num_box);
		all_box_bbox_scores.resize(num_box);
		const Mat& bottom_top_blobs = bottom_blobs[b];

		int w = bottom_top_blobs.w;
		int h = bottom_top_blobs.h;
		int channels = bottom_top_blobs.c;
		//printf("%d %d %d\n", w, h, channels);
		const int channels_per_box = channels / num_box;

		// anchor coord + box score + num_class
		if (channels_per_box != 4 + 1 + num_class)
			return -1;
		int mask_offset = b * num_box;
		int net_w = (int)(anchors_scale[b] * w);
		int net_h = (int)(anchors_scale[b] * h);
		//printf("%d %d\n", net_w, net_h);

		//printf("%d %d %d\n", w, h, channels);
#pragma omp parallel for num_threads(opt.num_threads)
		for (int pp = 0; pp < num_box; pp++)
		{
			int p = pp * channels_per_box;
			int biases_index = mask[pp + mask_offset];
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
					// region box
					float bbox_cx = (j + sigmoid(xptr[0])) / w;
					float bbox_cy = (i + sigmoid(yptr[0])) / h;
					float bbox_w = exp(wptr[0]) * bias_w / 416;
					float bbox_h = exp(hptr[0]) * bias_h / 416;

					float bbox_xmin = bbox_cx - bbox_w * 0.5f;
					float bbox_ymin = bbox_cy - bbox_h * 0.5f;
					float bbox_xmax = bbox_cx + bbox_w * 0.5f;
					float bbox_ymax = bbox_cy + bbox_h * 0.5f;

					// box score
					float box_score = sigmoid(box_score_ptr[0]);

					// find class index with max class score
					int class_index = 0;
					float class_score = 0.f;
					for (int q = 0; q < num_class; q++)
					{
						float score = sigmoid(scores.channel(q).row(i)[j]);
						if (score > class_score)
						{
							class_index = q;
							class_score = score;
						}
					}

					//printf( "%d %f %f\n", class_index, box_score, class_score);

					float confidence = box_score * class_score;
					if (confidence >= confidence_threshold)
					{
						BBoxRect c = { bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, class_index };
						all_box_bbox_rects[pp].push_back(c);
						all_box_bbox_scores[pp].push_back(confidence);
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
			const std::vector<float>& box_bbox_scores = all_box_bbox_scores[i];

			all_bbox_rects.insert(all_bbox_rects.end(), box_bbox_rects.begin(), box_bbox_rects.end());
			all_bbox_scores.insert(all_bbox_scores.end(), box_bbox_scores.begin(), box_bbox_scores.end());
		}

	}
	

    // global sort inplace
    qsort_descent_inplace(all_bbox_rects, all_bbox_scores);

    // apply nms
    std::vector<int> picked;
    nms_sorted_bboxes(all_bbox_rects, picked, nms_threshold);

    // select
    std::vector<BBoxRect> bbox_rects;
    std::vector<float> bbox_scores;

    for (int i = 0; i < (int)picked.size(); i++)
    {
        int z = picked[i];
        bbox_rects.push_back(all_bbox_rects[z]);
        bbox_scores.push_back(all_bbox_scores[z]);
    }

    // fill result
    int num_detected = bbox_rects.size();
	Mat& top_blob = top_blobs[0];
	top_blob.create(6, num_detected, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    for (int i = 0; i < num_detected; i++)
    {
        const BBoxRect& r = bbox_rects[i];
        float score = bbox_scores[i];
        float* outptr = top_blob.row(i);

        outptr[0] = r.label + 1;// +1 for prepend background class
        outptr[1] = score;
        outptr[2] = r.xmin;
        outptr[3] = r.ymin;
        outptr[4] = r.xmax;
        outptr[5] = r.ymax;
    }

    return 0;
}

} // namespace ncnn
