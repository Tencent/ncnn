// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// 1. install
//      pip3 install -U ultralytics pnnx ncnn
// 2. export yolo11-pose torchscript
//      yolo export model=yolo11n-pose.pt format=torchscript
// 3. convert torchscript with static shape
//      pnnx yolo11n-pose.torchscript
// 4. modify yolo11n_pose_pnnx.py for dynamic shape inference
//      A. modify reshape to support dynamic image sizes
//      B. permute tensor before concat and adjust concat axis
//      C. drop post-process part
//      before:
//          v_195 = v_194.view(1, 51, 6400)
//          v_201 = v_200.view(1, 51, 1600)
//          v_207 = v_206.view(1, 51, 400)
//          v_208 = torch.cat((v_195, v_201, v_207), dim=-1)
//          ...
//          v_254 = v_223.view(1, 65, 6400)
//          v_255 = v_238.view(1, 65, 1600)
//          v_256 = v_253.view(1, 65, 400)
//          v_257 = torch.cat((v_254, v_255, v_256), dim=2)
//          ...
//      after:
//          v_195 = v_194.view(1, 51, -1).transpose(1, 2)
//          v_201 = v_200.view(1, 51, -1).transpose(1, 2)
//          v_207 = v_206.view(1, 51, -1).transpose(1, 2)
//          v_208 = torch.cat((v_195, v_201, v_207), dim=1)
//          ...
//          v_254 = v_223.view(1, 65, -1).transpose(1, 2)
//          v_255 = v_238.view(1, 65, -1).transpose(1, 2)
//          v_256 = v_253.view(1, 65, -1).transpose(1, 2)
//          v_257 = torch.cat((v_254, v_255, v_256), dim=1)
//          return v_257, v_208
//      D. modify area attention for dynamic shape inference
//      before:
//          v_95 = self.model_10_m_0_attn_qkv_conv(v_94)
//          v_96 = v_95.view(1, 2, 128, 400)
//          v_97, v_98, v_99 = torch.split(tensor=v_96, dim=2, split_size_or_sections=(32,32,64))
//          v_100 = torch.transpose(input=v_97, dim0=-2, dim1=-1)
//          v_101 = torch.matmul(input=v_100, other=v_98)
//          v_102 = (v_101 * 0.176777)
//          v_103 = F.softmax(input=v_102, dim=-1)
//          v_104 = torch.transpose(input=v_103, dim0=-2, dim1=-1)
//          v_105 = torch.matmul(input=v_99, other=v_104)
//          v_106 = v_105.view(1, 128, 20, 20)
//          v_107 = v_99.reshape(1, 128, 20, 20)
//          v_108 = self.model_10_m_0_attn_pe_conv(v_107)
//          v_109 = (v_106 + v_108)
//          v_110 = self.model_10_m_0_attn_proj_conv(v_109)
//      after:
//          v_95 = self.model_10_m_0_attn_qkv_conv(v_94)
//          v_96 = v_95.view(1, 2, 128, -1)
//          v_97, v_98, v_99 = torch.split(tensor=v_96, dim=2, split_size_or_sections=(32,32,64))
//          v_100 = torch.transpose(input=v_97, dim0=-2, dim1=-1)
//          v_101 = torch.matmul(input=v_100, other=v_98)
//          v_102 = (v_101 * 0.176777)
//          v_103 = F.softmax(input=v_102, dim=-1)
//          v_104 = torch.transpose(input=v_103, dim0=-2, dim1=-1)
//          v_105 = torch.matmul(input=v_99, other=v_104)
//          v_106 = v_105.view(1, 128, v_95.size(2), v_95.size(3))
//          v_107 = v_99.reshape(1, 128, v_95.size(2), v_95.size(3))
//          v_108 = self.model_10_m_0_attn_pe_conv(v_107)
//          v_109 = (v_106 + v_108)
//          v_110 = self.model_10_m_0_attn_proj_conv(v_109)
// 5. re-export yolo11-pose torchscript
//      python3 -c 'import yolo11n_pose_pnnx; yolo11n_pose_pnnx.export_torchscript()'
// 6. convert new torchscript with dynamic shape
//      pnnx yolo11n_pose_pnnx.py.pt inputshape=[1,3,640,640] inputshape2=[1,3,320,320]
// 7. now you get ncnn model files
//      mv yolo11n_pose_pnnx.py.ncnn.param yolo11n_pose.ncnn.param
//      mv yolo11n_pose_pnnx.py.ncnn.bin yolo11n_pose.ncnn.bin

// the out blob would be a 2-dim tensor with w=65 h=8400
//
//        | bbox-reg 16 x 4       |score(1)|
//        +-----+-----+-----+-----+--------+
//        | dx0 | dy0 | dx1 | dy1 |   0.1  |
//   all /|     |     |     |     |        |
//  boxes |  .. |  .. |  .. |  .. |   0.0  |
//  (8400)|     |     |     |     |   .    |
//       \|     |     |     |     |   .    |
//        +-----+-----+-----+-----+--------+
//

//
//        | pose (51) |
//        +-----------+
//        |0.1........|
//   all /|           |
//  boxes |0.0........|
//  (8400)|     .     |
//       \|     .     |
//        +-----------+
//

#include "layer.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>

struct KeyPoint
{
    cv::Point2f p;
    float prob;
};

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<KeyPoint> keypoints;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    // #pragma omp parallel sections
    {
        // #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        // #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static void generate_proposals(const ncnn::Mat& pred, const ncnn::Mat& pred_points, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;

    const int reg_max_1 = 16;
    const int num_points = pred_points.w / 3;

    for (int y = 0; y < num_grid_y; y++)
    {
        for (int x = 0; x < num_grid_x; x++)
        {
            const ncnn::Mat pred_grid = pred.row_range(y * num_grid_x + x, 1);
            const ncnn::Mat pred_points_grid = pred_points.row_range(y * num_grid_x + x, 1).reshape(3, num_points);

            // find label with max score
            int label = 0;
            float score = sigmoid(pred_grid[reg_max_1 * 4]);

            if (score >= prob_threshold)
            {
                ncnn::Mat pred_bbox = pred_grid.range(0, reg_max_1 * 4).reshape(reg_max_1, 4).clone();

                {
                    ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                    ncnn::ParamDict pd;
                    pd.set(0, 1); // axis
                    pd.set(1, 1);
                    softmax->load_param(pd);

                    ncnn::Option opt;
                    opt.num_threads = 1;
                    opt.use_packing_layout = false;

                    softmax->create_pipeline(opt);

                    softmax->forward_inplace(pred_bbox, opt);

                    softmax->destroy_pipeline(opt);

                    delete softmax;
                }

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    const float* dis_after_sm = pred_bbox.row(k);
                    for (int l = 0; l < reg_max_1; l++)
                    {
                        dis += l * dis_after_sm[l];
                    }

                    pred_ltrb[k] = dis * stride;
                }

                float pb_cx = (x + 0.5f) * stride;
                float pb_cy = (y + 0.5f) * stride;

                float x0 = pb_cx - pred_ltrb[0];
                float y0 = pb_cy - pred_ltrb[1];
                float x1 = pb_cx + pred_ltrb[2];
                float y1 = pb_cy + pred_ltrb[3];

                std::vector<KeyPoint> keypoints;
                for (int k = 0; k < num_points; k++)
                {
                    KeyPoint keypoint;
                    keypoint.p.x = (x + pred_points_grid.row(k)[0] * 2) * stride;
                    keypoint.p.y = (y + pred_points_grid.row(k)[1] * 2) * stride;
                    keypoint.prob = sigmoid(pred_points_grid.row(k)[2]);
                    keypoints.push_back(keypoint);
                }

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob = score;
                obj.keypoints = keypoints;

                objects.push_back(obj);
            }
        }
    }
}

static void generate_proposals(const ncnn::Mat& pred, const ncnn::Mat& pred_points, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    int pred_row_offset = 0;
    for (size_t i = 0; i < strides.size(); i++)
    {
        const int stride = strides[i];

        const int num_grid_x = w / stride;
        const int num_grid_y = h / stride;
        const int num_grid = num_grid_x * num_grid_y;

        generate_proposals(pred.row_range(pred_row_offset, num_grid), pred_points.row_range(pred_row_offset, num_grid), stride, in_pad, prob_threshold, objects);

        pred_row_offset += num_grid;
    }
}

static int detect_yolo11_pose(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net yolo11;

    yolo11.opt.use_vulkan_compute = true;
    // yolo11.opt.use_bf16_storage = true;

    // https://github.com/nihui/ncnn-android-yolo11/tree/master/app/src/main/assets
    yolo11.load_param("yolo11n_pose.ncnn.param");
    yolo11.load_model("yolo11n_pose.ncnn.bin");
    // yolo11.load_param("yolo11s_pose.ncnn.param");
    // yolo11.load_model("yolo11s_pose.ncnn.bin");
    // yolo11.load_param("yolo11m_pose.ncnn.param");
    // yolo11.load_model("yolo11m_pose.ncnn.bin");

    const int target_size = 640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;
    const float mask_threshold = 0.5f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // ultralytics/cfg/models/v8/yolo11.yaml
    std::vector<int> strides(3);
    strides[0] = 8;
    strides[1] = 16;
    strides[2] = 32;
    const int max_stride = 32;

    // letterbox pad to multiple of max_stride
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // letterbox pad to target_size rectangle
    int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolo11.create_extractor();

    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    ncnn::Mat out_points;
    ex.extract("out1", out_points);

    std::vector<Object> proposals;
    generate_proposals(out, out_points, strides, in_pad, prob_threshold, proposals);

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    if (count == 0)
        return 0;

    const int num_points = out_points.w / 3;

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        for (int j = 0; j < num_points; j++)
        {
            objects[i].keypoints[j].p.x = (objects[i].keypoints[j].p.x - (wpad / 2)) / scale;
            objects[i].keypoints[j].p.y = (objects[i].keypoints[j].p.y - (hpad / 2)) / scale;
        }

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"person"};

    static const cv::Scalar colors[] = {
        cv::Scalar(244, 67, 54),
        cv::Scalar(233, 30, 99),
        cv::Scalar(156, 39, 176),
        cv::Scalar(103, 58, 183),
        cv::Scalar(63, 81, 181),
        cv::Scalar(33, 150, 243),
        cv::Scalar(3, 169, 244),
        cv::Scalar(0, 188, 212),
        cv::Scalar(0, 150, 136),
        cv::Scalar(76, 175, 80),
        cv::Scalar(139, 195, 74),
        cv::Scalar(205, 220, 57),
        cv::Scalar(255, 235, 59),
        cv::Scalar(255, 193, 7),
        cv::Scalar(255, 152, 0),
        cv::Scalar(255, 87, 34),
        cv::Scalar(121, 85, 72),
        cv::Scalar(158, 158, 158),
        cv::Scalar(96, 125, 139)
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const cv::Scalar& color = colors[i % 19];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        // draw bone
        static const int joint_pairs[16][2] = {
            {0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}
        };
        static const cv::Scalar bone_colors[] = {
            cv::Scalar(0, 255, 0),
            cv::Scalar(0, 255, 0),
            cv::Scalar(0, 255, 0),
            cv::Scalar(0, 255, 0),
            cv::Scalar(255, 128, 0),
            cv::Scalar(255, 128, 0),
            cv::Scalar(255, 128, 0),
            cv::Scalar(255, 128, 0),
            cv::Scalar(255, 128, 0),
            cv::Scalar(255, 51, 255),
            cv::Scalar(255, 51, 255),
            cv::Scalar(255, 51, 255),
            cv::Scalar(51, 153, 255),
            cv::Scalar(51, 153, 255),
            cv::Scalar(51, 153, 255),
            cv::Scalar(51, 153, 255),
        };

        for (int j = 0; j < 16; j++)
        {
            const KeyPoint& p1 = obj.keypoints[joint_pairs[j][0]];
            const KeyPoint& p2 = obj.keypoints[joint_pairs[j][1]];

            if (p1.prob < 0.2f || p2.prob < 0.2f)
                continue;

            cv::line(image, p1.p, p2.p, bone_colors[j], 2);
        }

        // draw joint
        for (size_t j = 0; j < obj.keypoints.size(); j++)
        {
            const KeyPoint& keypoint = obj.keypoints[j];

            fprintf(stderr, "%.2f %.2f = %.5f\n", keypoint.p.x, keypoint.p.y, keypoint.prob);

            if (keypoint.prob < 0.2f)
                continue;

            cv::circle(image, keypoint.p, 3, color, -1);
        }

        cv::rectangle(image, obj.rect, color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_yolo11_pose(m, objects);

    draw_objects(m, objects);

    return 0;
}
