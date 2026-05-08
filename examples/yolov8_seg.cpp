// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// 1. install
//      pip3 install -U ultralytics pnnx ncnn
// 2. export yolov8-seg torchscript
//      yolo export model=yolov8n-seg.pt format=torchscript
// 3. convert torchscript with static shape
//      pnnx yolov8n-seg.torchscript
// 4. modify yolov8n_seg_pnnx.py for dynamic shape inference
//      A. modify reshape to support dynamic image sizes
//      B. permute tensor before concat and adjust concat axis
//      C. drop post-process part
//      before:
//          v_144 = v_143.view(1, 32, 6400)
//          v_150 = v_149.view(1, 32, 1600)
//          v_156 = v_155.view(1, 32, 400)
//          v_157 = torch.cat((v_144, v_150, v_156), dim=2)
//          ...
//          v_191 = v_168.view(1, 144, 6400)
//          v_192 = v_179.view(1, 144, 1600)
//          v_193 = v_190.view(1, 144, 400)
//          v_194 = torch.cat((v_191, v_192, v_193), dim=2)
//          ...
//          v_215 = (v_214, v_138, )
//          return v_215
//      after:
//          v_144 = v_143.view(1, 32, -1).transpose(1, 2)
//          v_150 = v_149.view(1, 32, -1).transpose(1, 2)
//          v_156 = v_155.view(1, 32, -1).transpose(1, 2)
//          v_157 = torch.cat((v_144, v_150, v_156), dim=1)
//          ...
//          v_191 = v_168.view(1, 144, -1).transpose(1, 2)
//          v_192 = v_179.view(1, 144, -1).transpose(1, 2)
//          v_193 = v_190.view(1, 144, -1).transpose(1, 2)
//          v_194 = torch.cat((v_191, v_192, v_193), dim=1)
//          return v_194, v_157, v_138
// 5. re-export yolov8-seg torchscript
//      python3 -c 'import yolov8n_seg_pnnx; yolov8n_seg_pnnx.export_torchscript()'
// 6. convert new torchscript with dynamic shape
//      pnnx yolov8n_seg_pnnx.py.pt inputshape=[1,3,640,640] inputshape2=[1,3,320,320]
// 7. now you get ncnn model files
//      mv yolov8n_seg_pnnx.py.ncnn.param yolov8n_seg.ncnn.param
//      mv yolov8n_seg_pnnx.py.ncnn.bin yolov8n_seg.ncnn.bin

// the out blob would be a 2-dim tensor with w=176 h=8400
//
//        | bbox-reg 16 x 4       | per-class scores(80) |
//        +-----+-----+-----+-----+----------------------+
//        | dx0 | dy0 | dx1 | dy1 |0.1 0.0 0.0 0.5 ......|
//   all /|     |     |     |     |           .          |
//  boxes |  .. |  .. |  .. |  .. |0.0 0.9 0.0 0.0 ......|
//  (8400)|     |     |     |     |           .          |
//       \|     |     |     |     |           .          |
//        +-----+-----+-----+-----+----------------------+
//

//
//        | mask (32) |
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

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    int gindex;
    cv::Mat mask;
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

static void generate_proposals(const ncnn::Mat& pred, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;

    const int reg_max_1 = 16;
    const int num_class = pred.w - reg_max_1 * 4; // number of classes. 80 for COCO

    for (int y = 0; y < num_grid_y; y++)
    {
        for (int x = 0; x < num_grid_x; x++)
        {
            const ncnn::Mat pred_grid = pred.row_range(y * num_grid_x + x, 1);

            // find label with max score
            int label = -1;
            float score = -FLT_MAX;
            {
                const ncnn::Mat pred_score = pred_grid.range(reg_max_1 * 4, num_class);

                for (int k = 0; k < num_class; k++)
                {
                    float s = pred_score[k];
                    if (s > score)
                    {
                        label = k;
                        score = s;
                    }
                }

                score = sigmoid(score);
            }

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

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob = score;
                obj.gindex = y * num_grid_x + x;

                objects.push_back(obj);
            }
        }
    }
}

static void generate_proposals(const ncnn::Mat& pred, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
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

        std::vector<Object> objects_stride;
        generate_proposals(pred.row_range(pred_row_offset, num_grid), stride, in_pad, prob_threshold, objects_stride);

        for (size_t j = 0; j < objects_stride.size(); j++)
        {
            Object obj = objects_stride[j];
            obj.gindex += pred_row_offset;
            objects.push_back(obj);
        }

        pred_row_offset += num_grid;
    }
}

static int detect_yolov8_seg(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net yolov8;

    yolov8.opt.use_vulkan_compute = true;
    // yolov8.opt.use_bf16_storage = true;

    // https://github.com/nihui/ncnn-android-yolov8/tree/master/app/src/main/assets
    yolov8.load_param("yolov8n_seg.ncnn.param");
    yolov8.load_model("yolov8n_seg.ncnn.bin");
    // yolov8.load_param("yolov8s_seg.ncnn.param");
    // yolov8.load_model("yolov8s_seg.ncnn.bin");
    // yolov8.load_param("yolov8m_seg.ncnn.param");
    // yolov8.load_model("yolov8m_seg.ncnn.bin");

    const int target_size = 640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;
    const float mask_threshold = 0.5f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // ultralytics/cfg/models/v8/yolov8.yaml
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

    ncnn::Extractor ex = yolov8.create_extractor();

    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    std::vector<Object> proposals;
    generate_proposals(out, strides, in_pad, prob_threshold, proposals);

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    if (count == 0)
        return 0;

    ncnn::Mat mask_feat;
    ex.extract("out1", mask_feat);

    ncnn::Mat mask_protos;
    ex.extract("out2", mask_protos);

    ncnn::Mat objects_mask_feat(mask_feat.w, 1, count);

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;

        // pick mask feat
        memcpy(objects_mask_feat.channel(i), mask_feat.row(objects[i].gindex), mask_feat.w * sizeof(float));
    }

    // process mask
    ncnn::Mat objects_mask;
    {
        ncnn::Layer* gemm = ncnn::create_layer("Gemm");

        ncnn::ParamDict pd;
        pd.set(6, 1);                             // constantC
        pd.set(7, count);                         // constantM
        pd.set(8, mask_protos.w * mask_protos.h); // constantN
        pd.set(9, mask_feat.w);                   // constantK
        pd.set(10, -1);                           // constant_broadcast_type_C
        pd.set(11, 1);                            // output_N1M
        gemm->load_param(pd);

        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;

        gemm->create_pipeline(opt);

        std::vector<ncnn::Mat> gemm_inputs(2);
        gemm_inputs[0] = objects_mask_feat;
        gemm_inputs[1] = mask_protos.reshape(mask_protos.w * mask_protos.h, 1, mask_protos.c);
        std::vector<ncnn::Mat> gemm_outputs(1);
        gemm->forward(gemm_inputs, gemm_outputs, opt);
        objects_mask = gemm_outputs[0].reshape(mask_protos.w, mask_protos.h, count);

        gemm->destroy_pipeline(opt);

        delete gemm;
    }
    {
        ncnn::Layer* sigmoid = ncnn::create_layer("Sigmoid");

        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;

        sigmoid->create_pipeline(opt);

        sigmoid->forward_inplace(objects_mask, opt);

        sigmoid->destroy_pipeline(opt);

        delete sigmoid;
    }

    // resize mask map
    {
        ncnn::Mat objects_mask_resized;
        ncnn::resize_bilinear(objects_mask, objects_mask_resized, in_pad.w / scale, in_pad.h / scale);
        objects_mask = objects_mask_resized;
    }

    // create per-object mask
    for (int i = 0; i < count; i++)
    {
        Object& obj = objects[i];

        const ncnn::Mat mm = objects_mask.channel(i);

        obj.mask = cv::Mat((int)obj.rect.height, (int)obj.rect.width, CV_8UC1);

        // adjust offset to original unpadded and clip inside object box
        for (int y = 0; y < (int)obj.rect.height; y++)
        {
            const float* pmm = mm.row((int)(hpad / 2 / scale + obj.rect.y + y)) + (int)(wpad / 2 / scale + obj.rect.x);
            uchar* pmask = obj.mask.ptr<uchar>(y);
            for (int x = 0; x < (int)obj.rect.width; x++)
            {
                pmask[x] = pmm[x] > mask_threshold ? 1 : 0;
            }
        }
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    static cv::Scalar colors[] = {
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

        for (int y = 0; y < (int)obj.rect.height; y++)
        {
            const uchar* maskptr = obj.mask.ptr<const uchar>(y);
            uchar* bgrptr = image.ptr<uchar>((int)obj.rect.y + y) + (int)obj.rect.x * 3;
            for (int x = 0; x < (int)obj.rect.width; x++)
            {
                if (maskptr[x])
                {
                    bgrptr[0] = bgrptr[0] * 0.5 + color[0] * 0.5;
                    bgrptr[1] = bgrptr[1] * 0.5 + color[1] * 0.5;
                    bgrptr[2] = bgrptr[2] * 0.5 + color[2] * 0.5;
                }
                bgrptr += 3;
            }
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
    detect_yolov8_seg(m, objects);

    draw_objects(m, objects);

    return 0;
}
