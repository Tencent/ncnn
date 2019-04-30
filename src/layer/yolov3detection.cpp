/**
 * @File   : yolov3_detection.cpp
 * @Author : damone (damonexw@gmail.com)
 * @Link   : 
 * @Date   : 11/3/2018, 3:24:10 PM
 */

#include <math.h>
#include "layer_type.h"
#include "objectdetection.h"
#include "yolov3detection.h"

namespace ncnn
{

DEFINE_LAYER_CREATOR(Yolov3Detection)

Yolov3Detection::Yolov3Detection()
{
    softmax = ncnn::create_layer(ncnn::LayerType::Softmax);
    ncnn::ParamDict pd;
    pd.set(0, 0);
    softmax->load_param(pd);

    sigmoid = ncnn::create_layer(ncnn::LayerType::Sigmoid);
}

Yolov3Detection::~Yolov3Detection()
{
    if (NULL != softmax)
    {
        delete softmax;
        softmax = NULL;
    }

    if (NULL != sigmoid)
    {
        delete sigmoid;
        sigmoid = NULL;
    }
}

int Yolov3Detection::load_param(const ParamDict &pd)
{
    classes = pd.get(0, 20);
    box_num = pd.get(1, 5);
    softmax_enable = pd.get(2, 1);
    confidence_threshold = pd.get(3, 0.25f);
    nms_threshold = pd.get(4, 0.45f);
    net_width = pd.get(5, 0);
    net_height = pd.get(6, 0);
    biases = pd.get(7, Mat());
    return 0;
}

int Yolov3Detection::forward(const std::vector<Mat> &bottom_blobs, std::vector<Mat> &top_blobs, const Option &opt) const
{
    int channels = bottom_blobs[0].c;
    std::vector<int> mat_steps_channels;
    mat_steps_channels.push_back(channels);
    int prob_steps = bottom_blobs[0].w * bottom_blobs[0].h;
    for (int b = 1; b < bottom_blobs.size(); b++)
    {
        channels += bottom_blobs[b].c;
        mat_steps_channels.push_back(channels);
        prob_steps += bottom_blobs[b].w * bottom_blobs[b].h;
        prob_steps *= 0.5f;
    }

    const int channels_per_box = channels / box_num;

    // anchor coord + box score + num_class
    if (channels_per_box != 4 + 1 + classes)
        return -1;

    ObjectsManager objects(prob_steps * box_num / 20);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < box_num; pp++)
    {
        int p = pp * channels_per_box;

        int current_mat_index = 0;
        int front_mat_channels = 0;
        while (p >= mat_steps_channels[current_mat_index])
        {
            front_mat_channels = mat_steps_channels[current_mat_index];
            current_mat_index++;
        }

        const Mat &bottom_blob = bottom_blobs[current_mat_index];
        p -= front_mat_channels;

        Mat locations = bottom_blob.channel_range(p, 2);
        sigmoid->forward_inplace(locations, opt);
        Mat boxes_score = bottom_blob.channel_range(p + 4, 1);
        sigmoid->forward_inplace(boxes_score);

        const float *xptr = locations.channel(0);
        const float *yptr = locations.channel(1);
        const float *wptr = bottom_blob.channel(p + 2);
        const float *hptr = bottom_blob.channel(p + 3);

        const float *box_score_ptr = boxes_score.channel(0);

        Mat scores = bottom_blob.channel_range(p + 5, classes);
        if (softmax_enable)
        {
            // softmax class scores
            softmax->forward_inplace(scores, opt);
        }
        else
        {
            // Not use softmax to avoid class competition
            sigmoid->forward_inplace(scores, opt);
        }

        int w = bottom_blob.w;
        int h = bottom_blob.h;

        int real_w = w, real_h = h;
        if (net_height != 0 && net_width != 0)
        {
            real_w = net_width;
            real_h = net_height;
        }

        const float bias_w = biases[pp * 2] / real_w;
        const float bias_h = biases[pp * 2 + 1] / real_h;

        for (int y_offset = 0; y_offset < h; y_offset++)
        {
            for (int x_offset = 0; x_offset < w; x_offset++)
            {

                float box_w = exp(wptr[0]) * bias_w;
                float box_h = exp(hptr[0]) * bias_h;

                float xmin = (xptr[0] + x_offset) / w - box_w * 0.5f;
                float ymin = (yptr[0] + y_offset) / h - box_h * 0.5f;
                float xmax = xmin + box_w;
                float ymax = ymin + box_h;

                // box score
                float box_score = box_score_ptr[0];

                // find class index with max class score
                int class_index = 0;
                float class_score = 0.f;
                for (int q = 0; q < classes; q ++)
                {
                    float score = scores.channel(q).row(y_offset)[x_offset];

                    if (score > class_score)
                    {
                        class_index = q;
                        class_score = score;
                    }
                }

                float prob = box_score * class_score;
                if (prob >= confidence_threshold)
                {
                    ObjectBox object = {
                        .xmin = xmin,
                        .ymin = ymin,
                        .xmax = xmax,
                        .ymax = ymax,
                        .prob = prob,
                        .classid = class_index,
                    };
                    objects.add_new_object_box(object);
                }

                xptr++;
                yptr++;
                wptr++;
                hptr++;

                box_score_ptr++;
            }
        }
    }

    std::vector<ObjectBox> detected_objects;
    objects.do_objects_nms(detected_objects, nms_threshold, confidence_threshold);

    Mat &top_blob = top_blobs[0];
    top_blob.create((int)6, (int)detected_objects.size(), (size_t)4u, (Allocator*)opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    for (int i = 0; i < detected_objects.size(); i++)
    {
        const ObjectBox &r = detected_objects[i];
        float *outptr = top_blob.row(i);

        outptr[0] = r.classid + 1; // 0 reserve for background
        outptr[1] = r.prob;
        outptr[2] = r.xmin;
        outptr[3] = r.ymin;
        outptr[4] = r.xmax;
        outptr[5] = r.ymax;
    }

    return 0;
}

} // namespace ncnn
