/**
 * @File   : yolov3_detection.h
 * @Author : damone (damonexw@gmail.com)
 * @Link   : 
 * @Date   : 11/3/2018, 3:23:53 PM
 */

#ifndef _YOLO_V3_DETECTION_H
#define _YOLO_V3_DETECTION_H

#include "layer.h"

namespace ncnn
{

::ncnn::Layer *Yolov3Detection_layer_creator();
class Yolov3Detection : public ncnn::Layer
{
public:
  Yolov3Detection();
  ~Yolov3Detection();

  virtual int load_param(const ParamDict &pd) override;
  virtual int forward(const std::vector<Mat> &bottom_blobs, std::vector<Mat> &top_blobs, const Option &opt) const override;

public:
  int classes;
  int box_num;
  int net_width;
  int net_height;
  int softmax_enable;

  float confidence_threshold;
  float nms_threshold;

  Mat biases;

  ncnn::Layer *softmax;
  ncnn::Layer *sigmoid;
};

} // namespace ncnn


#endif
