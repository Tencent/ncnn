/**
 * @File   : darknet_shortcut.h
 * @Author : damone (damonexw@gmail.com)
 * @Link   :
 * @Date   : 10/31/2018, 3:21:34 PM
 */

#ifndef _DARKNET_SHORTCUT_H
#define _DARKNET_SHORTCUT_H

#include "layer.h"

namespace ncnn
{

::ncnn::Layer *DarknetShortcut_layer_creator();
class DarknetShortcut : public Layer
{
public:
  DarknetShortcut();
  virtual int load_param(const ParamDict &pd);
  virtual int forward(const std::vector<Mat> &bottom_blobs, std::vector<Mat> &top_blobs, const Option &opt) const;

public:
  float alpha;
  float beta;
};

} // namespace ncnn

#endif