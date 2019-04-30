/**
 * @File   : darknet_activation.h
 * @Author : damone (damonexw@gmail.com)
 * @Link   :
 * @Date   : 10/30/2018, 3:38:19 PM
 */

#ifndef _DARKNET_ACTIVATION_H
#define _DARKNET_ACTIVATION_H

#include "layer.h"

namespace ncnn
{

::ncnn::Layer *DarknetActivation_layer_creator();
class DarknetActivation : public Layer
{
public:
  DarknetActivation();
  virtual int load_param(const ParamDict &pd);
  virtual int forward_inplace(Mat &bottom_top_blob, const Option &opt) const;

public:
  enum Activation_Type
  {
    Activate_LOGISTIC,
    Activate_RELU,
    Activate_RELIE,
    Activate_LINEAR,
    Activate_RAMP,
    Activate_TANH,
    Activate_PLSE,
    Activate_LEAKY,
    Activate_ELU,
    Activate_LOGGY,
    Activate_STAIR,
    Activate_HARDTAN,
    Activate_LHTAN,
    Activate_SELU
  };

public:
  int activate_type;
};

} // namespace ncnn

#endif