/**
 * @File   : darknet_activation.cpp
 * @Author : damone (damonexw@gmail.com)
 * @Link   :
 * @Date   : 10/30/2018, 3:40:37 PM
 */

#include "darknetactivation.h"
#include <cmath>

namespace ncnn
{

DEFINE_LAYER_CREATOR(DarknetActivation);

DarknetActivation::DarknetActivation()
{
  one_blob_only = true;
  support_inplace = true;
}

int DarknetActivation::load_param(const ParamDict &pd)
{
  activate_type = pd.get(0, 0);
  return 0;
}

template <typename Op>
static int activate_op_inplace(Mat &a, const Option &opt)
{
  Op op;
  int channels = a.c;
  int size = a.w * a.h;

  #pragma omp parallel for num_threads(opt.num_threads)
  for (int c = 0; c < channels; c++)
  {
    Mat cmat = a.channel(c);
    for (int i = 0; i < size; i++)
      cmat[i] = op(cmat[i]);
  }
  return 0;
}

template <typename T>
struct activate_op_relie : std::unary_function<T, T>
{
  T operator()(const T &x) const { return (x > 0) ? x : .01 * x; }
};

template <typename T>
struct activate_op_ramp : std::unary_function<T, T>
{
  T operator()(const T &x) const { return x * (x > 0) + .1 * x; }
};

template <typename T>
struct activate_op_linear : std::unary_function<T, T>
{
  T operator()(const T &x) const { return x; }
};

template <typename T>
struct activate_op_loggy : std::unary_function<T, T>
{
  T operator()(const T &x) const { return 2. / (1. + exp(-x)) - 1; }
};

template <typename T>
struct activate_op_plse : std::unary_function<T, T>
{
  T operator()(const T &x) const
  {
    if (x < -4)
      return .01 * (x + 4);
    if (x > 4)
      return .01 * (x - 4) + 1;
    return .125 * x + .5;
  }
};

template <typename T>
struct activate_op_stair : std::unary_function<T, T>
{
  T operator()(const T &x) const
  {
    int n = floor(x);
    if (n % 2 == 0)
      return floor(x / 2.);
    else
      return (x - n) + floor(x / 2.);
  }
};

template <typename T>
struct activate_op_hardtan : std::unary_function<T, T>
{
  T operator()(const T &x) const
  {
    if (x < -1)
      return -1;
    if (x > 1)
      return 1;
    return x;
  }
};

template <typename T>
struct activate_op_lhtan : std::unary_function<T, T>
{
  T operator()(const T &x) const
  {
    if (x < 0)
      return .001 * x;
    if (x > 1)
      return .001 * (x - 1) + 1;
    return x;
  }
};

int DarknetActivation::forward_inplace(Mat &bottom_top_blob,
                                       const Option &opt) const
{
  if (activate_type == Activate_LINEAR)
    return 0;
  if (activate_type == Activate_RELIE)
    return activate_op_inplace< activate_op_relie<float> >(bottom_top_blob, opt);
  if (activate_type == Activate_RAMP)
    return activate_op_inplace< activate_op_ramp<float> >(bottom_top_blob, opt);
  if (activate_type == Activate_LOGGY)
    return activate_op_inplace <activate_op_loggy<float> >(bottom_top_blob, opt);
  if (activate_type == Activate_PLSE)
    return activate_op_inplace< activate_op_plse<float> >(bottom_top_blob, opt);
  if (activate_type == Activate_STAIR)
    return activate_op_inplace< activate_op_stair<float> >(bottom_top_blob, opt);
  if (activate_type == Activate_HARDTAN)
    return activate_op_inplace< activate_op_hardtan<float> >(bottom_top_blob, opt);
  if (activate_type == Activate_LHTAN)
    return activate_op_inplace< activate_op_lhtan<float> >(bottom_top_blob, opt);

  return 0;
}
} // namespace ncnn
