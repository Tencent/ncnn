/**
 * @File   : darknet_shortcut.cpp
 * @Author : damone (damonexw@gmail.com)
 * @Link   :
 * @Date   : 10/31/2018, 3:21:56 PM
 */

#include "darknetshortcut.h"
#include <algorithm>

namespace ncnn
{

DEFINE_LAYER_CREATOR(DarknetShortcut)

DarknetShortcut::DarknetShortcut() {}

int DarknetShortcut::load_param(const ParamDict &pd)
{
  alpha = pd.get(0, 1.f);
  beta = pd.get(1, 1.f);
  return 0;
}

int DarknetShortcut::forward(const std::vector<Mat> &bottom_blobs,
                             std::vector<Mat> &top_blobs,
                             const Option &opt) const
{
  const Mat &bottom_blob = bottom_blobs[0];
  int w = bottom_blob.w;
  int h = bottom_blob.h;
  int c = bottom_blob.c;

  const Mat &bottom_blob1 = bottom_blobs[1];
  int o_w = bottom_blob1.w;
  int o_h = bottom_blob1.h;
  int o_c = bottom_blob1.c;

  int i_stride = w / o_w;
  int o_stride = o_w / w;
  if (i_stride != h / o_h) return -1;
  if (o_stride != o_h / h) return -1;
  i_stride = i_stride < 1 ? 1 : i_stride;
  o_stride = o_stride < 1 ? 1 : o_stride;

  int op_w = w < o_w ? w : o_w;
  int op_h = h < o_h ? h : o_h;
  int op_c = c < o_c ? c : o_c;

  Mat &top_blob = top_blobs[0];
  size_t elemsize = bottom_blob.elemsize;
  top_blob = bottom_blob1.clone();
  if (top_blob.empty())
    return -100;

  #pragma omp parallel for num_threads(opt.num_threads)
  for (int q = 0; q < op_c; q++)
  {

    int out_offset_h = 0;
    int in_offset_h = 0;

    for (int hi = 0; hi < op_h; hi++)
    {

      const float *ptr1 = bottom_blob.channel(q).row(in_offset_h);
      const float *ptr2 = bottom_blob1.channel(q).row(out_offset_h);
      float *outptr = top_blob.channel(q).row(out_offset_h);

      out_offset_h += o_stride;
      in_offset_h += i_stride;

      for (int i = 0; i < op_w; i++)
      {
        *outptr = alpha * (*ptr2) + beta * (*ptr1);

        ptr2 += o_stride;
        ptr1 += i_stride;
        outptr += o_stride;
      }
    }
  }

  return 0;
}

} // namespace ncnn
