#include <iostream>

#include "mat.h"
#include "net.h"

template <typename T = float>
void print_mat_1d(ncnn::Mat &m, int start_w, int end_w) {
  const T *p = m;
  if (end_w == -1) {
    end_w = m.w;
  }
  for (int w = start_w; w != end_w; ++w) {
    std::cout << p[w] << ", ";
  }
  std::cout << "\n";
}

template <typename T = float>
void print_mat_2d(ncnn::Mat &m, int start_h, int end_h, int start_w,
                  int end_w) {
  if (end_h == -1) {
    end_h = m.h;
  }
  for (int h = start_h; h != end_h; ++h) {
    ncnn::Mat sub = m.row_range(h, 1);
    print_mat_1d<T>(sub, start_w, end_w);
  }
}

template <typename T = float>
void print_mat_3d(ncnn::Mat &m, int start_c, int end_c, int start_h, int end_h,
                  int start_w, int end_w) {
  if (end_c == -1) {
    end_c = m.c;
  }

  for (int c = start_c; c != end_c; ++c) {
    std::cout << "c " << c << "\n";
    ncnn::Mat sub = m.channel_range(c, 1);
    print_mat_2d<T>(sub, start_h, end_h, start_w, end_w);
  }
}

int main() {
  int c = 1;
  int h = 6;
  int w = 8;
  int size = c * h * w;
  // std::vector<float> data(size);
  std::vector<int> data = {1, 3, 5, 4, 2};
  // for (int i = 0; i != size; ++i) {
  //   data[i] = i;
  // }
  ncnn::Option opt;
  opt.num_threads = 1;
  ncnn::Net net;
  net.opt = opt;
  net.load_param("foo/make_pad_mask.ncnn.param");
  net.load_model("foo/make_pad_mask.ncnn.bin");

  ncnn::Extractor ex = net.create_extractor();

  ncnn::Mat m(data.size(), data.data());
  m = m.clone();
  std::cout << "in\n";
  print_mat_1d<int>(m, 0, -1);
  std::cout << "\n";

  ncnn::Mat out;

  ex.input("in0", m);
  ex.extract("out0", out);
  print_mat_2d<int>(out, 0, -1, 0, -1);
  ex.clear();
  net.clear();
}
