// Copyright (c) 2022 Xiaomi Corp.        (author: Fangjun Kuang)
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "relshift.h"

namespace ncnn {

RelShift ::RelShift() {
  one_blob_only = true;
  support_inplace = false;
}

int RelShift::forward(const Mat &bottom_blob, Mat &top_blob,
                      const Option &opt) const {
  // Suppose w = 2*t - 1
  // out_row0: t-1, t, t+1, ..., 2*t-2
  // out_row1: t-2, t-1, t, ..., 2*t-3
  // out_row2: t-3, t-2, t-1, ..., 2*t-4
  // out_row3: t-4, t-3, t-2, ..., 2*t-5
  // ...
  // last_row: 0, 1, 2, ..., t-1
  //
  // We assume left context is 0
  int w = bottom_blob.w;
  int h = bottom_blob.h;  // w = 2 * h - 1
  int c = bottom_blob.c;

  top_blob.create(h, h, c, sizeof(float), opt.blob_allocator);

#pragma omp parallel for num_threads(opt.num_threads)
  for (int q = 0; q < c; ++q) {
    const float *in_ptr = bottom_blob.channel(q);
    float *out_ptr = top_blob.channel(q);

    for (int y = 0; y < h; ++y) {
      int start = h - 1 - y;

      memcpy(out_ptr, in_ptr + start, h * sizeof(float));

      in_ptr += w;
      out_ptr += h;
    }
  }

  return 0;
}

}  // namespace ncnn
