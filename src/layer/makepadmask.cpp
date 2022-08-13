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

#include "makepadmask.h"

namespace ncnn {

MakePadMask::MakePadMask() {
  one_blob_only = true;
  support_inplace = false;
}

int MakePadMask::forward(const Mat &bottom_blob, Mat &top_blob,
                         const Option &opt) const {
  // NOTE: We assume the input data type is int32_t and the output data
  // type is also int32_t. We may change them later.
  int batch_size = bottom_blob.w;  // batch size
  int max_len =
      *std::max_element(static_cast<const int *>(bottom_blob),
                        static_cast<const int *>(bottom_blob) + batch_size);

  top_blob.create(max_len, batch_size, sizeof(int), opt.blob_allocator);

  const int *ptr = bottom_blob;

#pragma omp parallel for num_threads(opt.num_threads)
  for (int b = 0; b < batch_size; ++b) {
    int *out_ptr = static_cast<int *>(top_blob) + b * max_len;
    int size = ptr[b];
    for (int c = 0; c != max_len; ++c) {
      if (c < size) {
        out_ptr[c] = 0;
      } else {
        out_ptr[c] = 1;
      }
    }
  }

  return 0;
}

}  // namespace ncnn
