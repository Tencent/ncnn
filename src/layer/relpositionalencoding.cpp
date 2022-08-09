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

#include "relpositionalencoding.h"

#include "crop.h"

namespace ncnn {

RelPositionalEncoding::RelPositionalEncoding() {
  one_blob_only = true;
  support_inplace = false;
}

int RelPositionalEncoding::load_param(const ParamDict &pd) {
  w = pd.get(0, 0);
  h = pd.get(1, 0);

  return 0;
}

int RelPositionalEncoding::load_model(const ModelBin &mb) {
  pe = mb.load(w, h, 1);
  if (pe.empty()) return -100;
  return 0;
}

int RelPositionalEncoding::forward(const Mat &bottom_blob, Mat &top_blob,
                                   const Option &opt) const {
  int in_h = bottom_blob.h;
  int start_h = (h >> 1) - in_h + 1;  // assume left context is 0

  ParamDict pd;
  pd.set(1, start_h);
  pd.set(3, bottom_blob.w);
  pd.set(4, bottom_blob.h * 2 - 1);

  Crop crop;
  crop.load_param(pd);

  return crop.forward(pe, top_blob, opt);
}

}  // namespace ncnn
