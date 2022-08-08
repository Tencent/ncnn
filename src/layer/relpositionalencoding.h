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

#ifndef LAYER_RELPOSITIONALENCODING_H
#define LAYER_RELPOSITIONALENCODING_H

#include "layer.h"

namespace ncnn {

class RelPositionalEncoding : public Layer {
 public:
  RelPositionalEncoding();

  virtual int load_param(const ParamDict &pd);

  virtual int load_model(const ModelBin &mb);

  virtual int forward(const Mat &bottom_blob, Mat &top_blob,
                      const Option &opt) const;

 public:
  // param
  int w;
  int h;

  // model
  Mat pe;
};

}  // namespace ncnn

#endif  // LAYER_RELPOSITIONALENCODING_H
