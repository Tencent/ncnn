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

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

// ncnn outputs a 2-D tensor after conv1d.
// This function does unsqueeze(0) on the output of conv1d
void icefall_insert_expanddims_conv1d(Graph &graph);

}  // namespace ncnn

}  // namespace pnnx
