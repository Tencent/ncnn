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

#include <algorithm>

#include "fuse_rnn_unpack.h"
#include "pass_level2.h"

namespace pnnx {

void fuse_relpositionalencoding_unpack(Graph &graph) {
  while (1) {
    bool matched = false;

    for (size_t i = 0; i < graph.ops.size(); i++) {
      Operator *op = graph.ops[i];

      if (op->type != "icefall.RelPositionalEncoding") {
        continue;
      }

      if (op->outputs.size() != 1) {
        continue;
      }

      if (op->outputs[0]->consumers.size() != 1) {
        continue;
      }

      Operator *op2 = op->outputs[0]->consumers[0];
      if (op2->type != "prim::TupleUnpack") {
        continue;
      }

      matched = true;

      op->outputs[0]->producer = 0;
      op->outputs[0]->remove_consumer(op2);

      for (auto &x : op2->outputs) {
        x->producer = op;
      }

      op->outputs = op2->outputs;

      op2->inputs.clear();
      op2->outputs.clear();

      graph.ops.erase(std::find(graph.ops.begin(), graph.ops.end(), op2));

      delete op2;

      // RelPositionalEncoding has two outputs. However, the
      // first output is identical with the input. The following
      // code block transforms the number of outputs from 2 to 1.
      Operand *in = op->inputs[0];
      Operand *out = op->outputs[0];
      for (auto &op : graph.ops) {
        for (auto &oi : op->inputs) {
          if (oi == out) {
            oi = in;
            out->remove_consumer(op);
            in->consumers.push_back(op);
          }
        }
      }
      op->outputs.erase(op->outputs.begin());

      break;
    }

    if (!matched) break;
  }
}

}  // namespace pnnx
