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

#include "icefall_insert_expanddims_conv1d.h"

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

void icefall_insert_expanddims_conv1d(Graph &graph) {
  while (1) {
    bool matched = false;

    for (size_t i = 0; i < graph.ops.size(); i++) {
      Operator *op = graph.ops[i];

      if (op->type != "nn.Conv1d") {
        continue;
      }

      Operand *conv1d_out = op->outputs[0];

      if (conv1d_out->consumers.size() != 1) {
        continue;
      }

      Operator *op2 = conv1d_out->consumers[0];
      if (op2->type != "F.glu") {
        continue;
      }

      matched = true;

      fprintf(stderr, "icefall_insert_expandims_conv1d\n");
      Operator *new_op = graph.new_operator_after(
          "ExpandDims", op->name + "_ncnnexpanddims0", op);

      op->outputs[0]->remove_consumer(op2);
      op->outputs[0]->consumers.push_back(new_op);
      new_op->inputs.push_back(op->outputs[0]);

      Operand *new_operand = graph.new_operand(new_op->name + "_output");
      new_op->outputs.push_back(new_operand);
      new_operand->producer = new_op;
      new_operand->consumers.push_back(op2);
      op2->inputs[0] = new_operand;

      std::vector<int> axes = {0};
      new_op->params["3"] = axes;

      break;
    }  // for (size_t i = 0, i < graph.ops.size(); i++)

    if (!matched) break;
  }  // while(1)
}

}  // namespace ncnn

}  // namespace pnnx
