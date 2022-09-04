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

#include "../utils.h"
#include "pass_level1.h"

namespace pnnx {

class RelPositionalEncoding : public FuseModulePass {
 public:
  const char *match_type_str() const {
    return "__torch__.conformer.RelPositionalEncoding";
  }

  const char *type_str() const { return "icefall.RelPositionalEncoding"; }

  void write(Operator *op, const std::shared_ptr<torch::jit::Graph> &graph,
             const torch::jit::Module &mod) const {
    const auto &pe = mod.attr("pe").toTensor();
    op->attrs["pe"] = pe;
    // pe.sizes() is something like [1, 9999, 512]
    // std::cout << pe.sizes() << "\n";
  }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(RelPositionalEncoding)

}  // namespace pnnx
