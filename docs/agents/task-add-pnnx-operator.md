# Task: Add a New Operator to PNNX

To support a new PyTorch op in PNNX conversion:

1. **Add a pass_level2 converter** — create `tools/pnnx/src/pass_level2/<op_name>.cpp`

   For `torch.nn` modules (e.g., `nn.NewOp`):
   ```cpp
   #include "pass_level2.h"
   namespace pnnx {
   class nn_NewOp : public GraphRewriterPass
   {
   public:
       const char* match_pattern_graph() const
       {
           return R"PNNXIR(7767517
   3 2
   pnnx.Input              input       0 1 input
   aten::new_op            op_0        1 1 input out attr0=%param0 attr1=%param1
   pnnx.Output             output      1 0 out
   )PNNXIR";
       }
       const char* type_str() const { return "nn.NewOp"; }
       const char* name_str() const { return "newop"; }
       void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
       {
           op->params["param0"] = captured_params.at("param0");
           op->params["param1"] = captured_params.at("param1");
       }
   };
   REGISTER_GLOBAL_PNNX_GRAPH_REWRITER_PASS(nn_NewOp, 20)
   } // namespace pnnx
   ```

   For `torch.nn.functional` functions (e.g., `F.new_op`):
   - Create `tools/pnnx/src/pass_level2/F_new_op.cpp` with a similar pattern.

2. **Add ncnn lowering** — create `tools/pnnx/src/pass_ncnn/nn_NewOp.cpp` to map PNNX params to ncnn param IDs.

3. **Add a PNNX test** — create `tools/pnnx/tests/test_nn_NewOp.py`:
   ```python
   import torch
   import torch.nn as nn
   class Model(nn.Module):
       def __init__(self):
           super().__init__()
           self.op = nn.NewOp(...)
       def forward(self, x):
           return self.op(x)
   # ... export and test
   ```

4. **Register the test** — add to `tools/pnnx/tests/CMakeLists.txt`.
