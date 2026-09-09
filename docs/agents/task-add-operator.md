# Task: Add a New Operator to ncnn

**Files to create/modify:**

1. **Declare the layer** — `src/layer/<newop>.h`
   ```cpp
   #ifndef LAYER_NEWOP_H
   #define LAYER_NEWOP_H
   #include "layer.h"
   namespace ncnn {
   class NewOp : public Layer
   {
   public:
       NewOp();
       virtual int load_param(const ParamDict& pd);
       virtual int load_model(const ModelBin& mb);  // only if layer has weights
       virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
       // or forward_inplace() if support_inplace = true
   public:
       int param0;
       float param1;
       Mat weight_data;  // only if layer has weights
   };
   } // namespace ncnn
   #endif
   ```

2. **Implement the layer** — `src/layer/<newop>.cpp`
   ```cpp
   #include "newop.h"
   namespace ncnn {
   NewOp::NewOp()
   {
       one_blob_only = true;
       support_inplace = false;
   }
   int NewOp::load_param(const ParamDict& pd)
   {
       param0 = pd.get(0, 0);
       param1 = pd.get(1, 1.f);
       return 0;
   }
   int NewOp::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
   {
       // implementation
       return 0;
   }
   } // namespace ncnn
   ```

3. **Register the layer** — add to `src/CMakeLists.txt`:
   ```cmake
   ncnn_add_layer(NewOp)
   ```
   This must be inserted at the **end** of the layer list. Never insert in the middle — layer type indices are assigned sequentially and must remain stable for backward compatibility.

4. **Add a test** — `tests/test_newop.cpp`
   ```cpp
   #include "testutil.h"
   static int test_newop(const ncnn::Mat& a)
   {
       ncnn::ParamDict pd;
       pd.set(0, 1);
       pd.set(1, 2.f);
       std::vector<ncnn::Mat> weights(0);
       return test_layer("NewOp", pd, weights, a);
   }
   int main()
   {
       SRAND(7767517);
       return test_newop(RandomMat(5, 6, 7, 24))
           || test_newop(RandomMat(7, 9, 12))
           || test_newop(RandomMat(15, 24))
           || test_newop(RandomMat(128));
   }
   ```

5. **Register the test** — add to `tests/CMakeLists.txt`:
   ```cmake
   ncnn_add_test(test_newop)
   ```

6. **Document the operator** — update `docs/developer-guide/operation-param-weight-table.md` with param IDs and weight layout.
