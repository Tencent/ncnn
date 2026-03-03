### expression

expression is used in the reshape slice parameter to express the dynamic shape or subscript value based on the expression formula and input shape

Compared with directly converting the expression calculation process into multiple operators, the motivation for using expression
* No additional shape concat and other operators will be generated due to dynamic calculation, which greatly reduces the number of layers of the ncnn model and makes it easier to view the model structure and modify expression
* Shape or subscript evaluations are usually single-digit operations, which are more suitable for direct completion on the CPU without layout conversion and kernel call overhead

In the param file, `Reshape` layer can contain 6=expression

The pnnx tool can automatically convert `pnnx.Expression` to the expr parameter of ncnn `Reshape`

* Convert to 0w, 0h, 0d or 0c according to the input shape rank and `size(@0,1)`
* Automatically remove the batch dimension according to the input batch index
* Convert `pnnx.Expression` and `Tensor.reshape`/`Tensor.view` two operators are fused into ncnn `Reshape`
* Automatically summarize the number of references, exclude duplicate references and sort the indexes of references
* Convert the customary shape representation order, such as CHW to WHC

Example pnnx.param where A and B are 3D tensors
```
pnnx.Expression  expr     2 1 A B shape expr=[add(size(@1,0),2),mul(size(@0,1),2),-1]
Tensor.reshape   reshape  2 1 A shape out
```

pnnx.py
```python
shape = [(B.size(0) + 2), (A.size(1) * 2), -1]
out = A.reshape(*shape)
```

Converted to ncnn.param
```
Reshape          reshape  2 1 A B out 6="-1,*(0h,2),+(1c,2)"
```

### syntax

Use infix expression, format is `op(arg0,arg1,...)`, multiple operations can be nested, multiple sizes are separated by commas, and numbers can be integers or decimals

Among them, the commonly used `add` `sub` `mul` `div` `floor_div` are abbreviated as `+` `-` `*` `/` `//`, and other arithmetic operations use names, such as `sin` `ceil` `max`, etc.

* `max(2,3)`
* `floor(sin(3.14))`
* `+(*(-2,1),10)` means (-2 * 1) + 10
* `1,2,+(3,2)` list can represent output shape with 3-rank

The input shape can be referenced at runtime, format is `id(w|h|d|c)`, the maximum id is 9, which means that up to 10 inputs can be referenced

Assuming that the Reshape layer has two input blobs, A and B, then

* `0w,1h` means A.w, B.h
* `*(+(0c,1c),2)` means (A.c + B.c) * 2

### helper api

```cpp
#include "expression.h"

int count_expression_blobs(const std::string& expr);

int eval_list_expression(const std::string& expr, const std::vector<Mat>& blobs, std::vector<int>& outlist);
```

* `count_expression_blobs`

Pass expression to get the number of inputs it references, such as `0w,1h` returns 2

* `eval_list_expression`

Evaluate the result list according to expression and input blob calculate. If the calculation result is a floating point number, it will be automatically truncated to an integer.

### supported operator

|type|operators|
|---|---|
|float to int|`trunc` `ceil` `floor` `round`|
|binary arithmetic|`+` `-` `*` `/` `//` `max` `min` `pow` `fmod` `remainder` `atan2` `logaddexp`|
|unary arithmetic|`abs` `neg` `sign` `square` `sqrt` `rsqrt` `reciprocal` `exp` `log` `log10` `sin` `asin` `cos` `acos` `tan` `atan` `sinh` `asinh` `cosh` `acosh` `tanh` `atanh`|
|integer bitwise|`and` `or` `xor` `lshift` `rshift`|
