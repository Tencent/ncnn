## batchnorm
```
Input       A            0 1 A 0 0 0
MemoryData  sub/y        0 1 sub/y 16 0 0
BinaryOp    sub          2 1 A sub/y sub 1
MemoryData  div/y        0 1 div/y 16 0 0
BinaryOp    div          2 1 sub div/y div 3
MemoryData  mul/y        0 1 mul/y 16 0 0
BinaryOp    mul          2 1 div mul/y mul 2
MemoryData  BiasAdd/bias 0 1 BiasAdd/bias 16 0 0
BinaryOp    BiasAdd      2 1 mul BiasAdd/bias BiasAdd 0
```
## convolution
```
Input       A            0 1 A 0 0 0
Convolution Conv2D       1 1 A Conv2D 10 3 1 1 0 0 270
MemoryData  biases/read  0 1 biases/read 10 0 0
BinaryOp    BiasAdd      2 1 Conv2D biases/read BiasAdd 0
```
## innerproduct
```
Input        A           0 1 A 0 0 0
MemoryData   biases/read 0 1 biases/read 10 0 0
InnerProduct MatMul      1 1 A MatMul 10 0 2560
BinaryOp     conv6       2 1 MatMul biases/read conv6 0
```
## leakyrelu
```
Input       A            0 1 A 0 0 0
Split       splitncnn_0  1 2 A A_splitncnn_0 A_splitncnn_1
MemoryData  mul_1/x      0 1 mul_1/x 0 0 0
BinaryOp    mul_1        2 1 mul_1/x A_splitncnn_1 mul_1 2
BinaryOp    leaky        2 1 mul_1 A_splitncnn_0 leaky 4
```
## prelu
```
Input       A            0 1 A 0 0 0
Split       splitncnn_0  1 2 A A_splitncnn_0 A_splitncnn_1
MemoryData  prelu/alpha  0 1 prelu/alpha 10 0 0
ReLU        prelu/Relu   1 1 A_splitncnn_1 prelu/Relu 0.000000
UnaryOp     prelu/Neg    1 1 A_splitncnn_0 prelu/Neg 1
ReLU        prelu/Relu_1 1 1 prelu/Neg prelu/Relu_1 0.000000
UnaryOp     prelu/Neg_1  1 1 prelu/Relu_1 prelu/Neg_1 1
BinaryOp    prelu/Mul    2 1 prelu/alpha prelu/Neg_1 prelu/Mul 2
BinaryOp    prelu/add    2 1 prelu/Relu prelu/Mul prelu/add 0
```
## softmax
```
Input       A            0 1 A 0 0 0
Split       splitncnn_4  1 2 A A_splitncnn_0 A_splitncnn_1
Reduction   Max          1 1 A_splitncnn_1 Max 4 -2 1.000000
BinaryOp    sub          2 1 A_splitncnn_0 Max sub 1
UnaryOp     Exp          1 1 sub Exp 7
Split       splitncnn_5  1 2 Exp Exp_splitncnn_0 Exp_splitncnn_1
Reduction   Sum          1 1 Exp_splitncnn_1 Sum 0 -2 1.000000
BinaryOp    prob         2 1 Exp_splitncnn_0 Sum prob 3
```