### Non ARM Linux Platform

the typical usage
```
ncnnoptimize mobilenet.param mobilenet.bin mobilenet-opt.param mobilenet-opt.bin 65536 
```

operator fusion
* batchnorm - scale
* convolution - batchnorm
* convolutiondepthwise - batchnorm
* deconvolution - batchnorm
* deconvolutiondepthwise - batchnorm
* innerproduct - batchnorm
* convolution - relu
* convolutiondepthwise - relu
* deconvolution - relu
* deconvolutiondepthwise - relu
* innerproduct - relu

eliminate noop operator
* innerproduct - dropout
* flatten after global pooling

prefer better operator
* replace convolution with innerproduct after global pooling

### ARM Linux Platform
usage
```
ncnnoptimize squeezenet.param squeezenet.bin squeezenet-opt.param squeezenet-opt.bin 0 data 227 224 3
```

explanation

|parameter|meaning|
|---|---|
|data|input data node, currently support one input|
|227|input weight|
|224|input height|
|3|input channel|

this feature would auto choose the fastest convolution implementation, normally speedup 10%.
