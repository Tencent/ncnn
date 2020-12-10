### param is too old, please regenerate

Your model file is being the old format converted by an old caffe2ncnn tool.

Checkout the latest ncnn code, build it and regenerate param and model binary files, and that should work.

Make sure that your param file starts with the magic number 7767517.

you may find more info on [use-ncnn-with-alexnet](use-ncnn-with-alexnet)

### find_blob_index_by_name XYZ failed

That means ncnn couldn't find the XYZ blob in the network. 

You shall call Extractor::input()/extract() by blob name instead of layer name.

For models loaded from binary param file or external memory, you shall call Extractor::input()/extract() by the enum defined in xxx.id.h because all the visible string literals have been stripped in binary form.

This error usually happens when the input layer is not properly converted.

You shall upgrade caffe prototxt/caffemodel before converting it to ncnn. Following snnipet type shall be ok. 

```
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape: { dim: 1 dim: 3 dim: 227 dim: 227 } }
}
```

you may find more info on [use-ncnn-with-alexnet](use-ncnn-with-alexnet).

### layer XYZ not exists or registered

Your network contains some operations that are not implemented in ncnn.

You may implement them as custom layer followed in [how-to-implement-custom-layer-step-by-step](how-to-implement-custom-layer-step-by-step).

Or you could simply register them as no-op if you are sure those operations make no sense.

```cpp
class Noop : public ncnn::Layer {};
DEFINE_LAYER_CREATOR(Noop)

net.register_custom_layer("LinearRegressionOutput", Noop_layer_creator);
net.register_custom_layer("MAERegressionOutput", Noop_layer_creator);
```

### fopen XYZ.param/XYZ.bin failed

File not found or not readable. Make sure that XYZ.param/XYZ.bin is accessible.

### network graph not ready

You shall call Net::load_param() first, then Net::load_model().

This error may also happens when Net::load_param() failed, but not properly handled.

For more information about the ncnn model load api, see [ncnn-load-model](ncnn-load-model)

### memory not 32-bit aligned at XYZ

The pointer passed to Net::load_param() or Net::load_model() is not 32bit aligned.

In practice, the head pointer of std::vector<unsigned char> is not guaranteed to be 32bit aligned.

you can store your binary buffer in ncnn::Mat structure, its internal memory is aligned.

### undefined reference to '__kmpc_XYZ_XYZ'

use clang for building android shared library

comment the following line in your Application.mk
```
NDK_TOOLCHAIN_VERSION := 4.9
```

### crash on android with '__kmp_abort_process'

This usually happens if you bundle multiple shared library with openmp linked

It is actually an issue of the android ndk https://github.com/android/ndk/issues/1028

On old android ndk, modify the link flags as

```
-Wl,-Bstatic -lomp -Wl,-Bdynamic
```

For recent ndk >= 21

```
-fstatic-openmp
```

### dlopen failed: library "libomp.so" not found

Newer android ndk defaults to dynamic openmp runtime

modify the link flags as

```
-fstatic-openmp -fopenmp
```

### crash when freeing a ncnn dynamic library(*.dll/*.so) built with openMP

for optimal performance, the openmp threadpool spin waits for about a second prior to shutting down in case more work becomes available. 

If you unload a dynamic library that's in the process of spin-waiting, it will crash in the manner you see (most of the time).

Just set OMP_WAIT_POLICY=passive in your environment, before calling loadlibrary. or Just wait a few seconds before calling freelibrary.

You can also use the following method to set environment variables in your code:

for msvc++:

```
SetEnvironmentVariable(_T("OMP_WAIT_POLICY"), _T("passive"));
```

for g++:

```
setenv("OMP_WAIT_POLICY", "passive", 1)
```

reference: https://stackoverflow.com/questions/34439956/vc-crash-when-freeing-a-dll-built-with-openmp
