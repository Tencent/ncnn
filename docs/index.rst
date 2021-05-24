.. ncnn documentation master file, created by
   sphinx-quickstart on Wed Mar 24 22:40:10 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ncnn's documentation!
===================================


*Please select a specific version of the document in the lower left corner of the page.*

.. toctree::
  :maxdepth: 1 
  :caption: how to build
  :name: sec-introduction

  how-to-build/how-to-build
  how-to-build/build-mlir2ncnn
  how-to-build/build-for-VS2017.zh
  how-to-build/build-minimal-library

.. toctree::
  :maxdepth: 1
  :caption: how to use 
  :name: sec-how-to-use

  how-to-use-and-FAQ/efficient-roi-resize-rotate
  how-to-use-and-FAQ/FAQ-ncnn-protobuf-problem.zh
  how-to-use-and-FAQ/FAQ-ncnn-produce-wrong-result
  how-to-use-and-FAQ/FAQ-ncnn-throw-error
  how-to-use-and-FAQ/FAQ-ncnn-vulkan
  how-to-use-and-FAQ/ncnn-load-model
  how-to-use-and-FAQ/openmp-best-practice
  how-to-use-and-FAQ/openmp-best-practice.zh
  how-to-use-and-FAQ/quantized-int8-inference
  how-to-use-and-FAQ/use-ncnnoptimize-to-optimize-model
  how-to-use-and-FAQ/use-ncnn-with-alexnet
  how-to-use-and-FAQ/use-ncnn-with-alexnet.zh
  how-to-use-and-FAQ/use-ncnn-with-opencv
  how-to-use-and-FAQ/use-ncnn-with-own-project
  how-to-use-and-FAQ/use-ncnn-with-pytorch-or-onnx
  how-to-use-and-FAQ/vulkan-notes

.. toctree::
  :maxdepth: 1
  :caption: benchmark
  :name: sec-benchmark
  
  benchmark/the-benchmark-of-caffe-android-lib,-mini-caffe,-and-ncnn
  benchmark/vulkan-conformance-test

.. toctree::
  :maxdepth: 1
  :caption: developer guide
  :name: sec-dev-guide

  developer-guide/aarch64-mix-assembly-and-intrinsic
  developer-guide/add-custom-layer.zh
  developer-guide/arm-a53-a55-dual-issue
  developer-guide/armv7-mix-assembly-and-intrinsic
  developer-guide/binaryop-broadcasting
  developer-guide/custom-allocator
  developer-guide/element-packing
  developer-guide/how-to-implement-custom-layer-step-by-step
  developer-guide/how-to-write-a-neon-optimized-op-kernel
  developer-guide/how-to-be-a-contributor.zh
  developer-guide/low-level-operation-api
  developer-guide/ncnn-tips-and-tricks.zh
  developer-guide/new-model-load-api
  developer-guide/new-param-load-api
  developer-guide/operation-param-weight-table
  developer-guide/operators
  developer-guide/preload-practice.zh
  developer-guide/param-and-model-file-structure
  developer-guide/tensorflow-op-combination
  
.. toctree::
  :maxdepth: 1
  :caption: home

  home
  faq
  application-with-ncnn-inside

