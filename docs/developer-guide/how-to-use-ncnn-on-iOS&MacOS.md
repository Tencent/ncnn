# How to use ncnn framework on iOS or MacOS?
## [Build on myself](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-ios-on-macos-with-xcode)
## Using GitHub Release package

Download the release version in [Github Release Page](https://github.com/Tencent/ncnn/releases)

Copy the Lib and Include folder to youself program

In Xcode, add the ncnn.framework,libomp ,libglslang,libMachineIndependent,libGenericCodeGen,libSPIRV,libOGLCompiler and libOSDependent libraries to General → Embedded Binaries

In Xcode, add Include folder to Header search path

Use
```
#include <ncnn/mat.h>
#include <ncnn/xxx.h>
```