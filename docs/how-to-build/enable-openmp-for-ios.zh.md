### 背景知识

目前，最新版本的 Xcode-8.2.1 携带的 clang 编译器不具有 openmp 特性，Apple 方面比较推 GCD 技术，网络上能找到的大多是教你如何把 openmp 代码改用 GCD 实现，而关于如何真正用 openmp，这是第一篇吧

这篇文章记录 up 主在 Linux 上用 clang 编译器和交叉编译方式实现 ios 的 openmp 加速

### 准备交叉编译环境

系统：fedora 25

工具链：llvm 3.8.1，clang 3.8.0，fuse 2.9.7

从 Apple 官网上下载 Xcode_7.3.1.dmg，因为 cctools-port 还没有支持最新版的 Xcode
https://developer.apple.com/download/more/

安装 Xcode dmg 挂载工具
https://github.com/darlinghq/darling-dmg
```
$ mkdir xcode
$ ./darling-dmg/build/darling-dmg Xcode_7.3.1.dmg xcode
Skipping partition of type Primary GPT Header
Skipping partition of type Primary GPT Table
Skipping partition of type Apple_Free
Skipping partition of type C12A7328-F81F-11D2-BA4B-00A0C93EC93B
Using partition #4 of type Apple_HFS
Everything looks OK, disk mounted
```
提取 ios sdk，卸载 dmg
```
$ mkdir -p iPhoneSDK/iPhoneOS9.3.sdk
$ cp -r xcode/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk/* iPhoneSDK/iPhoneOS9.3.sdk
$ cp -r xcode/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/* iPhoneSDK/iPhoneOS9.3.sdk/usr/include/c++
$ fusermount -u xcode  # unmount the image
```
打包 ios sdk
```
$ cd iPhoneSDK
$ tar -cf - iPhoneOS9.3.sdk | xz -9 -c - > iPhoneOS9.3.sdk.tar.xz
```
安装 cctools 移植版本，感谢 cjacker 和 tpoechtrager 的辛勤付出！
https://github.com/tpoechtrager/cctools-port
```
$ cd cctools-port/usage_examples/ios_toolchain
$ ./build.sh iPhoneOS9.3.sdk.tar.xz armv7
```
交叉编译的工具链在 cctools-port/usage_examples/ios_toolchain/target/bin 目录下

### 编译 libomp

下载 llvm 官网上的 openmp 运行时
```
svn co http://llvm.org/svn/llvm-project/openmp/trunk openmp
```
准备好 ios 交叉编译的 cmake toolchain，文件名 iosxc.toolchain.cmake，放在 openmp 目录中
```cmake
# standard settings
set (CMAKE_SYSTEM_NAME Darwin)
set (CMAKE_SYSTEM_VERSION 1)
set (UNIX True)
set (APPLE True)
set (IOS True)

set(CMAKE_C_COMPILER arm-apple-darwin11-clang)
set(CMAKE_CXX_COMPILER arm-apple-darwin11-clang++)

set(_CMAKE_TOOLCHAIN_PREFIX arm-apple-darwin11-)

# 这里指定交叉编译工具链中的 sdk 目录
set(CMAKE_IOS_SDK_ROOT "/home/nihui/osd/cctools-port/usage_examples/ios_toolchain/target/SDK/")

# set the sysroot default to the most recent SDK
set(CMAKE_OSX_SYSROOT ${CMAKE_IOS_SDK_ROOT} CACHE PATH "Sysroot used for iOS support")

# set the architecture for iOS 双架构
set(IOS_ARCH armv7;arm64)
set(CMAKE_OSX_ARCHITECTURES ${IOS_ARCH} CACHE string "Build architecture for iOS")

# set the find root to the iOS developer roots and to user defined paths
set(CMAKE_FIND_ROOT_PATH ${CMAKE_IOS_DEVELOPER_ROOT} ${CMAKE_IOS_SDK_ROOT} ${CMAKE_PREFIX_PATH} CACHE string "iOS find search path root")

# searching for frameworks only
set(CMAKE_FIND_FRAMEWORK FIRST)

# set up the default search directories for frameworks
set(CMAKE_SYSTEM_FRAMEWORK_PATH
    ${CMAKE_IOS_SDK_ROOT}/System/Library/Frameworks
)
```
直接编译的话会失败，所以 up 主自己弄了下面三个补丁，放在附件里可以下载
编译补丁A，ios sdk 没有 crt_externs.h 头文件，改为经典声明
```
diff --git a/runtime/src/kmp_environment.cpp b/runtime/src/kmp_environment.cpp
index d4d95df..c8c2970 100644
--- a/runtime/src/kmp_environment.cpp
+++ b/runtime/src/kmp_environment.cpp
@@ -64,12 +64,12 @@
 #if KMP_OS_UNIX
     #include <stdlib.h>    // getenv, setenv, unsetenv.
     #include <string.h>    // strlen, strcpy.
-    #if KMP_OS_DARWIN
-        #include <crt_externs.h>
-        #define environ (*_NSGetEnviron())
-    #else
+//     #if KMP_OS_DARWIN
+//         #include <crt_externs.h>
+//         #define environ (*_NSGetEnviron())
+//     #else
         extern char * * environ;
-    #endif
+//     #endif
 #elif KMP_OS_WINDOWS
     #include <windows.h>   // GetEnvironmentVariable, SetEnvironmentVariable, GetLastError.
 #else

```
```
$ patch -p1 -i openmp-ios-classic-environ.patch
```
编译补丁B，clang 不支持 .size 的语法，删除，并为符号名字前补上额外下划线，不然会链接失败
```
diff --git a/runtime/src/z_Linux_asm.s b/runtime/src/z_Linux_asm.s
index d6e1c0b..69f94ef 100644
--- a/runtime/src/z_Linux_asm.s
+++ b/runtime/src/z_Linux_asm.s
@@ -1781,10 +1781,10 @@ __kmp_invoke_microtask:
     .comm .gomp_critical_user_,32,8
     .data
     .align 4
-    .global __kmp_unnamed_critical_addr
-__kmp_unnamed_critical_addr:
+    .global ___kmp_unnamed_critical_addr
+___kmp_unnamed_critical_addr:
     .4byte .gomp_critical_user_
-    .size __kmp_unnamed_critical_addr,4
+//    .size __kmp_unnamed_critical_addr,4
 #endif /* KMP_ARCH_ARM */
 
 #if KMP_ARCH_PPC64 || KMP_ARCH_AARCH64 || KMP_ARCH_MIPS64
@@ -1792,10 +1792,10 @@ __kmp_unnamed_critical_addr:
     .comm .gomp_critical_user_,32,8
     .data
     .align 8
-    .global __kmp_unnamed_critical_addr
-__kmp_unnamed_critical_addr:
+    .global ___kmp_unnamed_critical_addr
+___kmp_unnamed_critical_addr:
     .8byte .gomp_critical_user_
-    .size __kmp_unnamed_critical_addr,8
+//    .size __kmp_unnamed_critical_addr,8
 #endif /* KMP_ARCH_PPC64 || KMP_ARCH_AARCH64 */
 
 #if KMP_OS_LINUX

```
```
$ patch -p1 -i openmp-kmp_unnamed_critical_addr-clang-arm-build-fix.patch
```
编译补丁C，交叉编译的工具链会把 complex 类型的除法放在 compiler-rt builtin library 实现，但是 ios sdk 本身没有，为了免去麻烦就直接删掉了。这个补丁也许在正常的 macos 下不需要，不过 up 主不用 macos 也就不管了
```
diff --git a/runtime/src/kmp_atomic.cpp b/runtime/src/kmp_atomic.cpp
index 3831165..b969175 100644
--- a/runtime/src/kmp_atomic.cpp
+++ b/runtime/src/kmp_atomic.cpp
@@ -1139,23 +1139,23 @@ ATOMIC_CRITICAL( float16, div, QUAD_LEGACY,     /, 16r,   1 )            // __km
 ATOMIC_CMPXCHG_WORKAROUND( cmplx4, add, kmp_cmplx32, 64, +, 8c, 7, 1 )   // __kmpc_atomic_cmplx4_add
 ATOMIC_CMPXCHG_WORKAROUND( cmplx4, sub, kmp_cmplx32, 64, -, 8c, 7, 1 )   // __kmpc_atomic_cmplx4_sub
 ATOMIC_CMPXCHG_WORKAROUND( cmplx4, mul, kmp_cmplx32, 64, *, 8c, 7, 1 )   // __kmpc_atomic_cmplx4_mul
-ATOMIC_CMPXCHG_WORKAROUND( cmplx4, div, kmp_cmplx32, 64, /, 8c, 7, 1 )   // __kmpc_atomic_cmplx4_div
+// ATOMIC_CMPXCHG_WORKAROUND( cmplx4, div, kmp_cmplx32, 64, /, 8c, 7, 1 )   // __kmpc_atomic_cmplx4_div
 // end of the workaround for C78287
 #else
 ATOMIC_CRITICAL( cmplx4,  add, kmp_cmplx32,     +,  8c,   1 )            // __kmpc_atomic_cmplx4_add
 ATOMIC_CRITICAL( cmplx4,  sub, kmp_cmplx32,     -,  8c,   1 )            // __kmpc_atomic_cmplx4_sub
 ATOMIC_CRITICAL( cmplx4,  mul, kmp_cmplx32,     *,  8c,   1 )            // __kmpc_atomic_cmplx4_mul
-ATOMIC_CRITICAL( cmplx4,  div, kmp_cmplx32,     /,  8c,   1 )            // __kmpc_atomic_cmplx4_div
+// ATOMIC_CRITICAL( cmplx4,  div, kmp_cmplx32,     /,  8c,   1 )            // __kmpc_atomic_cmplx4_div
 #endif // USE_CMPXCHG_FIX
 
 ATOMIC_CRITICAL( cmplx8,  add, kmp_cmplx64,     +, 16c,   1 )            // __kmpc_atomic_cmplx8_add
 ATOMIC_CRITICAL( cmplx8,  sub, kmp_cmplx64,     -, 16c,   1 )            // __kmpc_atomic_cmplx8_sub
 ATOMIC_CRITICAL( cmplx8,  mul, kmp_cmplx64,     *, 16c,   1 )            // __kmpc_atomic_cmplx8_mul
-ATOMIC_CRITICAL( cmplx8,  div, kmp_cmplx64,     /, 16c,   1 )            // __kmpc_atomic_cmplx8_div
+// ATOMIC_CRITICAL( cmplx8,  div, kmp_cmplx64,     /, 16c,   1 )            // __kmpc_atomic_cmplx8_div
 ATOMIC_CRITICAL( cmplx10, add, kmp_cmplx80,     +, 20c,   1 )            // __kmpc_atomic_cmplx10_add
 ATOMIC_CRITICAL( cmplx10, sub, kmp_cmplx80,     -, 20c,   1 )            // __kmpc_atomic_cmplx10_sub
 ATOMIC_CRITICAL( cmplx10, mul, kmp_cmplx80,     *, 20c,   1 )            // __kmpc_atomic_cmplx10_mul
-ATOMIC_CRITICAL( cmplx10, div, kmp_cmplx80,     /, 20c,   1 )            // __kmpc_atomic_cmplx10_div
+// ATOMIC_CRITICAL( cmplx10, div, kmp_cmplx80,     /, 20c,   1 )            // __kmpc_atomic_cmplx10_div
 #if KMP_HAVE_QUAD
 ATOMIC_CRITICAL( cmplx16, add, CPLX128_LEG,     +, 32c,   1 )            // __kmpc_atomic_cmplx16_add
 ATOMIC_CRITICAL( cmplx16, sub, CPLX128_LEG,     -, 32c,   1 )            // __kmpc_atomic_cmplx16_sub
@@ -1541,7 +1541,7 @@ ATOMIC_BEGIN_MIX(TYPE_ID,TYPE,OP_ID,RTYPE_ID,RTYPE)
 ATOMIC_CMPXCHG_CMPLX( cmplx4, kmp_cmplx32, add, 64, +, cmplx8,  kmp_cmplx64,  8c, 7, KMP_ARCH_X86 ) // __kmpc_atomic_cmplx4_add_cmplx8
 ATOMIC_CMPXCHG_CMPLX( cmplx4, kmp_cmplx32, sub, 64, -, cmplx8,  kmp_cmplx64,  8c, 7, KMP_ARCH_X86 ) // __kmpc_atomic_cmplx4_sub_cmplx8
 ATOMIC_CMPXCHG_CMPLX( cmplx4, kmp_cmplx32, mul, 64, *, cmplx8,  kmp_cmplx64,  8c, 7, KMP_ARCH_X86 ) // __kmpc_atomic_cmplx4_mul_cmplx8
-ATOMIC_CMPXCHG_CMPLX( cmplx4, kmp_cmplx32, div, 64, /, cmplx8,  kmp_cmplx64,  8c, 7, KMP_ARCH_X86 ) // __kmpc_atomic_cmplx4_div_cmplx8
+// ATOMIC_CMPXCHG_CMPLX( cmplx4, kmp_cmplx32, div, 64, /, cmplx8,  kmp_cmplx64,  8c, 7, KMP_ARCH_X86 ) // __kmpc_atomic_cmplx4_div_cmplx8
 
 // READ, WRITE, CAPTURE are supported only on IA-32 architecture and Intel(R) 64
 #if KMP_ARCH_X86 || KMP_ARCH_X86_64
diff --git a/runtime/src/kmp_atomic.h b/runtime/src/kmp_atomic.h
index 7a98de6..d3d37c2 100644
--- a/runtime/src/kmp_atomic.h
+++ b/runtime/src/kmp_atomic.h
@@ -573,15 +573,15 @@ void __kmpc_atomic_float16_div( ident_t *id_ref, int gtid, QUAD_LEGACY * lhs, QU
 void __kmpc_atomic_cmplx4_add(  ident_t *id_ref, int gtid, kmp_cmplx32 * lhs, kmp_cmplx32 rhs );
 void __kmpc_atomic_cmplx4_sub(  ident_t *id_ref, int gtid, kmp_cmplx32 * lhs, kmp_cmplx32 rhs );
 void __kmpc_atomic_cmplx4_mul(  ident_t *id_ref, int gtid, kmp_cmplx32 * lhs, kmp_cmplx32 rhs );
-void __kmpc_atomic_cmplx4_div(  ident_t *id_ref, int gtid, kmp_cmplx32 * lhs, kmp_cmplx32 rhs );
+// void __kmpc_atomic_cmplx4_div(  ident_t *id_ref, int gtid, kmp_cmplx32 * lhs, kmp_cmplx32 rhs );
 void __kmpc_atomic_cmplx8_add(  ident_t *id_ref, int gtid, kmp_cmplx64 * lhs, kmp_cmplx64 rhs );
 void __kmpc_atomic_cmplx8_sub(  ident_t *id_ref, int gtid, kmp_cmplx64 * lhs, kmp_cmplx64 rhs );
 void __kmpc_atomic_cmplx8_mul(  ident_t *id_ref, int gtid, kmp_cmplx64 * lhs, kmp_cmplx64 rhs );
-void __kmpc_atomic_cmplx8_div(  ident_t *id_ref, int gtid, kmp_cmplx64 * lhs, kmp_cmplx64 rhs );
+// void __kmpc_atomic_cmplx8_div(  ident_t *id_ref, int gtid, kmp_cmplx64 * lhs, kmp_cmplx64 rhs );
 void __kmpc_atomic_cmplx10_add( ident_t *id_ref, int gtid, kmp_cmplx80 * lhs, kmp_cmplx80 rhs );
 void __kmpc_atomic_cmplx10_sub( ident_t *id_ref, int gtid, kmp_cmplx80 * lhs, kmp_cmplx80 rhs );
 void __kmpc_atomic_cmplx10_mul( ident_t *id_ref, int gtid, kmp_cmplx80 * lhs, kmp_cmplx80 rhs );
-void __kmpc_atomic_cmplx10_div( ident_t *id_ref, int gtid, kmp_cmplx80 * lhs, kmp_cmplx80 rhs );
+// void __kmpc_atomic_cmplx10_div( ident_t *id_ref, int gtid, kmp_cmplx80 * lhs, kmp_cmplx80 rhs );
 #if KMP_HAVE_QUAD
 void __kmpc_atomic_cmplx16_add( ident_t *id_ref, int gtid, CPLX128_LEG * lhs, CPLX128_LEG rhs );
 void __kmpc_atomic_cmplx16_sub( ident_t *id_ref, int gtid, CPLX128_LEG * lhs, CPLX128_LEG rhs );
@@ -753,7 +753,7 @@ void __kmpc_atomic_float10_div_rev_fp( ident_t *id_ref, int gtid, long double *
 void __kmpc_atomic_cmplx4_add_cmplx8( ident_t *id_ref, int gtid, kmp_cmplx32 * lhs, kmp_cmplx64 rhs );
 void __kmpc_atomic_cmplx4_sub_cmplx8( ident_t *id_ref, int gtid, kmp_cmplx32 * lhs, kmp_cmplx64 rhs );
 void __kmpc_atomic_cmplx4_mul_cmplx8( ident_t *id_ref, int gtid, kmp_cmplx32 * lhs, kmp_cmplx64 rhs );
-void __kmpc_atomic_cmplx4_div_cmplx8( ident_t *id_ref, int gtid, kmp_cmplx32 * lhs, kmp_cmplx64 rhs );
+// void __kmpc_atomic_cmplx4_div_cmplx8( ident_t *id_ref, int gtid, kmp_cmplx32 * lhs, kmp_cmplx64 rhs );
 
 // generic atomic routines
 void __kmpc_atomic_1(  ident_t *id_ref, int gtid, void* lhs, void* rhs, void (*f)( void *, void *, void * ) );

```
```
$ patch -p1 -i openmp-atomic-drop-complex-div.patch
```
编译 libomp 静态库
```
$ mkdir build-ios
$ cd build-ios
$ cmake -DCMAKE_TOOLCHAIN_FILE=../iosxc.toolchain.cmake -DLIBOMP_ENABLE_SHARED=off ..
$ make
```
编译完成后，libomp.a 和 omp.h 在 openmp/build-ios/runtime/src 目录里
把这两个文件分别放在 cctools-port/usage_examples/ios_toolchain/target/SDK/usr/lib 和 cctools-port/usage_examples/ios_toolchain/target/SDK/usr/include 里面，成为 sdk 的一部分

### openmp 测试程序
```
#include <stdio.h>
#include "omp.h"

int main()
{
    #pragma omp parallel for
    for (int i=0; i<20; i++)
    {
        fprintf(stderr, "%d\n", i);
    }
}
```
编译方法，增加 -fopenmp 参数
```
$ cctools-port/usage_examples/ios_toolchain/target/bin/arm-apple-darwin11-clang -fopenmp testomp.c -o testomp
```
找一台双核cpu的越狱设备，比如这个 ipad2，把程序上传后运行，乱许输出表明 openmp 是可用的了
```
Chengjiede-iPad:~ root# ./testomp
0
1
2
3
10
4
11
5
12
6
13
7
14
8
15
9
16
17
18
19
```
