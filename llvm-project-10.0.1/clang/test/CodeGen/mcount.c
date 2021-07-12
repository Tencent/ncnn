// RUN: %clang_cc1 -pg -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -pg -triple i386-unknown-unknown -emit-llvm -O2 -o - %s | FileCheck %s
// RUN: %clang_cc1 -pg -triple powerpc-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple powerpc64-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple powerpc64le-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple i386-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple x86_64-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple arm-netbsd-eabi -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple aarch64-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple mips-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple mips-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple mipsel-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple mips64-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple mips64el-unknown-gnu-linux -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple powerpc-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple powerpc64-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple powerpc64le-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple sparc-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -pg -triple sparc64-netbsd -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK-DOUBLE-PREFIXED,NO-MCOUNT1 %s
// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s -check-prefix=NO-MCOUNT

int bar(void) {
  return 0;
}

int foo(void) {
  return bar();
}

int __attribute__((no_instrument_function)) no_instrument(void) {
  return foo();
}

int main(void) {
  return no_instrument();
}

// CHECK: attributes #0 = { {{.*}}"instrument-function-entry-inlined"="mcount"{{.*}} }
// CHECK: attributes #1 = { {{.*}} }
// CHECK-PREFIXED: attributes #0 = { {{.*}}"instrument-function-entry-inlined"="_mcount"{{.*}} }
// CHECK-PREFIXED: attributes #1 = { {{.*}} }
// CHECK-DOUBLE-PREFIXED: attributes #0 = { {{.*}}"instrument-function-entry-inlined"="__mcount"{{.*}} }
// CHECK-DOUBLE-PREFIXED: attributes #1 = { {{.*}} }
// NO-MCOUNT-NOT: attributes #{{[0-9]}} = { {{.*}}"instrument-function-entry-inlined"={{.*}} }
// NO-MCOUNT1-NOT: attributes #1 = { {{.*}}"instrument-function-entry-inlined"={{.*}} }
