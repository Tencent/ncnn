; RUN: llc -mtriple armeb-eabi -mattr v7,neon -float-abi soft %s -o - | FileCheck %s -check-prefix CHECK -check-prefix SOFT
; RUN: llc -mtriple armeb-eabi -mattr v7,neon -float-abi hard %s -o - | FileCheck %s -check-prefix CHECK -check-prefix HARD

; CHECK-LABEL: test_i64_f64:
declare i64 @test_i64_f64_helper(double %p)
define void @test_i64_f64(double* %p, i64* %q) {
; SOFT: vadd.f64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.f64 d0
    %1 = load double, double* %p
    %2 = fadd double %1, %1
    %3 = call i64 @test_i64_f64_helper(double %2)
    %4 = add i64 %3, %3
    store i64 %4, i64* %q
    ret void
; CHECK: adds r1
; CHECK: adc r0
}

; CHECK-LABEL: test_i64_v1i64:
declare i64 @test_i64_v1i64_helper(<1 x i64> %p)
define void @test_i64_v1i64(<1 x i64>* %p, i64* %q) {
; SOFT: vadd.i64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.i64 d0
    %1 = load <1 x i64>, <1 x i64>* %p
    %2 = add <1 x i64> %1, %1
    %3 = call i64 @test_i64_v1i64_helper(<1 x i64> %2)
    %4 = add i64 %3, %3
    store i64 %4, i64* %q
    ret void
; CHECK: adds r1
; CHECK: adc r0
}

; CHECK-LABEL: test_i64_v2f32:
declare i64 @test_i64_v2f32_helper(<2 x float> %p)
define void @test_i64_v2f32(<2 x float>* %p, i64* %q) {
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
    %1 = load <2 x float>, <2 x float>* %p
    %2 = fadd <2 x float> %1, %1
    %3 = call i64 @test_i64_v2f32_helper(<2 x float> %2)
    %4 = add i64 %3, %3
    store i64 %4, i64* %q
    ret void
; CHECK: adds r1
; CHECK: adc r0
}

; CHECK-LABEL: test_i64_v2i32:
declare i64 @test_i64_v2i32_helper(<2 x i32> %p)
define void @test_i64_v2i32(<2 x i32>* %p, i64* %q) {
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
    %1 = load <2 x i32>, <2 x i32>* %p
    %2 = add <2 x i32> %1, %1
    %3 = call i64 @test_i64_v2i32_helper(<2 x i32> %2)
    %4 = add i64 %3, %3
    store i64 %4, i64* %q
    ret void
; CHECK: adds r1
; CHECK: adc r0
}

; CHECK-LABEL: test_i64_v4i16:
declare i64 @test_i64_v4i16_helper(<4 x i16> %p)
define void @test_i64_v4i16(<4 x i16>* %p, i64* %q) {
; SOFT: vrev64.16 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.16 d0
    %1 = load <4 x i16>, <4 x i16>* %p
    %2 = add <4 x i16> %1, %1
    %3 = call i64 @test_i64_v4i16_helper(<4 x i16> %2)
    %4 = add i64 %3, %3
    store i64 %4, i64* %q
    ret void
; CHECK: adds r1
; CHECK: adc r0
}

; CHECK-LABEL: test_i64_v8i8:
declare i64 @test_i64_v8i8_helper(<8 x i8> %p)
define void @test_i64_v8i8(<8 x i8>* %p, i64* %q) {
; SOFT: vrev64.8 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.8 d0
    %1 = load <8 x i8>, <8 x i8>* %p
    %2 = add <8 x i8> %1, %1
    %3 = call i64 @test_i64_v8i8_helper(<8 x i8> %2)
    %4 = add i64 %3, %3
    store i64 %4, i64* %q
    ret void
; CHECK: adds r1
; CHECK: adc r0
}

; CHECK-LABEL: test_f64_i64:
declare double @test_f64_i64_helper(i64 %p)
define void @test_f64_i64(i64* %p, double* %q) {
; CHECK: adds r1
; CHECK: adc r0
    %1 = load i64, i64* %p
    %2 = add i64 %1, %1
    %3 = call double @test_f64_i64_helper(i64 %2)
    %4 = fadd double %3, %3
    store double %4, double* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.f64 [[REG]]
; HARD: vadd.f64 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_f64_v1i64:
declare double @test_f64_v1i64_helper(<1 x i64> %p)
define void @test_f64_v1i64(<1 x i64>* %p, double* %q) {
; SOFT: vadd.i64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.i64 d0
    %1 = load <1 x i64>, <1 x i64>* %p
    %2 = add <1 x i64> %1, %1
    %3 = call double @test_f64_v1i64_helper(<1 x i64> %2)
    %4 = fadd double %3, %3
    store double %4, double* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.f64 [[REG]]
; HARD: vadd.f64 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_f64_v2f32:
declare double @test_f64_v2f32_helper(<2 x float> %p)
define void @test_f64_v2f32(<2 x float>* %p, double* %q) {
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
    %1 = load <2 x float>, <2 x float>* %p
    %2 = fadd <2 x float> %1, %1
    %3 = call double @test_f64_v2f32_helper(<2 x float> %2)
    %4 = fadd double %3, %3
    store double %4, double* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.f64 [[REG]]
; HARD: vadd.f64 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_f64_v2i32:
declare double @test_f64_v2i32_helper(<2 x i32> %p)
define void @test_f64_v2i32(<2 x i32>* %p, double* %q) {
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
    %1 = load <2 x i32>, <2 x i32>* %p
    %2 = add <2 x i32> %1, %1
    %3 = call double @test_f64_v2i32_helper(<2 x i32> %2)
    %4 = fadd double %3, %3
    store double %4, double* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.f64 [[REG]]
; HARD: vadd.f64 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_f64_v4i16:
declare double @test_f64_v4i16_helper(<4 x i16> %p)
define void @test_f64_v4i16(<4 x i16>* %p, double* %q) {
; SOFT: vrev64.16 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.16 d0
    %1 = load <4 x i16>, <4 x i16>* %p
    %2 = add <4 x i16> %1, %1
    %3 = call double @test_f64_v4i16_helper(<4 x i16> %2)
    %4 = fadd double %3, %3
    store double %4, double* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.f64 [[REG]]
; HARD: vadd.f64 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_f64_v8i8:
declare double @test_f64_v8i8_helper(<8 x i8> %p)
define void @test_f64_v8i8(<8 x i8>* %p, double* %q) {
; SOFT: vrev64.8 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.8 d0
    %1 = load <8 x i8>, <8 x i8>* %p
    %2 = add <8 x i8> %1, %1
    %3 = call double @test_f64_v8i8_helper(<8 x i8> %2)
    %4 = fadd double %3, %3
    store double %4, double* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.f64 [[REG]]
; HARD: vadd.f64 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v1i64_i64:
declare <1 x i64> @test_v1i64_i64_helper(i64 %p)
define void @test_v1i64_i64(i64* %p, <1 x i64>* %q) {
; CHECK: adds r1
; CHECK: adc r0
    %1 = load i64, i64* %p
    %2 = add i64 %1, %1
    %3 = call <1 x i64> @test_v1i64_i64_helper(i64 %2)
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, <1 x i64>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.i64 [[REG]]
; HARD: vadd.i64 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v1i64_f64:
declare <1 x i64> @test_v1i64_f64_helper(double %p)
define void @test_v1i64_f64(double* %p, <1 x i64>* %q) {
; SOFT: vadd.f64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.f64 d0
    %1 = load double, double* %p
    %2 = fadd double %1, %1
    %3 = call <1 x i64> @test_v1i64_f64_helper(double %2)
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, <1 x i64>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.i64 [[REG]]
; HARD: vadd.i64 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v1i64_v2f32:
declare <1 x i64> @test_v1i64_v2f32_helper(<2 x float> %p)
define void @test_v1i64_v2f32(<2 x float>* %p, <1 x i64>* %q) {
; HARD: vrev64.32 d0
; SOFT: vadd.f32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
    %1 = load <2 x float>, <2 x float>* %p
    %2 = fadd <2 x float> %1, %1
    %3 = call <1 x i64> @test_v1i64_v2f32_helper(<2 x float> %2)
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, <1 x i64>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.i64 [[REG]]
; HARD: vadd.i64 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v1i64_v2i32:
declare <1 x i64> @test_v1i64_v2i32_helper(<2 x i32> %p)
define void @test_v1i64_v2i32(<2 x i32>* %p, <1 x i64>* %q) {
; HARD: vrev64.32 d0
; SOFT: vadd.i32 [[REG:d[0-9]+]]
; SOFT: vrev64.32 [[REG]]
; SOFT: vmov r1, r0, [[REG]]
    %1 = load <2 x i32>, <2 x i32>* %p
    %2 = add <2 x i32> %1, %1
    %3 = call <1 x i64> @test_v1i64_v2i32_helper(<2 x i32> %2)
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, <1 x i64>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.i64 [[REG]]
; HARD: vadd.i64 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v1i64_v4i16:
declare <1 x i64> @test_v1i64_v4i16_helper(<4 x i16> %p)
define void @test_v1i64_v4i16(<4 x i16>* %p, <1 x i64>* %q) {
; SOFT: vrev64.16 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.16 d0
    %1 = load <4 x i16>, <4 x i16>* %p
    %2 = add <4 x i16> %1, %1
    %3 = call <1 x i64> @test_v1i64_v4i16_helper(<4 x i16> %2)
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, <1 x i64>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.i64 [[REG]]
; HARD: vadd.i64 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v1i64_v8i8:
declare <1 x i64> @test_v1i64_v8i8_helper(<8 x i8> %p)
define void @test_v1i64_v8i8(<8 x i8>* %p, <1 x i64>* %q) {
; SOFT: vrev64.8 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.8 d0
    %1 = load <8 x i8>, <8 x i8>* %p
    %2 = add <8 x i8> %1, %1
    %3 = call <1 x i64> @test_v1i64_v8i8_helper(<8 x i8> %2)
    %4 = add <1 x i64> %3, %3
    store <1 x i64> %4, <1 x i64>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vadd.i64 [[REG]]
; HARD: vadd.i64 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v2f32_i64:
declare <2 x float> @test_v2f32_i64_helper(i64 %p)
define void @test_v2f32_i64(i64* %p, <2 x float>* %q) {
; CHECK: adds r1
; CHECK: adc r0
    %1 = load i64, i64* %p
    %2 = add i64 %1, %1
    %3 = call <2 x float> @test_v2f32_i64_helper(i64 %2)
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, <2 x float>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v2f32_f64:
declare <2 x float> @test_v2f32_f64_helper(double %p)
define void @test_v2f32_f64(double* %p, <2 x float>* %q) {
; SOFT: vadd.f64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.f64 d0
    %1 = load double, double* %p
    %2 = fadd double %1, %1
    %3 = call <2 x float> @test_v2f32_f64_helper(double %2)
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, <2 x float>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v2f32_v1i64:
declare <2 x float> @test_v2f32_v1i64_helper(<1 x i64> %p)
define void @test_v2f32_v1i64(<1 x i64>* %p, <2 x float>* %q) {
; SOFT: vadd.i64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.i64 d0
    %1 = load <1 x i64>, <1 x i64>* %p
    %2 = add <1 x i64> %1, %1
    %3 = call <2 x float> @test_v2f32_v1i64_helper(<1 x i64> %2)
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, <2 x float>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v2f32_v2i32:
declare <2 x float> @test_v2f32_v2i32_helper(<2 x i32> %p)
define void @test_v2f32_v2i32(<2 x i32>* %p, <2 x float>* %q) {
; HARD: vrev64.32 d0
; SOFT: vadd.i32 [[REG:d[0-9]+]]
; SOFT: vrev64.32 [[REG]]
; SOFT: vmov r1, r0, [[REG]]
    %1 = load <2 x i32>, <2 x i32>* %p
    %2 = add <2 x i32> %1, %1
    %3 = call <2 x float> @test_v2f32_v2i32_helper(<2 x i32> %2)
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, <2 x float>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v2f32_v4i16:
declare <2 x float> @test_v2f32_v4i16_helper(<4 x i16> %p)
define void @test_v2f32_v4i16(<4 x i16>* %p, <2 x float>* %q) {
; SOFT: vrev64.16 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.16 d0
    %1 = load <4 x i16>, <4 x i16>* %p
    %2 = add <4 x i16> %1, %1
    %3 = call <2 x float> @test_v2f32_v4i16_helper(<4 x i16> %2)
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, <2 x float>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v2f32_v8i8:
declare <2 x float> @test_v2f32_v8i8_helper(<8 x i8> %p)
define void @test_v2f32_v8i8(<8 x i8>* %p, <2 x float>* %q) {
; SOFT: vrev64.8 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.8 d0
    %1 = load <8 x i8>, <8 x i8>* %p
    %2 = add <8 x i8> %1, %1
    %3 = call <2 x float> @test_v2f32_v8i8_helper(<8 x i8> %2)
    %4 = fadd <2 x float> %3, %3
    store <2 x float> %4, <2 x float>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v2i32_i64:
declare <2 x i32> @test_v2i32_i64_helper(i64 %p)
define void @test_v2i32_i64(i64* %p, <2 x i32>* %q) {
; CHECK: adds r1
; CHECK: adc r0
    %1 = load i64, i64* %p
    %2 = add i64 %1, %1
    %3 = call <2 x i32> @test_v2i32_i64_helper(i64 %2)
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, <2 x i32>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v2i32_f64:
declare <2 x i32> @test_v2i32_f64_helper(double %p)
define void @test_v2i32_f64(double* %p, <2 x i32>* %q) {
; SOFT: vadd.f64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.f64 d0
    %1 = load double, double* %p
    %2 = fadd double %1, %1
    %3 = call <2 x i32> @test_v2i32_f64_helper(double %2)
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, <2 x i32>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v2i32_v1i64:
declare <2 x i32> @test_v2i32_v1i64_helper(<1 x i64> %p)
define void @test_v2i32_v1i64(<1 x i64>* %p, <2 x i32>* %q) {
; SOFT: vadd.i64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.i64 d0
    %1 = load <1 x i64>, <1 x i64>* %p
    %2 = add <1 x i64> %1, %1
    %3 = call <2 x i32> @test_v2i32_v1i64_helper(<1 x i64> %2)
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, <2 x i32>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v2i32_v2f32:
declare <2 x i32> @test_v2i32_v2f32_helper(<2 x float> %p)
define void @test_v2i32_v2f32(<2 x float>* %p, <2 x i32>* %q) {
; HARD: vadd.f32 [[REG:d[0-9]+]]
; HARD: vrev64.32 d0, [[REG]]
; SOFT: vadd.f32 [[REG:d[0-9]+]]
; SOFT: vrev64.32 [[REG]]
; SOFT: vmov r1, r0, [[REG]]
    %1 = load <2 x float>, <2 x float>* %p
    %2 = fadd <2 x float> %1, %1
    %3 = call <2 x i32> @test_v2i32_v2f32_helper(<2 x float> %2)
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, <2 x i32>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v2i32_v4i16:
declare <2 x i32> @test_v2i32_v4i16_helper(<4 x i16> %p)
define void @test_v2i32_v4i16(<4 x i16>* %p, <2 x i32>* %q) {
; SOFT: vrev64.16 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.16 d0
    %1 = load <4 x i16>, <4 x i16>* %p
    %2 = add <4 x i16> %1, %1
    %3 = call <2 x i32> @test_v2i32_v4i16_helper(<4 x i16> %2)
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, <2 x i32>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v2i32_v8i8:
declare <2 x i32> @test_v2i32_v8i8_helper(<8 x i8> %p)
define void @test_v2i32_v8i8(<8 x i8>* %p, <2 x i32>* %q) {
; SOFT: vrev64.8 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.8 d0
    %1 = load <8 x i8>, <8 x i8>* %p
    %2 = add <8 x i8> %1, %1
    %3 = call <2 x i32> @test_v2i32_v8i8_helper(<8 x i8> %2)
    %4 = add <2 x i32> %3, %3
    store <2 x i32> %4, <2 x i32>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.32 [[REG]]
; HARD: vrev64.32 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v4i16_i64:
declare <4 x i16> @test_v4i16_i64_helper(i64 %p)
define void @test_v4i16_i64(i64* %p, <4 x i16>* %q) {
; CHECK: adds r1
; CHECK: adc r0
    %1 = load i64, i64* %p
    %2 = add i64 %1, %1
    %3 = call <4 x i16> @test_v4i16_i64_helper(i64 %2)
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, <4 x i16>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.16 [[REG]]
; HARD: vrev64.16 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v4i16_f64:
declare <4 x i16> @test_v4i16_f64_helper(double %p)
define void @test_v4i16_f64(double* %p, <4 x i16>* %q) {
; SOFT: vadd.f64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.f64 d0
    %1 = load double, double* %p
    %2 = fadd double %1, %1
    %3 = call <4 x i16> @test_v4i16_f64_helper(double %2)
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, <4 x i16>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.16 [[REG]]
; HARD: vrev64.16 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v4i16_v1i64:
declare <4 x i16> @test_v4i16_v1i64_helper(<1 x i64> %p)
define void @test_v4i16_v1i64(<1 x i64>* %p, <4 x i16>* %q) {
; SOFT: vadd.i64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.i64 d0
    %1 = load <1 x i64>, <1 x i64>* %p
    %2 = add <1 x i64> %1, %1
    %3 = call <4 x i16> @test_v4i16_v1i64_helper(<1 x i64> %2)
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, <4 x i16>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.16 [[REG]]
; HARD: vrev64.16 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v4i16_v2f32:
declare <4 x i16> @test_v4i16_v2f32_helper(<2 x float> %p)
define void @test_v4i16_v2f32(<2 x float>* %p, <4 x i16>* %q) {
; HARD: vadd.f32 [[REG:d[0-9]+]]
; HARD: vrev64.32 d0, [[REG]]
; SOFT: vadd.f32 [[REG:d[0-9]+]]
; SOFT: vrev64.32 [[REG]]
; SOFT: vmov r1, r0, [[REG]]
    %1 = load <2 x float>, <2 x float>* %p
    %2 = fadd <2 x float> %1, %1
    %3 = call <4 x i16> @test_v4i16_v2f32_helper(<2 x float> %2)
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, <4 x i16>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.16 [[REG]]
; HARD: vrev64.16 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v4i16_v2i32:
declare <4 x i16> @test_v4i16_v2i32_helper(<2 x i32> %p)
define void @test_v4i16_v2i32(<2 x i32>* %p, <4 x i16>* %q) {
; HARD: vadd.i32 [[REG:d[0-9]+]]
; HARD: vrev64.32 d0, [[REG]]
; SOFT: vadd.i32 [[REG:d[0-9]+]]
; SOFT: vrev64.32 [[REG]]
; SOFT: vmov r1, r0, [[REG]]
    %1 = load <2 x i32>, <2 x i32>* %p
    %2 = add <2 x i32> %1, %1
    %3 = call <4 x i16> @test_v4i16_v2i32_helper(<2 x i32> %2)
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, <4 x i16>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.16 [[REG]]
; HARD: vrev64.16 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v4i16_v8i8:
declare <4 x i16> @test_v4i16_v8i8_helper(<8 x i8> %p)
define void @test_v4i16_v8i8(<8 x i8>* %p, <4 x i16>* %q) {
; SOFT: vrev64.8 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.8 d0
    %1 = load <8 x i8>, <8 x i8>* %p
    %2 = add <8 x i8> %1, %1
    %3 = call <4 x i16> @test_v4i16_v8i8_helper(<8 x i8> %2)
    %4 = add <4 x i16> %3, %3
    store <4 x i16> %4, <4 x i16>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.16 [[REG]]
; HARD: vrev64.16 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v8i8_i64:
declare <8 x i8> @test_v8i8_i64_helper(i64 %p)
define void @test_v8i8_i64(i64* %p, <8 x i8>* %q) {
; CHECK: adds r1
; CHECK: adc r0
    %1 = load i64, i64* %p
    %2 = add i64 %1, %1
    %3 = call <8 x i8> @test_v8i8_i64_helper(i64 %2)
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, <8 x i8>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.8 [[REG]]
; HARD: vrev64.8 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v8i8_f64:
declare <8 x i8> @test_v8i8_f64_helper(double %p)
define void @test_v8i8_f64(double* %p, <8 x i8>* %q) {
; SOFT: vadd.f64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.f64 d0
    %1 = load double, double* %p
    %2 = fadd double %1, %1
    %3 = call <8 x i8> @test_v8i8_f64_helper(double %2)
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, <8 x i8>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.8 [[REG]]
; HARD: vrev64.8 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v8i8_v1i64:
declare <8 x i8> @test_v8i8_v1i64_helper(<1 x i64> %p)
define void @test_v8i8_v1i64(<1 x i64>* %p, <8 x i8>* %q) {
; SOFT: vadd.i64 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vadd.i64 d0
    %1 = load <1 x i64>, <1 x i64>* %p
    %2 = add <1 x i64> %1, %1
    %3 = call <8 x i8> @test_v8i8_v1i64_helper(<1 x i64> %2)
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, <8 x i8>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.8 [[REG]]
; HARD: vrev64.8 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v8i8_v2f32:
declare <8 x i8> @test_v8i8_v2f32_helper(<2 x float> %p)
define void @test_v8i8_v2f32(<2 x float>* %p, <8 x i8>* %q) {
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
    %1 = load <2 x float>, <2 x float>* %p
    %2 = fadd <2 x float> %1, %1
    %3 = call <8 x i8> @test_v8i8_v2f32_helper(<2 x float> %2)
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, <8 x i8>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.8 [[REG]]
; HARD: vrev64.8 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v8i8_v2i32:
declare <8 x i8> @test_v8i8_v2i32_helper(<2 x i32> %p)
define void @test_v8i8_v2i32(<2 x i32>* %p, <8 x i8>* %q) {
; SOFT: vrev64.32 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.32 d0
    %1 = load <2 x i32>, <2 x i32>* %p
    %2 = add <2 x i32> %1, %1
    %3 = call <8 x i8> @test_v8i8_v2i32_helper(<2 x i32> %2)
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, <8 x i8>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.8 [[REG]]
; HARD: vrev64.8 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_v8i8_v4i16:
declare <8 x i8> @test_v8i8_v4i16_helper(<4 x i16> %p)
define void @test_v8i8_v4i16(<4 x i16>* %p, <8 x i8>* %q) {
; SOFT: vrev64.16 [[REG:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG]]
; HARD: vrev64.16 d0
    %1 = load <4 x i16>, <4 x i16>* %p
    %2 = add <4 x i16> %1, %1
    %3 = call <8 x i8> @test_v8i8_v4i16_helper(<4 x i16> %2)
    %4 = add <8 x i8> %3, %3
    store <8 x i8> %4, <8 x i8>* %q
    ret void
; SOFT: vmov [[REG:d[0-9]+]], r1, r0
; SOFT: vrev64.8 [[REG]]
; HARD: vrev64.8 {{d[0-9]+}}, d0
}

; CHECK-LABEL: test_f128_v2f64:
declare fp128 @test_f128_v2f64_helper(<2 x double> %p)
define void @test_f128_v2f64(<2 x double>* %p, fp128* %q) {
; SOFT: vadd.f64 [[REG2:d[0-9]+]]
; SOFT: vadd.f64 [[REG1:d[0-9]+]]
; SOFT: vmov r1, r0, [[REG1]]
; SOFT: vmov r3, r2, [[REG2]]
; HARD: vadd.f64 d1
; HARD: vadd.f64 d0
    %1 = load <2 x double>, <2 x double>* %p
    %2 = fadd <2 x double> %1, %1
    %3 = call fp128 @test_f128_v2f64_helper(<2 x double> %2)
    %4 = fadd fp128 %3, %3
    store fp128 %4, fp128* %q
    ret void
; CHECK: stm sp, {r0, r1, r2, r3}
}

; CHECK-LABEL: test_f128_v2i64:
declare fp128 @test_f128_v2i64_helper(<2 x i64> %p)
define void @test_f128_v2i64(<2 x i64>* %p, fp128* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vadd.i64 q0
    %1 = load <2 x i64>, <2 x i64>* %p
    %2 = add <2 x i64> %1, %1
    %3 = call fp128 @test_f128_v2i64_helper(<2 x i64> %2)
    %4 = fadd fp128 %3, %3
    store fp128 %4, fp128* %q
    ret void
; CHECK: stm sp, {r0, r1, r2, r3}
}

; CHECK-LABEL: test_f128_v4f32:
declare fp128 @test_f128_v4f32_helper(<4 x float> %p)
define void @test_f128_v4f32(<4 x float>* %p, fp128* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
    %1 = load <4 x float>, <4 x float>* %p
    %2 = fadd <4 x float> %1, %1
    %3 = call fp128 @test_f128_v4f32_helper(<4 x float> %2)
    %4 = fadd fp128 %3, %3
    store fp128 %4, fp128* %q
    ret void
; CHECK: stm sp, {r0, r1, r2, r3}
}

; CHECK-LABEL: test_f128_v4i32:
declare fp128 @test_f128_v4i32_helper(<4 x i32> %p)
define void @test_f128_v4i32(<4 x i32>* %p, fp128* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
    %1 = load <4 x i32>, <4 x i32>* %p
    %2 = add <4 x i32> %1, %1
    %3 = call fp128 @test_f128_v4i32_helper(<4 x i32> %2)
    %4 = fadd fp128 %3, %3
    store fp128 %4, fp128* %q
    ret void
; CHECK: stm sp, {r0, r1, r2, r3}
}

; CHECK-LABEL: test_f128_v8i16:
declare fp128 @test_f128_v8i16_helper(<8 x i16> %p)
define void @test_f128_v8i16(<8 x i16>* %p, fp128* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.16 q0
    %1 = load <8 x i16>, <8 x i16>* %p
    %2 = add <8 x i16> %1, %1
    %3 = call fp128 @test_f128_v8i16_helper(<8 x i16> %2)
    %4 = fadd fp128 %3, %3
    store fp128 %4, fp128* %q
    ret void
; CHECK: stm sp, {r0, r1, r2, r3}
}

; CHECK-LABEL: test_f128_v16i8:
declare fp128 @test_f128_v16i8_helper(<16 x i8> %p)
define void @test_f128_v16i8(<16 x i8>* %p, fp128* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.8 q0
    %1 = load <16 x i8>, <16 x i8>* %p
    %2 = add <16 x i8> %1, %1
    %3 = call fp128 @test_f128_v16i8_helper(<16 x i8> %2)
    %4 = fadd fp128 %3, %3
    store fp128 %4, fp128* %q
    ret void
; CHECK: stm sp, {r0, r1, r2, r3}
}

; CHECK-LABEL: test_v2f64_f128:
declare <2 x double> @test_v2f64_f128_helper(fp128 %p)
define void @test_v2f64_f128(fp128* %p, <2 x double>* %q) {
    %1 = load fp128, fp128* %p
    %2 = fadd fp128 %1, %1
    %3 = call <2 x double> @test_v2f64_f128_helper(fp128 %2)
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, <2 x double>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0

}

; CHECK-LABEL: test_v2f64_v2i64:
declare <2 x double> @test_v2f64_v2i64_helper(<2 x i64> %p)
define void @test_v2f64_v2i64(<2 x i64>* %p, <2 x double>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vadd.i64 q0
    %1 = load <2 x i64>, <2 x i64>* %p
    %2 = add <2 x i64> %1, %1
    %3 = call <2 x double> @test_v2f64_v2i64_helper(<2 x i64> %2)
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, <2 x double>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v2f64_v4f32:
declare <2 x double> @test_v2f64_v4f32_helper(<4 x float> %p)
define void @test_v2f64_v4f32(<4 x float>* %p, <2 x double>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
    %1 = load <4 x float>, <4 x float>* %p
    %2 = fadd <4 x float> %1, %1
    %3 = call <2 x double> @test_v2f64_v4f32_helper(<4 x float> %2)
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, <2 x double>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v2f64_v4i32:
declare <2 x double> @test_v2f64_v4i32_helper(<4 x i32> %p)
define void @test_v2f64_v4i32(<4 x i32>* %p, <2 x double>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
    %1 = load <4 x i32>, <4 x i32>* %p
    %2 = add <4 x i32> %1, %1
    %3 = call <2 x double> @test_v2f64_v4i32_helper(<4 x i32> %2)
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, <2 x double>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v2f64_v8i16:
declare <2 x double> @test_v2f64_v8i16_helper(<8 x i16> %p)
define void @test_v2f64_v8i16(<8 x i16>* %p, <2 x double>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.16 q0
    %1 = load <8 x i16>, <8 x i16>* %p
    %2 = add <8 x i16> %1, %1
    %3 = call <2 x double> @test_v2f64_v8i16_helper(<8 x i16> %2)
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, <2 x double>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v2f64_v16i8:
declare <2 x double> @test_v2f64_v16i8_helper(<16 x i8> %p)
define void @test_v2f64_v16i8(<16 x i8>* %p, <2 x double>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.8 q0
    %1 = load <16 x i8>, <16 x i8>* %p
    %2 = add <16 x i8> %1, %1
    %3 = call <2 x double> @test_v2f64_v16i8_helper(<16 x i8> %2)
    %4 = fadd <2 x double> %3, %3
    store <2 x double> %4, <2 x double>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v2i64_f128:
declare <2 x i64> @test_v2i64_f128_helper(fp128 %p)
define void @test_v2i64_f128(fp128* %p, <2 x i64>* %q) {
    %1 = load fp128, fp128* %p
    %2 = fadd fp128 %1, %1
    %3 = call <2 x i64> @test_v2i64_f128_helper(fp128 %2)
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, <2 x i64>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v2i64_v2f64:
declare <2 x i64> @test_v2i64_v2f64_helper(<2 x double> %p)
define void @test_v2i64_v2f64(<2 x double>* %p, <2 x i64>* %q) {
; SOFT: vmov r1, r0, [[REG1]]
; SOFT: vmov r3, r2, [[REG2]]
; HARD: vadd.f64 d1
; HARD: vadd.f64 d0
    %1 = load <2 x double>, <2 x double>* %p
    %2 = fadd <2 x double> %1, %1
    %3 = call <2 x i64> @test_v2i64_v2f64_helper(<2 x double> %2)
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, <2 x i64>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v2i64_v4f32:
declare <2 x i64> @test_v2i64_v4f32_helper(<4 x float> %p) 
define void @test_v2i64_v4f32(<4 x float>* %p, <2 x i64>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
    %1 = load <4 x float>, <4 x float>* %p
    %2 = fadd <4 x float> %1, %1
    %3 = call <2 x i64> @test_v2i64_v4f32_helper(<4 x float> %2)
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, <2 x i64>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v2i64_v4i32:
declare <2 x i64> @test_v2i64_v4i32_helper(<4 x i32> %p)
define void @test_v2i64_v4i32(<4 x i32>* %p, <2 x i64>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
    %1 = load <4 x i32>, <4 x i32>* %p
    %2 = add <4 x i32> %1, %1
    %3 = call <2 x i64> @test_v2i64_v4i32_helper(<4 x i32> %2)
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, <2 x i64>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v2i64_v8i16:
declare <2 x i64> @test_v2i64_v8i16_helper(<8 x i16> %p)
define void @test_v2i64_v8i16(<8 x i16>* %p, <2 x i64>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.16 q0
    %1 = load <8 x i16>, <8 x i16>* %p
    %2 = add <8 x i16> %1, %1
    %3 = call <2 x i64> @test_v2i64_v8i16_helper(<8 x i16> %2)
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, <2 x i64>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v2i64_v16i8:
declare <2 x i64> @test_v2i64_v16i8_helper(<16 x i8> %p)
define void @test_v2i64_v16i8(<16 x i8>* %p, <2 x i64>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.8 q0
    %1 = load <16 x i8>, <16 x i8>* %p
    %2 = add <16 x i8> %1, %1
    %3 = call <2 x i64> @test_v2i64_v16i8_helper(<16 x i8> %2)
    %4 = add <2 x i64> %3, %3
    store <2 x i64> %4, <2 x i64>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v4f32_f128:
declare <4 x float> @test_v4f32_f128_helper(fp128 %p)
define void @test_v4f32_f128(fp128* %p, <4 x float>* %q) {
    %1 = load fp128, fp128* %p
    %2 = fadd fp128 %1, %1
    %3 = call <4 x float> @test_v4f32_f128_helper(fp128 %2)
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, <4 x float>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v4f32_v2f64:
declare <4 x float> @test_v4f32_v2f64_helper(<2 x double> %p)
define void @test_v4f32_v2f64(<2 x double>* %p, <4 x float>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vadd.f64  d1
; HARD: vadd.f64  d0
    %1 = load <2 x double>, <2 x double>* %p
    %2 = fadd <2 x double> %1, %1
    %3 = call <4 x float> @test_v4f32_v2f64_helper(<2 x double> %2)
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, <4 x float>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v4f32_v2i64:
declare <4 x float> @test_v4f32_v2i64_helper(<2 x i64> %p)
define void @test_v4f32_v2i64(<2 x i64>* %p, <4 x float>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vadd.i64 q0
    %1 = load <2 x i64>, <2 x i64>* %p
    %2 = add <2 x i64> %1, %1
    %3 = call <4 x float> @test_v4f32_v2i64_helper(<2 x i64> %2)
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, <4 x float>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v4f32_v4i32:
declare <4 x float> @test_v4f32_v4i32_helper(<4 x i32> %p)
define void @test_v4f32_v4i32(<4 x i32>* %p, <4 x float>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
    %1 = load <4 x i32>, <4 x i32>* %p
    %2 = add <4 x i32> %1, %1
    %3 = call <4 x float> @test_v4f32_v4i32_helper(<4 x i32> %2)
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, <4 x float>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v4f32_v8i16:
declare <4 x float> @test_v4f32_v8i16_helper(<8 x i16> %p)
define void @test_v4f32_v8i16(<8 x i16>* %p, <4 x float>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.16 q0
    %1 = load <8 x i16>, <8 x i16>* %p
    %2 = add <8 x i16> %1, %1
    %3 = call <4 x float> @test_v4f32_v8i16_helper(<8 x i16> %2)
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, <4 x float>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v4f32_v16i8:
declare <4 x float> @test_v4f32_v16i8_helper(<16 x i8> %p)
define void @test_v4f32_v16i8(<16 x i8>* %p, <4 x float>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.8 q0
    %1 = load <16 x i8>, <16 x i8>* %p
    %2 = add <16 x i8> %1, %1
    %3 = call <4 x float> @test_v4f32_v16i8_helper(<16 x i8> %2)
    %4 = fadd <4 x float> %3, %3
    store <4 x float> %4, <4 x float>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v4i32_f128:
declare <4 x i32> @test_v4i32_f128_helper(fp128 %p)
define void @test_v4i32_f128(fp128* %p, <4 x i32>* %q) {
    %1 = load fp128, fp128* %p
    %2 = fadd fp128 %1, %1
    %3 = call <4 x i32> @test_v4i32_f128_helper(fp128 %2)
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, <4 x i32>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v4i32_v2f64:
declare <4 x i32> @test_v4i32_v2f64_helper(<2 x double> %p)
define void @test_v4i32_v2f64(<2 x double>* %p, <4 x i32>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vadd.f64 d1
; HARD: vadd.f64 d0
    %1 = load <2 x double>, <2 x double>* %p
    %2 = fadd <2 x double> %1, %1
    %3 = call <4 x i32> @test_v4i32_v2f64_helper(<2 x double> %2)
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, <4 x i32>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v4i32_v2i64:
declare <4 x i32> @test_v4i32_v2i64_helper(<2 x i64> %p)
define void @test_v4i32_v2i64(<2 x i64>* %p, <4 x i32>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vadd.i64 q0
    %1 = load <2 x i64>, <2 x i64>* %p
    %2 = add <2 x i64> %1, %1
    %3 = call <4 x i32> @test_v4i32_v2i64_helper(<2 x i64> %2)
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, <4 x i32>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v4i32_v4f32:
declare <4 x i32> @test_v4i32_v4f32_helper(<4 x float> %p)
define void @test_v4i32_v4f32(<4 x float>* %p, <4 x i32>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
    %1 = load <4 x float>, <4 x float>* %p
    %2 = fadd <4 x float> %1, %1
    %3 = call <4 x i32> @test_v4i32_v4f32_helper(<4 x float> %2)
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, <4 x i32>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v4i32_v8i16:
declare <4 x i32> @test_v4i32_v8i16_helper(<8 x i16> %p)
define void @test_v4i32_v8i16(<8 x i16>* %p, <4 x i32>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.16 q0
    %1 = load <8 x i16>, <8 x i16>* %p
    %2 = add <8 x i16> %1, %1
    %3 = call <4 x i32> @test_v4i32_v8i16_helper(<8 x i16> %2)
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, <4 x i32>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v4i32_v16i8:
declare <4 x i32> @test_v4i32_v16i8_helper(<16 x i8> %p)
define void @test_v4i32_v16i8(<16 x i8>* %p, <4 x i32>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.8 q0
    %1 = load <16 x i8>, <16 x i8>* %p
    %2 = add <16 x i8> %1, %1
    %3 = call <4 x i32> @test_v4i32_v16i8_helper(<16 x i8> %2)
    %4 = add <4 x i32> %3, %3
    store <4 x i32> %4, <4 x i32>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v8i16_f128:
declare <8 x i16> @test_v8i16_f128_helper(fp128 %p)
define void @test_v8i16_f128(fp128* %p, <8 x i16>* %q) {
    %1 = load fp128, fp128* %p
    %2 = fadd fp128 %1, %1
    %3 = call <8 x i16> @test_v8i16_f128_helper(fp128 %2)
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, <8 x i16>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v8i16_v2f64:
declare <8 x i16> @test_v8i16_v2f64_helper(<2 x double> %p)
define void @test_v8i16_v2f64(<2 x double>* %p, <8 x i16>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vadd.f64 d1
; HARD: vadd.f64 d0
    %1 = load <2 x double>, <2 x double>* %p
    %2 = fadd <2 x double> %1, %1
    %3 = call <8 x i16> @test_v8i16_v2f64_helper(<2 x double> %2)
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, <8 x i16>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v8i16_v2i64:
declare <8 x i16> @test_v8i16_v2i64_helper(<2 x i64> %p)
define void @test_v8i16_v2i64(<2 x i64>* %p, <8 x i16>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vadd.i64 q0
    %1 = load <2 x i64>, <2 x i64>* %p
    %2 = add <2 x i64> %1, %1
    %3 = call <8 x i16> @test_v8i16_v2i64_helper(<2 x i64> %2)
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, <8 x i16>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v8i16_v4f32:
declare <8 x i16> @test_v8i16_v4f32_helper(<4 x float> %p)
define void @test_v8i16_v4f32(<4 x float>* %p, <8 x i16>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
    %1 = load <4 x float>, <4 x float>* %p
    %2 = fadd <4 x float> %1, %1
    %3 = call <8 x i16> @test_v8i16_v4f32_helper(<4 x float> %2)
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, <8 x i16>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v8i16_v4i32:
declare <8 x i16> @test_v8i16_v4i32_helper(<4 x i32> %p)
define void @test_v8i16_v4i32(<4 x i32>* %p, <8 x i16>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
    %1 = load <4 x i32>, <4 x i32>* %p
    %2 = add <4 x i32> %1, %1
    %3 = call <8 x i16> @test_v8i16_v4i32_helper(<4 x i32> %2)
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, <8 x i16>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v8i16_v16i8:
declare <8 x i16> @test_v8i16_v16i8_helper(<16 x i8> %p)
define void @test_v8i16_v16i8(<16 x i8>* %p, <8 x i16>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.8 q0
    %1 = load <16 x i8>, <16 x i8>* %p
    %2 = add <16 x i8> %1, %1
    %3 = call <8 x i16> @test_v8i16_v16i8_helper(<16 x i8> %2)
    %4 = add <8 x i16> %3, %3
    store <8 x i16> %4, <8 x i16>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v16i8_f128:
declare <16 x i8> @test_v16i8_f128_helper(fp128 %p)
define void @test_v16i8_f128(fp128* %p, <16 x i8>* %q) {
    %1 = load fp128, fp128* %p
    %2 = fadd fp128 %1, %1
    %3 = call <16 x i8> @test_v16i8_f128_helper(fp128 %2)
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, <16 x i8>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v16i8_v2f64:
declare <16 x i8> @test_v16i8_v2f64_helper(<2 x double> %p)
define void @test_v16i8_v2f64(<2 x double>* %p, <16 x i8>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vadd.f64 d1
; HARD: vadd.f64 d0
    %1 = load <2 x double>, <2 x double>* %p
    %2 = fadd <2 x double> %1, %1
    %3 = call <16 x i8> @test_v16i8_v2f64_helper(<2 x double> %2)
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, <16 x i8>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v16i8_v2i64:
declare <16 x i8> @test_v16i8_v2i64_helper(<2 x i64> %p)
define void @test_v16i8_v2i64(<2 x i64>* %p, <16 x i8>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vadd.i64 q0
    %1 = load <2 x i64>, <2 x i64>* %p
    %2 = add <2 x i64> %1, %1
    %3 = call <16 x i8> @test_v16i8_v2i64_helper(<2 x i64> %2)
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, <16 x i8>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v16i8_v4f32:
declare <16 x i8> @test_v16i8_v4f32_helper(<4 x float> %p)
define void @test_v16i8_v4f32(<4 x float>* %p, <16 x i8>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
    %1 = load <4 x float>, <4 x float>* %p
    %2 = fadd <4 x float> %1, %1
    %3 = call <16 x i8> @test_v16i8_v4f32_helper(<4 x float> %2)
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, <16 x i8>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v16i8_v4i32:
declare <16 x i8> @test_v16i8_v4i32_helper(<4 x i32> %p)
define void @test_v16i8_v4i32(<4 x i32>* %p, <16 x i8>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.32 q0
    %1 = load <4 x i32>, <4 x i32>* %p
    %2 = add <4 x i32> %1, %1
    %3 = call <16 x i8> @test_v16i8_v4i32_helper(<4 x i32> %2)
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, <16 x i8>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}

; CHECK-LABEL: test_v16i8_v8i16:
declare <16 x i8> @test_v16i8_v8i16_helper(<8 x i16> %p)
define void @test_v16i8_v8i16(<8 x i16>* %p, <16 x i8>* %q) {
; SOFT: vmov r1, r0
; SOFT: vmov r3, r2
; HARD: vrev64.16 q0
    %1 = load <8 x i16>, <8 x i16>* %p
    %2 = add <8 x i16> %1, %1
    %3 = call <16 x i8> @test_v16i8_v8i16_helper(<8 x i16> %2)
    %4 = add <16 x i8> %3, %3
    store <16 x i8> %4, <16 x i8>* %q
    ret void
; SOFT: vmov {{d[0-9]+}}, r3, r2
; SOFT: vmov {{d[0-9]+}}, r1, r0
}
