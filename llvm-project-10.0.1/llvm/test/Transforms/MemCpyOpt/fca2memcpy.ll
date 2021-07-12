; RUN: opt -memcpyopt -S < %s | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%S = type { i8*, i8, i32 }

define void @copy(%S* %src, %S* %dst) {
; CHECK-LABEL: copy
; CHECK-NOT: load
; CHECK: call void @llvm.memmove.p0i8.p0i8.i64
; CHECK-NEXT: ret void
  %1 = load %S, %S* %src
  store %S %1, %S* %dst
  ret void
}

define void @noaliassrc(%S* noalias %src, %S* %dst) {
; CHECK-LABEL: noaliassrc
; CHECK-NOT: load
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
; CHECK-NEXT: ret void
  %1 = load %S, %S* %src
  store %S %1, %S* %dst
  ret void
}

define void @noaliasdst(%S* %src, %S* noalias %dst) {
; CHECK-LABEL: noaliasdst
; CHECK-NOT: load
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
; CHECK-NEXT: ret void
  %1 = load %S, %S* %src
  store %S %1, %S* %dst
  ret void
}

define void @destroysrc(%S* %src, %S* %dst) {
; CHECK-LABEL: destroysrc
; CHECK: load %S, %S* %src
; CHECK: call void @llvm.memset.p0i8.i64
; CHECK-NEXT: store %S %1, %S* %dst
; CHECK-NEXT: ret void
  %1 = load %S, %S* %src
  store %S zeroinitializer, %S* %src
  store %S %1, %S* %dst
  ret void
}

define void @destroynoaliassrc(%S* noalias %src, %S* %dst) {
; CHECK-LABEL: destroynoaliassrc
; CHECK-NOT: load
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
; CHECK-NEXT: call void @llvm.memset.p0i8.i64
; CHECK-NEXT: ret void
  %1 = load %S, %S* %src
  store %S zeroinitializer, %S* %src
  store %S %1, %S* %dst
  ret void
}

define void @copyalias(%S* %src, %S* %dst) {
; CHECK-LABEL: copyalias
; CHECK-NEXT: [[LOAD:%[a-z0-9\.]+]] = load %S, %S* %src
; CHECK-NOT: load
; CHECK: call void @llvm.memmove.p0i8.p0i8.i64
; CHECK-NEXT: store %S [[LOAD]], %S* %dst
; CHECK-NEXT: ret void
  %1 = load %S, %S* %src
  %2 = load %S, %S* %src
  store %S %1, %S* %dst
  store %S %2, %S* %dst
  ret void
}

; If the store address is computed in a complex manner, make
; sure we lift the computation as well if needed and possible.
define void @addrproducer(%S* %src, %S* %dst) {
; CHECK-LABEL: addrproducer(
; CHECK-NEXT: %[[DSTCAST:[0-9]+]] = bitcast %S* %dst to i8*
; CHECK-NEXT: %dst2 = getelementptr %S, %S* %dst, i64 1
; CHECK-NEXT: %[[DST2CAST:[0-9]+]] = bitcast %S* %dst2 to i8*
; CHECK-NEXT: %[[SRCCAST:[0-9]+]] = bitcast %S* %src to i8*
; CHECK-NEXT: call void @llvm.memmove.p0i8.p0i8.i64(i8* align 8 %[[DST2CAST]], i8* align 8 %[[SRCCAST]], i64 16, i1 false)
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* align 8 %[[DSTCAST]], i8 undef, i64 16, i1 false)
; CHECK-NEXT: ret void
  %1 = load %S, %S* %src
  store %S undef, %S* %dst
  %dst2 = getelementptr %S , %S* %dst, i64 1
  store %S %1, %S* %dst2
  ret void
}

define void @aliasaddrproducer(%S* %src, %S* %dst, i32* %dstidptr) {
; CHECK-LABEL: aliasaddrproducer(
; CHECK-NEXT: %[[SRC:[0-9]+]] = load %S, %S* %src
; CHECK-NEXT: %[[DSTCAST:[0-9]+]] = bitcast %S* %dst to i8*
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* align 8 %[[DSTCAST]], i8 undef, i64 16, i1 false)
; CHECK-NEXT: %dstindex = load i32, i32* %dstidptr
; CHECK-NEXT: %dst2 = getelementptr %S, %S* %dst, i32 %dstindex
; CHECK-NEXT: store %S %[[SRC]], %S* %dst2
; CHECK-NEXT: ret void
  %1 = load %S, %S* %src
  store %S undef, %S* %dst
  %dstindex = load i32, i32* %dstidptr
  %dst2 = getelementptr %S , %S* %dst, i32 %dstindex
  store %S %1, %S* %dst2
  ret void
}

define void @noaliasaddrproducer(%S* %src, %S* noalias %dst, i32* noalias %dstidptr) {
; CHECK-LABEL: noaliasaddrproducer(
; CHECK-NEXT: %[[SRCCAST:[0-9]+]] = bitcast %S* %src to i8*
; CHECK-NEXT: %[[LOADED:[0-9]+]] = load i32, i32* %dstidptr
; CHECK-NEXT: %dstindex = or i32 %[[LOADED]], 1
; CHECK-NEXT: %dst2 = getelementptr %S, %S* %dst, i32 %dstindex
; CHECK-NEXT: %[[DST2CAST:[0-9]+]] = bitcast %S* %dst2 to i8*
; CHECK-NEXT: %[[SRCCAST2:[0-9]+]] = bitcast %S* %src to i8*
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %[[DST2CAST]], i8* align 8 %[[SRCCAST2]], i64 16, i1 false)
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* align 8 %[[SRCCAST]], i8 undef, i64 16, i1 false)
; CHECK-NEXT: ret void
  %1 = load %S, %S* %src
  store %S undef, %S* %src
  %2 = load i32, i32* %dstidptr
  %dstindex = or i32 %2, 1
  %dst2 = getelementptr %S , %S* %dst, i32 %dstindex
  store %S %1, %S* %dst2
  ret void
}
