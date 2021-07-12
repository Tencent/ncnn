; RUN: opt %loadPolly -analyze -polly-scops < %s | FileCheck %s
; RUN: opt %loadPolly -S -polly-codegen < %s | FileCheck %s --check-prefix=IR
;
;    void f(long *A, long *ptr, long val) {
;      for (long i = 0; i < 100; i++) {
;        long ptrV = ((long)(ptr + 1)) + 1;
;        long valP = (long)(((long *)(val + 1)) + 1);
;        A[ptrV] += A[valP];
;      }
;    }
;
; CHECK:        ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:       [val, ptr] -> { Stmt_for_body[i0] -> MemRef_A[9 + val] };
; CHECK-NEXT:   ReadAccess :=	[Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:       [val, ptr] -> { Stmt_for_body[i0] -> MemRef_A[9 + ptr] };
; CHECK-NEXT:   MustWriteAccess :=	[Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:       [val, ptr] -> { Stmt_for_body[i0] -> MemRef_A[9 + ptr] };
;
; IR:      polly.stmt.for.body:
; IR-NEXT:   %p_tmp = ptrtoint i64* %scevgep to i64
; IR-NEXT:   %p_add = add nsw i64 %p_tmp, 1
; IR-NEXT:   %p_tmp1 = inttoptr i64 %[[r1:[a-zA-Z0-9]*]] to i64*
; IR-NEXT:   %p_add.ptr2 = getelementptr inbounds i64, i64* %p_tmp1, i64 1
; IR-NEXT:   %p_tmp2 = ptrtoint i64* %p_add.ptr2 to i64
; IR-NEXT:   %p_arrayidx = getelementptr inbounds i64, i64* %A, i64 %p_tmp2
; IR-NEXT:   %tmp3_p_scalar_ = load i64, i64* %p_arrayidx
; IR-NEXT:   %p_arrayidx3 = getelementptr inbounds i64, i64* %A, i64 %p_add
; IR-NEXT:   %tmp4_p_scalar_ = load i64, i64* %p_arrayidx3
; IR-NEXT:   %p_add4 = add nsw i64 %tmp4_p_scalar_, %tmp3_p_scalar_
; IR-NEXT:   store i64 %p_add4, i64* %p_arrayidx3
;
; IR:      polly.loop_preheader:
; IR-NEXT:   %scevgep = getelementptr i64, i64* %ptr, i32 1
; IR-NEXT:   %[[r1]] = add i64 %val, 1
; IR-NEXT:   br label %polly.loop_header
;
target datalayout = "e-p:32:32:32-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i64* %A, i64* %ptr, i64 %val) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %add.ptr = getelementptr inbounds i64, i64* %ptr, i64 1
  %tmp = ptrtoint i64* %add.ptr to i64
  %add = add nsw i64 %tmp, 1
  %add1 = add nsw i64 %val, 1
  %tmp1 = inttoptr i64 %add1 to i64*
  %add.ptr2 = getelementptr inbounds i64, i64* %tmp1, i64 1
  %tmp2 = ptrtoint i64* %add.ptr2 to i64
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %tmp2
  %tmp3 = load i64, i64* %arrayidx
  %arrayidx3 = getelementptr inbounds i64, i64* %A, i64 %add
  %tmp4 = load i64, i64* %arrayidx3
  %add4 = add nsw i64 %tmp4, %tmp3
  store i64 %add4, i64* %arrayidx3
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
