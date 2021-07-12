; RUN: opt -loop-idiom -mtriple=x86_64 -mcpu=core-avx2 < %s -S | FileCheck -check-prefix=LZCNT --check-prefix=ALL %s
; RUN: opt -loop-idiom -mtriple=x86_64 -mcpu=corei7 < %s -S | FileCheck -check-prefix=NOLZCNT --check-prefix=ALL %s

; Recognize CTLZ builtin pattern.
; Here we'll just convert loop to countable,
; so do not insert builtin if CPU do not support CTLZ
;
; int ctlz_and_other(int n, char *a)
; {
;   n = n >= 0 ? n : -n;
;   int i = 0, n0 = n;
;   while(n >>= 1) {
;     a[i] = (n0 & (1 << i)) ? 1 : 0;
;     i++;
;   }
;   return i;
; }
;
; LZCNT:  entry
; LZCNT:  %0 = call i32 @llvm.ctlz.i32(i32 %shr8, i1 true)
; LZCNT-NEXT:  %1 = sub i32 32, %0
; LZCNT-NEXT:  %2 = zext i32 %1 to i64
; LZCNT:  %indvars.iv.next.lcssa = phi i64 [ %2, %while.body ]
; LZCNT:  %4 = trunc i64 %indvars.iv.next.lcssa to i32
; LZCNT:  %i.0.lcssa = phi i32 [ 0, %entry ], [ %4, %while.end.loopexit ]
; LZCNT:  ret i32 %i.0.lcssa

; NOLZCNT:  entry
; NOLZCNT-NOT:  @llvm.ctlz

; Function Attrs: norecurse nounwind uwtable
define i32 @ctlz_and_other(i32 %n, i8* nocapture %a) {
entry:
  %c = icmp sgt i32 %n, 0
  %negn = sub nsw i32 0, %n
  %abs_n = select i1 %c, i32 %n, i32 %negn
  %shr8 = lshr i32 %abs_n, 1
  %tobool9 = icmp eq i32 %shr8, 0
  br i1 %tobool9, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %while.body ], [ 0, %while.body.preheader ]
  %shr11 = phi i32 [ %shr, %while.body ], [ %shr8, %while.body.preheader ]
  %0 = trunc i64 %indvars.iv to i32
  %shl = shl i32 1, %0
  %and = and i32 %shl, %abs_n
  %tobool1 = icmp ne i32 %and, 0
  %conv = zext i1 %tobool1 to i8
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %indvars.iv
  store i8 %conv, i8* %arrayidx, align 1
  %indvars.iv.next = add nuw i64 %indvars.iv, 1
  %shr = ashr i32 %shr11, 1
  %tobool = icmp eq i32 %shr, 0
  br i1 %tobool, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %while.body
  %1 = trunc i64 %indvars.iv.next to i32
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %i.0.lcssa = phi i32 [ 0, %entry ], [ %1, %while.end.loopexit ]
  ret i32 %i.0.lcssa
}

; Recognize CTLZ builtin pattern.
; Here it will replace the loop -
; assume builtin is always profitable.
;
; int ctlz_zero_check(int n)
; {
;   n = n >= 0 ? n : -n;
;   int i = 0;
;   while(n) {
;     n >>= 1;
;     i++;
;   }
;   return i;
; }
;
; ALL:  entry
; ALL:  %0 = call i32 @llvm.ctlz.i32(i32 %abs_n, i1 true)
; ALL-NEXT:  %1 = sub i32 32, %0
; ALL:  %inc.lcssa = phi i32 [ %1, %while.body ]
; ALL:  %i.0.lcssa = phi i32 [ 0, %entry ], [ %inc.lcssa, %while.end.loopexit ]
; ALL:  ret i32 %i.0.lcssa

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @ctlz_zero_check(i32 %n) {
entry:
  %c = icmp sgt i32 %n, 0
  %negn = sub nsw i32 0, %n
  %abs_n = select i1 %c, i32 %n, i32 %negn
  %tobool4 = icmp eq i32 %abs_n, 0
  br i1 %tobool4, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %i.06 = phi i32 [ %inc, %while.body ], [ 0, %while.body.preheader ]
  %n.addr.05 = phi i32 [ %shr, %while.body ], [ %abs_n, %while.body.preheader ]
  %shr = ashr i32 %n.addr.05, 1
  %inc = add nsw i32 %i.06, 1
  %tobool = icmp eq i32 %shr, 0
  br i1 %tobool, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %i.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.end.loopexit ]
  ret i32 %i.0.lcssa
}

; Recognize CTLZ builtin pattern.
; Here it will replace the loop -
; assume builtin is always profitable.
;
; int ctlz_zero_check_lshr(int n)
; {
;   int i = 0;
;   while(n) {
;     n >>= 1;
;     i++;
;   }
;   return i;
; }
;
; ALL:  entry
; ALL:  %0 = call i32 @llvm.ctlz.i32(i32 %n, i1 true)
; ALL-NEXT:  %1 = sub i32 32, %0
; ALL:  %inc.lcssa = phi i32 [ %1, %while.body ]
; ALL:  %i.0.lcssa = phi i32 [ 0, %entry ], [ %inc.lcssa, %while.end.loopexit ]
; ALL:  ret i32 %i.0.lcssa

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @ctlz_zero_check_lshr(i32 %n) {
entry:
  %tobool4 = icmp eq i32 %n, 0
  br i1 %tobool4, label %while.end, label %while.body.preheader

while.body.preheader:                             ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.preheader, %while.body
  %i.06 = phi i32 [ %inc, %while.body ], [ 0, %while.body.preheader ]
  %n.addr.05 = phi i32 [ %shr, %while.body ], [ %n, %while.body.preheader ]
  %shr = lshr i32 %n.addr.05, 1
  %inc = add nsw i32 %i.06, 1
  %tobool = icmp eq i32 %shr, 0
  br i1 %tobool, label %while.end.loopexit, label %while.body

while.end.loopexit:                               ; preds = %while.body
  br label %while.end

while.end:                                        ; preds = %while.end.loopexit, %entry
  %i.0.lcssa = phi i32 [ 0, %entry ], [ %inc, %while.end.loopexit ]
  ret i32 %i.0.lcssa
}

; Recognize CTLZ builtin pattern.
; Here it will replace the loop -
; assume builtin is always profitable.
;
; int ctlz(int n)
; {
;   n = n >= 0 ? n : -n;
;   int i = 0;
;   while(n >>= 1) {
;     i++;
;   }
;   return i;
; }
;
; ALL:  entry
; ALL:  %0 = ashr i32 %abs_n, 1
; ALL-NEXT:  %1 = call i32 @llvm.ctlz.i32(i32 %0, i1 false)
; ALL-NEXT:  %2 = sub i32 32, %1
; ALL-NEXT:  %3 = add i32 %2, 1
; ALL:  %i.0.lcssa = phi i32 [ %2, %while.cond ]
; ALL:  ret i32 %i.0.lcssa

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @ctlz(i32 %n) {
entry:
  %c = icmp sgt i32 %n, 0
  %negn = sub nsw i32 0, %n
  %abs_n = select i1 %c, i32 %n, i32 %negn
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %n.addr.0 = phi i32 [ %abs_n, %entry ], [ %shr, %while.cond ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %while.cond ]
  %shr = ashr i32 %n.addr.0, 1
  %tobool = icmp eq i32 %shr, 0
  %inc = add nsw i32 %i.0, 1
  br i1 %tobool, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  ret i32 %i.0
}

; Recognize CTLZ builtin pattern.
; Here it will replace the loop -
; assume builtin is always profitable.
;
; int ctlz_lshr(int n)
; {
;   int i = 0;
;   while(n >>= 1) {
;     i++;
;   }
;   return i;
; }
;
; ALL:  entry
; ALL:  %0 = lshr i32 %n, 1
; ALL-NEXT:  %1 = call i32 @llvm.ctlz.i32(i32 %0, i1 false)
; ALL-NEXT:  %2 = sub i32 32, %1
; ALL-NEXT:  %3 = add i32 %2, 1
; ALL:  %i.0.lcssa = phi i32 [ %2, %while.cond ]
; ALL:  ret i32 %i.0.lcssa

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @ctlz_lshr(i32 %n) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %n.addr.0 = phi i32 [ %n, %entry ], [ %shr, %while.cond ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %while.cond ]
  %shr = lshr i32 %n.addr.0, 1
  %tobool = icmp eq i32 %shr, 0
  %inc = add nsw i32 %i.0, 1
  br i1 %tobool, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  ret i32 %i.0
}

; Recognize CTLZ builtin pattern.
; Here it will replace the loop -
; assume builtin is always profitable.
;
; int ctlz_add(int n, int i0)
; {
;   n = n >= 0 ? n : -n;
;   int i = i0;
;   while(n >>= 1) {
;     i++;
;   }
;   return i;
; }
;
; ALL:  entry
; ALL:  %0 = ashr i32 %abs_n, 1
; ALL-NEXT:  %1 = call i32 @llvm.ctlz.i32(i32 %0, i1 false)
; ALL-NEXT:  %2 = sub i32 32, %1
; ALL-NEXT:  %3 = add i32 %2, 1
; ALL-NEXT:  %4 = add i32 %2, %i0
; ALL:  %i.0.lcssa = phi i32 [ %4, %while.cond ]
; ALL:  ret i32 %i.0.lcssa
;
; Function Attrs: norecurse nounwind readnone uwtable
define i32 @ctlz_add(i32 %n, i32 %i0) {
entry:
  %c = icmp sgt i32 %n, 0
  %negn = sub nsw i32 0, %n
  %abs_n = select i1 %c, i32 %n, i32 %negn
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %n.addr.0 = phi i32 [ %abs_n, %entry ], [ %shr, %while.cond ]
  %i.0 = phi i32 [ %i0, %entry ], [ %inc, %while.cond ]
  %shr = ashr i32 %n.addr.0, 1
  %tobool = icmp eq i32 %shr, 0
  %inc = add nsw i32 %i.0, 1
  br i1 %tobool, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  ret i32 %i.0
}

; Recognize CTLZ builtin pattern.
; Here it will replace the loop -
; assume builtin is always profitable.
;
; int ctlz_add_lshr(int n, int i0)
; {
;   int i = i0;
;   while(n >>= 1) {
;     i++;
;   }
;   return i;
; }
;
; ALL:  entry
; ALL:  %0 = lshr i32 %n, 1
; ALL-NEXT:  %1 = call i32 @llvm.ctlz.i32(i32 %0, i1 false)
; ALL-NEXT:  %2 = sub i32 32, %1
; ALL-NEXT:  %3 = add i32 %2, 1
; ALL-NEXT:  %4 = add i32 %2, %i0
; ALL:  %i.0.lcssa = phi i32 [ %4, %while.cond ]
; ALL:  ret i32 %i.0.lcssa
;
; Function Attrs: norecurse nounwind readnone uwtable
define i32 @ctlz_add_lshr(i32 %n, i32 %i0) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %n.addr.0 = phi i32 [ %n, %entry ], [ %shr, %while.cond ]
  %i.0 = phi i32 [ %i0, %entry ], [ %inc, %while.cond ]
  %shr = lshr i32 %n.addr.0, 1
  %tobool = icmp eq i32 %shr, 0
  %inc = add nsw i32 %i.0, 1
  br i1 %tobool, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  ret i32 %i.0
}

; Recognize CTLZ builtin pattern.
; Here it will replace the loop -
; assume builtin is always profitable.
;
; int ctlz_sext(short in)
; {
;   int n = in;
;   if (in < 0)
;     n = -n;
;   int i = 0;
;   while(n >>= 1) {
;     i++;
;   }
;   return i;
; }
;
; ALL:  entry
; ALL:  %0 = ashr i32 %abs_n, 1
; ALL-NEXT:  %1 = call i32 @llvm.ctlz.i32(i32 %0, i1 false)
; ALL-NEXT:  %2 = sub i32 32, %1
; ALL-NEXT:  %3 = add i32 %2, 1
; ALL:  %i.0.lcssa = phi i32 [ %2, %while.cond ]
; ALL:  ret i32 %i.0.lcssa

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @ctlz_sext(i16 %in) {
entry:
  %n = sext i16 %in to i32
  %c = icmp sgt i16 %in, 0
  %negn = sub nsw i32 0, %n
  %abs_n = select i1 %c, i32 %n, i32 %negn
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %n.addr.0 = phi i32 [ %abs_n, %entry ], [ %shr, %while.cond ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %while.cond ]
  %shr = ashr i32 %n.addr.0, 1
  %tobool = icmp eq i32 %shr, 0
  %inc = add nsw i32 %i.0, 1
  br i1 %tobool, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  ret i32 %i.0
}

; Recognize CTLZ builtin pattern.
; Here it will replace the loop -
; assume builtin is always profitable.
;
; int ctlz_sext_lshr(short in)
; {
;   int i = 0;
;   while(in >>= 1) {
;     i++;
;   }
;   return i;
; }
;
; ALL:  entry
; ALL:  %0 = lshr i32 %n, 1
; ALL-NEXT:  %1 = call i32 @llvm.ctlz.i32(i32 %0, i1 false)
; ALL-NEXT:  %2 = sub i32 32, %1
; ALL-NEXT:  %3 = add i32 %2, 1
; ALL:  %i.0.lcssa = phi i32 [ %2, %while.cond ]
; ALL:  ret i32 %i.0.lcssa

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @ctlz_sext_lshr(i16 %in) {
entry:
  %n = sext i16 %in to i32
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %n.addr.0 = phi i32 [ %n, %entry ], [ %shr, %while.cond ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %while.cond ]
  %shr = lshr i32 %n.addr.0, 1
  %tobool = icmp eq i32 %shr, 0
  %inc = add nsw i32 %i.0, 1
  br i1 %tobool, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  ret i32 %i.0
}

; This loop contains a volatile store. If x is initially negative,
; the code will be an infinite loop because the ashr will eventually produce
; all ones and continue doing so. This prevents the loop from terminating. If
; we convert this to a countable loop using ctlz that loop will only run 32
; times. This is different than the infinite number of times of the original.
define i32 @foo(i32 %x) {
; LZCNT-LABEL: @foo(
; LZCNT-NEXT:  entry:
; LZCNT-NEXT:    [[V:%.*]] = alloca i8, align 1
; LZCNT-NEXT:    [[TOBOOL4:%.*]] = icmp eq i32 [[X:%.*]], 0
; LZCNT-NEXT:    br i1 [[TOBOOL4]], label [[WHILE_END:%.*]], label [[WHILE_BODY_LR_PH:%.*]]
; LZCNT:       while.body.lr.ph:
; LZCNT-NEXT:    br label [[WHILE_BODY:%.*]]
; LZCNT:       while.body:
; LZCNT-NEXT:    [[CNT_06:%.*]] = phi i32 [ 0, [[WHILE_BODY_LR_PH]] ], [ [[INC:%.*]], [[WHILE_BODY]] ]
; LZCNT-NEXT:    [[X_ADDR_05:%.*]] = phi i32 [ [[X]], [[WHILE_BODY_LR_PH]] ], [ [[SHR:%.*]], [[WHILE_BODY]] ]
; LZCNT-NEXT:    [[SHR]] = ashr i32 [[X_ADDR_05]], 1
; LZCNT-NEXT:    [[INC]] = add i32 [[CNT_06]], 1
; LZCNT-NEXT:    store volatile i8 42, i8* [[V]], align 1
; LZCNT-NEXT:    [[TOBOOL:%.*]] = icmp eq i32 [[SHR]], 0
; LZCNT-NEXT:    br i1 [[TOBOOL]], label [[WHILE_COND_WHILE_END_CRIT_EDGE:%.*]], label [[WHILE_BODY]]
; LZCNT:       while.cond.while.end_crit_edge:
; LZCNT-NEXT:    [[SPLIT:%.*]] = phi i32 [ [[INC]], [[WHILE_BODY]] ]
; LZCNT-NEXT:    br label [[WHILE_END]]
; LZCNT:       while.end:
; LZCNT-NEXT:    [[CNT_0_LCSSA:%.*]] = phi i32 [ [[SPLIT]], [[WHILE_COND_WHILE_END_CRIT_EDGE]] ], [ 0, [[ENTRY:%.*]] ]
; LZCNT-NEXT:    ret i32 [[CNT_0_LCSSA]]
;
; NOLZCNT-LABEL: @foo(
; NOLZCNT-NEXT:  entry:
; NOLZCNT-NEXT:    [[V:%.*]] = alloca i8, align 1
; NOLZCNT-NEXT:    [[TOBOOL4:%.*]] = icmp eq i32 [[X:%.*]], 0
; NOLZCNT-NEXT:    br i1 [[TOBOOL4]], label [[WHILE_END:%.*]], label [[WHILE_BODY_LR_PH:%.*]]
; NOLZCNT:       while.body.lr.ph:
; NOLZCNT-NEXT:    br label [[WHILE_BODY:%.*]]
; NOLZCNT:       while.body:
; NOLZCNT-NEXT:    [[CNT_06:%.*]] = phi i32 [ 0, [[WHILE_BODY_LR_PH]] ], [ [[INC:%.*]], [[WHILE_BODY]] ]
; NOLZCNT-NEXT:    [[X_ADDR_05:%.*]] = phi i32 [ [[X]], [[WHILE_BODY_LR_PH]] ], [ [[SHR:%.*]], [[WHILE_BODY]] ]
; NOLZCNT-NEXT:    [[SHR]] = ashr i32 [[X_ADDR_05]], 1
; NOLZCNT-NEXT:    [[INC]] = add i32 [[CNT_06]], 1
; NOLZCNT-NEXT:    store volatile i8 42, i8* [[V]], align 1
; NOLZCNT-NEXT:    [[TOBOOL:%.*]] = icmp eq i32 [[SHR]], 0
; NOLZCNT-NEXT:    br i1 [[TOBOOL]], label [[WHILE_COND_WHILE_END_CRIT_EDGE:%.*]], label [[WHILE_BODY]]
; NOLZCNT:       while.cond.while.end_crit_edge:
; NOLZCNT-NEXT:    [[SPLIT:%.*]] = phi i32 [ [[INC]], [[WHILE_BODY]] ]
; NOLZCNT-NEXT:    br label [[WHILE_END]]
; NOLZCNT:       while.end:
; NOLZCNT-NEXT:    [[CNT_0_LCSSA:%.*]] = phi i32 [ [[SPLIT]], [[WHILE_COND_WHILE_END_CRIT_EDGE]] ], [ 0, [[ENTRY:%.*]] ]
; NOLZCNT-NEXT:    ret i32 [[CNT_0_LCSSA]]
;
entry:
  %v = alloca i8, align 1
  %tobool4 = icmp eq i32 %x, 0
  br i1 %tobool4, label %while.end, label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %entry
  br label %while.body

while.body:                                       ; preds = %while.body.lr.ph, %while.body
  %cnt.06 = phi i32 [ 0, %while.body.lr.ph ], [ %inc, %while.body ]
  %x.addr.05 = phi i32 [ %x, %while.body.lr.ph ], [ %shr, %while.body ]
  %shr = ashr i32 %x.addr.05, 1
  %inc = add i32 %cnt.06, 1
  store volatile i8 42, i8* %v, align 1
  %tobool = icmp eq i32 %shr, 0
  br i1 %tobool, label %while.cond.while.end_crit_edge, label %while.body

while.cond.while.end_crit_edge:                   ; preds = %while.body
  %split = phi i32 [ %inc, %while.body ]
  br label %while.end

while.end:                                        ; preds = %while.cond.while.end_crit_edge, %entry
  %cnt.0.lcssa = phi i32 [ %split, %while.cond.while.end_crit_edge ], [ 0, %entry ]
  ret i32 %cnt.0.lcssa
}

; We can't easily transform this loop. It returns 1 for an input of both
; 0 and 1.
;
; int ctlz_bad(unsigned n)
; {
;   int i = 0;
;   do {
;     i++;
;     n >>= 1;
;   } while(n != 0) {
;   return i;
; }
;
; Function Attrs: norecurse nounwind readnone uwtable
define i32 @ctlz_bad(i32 %n) {
; ALL-LABEL: @ctlz_bad(
; ALL-NEXT:  entry:
; ALL-NEXT:    br label [[WHILE_COND:%.*]]
; ALL:       while.cond:
; ALL-NEXT:    [[N_ADDR_0:%.*]] = phi i32 [ [[N:%.*]], [[ENTRY:%.*]] ], [ [[SHR:%.*]], [[WHILE_COND]] ]
; ALL-NEXT:    [[I_0:%.*]] = phi i32 [ 0, [[ENTRY]] ], [ [[INC:%.*]], [[WHILE_COND]] ]
; ALL-NEXT:    [[SHR]] = lshr i32 [[N_ADDR_0]], 1
; ALL-NEXT:    [[TOBOOL:%.*]] = icmp eq i32 [[SHR]], 0
; ALL-NEXT:    [[INC]] = add nsw i32 [[I_0]], 1
; ALL-NEXT:    br i1 [[TOBOOL]], label [[WHILE_END:%.*]], label [[WHILE_COND]]
; ALL:       while.end:
; ALL-NEXT:    [[INC_LCSSA:%.*]] = phi i32 [ [[INC]], [[WHILE_COND]] ]
; ALL-NEXT:    ret i32 [[INC_LCSSA]]
;
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %n.addr.0 = phi i32 [ %n, %entry ], [ %shr, %while.cond ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %while.cond ]
  %shr = lshr i32 %n.addr.0, 1
  %tobool = icmp eq i32 %shr, 0
  %inc = add nsw i32 %i.0, 1
  br i1 %tobool, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  ret i32 %inc
}
