; RUN: opt < %s -ipconstprop -instcombine -S | FileCheck %s

; CHECK: ret i1 true
; CHECK: ret i1 true
; CHECK-NOT: ret i1 true

define internal i32 @foo(i1 %C) {
        br i1 %C, label %T, label %F

T:              ; preds = %0
        ret i32 52

F:              ; preds = %0
        ret i32 52
}

define i1 @caller(i1 %C) {
        %X = call i32 @foo( i1 %C )             ; <i32> [#uses=1]
        %Y = icmp ne i32 %X, 0          ; <i1> [#uses=1]
        ret i1 %Y
}

define i1 @invokecaller(i1 %C) personality i32 (...)* @__gxx_personality_v0 {
        %X = invoke i32 @foo( i1 %C ) to label %OK unwind label %FAIL             ; <i32> [#uses=1]
OK:
        %Y = icmp ne i32 %X, 0          ; <i1> [#uses=1]
        ret i1 %Y 
FAIL:
        %exn = landingpad {i8*, i32}
                 cleanup
        ret i1 false
}

declare i32 @__gxx_personality_v0(...)
