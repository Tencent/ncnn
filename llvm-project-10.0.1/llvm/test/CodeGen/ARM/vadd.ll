; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <8 x i8> @vaddi8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vaddi8:
;CHECK: vadd.i8
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = add <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vaddi16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vaddi16:
;CHECK: vadd.i16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = add <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vaddi32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vaddi32:
;CHECK: vadd.i32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = add <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <1 x i64> @vaddi64(<1 x i64>* %A, <1 x i64>* %B) nounwind {
;CHECK-LABEL: vaddi64:
;CHECK: vadd.i64
	%tmp1 = load <1 x i64>, <1 x i64>* %A
	%tmp2 = load <1 x i64>, <1 x i64>* %B
	%tmp3 = add <1 x i64> %tmp1, %tmp2
	ret <1 x i64> %tmp3
}

define <2 x float> @vaddf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK-LABEL: vaddf32:
;CHECK: vadd.f32
	%tmp1 = load <2 x float>, <2 x float>* %A
	%tmp2 = load <2 x float>, <2 x float>* %B
	%tmp3 = fadd <2 x float> %tmp1, %tmp2
	ret <2 x float> %tmp3
}

define <16 x i8> @vaddQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: vaddQi8:
;CHECK: vadd.i8
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = add <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vaddQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vaddQi16:
;CHECK: vadd.i16
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = add <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vaddQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vaddQi32:
;CHECK: vadd.i32
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = add <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <2 x i64> @vaddQi64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: vaddQi64:
;CHECK: vadd.i64
	%tmp1 = load <2 x i64>, <2 x i64>* %A
	%tmp2 = load <2 x i64>, <2 x i64>* %B
	%tmp3 = add <2 x i64> %tmp1, %tmp2
	ret <2 x i64> %tmp3
}

define <4 x float> @vaddQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK-LABEL: vaddQf32:
;CHECK: vadd.f32
	%tmp1 = load <4 x float>, <4 x float>* %A
	%tmp2 = load <4 x float>, <4 x float>* %B
	%tmp3 = fadd <4 x float> %tmp1, %tmp2
	ret <4 x float> %tmp3
}

define <8 x i8> @vraddhni16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vraddhni16:
;CHECK: vraddhn.i16
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vraddhn.v8i8(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i8> %tmp3
}

define <4 x i16> @vraddhni32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vraddhni32:
;CHECK: vraddhn.i32
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vraddhn.v4i16(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vraddhni64(<2 x i64>* %A, <2 x i64>* %B) nounwind {
;CHECK-LABEL: vraddhni64:
;CHECK: vraddhn.i64
	%tmp1 = load <2 x i64>, <2 x i64>* %A
	%tmp2 = load <2 x i64>, <2 x i64>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vraddhn.v2i32(<2 x i64> %tmp1, <2 x i64> %tmp2)
	ret <2 x i32> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vraddhn.v8i8(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vraddhn.v4i16(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vraddhn.v2i32(<2 x i64>, <2 x i64>) nounwind readnone

define <8 x i8> @vaddhni16_natural(<8 x i16> %A, <8 x i16> %B) nounwind {
; CHECK-LABEL: vaddhni16_natural:
; CHECK: vaddhn.i16
  %sum = add <8 x i16> %A, %B
  %shift = lshr <8 x i16> %sum, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %trunc = trunc <8 x i16> %shift to <8 x i8>
  ret <8 x i8> %trunc
}

define <4 x i16> @vaddhni32_natural(<4 x i32> %A, <4 x i32> %B) nounwind {
; CHECK-LABEL: vaddhni32_natural:
; CHECK: vaddhn.i32
  %sum = add <4 x i32> %A, %B
  %shift = lshr <4 x i32> %sum, <i32 16, i32 16, i32 16, i32 16>
  %trunc = trunc <4 x i32> %shift to <4 x i16>
  ret <4 x i16> %trunc
}

define <2 x i32> @vaddhni64_natural(<2 x i64> %A, <2 x i64> %B) nounwind {
; CHECK-LABEL: vaddhni64_natural:
; CHECK: vaddhn.i64
  %sum = add <2 x i64> %A, %B
  %shift = lshr <2 x i64> %sum, <i64 32, i64 32>
  %trunc = trunc <2 x i64> %shift to <2 x i32>
  ret <2 x i32> %trunc
}

define <8 x i16> @vaddls8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vaddls8:
;CHECK: vaddl.s8
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = sext <8 x i8> %tmp1 to <8 x i16>
	%tmp4 = sext <8 x i8> %tmp2 to <8 x i16>
	%tmp5 = add <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

define <4 x i32> @vaddls16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vaddls16:
;CHECK: vaddl.s16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = sext <4 x i16> %tmp1 to <4 x i32>
	%tmp4 = sext <4 x i16> %tmp2 to <4 x i32>
	%tmp5 = add <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

define <2 x i64> @vaddls32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vaddls32:
;CHECK: vaddl.s32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = sext <2 x i32> %tmp1 to <2 x i64>
	%tmp4 = sext <2 x i32> %tmp2 to <2 x i64>
	%tmp5 = add <2 x i64> %tmp3, %tmp4
	ret <2 x i64> %tmp5
}

define <8 x i16> @vaddlu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vaddlu8:
;CHECK: vaddl.u8
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = zext <8 x i8> %tmp1 to <8 x i16>
	%tmp4 = zext <8 x i8> %tmp2 to <8 x i16>
	%tmp5 = add <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

define <4 x i32> @vaddlu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vaddlu16:
;CHECK: vaddl.u16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = zext <4 x i16> %tmp1 to <4 x i32>
	%tmp4 = zext <4 x i16> %tmp2 to <4 x i32>
	%tmp5 = add <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

define <2 x i64> @vaddlu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vaddlu32:
;CHECK: vaddl.u32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = zext <2 x i32> %tmp1 to <2 x i64>
	%tmp4 = zext <2 x i32> %tmp2 to <2 x i64>
	%tmp5 = add <2 x i64> %tmp3, %tmp4
	ret <2 x i64> %tmp5
}

define <8 x i16> @vaddws8(<8 x i16>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vaddws8:
;CHECK: vaddw.s8
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = sext <8 x i8> %tmp2 to <8 x i16>
	%tmp4 = add <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vaddws16(<4 x i32>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vaddws16:
;CHECK: vaddw.s16
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = sext <4 x i16> %tmp2 to <4 x i32>
	%tmp4 = add <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @vaddws32(<2 x i64>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vaddws32:
;CHECK: vaddw.s32
	%tmp1 = load <2 x i64>, <2 x i64>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = sext <2 x i32> %tmp2 to <2 x i64>
	%tmp4 = add <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}

define <8 x i16> @vaddwu8(<8 x i16>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: vaddwu8:
;CHECK: vaddw.u8
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = zext <8 x i8> %tmp2 to <8 x i16>
	%tmp4 = add <8 x i16> %tmp1, %tmp3
	ret <8 x i16> %tmp4
}

define <4 x i32> @vaddwu16(<4 x i32>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vaddwu16:
;CHECK: vaddw.u16
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = zext <4 x i16> %tmp2 to <4 x i32>
	%tmp4 = add <4 x i32> %tmp1, %tmp3
	ret <4 x i32> %tmp4
}

define <2 x i64> @vaddwu32(<2 x i64>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vaddwu32:
;CHECK: vaddw.u32
	%tmp1 = load <2 x i64>, <2 x i64>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = zext <2 x i32> %tmp2 to <2 x i64>
	%tmp4 = add <2 x i64> %tmp1, %tmp3
	ret <2 x i64> %tmp4
}
