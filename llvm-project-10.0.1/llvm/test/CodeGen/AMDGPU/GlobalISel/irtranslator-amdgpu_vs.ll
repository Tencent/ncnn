; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=fiji -stop-after=irtranslator -global-isel %s -o - | FileCheck %s

; CHECK-LABEL: name: test_f32_inreg
; CHECK: [[S0:%[0-9]+]]:_(s32) = COPY $sgpr2
; CHECK: G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.exp), 32, 15, [[S0]]
define amdgpu_vs void @test_f32_inreg(float inreg %arg0) {
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %arg0, float undef, float undef, float undef, i1 false, i1 false) #0
  ret void
}

; CHECK-LABEL: name: test_f32
; CHECK: [[V0:%[0-9]+]]:_(s32) = COPY $vgpr0
; CHECK: G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.exp), 32, 15, [[V0]]
define amdgpu_vs void @test_f32(float %arg0) {
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %arg0, float undef, float undef, float undef, i1 false, i1 false) #0
  ret void
}

; CHECK-LABEL: name: test_ptr2_inreg
; CHECK: [[S2:%[0-9]+]]:_(s32) = COPY $sgpr2
; CHECK: [[S3:%[0-9]+]]:_(s32) = COPY $sgpr3
; CHECK: [[PTR:%[0-9]+]]:_(p4) = G_MERGE_VALUES [[S2]](s32), [[S3]](s32)
; CHECK: G_LOAD [[PTR]]
define amdgpu_vs void @test_ptr2_inreg(i32 addrspace(4)* inreg %arg0) {
  %tmp0 = load volatile i32, i32 addrspace(4)* %arg0
  ret void
}

; CHECK-LABEL: name: test_sgpr_alignment0
; CHECK: [[S2:%[0-9]+]]:_(s32) = COPY $sgpr2
; CHECK: [[S3:%[0-9]+]]:_(s32) = COPY $sgpr3
; CHECK: [[S4:%[0-9]+]]:_(s32) = COPY $sgpr4
; CHECK: [[S34:%[0-9]+]]:_(p4) = G_MERGE_VALUES [[S3]](s32), [[S4]](s32)
; CHECK: G_LOAD [[S34]]
; CHECK: G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.exp), 32, 15, [[S2]](s32)
define amdgpu_vs void @test_sgpr_alignment0(float inreg %arg0, i32 addrspace(4)* inreg %arg1) {
  %tmp0 = load volatile i32, i32 addrspace(4)* %arg1
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %arg0, float undef, float undef, float undef, i1 false, i1 false) #0
  ret void
}

; CHECK-LABEL: name: test_order
; CHECK: [[S0:%[0-9]+]]:_(s32) = COPY $sgpr2
; CHECK: [[S1:%[0-9]+]]:_(s32) = COPY $sgpr3
; CHECK: [[V0:%[0-9]+]]:_(s32) = COPY $vgpr0
; CHECK: [[V1:%[0-9]+]]:_(s32) = COPY $vgpr1
; CHECK: G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.exp), 32, 15, [[V0]](s32), [[S0]](s32), [[V1]](s32), [[S1]](s32)
define amdgpu_vs void @test_order(float inreg %arg0, float inreg %arg1, float %arg2, float %arg3) {
  call void @llvm.amdgcn.exp.f32(i32 32, i32 15, float %arg2, float %arg0, float %arg3, float %arg1, i1 false, i1 false) #0
  ret void
}

; CHECK-LABEL: name: ret_struct
; CHECK: [[S0:%[0-9]+]]:_(s32) = COPY $sgpr2
; CHECK: [[S1:%[0-9]+]]:_(s32) = COPY $sgpr3
; CHECK: $sgpr0 = COPY [[S0]]
; CHECK: $sgpr1 = COPY [[S1]]
; CHECK: SI_RETURN_TO_EPILOG implicit $sgpr0, implicit $sgpr1
define amdgpu_vs <{ i32, i32 }> @ret_struct(i32 inreg %arg0, i32 inreg %arg1) {
main_body:
  %tmp0 = insertvalue <{ i32, i32 }> undef, i32 %arg0, 0
  %tmp1 = insertvalue <{ i32, i32 }> %tmp0, i32 %arg1, 1
  ret <{ i32, i32 }> %tmp1
}

; CHECK_LABEL: name: non_void_ret
; CHECK: [[ZERO:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
; CHECK: $sgpr0 = COPY [[ZERO]]
; SI_RETURN_TO_EPILOG $sgpr0
define amdgpu_vs i32 @non_void_ret() {
  ret i32 0
}

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1)  #0

attributes #0 = { nounwind }
