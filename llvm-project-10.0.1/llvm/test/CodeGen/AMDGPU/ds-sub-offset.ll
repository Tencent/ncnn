; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

declare i32 @llvm.amdgcn.workitem.id.x() #0

@lds.obj = addrspace(3) global [256 x i32] undef, align 4

; GCN-LABEL: {{^}}write_ds_sub0_offset0_global:
; GCN: v_lshlrev_b32_e32 [[SHL:v[0-9]+]], 2, v0
; GCN: v_sub_{{[iu]}}32_e32 [[BASEPTR:v[0-9]+]], {{(vcc, )?}}lds.obj@abs32@lo, [[SHL]]
; GCN: v_mov_b32_e32 [[VAL:v[0-9]+]], 0x7b
; GCN: ds_write_b32 [[BASEPTR]], [[VAL]] offset:12
define amdgpu_kernel void @write_ds_sub0_offset0_global() #0 {
entry:
  %x.i = call i32 @llvm.amdgcn.workitem.id.x() #1
  %sub1 = sub i32 0, %x.i
  %tmp0 = getelementptr [256 x i32], [256 x i32] addrspace(3)* @lds.obj, i32 0, i32 %sub1
  %arrayidx = getelementptr inbounds i32, i32 addrspace(3)* %tmp0, i32 3
  store i32 123, i32 addrspace(3)* %arrayidx
  ret void
}

; GFX9-LABEL: {{^}}write_ds_sub0_offset0_global_clamp_bit:
; GFX9: v_sub_u32
; GFX9: s_endpgm
define amdgpu_kernel void @write_ds_sub0_offset0_global_clamp_bit(float %dummy.val) #0 {
entry:
  %x.i = call i32 @llvm.amdgcn.workitem.id.x() #1
  %sub1 = sub i32 0, %x.i
  %tmp0 = getelementptr [256 x i32], [256 x i32] addrspace(3)* @lds.obj, i32 0, i32 %sub1
  %arrayidx = getelementptr inbounds i32, i32 addrspace(3)* %tmp0, i32 3
  store i32 123, i32 addrspace(3)* %arrayidx
  %fmas = call float @llvm.amdgcn.div.fmas.f32(float %dummy.val, float %dummy.val, float %dummy.val, i1 false)
  store volatile float %fmas, float addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}add_x_shl_neg_to_sub_max_offset:
; GCN-DAG: v_lshlrev_b32_e32 [[SCALED:v[0-9]+]], 2, v0
; CI-DAG: v_sub_i32_e32 [[NEG:v[0-9]+]], vcc, 0, [[SCALED]]
; GFX9-DAG: v_sub_u32_e32 [[NEG:v[0-9]+]], 0, [[SCALED]]
; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 13
; GCN: ds_write_b8 [[NEG]], [[K]] offset:65535
define amdgpu_kernel void @add_x_shl_neg_to_sub_max_offset() #1 {
  %x.i = call i32 @llvm.amdgcn.workitem.id.x() #0
  %neg = sub i32 0, %x.i
  %shl = shl i32 %neg, 2
  %add = add i32 65535, %shl
  %ptr = inttoptr i32 %add to i8 addrspace(3)*
  store i8 13, i8 addrspace(3)* %ptr
  ret void
}

; GCN-LABEL: {{^}}add_x_shl_neg_to_sub_max_offset_p1:
; GCN-DAG: v_lshlrev_b32_e32 [[SCALED:v[0-9]+]], 2, v0
; CI-DAG: v_sub_i32_e32 [[NEG:v[0-9]+]], vcc, 0x10000, [[SCALED]]
; GFX9-DAG: v_sub_u32_e32 [[NEG:v[0-9]+]], 0x10000, [[SCALED]]
; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 13
; GCN: ds_write_b8 [[NEG]], [[K]]{{$}}
define amdgpu_kernel void @add_x_shl_neg_to_sub_max_offset_p1() #1 {
  %x.i = call i32 @llvm.amdgcn.workitem.id.x() #0
  %neg = sub i32 0, %x.i
  %shl = shl i32 %neg, 2
  %add = add i32 65536, %shl
  %ptr = inttoptr i32 %add to i8 addrspace(3)*
  store i8 13, i8 addrspace(3)* %ptr
  ret void
}

; GCN-LABEL: {{^}}add_x_shl_neg_to_sub_multi_use:
; GCN-DAG: v_lshlrev_b32_e32 [[SCALED:v[0-9]+]], 2, v0
; CI-DAG: v_sub_i32_e32 [[NEG:v[0-9]+]], vcc, 0, [[SCALED]]
; GFX9-DAG: v_sub_u32_e32 [[NEG:v[0-9]+]], 0, [[SCALED]]
; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 13
; GCN-NOT: v_sub
; GCN: ds_write_b32 [[NEG]], [[K]] offset:123{{$}}
; GCN-NOT: v_sub
; GCN: ds_write_b32 [[NEG]], [[K]] offset:456{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @add_x_shl_neg_to_sub_multi_use() #1 {
  %x.i = call i32 @llvm.amdgcn.workitem.id.x() #0
  %neg = sub i32 0, %x.i
  %shl = shl i32 %neg, 2
  %add0 = add i32 123, %shl
  %add1 = add i32 456, %shl
  %ptr0 = inttoptr i32 %add0 to i32 addrspace(3)*
  store volatile i32 13, i32 addrspace(3)* %ptr0
  %ptr1 = inttoptr i32 %add1 to i32 addrspace(3)*
  store volatile i32 13, i32 addrspace(3)* %ptr1
  ret void
}

; GCN-LABEL: {{^}}add_x_shl_neg_to_sub_multi_use_same_offset:
; GCN-DAG: v_lshlrev_b32_e32 [[SCALED:v[0-9]+]], 2, v0
; CI-DAG: v_sub_i32_e32 [[NEG:v[0-9]+]], vcc, 0, [[SCALED]]
; GFX9-DAG: v_sub_u32_e32 [[NEG:v[0-9]+]], 0, [[SCALED]]
; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 13
; GCN-NOT: v_sub
; GCN: ds_write_b32 [[NEG]], [[K]] offset:123{{$}}
; GCN-NOT: v_sub
; GCN: ds_write_b32 [[NEG]], [[K]] offset:123{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @add_x_shl_neg_to_sub_multi_use_same_offset() #1 {
  %x.i = call i32 @llvm.amdgcn.workitem.id.x() #0
  %neg = sub i32 0, %x.i
  %shl = shl i32 %neg, 2
  %add = add i32 123, %shl
  %ptr = inttoptr i32 %add to i32 addrspace(3)*
  store volatile i32 13, i32 addrspace(3)* %ptr
  store volatile i32 13, i32 addrspace(3)* %ptr
  ret void
}

; GCN-LABEL: {{^}}add_x_shl_neg_to_sub_misaligned_i64_max_offset:
; GCN-DAG: v_lshlrev_b32_e32 [[SCALED:v[0-9]+]], 2, v0
; CI-DAG: v_sub_i32_e32 [[NEG:v[0-9]+]], vcc, 0, [[SCALED]]
; GFX9-DAG: v_sub_u32_e32 [[NEG:v[0-9]+]], 0, [[SCALED]]
; GCN: ds_write2_b32 [[NEG]], {{v[0-9]+}}, {{v[0-9]+}} offset0:254 offset1:255
define amdgpu_kernel void @add_x_shl_neg_to_sub_misaligned_i64_max_offset() #1 {
  %x.i = call i32 @llvm.amdgcn.workitem.id.x() #0
  %neg = sub i32 0, %x.i
  %shl = shl i32 %neg, 2
  %add = add i32 1019, %shl
  %ptr = inttoptr i32 %add to i64 addrspace(3)*
  store i64 123, i64 addrspace(3)* %ptr, align 4
  ret void
}

; GFX9-LABEL: {{^}}add_x_shl_neg_to_sub_misaligned_i64_max_offset_clamp_bit:
; GFX9: v_sub_u32
; GFX9: s_endpgm
define amdgpu_kernel void @add_x_shl_neg_to_sub_misaligned_i64_max_offset_clamp_bit(float %dummy.val) #1 {
  %x.i = call i32 @llvm.amdgcn.workitem.id.x() #0
  %neg = sub i32 0, %x.i
  %shl = shl i32 %neg, 2
  %add = add i32 1019, %shl
  %ptr = inttoptr i32 %add to i64 addrspace(3)*
  store i64 123, i64 addrspace(3)* %ptr, align 4
  %fmas = call float @llvm.amdgcn.div.fmas.f32(float %dummy.val, float %dummy.val, float %dummy.val, i1 false)
  store volatile float %fmas, float addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}add_x_shl_neg_to_sub_misaligned_i64_max_offset_p1:
; GCN-DAG: v_lshlrev_b32_e32 [[SCALED:v[0-9]+]], 2, v0
; CI-DAG: v_sub_i32_e32 [[NEG:v[0-9]+]], vcc, 0x3fc, [[SCALED]]
; GFX9-DAG: v_sub_u32_e32 [[NEG:v[0-9]+]], 0x3fc, [[SCALED]]
; GCN: ds_write2_b32 [[NEG]], {{v[0-9]+}}, {{v[0-9]+}} offset1:1{{$}}
define amdgpu_kernel void @add_x_shl_neg_to_sub_misaligned_i64_max_offset_p1() #1 {
  %x.i = call i32 @llvm.amdgcn.workitem.id.x() #0
  %neg = sub i32 0, %x.i
  %shl = shl i32 %neg, 2
  %add = add i32 1020, %shl
  %ptr = inttoptr i32 %add to i64 addrspace(3)*
  store i64 123, i64 addrspace(3)* %ptr, align 4
  ret void
}

declare float @llvm.amdgcn.div.fmas.f32(float, float, float, i1)

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
attributes #2 = { nounwind convergent }
