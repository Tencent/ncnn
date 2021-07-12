;RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck --check-prefix=SI --check-prefix=GCN --check-prefix=FUNC %s
;RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefix=VI --check-prefix=GCN --check-prefix=FUNC %s
;RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefix=VI --check-prefix=GCN --check-prefix=FUNC %s
;RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck --check-prefix=EG --check-prefix=FUNC %s

;FUNC-LABEL: {{^}}test_udiv:
;EG: RECIP_UINT
;EG: LSHL {{.*}}, 1,
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT

;GCN: v_mac_f32_e32 v{{[0-9]+}}, 0x4f800000,
;GCN: v_rcp_f32_e32
;GCN: v_mul_f32_e32 v{{[0-9]+}}, 0x5f7ffffc
;GCN: v_mul_f32_e32 v{{[0-9]+}}, 0x2f800000
;GCN: v_trunc_f32_e32
;GCN: v_mac_f32_e32 v{{[0-9]+}}, 0xcf800000
;GCN: s_endpgm
define amdgpu_kernel void @test_udiv(i64 addrspace(1)* %out, i64 %x, i64 %y) {
  %result = udiv i64 %x, %y
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

;FUNC-LABEL: {{^}}test_urem:
;EG: RECIP_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: BFE_UINT
;EG: AND_INT {{.*}}, 1,

;GCN: v_mac_f32_e32 v{{[0-9]+}}, 0x4f800000,
;GCN: v_rcp_f32_e32
;GCN: v_mul_f32_e32 v{{[0-9]+}}, 0x5f7ffffc
;GCN: v_mul_f32_e32 v{{[0-9]+}}, 0x2f800000
;GCN: v_trunc_f32_e32
;GCN: v_mac_f32_e32 v{{[0-9]+}}, 0xcf800000
;GCN: s_endpgm
define amdgpu_kernel void @test_urem(i64 addrspace(1)* %out, i64 %x, i64 %y) {
  %result = urem i64 %x, %y
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

;FUNC-LABEL: {{^}}test_udiv3264:
;EG: RECIP_UINT
;EG-NOT: BFE_UINT

;GCN-NOT: s_bfe_u32
;GCN-NOT: v_mad_f32
;SI-NOT: v_lshr_b64
;VI-NOT: v_lshrrev_b64
;GCN: s_endpgm
define amdgpu_kernel void @test_udiv3264(i64 addrspace(1)* %out, i64 %x, i64 %y) {
  %1 = lshr i64 %x, 33
  %2 = lshr i64 %y, 33
  %result = udiv i64 %1, %2
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

;FUNC-LABEL: {{^}}test_urem3264:
;EG: RECIP_UINT
;EG-NOT: BFE_UINT

;GCN-NOT: s_bfe_u32
;GCN-NOT: v_mad_f32
;SI-NOT: v_lshr_b64
;VI-NOT: v_lshrrev_b64
;GCN: s_endpgm
define amdgpu_kernel void @test_urem3264(i64 addrspace(1)* %out, i64 %x, i64 %y) {
  %1 = lshr i64 %x, 33
  %2 = lshr i64 %y, 33
  %result = urem i64 %1, %2
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

;FUNC-LABEL: {{^}}test_udiv2364:
;EG: UINT_TO_FLT
;EG: UINT_TO_FLT
;EG: FLT_TO_UINT
;EG-NOT: RECIP_UINT
;EG-NOT: BFE_UINT

;SI-NOT: v_lshr_b64
;VI-NOT: v_lshrrev_b64
;GCN: v_mad_f32
;GCN: s_endpgm
define amdgpu_kernel void @test_udiv2364(i64 addrspace(1)* %out, i64 %x, i64 %y) {
  %1 = lshr i64 %x, 41
  %2 = lshr i64 %y, 41
  %result = udiv i64 %1, %2
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

;FUNC-LABEL: {{^}}test_urem2364:
;EG: UINT_TO_FLT
;EG: UINT_TO_FLT
;EG: FLT_TO_UINT
;EG-NOT: RECIP_UINT
;EG-NOT: BFE_UINT

;SI-NOT: v_lshr_b64
;VI-NOT: v_lshrrev_b64
;GCN: v_mad_f32
;GCN: s_endpgm
define amdgpu_kernel void @test_urem2364(i64 addrspace(1)* %out, i64 %x, i64 %y) {
  %1 = lshr i64 %x, 41
  %2 = lshr i64 %y, 41
  %result = urem i64 %1, %2
  store i64 %result, i64 addrspace(1)* %out
  ret void
}

;FUNC-LABEL: {{^}}test_udiv_k:
;GCN: v_mul{{.+}} v{{[0-9]+}}, v{{[0-9]+}}, 24
;GCN: v_mul{{.+}} v{{[0-9]+}}, v{{[0-9]+}}, 24
;GCN: v_mul{{.+}} v{{[0-9]+}}, v{{[0-9]+}}, 24
;GCN: v_add
;GCN: v_addc
;GCN: v_addc
;GCN: s_endpgm
define amdgpu_kernel void @test_udiv_k(i64 addrspace(1)* %out, i64 %x) {
  %result = udiv i64 24, %x
  store i64 %result, i64 addrspace(1)* %out
  ret void
}
