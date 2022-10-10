include(CheckCXXSourceCompiles)

set(TEMP_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
set(CMAKE_REQUIRED_FLAGS "-march=rv64gcv")

check_cxx_source_compiles("
#include <riscv_vector.h>
int main(void)
{
    float in1[4] = {-1.f,0.f,+1.f,2.f};
    float out1=0;
    size_t vl = vsetvl_e32m8(4);
    vfloat32m8_t _add = vle32_v_f32m8(in1,vl);
    vfloat32m1_t _sum = vfmv_s_f_f32m1(vundefined_f32m1(),out1,vl);
    _sum = vfredsum_vs_f32m8_f32m1(_sum, _add, _sum, vl);
    out1 = vfmv_f_s_f32m1_f32(_sum);
    return 0;
}
" NCNN_COMPILER_USE_VFREDSUM)
check_cxx_source_compiles("
#include <riscv_vector.h>
int main(void)
{
    float in1[4] = {-1.f,0.f,+1.f,2.f};
    float out1=0;
    size_t vl = vsetvl_e32m8(4);
    vfloat32m8_t _add = vle32_v_f32m8(in1,vl);
    vfloat32m1_t _sum = vfmv_s_f_f32m1(vundefined_f32m1(),out1,vl);
    _sum = vfredusum_vs_f32m8_f32m1(_sum, _add, _sum, vl);
    out1 = vfmv_f_s_f32m1_f32(_sum);
    return 0;
};
" NCNN_COMPILER_USE_VFREDUSUM)

if(NCNN_COMPILER_USE_VFREDSUM AND NOT NCNN_COMPILER_USE_VFREDUSUM)
    message(WARNING "The compiler uses vfredsum. Upgrading your toolchain is strongly recommended.")
    foreach(LMUL 1 2 4 8)
        add_definitions(-Dvfredusum_vs_f32m${LMUL}_f32m1=vfredsum_vs_f32m${LMUL}_f32m1)
        if(NCNN_COMPILER_SUPPORT_RVV_ZFH OR NCNN_COMPILER_SUPPORT_RVV_ZVFH)
            add_definitions(-Dvfredusum_vs_f16m${LMUL}_f16m1=vfredsum_vs_f16m${LMUL}_f16m1)
        endif()
    endforeach()
endif()

set(CMAKE_REQUIRED_FLAGS ${TEMP_CMAKE_REQUIRED_FLAGS})
unset(TEMP_CMAKE_REQUIRED_FLAGS)
