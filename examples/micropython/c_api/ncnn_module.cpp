// 包含ncnn C++ API
#include "ncnn/c_api.h"

extern "C" {
    #include "py/runtime.h"
    #include "py/obj.h"
    #include "py/objstr.h"
    #include <string.h>
}

extern "C" {
    // NCNN版本获取函数
    mp_obj_t ncnn_version_bind(void) {
        const char* ver = ncnn_version();
        return mp_obj_new_str(ver, strlen(ver));
    }
}

// 定义函数对象
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_version_obj, ncnn_version_bind);

// 模块全局符号表
static const mp_rom_map_elem_t ncnn_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_ncnn) },
    { MP_ROM_QSTR(MP_QSTR_version), MP_ROM_PTR(&ncnn_version_obj) },
};

// 定义模块全局字典
static MP_DEFINE_CONST_DICT(ncnn_module_globals, ncnn_module_globals_table);

// 定义模块对象
extern "C" const mp_obj_module_t ncnn_user_cmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t*)&ncnn_module_globals,
};

// 注册模块
extern "C" {
    MP_REGISTER_MODULE(MP_QSTR_ncnn, ncnn_user_cmodule);
}