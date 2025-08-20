// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "py/runtime.h"
#include "py/obj.h"
#include "../../src/c_api.h"

#include <string.h>

// module globals
static ncnn_option_t g_default_option = NULL;
static ncnn_allocator_t g_default_allocator = NULL;

// object structures
typedef struct _ncnn_net_obj_t {
    mp_obj_base_t base;
    ncnn_net_t net;
} ncnn_net_obj_t;

typedef struct _ncnn_mat_obj_t {
    mp_obj_base_t base;
    ncnn_mat_t mat;
} ncnn_mat_obj_t;

typedef struct _ncnn_extractor_obj_t {
    mp_obj_base_t base;
    ncnn_extractor_t extractor;
} ncnn_extractor_obj_t;

typedef struct _ncnn_option_obj_t {
    mp_obj_base_t base;
    ncnn_option_t option;
} ncnn_option_obj_t;

typedef struct _ncnn_allocator_obj_t {
    mp_obj_base_t base;
    ncnn_allocator_t allocator;
} ncnn_allocator_obj_t;

// forward declarations
extern const mp_obj_type_t ncnn_net_type;
extern const mp_obj_type_t ncnn_mat_type;
extern const mp_obj_type_t ncnn_extractor_type;
extern const mp_obj_type_t ncnn_option_type;
extern const mp_obj_type_t ncnn_allocator_type;

// version function
static mp_obj_t ncnn_get_version(void) {
    const char* version_str = ncnn_version();
    return mp_obj_new_str(version_str, strlen(version_str));
}
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_get_version_obj, ncnn_get_version);

// Mat implementation
static mp_obj_t ncnn_mat_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 0, 4, false);
    
    ncnn_mat_obj_t *self = mp_obj_malloc(ncnn_mat_obj_t, &ncnn_mat_type);
    
    if (n_args == 0) {
        self->mat = ncnn_mat_create();
    } else if (n_args == 1) {
        int w = mp_obj_get_int(args[0]);
        self->mat = ncnn_mat_create_1d(w, g_default_allocator);
    } else if (n_args == 2) {
        int w = mp_obj_get_int(args[0]);
        int h = mp_obj_get_int(args[1]);
        self->mat = ncnn_mat_create_2d(w, h, g_default_allocator);
    } else if (n_args == 3) {
        int w = mp_obj_get_int(args[0]);
        int h = mp_obj_get_int(args[1]);
        int c = mp_obj_get_int(args[2]);
        self->mat = ncnn_mat_create_3d(w, h, c, g_default_allocator);
    } else if (n_args == 4) {
        int w = mp_obj_get_int(args[0]);
        int h = mp_obj_get_int(args[1]);
        int d = mp_obj_get_int(args[2]);
        int c = mp_obj_get_int(args[3]);
        self->mat = ncnn_mat_create_4d(w, h, d, c, g_default_allocator);
    }
    
    return MP_OBJ_FROM_PTR(self);
}

static void ncnn_mat_print(const mp_print_t *print, mp_obj_t self_in, mp_print_kind_t kind) {
    ncnn_mat_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int dims = ncnn_mat_get_dims(self->mat);
    int w = ncnn_mat_get_w(self->mat);
    int h = ncnn_mat_get_h(self->mat);
    int c = ncnn_mat_get_c(self->mat);
    mp_printf(print, "Mat(%d, %d, %d, %d)", dims, w, h, c);
}

static mp_obj_t ncnn_mat_dims(mp_obj_t self_in) {
    ncnn_mat_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int(ncnn_mat_get_dims(self->mat));
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_dims_obj, ncnn_mat_dims);

static mp_obj_t ncnn_mat_w(mp_obj_t self_in) {
    ncnn_mat_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int(ncnn_mat_get_w(self->mat));
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_w_obj, ncnn_mat_w);

static mp_obj_t ncnn_mat_h(mp_obj_t self_in) {
    ncnn_mat_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int(ncnn_mat_get_h(self->mat));
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_h_obj, ncnn_mat_h);

static mp_obj_t ncnn_mat_c(mp_obj_t self_in) {
    ncnn_mat_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int(ncnn_mat_get_c(self->mat));
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_c_obj, ncnn_mat_c);

static mp_obj_t ncnn_mat_fill(mp_obj_t self_in, mp_obj_t value) {
    ncnn_mat_obj_t *self = MP_OBJ_TO_PTR(self_in);
    float val = mp_obj_get_float(value);
    ncnn_mat_fill_float(self->mat, val);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mat_fill_obj, ncnn_mat_fill);

static const mp_rom_map_elem_t ncnn_mat_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_dims), MP_ROM_PTR(&ncnn_mat_dims_obj) },
    { MP_ROM_QSTR(MP_QSTR_w), MP_ROM_PTR(&ncnn_mat_w_obj) },
    { MP_ROM_QSTR(MP_QSTR_h), MP_ROM_PTR(&ncnn_mat_h_obj) },
    { MP_ROM_QSTR(MP_QSTR_c), MP_ROM_PTR(&ncnn_mat_c_obj) },
    { MP_ROM_QSTR(MP_QSTR_fill), MP_ROM_PTR(&ncnn_mat_fill_obj) },
};
static MP_DEFINE_CONST_DICT(ncnn_mat_locals_dict, ncnn_mat_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    ncnn_mat_type,
    MP_QSTR_Mat,
    MP_TYPE_FLAG_NONE,
    make_new, ncnn_mat_make_new,
    print, ncnn_mat_print,
    locals_dict, &ncnn_mat_locals_dict
);

// Net implementation
static mp_obj_t ncnn_net_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 0, 0, false);
    
    ncnn_net_obj_t *self = mp_obj_malloc(ncnn_net_obj_t, &ncnn_net_type);
    self->net = ncnn_net_create();
    
    if (g_default_option) {
        ncnn_net_set_option(self->net, g_default_option);
    }
    
    return MP_OBJ_FROM_PTR(self);
}

static void ncnn_net_print(const mp_print_t *print, mp_obj_t self_in, mp_print_kind_t kind) {
    ncnn_net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int input_count = ncnn_net_get_input_count(self->net);
    int output_count = ncnn_net_get_output_count(self->net);
    mp_printf(print, "Net(%d, %d)", input_count, output_count);
}

#if NCNN_STDIO && NCNN_STRING
static mp_obj_t ncnn_net_load_param_wrapper(mp_obj_t self_in, mp_obj_t path_obj) {
    ncnn_net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    const char *path = mp_obj_str_get_str(path_obj);
    int ret = ncnn_net_load_param(self->net, path);
    return mp_obj_new_int(ret);
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_load_param_obj, ncnn_net_load_param_wrapper);
#endif

#if NCNN_STDIO
static mp_obj_t ncnn_net_load_model_wrapper(mp_obj_t self_in, mp_obj_t path_obj) {
    ncnn_net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    const char *path = mp_obj_str_get_str(path_obj);
    int ret = ncnn_net_load_model(self->net, path);
    return mp_obj_new_int(ret);
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_load_model_obj, ncnn_net_load_model_wrapper);
#endif

static mp_obj_t ncnn_net_create_extractor(mp_obj_t self_in) {
    ncnn_net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    
    ncnn_extractor_obj_t *extractor = mp_obj_malloc(ncnn_extractor_obj_t, &ncnn_extractor_type);
    extractor->extractor = ncnn_extractor_create(self->net);
    
    if (g_default_option) {
        ncnn_extractor_set_option(extractor->extractor, g_default_option);
    }
    
    return MP_OBJ_FROM_PTR(extractor);
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_net_create_extractor_obj, ncnn_net_create_extractor);

static mp_obj_t ncnn_net_set_option_wrapper(mp_obj_t self_in, mp_obj_t option_obj) {
    ncnn_net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ncnn_option_obj_t *option = MP_OBJ_TO_PTR(option_obj);
    ncnn_net_set_option(self->net, option->option);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_set_option_obj, ncnn_net_set_option_wrapper);

static const mp_rom_map_elem_t ncnn_net_locals_dict_table[] = {
#if NCNN_STDIO && NCNN_STRING
    { MP_ROM_QSTR(MP_QSTR_load_param), MP_ROM_PTR(&ncnn_net_load_param_obj) },
#endif
#if NCNN_STDIO
    { MP_ROM_QSTR(MP_QSTR_load_model), MP_ROM_PTR(&ncnn_net_load_model_obj) },
#endif
    { MP_ROM_QSTR(MP_QSTR_create_extractor), MP_ROM_PTR(&ncnn_net_create_extractor_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_option), MP_ROM_PTR(&ncnn_net_set_option_obj) },
};
static MP_DEFINE_CONST_DICT(ncnn_net_locals_dict, ncnn_net_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    ncnn_net_type,
    MP_QSTR_Net,
    MP_TYPE_FLAG_NONE,
    make_new, ncnn_net_make_new,
    print, ncnn_net_print,
    locals_dict, &ncnn_net_locals_dict
);

// Extractor implementation
static void ncnn_extractor_print(const mp_print_t *print, mp_obj_t self_in, mp_print_kind_t kind) {
    mp_printf(print, "Extractor()");
}

static mp_obj_t ncnn_extractor_input_wrapper(mp_obj_t self_in, mp_obj_t index_obj, mp_obj_t mat_obj) {
    ncnn_extractor_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ncnn_mat_obj_t *mat = MP_OBJ_TO_PTR(mat_obj);
    int index = mp_obj_get_int(index_obj);
    int ret = ncnn_extractor_input_index(self->extractor, index, mat->mat);
    return mp_obj_new_int(ret);
}
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_extractor_input_obj, ncnn_extractor_input_wrapper);

static mp_obj_t ncnn_extractor_extract_wrapper(mp_obj_t self_in, mp_obj_t index_obj) {
    ncnn_extractor_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int index = mp_obj_get_int(index_obj);
    
    ncnn_mat_t result_mat;
    int ret = ncnn_extractor_extract_index(self->extractor, index, &result_mat);
    
    if (ret == 0) {
        ncnn_mat_obj_t *result = mp_obj_malloc(ncnn_mat_obj_t, &ncnn_mat_type);
        result->mat = result_mat;
        return MP_OBJ_FROM_PTR(result);
    } else {
        mp_raise_ValueError(MP_ERROR_TEXT("extract failed"));
    }
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_extractor_extract_obj, ncnn_extractor_extract_wrapper);

static const mp_rom_map_elem_t ncnn_extractor_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_input), MP_ROM_PTR(&ncnn_extractor_input_obj) },
    { MP_ROM_QSTR(MP_QSTR_extract), MP_ROM_PTR(&ncnn_extractor_extract_obj) },
};
static MP_DEFINE_CONST_DICT(ncnn_extractor_locals_dict, ncnn_extractor_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    ncnn_extractor_type,
    MP_QSTR_Extractor,
    MP_TYPE_FLAG_NONE,
    print, ncnn_extractor_print,
    locals_dict, &ncnn_extractor_locals_dict
);

// Option implementation
static mp_obj_t ncnn_option_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 0, 0, false);
    
    ncnn_option_obj_t *self = mp_obj_malloc(ncnn_option_obj_t, &ncnn_option_type);
    self->option = ncnn_option_create();
    
    return MP_OBJ_FROM_PTR(self);
}

static void ncnn_option_print(const mp_print_t *print, mp_obj_t self_in, mp_print_kind_t kind) {
    ncnn_option_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int num_threads = ncnn_option_get_num_threads(self->option);
    int use_vulkan = ncnn_option_get_use_vulkan_compute(self->option);
    mp_printf(print, "Option(threads=%d, vulkan=%d)", num_threads, use_vulkan);
}

static mp_obj_t ncnn_option_get_num_threads_wrapper(mp_obj_t self_in) {
    ncnn_option_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_int(ncnn_option_get_num_threads(self->option));
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_option_get_num_threads_obj, ncnn_option_get_num_threads_wrapper);

static mp_obj_t ncnn_option_set_num_threads_wrapper(mp_obj_t self_in, mp_obj_t threads) {
    ncnn_option_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int num_threads = mp_obj_get_int(threads);
    ncnn_option_set_num_threads(self->option, num_threads);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_option_set_num_threads_obj, ncnn_option_set_num_threads_wrapper);

static mp_obj_t ncnn_option_get_use_vulkan_wrapper(mp_obj_t self_in) {
    ncnn_option_obj_t *self = MP_OBJ_TO_PTR(self_in);
    return mp_obj_new_bool(ncnn_option_get_use_vulkan_compute(self->option));
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_option_get_use_vulkan_obj, ncnn_option_get_use_vulkan_wrapper);

static mp_obj_t ncnn_option_set_use_vulkan_wrapper(mp_obj_t self_in, mp_obj_t use_vulkan) {
    ncnn_option_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int vulkan_flag = mp_obj_is_true(use_vulkan);
    ncnn_option_set_use_vulkan_compute(self->option, vulkan_flag);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_option_set_use_vulkan_obj, ncnn_option_set_use_vulkan_wrapper);

static const mp_rom_map_elem_t ncnn_option_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_get_num_threads), MP_ROM_PTR(&ncnn_option_get_num_threads_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_num_threads), MP_ROM_PTR(&ncnn_option_set_num_threads_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_use_vulkan), MP_ROM_PTR(&ncnn_option_get_use_vulkan_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_use_vulkan), MP_ROM_PTR(&ncnn_option_set_use_vulkan_obj) },
};
static MP_DEFINE_CONST_DICT(ncnn_option_locals_dict, ncnn_option_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    ncnn_option_type,
    MP_QSTR_Option,
    MP_TYPE_FLAG_NONE,
    make_new, ncnn_option_make_new,
    print, ncnn_option_print,
    locals_dict, &ncnn_option_locals_dict
);

// Allocator implementation
static mp_obj_t ncnn_allocator_create_pool(void) {
    ncnn_allocator_obj_t *self = mp_obj_malloc(ncnn_allocator_obj_t, &ncnn_allocator_type);
    self->allocator = ncnn_allocator_create_pool_allocator();
    return MP_OBJ_FROM_PTR(self);
}
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_allocator_create_pool_obj, ncnn_allocator_create_pool);

static mp_obj_t ncnn_allocator_create_unlocked_pool(void) {
    ncnn_allocator_obj_t *self = mp_obj_malloc(ncnn_allocator_obj_t, &ncnn_allocator_type);
    self->allocator = ncnn_allocator_create_unlocked_pool_allocator();
    return MP_OBJ_FROM_PTR(self);
}
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_allocator_create_unlocked_pool_obj, ncnn_allocator_create_unlocked_pool);

static void ncnn_allocator_print(const mp_print_t *print, mp_obj_t self_in, mp_print_kind_t kind) {
    mp_printf(print, "Allocator()");
}

static const mp_rom_map_elem_t ncnn_allocator_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_create_pool), MP_ROM_PTR(&ncnn_allocator_create_pool_obj) },
    { MP_ROM_QSTR(MP_QSTR_create_unlocked_pool), MP_ROM_PTR(&ncnn_allocator_create_unlocked_pool_obj) },
};
static MP_DEFINE_CONST_DICT(ncnn_allocator_locals_dict, ncnn_allocator_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    ncnn_allocator_type,
    MP_QSTR_Allocator,
    MP_TYPE_FLAG_NONE,
    print, ncnn_allocator_print,
    locals_dict, &ncnn_allocator_locals_dict
);

// module init
static mp_obj_t ncnn_init(void) {
    if (!g_default_option) {
        g_default_option = ncnn_option_create();
        ncnn_option_set_num_threads(g_default_option, 1);
    }
    if (!g_default_allocator) {
        g_default_allocator = ncnn_allocator_create_pool_allocator();
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_init_obj, ncnn_init);

// module globals
static const mp_rom_map_elem_t ncnn_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_ncnn) },
    { MP_ROM_QSTR(MP_QSTR_version), MP_ROM_PTR(&ncnn_get_version_obj) },
    { MP_ROM_QSTR(MP_QSTR_init), MP_ROM_PTR(&ncnn_init_obj) },
    { MP_ROM_QSTR(MP_QSTR_Net), MP_ROM_PTR(&ncnn_net_type) },
    { MP_ROM_QSTR(MP_QSTR_Mat), MP_ROM_PTR(&ncnn_mat_type) },
    { MP_ROM_QSTR(MP_QSTR_Extractor), MP_ROM_PTR(&ncnn_extractor_type) },
    { MP_ROM_QSTR(MP_QSTR_Option), MP_ROM_PTR(&ncnn_option_type) },
    { MP_ROM_QSTR(MP_QSTR_Allocator), MP_ROM_PTR(&ncnn_allocator_type) },
};
static MP_DEFINE_CONST_DICT(mp_module_ncnn_globals, ncnn_module_globals_table);

const mp_obj_module_t mp_module_ncnn = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t*)&mp_module_ncnn_globals,
};

MP_REGISTER_MODULE(MP_QSTR_ncnn, mp_module_ncnn);