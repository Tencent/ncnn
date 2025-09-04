#include "py/runtime.h"
#include "py/obj.h"
#include "ncnn/c_api.h"

extern const mp_obj_type_t ncnn_mp_type_Allocator;
extern const mp_obj_type_t ncnn_mp_type_Option;
extern const mp_obj_type_t ncnn_mp_type_Mat;
extern const mp_obj_type_t ncnn_mp_type_Blob;
extern const mp_obj_type_t ncnn_mp_type_ParamDict;
extern const mp_obj_type_t ncnn_mp_type_DataReader;
extern const mp_obj_type_t ncnn_mp_type_ModelBin;
extern const mp_obj_type_t ncnn_mp_type_Layer;
extern const mp_obj_type_t ncnn_mp_type_Net;
extern const mp_obj_type_t ncnn_mp_type_Extractor;

typedef struct _ncnn_mp_Allocator_obj_t {
    mp_obj_base_t base;
    ncnn_allocator_t allocator;
} ncnn_mp_Allocator_obj_t;

typedef struct _ncnn_mp_Option_obj_t {
    mp_obj_base_t base;
    ncnn_option_t opt;
    bool is_wrapper;
} ncnn_mp_Option_obj_t;

typedef struct _ncnn_mp_Mat_obj_t {
    mp_obj_base_t base;
    ncnn_mat_t mat;
    bool is_wrapper;
} ncnn_mp_Mat_obj_t;

typedef struct _ncnn_mp_Blob_obj_t {
    mp_obj_base_t base;
    ncnn_blob_t blob;
} ncnn_mp_Blob_obj_t;

typedef struct _ncnn_mp_ParamDict_obj_t {
    mp_obj_base_t base;
    ncnn_paramdict_t pd;
    bool is_wrapper;
} ncnn_mp_ParamDict_obj_t;

typedef struct _ncnn_mp_DataReader_obj_t {
    mp_obj_base_t base;
    ncnn_datareader_t dr;
    mp_obj_t from_memory_obj;
    const unsigned char* mem_ptr;
} ncnn_mp_DataReader_obj_t;

typedef struct _ncnn_mp_ModelBin_obj_t {
    mp_obj_base_t base;
    ncnn_modelbin_t mb;
    bool is_wrapper;
} ncnn_mp_ModelBin_obj_t;

typedef struct _ncnn_mp_Layer_obj_t {
    mp_obj_base_t base;
    ncnn_layer_t layer;
} ncnn_mp_Layer_obj_t;

typedef struct _ncnn_mp_Net_obj_t {
    mp_obj_base_t base;
    ncnn_net_t net;
} ncnn_mp_Net_obj_t;

typedef struct _ncnn_mp_Extractor_obj_t {
    mp_obj_base_t base;
    ncnn_extractor_t ex;
} ncnn_mp_Extractor_obj_t;

static mp_obj_t custom_layer_instances = MP_OBJ_NULL;
static mp_obj_t layer_instance_map = MP_OBJ_NULL;

// ncnn_mp.version()
static mp_obj_t ncnn_mp_version(void) {
    const char* ver_str = ncnn_version();
    return mp_obj_new_str(ver_str, strlen(ver_str));
}
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_mp_version_obj, ncnn_mp_version);

// ------------------
/* allocator api */
// ------------------

// Constructor: Allocator.__new__ and Allocator.__init__
// Usage: Allocator() or Allocator(unlocked=True)
static mp_obj_t ncnn_mp_Allocator_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 0, 1, true);

    // deal with unlocked
    enum { ARG_unlocked };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_unlocked, MP_ARG_KW_ONLY | MP_ARG_BOOL, {.u_bool = false} },
    };
    mp_arg_val_t parsed_args[MP_ARRAY_SIZE(allowed_args)];
    mp_map_t kw_args;
    mp_map_init_fixed_table(&kw_args, n_kw, args + n_args);
    mp_arg_parse_all(0, NULL, &kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, parsed_args);

    ncnn_mp_Allocator_obj_t *self = mp_obj_malloc(ncnn_mp_Allocator_obj_t, type);

    if (parsed_args[ARG_unlocked].u_bool) {
        self->allocator = ncnn_allocator_create_unlocked_pool_allocator();
    } else {
        self->allocator = ncnn_allocator_create_pool_allocator();
    }
    return MP_OBJ_FROM_PTR(self);
}

// Destructor: Allocator.__del__()
// Usage: Auto
static mp_obj_t ncnn_mp_Allocator_deinit(mp_obj_t self_in) {
    ncnn_mp_Allocator_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->allocator) {
        ncnn_allocator_destroy(self->allocator);
        self->allocator = NULL;
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mp_Allocator_deinit_obj, ncnn_mp_Allocator_deinit);

// Allocator.fast_malloc()
// c_api: void* (*fast_malloc)(ncnn_allocator_t allocator, size_t size);
static mp_obj_t ncnn_mp_Allocator_fast_malloc(mp_obj_t self_in, mp_obj_t size_obj) {
    ncnn_mp_Allocator_obj_t *self = MP_OBJ_TO_PTR(self_in);
    size_t size = (size_t)mp_obj_get_int(size_obj);
    void* ptr = self->allocator->fast_malloc(self->allocator, size);
    return mp_obj_new_int_from_ull((uintptr_t)ptr);
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Allocator_fast_malloc_obj, ncnn_mp_Allocator_fast_malloc);

// Allocator.fast_free()
// c_api: void (*fast_free)(ncnn_allocator_t allocator, void* ptr);
static mp_obj_t ncnn_mp_Allocator_fast_free(mp_obj_t self_in, mp_obj_t ptr_obj) {
    ncnn_mp_Allocator_obj_t *self = MP_OBJ_TO_PTR(self_in);
    void* ptr = (void*)mp_obj_get_int(ptr_obj);
    self->allocator->fast_free(self->allocator, ptr);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Allocator_fast_free_obj, ncnn_mp_Allocator_fast_free);

// Allocator class.
static const mp_rom_map_elem_t ncnn_mp_Allocator_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&ncnn_mp_Allocator_deinit_obj) },
    { MP_ROM_QSTR(MP_QSTR_fast_malloc), MP_ROM_PTR(&ncnn_mp_Allocator_fast_malloc_obj) },
    { MP_ROM_QSTR(MP_QSTR_fast_free), MP_ROM_PTR(&ncnn_mp_Allocator_fast_free_obj) },
};
static MP_DEFINE_CONST_DICT(ncnn_mp_Allocator_locals_dict, ncnn_mp_Allocator_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    ncnn_mp_type_Allocator,
    MP_QSTR_Allocator,
    MP_TYPE_FLAG_NONE,
    make_new, ncnn_mp_Allocator_make_new,
    locals_dict, &ncnn_mp_Allocator_locals_dict
);

// ------------------
/* option api */
// ------------------

// Constructor: Option.__new__ and Option.__init__
// Usage: Option()
static mp_obj_t ncnn_mp_Option_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 0, 0, false);
    ncnn_mp_Option_obj_t *self = mp_obj_malloc(ncnn_mp_Option_obj_t, type);
    self->opt = ncnn_option_create();
    self->is_wrapper = false;
    return MP_OBJ_FROM_PTR(self);
}

// Destructor: Option.__del__()
// Usage: Auto
static mp_obj_t ncnn_mp_Option_deinit(mp_obj_t self_in) {
    ncnn_mp_Option_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->opt && !self->is_wrapper) {
        ncnn_option_destroy(self->opt);
        self->opt = NULL;
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mp_Option_deinit_obj, ncnn_mp_Option_deinit);

// Option.set_blob_allocator()
// c_api:  void ncnn_option_set_blob_allocator(ncnn_option_t opt, ncnn_allocator_t allocator);
static mp_obj_t ncnn_mp_Option_set_blob_allocator(mp_obj_t self_in, mp_obj_t allocator_obj) {
    ncnn_mp_Option_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ncnn_allocator_t allocator = NULL;
    if (mp_obj_is_type(allocator_obj, &ncnn_mp_type_Allocator)) {
        allocator = ((ncnn_mp_Allocator_obj_t *)MP_OBJ_TO_PTR(allocator_obj))->allocator;
    }
    ncnn_option_set_blob_allocator(self->opt, allocator);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Option_set_blob_allocator_obj, ncnn_mp_Option_set_blob_allocator);

// Option.set_workspace_allocator()
// c_api:  void ncnn_option_set_workspace_allocator(ncnn_option_t opt, ncnn_allocator_t allocator);
static mp_obj_t ncnn_mp_Option_set_workspace_allocator(mp_obj_t self_in, mp_obj_t allocator_obj) {
    ncnn_mp_Option_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ncnn_allocator_t allocator = NULL;
    if (mp_obj_is_type(allocator_obj, &ncnn_mp_type_Allocator)) {
        allocator = ((ncnn_mp_Allocator_obj_t *)MP_OBJ_TO_PTR(allocator_obj))->allocator;
    }
    ncnn_option_set_workspace_allocator(self->opt, allocator);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Option_set_workspace_allocator_obj, ncnn_mp_Option_set_workspace_allocator);

// Attributes
static void ncnn_mp_Option_attr(mp_obj_t self_in, qstr attr, mp_obj_t *dest) {
    ncnn_mp_Option_obj_t *self = MP_OBJ_TO_PTR(self_in);

    if (dest[0] == MP_OBJ_NULL) {
        // load attribute
        if (attr == MP_QSTR_num_threads) {
            dest[0] = mp_obj_new_int(ncnn_option_get_num_threads(self->opt));
        } else if (attr == MP_QSTR_use_local_pool_allocator) {
            dest[0] = mp_obj_new_int(ncnn_option_get_use_local_pool_allocator(self->opt));
        } else if (attr == MP_QSTR_use_vulkan_compute) {
            dest[0] = mp_obj_new_int(ncnn_option_get_use_vulkan_compute(self->opt));
        } else {
            dest[1] = MP_OBJ_SENTINEL;
        }
    } else if (dest[1] != MP_OBJ_NULL) {
        // store attribute
        if (attr == MP_QSTR_num_threads) {
            ncnn_option_set_num_threads(self->opt, mp_obj_get_int(dest[1]));
            dest[0] = MP_OBJ_NULL;
        } else if (attr == MP_QSTR_use_local_pool_allocator) {
            ncnn_option_set_use_local_pool_allocator(self->opt, mp_obj_get_int(dest[1]));
            dest[0] = MP_OBJ_NULL;
        } else if (attr == MP_QSTR_use_vulkan_compute) {
            ncnn_option_set_use_vulkan_compute(self->opt, mp_obj_get_int(dest[1]));
            dest[0] = MP_OBJ_NULL;
        } else {
            dest[1] = MP_OBJ_SENTINEL;
        }
    }
}

// Option class.
static const mp_rom_map_elem_t ncnn_mp_Option_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&ncnn_mp_Option_deinit_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_blob_allocator), MP_ROM_PTR(&ncnn_mp_Option_set_blob_allocator_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_workspace_allocator), MP_ROM_PTR(&ncnn_mp_Option_set_workspace_allocator_obj) },
};
static MP_DEFINE_CONST_DICT(ncnn_mp_Option_locals_dict, ncnn_mp_Option_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    ncnn_mp_type_Option,
    MP_QSTR_Option,
    MP_TYPE_FLAG_NONE,
    make_new, ncnn_mp_Option_make_new,
    attr, ncnn_mp_Option_attr,
    locals_dict, &ncnn_mp_Option_locals_dict
);

// ------------------
/* mat api */
// ------------------

// Constructor: Mat.__new__ and Mat.__init__
static mp_obj_t ncnn_mp_Mat_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    // Define allowed arguments
    enum { ARG_w, ARG_h, ARG_d, ARG_c, ARG_data, ARG_elemsize, ARG_elempack, ARG_allocator };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_w, MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_h, MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_d, MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_c, MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_data, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
        { MP_QSTR_elemsize, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_elempack, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = 1} },
        { MP_QSTR_allocator, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
    };

    mp_arg_val_t parsed_args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all_kw_array(n_args, n_kw, args, MP_ARRAY_SIZE(allowed_args), allowed_args, parsed_args);

    int w = parsed_args[ARG_w].u_int;
    int h = parsed_args[ARG_h].u_int;
    int d = parsed_args[ARG_d].u_int;
    int c = parsed_args[ARG_c].u_int;
    mp_obj_t data_obj = parsed_args[ARG_data].u_obj;
    size_t elemsize = parsed_args[ARG_elemsize].u_int;
    int elempack = parsed_args[ARG_elempack].u_int;
    mp_obj_t allocator_obj = parsed_args[ARG_allocator].u_obj;

    ncnn_allocator_t allocator = NULL;
    if (mp_obj_is_type(allocator_obj, &ncnn_mp_type_Allocator)) {
        allocator = ((ncnn_mp_Allocator_obj_t *)MP_OBJ_TO_PTR(allocator_obj))->allocator;
    }

    ncnn_mat_t mat = NULL;
    void *data_ptr = NULL;
    if (data_obj != mp_const_none) {
        mp_buffer_info_t bufinfo;
        mp_get_buffer_raise(data_obj, &bufinfo, MP_BUFFER_READ);
        data_ptr = bufinfo.buf;
    }

    if (data_ptr) {
        if (elemsize) {
            if (d) mat = ncnn_mat_create_external_4d_elem(w, h, d, c, data_ptr, elemsize, elempack, allocator);
            else if (c) mat = ncnn_mat_create_external_3d_elem(w, h, c, data_ptr, elemsize, elempack, allocator);
            else if (h) mat = ncnn_mat_create_external_2d_elem(w, h, data_ptr, elemsize, elempack, allocator);
            else if (w) mat = ncnn_mat_create_external_1d_elem(w, data_ptr, elemsize, elempack, allocator);
        } else {
            if (d) mat = ncnn_mat_create_external_4d(w, h, d, c, data_ptr, allocator);
            else if (c) mat = ncnn_mat_create_external_3d(w, h, c, data_ptr, allocator);
            else if (h) mat = ncnn_mat_create_external_2d(w, h, data_ptr, allocator);
            else if (w) mat = ncnn_mat_create_external_1d(w, data_ptr, allocator);
        }
    } else {
        if (elemsize) {
            if (d) mat = ncnn_mat_create_4d_elem(w, h, d, c, elemsize, elempack, allocator);
            else if (c) mat = ncnn_mat_create_3d_elem(w, h, c, elemsize, elempack, allocator);
            else if (h) mat = ncnn_mat_create_2d_elem(w, h, elemsize, elempack, allocator);
            else if (w) mat = ncnn_mat_create_1d_elem(w, elemsize, elempack, allocator);
        } else {
            if (d) mat = ncnn_mat_create_4d(w, h, d, c, allocator);
            else if (c) mat = ncnn_mat_create_3d(w, h, c, allocator);
            else if (h) mat = ncnn_mat_create_2d(w, h, allocator);
            else if (w) mat = ncnn_mat_create_1d(w, allocator);
            else mat = ncnn_mat_create();
        }
    }

    if (!mat) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Failed to create ncnn mat"));
    }

    ncnn_mp_Mat_obj_t *self = mp_obj_malloc(ncnn_mp_Mat_obj_t, type);
    self->mat = mat;
    self->is_wrapper = false;
    return MP_OBJ_FROM_PTR(self);
}

// Destructor: Mat.__del__()
// Usage: Auto
static mp_obj_t ncnn_mp_Mat_deinit(mp_obj_t self_in) {
    ncnn_mp_Mat_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->mat && !self->is_wrapper) {
        ncnn_mat_destroy(self->mat);
        self->mat = NULL;
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mp_Mat_deinit_obj, ncnn_mp_Mat_deinit);

// Mat.fill()
// c_api:  void ncnn_mat_fill_float(ncnn_mat_t mat, float v);
static mp_obj_t ncnn_mp_Mat_fill(mp_obj_t self_in, mp_obj_t value_obj) {
    ncnn_mp_Mat_obj_t *self = MP_OBJ_TO_PTR(self_in);
    float value = (float)mp_obj_get_float(value_obj);
    ncnn_mat_fill_float(self->mat, value);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Mat_fill_obj, ncnn_mp_Mat_fill);

// Mat.clone()
// c_api:  ncnn_mat_t ncnn_mat_clone(const ncnn_mat_t mat, ncnn_allocator_t allocator);
static mp_obj_t ncnn_mp_Mat_clone(size_t n_args, const mp_obj_t *args) {
    ncnn_mp_Mat_obj_t *self = MP_OBJ_TO_PTR(args[0]);
    ncnn_allocator_t allocator = NULL;
    if (n_args == 2 && mp_obj_is_type(args[1], &ncnn_mp_type_Allocator)) {
        allocator = ((ncnn_mp_Allocator_obj_t *)MP_OBJ_TO_PTR(args[1]))->allocator;
    }
    ncnn_mat_t cloned_mat = ncnn_mat_clone(self->mat, allocator);
    if (!cloned_mat) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Failed to clone ncnn mat"));
    }
    ncnn_mp_Mat_obj_t *cloned_self = mp_obj_malloc(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
    cloned_self->mat = cloned_mat;
    cloned_self->is_wrapper = false;
    return MP_OBJ_FROM_PTR(cloned_self);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mp_Mat_clone_obj, 1, 2, ncnn_mp_Mat_clone);

// Mat.reshape()
// c_api:  ncnn_mat_t ncnn_mat_reshape_xd(...)
static mp_obj_t ncnn_mp_Mat_reshape(size_t n_args, const mp_obj_t *args) {
    ncnn_mp_Mat_obj_t *self = MP_OBJ_TO_PTR(args[0]);
    ncnn_allocator_t allocator = NULL;
    int shape_dims = n_args - 1;
    if (mp_obj_is_type(args[n_args - 1], &ncnn_mp_type_Allocator)) {
        allocator = ((ncnn_mp_Allocator_obj_t *)MP_OBJ_TO_PTR(args[n_args - 1]))->allocator;
        shape_dims--;
    }

    ncnn_mat_t reshaped_mat = NULL;
    if (shape_dims == 1) {
        reshaped_mat = ncnn_mat_reshape_1d(self->mat, mp_obj_get_int(args[1]), allocator);
    } else if (shape_dims == 2) {
        reshaped_mat = ncnn_mat_reshape_2d(self->mat, mp_obj_get_int(args[1]), mp_obj_get_int(args[2]), allocator);
    } else if (shape_dims == 3) {
        reshaped_mat = ncnn_mat_reshape_3d(self->mat, mp_obj_get_int(args[1]), mp_obj_get_int(args[2]), mp_obj_get_int(args[3]), allocator);
    } else if (shape_dims == 4) {
        reshaped_mat = ncnn_mat_reshape_4d(self->mat, mp_obj_get_int(args[1]), mp_obj_get_int(args[2]), mp_obj_get_int(args[3]), mp_obj_get_int(args[4]), allocator);
    }
    if (!reshaped_mat) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Failed to reshape ncnn mat"));
    }

    ncnn_mp_Mat_obj_t *reshaped_self = mp_obj_malloc(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
    reshaped_self->mat = reshaped_mat;
    reshaped_self->is_wrapper = false;
    return MP_OBJ_FROM_PTR(reshaped_self);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR(ncnn_mp_Mat_reshape_obj, 2, ncnn_mp_Mat_reshape);

// Mat.flatten()
static mp_obj_t ncnn_mp_Mat_flatten(mp_obj_t self_in, mp_obj_t opt_obj) {
    ncnn_mat_t src = ((ncnn_mp_Mat_obj_t*)MP_OBJ_TO_PTR(self_in))->mat;
    ncnn_mp_Mat_obj_t *dst = mp_obj_malloc(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
    dst->mat = NULL;
    ncnn_option_t opt = ((ncnn_mp_Option_obj_t*)MP_OBJ_TO_PTR(opt_obj))->opt;

    ncnn_flatten(src, &dst->mat, opt);

    if (!dst->mat) {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Mat.flatten failed"));
    }
    dst->is_wrapper = false;
    return MP_OBJ_FROM_PTR(dst);
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Mat_flatten_obj, ncnn_mp_Mat_flatten);

// Mat.convert_packing()
static mp_obj_t ncnn_mp_Mat_convert_packing(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    enum { ARG_elempack, ARG_opt };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_elempack, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_opt, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
    };
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args - 1, pos_args + 1, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    ncnn_mat_t src = ((ncnn_mp_Mat_obj_t*)MP_OBJ_TO_PTR(pos_args[0]))->mat;
    ncnn_mp_Mat_obj_t *dst = mp_obj_malloc(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
    dst->mat = NULL;
    int elempack = args[ARG_elempack].u_int;
    ncnn_option_t opt = ((ncnn_mp_Option_obj_t*)MP_OBJ_TO_PTR(args[ARG_opt].u_obj))->opt;

    ncnn_convert_packing(src, &dst->mat, elempack, opt);

    if (!dst->mat) {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Mat.convert_packing failed"));
    }
    dst->is_wrapper = false;
    return MP_OBJ_FROM_PTR(dst);
}
static MP_DEFINE_CONST_FUN_OBJ_KW(ncnn_mp_Mat_convert_packing_obj, 1, ncnn_mp_Mat_convert_packing);

// Attributes: get methods
static void ncnn_mp_Mat_attr(mp_obj_t self_in, qstr attr, mp_obj_t *dest) {
    if (dest[0] != MP_OBJ_NULL) {
        return;
    }
    ncnn_mp_Mat_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (attr == MP_QSTR_dims) {
        dest[0] = mp_obj_new_int(ncnn_mat_get_dims(self->mat));
    } else if (attr == MP_QSTR_w) {
        dest[0] = mp_obj_new_int(ncnn_mat_get_w(self->mat));
    } else if (attr == MP_QSTR_h) {
        dest[0] = mp_obj_new_int(ncnn_mat_get_h(self->mat));
    } else if (attr == MP_QSTR_d) {
        dest[0] = mp_obj_new_int(ncnn_mat_get_d(self->mat));
    } else if (attr == MP_QSTR_c) {
        dest[0] = mp_obj_new_int(ncnn_mat_get_c(self->mat));
    } else if (attr == MP_QSTR_elemsize) {
        dest[0] = mp_obj_new_int(ncnn_mat_get_elemsize(self->mat));
    } else if (attr == MP_QSTR_elempack) {
        dest[0] = mp_obj_new_int(ncnn_mat_get_elempack(self->mat));
    } else if (attr == MP_QSTR_cstep) {
        dest[0] = mp_obj_new_int(ncnn_mat_get_cstep(self->mat));
    } else if (attr == MP_QSTR_data) {
        // pointer to the data; TODO
        dest[0] = mp_obj_new_int_from_ull((uintptr_t)ncnn_mat_get_data(self->mat));
    } else {
        dest[1] = MP_OBJ_SENTINEL;
    }
}

// Mat.get_channel_data(c)
// c_api:  void* ncnn_mat_get_channel_data(const ncnn_mat_t mat, int c);
static mp_obj_t ncnn_mp_Mat_get_channel_data(mp_obj_t self_in, mp_obj_t channel_obj) {
    ncnn_mp_Mat_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int channel = mp_obj_get_int(channel_obj);
    void *channel_data_ptr = ncnn_mat_get_channel_data(self->mat, channel);
    return mp_obj_new_int_from_ull((uintptr_t)channel_data_ptr);
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Mat_get_channel_data_obj, ncnn_mp_Mat_get_channel_data);

#if NCNN_PIXEL
// Mat.from_pixels()
static mp_obj_t ncnn_mp_Mat_from_pixels(size_t n_args, const mp_obj_t *args) {
    // args[0] is self, skip
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[1], &bufinfo, MP_BUFFER_READ);
    int type = mp_obj_get_int(args[2]);
    int w = mp_obj_get_int(args[3]);
    int h = mp_obj_get_int(args[4]);
    int stride = mp_obj_get_int(args[5]);
    ncnn_allocator_t allocator = NULL;
    if (n_args == 7 && mp_obj_is_type(args[6], &ncnn_mp_type_Allocator)) {
        allocator = ((ncnn_mp_Allocator_obj_t *)MP_OBJ_TO_PTR(args[6]))->allocator;
    }
    ncnn_mat_t mat = ncnn_mat_from_pixels((const unsigned char*)bufinfo.buf, type, w, h, stride, allocator);
    if (!mat) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Failed to create mat from pixels"));
    }
    ncnn_mp_Mat_obj_t *self = mp_obj_malloc(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
    self->mat = mat;
    self->is_wrapper = false;
    return MP_OBJ_FROM_PTR(self);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mp_Mat_from_pixels_obj, 6, 7, ncnn_mp_Mat_from_pixels);
MP_DEFINE_CONST_CLASSMETHOD_OBJ(ncnn_mp_Mat_from_pixels_classmethod_obj, MP_ROM_PTR(&ncnn_mp_Mat_from_pixels_obj));

// Mat.from_pixels_resize()
static mp_obj_t ncnn_mp_Mat_from_pixels_resize(size_t n_args, const mp_obj_t *args) {
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[1], &bufinfo, MP_BUFFER_READ);
    int type = mp_obj_get_int(args[2]);
    int w = mp_obj_get_int(args[3]);
    int h = mp_obj_get_int(args[4]);
    int stride = mp_obj_get_int(args[5]);
    int target_width = mp_obj_get_int(args[6]);
    int target_height = mp_obj_get_int(args[7]);
    ncnn_allocator_t allocator = NULL;
    if (n_args == 9 && mp_obj_is_type(args[8], &ncnn_mp_type_Allocator)) {
        allocator = ((ncnn_mp_Allocator_obj_t *)MP_OBJ_TO_PTR(args[8]))->allocator;
    }
    ncnn_mat_t mat = ncnn_mat_from_pixels_resize((const unsigned char*)bufinfo.buf, type, w, h, stride, target_width, target_height, allocator);
    if (!mat) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Failed to create mat from pixels resize"));
    }
    ncnn_mp_Mat_obj_t *self = mp_obj_malloc(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
    self->mat = mat;
    return MP_OBJ_FROM_PTR(self);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mp_Mat_from_pixels_resize_obj, 8, 9, ncnn_mp_Mat_from_pixels_resize);
MP_DEFINE_CONST_CLASSMETHOD_OBJ(ncnn_mp_Mat_from_pixels_resize_classmethod_obj, MP_ROM_PTR(&ncnn_mp_Mat_from_pixels_resize_obj));

// Mat.from_pixels_roi()
static mp_obj_t ncnn_mp_Mat_from_pixels_roi(size_t n_args, const mp_obj_t *args) {
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[1], &bufinfo, MP_BUFFER_READ);
    int type = mp_obj_get_int(args[2]);
    int w = mp_obj_get_int(args[3]);
    int h = mp_obj_get_int(args[4]);
    int stride = mp_obj_get_int(args[5]);
    int roix = mp_obj_get_int(args[6]);
    int roiy = mp_obj_get_int(args[7]);
    int roiw = mp_obj_get_int(args[8]);
    int roih = mp_obj_get_int(args[9]);
    ncnn_allocator_t allocator = NULL;
    if (n_args == 11 && mp_obj_is_type(args[10], &ncnn_mp_type_Allocator)) {
        allocator = ((ncnn_mp_Allocator_obj_t *)MP_OBJ_TO_PTR(args[10]))->allocator;
    }
    ncnn_mat_t mat = ncnn_mat_from_pixels_roi((const unsigned char*)bufinfo.buf, type, w, h, stride, roix, roiy, roiw, roih, allocator);
    if (!mat) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Failed to create mat from pixels roi"));
    }
    ncnn_mp_Mat_obj_t *self = mp_obj_malloc(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
    self->mat = mat;
    return MP_OBJ_FROM_PTR(self);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mp_Mat_from_pixels_roi_obj, 10, 11, ncnn_mp_Mat_from_pixels_roi);
MP_DEFINE_CONST_CLASSMETHOD_OBJ(ncnn_mp_Mat_from_pixels_roi_classmethod_obj, MP_ROM_PTR(&ncnn_mp_Mat_from_pixels_roi_obj));

// Mat.from_pixels_roi_resize()
static mp_obj_t ncnn_mp_Mat_from_pixels_roi_resize(size_t n_args, const mp_obj_t *args) {
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[1], &bufinfo, MP_BUFFER_READ);
    int type = mp_obj_get_int(args[2]);
    int w = mp_obj_get_int(args[3]);
    int h = mp_obj_get_int(args[4]);
    int stride = mp_obj_get_int(args[5]);
    int roix = mp_obj_get_int(args[6]);
    int roiy = mp_obj_get_int(args[7]);
    int roiw = mp_obj_get_int(args[8]);
    int roih = mp_obj_get_int(args[9]);
    int target_width = mp_obj_get_int(args[10]);
    int target_height = mp_obj_get_int(args[11]);
    ncnn_allocator_t allocator = NULL;
    if (n_args == 13 && mp_obj_is_type(args[12], &ncnn_mp_type_Allocator)) {
        allocator = ((ncnn_mp_Allocator_obj_t *)MP_OBJ_TO_PTR(args[12]))->allocator;
    }
    ncnn_mat_t mat = ncnn_mat_from_pixels_roi_resize((const unsigned char*)bufinfo.buf, type, w, h, stride, roix, roiy, roiw, roih, target_width, target_height, allocator);
    if (!mat) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Failed to create mat from pixels roi resize"));
    }
    ncnn_mp_Mat_obj_t *self = mp_obj_malloc(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
    self->mat = mat;
    return MP_OBJ_FROM_PTR(self);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mp_Mat_from_pixels_roi_resize_obj, 12, 13, ncnn_mp_Mat_from_pixels_roi_resize);
MP_DEFINE_CONST_CLASSMETHOD_OBJ(ncnn_mp_Mat_from_pixels_roi_resize_classmethod_obj, MP_ROM_PTR(&ncnn_mp_Mat_from_pixels_roi_resize_obj));

// Mat.to_pixels()
static mp_obj_t ncnn_mp_Mat_to_pixels(size_t n_args, const mp_obj_t *args) {
    ncnn_mp_Mat_obj_t *self = MP_OBJ_TO_PTR(args[0]);
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[1], &bufinfo, MP_BUFFER_WRITE);
    int type = mp_obj_get_int(args[2]);
    int stride = mp_obj_get_int(args[3]);
    ncnn_mat_to_pixels(self->mat, (unsigned char*)bufinfo.buf, type, stride);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mp_Mat_to_pixels_obj, 4, 4, ncnn_mp_Mat_to_pixels);

// Mat.to_pixels_resize()
static mp_obj_t ncnn_mp_Mat_to_pixels_resize(size_t n_args, const mp_obj_t *args) {
    ncnn_mp_Mat_obj_t *self = MP_OBJ_TO_PTR(args[0]);
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[1], &bufinfo, MP_BUFFER_WRITE);
    int type = mp_obj_get_int(args[2]);
    int target_width = mp_obj_get_int(args[3]);
    int target_height = mp_obj_get_int(args[4]);
    int target_stride = mp_obj_get_int(args[5]);
    ncnn_mat_to_pixels_resize(self->mat, (unsigned char*)bufinfo.buf, type, target_width, target_height, target_stride);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mp_Mat_to_pixels_resize_obj, 6, 6, ncnn_mp_Mat_to_pixels_resize);
#endif // NCNN_PIXEL

// Mat.substract_mean_normalize()
static mp_obj_t ncnn_mp_Mat_substract_mean_normalize(mp_obj_t self_in, mp_obj_t mean_obj, mp_obj_t norm_obj) {
    ncnn_mp_Mat_obj_t *self = MP_OBJ_TO_PTR(self_in);

    const float* mean_vals = NULL;
    if (mean_obj != mp_const_none) {
        mp_buffer_info_t mean_buf;
        mp_get_buffer_raise(mean_obj, &mean_buf, MP_BUFFER_READ);
        mean_vals = (const float*)mean_buf.buf;
    }
    const float* norm_vals = NULL;
    if (norm_obj != mp_const_none) {
        mp_buffer_info_t norm_buf;
        mp_get_buffer_raise(norm_obj, &norm_buf, MP_BUFFER_READ);
        norm_vals = (const float*)norm_buf.buf;
    }

    ncnn_mat_substract_mean_normalize(self->mat, mean_vals, norm_vals);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_mp_Mat_substract_mean_normalize_obj, ncnn_mp_Mat_substract_mean_normalize);

// Mat.from_bytes()
static mp_obj_t ncnn_mp_Mat_from_bytes(mp_obj_t self_in, mp_obj_t data_obj) {
    ncnn_mp_Mat_obj_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t src_bufinfo;
    mp_get_buffer_raise(data_obj, &src_bufinfo, MP_BUFFER_READ);

    void* dest_ptr = ncnn_mat_get_data(self->mat);
    if (dest_ptr == NULL) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Mat.from_bytes failed: Cannot write to an empty Mat"));
    }

    int c = ncnn_mat_get_c(self->mat);
    size_t cstep = ncnn_mat_get_cstep(self->mat);
    size_t elemsize = ncnn_mat_get_elemsize(self->mat);
    size_t total_dest_size = c * cstep * elemsize;

    if (src_bufinfo.len != total_dest_size) {
        mp_raise_msg_varg(&mp_type_ValueError,  
            MP_ERROR_TEXT("Mat.from_bytes failed: Source buffer size (%d) does not match Mat's data size (%d)"),   
            src_bufinfo.len, total_dest_size);
    }
    memcpy(dest_ptr, src_bufinfo.buf, total_dest_size);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Mat_from_bytes_obj, ncnn_mp_Mat_from_bytes);

// Mat.to_bytes()
static mp_obj_t ncnn_mp_Mat_to_bytes(mp_obj_t self_in) {
    ncnn_mp_Mat_obj_t *self = MP_OBJ_TO_PTR(self_in);
    const void* data_ptr = ncnn_mat_get_data(self->mat);

    if (data_ptr == NULL) {
        // empty mat
        return mp_const_empty_bytes;
    }

    int c = ncnn_mat_get_c(self->mat);
    size_t cstep = ncnn_mat_get_cstep(self->mat);
    size_t elemsize = ncnn_mat_get_elemsize(self->mat);
    size_t total_size = c * cstep * elemsize;
    
    return mp_obj_new_bytes(data_ptr, total_size);
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mp_Mat_to_bytes_obj, ncnn_mp_Mat_to_bytes);

// ------------------
/* mat process api */
// ------------------

// Mat.copy_make_border()
static mp_obj_t ncnn_mp_Mat_copy_make_border(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    mp_obj_t self_obj = pos_args[0];

    enum { ARG_top, ARG_bottom, ARG_left, ARG_right, ARG_front, ARG_behind, ARG_type, ARG_v, ARG_opt };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_top, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_bottom, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_left, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_right, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_front, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = -1} },  // 3d require
        { MP_QSTR_behind, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = -1} },  // 3d require
        { MP_QSTR_type, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_v, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
        { MP_QSTR_opt, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args - 1, pos_args + 1, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    ncnn_mat_t src = ((ncnn_mp_Mat_obj_t*)MP_OBJ_TO_PTR(self_obj))->mat;
    int top = args[ARG_top].u_int;
    int bottom = args[ARG_bottom].u_int;
    int left = args[ARG_left].u_int;
    int right = args[ARG_right].u_int;
    int front = args[ARG_front].u_int;
    int behind = args[ARG_behind].u_int;
    int type = args[ARG_type].u_int;
    float v = (float)mp_obj_get_float(args[ARG_v].u_obj);
    ncnn_option_t opt = ((ncnn_mp_Option_obj_t*)MP_OBJ_TO_PTR(args[ARG_opt].u_obj))->opt;

    ncnn_mp_Mat_obj_t *dst_obj = mp_obj_malloc(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
    dst_obj->mat = ncnn_mat_create();

    if (front != -1 && behind != -1) {
        ncnn_copy_make_border_3d(src, dst_obj->mat, top, bottom, left, right, front, behind, type, v, opt);
    } else {
        ncnn_copy_make_border(src, dst_obj->mat, top, bottom, left, right, type, v, opt);
    }
    dst_obj->is_wrapper = false;
    return MP_OBJ_FROM_PTR(dst_obj);
}
static MP_DEFINE_CONST_FUN_OBJ_KW(ncnn_mp_Mat_copy_make_border_obj, 1, ncnn_mp_Mat_copy_make_border);

// Mat.copy_cut_border()
static mp_obj_t ncnn_mp_Mat_copy_cut_border(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    mp_obj_t self_obj = pos_args[0];

    enum { ARG_top, ARG_bottom, ARG_left, ARG_right, ARG_front, ARG_behind, ARG_opt };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_top, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_bottom, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_left, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_right, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_front, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = -1} },
        { MP_QSTR_behind, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = -1} },
        { MP_QSTR_opt, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args - 1, pos_args + 1, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    ncnn_mat_t src = ((ncnn_mp_Mat_obj_t*)MP_OBJ_TO_PTR(self_obj))->mat;
    int top = args[ARG_top].u_int;
    int bottom = args[ARG_bottom].u_int;
    int left = args[ARG_left].u_int;
    int right = args[ARG_right].u_int;
    int front = args[ARG_front].u_int;
    int behind = args[ARG_behind].u_int;
    ncnn_option_t opt = ((ncnn_mp_Option_obj_t*)MP_OBJ_TO_PTR(args[ARG_opt].u_obj))->opt;

    ncnn_mp_Mat_obj_t *dst_obj = mp_obj_malloc(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
    dst_obj->mat = ncnn_mat_create();

    if (front != -1 && behind != -1) {
        ncnn_copy_cut_border_3d(src, dst_obj->mat, top, bottom, left, right, front, behind, opt);
    } else {
        ncnn_copy_cut_border(src, dst_obj->mat, top, bottom, left, right, opt);
    }
    dst_obj->is_wrapper = false;
    return MP_OBJ_FROM_PTR(dst_obj);
}
static MP_DEFINE_CONST_FUN_OBJ_KW(ncnn_mp_Mat_copy_cut_border_obj, 1, ncnn_mp_Mat_copy_cut_border);

// ------------------------
/* mat pixel drawing api*/
// ------------------------
#if NCNN_PIXEL_DRAWING
// Mat.draw_rectangle()
static mp_obj_t ncnn_mp_Mat_draw_rectangle(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    ncnn_mp_Mat_obj_t *self = MP_OBJ_TO_PTR(pos_args[0]);

    enum { ARG_rect, ARG_color, ARG_thickness };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_rect, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
        { MP_QSTR_color, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_thickness, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 1} },
    };
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args - 1, pos_args + 1, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    mp_obj_t *rect_items;
    size_t rect_len;
    mp_obj_get_array(args[ARG_rect].u_obj, &rect_len, &rect_items);
    if (rect_len != 4) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Rectangle must be like (x, y, w, h)"));
    }

    int rx = mp_obj_get_int(rect_items[0]);
    int ry = mp_obj_get_int(rect_items[1]);
    int rw = mp_obj_get_int(rect_items[2]);
    int rh = mp_obj_get_int(rect_items[3]);
    unsigned int color = args[ARG_color].u_int;
    int thickness = args[ARG_thickness].u_int;

    int channels = ncnn_mat_get_c(self->mat);
    int w = ncnn_mat_get_w(self->mat);
    int h = ncnn_mat_get_h(self->mat);
    unsigned char* pixels = (unsigned char*)ncnn_mat_get_data(self->mat);

    if (channels == 1)      ncnn_draw_rectangle_c1(pixels, w, h, rx, ry, rw, rh, color, thickness);
    else if (channels == 2) ncnn_draw_rectangle_c2(pixels, w, h, rx, ry, rw, rh, color, thickness);
    else if (channels == 3) ncnn_draw_rectangle_c3(pixels, w, h, rx, ry, rw, rh, color, thickness);
    else if (channels == 4) ncnn_draw_rectangle_c4(pixels, w, h, rx, ry, rw, rh, color, thickness);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_KW(ncnn_mp_Mat_draw_rectangle_obj, 1, ncnn_mp_Mat_draw_rectangle);

// Mat.draw_text()
static mp_obj_t ncnn_mp_Mat_draw_text(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    ncnn_mp_Mat_obj_t *self = MP_OBJ_TO_PTR(pos_args[0]);

    enum { ARG_text, ARG_origin, ARG_font_size, ARG_color };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_text, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
        { MP_QSTR_origin, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
        { MP_QSTR_font_size, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_color, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
    };
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args - 1, pos_args + 1, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    const char* text = mp_obj_str_get_str(args[ARG_text].u_obj);
    mp_obj_t *origin_items;
    size_t origin_len;
    mp_obj_get_array(args[ARG_origin].u_obj, &origin_len, &origin_items);
    if (origin_len != 2) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Origin must be like (x, y)"));
    }
    int x = mp_obj_get_int(origin_items[0]);
    int y = mp_obj_get_int(origin_items[1]);
    int fontpixelsize = args[ARG_font_size].u_int;
    unsigned int color = args[ARG_color].u_int;

    int channels = ncnn_mat_get_c(self->mat);
    int w = ncnn_mat_get_w(self->mat);
    int h = ncnn_mat_get_h(self->mat);
    unsigned char* pixels = (unsigned char*)ncnn_mat_get_data(self->mat);

    if (channels == 1)      ncnn_draw_text_c1(pixels, w, h, text, x, y, fontpixelsize, color);
    else if (channels == 2) ncnn_draw_text_c2(pixels, w, h, text, x, y, fontpixelsize, color);
    else if (channels == 3) ncnn_draw_text_c3(pixels, w, h, text, x, y, fontpixelsize, color);
    else if (channels == 4) ncnn_draw_text_c4(pixels, w, h, text, x, y, fontpixelsize, color);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_KW(ncnn_mp_Mat_draw_text_obj, 1, ncnn_mp_Mat_draw_text);

// Mat.draw_circle()
static mp_obj_t ncnn_mp_Mat_draw_circle(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    ncnn_mp_Mat_obj_t *self = MP_OBJ_TO_PTR(pos_args[0]);

    enum { ARG_center, ARG_radius, ARG_color, ARG_thickness };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_center, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
        { MP_QSTR_radius, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_color, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_thickness, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 1} },
    };
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args - 1, pos_args + 1, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    mp_obj_t *center_items;
    size_t center_len;
    mp_obj_get_array(args[ARG_center].u_obj, &center_len, &center_items);
    if (center_len != 2) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Center Point must be like (x, y)"));
    }
    int cx = mp_obj_get_int(center_items[0]);
    int cy = mp_obj_get_int(center_items[1]);
    int radius = args[ARG_radius].u_int;
    unsigned int color = args[ARG_color].u_int;
    int thickness = args[ARG_thickness].u_int;

    int channels = ncnn_mat_get_c(self->mat);
    int w = ncnn_mat_get_w(self->mat);
    int h = ncnn_mat_get_h(self->mat);
    unsigned char* pixels = (unsigned char*)ncnn_mat_get_data(self->mat);

    if (channels == 1)      ncnn_draw_circle_c1(pixels, w, h, cx, cy, radius, color, thickness);
    else if (channels == 2) ncnn_draw_circle_c2(pixels, w, h, cx, cy, radius, color, thickness);
    else if (channels == 3) ncnn_draw_circle_c3(pixels, w, h, cx, cy, radius, color, thickness);
    else if (channels == 4) ncnn_draw_circle_c4(pixels, w, h, cx, cy, radius, color, thickness);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_KW(ncnn_mp_Mat_draw_circle_obj, 1, ncnn_mp_Mat_draw_circle);

// Mat.draw_line()
static mp_obj_t ncnn_mp_Mat_draw_line(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    ncnn_mp_Mat_obj_t *self = MP_OBJ_TO_PTR(pos_args[0]);

    enum { ARG_pt1, ARG_pt2, ARG_color, ARG_thickness };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_pt1, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
        { MP_QSTR_pt2, MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
        { MP_QSTR_color, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 0} },
        { MP_QSTR_thickness, MP_ARG_REQUIRED | MP_ARG_INT, {.u_int = 1} },
    };
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args - 1, pos_args + 1, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    mp_obj_t *pt1_items; size_t pt1_len;
    mp_obj_get_array(args[ARG_pt1].u_obj, &pt1_len, &pt1_items);
    mp_obj_t *pt2_items; size_t pt2_len;
    mp_obj_get_array(args[ARG_pt2].u_obj, &pt2_len, &pt2_items);
    if (pt1_len != 2 || pt2_len != 2) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Point1 and Point2 must be like (x, y)"));
    }
    int x0 = mp_obj_get_int(pt1_items[0]);
    int y0 = mp_obj_get_int(pt1_items[1]);
    int x1 = mp_obj_get_int(pt2_items[0]);
    int y1 = mp_obj_get_int(pt2_items[1]);
    unsigned int color = args[ARG_color].u_int;
    int thickness = args[ARG_thickness].u_int;

    int channels = ncnn_mat_get_c(self->mat);
    int w = ncnn_mat_get_w(self->mat);
    int h = ncnn_mat_get_h(self->mat);
    unsigned char* pixels = (unsigned char*)ncnn_mat_get_data(self->mat);

    if (channels == 1)      ncnn_draw_line_c1(pixels, w, h, x0, y0, x1, y1, color, thickness);
    else if (channels == 2) ncnn_draw_line_c2(pixels, w, h, x0, y0, x1, y1, color, thickness);
    else if (channels == 3) ncnn_draw_line_c3(pixels, w, h, x0, y0, x1, y1, color, thickness);
    else if (channels == 4) ncnn_draw_line_c4(pixels, w, h, x0, y0, x1, y1, color, thickness);

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_KW(ncnn_mp_Mat_draw_line_obj, 1, ncnn_mp_Mat_draw_line);
#endif // NCNN_PIXEL_DRAWING

// Mat class.
static const mp_rom_map_elem_t ncnn_mp_Mat_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&ncnn_mp_Mat_deinit_obj) },
    { MP_ROM_QSTR(MP_QSTR_fill), MP_ROM_PTR(&ncnn_mp_Mat_fill_obj) },
    { MP_ROM_QSTR(MP_QSTR_clone), MP_ROM_PTR(&ncnn_mp_Mat_clone_obj) },
    { MP_ROM_QSTR(MP_QSTR_reshape), MP_ROM_PTR(&ncnn_mp_Mat_reshape_obj) },
    { MP_ROM_QSTR(MP_QSTR_flatten), MP_ROM_PTR(&ncnn_mp_Mat_flatten_obj) },
    { MP_ROM_QSTR(MP_QSTR_convert_packing), MP_ROM_PTR(&ncnn_mp_Mat_convert_packing_obj) },
    { MP_ROM_QSTR(MP_QSTR_substract_mean_normalize), MP_ROM_PTR(&ncnn_mp_Mat_substract_mean_normalize_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_channel_data), MP_ROM_PTR(&ncnn_mp_Mat_get_channel_data_obj) },
    { MP_ROM_QSTR(MP_QSTR_from_bytes), MP_ROM_PTR(&ncnn_mp_Mat_from_bytes_obj) },
    { MP_ROM_QSTR(MP_QSTR_to_bytes), MP_ROM_PTR(&ncnn_mp_Mat_to_bytes_obj) },
#if NCNN_PIXEL
    { MP_ROM_QSTR(MP_QSTR_to_pixels), MP_ROM_PTR(&ncnn_mp_Mat_to_pixels_obj) },
    { MP_ROM_QSTR(MP_QSTR_to_pixels_resize), MP_ROM_PTR(&ncnn_mp_Mat_to_pixels_resize_obj) },
    { MP_ROM_QSTR(MP_QSTR_from_pixels), MP_ROM_PTR(&ncnn_mp_Mat_from_pixels_classmethod_obj) },
    { MP_ROM_QSTR(MP_QSTR_from_pixels_resize), MP_ROM_PTR(&ncnn_mp_Mat_from_pixels_resize_classmethod_obj) },
    { MP_ROM_QSTR(MP_QSTR_from_pixels_roi), MP_ROM_PTR(&ncnn_mp_Mat_from_pixels_roi_classmethod_obj) },
    { MP_ROM_QSTR(MP_QSTR_from_pixels_roi_resize), MP_ROM_PTR(&ncnn_mp_Mat_from_pixels_roi_resize_classmethod_obj) },

    // Macros
    { MP_ROM_QSTR(MP_QSTR_PIXEL_RGB), MP_ROM_INT(1) },
    { MP_ROM_QSTR(MP_QSTR_PIXEL_BGR), MP_ROM_INT(2) },
    { MP_ROM_QSTR(MP_QSTR_PIXEL_GRAY), MP_ROM_INT(3) },
    { MP_ROM_QSTR(MP_QSTR_PIXEL_RGBA), MP_ROM_INT(4) },
    { MP_ROM_QSTR(MP_QSTR_PIXEL_BGRA), MP_ROM_INT(5) },
#endif
    { MP_ROM_QSTR(MP_QSTR_copy_make_border), MP_ROM_PTR(&ncnn_mp_Mat_copy_make_border_obj) },
    { MP_ROM_QSTR(MP_QSTR_copy_cut_border), MP_ROM_PTR(&ncnn_mp_Mat_copy_cut_border_obj) },
#if NCNN_PIXEL_DRAWING
    { MP_ROM_QSTR(MP_QSTR_draw_rectangle), MP_ROM_PTR(&ncnn_mp_Mat_draw_rectangle_obj) },
    { MP_ROM_QSTR(MP_QSTR_draw_text), MP_ROM_PTR(&ncnn_mp_Mat_draw_text_obj) },
    { MP_ROM_QSTR(MP_QSTR_draw_circle), MP_ROM_PTR(&ncnn_mp_Mat_draw_circle_obj) },
    { MP_ROM_QSTR(MP_QSTR_draw_line), MP_ROM_PTR(&ncnn_mp_Mat_draw_line_obj) },
#endif
};
static MP_DEFINE_CONST_DICT(ncnn_mp_Mat_locals_dict, ncnn_mp_Mat_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    ncnn_mp_type_Mat,
    MP_QSTR_Mat,
    MP_TYPE_FLAG_NONE,
    make_new, ncnn_mp_Mat_make_new,
    attr, ncnn_mp_Mat_attr,
    locals_dict, &ncnn_mp_Mat_locals_dict
);

// ------------------
/* blob api */
// ------------------

// Attributes
static void ncnn_mp_Blob_attr(mp_obj_t self_in, qstr attr, mp_obj_t *dest) {
    if (dest[0] != MP_OBJ_NULL) {
        return;
    }
    ncnn_mp_Blob_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (attr == MP_QSTR_name) {
        #if NCNN_STRING
        const char* name = ncnn_blob_get_name(self->blob);
        dest[0] = mp_obj_new_str(name, strlen(name));
        #else
        mp_raise_msg(&mp_type_AttributeError, MP_ERROR_TEXT("Accessing the 'Blob.name' attribute failed: This feature depends on the 'NCNN_STRING' option"));
        #endif
    } else if (attr == MP_QSTR_producer) {
        dest[0] = mp_obj_new_int(ncnn_blob_get_producer(self->blob));
    } else if (attr == MP_QSTR_consumer) {
        dest[0] = mp_obj_new_int(ncnn_blob_get_consumer(self->blob));
    } else if (attr == MP_QSTR_shape) {
        int dims, w, h, c;
        ncnn_blob_get_shape(self->blob, &dims, &w, &h, &c);
        mp_obj_t items[] = {mp_obj_new_int(dims), mp_obj_new_int(w), mp_obj_new_int(h), mp_obj_new_int(c)};
        dest[0] = mp_obj_new_tuple(4, items);
    } else {
        dest[1] = MP_OBJ_SENTINEL;
    }
}

// Blob class.
MP_DEFINE_CONST_OBJ_TYPE(
    ncnn_mp_type_Blob,
    MP_QSTR_Blob,
    MP_TYPE_FLAG_NONE,
    attr, ncnn_mp_Blob_attr
);

// ------------------
/* paramdict api */
// ------------------

// Constructor
// Usage: ParamDict()
static mp_obj_t ncnn_mp_ParamDict_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 0, 0, false);
    ncnn_mp_ParamDict_obj_t *self = mp_obj_malloc(ncnn_mp_ParamDict_obj_t, type);
    self->pd = ncnn_paramdict_create();
    self->is_wrapper = false;
    return MP_OBJ_FROM_PTR(self);
}

// Destructor
// Usage: Auto
static mp_obj_t ncnn_mp_ParamDict_deinit(mp_obj_t self_in) {
    ncnn_mp_ParamDict_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->pd && !self->is_wrapper) {
        ncnn_paramdict_destroy(self->pd);
        self->pd = NULL;
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mp_ParamDict_deinit_obj, ncnn_mp_ParamDict_deinit);

// ParamDict.get_type()
static mp_obj_t ncnn_mp_ParamDict_get_type(mp_obj_t self_in, mp_obj_t id_obj) {
    ncnn_mp_ParamDict_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int id = mp_obj_get_int(id_obj);
    return mp_obj_new_int(ncnn_paramdict_get_type(self->pd, id));
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_ParamDict_get_type_obj, ncnn_mp_ParamDict_get_type);

// ParamDict.get_int()
static mp_obj_t ncnn_mp_ParamDict_get_int(mp_obj_t self_in, mp_obj_t id_obj, mp_obj_t def_obj) {
    ncnn_mp_ParamDict_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int id = mp_obj_get_int(id_obj);
    int def = mp_obj_get_int(def_obj);
    return mp_obj_new_int(ncnn_paramdict_get_int(self->pd, id, def));
}
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_mp_ParamDict_get_int_obj, ncnn_mp_ParamDict_get_int);

// ParamDict.get_float()
static mp_obj_t ncnn_mp_ParamDict_get_float(mp_obj_t self_in, mp_obj_t id_obj, mp_obj_t def_obj) {
    ncnn_mp_ParamDict_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int id = mp_obj_get_int(id_obj);
    float def = (float)mp_obj_get_float(def_obj);
    return mp_obj_new_float(ncnn_paramdict_get_float(self->pd, id, def));
}
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_mp_ParamDict_get_float_obj, ncnn_mp_ParamDict_get_float);

// ParamDict.get_array()
static mp_obj_t ncnn_mp_ParamDict_get_array(mp_obj_t self_in, mp_obj_t id_obj, mp_obj_t def_obj) {
    ncnn_mp_ParamDict_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int id = mp_obj_get_int(id_obj);
    ncnn_mat_t def = ((ncnn_mp_Mat_obj_t*)MP_OBJ_TO_PTR(def_obj))->mat;
    ncnn_mat_t result = ncnn_paramdict_get_array(self->pd, id, def);
    if (!result) {
        return mp_const_none;
    }
    ncnn_mp_Mat_obj_t *mat_self = mp_obj_malloc(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
    mat_self->mat = result;
    return MP_OBJ_FROM_PTR(mat_self);
}
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_mp_ParamDict_get_array_obj, ncnn_mp_ParamDict_get_array);

// ParamDict.set_int()
static mp_obj_t ncnn_mp_ParamDict_set_int(mp_obj_t self_in, mp_obj_t id_obj, mp_obj_t i_obj) {
    ncnn_mp_ParamDict_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int id = mp_obj_get_int(id_obj);
    int i = mp_obj_get_int(i_obj);
    ncnn_paramdict_set_int(self->pd, id, i);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_mp_ParamDict_set_int_obj, ncnn_mp_ParamDict_set_int);

// ParamDict.set_float()
static mp_obj_t ncnn_mp_ParamDict_set_float(mp_obj_t self_in, mp_obj_t id_obj, mp_obj_t f_obj) {
    ncnn_mp_ParamDict_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int id = mp_obj_get_int(id_obj);
    float f = (float)mp_obj_get_float(f_obj);
    ncnn_paramdict_set_float(self->pd, id, f);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_mp_ParamDict_set_float_obj, ncnn_mp_ParamDict_set_float);

// ParamDict.set_array()
static mp_obj_t ncnn_mp_ParamDict_set_array(mp_obj_t self_in, mp_obj_t id_obj, mp_obj_t v_obj) {
    ncnn_mp_ParamDict_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int id = mp_obj_get_int(id_obj);
    ncnn_mat_t v = ((ncnn_mp_Mat_obj_t*)MP_OBJ_TO_PTR(v_obj))->mat;
    ncnn_paramdict_set_array(self->pd, id, v);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_mp_ParamDict_set_array_obj, ncnn_mp_ParamDict_set_array);

// ParamDict class.
static const mp_rom_map_elem_t ncnn_mp_ParamDict_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&ncnn_mp_ParamDict_deinit_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_type), MP_ROM_PTR(&ncnn_mp_ParamDict_get_type_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_int), MP_ROM_PTR(&ncnn_mp_ParamDict_get_int_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_float), MP_ROM_PTR(&ncnn_mp_ParamDict_get_float_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_array), MP_ROM_PTR(&ncnn_mp_ParamDict_get_array_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_int), MP_ROM_PTR(&ncnn_mp_ParamDict_set_int_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_float), MP_ROM_PTR(&ncnn_mp_ParamDict_set_float_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_array), MP_ROM_PTR(&ncnn_mp_ParamDict_set_array_obj) },
};
static MP_DEFINE_CONST_DICT(ncnn_mp_ParamDict_locals_dict, ncnn_mp_ParamDict_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    ncnn_mp_type_ParamDict,
    MP_QSTR_ParamDict,
    MP_TYPE_FLAG_NONE,
    make_new, ncnn_mp_ParamDict_make_new,
    locals_dict, &ncnn_mp_ParamDict_locals_dict
);

// ------------------
/* datareader api */
// ------------------

// Constructor: DataReader.__new__ and DataReader.__init__
// Usage: DataReader(), DataReader(from_memory='...'), or DataReader(from_stdio=...)
static mp_obj_t ncnn_mp_DataReader_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 0, 1, true);

    enum { ARG_from_memory, ARG_from_stdio };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_from_memory, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
        #if NCNN_STDIO
        { MP_QSTR_from_stdio, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
        #endif
    };

    mp_arg_val_t parsed_args[MP_ARRAY_SIZE(allowed_args)];
    mp_map_t kw_args;
    mp_map_init_fixed_table(&kw_args, n_kw, args + n_args);
    mp_arg_parse_all(0, NULL, &kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, parsed_args);
    mp_obj_t from_memory_obj = parsed_args[ARG_from_memory].u_obj;
    #if NCNN_STDIO
    mp_obj_t from_stdio_obj = parsed_args[ARG_from_stdio].u_obj;
    #endif

    ncnn_mp_DataReader_obj_t* self = mp_obj_malloc(ncnn_mp_DataReader_obj_t, type);
    self->from_memory_obj = mp_const_none;
    self->mem_ptr = NULL;
    self->dr = NULL;

    if (from_memory_obj != mp_const_none) {
        self->from_memory_obj = from_memory_obj;
        mp_buffer_info_t bufinfo;
        mp_get_buffer_raise(self->from_memory_obj, &bufinfo, MP_BUFFER_READ);
        self->mem_ptr = (const unsigned char*)bufinfo.buf;
        self->dr = ncnn_datareader_create_from_memory(&self->mem_ptr);
    }
    #if NCNN_STDIO
    else if (from_stdio_obj != mp_const_none) {
        FILE* fp = (FILE*)mp_obj_get_int(from_stdio_obj);
        self->dr = ncnn_datareader_create_from_stdio(fp);
    }
    #endif
    else {
        self->dr = ncnn_datareader_create();
    }

    if (!self->dr) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Failed to create datareader"));
    }

    return MP_OBJ_FROM_PTR(self);
}

// Destructor and other methods remain the same as the previous correct version...
// Destructor: DataReader.__del__()
static mp_obj_t ncnn_mp_DataReader_deinit(mp_obj_t self_in) {
    ncnn_mp_DataReader_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->dr) {
        ncnn_datareader_destroy(self->dr);
        self->dr = NULL;
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mp_DataReader_deinit_obj, ncnn_mp_DataReader_deinit);

// DataReader.scan()
#if NCNN_STRING
static mp_obj_t ncnn_mp_DataReader_scan(mp_obj_t self_in, mp_obj_t format_obj, mp_obj_t data_obj) {
    ncnn_mp_DataReader_obj_t *self = MP_OBJ_TO_PTR(self_in);
    const char* format = mp_obj_str_get_str(format_obj);
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(data_obj, &bufinfo, MP_BUFFER_WRITE);
    return mp_obj_new_int(self->dr->scan(self->dr, format, bufinfo.buf));
}
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_mp_DataReader_scan_obj, ncnn_mp_DataReader_scan);
#endif /* NCNN_STRING */

// DataReader.read()
static mp_obj_t ncnn_mp_DataReader_read(mp_obj_t self_in, mp_obj_t buffer_obj) {
    ncnn_mp_DataReader_obj_t *self = MP_OBJ_TO_PTR(self_in);
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(buffer_obj, &bufinfo, MP_BUFFER_WRITE);
    size_t bytes_read = self->dr->read(self->dr, bufinfo.buf, bufinfo.len);
    return mp_obj_new_int((int)bytes_read);
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_DataReader_read_obj, ncnn_mp_DataReader_read);

// DataReader class.
static const mp_rom_map_elem_t ncnn_mp_DataReader_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&ncnn_mp_DataReader_deinit_obj) },
    { MP_ROM_QSTR(MP_QSTR_read), MP_ROM_PTR(&ncnn_mp_DataReader_read_obj) },
    #if NCNN_STRING
    { MP_ROM_QSTR(MP_QSTR_scan), MP_ROM_PTR(&ncnn_mp_DataReader_scan_obj) },
    #endif
};
static MP_DEFINE_CONST_DICT(ncnn_mp_DataReader_locals_dict, ncnn_mp_DataReader_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    ncnn_mp_type_DataReader,
    MP_QSTR_DataReader,
    MP_TYPE_FLAG_NONE,
    make_new, ncnn_mp_DataReader_make_new,
    locals_dict, &ncnn_mp_DataReader_locals_dict
);

// ------------------
/* modelbin api */
// ------------------

// Constructor: ModelBin.__new__ and ModelBin.__init__ (Refactored)
// Usage: ModelBin(from_datareader=...) or ModelBin(from_mat_array=...)
static mp_obj_t ncnn_mp_ModelBin_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 0, 1, true);

    enum { ARG_from_datareader, ARG_from_mat_array };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_from_datareader, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
        { MP_QSTR_from_mat_array, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
    };

    mp_arg_val_t parsed_args[MP_ARRAY_SIZE(allowed_args)];
    mp_map_t kw_args;
    mp_map_init_fixed_table(&kw_args, n_kw, args + n_args);
    mp_arg_parse_all(0, NULL, &kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, parsed_args);

    ncnn_modelbin_t mb = NULL;
    mp_obj_t dr_obj = parsed_args[ARG_from_datareader].u_obj;
    mp_obj_t mats_obj = parsed_args[ARG_from_mat_array].u_obj;

    if (dr_obj != mp_const_none) {
        ncnn_datareader_t dr = ((ncnn_mp_DataReader_obj_t*)MP_OBJ_TO_PTR(dr_obj))->dr;
        mb = ncnn_modelbin_create_from_datareader(dr);
    } else if (mats_obj != mp_const_none) {
        mp_obj_t *items;
        size_t n;
        mp_obj_get_array(mats_obj, &n, &items);
        ncnn_mat_t* weights = m_new(ncnn_mat_t, n);
        for (size_t i = 0; i < n; i++) {
            weights[i] = ((ncnn_mp_Mat_obj_t*)MP_OBJ_TO_PTR(items[i]))->mat;
        }
        mb = ncnn_modelbin_create_from_mat_array(weights, n);
        m_del(ncnn_mat_t, weights, n);
    } else {
        // create an empty modelbin
        mb = ncnn_modelbin_create_from_mat_array(0, 0);
    }

    if (!mb) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Failed to create modelbin"));
    }

    ncnn_mp_ModelBin_obj_t *self = mp_obj_malloc(ncnn_mp_ModelBin_obj_t, type);
    self->mb = mb;
    self->is_wrapper = false;
    return MP_OBJ_FROM_PTR(self);
}

// Destructor: ModelBin.__del__()
static mp_obj_t ncnn_mp_ModelBin_deinit(mp_obj_t self_in) {
    ncnn_mp_ModelBin_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->mb && !self->is_wrapper) {
        ncnn_modelbin_destroy(self->mb);
        self->mb = NULL;
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mp_ModelBin_deinit_obj, ncnn_mp_ModelBin_deinit);

// ModelBin.load()
static mp_obj_t ncnn_mp_ModelBin_load(size_t n_args, const mp_obj_t *args) {
    ncnn_mp_ModelBin_obj_t *self = MP_OBJ_TO_PTR(args[0]);
    int type = mp_obj_get_int(args[n_args - 1]);
    ncnn_mat_t mat = NULL;

    if (n_args == 3) {
        mat = self->mb->load_1d(self->mb, mp_obj_get_int(args[1]), type);
    } else if (n_args == 4) {
        mat = self->mb->load_2d(self->mb, mp_obj_get_int(args[1]), mp_obj_get_int(args[2]), type);
    } else if (n_args == 5) {
        mat = self->mb->load_3d(self->mb, mp_obj_get_int(args[1]), mp_obj_get_int(args[2]), mp_obj_get_int(args[3]), type);
    }

    if (!mat) {
        return mp_const_none;
    }
    ncnn_mp_Mat_obj_t *mat_self = mp_obj_malloc(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
    mat_self->mat = mat;
    return MP_OBJ_FROM_PTR(mat_self);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mp_ModelBin_load_obj, 3, 5, ncnn_mp_ModelBin_load);

// ModelBin class.
static const mp_rom_map_elem_t ncnn_mp_ModelBin_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&ncnn_mp_ModelBin_deinit_obj) },
    { MP_ROM_QSTR(MP_QSTR_load), MP_ROM_PTR(&ncnn_mp_ModelBin_load_obj) },
};
static MP_DEFINE_CONST_DICT(ncnn_mp_ModelBin_locals_dict, ncnn_mp_ModelBin_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    ncnn_mp_type_ModelBin,
    MP_QSTR_ModelBin,
    MP_TYPE_FLAG_NONE,
    make_new, ncnn_mp_ModelBin_make_new,
    locals_dict, &ncnn_mp_ModelBin_locals_dict
);

// ------------------
/* layer api */
// ------------------

// Constructor: Layer.__new__ and Layer.__init__
// Usage: Layer(), Layer(type="...") or Layer(typeindex=5)
static mp_obj_t ncnn_mp_Layer_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 0, 1, true);

    enum { ARG_type, ARG_typeindex };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_type, MP_ARG_KW_ONLY | MP_ARG_OBJ, {.u_rom_obj = MP_ROM_NONE} },
        { MP_QSTR_typeindex, MP_ARG_KW_ONLY | MP_ARG_INT, {.u_int = -1} },  // layer_idx can't be negative
    };

    mp_arg_val_t parsed_args[MP_ARRAY_SIZE(allowed_args)];
    mp_map_t kw_args;
    mp_map_init_fixed_table(&kw_args, n_kw, args + n_args);
    mp_arg_parse_all(0, NULL, &kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, parsed_args);

    ncnn_layer_t layer = NULL;
    mp_obj_t type_obj = parsed_args[ARG_type].u_obj;
    int typeindex = parsed_args[ARG_typeindex].u_int;

    if (type_obj != mp_const_none) {
        #if NCNN_STRING
        layer = ncnn_layer_create_by_type(mp_obj_str_get_str(type_obj));
        #else
        mp_raise_msg(&mp_type_NotImplementedError, MP_ERROR_TEXT("Creating a Layer with the 'type' keyword failed: This feature depends on the 'NCNN_STRING' option. Or try using the 'typeindex' keyword."));
        #endif
    } else if (typeindex != -1) {
        layer = ncnn_layer_create_by_typeindex(typeindex);
    } else {
        layer = ncnn_layer_create();
    }

    if (!layer) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Failed to create layer"));
    }

    ncnn_mp_Layer_obj_t *self = mp_obj_malloc(ncnn_mp_Layer_obj_t, type);
    self->layer = layer;
    return MP_OBJ_FROM_PTR(self);
}

// Destructor: Layer.__del__()
static mp_obj_t ncnn_mp_Layer_deinit(mp_obj_t self_in) {
    ncnn_mp_Layer_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->layer) {
        ncnn_layer_destroy(self->layer);
        self->layer = NULL;
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mp_Layer_deinit_obj, ncnn_mp_Layer_deinit);

// Layer.type_to_index()
#if NCNN_STRING
static mp_obj_t ncnn_mp_Layer_type_to_index(mp_obj_t type_obj) {
    const char* type = mp_obj_str_get_str(type_obj);
    return mp_obj_new_int(ncnn_layer_type_to_index(type));
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mp_Layer_type_to_index_obj, ncnn_mp_Layer_type_to_index);
#endif // NCNN_STRING

// Layer.get_bottom()
static mp_obj_t ncnn_mp_Layer_get_bottom(mp_obj_t self_in, mp_obj_t i_obj) {
    ncnn_mp_Layer_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int i = mp_obj_get_int(i_obj);
    return mp_obj_new_int(ncnn_layer_get_bottom(self->layer, i));
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Layer_get_bottom_obj, ncnn_mp_Layer_get_bottom);

// Layer.get_top()
static mp_obj_t ncnn_mp_Layer_get_top(mp_obj_t self_in, mp_obj_t i_obj) {
    ncnn_mp_Layer_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int i = mp_obj_get_int(i_obj);
    return mp_obj_new_int(ncnn_layer_get_top(self->layer, i));
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Layer_get_top_obj, ncnn_mp_Layer_get_top);

// Layer.get_bottom_shape()
static mp_obj_t ncnn_mp_Layer_get_bottom_shape(mp_obj_t self_in, mp_obj_t i_obj) {
    ncnn_mp_Layer_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int i = mp_obj_get_int(i_obj);
    int dims, w, h, c;
    ncnn_blob_get_bottom_shape(self->layer, i, &dims, &w, &h, &c);
    mp_obj_t items[] = {mp_obj_new_int(dims), mp_obj_new_int(w), mp_obj_new_int(h), mp_obj_new_int(c)};
    return mp_obj_new_tuple(4, items);
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Layer_get_bottom_shape_obj, ncnn_mp_Layer_get_bottom_shape);

// Layer.get_top_shape()
static mp_obj_t ncnn_mp_Layer_get_top_shape(mp_obj_t self_in, mp_obj_t i_obj) {
    ncnn_mp_Layer_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int i = mp_obj_get_int(i_obj);
    int dims, w, h, c;
    ncnn_blob_get_top_shape(self->layer, i, &dims, &w, &h, &c);
    mp_obj_t items[] = {mp_obj_new_int(dims), mp_obj_new_int(w), mp_obj_new_int(h), mp_obj_new_int(c)};
    return mp_obj_new_tuple(4, items);
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Layer_get_top_shape_obj, ncnn_mp_Layer_get_top_shape);

// Attribute handler for Layer properties
static void ncnn_mp_Layer_attr(mp_obj_t self_in, qstr attr, mp_obj_t *dest) {
    ncnn_mp_Layer_obj_t *self = MP_OBJ_TO_PTR(self_in);

    if (dest[0] == MP_OBJ_NULL) {
        if (attr == MP_QSTR_name) {
            #if NCNN_STRING
            dest[0] = mp_obj_new_str(ncnn_layer_get_name(self->layer), strlen(ncnn_layer_get_name(self->layer)));
            #else
            mp_raise_msg(&mp_type_AttributeError, MP_ERROR_TEXT("Accessing the 'Layer.name' attribute failed: This feature depends on the 'NCNN_STRING' option"));
            #endif
        } else if (attr == MP_QSTR_typeindex) {
            dest[0] = mp_obj_new_int(ncnn_layer_get_typeindex(self->layer));
        } else if (attr == MP_QSTR_type) {
            #if NCNN_STRING
            dest[0] = mp_obj_new_str(ncnn_layer_get_type(self->layer), strlen(ncnn_layer_get_type(self->layer)));
            #else
            mp_raise_msg(&mp_type_AttributeError, MP_ERROR_TEXT("Accessing the 'Layer.type' attribute failed: This feature depends on the 'NCNN_STRING' option"));
            #endif
        } else if (attr == MP_QSTR_one_blob_only) {
            dest[0] = mp_obj_new_bool(ncnn_layer_get_one_blob_only(self->layer));
        } else if (attr == MP_QSTR_support_inplace) {
            dest[0] = mp_obj_new_bool(ncnn_layer_get_support_inplace(self->layer));
        } else if (attr == MP_QSTR_support_vulkan) {
            dest[0] = mp_obj_new_bool(ncnn_layer_get_support_vulkan(self->layer));
        } else if (attr == MP_QSTR_support_packing) {
            dest[0] = mp_obj_new_bool(ncnn_layer_get_support_packing(self->layer));
        } else if (attr == MP_QSTR_support_bf16_storage) {
            dest[0] = mp_obj_new_bool(ncnn_layer_get_support_bf16_storage(self->layer));
        } else if (attr == MP_QSTR_support_fp16_storage) {
            dest[0] = mp_obj_new_bool(ncnn_layer_get_support_fp16_storage(self->layer));
        } else if (attr == MP_QSTR_bottom_count) {
            dest[0] = mp_obj_new_int(ncnn_layer_get_bottom_count(self->layer));
        } else if (attr == MP_QSTR_top_count) {
            dest[0] = mp_obj_new_int(ncnn_layer_get_top_count(self->layer));
        } else {
            dest[1] = MP_OBJ_SENTINEL;
        }
    } else if (dest[1] != MP_OBJ_NULL) {
        if (attr == MP_QSTR_one_blob_only) {
            ncnn_layer_set_one_blob_only(self->layer, mp_obj_is_true(dest[1]));
            dest[0] = MP_OBJ_NULL;
        } else if (attr == MP_QSTR_support_inplace) {
            ncnn_layer_set_support_inplace(self->layer, mp_obj_is_true(dest[1]));
            dest[0] = MP_OBJ_NULL;
        } else if (attr == MP_QSTR_support_vulkan) {
            ncnn_layer_set_support_vulkan(self->layer, mp_obj_is_true(dest[1]));
            dest[0] = MP_OBJ_NULL;
        } else if (attr == MP_QSTR_support_packing) {
            ncnn_layer_set_support_packing(self->layer, mp_obj_is_true(dest[1]));
            dest[0] = MP_OBJ_NULL;
        } else if (attr == MP_QSTR_support_bf16_storage) {
            ncnn_layer_set_support_bf16_storage(self->layer, mp_obj_is_true(dest[1]));
            dest[0] = MP_OBJ_NULL;
        } else if (attr == MP_QSTR_support_fp16_storage) {
            ncnn_layer_set_support_fp16_storage(self->layer, mp_obj_is_true(dest[1]));
            dest[0] = MP_OBJ_NULL;
        } else {
            dest[1] = MP_OBJ_SENTINEL;
        }
    }
}

// ------------------
/* layer function pointer api */
// ------------------

// Layer.load_param()
static mp_obj_t ncnn_mp_Layer_load_param(mp_obj_t self_in, mp_obj_t pd_obj) {
    ncnn_mp_Layer_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ncnn_paramdict_t pd = ((ncnn_mp_ParamDict_obj_t*)MP_OBJ_TO_PTR(pd_obj))->pd;
    return mp_obj_new_int(self->layer->load_param(self->layer, pd));
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Layer_load_param_obj, ncnn_mp_Layer_load_param);

// Layer.load_model()
static mp_obj_t ncnn_mp_Layer_load_model(mp_obj_t self_in, mp_obj_t mb_obj) {
    ncnn_mp_Layer_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ncnn_modelbin_t mb = ((ncnn_mp_ModelBin_obj_t*)MP_OBJ_TO_PTR(mb_obj))->mb;
    return mp_obj_new_int(self->layer->load_model(self->layer, mb));
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Layer_load_model_obj, ncnn_mp_Layer_load_model);

// Layer.create_pipeline()
static mp_obj_t ncnn_mp_Layer_create_pipeline(mp_obj_t self_in, mp_obj_t opt_obj) {
    ncnn_mp_Layer_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ncnn_option_t opt = ((ncnn_mp_Option_obj_t*)MP_OBJ_TO_PTR(opt_obj))->opt;
    return mp_obj_new_int(self->layer->create_pipeline(self->layer, opt));
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Layer_create_pipeline_obj, ncnn_mp_Layer_create_pipeline);

// Layer.destroy_pipeline()
static mp_obj_t ncnn_mp_Layer_destroy_pipeline(mp_obj_t self_in, mp_obj_t opt_obj) {
    ncnn_mp_Layer_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ncnn_option_t opt = ((ncnn_mp_Option_obj_t*)MP_OBJ_TO_PTR(opt_obj))->opt;
    return mp_obj_new_int(self->layer->destroy_pipeline(self->layer, opt));
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Layer_destroy_pipeline_obj, ncnn_mp_Layer_destroy_pipeline);

// Layer.forward()
static mp_obj_t ncnn_mp_Layer_forward(size_t n_args, const mp_obj_t *args) {
    ncnn_mp_Layer_obj_t *self = MP_OBJ_TO_PTR(args[0]);
    mp_obj_t bottom_blobs_obj = args[1];
    ncnn_option_t opt = ((ncnn_mp_Option_obj_t*)MP_OBJ_TO_PTR(args[n_args - 1]))->opt;

    if (mp_obj_is_type(bottom_blobs_obj, &ncnn_mp_type_Mat)) {
        // forward_1
        ncnn_mat_t bottom_blob = ((ncnn_mp_Mat_obj_t*)MP_OBJ_TO_PTR(bottom_blobs_obj))->mat;
        ncnn_mat_t top_blob = NULL;
        self->layer->forward_1(self->layer, bottom_blob, &top_blob, opt);
        ncnn_mp_Mat_obj_t *top_mat_obj = mp_obj_malloc(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
        top_mat_obj->mat = top_blob;
        return MP_OBJ_FROM_PTR(top_mat_obj);
    } else {
        // forward_n, obj type is a list of Mat
        size_t n_bottom;
        mp_obj_t *bottom_items;
        mp_obj_get_array(bottom_blobs_obj, &n_bottom, &bottom_items);
        int n_top = mp_obj_get_int(args[2]);

        ncnn_mat_t* bottom_blobs = alloca(n_bottom * sizeof(ncnn_mat_t));
        for (size_t i = 0; i < n_bottom; i++) {
            bottom_blobs[i] = ((ncnn_mp_Mat_obj_t*)MP_OBJ_TO_PTR(bottom_items[i]))->mat;
        }
        ncnn_mat_t* top_blobs = alloca(n_top * sizeof(ncnn_mat_t));
        self->layer->forward_n(self->layer, bottom_blobs, n_bottom, top_blobs, n_top, opt);

        mp_obj_t* top_items = alloca(n_top * sizeof(mp_obj_t));
        for (int i = 0; i < n_top; i++) {
            ncnn_mp_Mat_obj_t *top_mat_obj = mp_obj_malloc(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
            top_mat_obj->mat = top_blobs[i];
            top_mat_obj->is_wrapper = false;
            top_items[i] = MP_OBJ_FROM_PTR(top_mat_obj);
        }
        return mp_obj_new_tuple(n_top, top_items);
    }
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mp_Layer_forward_obj, 3, 4, ncnn_mp_Layer_forward);

// Layer.forward_inplace()
static mp_obj_t ncnn_mp_Layer_forward_inplace(mp_obj_t self_in, mp_obj_t bottom_top_obj, mp_obj_t opt_obj) {
    ncnn_mp_Layer_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ncnn_option_t opt = ((ncnn_mp_Option_obj_t*)MP_OBJ_TO_PTR(opt_obj))->opt;

    if (mp_obj_is_type(bottom_top_obj, &ncnn_mp_type_Mat)) {
        ncnn_mat_t bottom_top_blob = ((ncnn_mp_Mat_obj_t*)MP_OBJ_TO_PTR(bottom_top_obj))->mat;
        return mp_obj_new_int(self->layer->forward_inplace_1(self->layer, bottom_top_blob, opt));
    } else {
        size_t n;
        mp_obj_t *items;
        mp_obj_get_array(bottom_top_obj, &n, &items);
        ncnn_mat_t* bottom_top_blobs = alloca(n * sizeof(ncnn_mat_t));
        for (size_t i = 0; i < n; i++) {
            bottom_top_blobs[i] = ((ncnn_mp_Mat_obj_t*)MP_OBJ_TO_PTR(items[i]))->mat;
        }
        int result = self->layer->forward_inplace_n(self->layer, bottom_top_blobs, n, opt);
        return mp_obj_new_int(result);
    }
}
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_mp_Layer_forward_inplace_obj, ncnn_mp_Layer_forward_inplace);

// Layer class.
static const mp_rom_map_elem_t ncnn_mp_Layer_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&ncnn_mp_Layer_deinit_obj) },
    #if NCNN_STRING
    { MP_ROM_QSTR(MP_QSTR_type_to_index), MP_ROM_PTR(&ncnn_mp_Layer_type_to_index_obj) },
    #endif
    { MP_ROM_QSTR(MP_QSTR_get_bottom), MP_ROM_PTR(&ncnn_mp_Layer_get_bottom_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_top), MP_ROM_PTR(&ncnn_mp_Layer_get_top_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_bottom_shape), MP_ROM_PTR(&ncnn_mp_Layer_get_bottom_shape_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_top_shape), MP_ROM_PTR(&ncnn_mp_Layer_get_top_shape_obj) },
    { MP_ROM_QSTR(MP_QSTR_load_param), MP_ROM_PTR(&ncnn_mp_Layer_load_param_obj) },
    { MP_ROM_QSTR(MP_QSTR_load_model), MP_ROM_PTR(&ncnn_mp_Layer_load_model_obj) },
    { MP_ROM_QSTR(MP_QSTR_create_pipeline), MP_ROM_PTR(&ncnn_mp_Layer_create_pipeline_obj) },
    { MP_ROM_QSTR(MP_QSTR_destroy_pipeline), MP_ROM_PTR(&ncnn_mp_Layer_destroy_pipeline_obj) },
    { MP_ROM_QSTR(MP_QSTR_forward), MP_ROM_PTR(&ncnn_mp_Layer_forward_obj) },
    { MP_ROM_QSTR(MP_QSTR_forward_inplace), MP_ROM_PTR(&ncnn_mp_Layer_forward_inplace_obj) },
};
static MP_DEFINE_CONST_DICT(ncnn_mp_Layer_locals_dict, ncnn_mp_Layer_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    ncnn_mp_type_Layer,
    MP_QSTR_Layer,
    MP_TYPE_FLAG_NONE,
    make_new, ncnn_mp_Layer_make_new,
    attr, ncnn_mp_Layer_attr,
    locals_dict, &ncnn_mp_Layer_locals_dict
);

// ------------------
/* net api */
// ------------------

// Constructor: Net.__new__ and Net.__init__
// Usage: Net()
static mp_obj_t ncnn_mp_Net_make_new(const mp_obj_type_t *type, size_t n_args, size_t n_kw, const mp_obj_t *args) {
    mp_arg_check_num(n_args, n_kw, 0, 0, false);
    ncnn_mp_Net_obj_t *self = mp_obj_malloc(ncnn_mp_Net_obj_t, type);
    self->net = ncnn_net_create();
    return MP_OBJ_FROM_PTR(self);
}

// Destructor: Net.__del__()
static mp_obj_t ncnn_mp_Net_deinit(mp_obj_t self_in) {
    ncnn_mp_Net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->net) {
        ncnn_net_destroy(self->net);
        self->net = NULL;
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mp_Net_deinit_obj, ncnn_mp_Net_deinit);

// Attributes
static void ncnn_mp_Net_attr(mp_obj_t self_in, qstr attr, mp_obj_t *dest) {
    ncnn_mp_Net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (dest[0] == MP_OBJ_NULL) {
        if (attr == MP_QSTR_option) {
            ncnn_mp_Option_obj_t *opt_obj = mp_obj_malloc(ncnn_mp_Option_obj_t, &ncnn_mp_type_Option);
            opt_obj->opt = ncnn_net_get_option(self->net);
            dest[0] = MP_OBJ_FROM_PTR(opt_obj);
        } else if (attr == MP_QSTR_input_count) {
            dest[0] = mp_obj_new_int(ncnn_net_get_input_count(self->net));
        } else if (attr == MP_QSTR_output_count) {
            dest[0] = mp_obj_new_int(ncnn_net_get_output_count(self->net));
        } else {
            dest[1] = MP_OBJ_SENTINEL;
        }
    } else if (dest[1] != MP_OBJ_NULL) {
        if (attr == MP_QSTR_option) {
            ncnn_option_t opt = ((ncnn_mp_Option_obj_t*)MP_OBJ_TO_PTR(dest[1]))->opt;
            ncnn_net_set_option(self->net, opt);
            dest[0] = MP_OBJ_NULL;
        } else {
            dest[1] = MP_OBJ_SENTINEL;
        }
    }
}

#if NCNN_VULKAN
// Net.set_vulkan_device()
static mp_obj_t ncnn_mp_Net_set_vulkan_device(mp_obj_t self_in, mp_obj_t device_index_obj) {
    ncnn_mp_Net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int device_index = mp_obj_get_int(device_index_obj);
    ncnn_net_set_vulkan_device(self->net, device_index);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Net_set_vulkan_device_obj, ncnn_mp_Net_set_vulkan_device);
#endif // NCNN_VULKAN

// Generic forward function
static int generic_forward_1(ncnn_layer_t layer, const ncnn_mat_t bottom_blob, ncnn_mat_t* top_blob, const ncnn_option_t opt) {
    // Get instance from C layer's userdata
    mp_obj_t self = mp_obj_dict_get(layer_instance_map, MP_OBJ_FROM_PTR(layer));
    if (self == MP_OBJ_NULL) {
        return -1;
    }

    // Find the 'forward' method in the Python instance.
    // dest[0] is method, dest[1] is self instance
    mp_obj_t dest[2];
    mp_load_method(self, MP_QSTR_forward, dest);

    // Wrap C handles into MicroPython objects.
    ncnn_mp_Mat_obj_t *bottom_obj = mp_obj_malloc_with_finaliser(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
    bottom_obj->mat = (ncnn_mat_t)bottom_blob;
    bottom_obj->is_wrapper = true;

    ncnn_mp_Option_obj_t *opt_obj = mp_obj_malloc_with_finaliser(ncnn_mp_Option_obj_t, &ncnn_mp_type_Option);
    opt_obj->opt = (ncnn_option_t)opt;
    opt_obj->is_wrapper = true;

    // Call Python method: instance.forward(bottom_blob, opt)
    mp_obj_t call_args[2 + 2];
    call_args[0] = dest[0];
    call_args[1] = dest[1];
    call_args[2] = MP_OBJ_FROM_PTR(bottom_obj);
    call_args[3] = MP_OBJ_FROM_PTR(opt_obj);

    mp_obj_t result = mp_call_method_n_kw(2, 0, call_args);

    if (!mp_obj_is_type(result, &ncnn_mp_type_Mat)) {
        return -1;
    }
    *top_blob = ((ncnn_mp_Mat_obj_t*)MP_OBJ_TO_PTR(result))->mat;

    return 0;
}

static int generic_forward_n(const ncnn_layer_t layer, const ncnn_mat_t* bottom_blobs, int n_bottom, ncnn_mat_t* top_blobs, int n_top, const ncnn_option_t opt) {
    mp_obj_t self = mp_obj_dict_get(layer_instance_map, MP_OBJ_FROM_PTR(layer));
    if (self == MP_OBJ_NULL) return -1;

    mp_obj_t dest[2];
    mp_load_method(self, MP_QSTR_forward, dest);

    // bottom_blobs -> tuple
    mp_obj_t* bottom_items = alloca(n_bottom * sizeof(mp_obj_t));
    for (int i = 0; i < n_bottom; i++) {
        ncnn_mp_Mat_obj_t *bottom_obj = mp_obj_malloc_with_finaliser(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
        bottom_obj->mat = bottom_blobs[i];
        bottom_obj->is_wrapper = true;
        bottom_items[i] = MP_OBJ_FROM_PTR(bottom_obj);
    }
    mp_obj_t py_bottom_blobs = mp_obj_new_tuple(n_bottom, bottom_items);

    ncnn_mp_Option_obj_t *opt_obj = mp_obj_malloc_with_finaliser(ncnn_mp_Option_obj_t, &ncnn_mp_type_Option);
    opt_obj->opt = (ncnn_option_t)opt;
    opt_obj->is_wrapper = true;

    // instance.forward((mat1, mat2, ...), opt)
    mp_obj_t call_args[2 + 2];
    call_args[0] = dest[0];
    call_args[1] = dest[1];
    call_args[2] = py_bottom_blobs;
    call_args[3] = MP_OBJ_FROM_PTR(opt_obj);
    mp_obj_t result_tuple = mp_call_method_n_kw(2, 0, call_args);

    // top_blobs
    if (!mp_obj_is_type(result_tuple, &mp_type_tuple)) {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Custom layer 'forward': expected tuple return for multi-blob output"));
    }
    size_t top_size;
    mp_obj_t *top_items;
    mp_obj_get_array(result_tuple, &top_size, &top_items);

    if (top_size != (size_t)n_top) {
        // Python != ncnn expectation
        mp_raise_msg_varg(&mp_type_RuntimeError, 
            MP_ERROR_TEXT("Custom layer 'forward': output blob count mismatch (returned %d, expected %d)"), 
            (int)top_size, n_top);
    }

    for (size_t i = 0; i < top_size; i++) {
        ncnn_mp_Mat_obj_t* py_mat = (ncnn_mp_Mat_obj_t*)MP_OBJ_TO_PTR(top_items[i]);
        if (!mp_obj_is_type(py_mat, &ncnn_mp_type_Mat)) {
            return -1;
        }
        top_blobs[i] = py_mat->mat;
    }

    return 0;
}

// Generic forward_inplace functions
static int generic_forward_inplace_1(const ncnn_layer_t layer, ncnn_mat_t bottom_top_blob, const ncnn_option_t opt) {
    mp_obj_t self = mp_obj_dict_get(layer_instance_map, MP_OBJ_FROM_PTR(layer));
    if (self == MP_OBJ_NULL) {
        return -1;
    }

    mp_obj_t dest[2];
    mp_load_method(self, MP_QSTR_forward_inplace, dest);

    ncnn_mp_Mat_obj_t *blob_obj = mp_obj_malloc_with_finaliser(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
    blob_obj->mat = bottom_top_blob;
    blob_obj->is_wrapper = true;

    ncnn_mp_Option_obj_t *opt_obj = mp_obj_malloc_with_finaliser(ncnn_mp_Option_obj_t, &ncnn_mp_type_Option);
    opt_obj->opt = (ncnn_option_t)opt;
    opt_obj->is_wrapper = true;

    mp_obj_t call_args[2 + 2];
    call_args[0] = dest[0];
    call_args[1] = dest[1];
    call_args[2] = MP_OBJ_FROM_PTR(blob_obj);
    call_args[3] = MP_OBJ_FROM_PTR(opt_obj);
    mp_obj_t result = mp_call_method_n_kw(2, 0, call_args);

    return mp_obj_get_int(result);
}

static int generic_forward_inplace_n(const ncnn_layer_t layer, ncnn_mat_t* bottom_top_blobs, int n, const ncnn_option_t opt) {
    mp_obj_t self = mp_obj_dict_get(layer_instance_map, MP_OBJ_FROM_PTR(layer));
    if (self == MP_OBJ_NULL) {
        return -1;
    }

    mp_obj_t dest[2];
    mp_load_method(self, MP_QSTR_forward_inplace, dest);

    mp_obj_t* py_blob_items = alloca(n * sizeof(mp_obj_t));
    for (int i = 0; i < n; i++) {
        ncnn_mp_Mat_obj_t *blob_obj = mp_obj_malloc_with_finaliser(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
        blob_obj->mat = bottom_top_blobs[i];
        blob_obj->is_wrapper = true;
        py_blob_items[i] = MP_OBJ_FROM_PTR(blob_obj);
    }
    mp_obj_t py_blobs_tuple = mp_obj_new_tuple(n, py_blob_items);

    ncnn_mp_Option_obj_t *opt_obj = mp_obj_malloc_with_finaliser(ncnn_mp_Option_obj_t, &ncnn_mp_type_Option);
    opt_obj->opt = (ncnn_option_t)opt;
    opt_obj->is_wrapper = true;

    mp_obj_t call_args[2 + 2];
    call_args[0] = dest[0];
    call_args[1] = dest[1];
    call_args[2] = py_blobs_tuple;
    call_args[3] = MP_OBJ_FROM_PTR(opt_obj);
    mp_obj_t result = mp_call_method_n_kw(2, 0, call_args);

    return mp_obj_get_int(result);
}

// Generic load_param function
static int generic_load_param(ncnn_layer_t layer, const ncnn_paramdict_t pd) {
    mp_obj_t self = mp_obj_dict_get(layer_instance_map, MP_OBJ_FROM_PTR(layer));
    if (self == MP_OBJ_NULL) {
        return -1;
    }

    mp_obj_t dest[2];
    mp_load_method(self, MP_QSTR_load_param, dest);

    // ncnn_paramdict_t -> Python: ncnn_mp.ParamDict()
    ncnn_mp_ParamDict_obj_t *pd_obj = mp_obj_malloc_with_finaliser(ncnn_mp_ParamDict_obj_t, &ncnn_mp_type_ParamDict);
    pd_obj->pd = (ncnn_paramdict_t)pd;
    pd_obj->is_wrapper = true;

    // instance.load_param(pd_obj)
    mp_obj_t call_args[2 + 1];
    call_args[0] = dest[0];
    call_args[1] = dest[1];
    call_args[2] = MP_OBJ_FROM_PTR(pd_obj);

    mp_obj_t result = mp_call_method_n_kw(1, 0, call_args);

    return mp_obj_get_int(result);
}

// Generic load_model function
static int generic_load_model(ncnn_layer_t layer, const ncnn_modelbin_t mb) {
    mp_obj_t self = mp_obj_dict_get(layer_instance_map, MP_OBJ_FROM_PTR(layer));
    if (self == MP_OBJ_NULL) {
        return -1;
    }

    mp_obj_t dest[2];
    mp_load_method(self, MP_QSTR_load_model, dest);

    ncnn_mp_ModelBin_obj_t *mb_obj = mp_obj_malloc_with_finaliser(ncnn_mp_ModelBin_obj_t, &ncnn_mp_type_ModelBin);
    mb_obj->mb = (ncnn_modelbin_t)mb;
    mb_obj->is_wrapper = true;

    // instance.load_model(mb_obj)
    mp_obj_t call_args[2 + 1];
    call_args[0] = dest[0];
    call_args[1] = dest[1];
    call_args[2] = MP_OBJ_FROM_PTR(mb_obj);

    mp_obj_t result = mp_call_method_n_kw(1, 0, call_args);

    return mp_obj_get_int(result);
}

// Generic create_pipeline function (for vulkan)
static int generic_create_pipeline(ncnn_layer_t layer, const ncnn_option_t opt) {
    mp_obj_t self = mp_obj_dict_get(layer_instance_map, MP_OBJ_FROM_PTR(layer));
    if (self == MP_OBJ_NULL) {
        return -1;
    }

    mp_obj_t dest[2];
    mp_load_method(self, MP_QSTR_create_pipeline, dest);

    ncnn_mp_Option_obj_t *opt_obj = mp_obj_malloc_with_finaliser(ncnn_mp_Option_obj_t, &ncnn_mp_type_Option);
    opt_obj->opt = (ncnn_option_t)opt;
    opt_obj->is_wrapper = true;

    // instance.create_pipeline(opt_obj)
    mp_obj_t call_args[2 + 1];
    call_args[0] = dest[0];
    call_args[1] = dest[1];
    call_args[2] = MP_OBJ_FROM_PTR(opt_obj);

    mp_obj_t result = mp_call_method_n_kw(1, 0, call_args);

    return mp_obj_get_int(result);
}

// Generic destroy_pipeline function (for vulkan)
static int generic_destroy_pipeline(ncnn_layer_t layer, const ncnn_option_t opt) {
    mp_obj_t self = mp_obj_dict_get(layer_instance_map, MP_OBJ_FROM_PTR(layer));
    if (self == MP_OBJ_NULL) {
        return -1;
    }

    mp_obj_t dest[2];
    mp_load_method(self, MP_QSTR_destroy_pipeline, dest);

    ncnn_mp_Option_obj_t *opt_obj = mp_obj_malloc_with_finaliser(ncnn_mp_Option_obj_t, &ncnn_mp_type_Option);
    opt_obj->opt = (ncnn_option_t)opt;
    opt_obj->is_wrapper = true;

    // instance.destroy_pipeline(opt_obj)
    mp_obj_t call_args[2 + 1];
    call_args[0] = dest[0];
    call_args[1] = dest[1];
    call_args[2] = MP_OBJ_FROM_PTR(opt_obj);

    mp_obj_t result = mp_call_method_n_kw(1, 0, call_args);

    return mp_obj_get_int(result);
}

// Generic layer creator function
static ncnn_layer_t generic_creator(void* userdata) {
    // Init
    if (custom_layer_instances == MP_OBJ_NULL) {
        custom_layer_instances = mp_obj_new_list(0, NULL);
    }

    if (layer_instance_map == MP_OBJ_NULL) {
        layer_instance_map = mp_obj_new_dict(0);
    }

    // 'userdata' is the Python Layer class passed from register_custom_layer.
    mp_obj_t class_obj = (mp_obj_t)userdata;

    // Instantiate the Python class to get an instance. Equivalent to `instance = MyLayer()`.
    mp_obj_t instance_obj = mp_call_function_0(class_obj);
    if (instance_obj == MP_OBJ_NULL) {
        return NULL;
    }
    mp_obj_list_append(custom_layer_instances, instance_obj);  // add to list

    // Create ncnn_layer_t object in C.
    ncnn_layer_t c_layer = ncnn_layer_create();
    if (!c_layer) {
        mp_obj_list_remove(custom_layer_instances, instance_obj);
        return NULL;
    }

    mp_obj_dict_store(layer_instance_map, MP_OBJ_FROM_PTR(c_layer), instance_obj);

    // Configure the C layer based on attributes from the Python instance.
    mp_obj_t attr;
    bool one_blob_only = false;
    bool support_inplace = false;
    bool support_vulkan = false;
    
    attr = mp_load_attr(instance_obj, MP_QSTR_one_blob_only);
    if (mp_obj_is_true(attr)) {
        ncnn_layer_set_one_blob_only(c_layer, 1);
        one_blob_only = true;
    }

    attr = mp_load_attr(instance_obj, MP_QSTR_support_inplace);
    if (mp_obj_is_true(attr)) {
        ncnn_layer_set_support_inplace(c_layer, 1);
        support_inplace = true;
    }

    attr = mp_load_attr(instance_obj, MP_QSTR_support_vulkan);
    if (mp_obj_is_true(attr)) {
        ncnn_layer_set_support_vulkan(c_layer, 1);
        support_vulkan = true;
    }

    attr = mp_load_attr(instance_obj, MP_QSTR_support_packing);
    if (mp_obj_is_true(attr)) {
        ncnn_layer_set_support_packing(c_layer, 1);
    }

    attr = mp_load_attr(instance_obj, MP_QSTR_support_bf16_storage);
    if (mp_obj_is_true(attr)) {
        ncnn_layer_set_support_bf16_storage(c_layer, 1);
    }

    attr = mp_load_attr(instance_obj, MP_QSTR_support_fp16_storage);
    if (mp_obj_is_true(attr)) {
        ncnn_layer_set_support_fp16_storage(c_layer, 1);
    }
    
    // Hook up the generic C funcs to call the C layer's function pointers.
    mp_obj_t dest[2];
    if (one_blob_only) {
        if (support_inplace) {
            mp_load_method_maybe(instance_obj, MP_QSTR_forward_inplace, dest);
            if (dest[0] == MP_OBJ_NULL) {
                mp_raise_msg(&mp_type_AttributeError, MP_ERROR_TEXT("Custom layer supports inplace but is missing 'forward_inplace' method"));
            }
            c_layer->forward_inplace_1 = generic_forward_inplace_1;
        } else {
            mp_load_method_maybe(instance_obj, MP_QSTR_forward, dest);
            if (dest[0] == MP_OBJ_NULL) {
                mp_raise_msg(&mp_type_AttributeError, MP_ERROR_TEXT("Custom layer is missing 'forward' method"));
            }
            c_layer->forward_1 = generic_forward_1;
        }
    } else {
        if (support_inplace) {
            mp_load_method_maybe(instance_obj, MP_QSTR_forward_inplace, dest);
            if (dest[0] == MP_OBJ_NULL) {
                mp_raise_msg(&mp_type_AttributeError, MP_ERROR_TEXT("Custom layer supports inplace (multi-blob) but is missing 'forward_inplace' method"));
            }
            c_layer->forward_inplace_n = generic_forward_inplace_n;
        } else {
            mp_load_method_maybe(instance_obj, MP_QSTR_forward, dest);
            if (dest[0] == MP_OBJ_NULL) {
                mp_raise_msg(&mp_type_AttributeError, MP_ERROR_TEXT("Custom layer (multi-blob) is missing 'forward' method"));
            }
            c_layer->forward_n = generic_forward_n;
        }
    }

    if (support_vulkan) {
        mp_load_method_maybe(instance_obj, MP_QSTR_create_pipeline, dest);
        if (dest[0] == MP_OBJ_NULL) {
            mp_raise_msg(&mp_type_AttributeError, MP_ERROR_TEXT("Custom layer supports vulkan but is missing 'create_pipeline' method"));
        }
        c_layer->create_pipeline = generic_create_pipeline;

        mp_load_method_maybe(instance_obj, MP_QSTR_destroy_pipeline, dest);
        if (dest[0] != MP_OBJ_NULL) {
            c_layer->destroy_pipeline = generic_destroy_pipeline;
        }
    }

    mp_load_method_maybe(instance_obj, MP_QSTR_load_param, dest);
    if (dest[0] != MP_OBJ_NULL) {
        c_layer->load_param = generic_load_param;
    }

    mp_load_method_maybe(instance_obj, MP_QSTR_load_model, dest);
    if (dest[0] != MP_OBJ_NULL) {
        c_layer->load_model = generic_load_model;
    }

    return c_layer;
}

// Generic layer destroyer function
static void generic_destroyer(ncnn_layer_t layer, void* /*userdata*/) {
    if (!layer) return;

    // Get the Python instance from userdata.
    mp_obj_t instance_obj = mp_obj_dict_delete(layer_instance_map, MP_OBJ_FROM_PTR(layer));

    // Remove the instance from our global list.
    if (instance_obj != MP_OBJ_NULL) {
        mp_obj_list_remove(custom_layer_instances, instance_obj);
    }
    
    ncnn_layer_destroy(layer);
}

// Net.register_custom_layer(type_or_index, layer_class)
static mp_obj_t ncnn_mp_Net_register_custom_layer(size_t n_args, const mp_obj_t *args) {
    ncnn_mp_Net_obj_t *self = MP_OBJ_TO_PTR(args[0]);
    mp_obj_t type_arg = args[1];
    mp_obj_t class_arg = args[2];
    if (!mp_obj_is_callable(class_arg)) {
        mp_raise_msg(&mp_type_TypeError, MP_ERROR_TEXT("Net.register_custom_layer failed: The second parameter must be a callable class"));
    }

    void* userdata = (void*)class_arg;

    if (mp_obj_is_str(type_arg)) {
        #if NCNN_STRING
        const char* type_name = mp_obj_str_get_str(type_arg);
        ncnn_net_register_custom_layer_by_type(self->net, type_name, generic_creator, generic_destroyer, userdata);
        #else
        mp_raise_msg(&mp_type_NotImplementedError, MP_ERROR_TEXT("Net.register_custom_layer failed: Register by 'type' requires NCNN_STRING=ON"));
        #endif
    } else if (mp_obj_is_int(type_arg)) {
        int type_index = mp_obj_get_int(type_arg);
        ncnn_net_register_custom_layer_by_typeindex(self->net, type_index, generic_creator, generic_destroyer, userdata);
    } else {
        mp_raise_msg(&mp_type_TypeError, MP_ERROR_TEXT("Net.register_custom_layer failed: The first parameter must be a string or an integer"));
    }

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mp_Net_register_custom_layer_obj, 3, 3, ncnn_mp_Net_register_custom_layer);

// Net.load_param()
static mp_obj_t ncnn_mp_Net_load_param(mp_obj_t self_in, mp_obj_t source_obj) {
    #if NCNN_STRING
    ncnn_mp_Net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int result = -1;
    if (mp_obj_is_str(source_obj)) {
        #if NCNN_STDIO
        result = ncnn_net_load_param(self->net, mp_obj_str_get_str(source_obj));
        #else
        mp_raise_msg(&mp_type_NotImplementedError, MP_ERROR_TEXT("Net.load_param failed: load_param from file path requires NCNN_STDIO"));
        #endif
    } else if (mp_obj_is_type(source_obj, &ncnn_mp_type_DataReader)) {
        ncnn_datareader_t dr = ((ncnn_mp_DataReader_obj_t*)MP_OBJ_TO_PTR(source_obj))->dr;
        result = ncnn_net_load_param_datareader(self->net, dr);
    } else {
        mp_raise_msg(&mp_type_TypeError, MP_ERROR_TEXT("Net.load_param failed: load_param source must be a path string or DataReader"));
    }

    // This return value is a kind of status code. TODO
    if (result != 0) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Failed to load param"));
    }
    #else
        mp_raise_msg(&mp_type_NotImplementedError, MP_ERROR_TEXT("Net.load_param failed: load_param requires NCNN_STRING=ON. Or try using 'Net.load_param_bin()'."));
    #endif

    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Net_load_param_obj, ncnn_mp_Net_load_param);

// Net.load_param_bin()
static mp_obj_t ncnn_mp_Net_load_param_bin(mp_obj_t self_in, mp_obj_t source_obj) {
    ncnn_mp_Net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int result = -1;

    if (mp_obj_is_str(source_obj)) {
        #if NCNN_STDIO
        const char* path = mp_obj_str_get_str(source_obj);
        result = ncnn_net_load_param_bin(self->net, path);
        #else
        mp_raise_msg(&mp_type_NotImplementedError, MP_ERROR_TEXT("Net.load_param_bin failed: load_param_bin from file path requires NCNN_STDIO"));
        #endif
    } else if (mp_obj_is_type(source_obj, &ncnn_mp_type_DataReader)) {
        ncnn_datareader_t dr = ((ncnn_mp_DataReader_obj_t*)MP_OBJ_TO_PTR(source_obj))->dr;
        result = ncnn_net_load_param_bin_datareader(self->net, dr);
    } else {
        mp_buffer_info_t bufinfo;
        mp_get_buffer_raise(source_obj, &bufinfo, MP_BUFFER_READ);
        result = ncnn_net_load_param_bin_memory(self->net, bufinfo.buf);
    }
    
    if (result != 0) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Failed to load param_bin"));
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Net_load_param_bin_obj, ncnn_mp_Net_load_param_bin);

// Net.load_model()
static mp_obj_t ncnn_mp_Net_load_model(mp_obj_t self_in, mp_obj_t source_obj) {
    ncnn_mp_Net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int result = -1;

    if (mp_obj_is_str(source_obj)) {
        #if NCNN_STDIO
        const char* path = mp_obj_str_get_str(source_obj);
        result = ncnn_net_load_model(self->net, path);
        #else
        mp_raise_msg(&mp_type_NotImplementedError, MP_ERROR_TEXT("Net.load_model failed: load_model from file path requires NCNN_STDIO"));
        #endif
    } else if (mp_obj_is_type(source_obj, &ncnn_mp_type_DataReader)) {
        ncnn_datareader_t dr = ((ncnn_mp_DataReader_obj_t*)MP_OBJ_TO_PTR(source_obj))->dr;
        result = ncnn_net_load_model_datareader(self->net, dr);
    } else {
        mp_buffer_info_t bufinfo;
        mp_get_buffer_raise(source_obj, &bufinfo, MP_BUFFER_READ);
        result = ncnn_net_load_model_memory(self->net, bufinfo.buf);
    }
    
    if (result != 0) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Failed to load model"));
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Net_load_model_obj, ncnn_mp_Net_load_model);

// Net.clear()
static mp_obj_t ncnn_mp_Net_clear(mp_obj_t self_in) {
    ncnn_mp_Net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ncnn_net_clear(self->net);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mp_Net_clear_obj, ncnn_mp_Net_clear);

// Net.get_input_name()
#if NCNN_STRING
static mp_obj_t ncnn_mp_Net_get_input_name(mp_obj_t self_in, mp_obj_t i_obj) {
    ncnn_mp_Net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int i = mp_obj_get_int(i_obj);
    const char* name = ncnn_net_get_input_name(self->net, i);
    return mp_obj_new_str(name, strlen(name));
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Net_get_input_name_obj, ncnn_mp_Net_get_input_name);
#endif

// Net.get_output_name()
#if NCNN_STRING
static mp_obj_t ncnn_mp_Net_get_output_name(mp_obj_t self_in, mp_obj_t i_obj) {
    ncnn_mp_Net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int i = mp_obj_get_int(i_obj);
    const char* name = ncnn_net_get_output_name(self->net, i);
    return mp_obj_new_str(name, strlen(name));
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Net_get_output_name_obj, ncnn_mp_Net_get_output_name);
#endif

// Net.get_input_index()
static mp_obj_t ncnn_mp_Net_get_input_index(mp_obj_t self_in, mp_obj_t i_obj) {
    ncnn_mp_Net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int i = mp_obj_get_int(i_obj);
    return mp_obj_new_int(ncnn_net_get_input_index(self->net, i));
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Net_get_input_index_obj, ncnn_mp_Net_get_input_index);

// Net.get_output_index()
static mp_obj_t ncnn_mp_Net_get_output_index(mp_obj_t self_in, mp_obj_t i_obj) {
    ncnn_mp_Net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    int i = mp_obj_get_int(i_obj);
    return mp_obj_new_int(ncnn_net_get_output_index(self->net, i));
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Net_get_output_index_obj, ncnn_mp_Net_get_output_index);

// Net.create_extractor()
extern const mp_obj_type_t ncnn_mp_type_Extractor;
static mp_obj_t ncnn_mp_Net_create_extractor(mp_obj_t self_in) {
    ncnn_mp_Net_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ncnn_extractor_t ex = ncnn_extractor_create(self->net);
    if (!ex) {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Failed to create extractor"));
    }
    ncnn_mp_Extractor_obj_t *ex_obj = mp_obj_malloc(ncnn_mp_Extractor_obj_t, &ncnn_mp_type_Extractor);
    ex_obj->ex = ex;
    return MP_OBJ_FROM_PTR(ex_obj);
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mp_Net_create_extractor_obj, ncnn_mp_Net_create_extractor);

// Net class.
static const mp_rom_map_elem_t ncnn_mp_Net_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&ncnn_mp_Net_deinit_obj) },
    { MP_ROM_QSTR(MP_QSTR_clear), MP_ROM_PTR(&ncnn_mp_Net_clear_obj) },
    { MP_ROM_QSTR(MP_QSTR_load_param), MP_ROM_PTR(&ncnn_mp_Net_load_param_obj) },
    { MP_ROM_QSTR(MP_QSTR_load_param_bin), MP_ROM_PTR(&ncnn_mp_Net_load_param_bin_obj) },
    { MP_ROM_QSTR(MP_QSTR_load_model), MP_ROM_PTR(&ncnn_mp_Net_load_model_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_input_index), MP_ROM_PTR(&ncnn_mp_Net_get_input_index_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_output_index), MP_ROM_PTR(&ncnn_mp_Net_get_output_index_obj) },
    #if NCNN_STRING
    { MP_ROM_QSTR(MP_QSTR_get_input_name), MP_ROM_PTR(&ncnn_mp_Net_get_input_name_obj) },
    { MP_ROM_QSTR(MP_QSTR_get_output_name), MP_ROM_PTR(&ncnn_mp_Net_get_output_name_obj) },
    #endif
    { MP_ROM_QSTR(MP_QSTR_register_custom_layer), MP_ROM_PTR(&ncnn_mp_Net_register_custom_layer_obj) },
    #if NCNN_VULKAN
    { MP_ROM_QSTR(MP_QSTR_set_vulkan_device), MP_ROM_PTR(&ncnn_mp_Net_set_vulkan_device_obj) },
    #endif
    { MP_ROM_QSTR(MP_QSTR_create_extractor), MP_ROM_PTR(&ncnn_mp_Net_create_extractor_obj) },
};
static MP_DEFINE_CONST_DICT(ncnn_mp_Net_locals_dict, ncnn_mp_Net_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    ncnn_mp_type_Net,
    MP_QSTR_Net,
    MP_TYPE_FLAG_NONE,
    make_new, ncnn_mp_Net_make_new,
    attr, ncnn_mp_Net_attr,
    locals_dict, &ncnn_mp_Net_locals_dict
);

// ------------------
/* extractor api */
// ------------------

// No Constructor for extractor. This object is created by Net.create_extractor()

// Destructor: Extractor.__del__()
static mp_obj_t ncnn_mp_Extractor_deinit(mp_obj_t self_in) {
    ncnn_mp_Extractor_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->ex) {
        ncnn_extractor_destroy(self->ex);
        self->ex = NULL;
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mp_Extractor_deinit_obj, ncnn_mp_Extractor_deinit);

// Extractor.set_option()
static mp_obj_t ncnn_mp_Extractor_set_option(mp_obj_t self_in, mp_obj_t opt_obj) {
    ncnn_mp_Extractor_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ncnn_option_t opt = ((ncnn_mp_Option_obj_t*)MP_OBJ_TO_PTR(opt_obj))->opt;
    ncnn_extractor_set_option(self->ex, opt);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Extractor_set_option_obj, ncnn_mp_Extractor_set_option);

// Extractor.input()
static mp_obj_t ncnn_mp_Extractor_input(mp_obj_t self_in, mp_obj_t id_obj, mp_obj_t mat_obj) {
    ncnn_mp_Extractor_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ncnn_mat_t mat = ((ncnn_mp_Mat_obj_t*)MP_OBJ_TO_PTR(mat_obj))->mat;
    int result = -1;
    if (mp_obj_is_str(id_obj)) {
        #if NCNN_STRING
        result = ncnn_extractor_input(self->ex, mp_obj_str_get_str(id_obj), mat);
        #else
        mp_raise_msg(&mp_type_NotImplementedError, MP_ERROR_TEXT("Extractor.input failed: Inputting by name requires NCNN_STRING"));
        #endif
    } else {
        result = ncnn_extractor_input_index(self->ex, mp_obj_get_int(id_obj), mat);
    }
    if (result != 0) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Extractor.input failed: invalid input name or index"));
    }
    return mp_obj_new_int(result);
}
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_mp_Extractor_input_obj, ncnn_mp_Extractor_input);

// Extractor.extract()
static mp_obj_t ncnn_mp_Extractor_extract(mp_obj_t self_in, mp_obj_t id_obj) {
    ncnn_mp_Extractor_obj_t *self = MP_OBJ_TO_PTR(self_in);
    ncnn_mat_t mat = NULL;
    int result = -1;
    if (mp_obj_is_str(id_obj)) {
        #if NCNN_STRING
        result = ncnn_extractor_extract(self->ex, mp_obj_str_get_str(id_obj), &mat);
        #else
        mp_raise_msg(&mp_type_NotImplementedError, MP_ERROR_TEXT("Extractor.extract failed: Extracting by name requires NCNN_STRING"));
        #endif
    } else {
        result = ncnn_extractor_extract_index(self->ex, mp_obj_get_int(id_obj), &mat);
    }
    if (result != 0) {
        mp_raise_msg(&mp_type_ValueError, MP_ERROR_TEXT("Extractor.extract failed: invalid input name or index"));
    }
    ncnn_mp_Mat_obj_t *mat_obj = mp_obj_malloc(ncnn_mp_Mat_obj_t, &ncnn_mp_type_Mat);
    mat_obj->mat = mat;
    return MP_OBJ_FROM_PTR(mat_obj);
}
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mp_Extractor_extract_obj, ncnn_mp_Extractor_extract);

// Extractor class.
static const mp_rom_map_elem_t ncnn_mp_Extractor_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR___del__), MP_ROM_PTR(&ncnn_mp_Extractor_deinit_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_option), MP_ROM_PTR(&ncnn_mp_Extractor_set_option_obj) },
    { MP_ROM_QSTR(MP_QSTR_input), MP_ROM_PTR(&ncnn_mp_Extractor_input_obj) },
    { MP_ROM_QSTR(MP_QSTR_extract), MP_ROM_PTR(&ncnn_mp_Extractor_extract_obj) },
};
static MP_DEFINE_CONST_DICT(ncnn_mp_Extractor_locals_dict, ncnn_mp_Extractor_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    ncnn_mp_type_Extractor,
    MP_QSTR_Extractor,
    MP_TYPE_FLAG_NONE,
    locals_dict, &ncnn_mp_Extractor_locals_dict
);

static const mp_rom_map_elem_t ncnn_mp_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_ncnn_mp) },
    { MP_ROM_QSTR(MP_QSTR_version), MP_ROM_PTR(&ncnn_mp_version_obj) },

    { MP_ROM_QSTR(MP_QSTR_Allocator), MP_ROM_PTR(&ncnn_mp_type_Allocator) },
    { MP_ROM_QSTR(MP_QSTR_Option), MP_ROM_PTR(&ncnn_mp_type_Option) },
    { MP_ROM_QSTR(MP_QSTR_Mat), MP_ROM_PTR(&ncnn_mp_type_Mat) },
    { MP_ROM_QSTR(MP_QSTR_Blob), MP_ROM_PTR(&ncnn_mp_type_Blob) },
    { MP_ROM_QSTR(MP_QSTR_ParamDict), MP_ROM_PTR(&ncnn_mp_type_ParamDict) },
    { MP_ROM_QSTR(MP_QSTR_DataReader), MP_ROM_PTR(&ncnn_mp_type_DataReader) },
    { MP_ROM_QSTR(MP_QSTR_ModelBin), MP_ROM_PTR(&ncnn_mp_type_ModelBin) },
    { MP_ROM_QSTR(MP_QSTR_Layer), MP_ROM_PTR(&ncnn_mp_type_Layer) },
    { MP_ROM_QSTR(MP_QSTR_Net), MP_ROM_PTR(&ncnn_mp_type_Net) },
    { MP_ROM_QSTR(MP_QSTR_Extractor), MP_ROM_PTR(&ncnn_mp_type_Extractor) }
};
static MP_DEFINE_CONST_DICT(ncnn_mp_module_globals, ncnn_mp_module_globals_table);

const mp_obj_module_t ncnn_mp_user_cmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&ncnn_mp_module_globals,
};
MP_REGISTER_MODULE(MP_QSTR_ncnn_mp, ncnn_mp_user_cmodule);