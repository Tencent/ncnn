#include "ncnn/c_api.h"

extern "C" {
#include "py/runtime.h"

mp_obj_t mp_ncnn_layer_create(void)
{
    return mp_obj_new_int_from_uint((uintptr_t)ncnn_layer_create());
}
mp_obj_t mp_ncnn_layer_create_by_typeindex(mp_obj_t typeindex_obj)
{
    int typeindex = mp_obj_get_int(typeindex_obj);
    return mp_obj_new_int_from_uint((uintptr_t)ncnn_layer_create_by_typeindex(typeindex));
}
#if NCNN_STRING
mp_obj_t mp_ncnn_layer_create_by_type(mp_obj_t type_obj)
{
    const char* type = mp_obj_str_get_str(type_obj);
    return mp_obj_new_int_from_uint((uintptr_t)ncnn_layer_create_by_type(type));
}
mp_obj_t mp_ncnn_layer_type_to_index(mp_obj_t type_obj)
{
    const char* type = mp_obj_str_get_str(type_obj);
    return mp_obj_new_int(ncnn_layer_type_to_index(type));
}
#endif /* NCNN_STRING */
mp_obj_t mp_ncnn_layer_destroy(mp_obj_t layer_obj)
{
    ncnn_layer_destroy((ncnn_layer_t)mp_obj_get_int(layer_obj));
    return mp_const_none;
}

#if NCNN_STRING
mp_obj_t mp_ncnn_layer_get_name(mp_obj_t layer_obj)
{
    const ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    const char* name = ncnn_layer_get_name(layer);
    return mp_obj_new_str(name, strlen(name));
}
#endif /* NCNN_STRING */

mp_obj_t mp_ncnn_layer_get_typeindex(mp_obj_t layer_obj)
{
    const ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    return mp_obj_new_int(ncnn_layer_get_typeindex(layer));
}
#if NCNN_STRING
mp_obj_t mp_ncnn_layer_get_type(mp_obj_t layer_obj)
{
    const ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    const char* type = ncnn_layer_get_type(layer);
    return mp_obj_new_str(type, strlen(type));
}
#endif /* NCNN_STRING */

mp_obj_t mp_ncnn_layer_get_one_blob_only(mp_obj_t layer_obj)
{
    const ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    return mp_obj_new_int(ncnn_layer_get_one_blob_only(layer));
}
mp_obj_t mp_ncnn_layer_get_support_inplace(mp_obj_t layer_obj)
{
    const ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    return mp_obj_new_int(ncnn_layer_get_support_inplace(layer));
}
mp_obj_t mp_ncnn_layer_get_support_vulkan(mp_obj_t layer_obj)
{
    const ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    return mp_obj_new_int(ncnn_layer_get_support_vulkan(layer));
}
mp_obj_t mp_ncnn_layer_get_support_packing(mp_obj_t layer_obj)
{
    const ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    return mp_obj_new_int(ncnn_layer_get_support_packing(layer));
}
mp_obj_t mp_ncnn_layer_get_support_bf16_storage(mp_obj_t layer_obj)
{
    const ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    return mp_obj_new_int(ncnn_layer_get_support_bf16_storage(layer));
}
mp_obj_t mp_ncnn_layer_get_support_fp16_storage(mp_obj_t layer_obj)
{
    const ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    return mp_obj_new_int(ncnn_layer_get_support_fp16_storage(layer));
}

mp_obj_t mp_ncnn_layer_set_one_blob_only(mp_obj_t layer_obj, mp_obj_t enable_obj)
{
    ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    int enable = mp_obj_get_int(enable_obj);
    ncnn_layer_set_one_blob_only(layer, enable);
    return mp_const_none;
}
mp_obj_t mp_ncnn_layer_set_support_inplace(mp_obj_t layer_obj, mp_obj_t enable_obj)
{
    ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    int enable = mp_obj_get_int(enable_obj);
    ncnn_layer_set_support_inplace(layer, enable);
    return mp_const_none;
}
mp_obj_t mp_ncnn_layer_set_support_vulkan(mp_obj_t layer_obj, mp_obj_t enable_obj)
{
    ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    int enable = mp_obj_get_int(enable_obj);
    ncnn_layer_set_support_vulkan(layer, enable);
    return mp_const_none;
}
mp_obj_t mp_ncnn_layer_set_support_packing(mp_obj_t layer_obj, mp_obj_t enable_obj)
{
    ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    int enable = mp_obj_get_int(enable_obj);
    ncnn_layer_set_support_packing(layer, enable);
    return mp_const_none;
}
mp_obj_t mp_ncnn_layer_set_support_bf16_storage(mp_obj_t layer_obj, mp_obj_t enable_obj)
{
    ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    int enable = mp_obj_get_int(enable_obj);
    ncnn_layer_set_support_bf16_storage(layer, enable);
    return mp_const_none;
}
mp_obj_t mp_ncnn_layer_set_support_fp16_storage(mp_obj_t layer_obj, mp_obj_t enable_obj)
{
    ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    int enable = mp_obj_get_int(enable_obj);
    ncnn_layer_set_support_fp16_storage(layer, enable);
    return mp_const_none;
}

mp_obj_t mp_ncnn_layer_get_bottom_count(mp_obj_t layer_obj)
{
    const ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    return mp_obj_new_int(ncnn_layer_get_bottom_count(layer));
}
mp_obj_t mp_ncnn_layer_get_bottom(mp_obj_t layer_obj, mp_obj_t i_obj)
{
    const ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    int i = mp_obj_get_int(i_obj);
    return mp_obj_new_int(ncnn_layer_get_bottom(layer, i));
}
mp_obj_t mp_ncnn_layer_get_top_count(mp_obj_t layer_obj)
{
    const ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    return mp_obj_new_int(ncnn_layer_get_top_count(layer));
}
mp_obj_t mp_ncnn_layer_get_top(mp_obj_t layer_obj, mp_obj_t i_obj)
{
    const ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(layer_obj);
    int i = mp_obj_get_int(i_obj);
    return mp_obj_new_int(ncnn_layer_get_top(layer, i));
}

mp_obj_t mp_ncnn_blob_get_bottom_shape(size_t n_args, const mp_obj_t* args)
{
    const ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(args[0]);
    int i = mp_obj_get_int(args[1]);
    int dims = (int)mp_obj_get_int(args[2]);
    int w = (int)mp_obj_get_int(args[3]);
    int h = (int)mp_obj_get_int(args[4]);
    int c = (int)mp_obj_get_int(args[5]);
    ncnn_blob_get_bottom_shape(layer, i, &dims, &w, &h, &c);
    return mp_const_none;
}
mp_obj_t mp_ncnn_blob_get_top_shape(size_t n_args, const mp_obj_t* args)
{
    const ncnn_layer_t layer = (ncnn_layer_t)mp_obj_get_int(args[0]);
    int i = mp_obj_get_int(args[1]);
    int dims = (int)mp_obj_get_int(args[2]);
    int w = (int)mp_obj_get_int(args[3]);
    int h = (int)mp_obj_get_int(args[4]);
    int c = (int)mp_obj_get_int(args[5]);
    ncnn_blob_get_top_shape(layer, i, &dims, &w, &h, &c);
    return mp_const_none;
}
}