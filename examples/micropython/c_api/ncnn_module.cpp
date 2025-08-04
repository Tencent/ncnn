#include "ncnn/c_api.h"

extern "C" {
#include "py/runtime.h"
#include "py/obj.h"
#include "py/objstr.h"
#include <string.h>
}

extern "C" {
static mp_obj_t mp_ncnn_version(void)
{
    const char* ver = ncnn_version();
    return mp_obj_new_str(ver, strlen(ver));
}

/* option api */
static mp_obj_t mp_ncnn_option_create(void)
{
    return mp_obj_new_int_from_uint((uintptr_t)ncnn_option_create());
}
static mp_obj_t mp_ncnn_option_destroy(mp_obj_t ncnn_option_obj)
{
    ncnn_option_destroy((ncnn_option_t)mp_obj_get_int(ncnn_option_obj));
    return mp_const_none;
}
static mp_obj_t mp_ncnn_option_get_num_threads(mp_obj_t ncnn_option_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(ncnn_option_obj);
    return mp_obj_new_int(ncnn_option_get_num_threads(opt));
}
static mp_obj_t mp_ncnn_option_set_num_threads(mp_obj_t ncnn_option_obj, mp_obj_t num_threads_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(ncnn_option_obj);
    int num_threads = mp_obj_get_int(num_threads_obj);
    ncnn_option_set_num_threads(opt, num_threads);
    return mp_const_none;
}
static mp_obj_t mp_ncnn_option_get_use_local_pool_allocator(mp_obj_t ncnn_option_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(ncnn_option_obj);
    return mp_obj_new_int(ncnn_option_get_use_local_pool_allocator(opt));
}
static mp_obj_t mp_ncnn_option_set_use_local_pool_allocator(mp_obj_t ncnn_option_obj, mp_obj_t use_local_pool_allocator_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(ncnn_option_obj);
    int use_local_pool_allocator = mp_obj_get_int(use_local_pool_allocator_obj);
    ncnn_option_set_use_local_pool_allocator(opt, use_local_pool_allocator);
    return mp_const_none;
}
static mp_obj_t mp_ncnn_option_set_blob_allocator(mp_obj_t ncnn_option_obj, mp_obj_t allocator_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(ncnn_option_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_option_set_blob_allocator(opt, allocator);
    return mp_const_none;
}
static mp_obj_t mp_ncnn_option_set_workspace_allocator(mp_obj_t ncnn_option_obj, mp_obj_t allocator_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(ncnn_option_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_option_set_workspace_allocator(opt, allocator);
    return mp_const_none;
}
static mp_obj_t mp_ncnn_option_get_use_vulkan_compute(mp_obj_t ncnn_option_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(ncnn_option_obj);
    return mp_obj_new_int(ncnn_option_get_use_vulkan_compute(opt));
}
static mp_obj_t mp_ncnn_option_set_use_vulkan_compute(mp_obj_t ncnn_option_obj, mp_obj_t use_vulkan_compute_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(ncnn_option_obj);
    int use_vulkan_compute = mp_obj_get_int(use_vulkan_compute_obj);
    ncnn_option_set_use_vulkan_compute(opt, use_vulkan_compute);
    return mp_const_none;
}

/* mat api */
static mp_obj_t mp_ncnn_mat_create(void)
{
    ncnn_mat_t mat = ncnn_mat_create();
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
static mp_obj_t mp_ncnn_mat_create_1d(mp_obj_t w_obj, mp_obj_t allocator_obj)
{
    int w = mp_obj_get_int(w_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_mat_t mat = ncnn_mat_create_1d(w, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
static mp_obj_t mp_ncnn_mat_create_2d(mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t allocator_obj)
{
    int w = mp_obj_get_int(w_obj);
    int h = mp_obj_get_int(h_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_mat_t mat = ncnn_mat_create_2d(w, h, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
// static mp_obj_t mp_ncnn_mat_create_3d(mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t c_obj, mp_obj_t allocator_obj)
// {
//     int w = mp_obj_get_int(w_obj);
//     int h = mp_obj_get_int(h_obj);
//     int c = mp_obj_get_int(c_obj);
//     ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
//     ncnn_mat_t mat = ncnn_mat_create_3d(w, h, c, allocator);
//     return mp_obj_new_int_from_uint((uintptr_t)mat);
// }
// static mp_obj_t mp_ncnn_mat_create_4d(mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t d_obj, mp_obj_t c_obj, mp_obj_t allocator_obj)
// {
//     int w = mp_obj_get_int(w_obj);
//     int h = mp_obj_get_int(h_obj);
//     int d = mp_obj_get_int(d_obj);
//     int c = mp_obj_get_int(c_obj);
//     ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
//     ncnn_mat_t mat = ncnn_mat_create_4d(w, h, d, c, allocator);
//     return mp_obj_new_int_from_uint((uintptr_t)mat);
// }
static mp_obj_t mp_ncnn_mat_create_external_1d(mp_obj_t w_obj, mp_obj_t data_obj, mp_obj_t allocator_obj)
{
    int w = mp_obj_get_int(w_obj);
    void* data = (void*)mp_obj_get_int(data_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_mat_t mat = ncnn_mat_create_external_1d(w, data, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
// static mp_obj_t mp_ncnn_mat_create_external_2d(mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t data_obj, mp_obj_t allocator_obj)
// {
//     int w = mp_obj_get_int(w_obj);
//     int h = mp_obj_get_int(h_obj);
//     void* data = (void*)mp_obj_get_int(data_obj);
//     ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
//     ncnn_mat_t mat = ncnn_mat_create_external_2d(w, h, data, allocator);
//     return mp_obj_new_int_from_uint((uintptr_t)mat);
// }
// static mp_obj_t mp_ncnn_mat_create_external_3d(mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t c_obj, mp_obj_t data_obj, mp_obj_t allocator_obj)
// {
//     int w = mp_obj_get_int(w_obj);
//     int h = mp_obj_get_int(h_obj);
//     int c = mp_obj_get_int(c_obj);
//     void* data = (void*)mp_obj_get_int(data_obj);
//     ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
//     ncnn_mat_t mat = ncnn_mat_create_external_3d(w, h, c, data, allocator);
//     return mp_obj_new_int_from_uint((uintptr_t)mat);
// }
// static mp_obj_t mp_ncnn_mat_create_external_4d(mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t d_obj, mp_obj_t c_obj, mp_obj_t data_obj, mp_obj_t allocator_obj)
// {
//     int w = mp_obj_get_int(w_obj);
//     int h = mp_obj_get_int(h_obj);
//     int d = mp_obj_get_int(d_obj);
//     int c = mp_obj_get_int(c_obj);
//     void* data = (void*)mp_obj_get_int(data_obj);
//     ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
//     ncnn_mat_t mat = ncnn_mat_create_external_4d(w, h, d, c, data, allocator);
//     return mp_obj_new_int_from_uint((uintptr_t)mat);
// }
// static mp_obj_t mp_ncnn_mat_create_1d_elem(mp_obj_t w_obj, mp_obj_t elemsize_obj, mp_obj_t elempack_obj, mp_obj_t allocator_obj)
// {
//     int w = mp_obj_get_int(w_obj);
//     size_t elemsize = (size_t)mp_obj_get_int(elemsize_obj);
//     int elempack = mp_obj_get_int(elempack_obj);
//     ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
//     ncnn_mat_t mat = ncnn_mat_create_1d_elem(w, elemsize, elempack, allocator);
//     return mp_obj_new_int_from_uint((uintptr_t)mat);
// }
// static mp_obj_t mp_ncnn_mat_create_2d_elem(mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t elemsize_obj, mp_obj_t elempack_obj, mp_obj_t allocator_obj)
// {
//     int w = mp_obj_get_int(w_obj);
//     int h = mp_obj_get_int(h_obj);
//     size_t elemsize = (size_t)mp_obj_get_int(elemsize_obj);
//     int elempack = mp_obj_get_int(elempack_obj);
//     ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
//     ncnn_mat_t mat = ncnn_mat_create_2d_elem(w, h, elemsize, elempack, allocator);
//     return mp_obj_new_int_from_uint((uintptr_t)mat);
// }
// static mp_obj_t mp_ncnn_mat_create_3d_elem(mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t c_obj, mp_obj_t elemsize_obj, mp_obj_t elempack_obj, mp_obj_t allocator_obj)
// {
//     int w = mp_obj_get_int(w_obj);
//     int h = mp_obj_get_int(h_obj);
//     int c = mp_obj_get_int(c_obj);
//     size_t elemsize = (size_t)mp_obj_get_int(elemsize_obj);
//     int elempack = mp_obj_get_int(elempack_obj);
//     ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
//     ncnn_mat_t mat = ncnn_mat_create_3d_elem(w, h, c, elemsize, elempack, allocator);
//     return mp_obj_new_int_from_uint((uintptr_t)mat);
// }
// static mp_obj_t mp_ncnn_mat_create_4d_elem(mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t d_obj, mp_obj_t c_obj, mp_obj_t elemsize_obj, mp_obj_t elempack_obj, mp_obj_t allocator_obj)
// {
//     int w = mp_obj_get_int(w_obj);
//     int h = mp_obj_get_int(h_obj);
//     int d = mp_obj_get_int(d_obj);
//     int c = mp_obj_get_int(c_obj);
//     size_t elemsize = (size_t)mp_obj_get_int(elemsize_obj);
//     int elempack = mp_obj_get_int(elempack_obj);
//     ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
//     ncnn_mat_t mat = ncnn_mat_create_4d_elem(w, h, d, c, elemsize, elempack, allocator);
//     return mp_obj_new_int_from_uint((uintptr_t)mat);
// }
// static mp_obj_t mp_ncnn_mat_create_external_1d_elem(mp_obj_t w_obj, mp_obj_t data_obj, mp_obj_t elemsize_obj, mp_obj_t elempack_obj, mp_obj_t allocator_obj)
// {
//     int w = mp_obj_get_int(w_obj);
//     void* data = (void*)mp_obj_get_int(data_obj);
//     size_t elemsize = (size_t)mp_obj_get_int(elemsize_obj);
//     int elempack = mp_obj_get_int(elempack_obj);
//     ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
//     ncnn_mat_t mat = ncnn_mat_create_external_1d_elem(w, data, elemsize, elempack, allocator);
//     return mp_obj_new_int_from_uint((uintptr_t)mat);
// }
// static mp_obj_t mp_ncnn_mat_create_external_2d_elem(mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t data_obj, mp_obj_t elemsize_obj, mp_obj_t elempack_obj, mp_obj_t allocator_obj)
// {
//     int w = mp_obj_get_int(w_obj);
//     int h = mp_obj_get_int(h_obj);
//     void* data = (void*)mp_obj_get_int(data_obj);
//     size_t elemsize = (size_t)mp_obj_get_int(elemsize_obj);
//     int elempack = mp_obj_get_int(elempack_obj);
//     ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
//     ncnn_mat_t mat = ncnn_mat_create_external_2d_elem(w, h, data, elemsize, elempack, allocator);
//     return mp_obj_new_int_from_uint((uintptr_t)mat);
// }
// static mp_obj_t mp_ncnn_mat_create_external_3d_elem(mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t c_obj, mp_obj_t data_obj, mp_obj_t elemsize_obj, mp_obj_t elempack_obj, mp_obj_t allocator_obj)
// {
//     int w = mp_obj_get_int(w_obj);
//     int h = mp_obj_get_int(h_obj);
//     int c = mp_obj_get_int(c_obj);
//     void* data = (void*)mp_obj_get_int(data_obj);
//     size_t elemsize = (size_t)mp_obj_get_int(elemsize_obj);
//     int elempack = mp_obj_get_int(elempack_obj);
//     ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
//     ncnn_mat_t mat = ncnn_mat_create_external_3d_elem(w, h, c, data, elemsize, elempack, allocator);
//     return mp_obj_new_int_from_uint((uintptr_t)mat);
// }
// static mp_obj_t mp_ncnn_mat_create_external_4d_elem(mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t d_obj, mp_obj_t c_obj, mp_obj_t data_obj, mp_obj_t elemsize_obj, mp_obj_t elempack_obj, mp_obj_t allocator_obj)
// {
//     int w = mp_obj_get_int(w_obj);
//     int h = mp_obj_get_int(h_obj);
//     int d = mp_obj_get_int(d_obj);
//     int c = mp_obj_get_int(c_obj);
//     void* data = (void*)mp_obj_get_int(data_obj);
//     size_t elemsize = (size_t)mp_obj_get_int(elemsize_obj);
//     int elempack = mp_obj_get_int(elempack_obj);
//     ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
//     ncnn_mat_t mat = ncnn_mat_create_external_4d_elem(w, h, d, c, data, elemsize, elempack, allocator);
//     return mp_obj_new_int_from_uint((uintptr_t)mat);
// }
static mp_obj_t mp_ncnn_mat_destroy(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_destroy((ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj));
    return mp_const_none;
}

static mp_obj_t mp_ncnn_mat_fill_float(mp_obj_t ncnn_mat_obj, mp_obj_t v_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    float v = (float)mp_obj_get_float(v_obj);
    ncnn_mat_fill_float(mat, v);
    return mp_const_none;
}

static mp_obj_t mp_ncnn_mat_clone(mp_obj_t ncnn_mat_obj, mp_obj_t allocator_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_mat_t cloned_mat = ncnn_mat_clone(mat, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)cloned_mat);
}
static mp_obj_t mp_ncnn_mat_reshape_1d(mp_obj_t ncnn_mat_obj, mp_obj_t w_obj, mp_obj_t allocator_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    int w = mp_obj_get_int(w_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_mat_t reshaped_mat = ncnn_mat_reshape_1d(mat, w, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)reshaped_mat);
}
// static mp_obj_t mp_ncnn_mat_reshape_2d(mp_obj_t ncnn_mat_obj, mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t allocator_obj)
// {
//     ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
//     int w = mp_obj_get_int(w_obj);
//     int h = mp_obj_get_int(h_obj);
//     ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
//     ncnn_mat_t reshaped_mat = ncnn_mat_reshape_2d(mat, w, h, allocator);
//     return mp_obj_new_int_from_uint((uintptr_t)reshaped_mat);
// }
// static mp_obj_t mp_ncnn_mat_reshape_3d(mp_obj_t ncnn_mat_obj, mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t c_obj, mp_obj_t allocator_obj)
// {
//     ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
//     int w = mp_obj_get_int(w_obj);
//     int h = mp_obj_get_int(h_obj);
//     int c = mp_obj_get_int(c_obj);
//     ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
//     ncnn_mat_t reshaped_mat = ncnn_mat_reshape_3d(mat, w, h, c, allocator);
//     return mp_obj_new_int_from_uint((uintptr_t)reshaped_mat);
// }
// static mp_obj_t mp_ncnn_mat_reshape_4d(mp_obj_t ncnn_mat_obj, mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t d_obj, mp_obj_t c_obj, mp_obj_t allocator_obj)
// {
//     ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
//     int w = mp_obj_get_int(w_obj);
//     int h = mp_obj_get_int(h_obj);
//     int d = mp_obj_get_int(d_obj);
//     int c = mp_obj_get_int(c_obj);
//     ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
//     ncnn_mat_t reshaped_mat = ncnn_mat_reshape_4d(mat, w, h, d, c, allocator);
//     return mp_obj_new_int_from_uint((uintptr_t)reshaped_mat);
// }

static mp_obj_t mp_ncnn_mat_get_dims(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    return mp_obj_new_int(ncnn_mat_get_dims(mat));
}
static mp_obj_t mp_ncnn_mat_get_w(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    return mp_obj_new_int(ncnn_mat_get_w(mat));
}
static mp_obj_t mp_ncnn_mat_get_h(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    return mp_obj_new_int(ncnn_mat_get_h(mat));
}
static mp_obj_t mp_ncnn_mat_get_d(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    return mp_obj_new_int(ncnn_mat_get_d(mat));
}
static mp_obj_t mp_ncnn_mat_get_c(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    return mp_obj_new_int(ncnn_mat_get_c(mat));
}
static mp_obj_t mp_ncnn_mat_get_elemsize(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    return mp_obj_new_int(ncnn_mat_get_elemsize(mat));
}
static mp_obj_t mp_ncnn_mat_get_elempack(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    return mp_obj_new_int(ncnn_mat_get_elempack(mat));
}
static mp_obj_t mp_ncnn_mat_get_cstep(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    return mp_obj_new_int(ncnn_mat_get_cstep(mat));
}
static mp_obj_t mp_ncnn_mat_get_data(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    void* data = ncnn_mat_get_data(mat);
    return mp_obj_new_int_from_uint((uintptr_t)data);
}

static mp_obj_t mp_ncnn_mat_get_channel_data(mp_obj_t ncnn_mat_obj, mp_obj_t c_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    int c = mp_obj_get_int(c_obj);
    void* channel_data = ncnn_mat_get_channel_data(mat, c);
    return mp_obj_new_int_from_uint((uintptr_t)channel_data);
}
}

// define function objects
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_version_obj, mp_ncnn_version);

// define option API function objects
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_option_create_obj, mp_ncnn_option_create);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_option_destroy_obj, mp_ncnn_option_destroy);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_option_get_num_threads_obj, mp_ncnn_option_get_num_threads);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_option_set_num_threads_obj, mp_ncnn_option_set_num_threads);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_option_get_use_local_pool_allocator_obj, mp_ncnn_option_get_use_local_pool_allocator);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_option_set_use_local_pool_allocator_obj, mp_ncnn_option_set_use_local_pool_allocator);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_option_set_blob_allocator_obj, mp_ncnn_option_set_blob_allocator);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_option_set_workspace_allocator_obj, mp_ncnn_option_set_workspace_allocator);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_option_get_use_vulkan_compute_obj, mp_ncnn_option_get_use_vulkan_compute);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_option_set_use_vulkan_compute_obj, mp_ncnn_option_set_use_vulkan_compute);

// define mat API function objects
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_mat_create_obj, mp_ncnn_mat_create);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mat_create_1d_obj, mp_ncnn_mat_create_1d);
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_mat_create_2d_obj, mp_ncnn_mat_create_2d);
// static MP_DEFINE_CONST_FUN_OBJ_4(ncnn_mat_create_3d_obj, mp_ncnn_mat_create_3d);
// static MP_DEFINE_CONST_FUN_OBJ_5(ncnn_mat_create_4d_obj, mp_ncnn_mat_create_4d);
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_mat_create_external_1d_obj, mp_ncnn_mat_create_external_1d);
// static MP_DEFINE_CONST_FUN_OBJ_4(ncnn_mat_create_external_2d_obj, mp_ncnn_mat_create_external_2d);
// static MP_DEFINE_CONST_FUN_OBJ_5(ncnn_mat_create_external_3d_obj, mp_ncnn_mat_create_external_3d);
// static MP_DEFINE_CONST_FUN_OBJ_6(ncnn_mat_create_external_4d_obj, mp_ncnn_mat_create_external_4d);
// static MP_DEFINE_CONST_FUN_OBJ_4(ncnn_mat_create_1d_elem_obj, mp_ncnn_mat_create_1d_elem);
// static MP_DEFINE_CONST_FUN_OBJ_4(ncnn_mat_create_2d_elem_obj, mp_ncnn_mat_create_2d_elem);
// static MP_DEFINE_CONST_FUN_OBJ_5(ncnn_mat_create_3d_elem_obj, mp_ncnn_mat_create_3d_elem);
// static MP_DEFINE_CONST_FUN_OBJ_6(ncnn_mat_create_4d_elem_obj, mp_ncnn_mat_create_4d_elem);
// static MP_DEFINE_CONST_FUN_OBJ_5(ncnn_mat_create_external_1d_elem_obj, mp_ncnn_mat_create_external_1d_elem);
// static MP_DEFINE_CONST_FUN_OBJ_4(ncnn_mat_create_external_2d_elem_obj, mp_ncnn_mat_create_external_2d_elem);
// static MP_DEFINE_CONST_FUN_OBJ_5(ncnn_mat_create_external_3d_elem_obj, mp_ncnn_mat_create_external_3d_elem);
// static MP_DEFINE_CONST_FUN_OBJ_6(ncnn_mat_create_external_4d_elem_obj, mp_ncnn_mat_create_external_4d_elem);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_destroy_obj, mp_ncnn_mat_destroy);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mat_fill_float_obj, mp_ncnn_mat_fill_float);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mat_clone_obj, mp_ncnn_mat_clone);
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_mat_reshape_1d_obj, mp_ncnn_mat_reshape_1d);
// static MP_DEFINE_CONST_FUN_OBJ_4(ncnn_mat_reshape_2d_obj, mp_ncnn_mat_reshape_2d);
// static MP_DEFINE_CONST_FUN_OBJ_5(ncnn_mat_reshape_3d_obj, mp_ncnn_mat_reshape_3d);
// static MP_DEFINE_CONST_FUN_OBJ_6(ncnn_mat_reshape_4d_obj, mp_ncnn_mat_reshape_4d);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_dims_obj, mp_ncnn_mat_get_dims);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_w_obj, mp_ncnn_mat_get_w);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_h_obj, mp_ncnn_mat_get_h);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_d_obj, mp_ncnn_mat_get_d);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_c_obj, mp_ncnn_mat_get_c);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_elemsize_obj, mp_ncnn_mat_get_elemsize);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_elempack_obj, mp_ncnn_mat_get_elempack);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_cstep_obj, mp_ncnn_mat_get_cstep);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_data_obj, mp_ncnn_mat_get_data);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mat_get_channel_data_obj, mp_ncnn_mat_get_channel_data);

// globals table
static const mp_rom_map_elem_t ncnn_module_globals_table[] = {
    {MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_ncnn)},
    {MP_ROM_QSTR(MP_QSTR_version), MP_ROM_PTR(&ncnn_version_obj)},

    // Option API
    {MP_ROM_QSTR(MP_QSTR_option_create), MP_ROM_PTR(&ncnn_option_create_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_destroy), MP_ROM_PTR(&ncnn_option_destroy_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_get_num_threads), MP_ROM_PTR(&ncnn_option_get_num_threads_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_set_num_threads), MP_ROM_PTR(&ncnn_option_set_num_threads_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_get_use_local_pool_allocator), MP_ROM_PTR(&ncnn_option_get_use_local_pool_allocator_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_set_use_local_pool_allocator), MP_ROM_PTR(&ncnn_option_set_use_local_pool_allocator_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_set_blob_allocator), MP_ROM_PTR(&ncnn_option_set_blob_allocator_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_set_workspace_allocator), MP_ROM_PTR(&ncnn_option_set_workspace_allocator_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_get_use_vulkan_compute), MP_ROM_PTR(&ncnn_option_get_use_vulkan_compute_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_set_use_vulkan_compute), MP_ROM_PTR(&ncnn_option_set_use_vulkan_compute_obj)},

    // Mat API
    {MP_ROM_QSTR(MP_QSTR_mat_create), MP_ROM_PTR(&ncnn_mat_create_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_1d), MP_ROM_PTR(&ncnn_mat_create_1d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_2d), MP_ROM_PTR(&ncnn_mat_create_2d_obj)},
    // {MP_ROM_QSTR(MP_QSTR_mat_create_3d), MP_ROM_PTR(&ncnn_mat_create_3d_obj)},
    // {MP_ROM_QSTR(MP_QSTR_mat_create_4d), MP_ROM_PTR(&ncnn_mat_create_4d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_external_1d), MP_ROM_PTR(&ncnn_mat_create_external_1d_obj)},
    // {MP_ROM_QSTR(MP_QSTR_mat_create_external_2d), MP_ROM_PTR(&ncnn_mat_create_external_2d_obj)},
    // {MP_ROM_QSTR(MP_QSTR_mat_create_external_3d), MP_ROM_PTR(&ncnn_mat_create_external_3d_obj)},
    // {MP_ROM_QSTR(MP_QSTR_mat_create_external_4d), MP_ROM_PTR(&ncnn_mat_create_external_4d_obj)},
    // {MP_ROM_QSTR(MP_QSTR_mat_create_1d_elem), MP_ROM_PTR(&ncnn_mat_create_1d_elem_obj)},
    // {MP_ROM_QSTR(MP_QSTR_mat_create_2d_elem), MP_ROM_PTR(&ncnn_mat_create_2d_elem_obj)},
    // {MP_ROM_QSTR(MP_QSTR_mat_create_3d_elem), MP_ROM_PTR(&ncnn_mat_create_3d_elem_obj)},
    // {MP_ROM_QSTR(MP_QSTR_mat_create_4d_elem), MP_ROM_PTR(&ncnn_mat_create_4d_elem_obj)},
    // {MP_ROM_QSTR(MP_QSTR_mat_create_external_1d_elem), MP_ROM_PTR(&ncnn_mat_create_external_1d_elem_obj)},
    // {MP_ROM_QSTR(MP_QSTR_mat_create_external_2d_elem), MP_ROM_PTR(&ncnn_mat_create_external_2d_elem_obj)},
    // {MP_ROM_QSTR(MP_QSTR_mat_create_external_3d_elem), MP_ROM_PTR(&ncnn_mat_create_external_3d_elem_obj)},
    // {MP_ROM_QSTR(MP_QSTR_mat_create_external_4d_elem), MP_ROM_PTR(&ncnn_mat_create_external_4d_elem_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_destroy), MP_ROM_PTR(&ncnn_mat_destroy_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_fill_float), MP_ROM_PTR(&ncnn_mat_fill_float_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_clone), MP_ROM_PTR(&ncnn_mat_clone_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_reshape_1d), MP_ROM_PTR(&ncnn_mat_reshape_1d_obj)},
    // {MP_ROM_QSTR(MP_QSTR_mat_reshape_2d), MP_ROM_PTR(&ncnn_mat_reshape_2d_obj)},
    // {MP_ROM_QSTR(MP_QSTR_mat_reshape_3d), MP_ROM_PTR(&ncnn_mat_reshape_3d_obj)},
    // {MP_ROM_QSTR(MP_QSTR_mat_reshape_4d), MP_ROM_PTR(&ncnn_mat_reshape_4d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_dims), MP_ROM_PTR(&ncnn_mat_get_dims_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_w), MP_ROM_PTR(&ncnn_mat_get_w_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_h), MP_ROM_PTR(&ncnn_mat_get_h_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_d), MP_ROM_PTR(&ncnn_mat_get_d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_c), MP_ROM_PTR(&ncnn_mat_get_c_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_elemsize), MP_ROM_PTR(&ncnn_mat_get_elemsize_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_elempack), MP_ROM_PTR(&ncnn_mat_get_elempack_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_cstep), MP_ROM_PTR(&ncnn_mat_get_cstep_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_data), MP_ROM_PTR(&ncnn_mat_get_data_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_channel_data), MP_ROM_PTR(&ncnn_mat_get_channel_data_obj)},
};

// define the module globals dictionary
static MP_DEFINE_CONST_DICT(ncnn_module_globals, ncnn_module_globals_table);

// define the module object
extern "C" const mp_obj_module_t ncnn_user_cmodule = {
    .base = {&mp_type_module},
    .globals = (mp_obj_dict_t*)&ncnn_module_globals,
};

// register module
MP_REGISTER_MODULE(MP_QSTR_ncnn, ncnn_user_cmodule);