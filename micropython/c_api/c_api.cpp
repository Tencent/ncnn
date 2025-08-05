#include "ncnn/c_api.h"

extern "C" {
#include "ncnn_module.h"
#include "py/objstr.h"
#include <string.h>

mp_obj_t mp_ncnn_version(void)
{
    const char* ver = ncnn_version();
    return mp_obj_new_str(ver, strlen(ver));
}

/* option api */
mp_obj_t mp_ncnn_option_create(void)
{
    return mp_obj_new_int_from_uint((uintptr_t)ncnn_option_create());
}
mp_obj_t mp_ncnn_option_destroy(mp_obj_t ncnn_option_obj)
{
    ncnn_option_destroy((ncnn_option_t)mp_obj_get_int(ncnn_option_obj));
    return mp_const_none;
}
mp_obj_t mp_ncnn_option_get_num_threads(mp_obj_t ncnn_option_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(ncnn_option_obj);
    return mp_obj_new_int(ncnn_option_get_num_threads(opt));
}
mp_obj_t mp_ncnn_option_set_num_threads(mp_obj_t ncnn_option_obj, mp_obj_t num_threads_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(ncnn_option_obj);
    int num_threads = mp_obj_get_int(num_threads_obj);
    ncnn_option_set_num_threads(opt, num_threads);
    return mp_const_none;
}
mp_obj_t mp_ncnn_option_get_use_local_pool_allocator(mp_obj_t ncnn_option_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(ncnn_option_obj);
    return mp_obj_new_int(ncnn_option_get_use_local_pool_allocator(opt));
}
mp_obj_t mp_ncnn_option_set_use_local_pool_allocator(mp_obj_t ncnn_option_obj, mp_obj_t use_local_pool_allocator_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(ncnn_option_obj);
    int use_local_pool_allocator = mp_obj_get_int(use_local_pool_allocator_obj);
    ncnn_option_set_use_local_pool_allocator(opt, use_local_pool_allocator);
    return mp_const_none;
}
mp_obj_t mp_ncnn_option_set_blob_allocator(mp_obj_t ncnn_option_obj, mp_obj_t allocator_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(ncnn_option_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_option_set_blob_allocator(opt, allocator);
    return mp_const_none;
}
mp_obj_t mp_ncnn_option_set_workspace_allocator(mp_obj_t ncnn_option_obj, mp_obj_t allocator_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(ncnn_option_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_option_set_workspace_allocator(opt, allocator);
    return mp_const_none;
}
mp_obj_t mp_ncnn_option_get_use_vulkan_compute(mp_obj_t ncnn_option_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(ncnn_option_obj);
    return mp_obj_new_int(ncnn_option_get_use_vulkan_compute(opt));
}
mp_obj_t mp_ncnn_option_set_use_vulkan_compute(mp_obj_t ncnn_option_obj, mp_obj_t use_vulkan_compute_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(ncnn_option_obj);
    int use_vulkan_compute = mp_obj_get_int(use_vulkan_compute_obj);
    ncnn_option_set_use_vulkan_compute(opt, use_vulkan_compute);
    return mp_const_none;
}

/* mat api */
mp_obj_t mp_ncnn_mat_create(void)
{
    ncnn_mat_t mat = ncnn_mat_create();
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_1d(mp_obj_t w_obj, mp_obj_t allocator_obj)
{
    int w = mp_obj_get_int(w_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_mat_t mat = ncnn_mat_create_1d(w, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_2d(mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t allocator_obj)
{
    int w = mp_obj_get_int(w_obj);
    int h = mp_obj_get_int(h_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_mat_t mat = ncnn_mat_create_2d(w, h, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_3d(size_t n_args, const mp_obj_t* args)
{
    int w = mp_obj_get_int(args[0]);
    int h = mp_obj_get_int(args[1]);
    int c = mp_obj_get_int(args[2]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[3]);
    ncnn_mat_t mat = ncnn_mat_create_3d(w, h, c, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_4d(size_t n_args, const mp_obj_t* args)
{
    int w = mp_obj_get_int(args[0]);
    int h = mp_obj_get_int(args[1]);
    int d = mp_obj_get_int(args[2]);
    int c = mp_obj_get_int(args[3]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[4]);
    ncnn_mat_t mat = ncnn_mat_create_4d(w, h, d, c, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_external_1d(mp_obj_t w_obj, mp_obj_t data_obj, mp_obj_t allocator_obj)
{
    int w = mp_obj_get_int(w_obj);
    void* data = (void*)mp_obj_get_int(data_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_mat_t mat = ncnn_mat_create_external_1d(w, data, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_external_2d(size_t n_args, const mp_obj_t* args)
{
    int w = mp_obj_get_int(args[0]);
    int h = mp_obj_get_int(args[1]);
    void* data = (void*)mp_obj_get_int(args[2]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[3]);
    ncnn_mat_t mat = ncnn_mat_create_external_2d(w, h, data, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_external_3d(size_t n_args, const mp_obj_t* args)
{
    int w = mp_obj_get_int(args[0]);
    int h = mp_obj_get_int(args[1]);
    int c = mp_obj_get_int(args[2]);
    void* data = (void*)mp_obj_get_int(args[3]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[4]);
    ncnn_mat_t mat = ncnn_mat_create_external_3d(w, h, c, data, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_external_4d(size_t n_args, const mp_obj_t* args)
{
    int w = mp_obj_get_int(args[0]);
    int h = mp_obj_get_int(args[1]);
    int d = mp_obj_get_int(args[2]);
    int c = mp_obj_get_int(args[3]);
    void* data = (void*)mp_obj_get_int(args[4]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[5]);
    ncnn_mat_t mat = ncnn_mat_create_external_4d(w, h, d, c, data, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_1d_elem(size_t n_args, const mp_obj_t* args)
{
    int w = mp_obj_get_int(args[0]);
    size_t elemsize = (size_t)mp_obj_get_int(args[1]);
    int elempack = mp_obj_get_int(args[2]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[3]);
    ncnn_mat_t mat = ncnn_mat_create_1d_elem(w, elemsize, elempack, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_2d_elem(size_t n_args, const mp_obj_t* args)
{
    int w = mp_obj_get_int(args[0]);
    int h = mp_obj_get_int(args[1]);
    size_t elemsize = (size_t)mp_obj_get_int(args[2]);
    int elempack = mp_obj_get_int(args[3]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[4]);
    ncnn_mat_t mat = ncnn_mat_create_2d_elem(w, h, elemsize, elempack, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_3d_elem(size_t n_args, const mp_obj_t* args)
{
    int w = mp_obj_get_int(args[0]);
    int h = mp_obj_get_int(args[1]);
    int c = mp_obj_get_int(args[2]);
    size_t elemsize = (size_t)mp_obj_get_int(args[3]);
    int elempack = mp_obj_get_int(args[4]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[5]);
    ncnn_mat_t mat = ncnn_mat_create_3d_elem(w, h, c, elemsize, elempack, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_4d_elem(size_t n_args, const mp_obj_t* args)
{
    int w = mp_obj_get_int(args[0]);
    int h = mp_obj_get_int(args[1]);
    int d = mp_obj_get_int(args[2]);
    int c = mp_obj_get_int(args[3]);
    size_t elemsize = (size_t)mp_obj_get_int(args[4]);
    int elempack = mp_obj_get_int(args[5]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[6]);
    ncnn_mat_t mat = ncnn_mat_create_4d_elem(w, h, d, c, elemsize, elempack, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_external_1d_elem(size_t n_args, const mp_obj_t* args)
{
    int w = mp_obj_get_int(args[0]);
    void* data = (void*)mp_obj_get_int(args[1]);
    size_t elemsize = (size_t)mp_obj_get_int(args[2]);
    int elempack = mp_obj_get_int(args[3]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[4]);
    ncnn_mat_t mat = ncnn_mat_create_external_1d_elem(w, data, elemsize, elempack, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_external_2d_elem(size_t n_args, const mp_obj_t* args)
{
    int w = mp_obj_get_int(args[0]);
    int h = mp_obj_get_int(args[1]);
    void* data = (void*)mp_obj_get_int(args[2]);
    size_t elemsize = (size_t)mp_obj_get_int(args[3]);
    int elempack = mp_obj_get_int(args[4]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[5]);
    ncnn_mat_t mat = ncnn_mat_create_external_2d_elem(w, h, data, elemsize, elempack, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_external_3d_elem(size_t n_args, const mp_obj_t* args)
{
    int w = mp_obj_get_int(args[0]);
    int h = mp_obj_get_int(args[1]);
    int c = mp_obj_get_int(args[2]);
    void* data = (void*)mp_obj_get_int(args[3]);
    size_t elemsize = (size_t)mp_obj_get_int(args[4]);
    int elempack = mp_obj_get_int(args[5]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[6]);
    ncnn_mat_t mat = ncnn_mat_create_external_3d_elem(w, h, c, data, elemsize, elempack, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_external_4d_elem(size_t n_args, const mp_obj_t* args)
{
    int w = mp_obj_get_int(args[0]);
    int h = mp_obj_get_int(args[1]);
    int d = mp_obj_get_int(args[2]);
    int c = mp_obj_get_int(args[3]);
    void* data = (void*)mp_obj_get_int(args[4]);
    size_t elemsize = (size_t)mp_obj_get_int(args[5]);
    int elempack = mp_obj_get_int(args[6]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[7]);
    ncnn_mat_t mat = ncnn_mat_create_external_4d_elem(w, h, d, c, data, elemsize, elempack, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_destroy(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_destroy((ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj));
    return mp_const_none;
}

mp_obj_t mp_ncnn_mat_fill_float(mp_obj_t ncnn_mat_obj, mp_obj_t v_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    float v = (float)mp_obj_get_float(v_obj);
    ncnn_mat_fill_float(mat, v);
    return mp_const_none;
}

mp_obj_t mp_ncnn_mat_clone(mp_obj_t ncnn_mat_obj, mp_obj_t allocator_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_mat_t cloned_mat = ncnn_mat_clone(mat, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)cloned_mat);
}
mp_obj_t mp_ncnn_mat_reshape_1d(mp_obj_t ncnn_mat_obj, mp_obj_t w_obj, mp_obj_t allocator_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    int w = mp_obj_get_int(w_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_mat_t reshaped_mat = ncnn_mat_reshape_1d(mat, w, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)reshaped_mat);
}
mp_obj_t mp_ncnn_mat_reshape_2d(size_t n_args, const mp_obj_t* args)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(args[0]);
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[3]);
    ncnn_mat_t reshaped_mat = ncnn_mat_reshape_2d(mat, w, h, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)reshaped_mat);
}
mp_obj_t mp_ncnn_mat_reshape_3d(size_t n_args, const mp_obj_t* args)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(args[0]);
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int c = mp_obj_get_int(args[3]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[4]);
    ncnn_mat_t reshaped_mat = ncnn_mat_reshape_3d(mat, w, h, c, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)reshaped_mat);
}
mp_obj_t mp_ncnn_mat_reshape_4d(size_t n_args, const mp_obj_t* args)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(args[0]);
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int d = mp_obj_get_int(args[3]);
    int c = mp_obj_get_int(args[4]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[5]);
    ncnn_mat_t reshaped_mat = ncnn_mat_reshape_4d(mat, w, h, d, c, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)reshaped_mat);
}

mp_obj_t mp_ncnn_mat_get_dims(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    return mp_obj_new_int(ncnn_mat_get_dims(mat));
}
mp_obj_t mp_ncnn_mat_get_w(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    return mp_obj_new_int(ncnn_mat_get_w(mat));
}
mp_obj_t mp_ncnn_mat_get_h(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    return mp_obj_new_int(ncnn_mat_get_h(mat));
}
mp_obj_t mp_ncnn_mat_get_d(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    return mp_obj_new_int(ncnn_mat_get_d(mat));
}
mp_obj_t mp_ncnn_mat_get_c(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    return mp_obj_new_int(ncnn_mat_get_c(mat));
}
mp_obj_t mp_ncnn_mat_get_elemsize(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    return mp_obj_new_int(ncnn_mat_get_elemsize(mat));
}
mp_obj_t mp_ncnn_mat_get_elempack(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    return mp_obj_new_int(ncnn_mat_get_elempack(mat));
}
mp_obj_t mp_ncnn_mat_get_cstep(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    return mp_obj_new_int(ncnn_mat_get_cstep(mat));
}
mp_obj_t mp_ncnn_mat_get_data(mp_obj_t ncnn_mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    void* data = ncnn_mat_get_data(mat);
    return mp_obj_new_int_from_uint((uintptr_t)data);
}

mp_obj_t mp_ncnn_mat_get_channel_data(mp_obj_t ncnn_mat_obj, mp_obj_t c_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(ncnn_mat_obj);
    int c = mp_obj_get_int(c_obj);
    void* channel_data = ncnn_mat_get_channel_data(mat, c);
    return mp_obj_new_int_from_uint((uintptr_t)channel_data);
}
}
