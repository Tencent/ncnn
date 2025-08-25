#include "ncnn/c_api.h"

extern "C" {
#include "py/runtime.h"

mp_obj_t mp_ncnn_option_create(void)
{
    return mp_obj_new_int_from_uint((uintptr_t)ncnn_option_create());
}
mp_obj_t mp_ncnn_option_destroy(mp_obj_t option_obj)
{
    ncnn_option_destroy((ncnn_option_t)mp_obj_get_int(option_obj));
    return mp_const_none;
}
mp_obj_t mp_ncnn_option_get_num_threads(mp_obj_t option_obj)
{
    const ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(option_obj);
    return mp_obj_new_int(ncnn_option_get_num_threads(opt));
}
mp_obj_t mp_ncnn_option_set_num_threads(mp_obj_t option_obj, mp_obj_t num_threads_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(option_obj);
    int num_threads = mp_obj_get_int(num_threads_obj);
    ncnn_option_set_num_threads(opt, num_threads);
    return mp_const_none;
}
mp_obj_t mp_ncnn_option_get_use_local_pool_allocator(mp_obj_t option_obj)
{
    const ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(option_obj);
    return mp_obj_new_int(ncnn_option_get_use_local_pool_allocator(opt));
}
mp_obj_t mp_ncnn_option_set_use_local_pool_allocator(mp_obj_t option_obj, mp_obj_t use_local_pool_allocator_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(option_obj);
    int use_local_pool_allocator = mp_obj_get_int(use_local_pool_allocator_obj);
    ncnn_option_set_use_local_pool_allocator(opt, use_local_pool_allocator);
    return mp_const_none;
}
mp_obj_t mp_ncnn_option_set_blob_allocator(mp_obj_t option_obj, mp_obj_t allocator_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(option_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_option_set_blob_allocator(opt, allocator);
    return mp_const_none;
}
mp_obj_t mp_ncnn_option_set_workspace_allocator(mp_obj_t option_obj, mp_obj_t allocator_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(option_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_option_set_workspace_allocator(opt, allocator);
    return mp_const_none;
}
mp_obj_t mp_ncnn_option_get_use_vulkan_compute(mp_obj_t option_obj)
{
    const ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(option_obj);
    return mp_obj_new_int(ncnn_option_get_use_vulkan_compute(opt));
}
mp_obj_t mp_ncnn_option_set_use_vulkan_compute(mp_obj_t option_obj, mp_obj_t use_vulkan_compute_obj)
{
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(option_obj);
    int use_vulkan_compute = mp_obj_get_int(use_vulkan_compute_obj);
    ncnn_option_set_use_vulkan_compute(opt, use_vulkan_compute);
    return mp_const_none;
}
}