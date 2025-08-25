#include "ncnn/c_api.h"

extern "C" {
#include "py/runtime.h"

mp_obj_t mp_ncnn_allocator_create_pool_allocator(void)
{
    ncnn_allocator_t allocator = ncnn_allocator_create_pool_allocator();
    return mp_obj_new_int_from_uint((uintptr_t)allocator);
}
mp_obj_t mp_ncnn_allocator_create_unlocked_pool_allocator(void)
{
    ncnn_allocator_t allocator = ncnn_allocator_create_unlocked_pool_allocator();
    return mp_obj_new_int_from_uint((uintptr_t)allocator);
}
mp_obj_t mp_ncnn_allocator_destroy(mp_obj_t ncnn_allocator_obj)
{
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(ncnn_allocator_obj);
    ncnn_allocator_destroy(allocator);
    return mp_const_none;
}
}
