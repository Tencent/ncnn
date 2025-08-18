#include "ncnn/c_api.h"

extern "C" {
#include "ncnn_module.h"

mp_obj_t mp_ncnn_version(void)
{
    const char* ver = ncnn_version();
    return mp_obj_new_str(ver, strlen(ver));
}

/* allocator api */
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

/* option api */
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
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(data_obj, &bufinfo, MP_BUFFER_READ);
    void* data = bufinfo.buf;
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_mat_t mat = ncnn_mat_create_external_1d(w, data, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_external_2d(size_t n_args, const mp_obj_t* args)
{
    int w = mp_obj_get_int(args[0]);
    int h = mp_obj_get_int(args[1]);
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[2], &bufinfo, MP_BUFFER_READ);
    void* data = bufinfo.buf;
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[3]);
    ncnn_mat_t mat = ncnn_mat_create_external_2d(w, h, data, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_create_external_3d(size_t n_args, const mp_obj_t* args)
{
    int w = mp_obj_get_int(args[0]);
    int h = mp_obj_get_int(args[1]);
    int c = mp_obj_get_int(args[2]);
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[3], &bufinfo, MP_BUFFER_READ);
    void* data = bufinfo.buf;
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
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[4], &bufinfo, MP_BUFFER_READ);
    void* data = bufinfo.buf;
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
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[4], &bufinfo, MP_BUFFER_READ);
    void* data = bufinfo.buf;
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
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[2], &bufinfo, MP_BUFFER_READ);
    void* data = bufinfo.buf;
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
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[3], &bufinfo, MP_BUFFER_READ);
    void* data = bufinfo.buf;
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
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[4], &bufinfo, MP_BUFFER_READ);
    void* data = bufinfo.buf;
    size_t elemsize = (size_t)mp_obj_get_int(args[5]);
    int elempack = mp_obj_get_int(args[6]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[7]);
    ncnn_mat_t mat = ncnn_mat_create_external_4d_elem(w, h, d, c, data, elemsize, elempack, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_destroy(mp_obj_t mat_obj)
{
    ncnn_mat_destroy((ncnn_mat_t)mp_obj_get_int(mat_obj));
    return mp_const_none;
}

mp_obj_t mp_ncnn_mat_fill_float(mp_obj_t mat_obj, mp_obj_t v_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
    float v = (float)mp_obj_get_float(v_obj);
    ncnn_mat_fill_float(mat, v);
    return mp_const_none;
}

mp_obj_t mp_ncnn_mat_clone(mp_obj_t mat_obj, mp_obj_t allocator_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(allocator_obj);
    ncnn_mat_t cloned_mat = ncnn_mat_clone(mat, allocator);
    return mp_obj_new_int_from_uint((uintptr_t)cloned_mat);
}
mp_obj_t mp_ncnn_mat_reshape_1d(mp_obj_t mat_obj, mp_obj_t w_obj, mp_obj_t allocator_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
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

mp_obj_t mp_ncnn_mat_get_dims(mp_obj_t mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
    return mp_obj_new_int(ncnn_mat_get_dims(mat));
}
mp_obj_t mp_ncnn_mat_get_w(mp_obj_t mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
    return mp_obj_new_int(ncnn_mat_get_w(mat));
}
mp_obj_t mp_ncnn_mat_get_h(mp_obj_t mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
    return mp_obj_new_int(ncnn_mat_get_h(mat));
}
mp_obj_t mp_ncnn_mat_get_d(mp_obj_t mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
    return mp_obj_new_int(ncnn_mat_get_d(mat));
}
mp_obj_t mp_ncnn_mat_get_c(mp_obj_t mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
    return mp_obj_new_int(ncnn_mat_get_c(mat));
}
mp_obj_t mp_ncnn_mat_get_elemsize(mp_obj_t mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
    return mp_obj_new_int(ncnn_mat_get_elemsize(mat));
}
mp_obj_t mp_ncnn_mat_get_elempack(mp_obj_t mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
    return mp_obj_new_int(ncnn_mat_get_elempack(mat));
}
mp_obj_t mp_ncnn_mat_get_cstep(mp_obj_t mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
    return mp_obj_new_int(ncnn_mat_get_cstep(mat));
}
mp_obj_t mp_ncnn_mat_get_data(mp_obj_t mat_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
    void* data = ncnn_mat_get_data(mat);
    return mp_obj_new_int_from_uint((uintptr_t)data);
}

mp_obj_t mp_ncnn_mat_get_channel_data(mp_obj_t mat_obj, mp_obj_t c_obj)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
    int c = mp_obj_get_int(c_obj);
    void* channel_data = ncnn_mat_get_channel_data(mat, c);
    return mp_obj_new_int_from_uint((uintptr_t)channel_data);
}

#if NCNN_PIXEL
/* mat pixel api */
mp_obj_t mp_ncnn_mat_from_pixels(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_READ);
    const unsigned char* pixels = (const unsigned char*)bufinfo.buf;
    int type = mp_obj_get_int(args[1]);
    int w = mp_obj_get_int(args[2]);
    int h = mp_obj_get_int(args[3]);
    int stride = mp_obj_get_int(args[4]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[5]);
    ncnn_mat_t mat = ncnn_mat_from_pixels(pixels, type, w, h, stride, allocator);
    if (mat == NULL)
    {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Failed to create NCNN Mat from pixels"));
    }
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_from_pixels_resize(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int type = mp_obj_get_int(args[1]);
    int w = mp_obj_get_int(args[2]);
    int h = mp_obj_get_int(args[3]);
    int stride = mp_obj_get_int(args[4]);
    int target_width = mp_obj_get_int(args[5]);
    int target_height = mp_obj_get_int(args[6]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[7]);
    ncnn_mat_t mat = ncnn_mat_from_pixels_resize(pixels, type, w, h, stride, target_width, target_height, allocator);
    if (mat == NULL)
    {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Failed to create NCNN Mat from resized pixels"));
    }
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_from_pixels_roi(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int type = mp_obj_get_int(args[1]);
    int w = mp_obj_get_int(args[2]);
    int h = mp_obj_get_int(args[3]);
    int stride = mp_obj_get_int(args[4]);
    int roix = mp_obj_get_int(args[5]);
    int roiy = mp_obj_get_int(args[6]);
    int roiw = mp_obj_get_int(args[7]);
    int roih = mp_obj_get_int(args[8]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[9]);
    ncnn_mat_t mat = ncnn_mat_from_pixels_roi(pixels, type, w, h, stride, roix, roiy, roiw, roih, allocator);
    if (mat == NULL)
    {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Failed to create NCNN Mat from ROI pixels"));
    }
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_from_pixels_roi_resize(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_READ);
    const unsigned char* pixels = (const unsigned char*)bufinfo.buf;
    int type = mp_obj_get_int(args[1]);
    int w = mp_obj_get_int(args[2]);
    int h = mp_obj_get_int(args[3]);
    int stride = mp_obj_get_int(args[4]);
    int roix = mp_obj_get_int(args[5]);
    int roiy = mp_obj_get_int(args[6]);
    int roiw = mp_obj_get_int(args[7]);
    int roih = mp_obj_get_int(args[8]);
    int target_width = mp_obj_get_int(args[9]);
    int target_height = mp_obj_get_int(args[10]);
    ncnn_allocator_t allocator = (ncnn_allocator_t)mp_obj_get_int(args[11]);
    ncnn_mat_t mat = ncnn_mat_from_pixels_roi_resize(pixels, type, w, h, stride, roix, roiy, roiw, roih, target_width, target_height, allocator);
    if (mat == NULL)
    {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Failed to create NCNN Mat from ROI pixels"));
    }
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_mat_to_pixels(size_t n_args, const mp_obj_t* args)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(args[0]);
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[1], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int type = mp_obj_get_int(args[2]);
    int stride = mp_obj_get_int(args[3]);
    ncnn_mat_to_pixels(mat, pixels, type, stride);
    return mp_const_none;
}
mp_obj_t mp_ncnn_mat_to_pixels_resize(size_t n_args, const mp_obj_t* args)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(args[0]);
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[1], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int type = mp_obj_get_int(args[2]);
    int target_width = mp_obj_get_int(args[3]);
    int target_height = mp_obj_get_int(args[4]);
    int target_stride = mp_obj_get_int(args[5]);
    ncnn_mat_to_pixels_resize(mat, pixels, type, target_width, target_height, target_stride);
    return mp_const_none;
}
#endif /* NCNN_PIXEL */

mp_obj_t mp_ncnn_mat_substract_mean_normalize(size_t n_args, const mp_obj_t* args)
{
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(args[0]);
    mp_obj_t mean_list = args[1];
    size_t mean_len;
    mp_obj_t* mean_items;
    mp_obj_list_get(mean_list, &mean_len, &mean_items);

    float* mean_vals = (float*)malloc(mean_len * sizeof(float));
    for (size_t i = 0; i < mean_len; i++)
    {
        mean_vals[i] = (float)mp_obj_get_float(mean_items[i]);
    }

    mp_obj_t norm_list = args[2];
    size_t norm_len;
    mp_obj_t* norm_items;
    mp_obj_list_get(norm_list, &norm_len, &norm_items);

    float* norm_vals = (float*)malloc(norm_len * sizeof(float));
    for (size_t i = 0; i < norm_len; i++)
    {
        norm_vals[i] = (float)mp_obj_get_float(norm_items[i]);
    }
    ncnn_mat_substract_mean_normalize(mat, mean_vals, norm_vals);

    free(mean_vals);
    free(norm_vals);
    return mp_const_none;
}
mp_obj_t mp_ncnn_convert_packing(size_t n_args, const mp_obj_t* args)
{
    const ncnn_mat_t src = (ncnn_mat_t)mp_obj_get_int(args[0]);
    ncnn_mat_t dst = (ncnn_mat_t)mp_obj_get_int(args[1]);
    int elempack = mp_obj_get_int(args[2]);
    const ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(args[3]);
    ncnn_convert_packing(src, &dst, elempack, opt);
    return mp_const_none;
}
mp_obj_t mp_ncnn_flatten(size_t n_args, const mp_obj_t* args)
{
    ncnn_mat_t src = (ncnn_mat_t)mp_obj_get_int(args[0]);
    ncnn_mat_t dst = (ncnn_mat_t)mp_obj_get_int(args[1]);
    const ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(args[2]);
    ncnn_flatten(src, &dst, opt);
    return mp_const_none;
}

/* blob api */
#if NCNN_STRING
mp_obj_t mp_ncnn_blob_get_name(mp_obj_t blob_obj)
{
    const ncnn_blob_t blob = (ncnn_blob_t)mp_obj_get_int(blob_obj);
    return mp_obj_new_str(ncnn_blob_get_name(blob), strlen(ncnn_blob_get_name(blob)));
}
#endif

mp_obj_t mp_ncnn_blob_get_producer(mp_obj_t blob_obj)
{
    const ncnn_blob_t blob = (ncnn_blob_t)mp_obj_get_int(blob_obj);
    return mp_obj_new_int(ncnn_blob_get_producer(blob));
}
mp_obj_t mp_ncnn_blob_get_consumer(mp_obj_t blob_obj)
{
    const ncnn_blob_t blob = (ncnn_blob_t)mp_obj_get_int(blob_obj);
    return mp_obj_new_int(ncnn_blob_get_consumer(blob));
}
mp_obj_t mp_ncnn_blob_get_shape(size_t n_args, const mp_obj_t* args)
{
    const ncnn_blob_t blob = (ncnn_blob_t)mp_obj_get_int(args[0]);
    int dims = mp_obj_get_int(args[1]);
    int w = mp_obj_get_int(args[2]);
    int h = mp_obj_get_int(args[3]);
    int c = mp_obj_get_int(args[4]);
    ncnn_blob_get_shape(blob, &dims, &w, &h, &c);
    return mp_const_none;
}

/* paramdict api */
mp_obj_t mp_ncnn_paramdict_create(void)
{
    return mp_obj_new_int_from_uint((uintptr_t)ncnn_paramdict_create());
}
mp_obj_t mp_ncnn_paramdict_destroy(mp_obj_t pd_obj)
{
    ncnn_paramdict_destroy((ncnn_paramdict_t)mp_obj_get_int(pd_obj));
    return mp_const_none;
}
mp_obj_t mp_ncnn_paramdict_get_type(mp_obj_t pd_obj, mp_obj_t id_obj)
{
    const ncnn_paramdict_t pd = (ncnn_paramdict_t)mp_obj_get_int(pd_obj);
    int id = mp_obj_get_int(id_obj);
    return mp_obj_new_int(ncnn_paramdict_get_type(pd, id));
}
mp_obj_t mp_ncnn_paramdict_get_int(size_t n_args, const mp_obj_t* args)
{
    const ncnn_paramdict_t pd = (ncnn_paramdict_t)mp_obj_get_int(args[0]);
    int id = mp_obj_get_int(args[1]);
    int def = mp_obj_get_int(args[2]);
    return mp_obj_new_int(ncnn_paramdict_get_int(pd, id, def));
}
mp_obj_t mp_ncnn_paramdict_get_float(size_t n_args, const mp_obj_t* args)
{
    const ncnn_paramdict_t pd = (ncnn_paramdict_t)mp_obj_get_int(args[0]);
    int id = mp_obj_get_int(args[1]);
    float def = (float)mp_obj_get_float(args[2]);
    return mp_obj_new_float(ncnn_paramdict_get_float(pd, id, def));
}
mp_obj_t mp_ncnn_paramdict_get_array(size_t n_args, const mp_obj_t* args)
{
    const ncnn_paramdict_t pd = (ncnn_paramdict_t)mp_obj_get_int(args[0]);
    int id = mp_obj_get_int(args[1]);
    const ncnn_mat_t def = (ncnn_mat_t)mp_obj_get_int(args[2]);
    ncnn_mat_t mat = ncnn_paramdict_get_array(pd, id, def);
    if (mat == NULL)
    {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Failed to get NCNN Mat from paramdict"));
    }
    return mp_obj_new_int_from_uint((uintptr_t)mat);
}
mp_obj_t mp_ncnn_paramdict_set_int(size_t n_args, const mp_obj_t* args)
{
    ncnn_paramdict_t pd = (ncnn_paramdict_t)mp_obj_get_int(args[0]);
    int id = mp_obj_get_int(args[1]);
    int i = mp_obj_get_int(args[2]);
    ncnn_paramdict_set_int(pd, id, i);
    return mp_const_none;
}
mp_obj_t mp_ncnn_paramdict_set_float(size_t n_args, const mp_obj_t* args)
{
    ncnn_paramdict_t pd = (ncnn_paramdict_t)mp_obj_get_int(args[0]);
    int id = mp_obj_get_int(args[1]);
    float f = (float)mp_obj_get_float(args[2]);
    ncnn_paramdict_set_float(pd, id, f);
    return mp_const_none;
}
mp_obj_t mp_ncnn_paramdict_set_array(size_t n_args, const mp_obj_t* args)
{
    ncnn_paramdict_t pd = (ncnn_paramdict_t)mp_obj_get_int(args[0]);
    int id = mp_obj_get_int(args[1]);
    const ncnn_mat_t v = (ncnn_mat_t)mp_obj_get_int(args[2]);
    ncnn_paramdict_set_array(pd, id, v);
    return mp_const_none;
}

/* datareader api */
mp_obj_t mp_ncnn_datareader_create(void)
{
    ncnn_datareader_t dr = ncnn_datareader_create();
    return mp_obj_new_int_from_uint((uintptr_t)dr);
}
#if NCNN_STDIO
mp_obj_t mp_ncnn_datareader_create_from_stdio(mp_obj_t fp_obj)
{
    ncnn_datareader_t dr = ncnn_datareader_create_from_stdio((FILE*)mp_obj_get_int(fp_obj));
    return mp_obj_new_int_from_uint((uintptr_t)dr);
}
#endif /* NCNN_STDIO */
mp_obj_t mp_ncnn_datareader_create_from_memory(mp_obj_t mem_obj)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(mem_obj, &bufinfo, MP_BUFFER_READ);
    if (bufinfo.len == 0 || bufinfo.buf == NULL)
    {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Memory buffer is empty or NULL"));
    }
    const unsigned char* mem_ptr = (const unsigned char*)bufinfo.buf;
    const unsigned char** mem = &mem_ptr;
    ncnn_datareader_t dr = ncnn_datareader_create_from_memory(mem);
    return mp_obj_new_int_from_uint((uintptr_t)dr);
}
mp_obj_t mp_ncnn_datareader_destroy(mp_obj_t dr_obj)
{
    ncnn_datareader_destroy((ncnn_datareader_t)mp_obj_get_int(dr_obj));
    return mp_const_none;
}

/* modelbin api */
mp_obj_t mp_ncnn_modelbin_create_from_datareader(mp_obj_t dr_obj)
{
    ncnn_modelbin_t mb = ncnn_modelbin_create_from_datareader((ncnn_datareader_t)mp_obj_get_int(dr_obj));
    return mp_obj_new_int_from_uint((uintptr_t)mb);
}
mp_obj_t mp_ncnn_modelbin_create_from_mat_array(mp_obj_t weights_obj, mp_obj_t n_obj)
{
    int n = mp_obj_get_int(n_obj);
    size_t list_len;
    mp_obj_t* list_items;
    mp_obj_list_get(weights_obj, &list_len, &list_items);

    if ((int)list_len != n)
    {
        mp_raise_ValueError(MP_ERROR_TEXT("Array length mismatch with parameter n"));
    }

    ncnn_mat_t* weights = (ncnn_mat_t*)malloc(n * sizeof(ncnn_mat_t));
    if (weights == NULL)
    {
        mp_raise_msg(&mp_type_MemoryError, MP_ERROR_TEXT("Failed to allocate memory for weights array"));
    }

    for (int i = 0; i < n; i++)
    {
        weights[i] = (ncnn_mat_t)mp_obj_get_int(list_items[i]);
    }
    ncnn_modelbin_t mb = ncnn_modelbin_create_from_mat_array(weights, n);
    free(weights);
    return mp_obj_new_int_from_uint((uintptr_t)mb);
}
mp_obj_t mp_ncnn_modelbin_destroy(mp_obj_t mb_obj)
{
    ncnn_modelbin_destroy((ncnn_modelbin_t)mp_obj_get_int(mb_obj));
    return mp_const_none;
}

/* layer api */
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

/* net api */
mp_obj_t mp_ncnn_net_create(void)
{
    return mp_obj_new_int_from_uint((uintptr_t)ncnn_net_create());
}
mp_obj_t mp_ncnn_net_destroy(mp_obj_t net_obj)
{
    ncnn_net_destroy((ncnn_net_t)mp_obj_get_int(net_obj));
    return mp_const_none;
}
mp_obj_t mp_ncnn_net_get_option(mp_obj_t net_obj)
{
    ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    ncnn_option_t opt = ncnn_net_get_option(net);
    return mp_obj_new_int_from_uint((uintptr_t)opt);
}
mp_obj_t mp_ncnn_net_set_option(mp_obj_t net_obj, mp_obj_t opt_obj)
{
    ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(opt_obj);
    ncnn_net_set_option(net, opt);
    return mp_const_none;
}
#if NCNN_VULKAN
mp_obj_t mp_ncnn_net_set_vulkan_device(mp_obj_t net_obj, mp_obj_t device_index_obj)
{
    ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    int device_index = mp_obj_get_int(device_index_obj);
    ncnn_net_set_vulkan_device(net, device_index);
    return mp_const_none;
}
#endif
#if NCNN_STRING
mp_obj_t mp_ncnn_net_register_custom_layer_by_type(size_t n_args, const mp_obj_t* args)
{
    ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(args[0]);
    const char* type = mp_obj_str_get_str(args[1]);
    ncnn_layer_creator_t creator = (ncnn_layer_creator_t)mp_obj_get_int(args[2]);
    ncnn_layer_destroyer_t destroyer = (ncnn_layer_destroyer_t)mp_obj_get_int(args[3]);
    void* userdata = (void*)mp_obj_get_int(args[4]);
    ncnn_net_register_custom_layer_by_type(net, type, creator, destroyer, userdata);
    return mp_const_none;
}
#endif
mp_obj_t mp_ncnn_net_register_custom_layer_by_typeindex(size_t n_args, const mp_obj_t* args)
{
    ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(args[0]);
    int typeindex = mp_obj_get_int(args[1]);
    ncnn_layer_creator_t creator = (ncnn_layer_creator_t)mp_obj_get_int(args[2]);
    ncnn_layer_destroyer_t destroyer = (ncnn_layer_destroyer_t)mp_obj_get_int(args[3]);
    void* userdata = (void*)mp_obj_get_int(args[4]);
    ncnn_net_register_custom_layer_by_typeindex(net, typeindex, creator, destroyer, userdata);
    return mp_const_none;
}

#if NCNN_STDIO
#if NCNN_STRING
mp_obj_t mp_ncnn_net_load_param(mp_obj_t net_obj, mp_obj_t path_obj)
{
    ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    const char* path = mp_obj_str_get_str(path_obj);
    return mp_obj_new_int(ncnn_net_load_param(net, path));
}
#endif /* NCNN_STRING */
mp_obj_t mp_ncnn_net_load_param_bin(mp_obj_t net_obj, mp_obj_t path_obj)
{
    ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    const char* path = mp_obj_str_get_str(path_obj);
    return mp_obj_new_int(ncnn_net_load_param_bin(net, path));
}
mp_obj_t mp_ncnn_net_load_model(mp_obj_t net_obj, mp_obj_t path_obj)
{
    ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    const char* path = mp_obj_str_get_str(path_obj);
    return mp_obj_new_int(ncnn_net_load_model(net, path));
}
#endif /* NCNN_STDIO */

#if NCNN_STDIO
#if NCNN_STRING
mp_obj_t mp_ncnn_net_load_param_memory(mp_obj_t net_obj, mp_obj_t mem_obj)
{
    ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(mem_obj, &bufinfo, MP_BUFFER_READ);
    if (bufinfo.len == 0 || bufinfo.buf == NULL)
    {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Memory buffer is empty or NULL"));
    }
    return mp_obj_new_int(ncnn_net_load_param_memory(net, (const char*)bufinfo.buf));
}
#endif /* NCNN_STRING */
#endif /* NCNN_STDIO */
mp_obj_t mp_ncnn_net_load_param_bin_memory(mp_obj_t net_obj, mp_obj_t mem_obj)
{
    ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(mem_obj, &bufinfo, MP_BUFFER_READ);
    if (bufinfo.len == 0 || bufinfo.buf == NULL)
    {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Memory buffer is empty or NULL"));
    }
    return mp_obj_new_int(ncnn_net_load_param_bin_memory(net, (const unsigned char*)bufinfo.buf));
}
mp_obj_t mp_ncnn_net_load_model_memory(mp_obj_t net_obj, mp_obj_t mem_obj)
{
    ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(mem_obj, &bufinfo, MP_BUFFER_READ);
    if (bufinfo.len == 0 || bufinfo.buf == NULL)
    {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("Memory buffer is empty or NULL"));
    }
    return mp_obj_new_int(ncnn_net_load_model_memory(net, (const unsigned char*)bufinfo.buf));
}

#if NCNN_STRING
mp_obj_t mp_ncnn_net_load_param_datareader(mp_obj_t net_obj, mp_obj_t dr_obj)
{
    ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    const ncnn_datareader_t dr = (ncnn_datareader_t)mp_obj_get_int(dr_obj);
    return mp_obj_new_int(ncnn_net_load_param_datareader(net, dr));
}
#endif /* NCNN_STRING */
mp_obj_t mp_ncnn_net_load_param_bin_datareader(mp_obj_t net_obj, mp_obj_t dr_obj)
{
    ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    const ncnn_datareader_t dr = (ncnn_datareader_t)mp_obj_get_int(dr_obj);
    return mp_obj_new_int(ncnn_net_load_param_bin_datareader(net, dr));
}
mp_obj_t mp_ncnn_net_load_model_datareader(mp_obj_t net_obj, mp_obj_t dr_obj)
{
    ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    const ncnn_datareader_t dr = (ncnn_datareader_t)mp_obj_get_int(dr_obj);
    return mp_obj_new_int(ncnn_net_load_model_datareader(net, dr));
}

mp_obj_t mp_ncnn_net_clear(mp_obj_t net_obj)
{
    ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    ncnn_net_clear(net);
    return mp_const_none;
}

mp_obj_t mp_ncnn_net_get_input_count(mp_obj_t net_obj)
{
    const ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    return mp_obj_new_int(ncnn_net_get_input_count(net));
}
mp_obj_t mp_ncnn_net_get_output_count(mp_obj_t net_obj)
{
    const ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    return mp_obj_new_int(ncnn_net_get_output_count(net));
}
#if NCNN_STRING
mp_obj_t mp_ncnn_net_get_input_name(mp_obj_t net_obj, mp_obj_t i_obj)
{
    const ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    int i = mp_obj_get_int(i_obj);
    const char* name = ncnn_net_get_input_name(net, i);
    return mp_obj_new_str(name, strlen(name));
}
mp_obj_t mp_ncnn_net_get_output_name(mp_obj_t net_obj, mp_obj_t i_obj)
{
    const ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    int i = mp_obj_get_int(i_obj);
    const char* name = ncnn_net_get_output_name(net, i);
    return mp_obj_new_str(name, strlen(name));
}
#endif /* NCNN_STRING */
mp_obj_t mp_ncnn_net_get_input_index(mp_obj_t net_obj, mp_obj_t i_obj)
{
    const ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    int i = mp_obj_get_int(i_obj);
    return mp_obj_new_int(ncnn_net_get_input_index(net, i));
}
mp_obj_t mp_ncnn_net_get_output_index(mp_obj_t net_obj, mp_obj_t i_obj)
{
    const ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    int i = mp_obj_get_int(i_obj);
    return mp_obj_new_int(ncnn_net_get_output_index(net, i));
}

/* extractor api */
mp_obj_t mp_ncnn_extractor_create(mp_obj_t net_obj)
{
    ncnn_net_t net = (ncnn_net_t)mp_obj_get_int(net_obj);
    return mp_obj_new_int_from_uint((uintptr_t)ncnn_extractor_create(net));
}
mp_obj_t mp_ncnn_extractor_destroy(mp_obj_t ex_obj)
{
    ncnn_extractor_t ex = (ncnn_extractor_t)mp_obj_get_int(ex_obj);
    ncnn_extractor_destroy(ex);
    return mp_const_none;
}
mp_obj_t mp_ncnn_extractor_set_option(mp_obj_t ex_obj, mp_obj_t opt_obj)
{
    ncnn_extractor_t ex = (ncnn_extractor_t)mp_obj_get_int(ex_obj);
    const ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(opt_obj);
    ncnn_extractor_set_option(ex, opt);
    return mp_const_none;
}
#if NCNN_STRING
mp_obj_t mp_ncnn_extractor_input(mp_obj_t ex_obj, mp_obj_t name_obj, mp_obj_t mat_obj)
{
    ncnn_extractor_t ex = (ncnn_extractor_t)mp_obj_get_int(ex_obj);
    const char* name = mp_obj_str_get_str(name_obj);
    const ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
    return mp_obj_new_int(ncnn_extractor_input(ex, name, mat));
}
mp_obj_t mp_ncnn_extractor_extract(mp_obj_t ex_obj, mp_obj_t name_obj, mp_obj_t mat_obj)
{
    ncnn_extractor_t ex = (ncnn_extractor_t)mp_obj_get_int(ex_obj);
    const char* name = mp_obj_str_get_str(name_obj);
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
    return mp_obj_new_int(ncnn_extractor_extract(ex, name, &mat));
}
#endif /* NCNN_STRING */
mp_obj_t mp_ncnn_extractor_input_index(mp_obj_t ex_obj, mp_obj_t index_obj, mp_obj_t mat_obj)
{
    ncnn_extractor_t ex = (ncnn_extractor_t)mp_obj_get_int(ex_obj);
    int index = mp_obj_get_int(index_obj);
    const ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
    return mp_obj_new_int(ncnn_extractor_input_index(ex, index, mat));
}
mp_obj_t mp_ncnn_extractor_extract_index(mp_obj_t ex_obj, mp_obj_t index_obj, mp_obj_t mat_obj)
{
    ncnn_extractor_t ex = (ncnn_extractor_t)mp_obj_get_int(ex_obj);
    int index = mp_obj_get_int(index_obj);
    ncnn_mat_t mat = (ncnn_mat_t)mp_obj_get_int(mat_obj);
    return mp_obj_new_int(ncnn_extractor_extract_index(ex, index, &mat));
}

/* mat process api */
mp_obj_t mp_ncnn_copy_make_border(size_t n_args, const mp_obj_t* args)
{
    const ncnn_mat_t src = (ncnn_mat_t)mp_obj_get_int(args[0]);
    ncnn_mat_t dst = (ncnn_mat_t)mp_obj_get_int(args[1]);
    int top = mp_obj_get_int(args[2]);
    int bottom = mp_obj_get_int(args[3]);
    int left = mp_obj_get_int(args[4]);
    int right = mp_obj_get_int(args[5]);
    int type = mp_obj_get_int(args[6]);
    float v = (float)mp_obj_get_float(args[7]);
    const ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(args[8]);
    ncnn_copy_make_border(src, dst, top, bottom, left, right, type, v, opt);
    return mp_const_none;
}
mp_obj_t mp_ncnn_copy_make_border_3d(size_t n_args, const mp_obj_t* args)
{
    const ncnn_mat_t src = (ncnn_mat_t)mp_obj_get_int(args[0]);
    ncnn_mat_t dst = (ncnn_mat_t)mp_obj_get_int(args[1]);
    int top = mp_obj_get_int(args[2]);
    int bottom = mp_obj_get_int(args[3]);
    int left = mp_obj_get_int(args[4]);
    int right = mp_obj_get_int(args[5]);
    int front = mp_obj_get_int(args[6]);
    int behind = mp_obj_get_int(args[7]);
    int type = mp_obj_get_int(args[8]);
    float v = (float)mp_obj_get_float(args[9]);
    const ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(args[10]);
    ncnn_copy_make_border_3d(src, dst, top, bottom, left, right, front, behind, type, v, opt);
    return mp_const_none;
}
mp_obj_t mp_ncnn_copy_cut_border(size_t n_args, const mp_obj_t* args)
{
    const ncnn_mat_t src = (ncnn_mat_t)mp_obj_get_int(args[0]);
    ncnn_mat_t dst = (ncnn_mat_t)mp_obj_get_int(args[1]);
    int top = mp_obj_get_int(args[2]);
    int bottom = mp_obj_get_int(args[3]);
    int left = mp_obj_get_int(args[4]);
    int right = mp_obj_get_int(args[5]);
    const ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(args[6]);
    ncnn_copy_cut_border(src, dst, top, bottom, left, right, opt);
    return mp_const_none;
}
mp_obj_t mp_ncnn_copy_cut_border_3d(size_t n_args, const mp_obj_t* args)
{
    const ncnn_mat_t src = (ncnn_mat_t)mp_obj_get_int(args[0]);
    ncnn_mat_t dst = (ncnn_mat_t)mp_obj_get_int(args[1]);
    int top = mp_obj_get_int(args[2]);
    int bottom = mp_obj_get_int(args[3]);
    int left = mp_obj_get_int(args[4]);
    int right = mp_obj_get_int(args[5]);
    int front = mp_obj_get_int(args[6]);
    int behind = mp_obj_get_int(args[7]);
    const ncnn_option_t opt = (ncnn_option_t)mp_obj_get_int(args[8]);
    ncnn_copy_cut_border_3d(src, dst, top, bottom, left, right, front, behind, opt);
    return mp_const_none;
}

#if NCNN_PIXEL_DRAWING
/* mat pixel drawing api*/
mp_obj_t mp_ncnn_draw_rectangle_c1(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int rx = mp_obj_get_int(args[3]);
    int ry = mp_obj_get_int(args[4]);
    int rw = mp_obj_get_int(args[5]);
    int rh = mp_obj_get_int(args[6]);
    int color = mp_obj_get_int(args[7]);
    int thickness = mp_obj_get_int(args[8]);
    ncnn_draw_rectangle_c1(pixels, w, h, rx, ry, rw, rh, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_rectangle_c2(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int rx = mp_obj_get_int(args[3]);
    int ry = mp_obj_get_int(args[4]);
    int rw = mp_obj_get_int(args[5]);
    int rh = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    int thickness = mp_obj_get_int(args[8]);
    ncnn_draw_rectangle_c2(pixels, w, h, rx, ry, rw, rh, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_rectangle_c3(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int rx = mp_obj_get_int(args[3]);
    int ry = mp_obj_get_int(args[4]);
    int rw = mp_obj_get_int(args[5]);
    int rh = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    int thickness = mp_obj_get_int(args[8]);
    ncnn_draw_rectangle_c3(pixels, w, h, rx, ry, rw, rh, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_rectangle_c4(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int rx = mp_obj_get_int(args[3]);
    int ry = mp_obj_get_int(args[4]);
    int rw = mp_obj_get_int(args[5]);
    int rh = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    int thickness = mp_obj_get_int(args[8]);
    ncnn_draw_rectangle_c4(pixels, w, h, rx, ry, rw, rh, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_text_c1(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    const char* text = mp_obj_str_get_str(args[3]);
    int x = mp_obj_get_int(args[4]);
    int y = mp_obj_get_int(args[5]);
    int fontpixelsize = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    ncnn_draw_text_c1(pixels, w, h, text, x, y, fontpixelsize, color);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_text_c2(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    const char* text = mp_obj_str_get_str(args[3]);
    int x = mp_obj_get_int(args[4]);
    int y = mp_obj_get_int(args[5]);
    int fontpixelsize = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    ncnn_draw_text_c2(pixels, w, h, text, x, y, fontpixelsize, color);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_text_c3(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    const char* text = mp_obj_str_get_str(args[3]);
    int x = mp_obj_get_int(args[4]);
    int y = mp_obj_get_int(args[5]);
    int fontpixelsize = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    ncnn_draw_text_c3(pixels, w, h, text, x, y, fontpixelsize, color);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_text_c4(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    const char* text = mp_obj_str_get_str(args[3]);
    int x = mp_obj_get_int(args[4]);
    int y = mp_obj_get_int(args[5]);
    int fontpixelsize = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    ncnn_draw_text_c4(pixels, w, h, text, x, y, fontpixelsize, color);
    return mp_const_none;
}

mp_obj_t mp_ncnn_draw_circle_c1(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int cx = mp_obj_get_int(args[3]);
    int cy = mp_obj_get_int(args[4]);
    int radius = mp_obj_get_int(args[5]);
    unsigned int color = mp_obj_get_int(args[6]);
    int thickness = mp_obj_get_int(args[7]);
    ncnn_draw_circle_c1(pixels, w, h, cx, cy, radius, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_circle_c2(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int cx = mp_obj_get_int(args[3]);
    int cy = mp_obj_get_int(args[4]);
    int radius = mp_obj_get_int(args[5]);
    unsigned int color = mp_obj_get_int(args[6]);
    int thickness = mp_obj_get_int(args[7]);
    ncnn_draw_circle_c2(pixels, w, h, cx, cy, radius, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_circle_c3(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int cx = mp_obj_get_int(args[3]);
    int cy = mp_obj_get_int(args[4]);
    int radius = mp_obj_get_int(args[5]);
    unsigned int color = mp_obj_get_int(args[6]);
    int thickness = mp_obj_get_int(args[7]);
    ncnn_draw_circle_c3(pixels, w, h, cx, cy, radius, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_circle_c4(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int cx = mp_obj_get_int(args[3]);
    int cy = mp_obj_get_int(args[4]);
    int radius = mp_obj_get_int(args[5]);
    unsigned int color = mp_obj_get_int(args[6]);
    int thickness = mp_obj_get_int(args[7]);
    ncnn_draw_circle_c4(pixels, w, h, cx, cy, radius, color, thickness);
    return mp_const_none;
}

mp_obj_t mp_ncnn_draw_line_c1(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int x0 = mp_obj_get_int(args[3]);
    int y0 = mp_obj_get_int(args[4]);
    int x1 = mp_obj_get_int(args[5]);
    int y1 = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    int thickness = mp_obj_get_int(args[8]);
    ncnn_draw_line_c1(pixels, w, h, x0, y0, x1, y1, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_line_c2(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int x0 = mp_obj_get_int(args[3]);
    int y0 = mp_obj_get_int(args[4]);
    int x1 = mp_obj_get_int(args[5]);
    int y1 = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    int thickness = mp_obj_get_int(args[8]);
    ncnn_draw_line_c2(pixels, w, h, x0, y0, x1, y1, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_line_c3(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int x0 = mp_obj_get_int(args[3]);
    int y0 = mp_obj_get_int(args[4]);
    int x1 = mp_obj_get_int(args[5]);
    int y1 = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    int thickness = mp_obj_get_int(args[8]);
    ncnn_draw_line_c3(pixels, w, h, x0, y0, x1, y1, color, thickness);
    return mp_const_none;
}
mp_obj_t mp_ncnn_draw_line_c4(size_t n_args, const mp_obj_t* args)
{
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_WRITE);
    unsigned char* pixels = (unsigned char*)bufinfo.buf;
    int w = mp_obj_get_int(args[1]);
    int h = mp_obj_get_int(args[2]);
    int x0 = mp_obj_get_int(args[3]);
    int y0 = mp_obj_get_int(args[4]);
    int x1 = mp_obj_get_int(args[5]);
    int y1 = mp_obj_get_int(args[6]);
    unsigned int color = mp_obj_get_int(args[7]);
    int thickness = mp_obj_get_int(args[8]);
    ncnn_draw_line_c4(pixels, w, h, x0, y0, x1, y1, color, thickness);
    return mp_const_none;
}
#endif /* NCNN_PIXEL_DRAWING */
}
