#include "ncnn/c_api.h"

extern "C" {
#include "py/runtime.h"

#if NCNN_PIXEL
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
}