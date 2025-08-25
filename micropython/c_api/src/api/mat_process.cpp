#include "ncnn/c_api.h"

extern "C" {
#include "py/runtime.h"

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
}