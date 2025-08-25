#include "ncnn/c_api.h"

extern "C" {
#include "py/runtime.h"

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
}