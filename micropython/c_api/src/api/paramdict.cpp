#include "ncnn/c_api.h"

extern "C" {
#include "py/runtime.h"

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
}