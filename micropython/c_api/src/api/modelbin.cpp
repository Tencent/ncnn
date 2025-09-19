#include "ncnn/c_api.h"

extern "C" {
#include "py/runtime.h"

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
}