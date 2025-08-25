#include "ncnn/c_api.h"

extern "C" {
#include "py/runtime.h"

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
}