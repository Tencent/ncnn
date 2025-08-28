#include "ncnn/c_api.h"

extern "C" {
#include "py/runtime.h"

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
}