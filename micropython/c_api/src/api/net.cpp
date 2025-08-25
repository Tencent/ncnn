#include "ncnn/c_api.h"

extern "C" {
#include "py/runtime.h"

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
}