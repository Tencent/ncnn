#include "ncnn_module.h"

// define function objects
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_version_obj, mp_ncnn_version);

// define option API function objects
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_option_create_obj, mp_ncnn_option_create);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_option_destroy_obj, mp_ncnn_option_destroy);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_option_get_num_threads_obj, mp_ncnn_option_get_num_threads);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_option_set_num_threads_obj, mp_ncnn_option_set_num_threads);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_option_get_use_local_pool_allocator_obj, mp_ncnn_option_get_use_local_pool_allocator);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_option_set_use_local_pool_allocator_obj, mp_ncnn_option_set_use_local_pool_allocator);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_option_set_blob_allocator_obj, mp_ncnn_option_set_blob_allocator);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_option_set_workspace_allocator_obj, mp_ncnn_option_set_workspace_allocator);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_option_get_use_vulkan_compute_obj, mp_ncnn_option_get_use_vulkan_compute);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_option_set_use_vulkan_compute_obj, mp_ncnn_option_set_use_vulkan_compute);

// define mat API function objects
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_mat_create_obj, mp_ncnn_mat_create);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mat_create_1d_obj, mp_ncnn_mat_create_1d);
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_mat_create_2d_obj, mp_ncnn_mat_create_2d);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_create_3d_obj, 4, 4, mp_ncnn_mat_create_3d);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_create_4d_obj, 5, 5, mp_ncnn_mat_create_4d);
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_mat_create_external_1d_obj, mp_ncnn_mat_create_external_1d);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_create_external_2d_obj, 4, 4, mp_ncnn_mat_create_external_2d);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_create_external_3d_obj, 5, 5, mp_ncnn_mat_create_external_3d);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_create_external_4d_obj, 6, 6, mp_ncnn_mat_create_external_4d);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_create_1d_elem_obj, 4, 4, mp_ncnn_mat_create_1d_elem);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_create_2d_elem_obj, 5, 5, mp_ncnn_mat_create_2d_elem);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_create_3d_elem_obj, 6, 6, mp_ncnn_mat_create_3d_elem);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_create_4d_elem_obj, 7, 7, mp_ncnn_mat_create_4d_elem);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_create_external_1d_elem_obj, 5, 5, mp_ncnn_mat_create_external_1d_elem);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_create_external_2d_elem_obj, 6, 6, mp_ncnn_mat_create_external_2d_elem);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_create_external_3d_elem_obj, 7, 7, mp_ncnn_mat_create_external_3d_elem);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_create_external_4d_elem_obj, 8, 8, mp_ncnn_mat_create_external_4d_elem);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_destroy_obj, mp_ncnn_mat_destroy);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mat_fill_float_obj, mp_ncnn_mat_fill_float);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mat_clone_obj, mp_ncnn_mat_clone);
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_mat_reshape_1d_obj, mp_ncnn_mat_reshape_1d);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_reshape_2d_obj, 4, 4, mp_ncnn_mat_reshape_2d);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_reshape_3d_obj, 5, 5, mp_ncnn_mat_reshape_3d);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_reshape_4d_obj, 6, 6, mp_ncnn_mat_reshape_4d);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_dims_obj, mp_ncnn_mat_get_dims);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_w_obj, mp_ncnn_mat_get_w);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_h_obj, mp_ncnn_mat_get_h);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_d_obj, mp_ncnn_mat_get_d);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_c_obj, mp_ncnn_mat_get_c);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_elemsize_obj, mp_ncnn_mat_get_elemsize);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_elempack_obj, mp_ncnn_mat_get_elempack);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_cstep_obj, mp_ncnn_mat_get_cstep);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_mat_get_data_obj, mp_ncnn_mat_get_data);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_mat_get_channel_data_obj, mp_ncnn_mat_get_channel_data);

// globals table
static const mp_rom_map_elem_t ncnn_module_globals_table[] = {
    {MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_ncnn)},
    {MP_ROM_QSTR(MP_QSTR_version), MP_ROM_PTR(&ncnn_version_obj)},

    // Option API
    {MP_ROM_QSTR(MP_QSTR_option_create), MP_ROM_PTR(&ncnn_option_create_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_destroy), MP_ROM_PTR(&ncnn_option_destroy_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_get_num_threads), MP_ROM_PTR(&ncnn_option_get_num_threads_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_set_num_threads), MP_ROM_PTR(&ncnn_option_set_num_threads_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_get_use_local_pool_allocator), MP_ROM_PTR(&ncnn_option_get_use_local_pool_allocator_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_set_use_local_pool_allocator), MP_ROM_PTR(&ncnn_option_set_use_local_pool_allocator_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_set_blob_allocator), MP_ROM_PTR(&ncnn_option_set_blob_allocator_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_set_workspace_allocator), MP_ROM_PTR(&ncnn_option_set_workspace_allocator_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_get_use_vulkan_compute), MP_ROM_PTR(&ncnn_option_get_use_vulkan_compute_obj)},
    {MP_ROM_QSTR(MP_QSTR_option_set_use_vulkan_compute), MP_ROM_PTR(&ncnn_option_set_use_vulkan_compute_obj)},

    // Mat API
    {MP_ROM_QSTR(MP_QSTR_mat_create), MP_ROM_PTR(&ncnn_mat_create_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_1d), MP_ROM_PTR(&ncnn_mat_create_1d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_2d), MP_ROM_PTR(&ncnn_mat_create_2d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_3d), MP_ROM_PTR(&ncnn_mat_create_3d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_4d), MP_ROM_PTR(&ncnn_mat_create_4d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_external_1d), MP_ROM_PTR(&ncnn_mat_create_external_1d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_external_2d), MP_ROM_PTR(&ncnn_mat_create_external_2d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_external_3d), MP_ROM_PTR(&ncnn_mat_create_external_3d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_external_4d), MP_ROM_PTR(&ncnn_mat_create_external_4d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_1d_elem), MP_ROM_PTR(&ncnn_mat_create_1d_elem_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_2d_elem), MP_ROM_PTR(&ncnn_mat_create_2d_elem_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_3d_elem), MP_ROM_PTR(&ncnn_mat_create_3d_elem_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_4d_elem), MP_ROM_PTR(&ncnn_mat_create_4d_elem_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_external_1d_elem), MP_ROM_PTR(&ncnn_mat_create_external_1d_elem_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_external_2d_elem), MP_ROM_PTR(&ncnn_mat_create_external_2d_elem_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_external_3d_elem), MP_ROM_PTR(&ncnn_mat_create_external_3d_elem_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_create_external_4d_elem), MP_ROM_PTR(&ncnn_mat_create_external_4d_elem_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_destroy), MP_ROM_PTR(&ncnn_mat_destroy_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_fill_float), MP_ROM_PTR(&ncnn_mat_fill_float_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_clone), MP_ROM_PTR(&ncnn_mat_clone_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_reshape_1d), MP_ROM_PTR(&ncnn_mat_reshape_1d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_reshape_2d), MP_ROM_PTR(&ncnn_mat_reshape_2d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_reshape_3d), MP_ROM_PTR(&ncnn_mat_reshape_3d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_reshape_4d), MP_ROM_PTR(&ncnn_mat_reshape_4d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_dims), MP_ROM_PTR(&ncnn_mat_get_dims_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_w), MP_ROM_PTR(&ncnn_mat_get_w_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_h), MP_ROM_PTR(&ncnn_mat_get_h_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_d), MP_ROM_PTR(&ncnn_mat_get_d_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_c), MP_ROM_PTR(&ncnn_mat_get_c_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_elemsize), MP_ROM_PTR(&ncnn_mat_get_elemsize_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_elempack), MP_ROM_PTR(&ncnn_mat_get_elempack_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_cstep), MP_ROM_PTR(&ncnn_mat_get_cstep_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_data), MP_ROM_PTR(&ncnn_mat_get_data_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_get_channel_data), MP_ROM_PTR(&ncnn_mat_get_channel_data_obj)},
};

// define the module globals dictionary
static MP_DEFINE_CONST_DICT(ncnn_module_globals, ncnn_module_globals_table);

// define the module object
const mp_obj_module_t ncnn_user_cmodule = {
    .base = {&mp_type_module},
    .globals = (mp_obj_dict_t*)&ncnn_module_globals,
};

// register module
MP_REGISTER_MODULE(MP_QSTR_ncnn, ncnn_user_cmodule);
