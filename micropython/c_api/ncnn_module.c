#include "ncnn_module.h"

/* define function objects */
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_version_obj, mp_ncnn_version);

/* allocator api */
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_allocator_create_pool_allocator_obj, mp_ncnn_allocator_create_pool_allocator);
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_allocator_create_unlocked_pool_allocator_obj, mp_ncnn_allocator_create_unlocked_pool_allocator);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_allocator_destroy_obj, mp_ncnn_allocator_destroy);

/* option api */
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

/* mat api */
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

/* mat pixel api */
#if NCNN_PIXEL
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_from_pixels_obj, 6, 6, mp_ncnn_mat_from_pixels);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_from_pixels_resize_obj, 8, 8, mp_ncnn_mat_from_pixels_resize);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_from_pixels_roi_obj, 10, 10, mp_ncnn_mat_from_pixels_roi);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_from_pixels_roi_resize_obj, 12, 12, mp_ncnn_mat_from_pixels_roi_resize);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_to_pixels_obj, 4, 4, mp_ncnn_mat_to_pixels);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_to_pixels_resize_obj, 6, 6, mp_ncnn_mat_to_pixels_resize);
#endif

static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_mat_substract_mean_normalize_obj, 3, 3, mp_ncnn_mat_substract_mean_normalize);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_convert_packing_obj, 4, 4, mp_ncnn_convert_packing);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_flatten_obj, 3, 3, mp_ncnn_flatten);

/* blob api */
#if NCNN_STRING
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_blob_get_name_obj, mp_ncnn_blob_get_name);
#endif /* NCNN_STRING */

static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_blob_get_producer_obj, mp_ncnn_blob_get_producer);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_blob_get_consumer_obj, mp_ncnn_blob_get_consumer);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_blob_get_shape_obj, 5, 5, mp_ncnn_blob_get_shape);

/* paramdict api */
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_paramdict_create_obj, mp_ncnn_paramdict_create);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_paramdict_destroy_obj, mp_ncnn_paramdict_destroy);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_paramdict_get_type_obj, mp_ncnn_paramdict_get_type);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_paramdict_get_int_obj, 3, 3, mp_ncnn_paramdict_get_int);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_paramdict_get_float_obj, 3, 3, mp_ncnn_paramdict_get_float);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_paramdict_get_array_obj, 3, 3, mp_ncnn_paramdict_get_array);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_paramdict_set_int_obj, 3, 3, mp_ncnn_paramdict_set_int);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_paramdict_set_float_obj, 3, 3, mp_ncnn_paramdict_set_float);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_paramdict_set_array_obj, 3, 3, mp_ncnn_paramdict_set_array);

/* datareader api */
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_datareader_create_obj, mp_ncnn_datareader_create);
#if NCNN_STDIO
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_datareader_create_from_stdio_obj, mp_ncnn_datareader_create_from_stdio);
#endif /* NCNN_STDIO */
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_datareader_create_from_memory_obj, mp_ncnn_datareader_create_from_memory);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_datareader_destroy_obj, mp_ncnn_datareader_destroy);

/* modelbin api */
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_modelbin_create_from_datareader_obj, mp_ncnn_modelbin_create_from_datareader);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_modelbin_create_from_mat_array_obj, mp_ncnn_modelbin_create_from_mat_array);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_modelbin_destroy_obj, mp_ncnn_modelbin_destroy);

/* layer api */
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_layer_create_obj, mp_ncnn_layer_create);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_layer_create_by_typeindex_obj, mp_ncnn_layer_create_by_typeindex);
#if NCNN_STRING
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_layer_create_by_type_obj, mp_ncnn_layer_create_by_type);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_layer_type_to_index_obj, mp_ncnn_layer_type_to_index);
#endif /* NCNN_STRING */
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_layer_destroy_obj, mp_ncnn_layer_destroy);

#if NCNN_STRING
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_layer_get_name_obj, mp_ncnn_layer_get_name);
#endif /* NCNN_STRING */

static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_layer_get_typeindex_obj, mp_ncnn_layer_get_typeindex);

#if NCNN_STRING
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_layer_get_type_obj, mp_ncnn_layer_get_type);
#endif /* NCNN_STRING */

static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_layer_get_one_blob_only_obj, mp_ncnn_layer_get_one_blob_only);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_layer_get_support_inplace_obj, mp_ncnn_layer_get_support_inplace);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_layer_get_support_vulkan_obj, mp_ncnn_layer_get_support_vulkan);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_layer_get_support_packing_obj, mp_ncnn_layer_get_support_packing);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_layer_get_support_bf16_storage_obj, mp_ncnn_layer_get_support_bf16_storage);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_layer_get_support_fp16_storage_obj, mp_ncnn_layer_get_support_fp16_storage);

static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_layer_set_one_blob_only_obj, mp_ncnn_layer_set_one_blob_only);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_layer_set_support_inplace_obj, mp_ncnn_layer_set_support_inplace);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_layer_set_support_vulkan_obj, mp_ncnn_layer_set_support_vulkan);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_layer_set_support_packing_obj, mp_ncnn_layer_set_support_packing);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_layer_set_support_bf16_storage_obj, mp_ncnn_layer_set_support_bf16_storage);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_layer_set_support_fp16_storage_obj, mp_ncnn_layer_set_support_fp16_storage);

static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_layer_get_bottom_count_obj, mp_ncnn_layer_get_bottom_count);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_layer_get_bottom_obj, mp_ncnn_layer_get_bottom);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_layer_get_top_count_obj, mp_ncnn_layer_get_top_count);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_layer_get_top_obj, mp_ncnn_layer_get_top);

static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_blob_get_bottom_shape_obj, 6, 6, mp_ncnn_blob_get_bottom_shape);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_blob_get_top_shape_obj, 6, 6, mp_ncnn_blob_get_top_shape);

/* net api */
static MP_DEFINE_CONST_FUN_OBJ_0(ncnn_net_create_obj, mp_ncnn_net_create);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_net_destroy_obj, mp_ncnn_net_destroy);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_net_get_option_obj, mp_ncnn_net_get_option);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_set_option_obj, mp_ncnn_net_set_option);

#if NCNN_VULKAN
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_set_vulkan_device_obj, mp_ncnn_net_set_vulkan_device);
#endif

#if NCNN_STRING
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_net_register_custom_layer_by_type_obj, 5, 5, mp_ncnn_net_register_custom_layer_by_type);
#endif /* NCNN_STRING */
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_net_register_custom_layer_by_typeindex_obj, 5, 5, mp_ncnn_net_register_custom_layer_by_typeindex);

#if NCNN_STDIO
#if NCNN_STRING
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_load_param_obj, mp_ncnn_net_load_param);
#endif /* NCNN_STRING */
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_load_param_bin_obj, mp_ncnn_net_load_param_bin);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_load_model_obj, mp_ncnn_net_load_model);
#endif /* NCNN_STDIO */

#if NCNN_STDIO
#if NCNN_STRING
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_load_param_memory_obj, mp_ncnn_net_load_param_memory);
#endif /* NCNN_STRING */
#endif /* NCNN_STDIO */
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_load_param_bin_memory_obj, mp_ncnn_net_load_param_bin_memory);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_load_model_memory_obj, mp_ncnn_net_load_model_memory);

#if NCNN_STRING
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_load_param_datareader_obj, mp_ncnn_net_load_param_datareader);
#endif /* NCNN_STRING */
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_load_param_bin_datareader_obj, mp_ncnn_net_load_param_bin_datareader);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_load_model_datareader_obj, mp_ncnn_net_load_model_datareader);

static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_net_clear_obj, mp_ncnn_net_clear);

static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_net_get_input_count_obj, mp_ncnn_net_get_input_count);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_net_get_output_count_obj, mp_ncnn_net_get_output_count);

#if NCNN_STRING
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_get_input_name_obj, mp_ncnn_net_get_input_name);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_get_output_name_obj, mp_ncnn_net_get_output_name);
#endif /* NCNN_STRING */
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_get_input_index_obj, mp_ncnn_net_get_input_index);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_net_get_output_index_obj, mp_ncnn_net_get_output_index);

/* extractor api */
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_extractor_create_obj, mp_ncnn_extractor_create);
static MP_DEFINE_CONST_FUN_OBJ_1(ncnn_extractor_destroy_obj, mp_ncnn_extractor_destroy);
static MP_DEFINE_CONST_FUN_OBJ_2(ncnn_extractor_set_option_obj, mp_ncnn_extractor_set_option);

#if NCNN_STRING
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_extractor_input_obj, mp_ncnn_extractor_input);
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_extractor_extract_obj, mp_ncnn_extractor_extract);
#endif /* NCNN_STRING */
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_extractor_input_index_obj, mp_ncnn_extractor_input_index);
static MP_DEFINE_CONST_FUN_OBJ_3(ncnn_extractor_extract_index_obj, mp_ncnn_extractor_extract_index);

/* mat process api */
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_copy_make_border_obj, 9, 9, mp_ncnn_copy_make_border);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_copy_make_border_3d_obj, 11, 11, mp_ncnn_copy_make_border_3d);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_copy_cut_border_obj, 7, 7, mp_ncnn_copy_cut_border);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_copy_cut_border_3d_obj, 9, 9, mp_ncnn_copy_cut_border_3d);

#if NCNN_PIXEL_DRAWING
/* mat pixel drawing api*/
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_draw_rectangle_c1_obj, 9, 9, mp_ncnn_draw_rectangle_c1);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_draw_rectangle_c2_obj, 9, 9, mp_ncnn_draw_rectangle_c2);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_draw_rectangle_c3_obj, 9, 9, mp_ncnn_draw_rectangle_c3);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_draw_rectangle_c4_obj, 9, 9, mp_ncnn_draw_rectangle_c4);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_draw_text_c1_obj, 8, 8, mp_ncnn_draw_text_c1);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_draw_text_c2_obj, 8, 8, mp_ncnn_draw_text_c2);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_draw_text_c3_obj, 8, 8, mp_ncnn_draw_text_c3);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_draw_text_c4_obj, 8, 8, mp_ncnn_draw_text_c4);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_draw_circle_c1_obj, 8, 8, mp_ncnn_draw_circle_c1);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_draw_circle_c2_obj, 8, 8, mp_ncnn_draw_circle_c2);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_draw_circle_c3_obj, 8, 8, mp_ncnn_draw_circle_c3);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_draw_circle_c4_obj, 8, 8, mp_ncnn_draw_circle_c4);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_draw_line_c1_obj, 9, 9, mp_ncnn_draw_line_c1);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_draw_line_c2_obj, 9, 9, mp_ncnn_draw_line_c2);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_draw_line_c3_obj, 9, 9, mp_ncnn_draw_line_c3);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ncnn_draw_line_c4_obj, 9, 9, mp_ncnn_draw_line_c4);
#endif /* NCNN_PIXEL_DRAWING */

/* globals table */
static const mp_rom_map_elem_t ncnn_module_globals_table[] = {
    {MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_ncnn)},
    {MP_ROM_QSTR(MP_QSTR_version), MP_ROM_PTR(&ncnn_version_obj)},

    /* allocator API */
    {MP_ROM_QSTR(MP_QSTR_allocator_create_pool_allocator), MP_ROM_PTR(&ncnn_allocator_create_pool_allocator_obj)},
    {MP_ROM_QSTR(MP_QSTR_allocator_create_unlocked_pool_allocator), MP_ROM_PTR(&ncnn_allocator_create_unlocked_pool_allocator_obj)},
    {MP_ROM_QSTR(MP_QSTR_allocator_destroy), MP_ROM_PTR(&ncnn_allocator_destroy_obj)},

    /* option API */
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

    /* mat API */
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

#if NCNN_PIXEL
    /* mat pixel api */
    {MP_ROM_QSTR(MP_QSTR_mat_from_pixels), MP_ROM_PTR(&ncnn_mat_from_pixels_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_from_pixels_resize), MP_ROM_PTR(&ncnn_mat_from_pixels_resize_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_from_pixels_roi), MP_ROM_PTR(&ncnn_mat_from_pixels_roi_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_from_pixels_roi_resize), MP_ROM_PTR(&ncnn_mat_from_pixels_roi_resize_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_to_pixels), MP_ROM_PTR(&ncnn_mat_to_pixels_obj)},
    {MP_ROM_QSTR(MP_QSTR_mat_to_pixels_resize), MP_ROM_PTR(&ncnn_mat_to_pixels_resize_obj)},
#endif

    {MP_ROM_QSTR(MP_QSTR_mat_substract_mean_normalize), MP_ROM_PTR(&ncnn_mat_substract_mean_normalize_obj)},
    {MP_ROM_QSTR(MP_QSTR_convert_packing), MP_ROM_PTR(&ncnn_convert_packing_obj)},
    {MP_ROM_QSTR(MP_QSTR_flatten), MP_ROM_PTR(&ncnn_flatten_obj)},

/* blob api */
#if NCNN_STRING
    {MP_ROM_QSTR(MP_QSTR_blob_get_name), MP_ROM_PTR(&ncnn_blob_get_name_obj)},
#endif

    {MP_ROM_QSTR(MP_QSTR_blob_get_producer), MP_ROM_PTR(&ncnn_blob_get_producer_obj)},
    {MP_ROM_QSTR(MP_QSTR_blob_get_consumer), MP_ROM_PTR(&ncnn_blob_get_consumer_obj)},

    {MP_ROM_QSTR(MP_QSTR_blob_get_shape), MP_ROM_PTR(&ncnn_blob_get_shape_obj)},

    /* paramdict api */
    {MP_ROM_QSTR(MP_QSTR_paramdict_create), MP_ROM_PTR(&ncnn_paramdict_create_obj)},
    {MP_ROM_QSTR(MP_QSTR_paramdict_destroy), MP_ROM_PTR(&ncnn_paramdict_destroy_obj)},
    {MP_ROM_QSTR(MP_QSTR_paramdict_get_type), MP_ROM_PTR(&ncnn_paramdict_get_type_obj)},
    {MP_ROM_QSTR(MP_QSTR_paramdict_get_int), MP_ROM_PTR(&ncnn_paramdict_get_int_obj)},
    {MP_ROM_QSTR(MP_QSTR_paramdict_get_float), MP_ROM_PTR(&ncnn_paramdict_get_float_obj)},
    {MP_ROM_QSTR(MP_QSTR_paramdict_get_array), MP_ROM_PTR(&ncnn_paramdict_get_array_obj)},
    {MP_ROM_QSTR(MP_QSTR_paramdict_set_int), MP_ROM_PTR(&ncnn_paramdict_set_int_obj)},
    {MP_ROM_QSTR(MP_QSTR_paramdict_set_float), MP_ROM_PTR(&ncnn_paramdict_set_float_obj)},
    {MP_ROM_QSTR(MP_QSTR_paramdict_set_array), MP_ROM_PTR(&ncnn_paramdict_set_array_obj)},

    /* datareader api */
    {MP_ROM_QSTR(MP_QSTR_datareader_create), MP_ROM_PTR(&ncnn_datareader_create_obj)},
#if NCNN_STDIO
    {MP_ROM_QSTR(MP_QSTR_datareader_create_from_stdio), MP_ROM_PTR(&ncnn_datareader_create_from_stdio_obj)},
#endif /* NCNN_STDIO */
    {MP_ROM_QSTR(MP_QSTR_datareader_create_from_memory), MP_ROM_PTR(&ncnn_datareader_create_from_memory_obj)},
    {MP_ROM_QSTR(MP_QSTR_datareader_destroy), MP_ROM_PTR(&ncnn_datareader_destroy_obj)},

    /* modelbin api */
    {MP_ROM_QSTR(MP_QSTR_modelbin_create_from_datareader), MP_ROM_PTR(&ncnn_modelbin_create_from_datareader_obj)},
    {MP_ROM_QSTR(MP_QSTR_modelbin_create_from_mat_array), MP_ROM_PTR(&ncnn_modelbin_create_from_mat_array_obj)},
    {MP_ROM_QSTR(MP_QSTR_modelbin_destroy), MP_ROM_PTR(&ncnn_modelbin_destroy_obj)},

    /* layer api */
    {MP_ROM_QSTR(MP_QSTR_layer_create), MP_ROM_PTR(&ncnn_layer_create_obj)},
    {MP_ROM_QSTR(MP_QSTR_layer_create_by_typeindex), MP_ROM_PTR(&ncnn_layer_create_by_typeindex_obj)},
#if NCNN_STRING
    {MP_ROM_QSTR(MP_QSTR_layer_create_by_type), MP_ROM_PTR(&ncnn_layer_create_by_type_obj)},
    {MP_ROM_QSTR(MP_QSTR_layer_type_to_index), MP_ROM_PTR(&ncnn_layer_type_to_index_obj)},
#endif /* NCNN_STRING */
    {MP_ROM_QSTR(MP_QSTR_layer_destroy), MP_ROM_PTR(&ncnn_layer_destroy_obj)},

#if NCNN_STRING
    {MP_ROM_QSTR(MP_QSTR_layer_get_name), MP_ROM_PTR(&ncnn_layer_get_name_obj)},
#endif /* NCNN_STRING */

    {MP_ROM_QSTR(MP_QSTR_layer_get_typeindex), MP_ROM_PTR(&ncnn_layer_get_typeindex_obj)},
#if NCNN_STRING
    {MP_ROM_QSTR(MP_QSTR_layer_get_type), MP_ROM_PTR(&ncnn_layer_get_type_obj)},
#endif /* NCNN_STRING */

    {MP_ROM_QSTR(MP_QSTR_layer_get_one_blob_only), MP_ROM_PTR(&ncnn_layer_get_one_blob_only_obj)},
    {MP_ROM_QSTR(MP_QSTR_layer_get_support_inplace), MP_ROM_PTR(&ncnn_layer_get_support_inplace_obj)},
    {MP_ROM_QSTR(MP_QSTR_layer_get_support_vulkan), MP_ROM_PTR(&ncnn_layer_get_support_vulkan_obj)},
    {MP_ROM_QSTR(MP_QSTR_layer_get_support_packing), MP_ROM_PTR(&ncnn_layer_get_support_packing_obj)},
    {MP_ROM_QSTR(MP_QSTR_layer_get_support_bf16_storage), MP_ROM_PTR(&ncnn_layer_get_support_bf16_storage_obj)},
    {MP_ROM_QSTR(MP_QSTR_layer_get_support_fp16_storage), MP_ROM_PTR(&ncnn_layer_get_support_fp16_storage_obj)},

    {MP_ROM_QSTR(MP_QSTR_layer_set_one_blob_only), MP_ROM_PTR(&ncnn_layer_set_one_blob_only_obj)},
    {MP_ROM_QSTR(MP_QSTR_layer_set_support_inplace), MP_ROM_PTR(&ncnn_layer_set_support_inplace_obj)},
    {MP_ROM_QSTR(MP_QSTR_layer_set_support_vulkan), MP_ROM_PTR(&ncnn_layer_set_support_vulkan_obj)},
    {MP_ROM_QSTR(MP_QSTR_layer_set_support_packing), MP_ROM_PTR(&ncnn_layer_set_support_packing_obj)},
    {MP_ROM_QSTR(MP_QSTR_layer_set_support_bf16_storage), MP_ROM_PTR(&ncnn_layer_set_support_bf16_storage_obj)},
    {MP_ROM_QSTR(MP_QSTR_layer_set_support_fp16_storage), MP_ROM_PTR(&ncnn_layer_set_support_fp16_storage_obj)},

    {MP_ROM_QSTR(MP_QSTR_layer_get_bottom_count), MP_ROM_PTR(&ncnn_layer_get_bottom_count_obj)},
    {MP_ROM_QSTR(MP_QSTR_layer_get_bottom), MP_ROM_PTR(&ncnn_layer_get_bottom_obj)},
    {MP_ROM_QSTR(MP_QSTR_layer_get_top_count), MP_ROM_PTR(&ncnn_layer_get_top_count_obj)},
    {MP_ROM_QSTR(MP_QSTR_layer_get_top), MP_ROM_PTR(&ncnn_layer_get_top_obj)},

    {MP_ROM_QSTR(MP_QSTR_blob_get_bottom_shape), MP_ROM_PTR(&ncnn_blob_get_bottom_shape_obj)},
    {MP_ROM_QSTR(MP_QSTR_blob_get_top_shape), MP_ROM_PTR(&ncnn_blob_get_top_shape_obj)},

    /* net api */
    {MP_ROM_QSTR(MP_QSTR_net_create), MP_ROM_PTR(&ncnn_net_create_obj)},
    {MP_ROM_QSTR(MP_QSTR_net_destroy), MP_ROM_PTR(&ncnn_net_destroy_obj)},
    {MP_ROM_QSTR(MP_QSTR_net_get_option), MP_ROM_PTR(&ncnn_net_get_option_obj)},
    {MP_ROM_QSTR(MP_QSTR_net_set_option), MP_ROM_PTR(&ncnn_net_set_option_obj)},
#if NCNN_VULKAN
    {MP_ROM_QSTR(MP_QSTR_net_set_vulkan_device), MP_ROM_PTR(&ncnn_net_set_vulkan_device_obj)},
#endif

#if NCNN_STRING
    {MP_ROM_QSTR(MP_QSTR_net_register_custom_layer_by_type), MP_ROM_PTR(&ncnn_net_register_custom_layer_by_type_obj)},
#endif /* NCNN_STRING */
    {MP_ROM_QSTR(MP_QSTR_net_register_custom_layer_by_typeindex), MP_ROM_PTR(&ncnn_net_register_custom_layer_by_typeindex_obj)},

#if NCNN_STDIO
#if NCNN_STRING
    {MP_ROM_QSTR(MP_QSTR_net_load_param), MP_ROM_PTR(&ncnn_net_load_param_obj)},
#endif /* NCNN_STRING */
    {MP_ROM_QSTR(MP_QSTR_net_load_param_bin), MP_ROM_PTR(&ncnn_net_load_param_bin_obj)},
    {MP_ROM_QSTR(MP_QSTR_net_load_model), MP_ROM_PTR(&ncnn_net_load_model_obj)},
#endif /* NCNN_STDIO */

#if NCNN_STDIO
#if NCNN_STRING
    {MP_ROM_QSTR(MP_QSTR_net_load_param_memory), MP_ROM_PTR(&ncnn_net_load_param_memory_obj)},
#endif /* NCNN_STRING */
#endif /* NCNN_STDIO */
    {MP_ROM_QSTR(MP_QSTR_net_load_param_bin_memory), MP_ROM_PTR(&ncnn_net_load_param_bin_memory_obj)},
    {MP_ROM_QSTR(MP_QSTR_net_load_model_memory), MP_ROM_PTR(&ncnn_net_load_model_memory_obj)},

#if NCNN_STRING
    {MP_ROM_QSTR(MP_QSTR_net_load_param_datareader), MP_ROM_PTR(&ncnn_net_load_param_datareader_obj)},
#endif /* NCNN_STRING */
    {MP_ROM_QSTR(MP_QSTR_net_load_param_bin_datareader), MP_ROM_PTR(&ncnn_net_load_param_bin_datareader_obj)},
    {MP_ROM_QSTR(MP_QSTR_net_load_model_datareader), MP_ROM_PTR(&ncnn_net_load_model_datareader_obj)},

    {MP_ROM_QSTR(MP_QSTR_net_clear), MP_ROM_PTR(&ncnn_net_clear_obj)},

    {MP_ROM_QSTR(MP_QSTR_net_get_input_count), MP_ROM_PTR(&ncnn_net_get_input_count_obj)},
    {MP_ROM_QSTR(MP_QSTR_net_get_output_count), MP_ROM_PTR(&ncnn_net_get_output_count_obj)},
#if NCNN_STRING
    {MP_ROM_QSTR(MP_QSTR_net_get_input_name), MP_ROM_PTR(&ncnn_net_get_input_name_obj)},
    {MP_ROM_QSTR(MP_QSTR_net_get_output_name), MP_ROM_PTR(&ncnn_net_get_output_name_obj)},
#endif /* NCNN_STRING */
    {MP_ROM_QSTR(MP_QSTR_net_get_input_index), MP_ROM_PTR(&ncnn_net_get_input_index_obj)},
    {MP_ROM_QSTR(MP_QSTR_net_get_output_index), MP_ROM_PTR(&ncnn_net_get_output_index_obj)},

    /* extractor api */
    {MP_ROM_QSTR(MP_QSTR_extractor_create), MP_ROM_PTR(&ncnn_extractor_create_obj)},
    {MP_ROM_QSTR(MP_QSTR_extractor_destroy), MP_ROM_PTR(&ncnn_extractor_destroy_obj)},
    {MP_ROM_QSTR(MP_QSTR_extractor_set_option), MP_ROM_PTR(&ncnn_extractor_set_option_obj)},

#if NCNN_STRING
    {MP_ROM_QSTR(MP_QSTR_extractor_input), MP_ROM_PTR(&ncnn_extractor_input_obj)},
    {MP_ROM_QSTR(MP_QSTR_extractor_extract), MP_ROM_PTR(&ncnn_extractor_extract_obj)},
#endif /* NCNN_STRING */
    {MP_ROM_QSTR(MP_QSTR_extractor_input_index), MP_ROM_PTR(&ncnn_extractor_input_index_obj)},
    {MP_ROM_QSTR(MP_QSTR_extractor_extract_index), MP_ROM_PTR(&ncnn_extractor_extract_index_obj)},

    /* mat process api */
    {MP_ROM_QSTR(MP_QSTR_copy_make_border), MP_ROM_PTR(&ncnn_copy_make_border_obj)},
    {MP_ROM_QSTR(MP_QSTR_copy_make_border_3d), MP_ROM_PTR(&ncnn_copy_make_border_3d_obj)},
    {MP_ROM_QSTR(MP_QSTR_copy_cut_border), MP_ROM_PTR(&ncnn_copy_cut_border_obj)},
    {MP_ROM_QSTR(MP_QSTR_copy_cut_border_3d), MP_ROM_PTR(&ncnn_copy_cut_border_3d_obj)},

#if NCNN_PIXEL_DRAWING
    /* mat pixel drawing api*/
    {MP_ROM_QSTR(MP_QSTR_draw_rectangle_c1), MP_ROM_PTR(&ncnn_draw_rectangle_c1_obj)},
    {MP_ROM_QSTR(MP_QSTR_draw_rectangle_c2), MP_ROM_PTR(&ncnn_draw_rectangle_c2_obj)},
    {MP_ROM_QSTR(MP_QSTR_draw_rectangle_c3), MP_ROM_PTR(&ncnn_draw_rectangle_c3_obj)},
    {MP_ROM_QSTR(MP_QSTR_draw_rectangle_c4), MP_ROM_PTR(&ncnn_draw_rectangle_c4_obj)},
    {MP_ROM_QSTR(MP_QSTR_draw_text_c1), MP_ROM_PTR(&ncnn_draw_text_c1_obj)},
    {MP_ROM_QSTR(MP_QSTR_draw_text_c2), MP_ROM_PTR(&ncnn_draw_text_c2_obj)},
    {MP_ROM_QSTR(MP_QSTR_draw_text_c3), MP_ROM_PTR(&ncnn_draw_text_c3_obj)},
    {MP_ROM_QSTR(MP_QSTR_draw_text_c4), MP_ROM_PTR(&ncnn_draw_text_c4_obj)},
    {MP_ROM_QSTR(MP_QSTR_draw_circle_c1), MP_ROM_PTR(&ncnn_draw_circle_c1_obj)},
    {MP_ROM_QSTR(MP_QSTR_draw_circle_c2), MP_ROM_PTR(&ncnn_draw_circle_c2_obj)},
    {MP_ROM_QSTR(MP_QSTR_draw_circle_c3), MP_ROM_PTR(&ncnn_draw_circle_c3_obj)},
    {MP_ROM_QSTR(MP_QSTR_draw_circle_c4), MP_ROM_PTR(&ncnn_draw_circle_c4_obj)},
    {MP_ROM_QSTR(MP_QSTR_draw_line_c1), MP_ROM_PTR(&ncnn_draw_line_c1_obj)},
    {MP_ROM_QSTR(MP_QSTR_draw_line_c2), MP_ROM_PTR(&ncnn_draw_line_c2_obj)},
    {MP_ROM_QSTR(MP_QSTR_draw_line_c3), MP_ROM_PTR(&ncnn_draw_line_c3_obj)},
    {MP_ROM_QSTR(MP_QSTR_draw_line_c4), MP_ROM_PTR(&ncnn_draw_line_c4_obj)},
#endif /* NCNN_PIXEL_DRAWING */
};

/* define the module globals dictionary */
static MP_DEFINE_CONST_DICT(ncnn_module_globals, ncnn_module_globals_table);

/* define the module object */
const mp_obj_module_t ncnn_user_cmodule = {
    .base = {&mp_type_module},
    .globals = (mp_obj_dict_t*)&ncnn_module_globals,
};

/* register module */
MP_REGISTER_MODULE(MP_QSTR_ncnn, ncnn_user_cmodule);
