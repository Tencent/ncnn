#ifndef NCNN_MODULE_H
#define NCNN_MODULE_H

#include "py/runtime.h"

extern mp_obj_t mp_ncnn_version(void);

/* allocator api */
extern mp_obj_t mp_ncnn_allocator_create_pool_allocator(void);
extern mp_obj_t mp_ncnn_allocator_create_unlocked_pool_allocator(void);
extern mp_obj_t mp_ncnn_allocator_destroy(mp_obj_t ncnn_allocator_obj);

/* option api */
extern mp_obj_t mp_ncnn_option_create(void);
extern mp_obj_t mp_ncnn_option_destroy(mp_obj_t option_obj);
extern mp_obj_t mp_ncnn_option_get_num_threads(mp_obj_t option_obj);
extern mp_obj_t mp_ncnn_option_set_num_threads(mp_obj_t option_obj, mp_obj_t num_threads_obj);
extern mp_obj_t mp_ncnn_option_get_use_local_pool_allocator(mp_obj_t option_obj);
extern mp_obj_t mp_ncnn_option_set_use_local_pool_allocator(mp_obj_t option_obj, mp_obj_t use_local_pool_allocator_obj);
extern mp_obj_t mp_ncnn_option_set_blob_allocator(mp_obj_t option_obj, mp_obj_t allocator_obj);
extern mp_obj_t mp_ncnn_option_set_workspace_allocator(mp_obj_t option_obj, mp_obj_t allocator_obj);
extern mp_obj_t mp_ncnn_option_get_use_vulkan_compute(mp_obj_t option_obj);
extern mp_obj_t mp_ncnn_option_set_use_vulkan_compute(mp_obj_t option_obj, mp_obj_t use_vulkan_compute_obj);

/* mat api */
extern mp_obj_t mp_ncnn_mat_create(void);
extern mp_obj_t mp_ncnn_mat_create_1d(mp_obj_t w_obj, mp_obj_t allocator_obj);
extern mp_obj_t mp_ncnn_mat_create_2d(mp_obj_t w_obj, mp_obj_t h_obj, mp_obj_t allocator_obj);
extern mp_obj_t mp_ncnn_mat_create_3d(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_4d(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_external_1d(mp_obj_t w_obj, mp_obj_t data_obj, mp_obj_t allocator_obj);
extern mp_obj_t mp_ncnn_mat_create_external_2d(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_external_3d(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_external_4d(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_1d_elem(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_2d_elem(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_3d_elem(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_4d_elem(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_external_1d_elem(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_external_2d_elem(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_external_3d_elem(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_create_external_4d_elem(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_destroy(mp_obj_t mat_obj);

extern mp_obj_t mp_ncnn_mat_fill_float(mp_obj_t mat_obj, mp_obj_t v_obj);

extern mp_obj_t mp_ncnn_mat_clone(mp_obj_t mat_obj, mp_obj_t allocator_obj);
extern mp_obj_t mp_ncnn_mat_reshape_1d(mp_obj_t mat_obj, mp_obj_t w_obj, mp_obj_t allocator_obj);
extern mp_obj_t mp_ncnn_mat_reshape_2d(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_reshape_3d(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_reshape_4d(size_t n_args, const mp_obj_t* args);

extern mp_obj_t mp_ncnn_mat_get_dims(mp_obj_t mat_obj);
extern mp_obj_t mp_ncnn_mat_get_w(mp_obj_t mat_obj);
extern mp_obj_t mp_ncnn_mat_get_h(mp_obj_t mat_obj);
extern mp_obj_t mp_ncnn_mat_get_d(mp_obj_t mat_obj);
extern mp_obj_t mp_ncnn_mat_get_c(mp_obj_t mat_obj);
extern mp_obj_t mp_ncnn_mat_get_elemsize(mp_obj_t mat_obj);
extern mp_obj_t mp_ncnn_mat_get_elempack(mp_obj_t mat_obj);
extern mp_obj_t mp_ncnn_mat_get_cstep(mp_obj_t mat_obj);
extern mp_obj_t mp_ncnn_mat_get_data(mp_obj_t mat_obj);

extern mp_obj_t mp_ncnn_mat_get_channel_data(mp_obj_t mat_obj, mp_obj_t c_obj);

/* pixel api */
extern mp_obj_t mp_ncnn_mat_from_pixels(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_from_pixels_resize(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_from_pixels_roi(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_from_pixels_roi_resize(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_to_pixels(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_mat_to_pixels_resize(size_t n_args, const mp_obj_t* args);

/* mat processing */
extern mp_obj_t mp_ncnn_mat_substract_mean_normalize(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_convert_packing(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_flatten(size_t n_args, const mp_obj_t* args);

/* blob api */
extern mp_obj_t mp_ncnn_blob_get_name(mp_obj_t blob_obj);
extern mp_obj_t mp_ncnn_blob_get_producer(mp_obj_t blob_obj);
extern mp_obj_t mp_ncnn_blob_get_consumer(mp_obj_t blob_obj);
extern mp_obj_t mp_ncnn_blob_get_shape(size_t n_args, const mp_obj_t* args);

/* paramdict api */
extern mp_obj_t mp_ncnn_paramdict_create(void);
extern mp_obj_t mp_ncnn_paramdict_destroy(mp_obj_t pd_obj);
extern mp_obj_t mp_ncnn_paramdict_get_type(mp_obj_t pd_obj, mp_obj_t id_obj);
extern mp_obj_t mp_ncnn_paramdict_get_int(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_paramdict_get_float(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_paramdict_get_array(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_paramdict_set_int(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_paramdict_set_float(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_paramdict_set_array(size_t n_args, const mp_obj_t* args);

/* datareader api */
extern mp_obj_t mp_ncnn_datareader_create(void);
extern mp_obj_t mp_ncnn_datareader_create_from_stdio(mp_obj_t fp_obj);
extern mp_obj_t mp_ncnn_datareader_create_from_memory(mp_obj_t mem_obj);
extern mp_obj_t mp_ncnn_datareader_destroy(mp_obj_t dr_obj);

/* modelbin api */
extern mp_obj_t mp_ncnn_modelbin_create_from_datareader(mp_obj_t dr_obj);
extern mp_obj_t mp_ncnn_modelbin_create_from_mat_array(mp_obj_t weights_obj, mp_obj_t n_obj);
extern mp_obj_t mp_ncnn_modelbin_destroy(mp_obj_t mb_obj);

/* layer api */
extern mp_obj_t mp_ncnn_layer_create(void);
extern mp_obj_t mp_ncnn_layer_create_by_typeindex(mp_obj_t typeindex_obj);
extern mp_obj_t mp_ncnn_layer_create_by_type(mp_obj_t type_obj);
extern mp_obj_t mp_ncnn_layer_type_to_index(mp_obj_t type_obj);
extern mp_obj_t mp_ncnn_layer_destroy(mp_obj_t layer_obj);
extern mp_obj_t mp_ncnn_layer_get_name(mp_obj_t layer_obj);
extern mp_obj_t mp_ncnn_layer_get_typeindex(mp_obj_t layer_obj);
extern mp_obj_t mp_ncnn_layer_get_type(mp_obj_t layer_obj);
extern mp_obj_t mp_ncnn_layer_get_one_blob_only(mp_obj_t layer_obj);
extern mp_obj_t mp_ncnn_layer_get_support_inplace(mp_obj_t layer_obj);
extern mp_obj_t mp_ncnn_layer_get_support_vulkan(mp_obj_t layer_obj);
extern mp_obj_t mp_ncnn_layer_get_support_packing(mp_obj_t layer_obj);
extern mp_obj_t mp_ncnn_layer_get_support_bf16_storage(mp_obj_t layer_obj);
extern mp_obj_t mp_ncnn_layer_get_support_fp16_storage(mp_obj_t layer_obj);
extern mp_obj_t mp_ncnn_layer_set_one_blob_only(mp_obj_t layer_obj, mp_obj_t enable_obj);
extern mp_obj_t mp_ncnn_layer_set_support_inplace(mp_obj_t layer_obj, mp_obj_t enable_obj);
extern mp_obj_t mp_ncnn_layer_set_support_vulkan(mp_obj_t layer_obj, mp_obj_t enable_obj);
extern mp_obj_t mp_ncnn_layer_set_support_packing(mp_obj_t layer_obj, mp_obj_t enable_obj);
extern mp_obj_t mp_ncnn_layer_set_support_bf16_storage(mp_obj_t layer_obj, mp_obj_t enable_obj);
extern mp_obj_t mp_ncnn_layer_set_support_fp16_storage(mp_obj_t layer_obj, mp_obj_t enable_obj);
extern mp_obj_t mp_ncnn_layer_get_bottom_count(mp_obj_t layer_obj);
extern mp_obj_t mp_ncnn_layer_get_bottom(mp_obj_t layer_obj, mp_obj_t i_obj);
extern mp_obj_t mp_ncnn_layer_get_top_count(mp_obj_t layer_obj);
extern mp_obj_t mp_ncnn_layer_get_top(mp_obj_t layer_obj, mp_obj_t i_obj);
extern mp_obj_t mp_ncnn_blob_get_bottom_shape(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_blob_get_top_shape(size_t n_args, const mp_obj_t* args);

/* net api */
extern mp_obj_t mp_ncnn_net_create(void);
extern mp_obj_t mp_ncnn_net_destroy(mp_obj_t net_obj);
extern mp_obj_t mp_ncnn_net_get_option(mp_obj_t net_obj);
extern mp_obj_t mp_ncnn_net_set_option(mp_obj_t net_obj, mp_obj_t opt_obj);
extern mp_obj_t mp_ncnn_net_set_vulkan_device(mp_obj_t net_obj, mp_obj_t device_index_obj);
extern mp_obj_t mp_ncnn_net_register_custom_layer_by_type(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_net_register_custom_layer_by_typeindex(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_net_load_param(mp_obj_t net_obj, mp_obj_t path_obj);
extern mp_obj_t mp_ncnn_net_load_param_bin(mp_obj_t net_obj, mp_obj_t path_obj);
extern mp_obj_t mp_ncnn_net_load_model(mp_obj_t net_obj, mp_obj_t path_obj);
extern mp_obj_t mp_ncnn_net_load_param_memory(mp_obj_t net_obj, mp_obj_t mem_obj);
extern mp_obj_t mp_ncnn_net_load_param_bin_memory(mp_obj_t net_obj, mp_obj_t mem_obj);
extern mp_obj_t mp_ncnn_net_load_model_memory(mp_obj_t net_obj, mp_obj_t mem_obj);
extern mp_obj_t mp_ncnn_net_load_param_datareader(mp_obj_t net_obj, mp_obj_t dr_obj);
extern mp_obj_t mp_ncnn_net_load_param_bin_datareader(mp_obj_t net_obj, mp_obj_t dr_obj);
extern mp_obj_t mp_ncnn_net_load_model_datareader(mp_obj_t net_obj, mp_obj_t dr_obj);
extern mp_obj_t mp_ncnn_net_clear(mp_obj_t net_obj);
extern mp_obj_t mp_ncnn_net_get_input_count(mp_obj_t net_obj);
extern mp_obj_t mp_ncnn_net_get_output_count(mp_obj_t net_obj);
extern mp_obj_t mp_ncnn_net_get_input_name(mp_obj_t net_obj, mp_obj_t i_obj);
extern mp_obj_t mp_ncnn_net_get_output_name(mp_obj_t net_obj, mp_obj_t i_obj);
extern mp_obj_t mp_ncnn_net_get_input_index(mp_obj_t net_obj, mp_obj_t i_obj);
extern mp_obj_t mp_ncnn_net_get_output_index(mp_obj_t net_obj, mp_obj_t i_obj);

/* extractor api */
extern mp_obj_t mp_ncnn_extractor_create(mp_obj_t net_obj);
extern mp_obj_t mp_ncnn_extractor_destroy(mp_obj_t ex_obj);
extern mp_obj_t mp_ncnn_extractor_set_option(mp_obj_t ex_obj, mp_obj_t opt_obj);
extern mp_obj_t mp_ncnn_extractor_input(mp_obj_t ex_obj, mp_obj_t name_obj, mp_obj_t mat_obj);
extern mp_obj_t mp_ncnn_extractor_extract(mp_obj_t ex_obj, mp_obj_t name_obj, mp_obj_t mat_obj);
extern mp_obj_t mp_ncnn_extractor_input_index(mp_obj_t ex_obj, mp_obj_t index_obj, mp_obj_t mat_obj);
extern mp_obj_t mp_ncnn_extractor_extract_index(mp_obj_t ex_obj, mp_obj_t name_obj, mp_obj_t mat_obj);

/* mat process api */
extern mp_obj_t mp_ncnn_copy_make_border(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_copy_make_border_3d(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_copy_cut_border(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_copy_cut_border_3d(size_t n_args, const mp_obj_t* args);

/* drawing api */
extern mp_obj_t mp_ncnn_draw_rectangle_c1(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_draw_rectangle_c2(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_draw_rectangle_c3(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_draw_rectangle_c4(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_draw_text_c1(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_draw_text_c2(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_draw_text_c3(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_draw_text_c4(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_draw_circle_c1(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_draw_circle_c2(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_draw_circle_c3(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_draw_circle_c4(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_draw_line_c1(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_draw_line_c2(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_draw_line_c3(size_t n_args, const mp_obj_t* args);
extern mp_obj_t mp_ncnn_draw_line_c4(size_t n_args, const mp_obj_t* args);

#endif